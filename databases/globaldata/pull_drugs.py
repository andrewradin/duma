#!/usr/bin/env python3

program_description='''\
Build attributes and dpi files from GlobalData API.
'''

from django.utils.http import urlencode
import requests
import json
from collections import Counter
from dtk.lazy_loader import LazyLoader
from dtk.files import FileDestination
from dtk.text import diffstr

# NOTE about API keys
# - you need a TokenId to access the GlobalData API
# - we store these keys in the global_data.json file in s3://2xar-duma-keys,
#   which then gets read into 2xar/ws/keys/global_data.json
# - these keys expire periodically; they can be renewed with scripts/apikeys.py
# - these are checked daily by a cron job, and renewed if they're within a
#   week of expiring
# - to prevent a renewal in the middle of an ETL operation (which would
#   invalidate the current key), the Makefile suggests forcing a renewal
#   prior to running the ETL

class ReportItem:
    # allows ordering and detection of instances supplying reports
    counter = 0
    @classmethod
    def next_token(cls):
        cls.counter += 1
        return cls.counter
    def __init__(self):
        self.sequence = self.next_token()
    @classmethod
    def ordered_items(cls,container):
        result = []
        for attr in dir(container):
            obj = getattr(container,attr)
            if isinstance(obj,cls):
                result.append((attr,obj))
        return sorted(result,key=lambda x:x[1].sequence)

class CaseCounter(ReportItem):
    def __init__(self,heading):
        super().__init__()
        self.heading=heading
        self.ctr = Counter()
    def count(self,case):
        self.ctr[case] += 1
    def report(self,f):
        print(self.heading,file=f)
        for k,v in self.ctr.most_common():
            print(f'{v:10}: {k}',file=f)

class ParseStats:
    def __init__(self):
        trace_fn = 'parse_errors.tsv'
        self.log = FileDestination(trace_fn)
        self.disposition = CaseCounter('Record Counts:')
        self.parse_errors = CaseCounter(
                f'Parse Errors by Attribute (details in {trace_fn}):'
                )
        self.highest_stage = CaseCounter('Highest Product Stage:')
        self.drug_type = CaseCounter('Drug Types:')
        self.moa_suffixes = CaseCounter('MOA Suffixes:')
    def report_stats(self,f):
        for attr,obj in ReportItem.ordered_items(self):
            obj.report(f)
    def parse_error(self,mode,attr,val,key):
        self.log.append((mode,attr,val,key))
        self.parse_errors.count(f'{mode}; {attr}')

class DrugWrapper(LazyLoader):
    _kwargs=['api_data','mode','stats']
    def _coll_key_loader(self):
        # GlobalData drugs have a numeric id; add GD prefix
        return f'GD{self.api_data["DrugID"]}'
    def count_cases(self,ctr,attr):
        ctr.count('; '.join([
                self.mode,
                self.api_data[attr],
                ]))
    def skip(self):
        keep, detail = self._prepare()
        self.stats.disposition.count('; '.join([
                    self.mode,
                    'processed' if keep else 'skipped',
                    detail,
                    ]))
        return not keep
    def _prepare(self):
        # gather any stats where we want to include skipped items
        self.count_cases(self.stats.highest_stage,'Highest_Product_Stage')
        self.count_cases(self.stats.drug_type,'Drug_Type')
        # decide whether to process and why
        if not self.canonical or len(self.canonical) >= 256:
            return (False,'bad canonical')
        if self.api_data["MonoCombinationDrug"] != "Mono":
            # XXX some combos are not marked (eg. GD467300)
            return (False,'combo')
        keep_reasons = []
        if self.targets:
            keep_reasons.append('targets')
        if self.max_phase:
            keep_reasons.append(f'max_phase {self.max_phase}')
        if self.cas:
            keep_reasons.append('cas')
        if keep_reasons:
            return (True,' and '.join(keep_reasons))
        return (False,'no targets and not in clinic')
    def _wr(self,f,*vals):
        f.write('\t'.join([self.coll_key]+[str(x) for x in vals])+'\n')
    def _prep_multi(self,src):
        val = self.api_data[src]
        if val:
            parts = val.split()
            generic_name = self.api_data["Generic_Name"]
            if len(parts) == 1:
                pass
            elif len(parts) == 2 and parts[0] == generic_name+':':
                val = parts[1]
            else:
                self.stats.parse_error(self.mode,src,val,self.coll_key)
                val = ''
        return val
    def _opt_copy_attr(self,f,attr,src):
        val = self._prep_multi(src)
        if val:
            self._wr(f,attr,val)
    phases = {
            'Marketed':4,
            'Archived (Marketed)':4,
            'Withdrawn (Marketed)':4,
            'Phase III':3,
            'Phase II':2,
            'Phase I':1,
            }
    def attr_write(self,f):
        self._wr(f,'canonical',self.canonical)
        for alias in self.api_data['Alias'].split(','):
            # regularize whitespace: this strips leading and trailing
            # whitespace, and replaces any internal runs of whitespace
            # with a single space; this eliminates all problematic \n's
            alias = ' '.join(alias.split())
            if alias:
                self._wr(f,'synonym',alias)
        # XXX Chemical_Name could potentially be an alias, but it
        # XXX requires parsing, and those long names aren't really useful
        self._opt_copy_attr(f,'mol_formula',"Chemical_Formula")
        if self.cas:
            self._wr(f,'cas',self.cas)
        self._wr(f,'max_phase',self.max_phase)
        # XXX ATC_Classification?
        # XXX - need to split code from description
        # XXX - not clear if there are ever multiple codes, or what the
        # XXX   format is
        # XXX - sometimes partial (N, B02)
        # XXX - often 'other' (L01X, L03AX)
    def _canonical_loader(self):
        return self.api_data['Drug_Name']
    def _max_phase_loader(self):
        return self.phases.get(self.api_data['Highest_Product_Stage'],0)
    def _cas_loader(self):
        return self._prep_multi('CAS')
    def _targets_loader(self):
        result = []
        import dtk.prot_search as ps
        raw_str = self.api_data['Target']
        if not raw_str:
            return result
        directions = self._get_directions_from_moa(raw_str)
        targets = raw_str.split(';')
        for target,direction in zip(targets,directions):
            p = ps.find_protein_for_global_data_target(target)
            if p:
                result.append((
                        p.uniprot,
                        0.5 if not direction else 0.9,
                        direction,
                        ))
            else:
                self.stats.parse_error(self.mode,'Target',target,self.coll_key)
        return result
    directions = {
            'Inhibitor':-1,
            'Antagonist':-1,
            'Blocker':-1,
            'Synthesis Inhibitor':-1,
            'Polymerase (EC 2.7.7.7) Inhibitor':-1,
            'Uptake Inhibitor':-1,
            'Gyrase (EC 5.99.1.3) Inhibitor':-1,
            'Disruptor':-1,
            'Agonist':1,
            'Activator':1,
            }
    def _get_directions_from_moa(self,targ_str):
        moa_str = self.api_data['Mechanism_of_Action']
        raw_targets = targ_str.split(';')
        raw_moas = moa_str.split(';')
        if len(raw_targets) != len(raw_moas):
            self.stats.parse_error(
                    self.mode,
                    'MOA Length',
                    moa_str or f'No MOA: {targ_str}',
                    self.coll_key,
                    )
            return [0]*len(raw_targets)
        result = []
        for targ,moa in zip(raw_targets,raw_moas):
            if moa.startswith(targ):
                suffix = moa[len(targ):].strip()
                self.stats.moa_suffixes.count(suffix)
                result.append(self.directions.get(suffix,0))
            else:
                self.stats.parse_error(
                        self.mode,
                        'MOA Mismatch',
                        json.dumps(diffstr(targ,moa)),
                        self.coll_key,
                        )
        return result
    def dpi_write(self,f):
        for uniprot,ev,dirn in self.targets:
            self._wr(f,uniprot,ev,dirn)

class GDApi:
    def __init__(self):
        from dtk.apikeys import GDApiKey
        self.apikey = GDApiKey()
        self.seen_keys = set()
        self.stats = ParseStats()
    def report_stats(self,f):
        self.stats.report_stats(f)
    def api_fetch(self,page,pagesize):
        return self.apikey.api_fetch(
                f'/api/Drugs/Get{self.mode}DrugDetails',
                TokenId=self.apikey.api_key,
                DisplayName='Aria',
                PageNumber=page,
                PageSize=pagesize,
                )
    def get_record_count(self):
        d = self.api_fetch(1,1)
        return d['TotalRecordsDetails'][0]['TotalRecords']
    def pull_api_drugs(self,limit,pagesize):
        page = 1
        num_pages = 1 # can be anything >= page to start cycle
        while page <= num_pages:
            d = self.api_fetch(page,pagesize)
            num_pages = d['TotalRecordsDetails'][0]['NoOfPages']
            for item in d[f'{self.mode}Drugs']:
                dw = DrugWrapper(
                        api_data=item,
                        stats=self.stats,
                        mode=self.mode,
                        )
                if dw.coll_key in self.seen_keys:
                    self.stats.disposition.count('; '.join([
                                self.mode,
                                'skipped',
                                'duplicate key',
                                ]))
                    continue
                self.seen_keys.add(dw.coll_key)
                yield dw
            page += 1
            if limit and page > limit:
                break

def scan_endpoint(api,mode,limit,f_attr,f_dpi,pagesize):
    api.mode = mode
    rec_count = api.get_record_count()
    from tqdm import tqdm
    for raw_rec in tqdm(
            api.pull_api_drugs(limit,pagesize),
            total=rec_count,
            desc=mode,
            smoothing=.1/pagesize,
            ):
        if raw_rec.skip():
            continue
        raw_rec.attr_write(f_attr)
        raw_rec.dpi_write(f_dpi)

def check_protein_table(load_uniprot):
    from dtk.s3_cache import S3Bucket
    uniprot_fns = S3Bucket('uniprot').list(cache_ok=False)
    from dtk.files import VersionedFileName
    uniprot_choices = [
            x[0]
            for x in VersionedFileName.get_choices(
                    file_class='uniprot',
                    paths=uniprot_fns,
                    )
            ]
    from browse.models import ProteinUploadStatus
    cur_prot_fn = ProteinUploadStatus.current_upload()
    if cur_prot_fn:
        cur_choice = '.'.join(cur_prot_fn.split('.')[1:3])
    else:
        cur_choice = ''
    if load_uniprot:
        if cur_choice != load_uniprot:
            from browse.models import import_proteins
            print(f'loading uniprot version {load_uniprot}')
            import_proteins(load_uniprot)
        # in either case we're now good to go
    else:
        if cur_choice != uniprot_choices[0]:
            if cur_choice:
                print(f'Currently loaded uniprot version is {cur_choice}.')
            else:
                print(f'There is no uniprot data loaded.')
            print(f'The latest version is {uniprot_choices[0]}.')
            print(f'Please re-run with the --load-uniprot option,')
            print(f'specifying one of the following choices:')
            for choice in uniprot_choices:
                print('   ',choice)
            print('Exiting')
            import sys
            sys.exit(1)
        # else the selected file is loaded, and we're good to go

def write_extract_files(limit,load_uniprot,pagesize):
    import django_setup
    check_protein_table(load_uniprot)
    attr_fn = f'tmp.attributes.tsv'
    dpi_fn = f'tmp.evidence.tsv'
    api = GDApi()
    force_error = False
    try:
        with open(attr_fn, 'w') as f_attr:
            with open(dpi_fn, 'w') as f_dpi:
                f_attr.write('\t'.join([
                        'globaldata_id',
                        'attribute',
                        'value',
                        ])+'\n')
                f_dpi.write('\t'.join([
                        'globaldata_id',
                        'uniprot_id',
                        'evidence',
                        'direction',
                        ])+'\n')
                scan_endpoint(api,'Marketed',limit,f_attr,f_dpi,pagesize)
                scan_endpoint(api,'Pipeline',limit,f_attr,f_dpi,pagesize)
    except (KeyboardInterrupt,requests.exceptions.HTTPError):
        force_error = True
    with open('last_stats.out', 'w') as f_stats:
        api.report_stats(f_stats)
    if force_error:
        raise RuntimeError('Terminated prematurely')
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=program_description,
            )
    parser.add_argument('--limit',default=0,type=int)
    # Note that large pagesizes can cause problems; a pagesize of 10000
    # causes a gateway timeout on the last page of the endpoint. Before
    # using a larger pagesize, you can test it with:
    # >>> from pull_drugs import GDApi
    # >>> api=GDApi()
    # >>> api.mode = 'Pipeline'
    # >>> test_page_size = 5000
    # >>> d = api.api_fetch(1,test_page_size)
    # >>> last_page = d['TotalRecordsDetails'][0]['NoOfPages']
    # >>> d = api.api_fetch(last_page,test_page_size)
    parser.add_argument('--pagesize',default=1000,type=int)
    parser.add_argument('--load-uniprot')
    args = parser.parse_args()

    write_extract_files(args.limit,args.load_uniprot,args.pagesize)
