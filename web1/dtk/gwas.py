import logging
logger = logging.getLogger(__name__)

def gwas_code(ds_id):
    return "gwds%d"%ds_id

def gwas_codes(ws):
    return [
            gwas_code(x[0])
            for x in ws.get_gwas_dataset_choices()
            ]

# a CM convenience function that returns a list of gwas codes
# corresponding to enabled settings is a parms dict
def selected_gwas(parms):
    return [k for k,v in parms.items() if k.startswith('gwds') and v]

def score_snp(val, v2g_val=1):
    """
    val is typically going to be a p-value relating to the variant-to-disease association.

    v2g_val will be a 0-1 score indicating strength of functional association between variant and gene.
    Older datasets didn't have v2g data, so assume a value of 1 for all.
    """
    #return 1.0 - val
    from math import log
    try:
        return -log(val,10) * v2g_val
    except ValueError:
        return 0.0
    #return 1.0/val**0.2

def scored_gwas(key,v2d_threshold=1.0,v2g_threshold=0.0,max_prot_assocs=None,exclude=False):
    assert key.startswith('gwds')
    ds_id = int(key[4:])
    from browse.models import GwasDataset
    from collections import defaultdict
    ds = GwasDataset.objects.get(pk=ds_id)

    # Keeps only results that pass the thresholds
    result = defaultdict(lambda: -1e99)

    # Tracks all results ignoring thresholds, used for tracking what gets left out.
    all_result = defaultdict(lambda: -1e99)
    for rec in ds.get_data():
        if rec.uniprot == '-':
            continue

        s = score_snp(rec.evidence, rec.v2g_evidence)

        # If there are duplicates, we take the maximum score.
        all_result[rec.uniprot] = max(all_result[rec.uniprot], s)

        # evidence is a p-value, must be under threshold.
        # v2g_evidence is a 0 to 1 evidence score, must be above that threshold.
        if rec.evidence <= v2d_threshold and rec.v2g_evidence >= v2g_threshold:
            result[rec.uniprot] = max(result[rec.uniprot], s)
    
    if max_prot_assocs is not None and len(result) > max_prot_assocs:
        scores = sorted(result.values(), reverse=True)
        threshold = scores[max_prot_assocs]
        result = {k:v for k,v in result.items() if v >= threshold}
    
    if exclude:
        # Return assocs for prots that aren't in 'result'.
        return {k:v for k,v in all_result.items() if k not in result}
    else:
        return dict(result)


class GwasSearchFilter:
    def __init__(self,ws):
        from browse.models import GwasDataset,GwasFilter
        self.ws = ws
        # get PMID filters
        self.filters=[
                x
                for x in GwasFilter.objects.filter(ws=ws,rejected=False)
                ]
        # get already-classified datasets
        self.selects=[]
        self.rejects=[]
        for gwds in GwasDataset.objects.filter(ws=ws):
            if gwds.rejected:
                self.rejects.append(gwds)
            else:
                self.selects.append(gwds)
        # set up use counters
        self.filtered=0
        self.selected=0
        self.rejected=0
        # set up filtering keys
        self.pmid_keys = set([str(x.pubmed_id) for x in self.filters])
        self.sel_keys = set([x.make_key() for x in self.selects])
        self.rej_keys = set([x.make_key() for x in self.rejects])
    def ok(self,key):
        phenotype,pmid = key.split('|')
        if key in self.sel_keys:
            self.selected += 1
            return False
        if key in self.rej_keys:
            self.rejected += 1
            return False
        if str(pmid) in self.pmid_keys:
            self.filtered += 1
            return False
        return True
    def used_counts_by_pmid(self):
        from collections import Counter
        return Counter([x.pubmed_id for x in self.selects])
    def ood_info(self):
        '''Return list of (gwds,date) for out-of-date datasets.

        A dataset is considered out-of-date if the data extraction file
        is older than the local duma_gwas master file.
        '''
        from browse.models import GwasDataset
        master_file = GwasDataset.get_master_S3File(self.ws)
        from dtk.files import modification_date
        master_date = modification_date(master_file.path())
        result = []
        for rec in self.selects:
            path = rec.make_path()
            try:
                mod = modification_date(path)
            except OSError:
                # if file doesn't exist, it can't be out of date
                # (it will be regenerated from the latest data on demand)
                continue
            if mod < master_date:
                result.append((rec,mod))
        return result

def search_gwas_studies_file(ws, search, ds_filter=None, exact=False):
    import datetime
    from browse.default_settings import duma_gwas_v2d, duma_gwas_v2g
    from dtk.files import get_file_records
    s3_file = duma_gwas_v2d.get_s3_file(ws=ws, role='studies')
    summary_s3_file = duma_gwas_v2g.get_s3_file(ws=ws, role='d2g_summary')

    matches=[]
    keys = []
    header=None
    if ds_filter:
        used_counts = ds_filter.used_counts_by_pmid()
    else:
        from collections import Counter
        used_counts = Counter()
    for rec in get_file_records(s3_file.path(),keep_header=True):
        if header is None:
            header = rec
            # we use this one a lot, so save it as a variable
            pheno_ind = header.index('Phenotype|PMID')
            continue

        # There are no spaces in the dataset, just underscores and dashes.
        # But people will usually search with a space, so make that work.
        search_pheno = rec[pheno_ind]
        search_pheno += search_pheno.replace('_', ' ').replace('-', ' ')

        # skip anything that doesn't match search criteria
        if ((exact and not all([word==rec[pheno_ind] for word in search]))
            or not all([word in search_pheno for word in search])
           ):
            continue
        # exclude filtered results
        if ds_filter and not ds_filter.ok(rec[pheno_ind]):
            continue
        # attempt to convert date; some aren't formatted correctly,
        # so just leave them as string
        try:
            rec[header.index('DatePub')]=datetime.datetime.strptime(
                                     rec[header.index('DatePub')],"%m/%d/%Y"
                                     ).date()
        except ValueError:
            pass
        # split key to make separate phenotype and pubmed fields;
        # convert phenotype to space-separated so it wraps
        phenotype,pubmed=rec[pheno_ind].split('|')
        phenotype = ' '.join(phenotype.split('_'))
        # Try converting sample counts to integers.
        try:
            rec[1] = int(rec[1])
        except (ValueError, TypeError) as e:
            pass
        # We insert 0's as placeholders for num_var and num_prot.
        matches.append([phenotype,pubmed,used_counts[pubmed], 0, 0]+rec)
        keys.append(rec[pheno_ind])
    
    d2g_data = lookup_d2g_summaries(summary_s3_file.path(), keys)
    assert len(d2g_data) == len(matches)
    for match, (num_var, num_prot) in zip(matches, d2g_data):
        match[3] = num_var
        match[4] = num_prot

    return matches


def lookup_variant_v2g(archive_fn, variants):
    """Variants should be chr:pos strings"""
    # This length is determined in the v2g makefile, currently hardcoded.
    PREFIX_LENGTH = 6
    def make_key(variant):
        return variant[:PREFIX_LENGTH]
    
    from dtk.data import MultiMap 
    grouped_variants = MultiMap((make_key(x), x) for x in variants).fwd_map()

    out = []
    import zipfile
    # v1 didn't have a score column.
    header = ['chrm_and_pos', 'allele', 'rs_ids', 'consequences', 'uniprot', 'score']
    with zipfile.ZipFile(archive_fn, 'r') as archive:
        for key, vars in grouped_variants.items():
            try:
                for line in archive.open(key):
                    parts = line.decode('utf8').strip('\n').split('\t')
                    if parts[0] in vars:
                        rec = dict(zip(header,parts))
                        # uniprot and consequences
                        out.append((rec['chrm_and_pos'], (rec['uniprot'], rec.get('score', '1'))))
            except KeyError:
                pass
    return MultiMap(out).fwd_map()

        

def extract_gwas_key_old(archive_fn, key):
    out = []
    import zipfile
    # We don't store '/'s in the key name in the zip, invalid path.
    key = key.replace('/', '_')

    with zipfile.ZipFile(archive_fn, 'r') as archive:
        try:
            archive.open(key)
        except KeyError:
            # In older workspaces, we sometimes have uppercased gwas keys, due to inconsistency in older datasets.
            # We can't universally convert to lower because that will break on those older datasets, but if we've
            # switched to a newer dataset in this workspace, let's make that case work by also trying out the lower
            # cased version of the key.
            key = key.lower()

        for line in archive.open(key):
            out.append(line.decode('utf8').strip('\n'))
    return out

def extract_gwas_key_new(v2d_fn, v2g_fn, key):
    key = key.replace('/', '_')
    import zipfile
    with zipfile.ZipFile(v2d_fn, 'r') as archive:
        try:
            archive.open(key)
        except KeyError:
            # In older workspaces, we sometimes have uppercased gwas keys, due to inconsistency in older datasets.
            # We can't universally convert to lower because that will break on those older datasets, but if we've
            # switched to a newer dataset in this workspace, let's make that case work by also trying out the lower
            # cased version of the key.
            key = key.lower()

        study_variant_data = [line.decode('utf8').strip('\n').split('\t') for line in archive.open(key)]

    # Assemble 'chrm:pos' strings
    variants = [f'{x[2]}:{x[3]}' for x in study_variant_data]
    variant2data = lookup_variant_v2g(v2g_fn, variants)

    out = []
    for study, variant in zip(study_variant_data, variants):
        data = variant2data.get(variant, [])
        # studykey, rsids, chrm, pos, pval
        row_pre = study[:5]
        # allele;maf, misc(?)
        row_suf = study[-2:]

        for entry in data:
            row = row_pre + list(entry) + row_suf
            out.append('\t'.join(row))
    return out

def lookup_d2g_summaries(fn, study_keys):
    from dtk.files import get_file_records
    s2d = {}
    for study, num_variants, num_prots in get_file_records(fn, select=(study_keys, 'study')):
        s2d[study] = (num_variants, num_prots)
    
    out = [s2d.get(k, ('', '')) for k in study_keys]
    return out


def lookup_otarg_alleles(variants):
    from browse.default_settings import duma_gwas_v2g
    from dtk.files import get_file_records
    fn = duma_gwas_v2g.get_s3_file(latest=True, role='otarg_alleles').path()
    v2data = []
    for rec in get_file_records(fn, select=(variants, 'chrm_and_pos')):
        v2data.append((rec[0], rec))
    
    from dtk.data import MultiMap
    return MultiMap(v2data).fwd_map()
    