#!/usr/bin/env python3

from builtins import str
from builtins import object
import sys,re
verbose=False

# XXX molecular weight is not available
# XXX molecular formula can be extracted from inchi

# A program to process the Ki, IC50 or EC50 data from drug-protein interaction data from The BindingDB
# The final product being drug names, uniprot IDs, Ki or IC50/EC50 (all in nM)
#drugName, uniprotID, value
#Lepirudin, P00734, 0.95

# A few subroutines to print out as I want

def message(*objs):
    print(*objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def useage(*objs):
    print("USEAGE: ", *objs, file=sys.stderr)
    
def printOut(*objs):
    print(*objs, file=sys.stdout)

_digits = re.compile('\d')
def contains_digits(d):
    return bool(_digits.search(d))

class InputData(object):
    # scan the file and:
    # - assign an id to each drug name
    # - accumulate other attributes for each drug name
    # - for each DPI, store any KI, IC50, or EC50 (keep list of each)
    # - output a single dpi file, with KI, IC50, EC50 as extra fields
    # - output an attributes file with drug keys
    def __init__(self):
        self.drug_info = {} # {key:{attr_name:set()}}
        self.dpi = {} # {key+uniprot:[key,uniprot,[IC50s], [EC50s],[Kis]]}
        self.attrs = (
                'canonical',
                'pubchem_cid',
                'drugbank_id',
                'kegg',
                'smiles_code',
                'inchi',
                'inchi_key',
                )
        self.output_attrs = self.attrs + (
                'synonym',
                'linked_chembl_id',
                )
        self.srcs = (
                'curation_id',
                'article_doi',
                'pubmed_id',
                'pubchem_aid',
                'patent_id'
                )
        self.sources_data = []
    def fix_canonical_conflicts(self):
        for key,di in self.drug_info.items():
            s = di['canonical']
            # some DPI records use the BDBM key as the canonical name,
            # and others have the 'real' canonical name; see if
            # removing the key makes it unique
            if len(s) > 1:
                s = s - set([key])
                if len(s) > 1:
                    # actually, the canonical name is a multipart field
                    # (see _split_multipart_canonical), so we can combine
                    # them into a single superset of all the parts
                    parts = set()
                    for variant in s:
                        parts |= set(variant.split('::'))
                    s = set(['::'.join(sorted(parts))])
                    #print(f'restructuring canonical for {key}')
                #assert len(s) == 1,f'ambiguous canonical {key} {s}'
                di['canonical'] = s
    def show_stats(self):
        print(len(self.drug_info),'drugs')
        print(len(self.dpi),'drug/protein interactions')
        to_delete={}
        for key,di in self.drug_info.items():
            for attr,s in di.items():
                if len(s) != 1:
                    s2=to_delete.setdefault(attr,set())
                    s2.add(key)

        for attr in to_delete:
            print(attr,'was in conflict on',len(to_delete[attr]),'drugs')
            # don't output anything for conflicting assignments
            for key in to_delete[attr]:
                del(self.drug_info[key][attr])
    def _split_multipart_canonical(self,key,di):
        # Now, bindingdb names consist of one or more synonyms separated
        # by '::', and the combined string can be too long to upload. So,
        # split them here, select one as the canonical name, and move others
        # to different attributes. This needs to happen after the consistency
        # checks in show_stats. Any new attr names created here need to appear
        # in self.output_attrs in order to be written to the file.
        #
        # Any name less than 3 characters is either an error or not useful,
        # so they get tossed. Things over 50 characters are likewise tossed
        # as unwieldy. CHEMBL ids and pubchem cids get moved to the associated
        # attribute. The shortest name left is chosen as canonical. The rest
        # are synonyms. If nothing is left, use the bindingdb key as the
        # canonical name.
        # The patent info could be useful, but seems to include incorrect
        # labels that tie unrelated molecules together during clustering,
        # so we toss that as well.
        assert 'canonical' in di,f'no canonical field for {key}: {di}'
        parts = next(iter(di['canonical'])).split('::')
        names = set()
        chembl_ids = set()
        pubchem_cids = set()
        patent_re = re.compile(r'US[0-9]{7}')
        for part in parts:
            if part.startswith(':'):
                part = part[1:] #fix some typos
            if len(part) < 3:
                continue
            if len(part) > 50:
                continue
            if patent_re.match(part):
                continue
            if part.startswith('CHEMBL'):
                chembl_ids.add(part)
            elif part.startswith('cid_'):
                pubchem_cids.add(part[4:])
            else:
                names.add(part)
        if not names:
            names.add(key)
        names = sorted(names,key=lambda x:(len(x),x))
        di['canonical'] = set(names[:1])
        if len(names) > 1:
            di['synonym'] = set(names[1:])
        if len(chembl_ids) == 1:
            di['linked_chembl_id'] = chembl_ids
        if len(pubchem_cids) == 1 and 'pubchem_cid' not in di:
            di['pubchem_cid'] = pubchem_cids
    def create_attr_file(self,filename):
        with open(filename,'w') as out:
            has_dpi = set([x[0] for x in list(self.dpi.values())])
            out.write('bindingdb_id\tattribute\tvalue\n')
            attr_renames={
                    'drugbank_id':'linked_drugbank_id',
                    }
            for key,di in self.drug_info.items():
                # skip drugs w/ no DPI info
                if key not in has_dpi:
                    continue
                self._split_multipart_canonical(key,di)
                for attr in self.output_attrs:
                    if attr not in di:
                        continue
                    s = di[attr]
                    attr = attr_renames.get(attr,attr)
                    for v in s:
                        out.write('\t'.join([key,attr,v])+'\n')
    def create_sources_file(self, filename):
        with open(filename, 'w') as out:
            out.write('bindingdb_id\t%s\n' % '\t'.join(self.srcs))
            for entry in self.sources_data:
                out.write('\t'.join(entry) + '\n')


    def create_dpi_file(self,filename, dpimode):
        with open(filename,'w') as out:
            third_col = dpimode
            columns = ['bindingdb_id','uniprot_id',third_col,'direction']
            out.write('\t'.join(columns)+'\n')
            dpis = list(self.dpi.values())
            dpis.sort(key=lambda x:x[0])
            for dpi in dpis:
                if dpimode == 'C50':
                    # output multiple records, one for each IC50 or EC50
                    for x in dpi[2]:
                        # ic50
                        out.write('\t'.join([dpi[0],dpi[1],str(x),'-1']) +'\n')
                    for x in dpi[3]:
                         # ec50
                        out.write('\t'.join([dpi[0],dpi[1],str(x),'1']) + '\n')
                elif dpimode == 'Ki':
                    for x in dpi[4]:
                        out.write('\t'.join([dpi[0],dpi[1],str(x),'0']) +'\n')
                else:
                    raise NotImplementedError(f"unknown dpimode '{dpimode}'")

    def parse_input(self, inp_file, relevant_uniprots):
        from collections import namedtuple
        from dtk.files import get_file_records

        InputRow = None
        field_cnt = None
        key_idx = None
        smiles_idx = None
        conc_idxs = []
        ki_idx = None
        src_idxs = []
        for fields in get_file_records(inp_file, progress=True, keep_header=True):
            if not InputRow:
                field_cnt = len(fields)
                for i,n in enumerate(fields):
                    if n in ('IC50','EC50'):
                        conc_idxs.append(i)
                    elif n == 'bindingdb_id':
                        key_idx = i
                    elif n == 'smiles_code':
                        smiles_idx = i
                    elif n == 'Ki':
                        ki_idx = i
                    elif n in self.srcs:
                        src_idxs.append(i)

                InputRow = namedtuple('InputRow', fields)
                continue
            assert len(fields) == field_cnt
            fields = [x.strip() for x in fields]
            # Ignore any values that are not =, <, or <=
            for i in conc_idxs+[ki_idx]:
                replace = None
                clean = re.sub('[<=]', '', fields[i])
                if clean:
                    try:
                        replace = format(float(clean),'f')
                    except ValueError:
                        if verbose:
                            warning('unable to clean', fields[i])
                        pass
                fields[i] = replace
            # add prefix to key
            fields[key_idx] = 'BDBM'+fields[key_idx]
            key = fields[key_idx]
            # store cleaned-up input drug information
            row = InputRow._make(fields)
            di = self.drug_info.setdefault(key,{})
            for attr in self.attrs:
                v = getattr(row,attr)
                if v:
                    s = di.setdefault(attr,set())
                    s.add(v)


            # Store information about where data was sourced
            self.sources_data.append([key] + [fields[i] for i in src_idxs])

            # process DPI information
            if ',' in row.Primary_Uniprot_ID:
                # In a small set of cases, this is a comma-separated field
                # rather than a single ID.  According to ACD "these are
                # fusion proteins in diseases, and as such aren't likely
                # to be in 'normal' tissue".  They used to be omitted as
                # a side effect (the entire list, including the commas, was
                # looked up in the relevant protein list, and failed).  Here
                # I exclude them explicitly, so it's easier to untangle if
                # we need to revisit this.
                prots = []
            else:
                prots = [row.Primary_Uniprot_ID]
            prots += row.Alternative_Uniprot_IDs.split(',')
            for prot in prots:
                prot = prot.strip()
                if prot in relevant_uniprots:
                    dpi_key = key+'\t'+prot
                    dpi = self.dpi.setdefault(dpi_key,[key,prot,[],[],[]])
                    if row.Ki:
                        dpi[4].append(float(row.Ki))
                    if row.IC50:
                        dpi[2].append(float(row.IC50))
                    if row.EC50:
                        dpi[3].append(float(row.EC50))

def get_relevant_uniprot(uni_ver):
    from dtk.s3_cache import S3File
    uni_s3f = S3File.get_versioned('uniprot',uni_ver,'Uniprot_data')
    uni_s3f.fetch()
    from dtk.readtext import parse_delim
    return set([u for u,a,v in parse_delim(open(uni_s3f.path()))])

if __name__ == '__main__':
    import argparse
    arguments = argparse.ArgumentParser(description="This will parse a file downloaded and partially parsed from bindingDB")
    
    arguments.add_argument("-u", help="Relevant Uniprot files")
    
    arguments.add_argument("-i", help="bindingDB file")
    
    args = arguments.parse_args()

    data = InputData()
    relevant_uniprot = get_relevant_uniprot(args.u)
    data.parse_input(args.i, relevant_uniprot)
    data.fix_canonical_conflicts()
    data.show_stats()
    data.create_attr_file('ds.tmp')
    data.create_dpi_file('c50.tmp', 'C50')
    data.create_dpi_file('ki.tmp', 'Ki')
    data.create_sources_file('srcs.tmp')
