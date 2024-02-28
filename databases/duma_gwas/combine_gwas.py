#!/usr/bin/env python

### This is the simplest way to do this:
### only studies which have not previously been seen are added.
### This was done b/c we originally had 2 datasets,
### and the first, GRASP is more trustworthy than the 2nd (GWAS Catalog) 
### As a result, the order in which the datasets are provided may change the results.
### We may also want to re-think this if we add a third source.
### UPDATE: we did add a third source, but it was totally non-overlapping
### so we didn't bother re-thinking this

import sys
try:
    from dtk.files import get_file_records
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
    from dtk.files import get_file_records

try:
    from parse_grasp import pmid_from_k
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../grasp")
    from parse_grasp import pmid_from_k
from filter_gwas import k_is_good

class gwas_combiner(object):
    def __init__(self,**kwargs):
        self.snp_file = kwargs.get('snp',None)
        self.studies_file = kwargs.get('studies',None)
        self.ofile = kwargs.get('out',None)
        self.osnps = kwargs.get('osnps',None)
        self.studies_to_add = set()
        self._setup()
    def _setup(self):
        self._load_snp_studies()
        self._copy_studies_file()
    def _copy_studies_file(self):
        header = None
        with open(self.ofile, 'w') as f:
            for frs in get_file_records(self.studies_file, keep_header=True):
                if header is None:
                    f.write("\t".join(frs) + "\n")
                    header = True
                if frs[0] in self.existing_keys:
                    f.write("\t".join(frs) + "\n")
    def _load_snp_studies(self):
        self.existing_studies = set()
        self.existing_keys = set()
        for frs in get_file_records(self.snp_file, parse_type='tsv',
                                    keep_header=False):
            if k_is_good(frs[0]):
                self.existing_keys.add(frs[0])
                self.existing_studies.add(pmid_from_k(frs[0]))
                print("\t".join(frs))
    def add_db(self, studies_file, snp_file):
        new_pmids = set()
        for frs in get_file_records(snp_file, keep_header=True):
            if not k_is_good(frs[0]):
                continue
            pmid = pmid_from_k(frs[0])
            if pmid in self.existing_studies:
                continue
            print("\t".join(frs))
            new_pmids.add(pmid)
        for frs in get_file_records(studies_file, keep_header=True):
            if pmid_from_k(frs[0]) in new_pmids:
                self.studies_to_add.add("\t".join(frs))
        self.existing_studies.update(new_pmids)
    def report(self):
        with open(self.ofile, 'a') as f:
            for x in self.studies_to_add:
                f.write(x+"\n")

if __name__=='__main__':
    import argparse
    
    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="Parse full GRASP DB into our format")
    arguments.add_argument("snp", help="GRASP SNP data")
    arguments.add_argument("studies", help="GRASP studies data")
    arguments.add_argument("ostudies", help="studies output file")
    arguments.add_argument("-s", nargs='*', help="Study data")
    arguments.add_argument("-i", nargs='*', help="SNP data; files must be in the same order as the study data files")
    args = arguments.parse_args()
    
    gc = gwas_combiner(snp=args.snp, studies=args.studies, out=args.ostudies)#, osnps=args.osnps)
    assert len(args.s) == len(args.i)
    for i in range(len(args.s)):
        gc.add_db(args.s[i], args.i[i])
    gc.report()


