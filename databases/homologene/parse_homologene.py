#!/usr/bin/env python

from dtk.files import get_file_records

# The format of this file is:
# clusterID, tax_ID, enrez_gene_id, gene_name...other identifiers


class parser:
    def __init__(self, infile, uniprot_file, taxids):
        self.uni_file = uniprot_file
        self.human_tax_id = '9606'
        self.ifile = infile
        self.other_tax_ids = taxids
    def run(self):
        self._load_homologene()
        self._convert_uni()
    def _convert_uni(self):
        self.out_d = {}
        for frs in get_file_records(self.uni_file, keep_header=False):
# the GeneID is to avoid duplicates b/c hgnc uses the same keys
            if frs[2] in self.entrez_d and frs[1]=='GeneID':
                for tax,gene in self.entrez_d[frs[2]].items():
                    if tax not in self.out_d:
                        self.out_d[tax]={}
                    self.out_d[tax][frs[0]] = gene
    def _load_homologene(self):
        self.entrez_d = {}
        self.current_key = None
        self._reset()
        for frs in get_file_records(self.ifile, parse_type='tsv'):
            if frs[1] in [self.human_tax_id]+self.other_tax_ids:
                if self.current_key is None:
                    self.current_key = frs[0]
                if self.current_key != frs[0]:
                    self._process()
                    self.current_key = frs[0]
                self._load_line(frs)
    def _process(self):
        if self.other and self.human:
            assert self.human not in self.entrez_d
            self.entrez_d[self.human] = self.other
        self._reset()
    def _reset(self):
        self.human = None
        self.other = {}
    def _load_line(self, frs):
        if frs[1] == self.human_tax_id:
            self.human = frs[2]
        elif frs[1] in self.other_tax_ids:
            self.other[frs[1]] = frs[2]
        else:
            print('something went wrong with', frs)
    def write(self):
        for tax,d in self.out_d.items():
            with open(f'{tax}.tmp', 'w') as f:
                f.write("\t".join(['Uniprot','otherSpeciesGeneID'])+"\n")
                for u,e in d.items():
                    f.write("\t".join([u,e]) + "\n")


if __name__=='__main__':
    import argparse
    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="parse the homologene for a specific taxID to generate a Human Uniprot to provided organism entrez ID")
    arguments.add_argument("i", help="input file")
    arguments.add_argument("u", help="UniProt file")
    arguments.add_argument("taxids", help="taxid", nargs='+')
    args = arguments.parse_args()
    print(args)

    from dtk.s3_cache import S3File
    uniprot_s3f=S3File.get_versioned(
            'uniprot',
            args.u,
            role='Uniprot_data',
            )
    uniprot_s3f.fetch()


    run=parser(args.i, uniprot_s3f.path(), args.taxids)
    run.run()
    run.write()
