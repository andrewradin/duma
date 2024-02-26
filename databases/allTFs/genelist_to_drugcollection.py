#!/usr/bin/env python
import sys

class combine(object):
    def __init__(self, **kwargs):
        self.input_file = kwargs.get('i', None)
        self.uniprot_file = kwargs.get('u', None)
        self.dpi_file = kwargs.get('o1', None)
        self.out_file = kwargs.get('o2', None)
        self.collection_name = kwargs.get('c', None)
    def run(self):
        try:
            from dtk.files import get_file_records
        except ImportError:
            sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
            from dtk.files import get_file_records
        all_genes = set()
        for frs in get_file_records(self.input_file):
            all_genes.add(frs[0])
        print len(all_genes), 'genes loaded'
        self.gene2prot={}
        for frs in get_file_records(self.uniprot_file):
            if frs[1] in all_genes:
                if frs[1] not in self.gene2prot:
                    self.gene2prot[frs[1]]=set()
                self.gene2prot[frs[1]].add(frs[0])
        print len(self.gene2prot), 'genes mapped to', len(set.union(*self.gene2prot.values())), 'uniprot IDs'
    def dump(self):
        with open(self.out_file, 'w') as f:
            with open(self.dpi_file, 'w') as f2:
                f.write("\t".join([self.collection_name+"_id",
                                   'attribute',
                                   'value'
                                  ])+ "\n"
                       )
                f2.write("\t".join([self.collection_name+"_id",
                                   'uniprot_id',
                                   'evidence',
                                   'direction'
                                  ])+ "\n"
                       )
                for k,s in self.gene2prot.iteritems():
                    f.write("\t".join([k, 'canonical', k]) + "\n")
                    for u in s:
                        f2.write("\t".join([k, u, '1.0', '0']) + "\n")

if __name__ == '__main__':
    import argparse

    arguments = argparse.ArgumentParser(description="Create a virtual drug collection for a list of genes (TFs in this case)")
    arguments.add_argument("i", help="e.g. TFs_Ensembl_v_$(V).txt")
    arguments.add_argument("u", help="e.g. HUMAN_9606_Protein_Ensembl.tsv")
    arguments.add_argument("o1", help="the dpi output")
    arguments.add_argument("o2", help="the create file output")
    arguments.add_argument("c", help="collection name")
    args = arguments.parse_args()

    c = combine(i=args.i,u=args.u,o1= args.o1, o2=args.o2,c=args.c)
    c.run()
    c.dump()
