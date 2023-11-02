#!/usr/bin/env python

from dtk.files import get_file_records
def load_converter(fn):
    d = {}

    # file order is:
    # Gene stable ID	Transcript stable ID	NCBI gene (formerly Entrezgene) ID
    # where the first two are ensembl

    for gene,transcript,entrez in get_file_records(fn, keep_header=False):
        if entrez not in d:
            d[entrez]=set()
        d[entrez].add(transcript)
    return d

if __name__=='__main__':
    import argparse
    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="convert from entrez gene ids to ensemblTRS ids for non human species")
    arguments.add_argument("i", help="input file")
    arguments.add_argument("c", help="entrez to ensembl file")
    arguments.add_argument("o", help="output file name")
    args = arguments.parse_args()
    print(args)

    converter = load_converter(args.c)

    from atomicwrites import atomic_write
    with atomic_write(args.o) as f:
        for uniprot,entrez in get_file_records(args.i, keep_header=False):
            if entrez in converter:
                for t in converter[entrez]:
                    f.write("\t".join([
                                       t,
                                       uniprot,
                                      ])
                           +"\n")

