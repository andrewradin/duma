#!/usr/bin/python
from __future__ import print_function
import sys, argparse

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def useage(*objs):
    print("USEAGE: ", *objs, file=sys.stderr)
    
def printOut(*objs):
    print(*objs, file=sys.stdout)

if __name__=='__main__':
    #=================================================
    # Read in the arguments/define options
    #=================================================
    
    arguments = argparse.ArgumentParser(description="This will parse a file downloaded and gunzipped from CPDB")

    arguments.add_argument("-u", help="ENSP to Uniprot conversion file")

    arguments.add_argument("-i", help="ConsensusPathDB_human_PPI.tsv")
    
    arguments.add_argument("-m", help="Median value of non-NA evidence scores")

    args = arguments.parse_args()
     
    # return usage information if no argvs given
    
    if not args.i or not args.u or not args.m:
        arguments.print_help()
        sys.exit(10)
    
    if float(args.m) == 0.0:
        warning("median evidence score was 0. Reporting an evidence score of 1 for all PPI.")
        args.m = '1.0'
    
    # Initialize a dictionary that has as a key a gene a name, and as a value a list of UniProt IDs
    uniProtConverter = {}
    # use a loop to read each line and pull out the info I need and store the pairs in a hash of arrays
    with open(args.u, 'r') as f:
        for l in f:
            u,a,v = l.rstrip().split("\t")
            if a != 'UniProtKB-ID':
                continue
            uniProtConverter.setdefault(v, set()).add(u)
    
    with open(args.i, 'r') as f:
        printOut("\t".join(['prot1', 'prot2', 'evidence', 'direction', 'sources']))
        for l in f:
            if(l.startswith("#")):
                continue
            info = l.rstrip().split("\t")
            # pull out the relevant bits
            prots = [p.rstrip('_HUMAN') for p in info[2].split(',')]
            sources = info[0]
            evidence = info[3] if info[3] != 'NA' else args.m
        #### From CPDB: "In cases when proteins are annotated only with genomic identifiers
        #### but no protein identifiers in the according source databases, and if the genomic identifiers map to more than one UniProt entry,
        #### the according UniProt entry names are concatenated (e.g. RL40_HUMAN.RS27A_HUMAN.UBIQ_HUMAN)
        #### as it is unclear which of the gene products interact."
        ##### We took the conservative approach of ignoring those interactions rather than putting in false interactions.
        ##### likely only one in the concatenated list is interacting, so the rest are noise
            for i in xrange(len(prots)):
                second_gen = (j for j in xrange(len(prots)) if j > i)
                for j in second_gen:
                    try:
                        lefts = uniProtConverter[prots[i]]
                        rights = uniProtConverter[prots[j]]
                    except KeyError:
                        continue
                    for left in lefts:
                        for right in rights:
                            printOut("\t".join([left, right, evidence, '0', sources]))
