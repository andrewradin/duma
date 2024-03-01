#!/usr/bin/env python

from __future__ import print_function
import sys
sys.path.insert(1,"../../web1")
from path_helper import PathHelper

def message(*objs):
    print(*objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def verboseOut(*objs):
    if verbose:
        print(*objs, file=sys.stderr)

def printOut(*objs):
    print(*objs, file=sys.stdout)

if __name__=='__main__':
    #=================================================
    # Read in the arguments/define options
    #=================================================
    import argparse
    arguments = argparse.ArgumentParser(description="convert gottlieb tsv file")
    
    arguments.add_argument("infile", help="UMLS-keyed input")
    arguments.add_argument("outfile", help="MedDRA-keyed output")
    
    args = arguments.parse_args()

    meddra_map = '../umls/umls_to_meddra.tsv'

    # read in mappings for UMLS ids
    umls2meddra = {}
    for line in open(meddra_map):
        rec = line.strip('\n').split('\t')
        if not rec:
            continue
        umls2meddra[rec[0]] = rec[1]

    # scan tsv file and output ADR records
    inp = open(args.infile)
    outp = open(args.outfile,'w')
    skipped = 0
    wrote = 0
    for line in inp:
        rec = line.strip('\n').split('\t')
        umls = rec[1].upper()
        if umls not in umls2meddra:
            skipped += 1
            warning(umls
                    ,'not in meddra map; skipping'
                    ,rec[0]
                    )
        else:
            wrote += 1
            outp.write('\t'.join([
                    rec[0],
                    umls2meddra[umls],
                    rec[2],
                    ])+'\n')
    print('wrote',wrote,'skipped',skipped)
