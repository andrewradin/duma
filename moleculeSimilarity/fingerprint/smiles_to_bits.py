#!/usr/bin/env python

import argparse

import sys

def get_bits(id,smiles):
    from similarity_csv import get_rdkit_fingerprint
    t = [[smiles,id]]
    return get_rdkit_fingerprint((t,0,2,True,False))

if __name__ == "__main__":
    parser=argparse.ArgumentParser(
            description='calculate similarities to specified drugs',
            )
    parser.add_argument('s'
            ,type=str
            ,default='smiles.tsv'
            ,help='file of smiles codes (default %(default)s)'
            )
    args = parser.parse_args()
    try:
        from dtk.files import get_file_records
    except ImportError:
        sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
        from dtk.files import get_file_records
    from dtk.mol_struct import getCanonSmilesFromSmiles

    for (id,smiles) in get_file_records(args.s):
        results = get_bits(id, getCanonSmilesFromSmiles(smiles))
        for k,v in results[-1].iteritems():
            print "\t".join([str(x) for x in [id,k,v]])

