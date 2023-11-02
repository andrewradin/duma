#!/usr/bin/env python3

import sys
sys.path.insert(1,"../../web1")

def extract_indications(outfile):
    q=ch.DrugIndication.select().join(ch.MoleculeDictionary)
    from dtk.files import FileDestination
    header=[
            'chembl_id',
            'drugname',
            'mesh_id',
            'indication',
            'max_phase',
            ]
    with FileDestination(outfile,header=header) as out:
        for r in q:
            out.append((
                    r.molregno.chembl_id,
                    r.molregno.pref_name,
                    r.mesh_id,
                    r.mesh_heading,
                    r.max_phase_for_ind,
                    ))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract Chembl Indications')
    parser.add_argument('outfile')
    parser.add_argument("chembl_version", help="source db version (like chembl_23)")
    args = parser.parse_args()

    import importlib
    ch = importlib.import_module(args.chembl_version+'_schema')

    extract_indications(args.outfile)
