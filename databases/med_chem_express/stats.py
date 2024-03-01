#!/usr/bin/env python3

def make_datas(data_fn):
    from dtk.files import get_file_records

    mols = set()
    cas = set()
    syns = set()
    for mol, attr, val in get_file_records(data_fn, keep_header=False, progress=True):
        mols.add(mol)
        if attr == 'cas':
            cas.add(val)
        elif attr == 'synonym':
            syns.add(val)
            

    return dict(
        mols=mols,
        cas=cas,
        synonyms=syns,
    )

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=[
        'med_chem_express.full.v{version}.attributes.tsv'
        ])
