#!/usr/bin/env python3

def make_datas(data_fn):
    from dtk.files import get_file_records

    mols = set()
    for mol, attr, val in get_file_records(data_fn, keep_header=False, progress=True):
        mols.add(mol)

    return dict(
        mols=mols,
    )

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=[
        'cayman.full.v{version}.attributes.tsv'
        ])
