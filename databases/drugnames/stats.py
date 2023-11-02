#!/usr/bin/env python3

def make_datas(fn):
    from dtk.files import get_file_records
    mols = set()
    names = set()
    pairs = set()

    for name, mol_id in get_file_records(fn, keep_header=None, progress=True):
        mols.add(mol_id)
        names.add(name)
        pairs.add(hash((name, mol_id)))

    return dict(mols=mols, names=names, mol_name_pairs=pairs)

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=['drugnames.v{version}.tsv'])
