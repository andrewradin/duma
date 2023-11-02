#!/usr/bin/env python3

def make_datas(attr_fn):
    from dtk.files import get_file_records
    attr_mols = set()
    attrs = set()
    for mol, attr_name, attr_val in get_file_records(attr_fn, keep_header=False, progress=True):
        attr_mols.add(mol)
        attrs.add((mol, attr_name, attr_val))

    return dict(attr_mols=attr_mols, attrs=attrs)

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=['moa.full.v{version}.attributes.tsv'])
