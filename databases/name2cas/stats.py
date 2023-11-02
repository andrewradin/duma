#!/usr/bin/env python3

def make_datas(fn):
    from dtk.files import get_file_records
    cas_set = set()
    names = set()
    pairs = set()

    for name, *cas_list in get_file_records(fn, keep_header=None, progress=True):
        cas_set.update(cas_list)
        names.add(name)
        for cas in cas_list:
            pairs.add(hash((name, cas)))

    return dict(cas_set=cas_set, names=names, cas_name_pairs=pairs)

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=['name2cas.v{version}.tsv'])
