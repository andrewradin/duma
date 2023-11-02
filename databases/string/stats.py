#!/usr/bin/env python3

def make_datas(fn):
    from dtk.files import get_file_records
    targets = set()
    assocs = set()
    targets90 = set()
    assocs90 = set()
    for t1, t2, score, dr in get_file_records(fn, keep_header=False, progress=True):
        targets.add(t1)
        targets.add(t2)
        assocs.add(hash((t1, t2)))

        if float(score) >= 0.9:
            targets90.add(t1)
            targets90.add(t2)
            assocs90.add(hash((t1, t2)))

    return dict(targets=targets, assocs=assocs, targets90=targets90, assocs90=assocs90)

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=['string.default.v{version}.tsv'])