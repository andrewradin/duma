#!/usr/bin/env python3

def make_datas(fn):
    from dtk.files import get_file_records
    diseases = set()
    targets = set()
    assocs = set()
    for dis, targ, score, val in get_file_records(fn, keep_header=False, progress=True):
        diseases.add(dis)
        targets.add(targ)
        assocs.add((dis, targ))

    return dict(diseases=diseases, targets=targets, assocs=assocs)

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=['disgenet.v{version}.curated_umls.tsv'])