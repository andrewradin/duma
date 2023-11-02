#!/usr/bin/env python3

def make_datas(fn):
    from dtk.files import get_file_records
    diseases = set()
    targets = set()
    assocs = set()
    for dis, targ, score, val in get_file_records(fn, keep_header=False, progress=True):
        diseases.add(dis)
        targets.add(targ)
        # Use a hash here because we run into memory limits.
        # In theory we could hit a hash collision, but wouldn't matter at all to the stats.
        # Checking v5, we have 8326444 assocs with no collisions.
        assoc = hash((dis, targ))
        assocs.add(assoc)

    return dict(diseases=diseases, targets=targets, assocs=assocs)

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=['openTargets.v{version}.data.tsv.gz'])