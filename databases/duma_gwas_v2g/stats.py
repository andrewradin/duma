#!/usr/bin/env python3

def make_datas(fn):
    from dtk.files import get_file_records
    variants = set()
    targets = set()
    assocs = set()
    for rec in get_file_records(fn, keep_header=False, progress=True):
        variant = rec[0]
        target = rec[4]
        if target == '-':
            # Ignore these, don't really care about unassociated variants.
            continue
        variants.add(variant)
        targets.add(target)
        assocs.add(hash((variant, target)))

    return dict(variants=variants, assocs=assocs, targets=targets)

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=['duma_gwas_v2g.v{version}.data.tsv.gz'])
