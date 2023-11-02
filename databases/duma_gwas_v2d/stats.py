#!/usr/bin/env python3

def make_datas(fn):
    from dtk.files import get_file_records
    variants = set()
    studies = set()
    assocs = set()
    for rec in get_file_records(fn, keep_header=None, progress=True):
        study = rec[0]
        variant = (rec[2], rec[3])

        variants.add(variant)
        studies.add(study)
        assocs.add(hash((variant, study)))

    return dict(variants=variants, assocs=assocs, studies=studies)

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=['duma_gwas_v2d.v{version}.data.tsv.gz'])