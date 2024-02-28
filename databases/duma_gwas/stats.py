#!/usr/bin/env python3

def make_datas(fn):
    from dtk.files import get_file_records
    studies = set()
    targets = set()
    assocs = set()
    for rec in get_file_records(fn, keep_header=False, progress=True):
        study = rec[0]
        gene = rec[5]
        studies.add(study)
        targets.add(gene)
        assocs.add(hash((study, gene)))

    return dict(studies=studies, assocs=assocs, targets=targets)

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=['duma_gwas.v{version}.data.tsv.gz'])