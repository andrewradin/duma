#!/usr/bin/env python3

def make_datas(rct_fn, combined_fn):
    # Earlier versions have only the reactome file; later have the combined.
    fn = combined_fn or rct_fn
    assert fn is not None, "Must have one of the files"
    from dtk.files import get_file_records
    pathways = set()
    targets = set()
    for pw, targets_str in get_file_records(fn, keep_header=False, progress=True):
        pathways.add(pw)
        targets.update(targets_str.split(','))


    return dict(pathways=pathways, targets=targets)

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=[
        'pathways.reactome.all.v{version}.genesets.tsv',
        'pathways.combined.all.v{version}.genesets.tsv',
        ])