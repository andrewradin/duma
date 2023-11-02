#!/usr/bin/env python3

def make_datas(data_fn):
    from dtk.files import get_file_records

    prots = set()
    gene_sets = set()
    for gs, uni_list in get_file_records(data_fn, keep_header=True, progress=True):
        prots.update(set(uni_list.split(',')))
        gene_sets.add(gs)

    return dict(
        prots=prots,
        gene_sets=gene_sets,
    )

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=[
        'gene_ontology.v{version}.genesets.tsv'
        ])
