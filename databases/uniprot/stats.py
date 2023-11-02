#!/usr/bin/env python3
def make_datas(fn):
    from dtk.files import get_file_records
    uniprots = set()

    attrs = {'Alt_uniprot', 'hgnc', 'Ensembl', 'GeneID'}
    from collections import defaultdict
    data = defaultdict(set)

    for uni, key, value in get_file_records(fn, keep_header=False, progress=True):
        uniprots.add(uni)
        if key in attrs:
            data[key].add((uni, value))

    return dict(uniprots=uniprots, **data)

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=['uniprot.HUMAN_9606.v{version}.Uniprot_data.tsv'])
