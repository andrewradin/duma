#!/usr/bin/env python3

from atomicwrites import atomic_write

"""
Each CSV file contains a table of uniprots and genes.
We associate each csv file with one (or more) global protein sets in the 
inputs list below.

These are put into a single table and pushed to S3.

On the client side, each protset here becomes a global protset, containing
all uniprots directly mentioned or associated with mentioned genes.
"""

def run(out_fn):
    inputs = [
        ('tier1.tsv', ('unwanted_tier1', 'Tier1 Intolerable Targets')),
        ('tier1.tsv', ('unwanted_tier2', 'Tier2 Intolerable Targets')),
        ('tier2.tsv', ('unwanted_tier2', 'Tier2 Intolerable Targets')),
    ]

    from dtk.files import get_file_records
    with atomic_write(out_fn, overwrite=True) as f:
        row = ["Uniprot", "Gene", "SetId", "SetName"]
        f.write('\t'.join(row) + "\n")

        for fn, (id, name) in inputs:
            for uniprot, gene in get_file_records(fn, keep_header=False):
                row = [uniprot, gene, id, name]
                f.write('\t'.join(row) + "\n")


if __name__ == "__main__":
    import argparse
    arguments = argparse.ArgumentParser(description="Generates global protsets file.")
    
    arguments.add_argument('-o', '--output', help="Where to write the output")
    args = arguments.parse_args()
    run(args.output)

