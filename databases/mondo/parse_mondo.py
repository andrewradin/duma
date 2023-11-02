#!/usr/bin/env python
def run(mondo_data, out_mappings, out_labels):
    from dtk.mondo import extract_mondo_xrefs
    from atomicwrites import atomic_write

    import json
    with open(mondo_data) as f:
        data = json.loads(f.read())

    mondo2other, mondo2name = extract_mondo_xrefs(data) 


    with atomic_write(out_mappings,overwrite=True) as f:
        for k, v in mondo2other.all_pairs():
            f.write(f'{k}\t{v}\n')

    with atomic_write(out_labels,overwrite=True) as f:
        for k, v in mondo2name.items():
            f.write(f'{k}\t{v}\n')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mondo-data', help='Mondo data file')
    parser.add_argument('--out-mappings', help='Mapping output file')
    parser.add_argument('--out-labels', help='Labels output file')
    args=parser.parse_args()
    run(**vars(args))