#!/usr/bin/env python
import os


from tqdm import tqdm
import logging
logger = logging.getLogger("pubchem")

from dtk.files import get_file_records

def grouped(records):
    out = []
    prev_key = None
    for rec in records:
        if rec[0] != prev_key and out:
            yield out
            out = []
        prev_key = rec[0]
        out.append(rec)
    if out:
        yield out

def load_clustered_set(clusters_fn):
    # If we match any of these collections, the molecule is going to get imported
    # regardless, so might as well include the chembl info.
    target_cols = [
        'drugbank',
        'chembl',
        ]
    target_cols = {x + '_id' for x in target_cols}

    from dtk.drug_clusters import assemble_pairs
    out = set()
    for rec in get_file_records(clusters_fn, keep_header=None, progress=True):
        my_ids = []
        other_ids = []
        in_target_col = False
        for idtype, idval in assemble_pairs(rec):
            in_target_col |= idtype in target_cols

            if idtype == 'pubchem_id':
                my_ids.append(idval)

        if in_target_col:
            out.update(my_ids)
    return out

def run(input_fn, clusters_fn, out_fn):
    ids_to_keep = load_clustered_set(clusters_fn)

    from dtk.data import MultiMap
    from atomicwrites import atomic_write

    mols_in = 0
    mols_out = 0
    with atomic_write(out_fn, overwrite=True) as f:
        recs = get_file_records(input_fn, progress=True, keep_header=True)
        header = next(recs)
        f.write('\t'.join(header) + '\n')
        for molrecs in grouped(recs):
            mols_in += 1
            pubchemid = molrecs[0][0]
            molattrs = [rec[1:] for rec in molrecs]
            moldata = MultiMap(molattrs).fwd_map()
            keep = False
            if pubchemid in ids_to_keep:
                keep = True

            if keep:
                mols_out += 1
                for molrec in molrecs:
                    f.write('\t'.join(molrec) + '\n')
    logger.info(f'Kept {mols_out} of {mols_in} molecules')



if __name__ == "__main__":
    from dtk.log_setup import setupLogging
    import argparse
    arguments = argparse.ArgumentParser(description="Filters data from pubchem.")

    arguments.add_argument('-i', '--input', help="Full attr file")
    arguments.add_argument('-c', '--clusters', help="Cluster output file")
    arguments.add_argument('-o', '--output', help="Where to write the output")
    args = arguments.parse_args()


    setupLogging()
    run(args.input, args.clusters, args.output)



