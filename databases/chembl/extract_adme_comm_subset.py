#!/usr/bin/env python3

import argparse
import logging
logger = logging.getLogger(__name__)
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

def load_adme_set(adme_fn):
    adme_set = set()
    for recs in get_file_records(adme_fn, progress=True, keep_header=None):
        adme_set.add(recs[0])
    return adme_set

def load_clustered_set(clusters_fn):
    # If we match any of these collections, the molecule is going to get imported
    # regardless, so might as well include the chembl info.
    target_cols = [
        'drugbank',
        'duma',
        'cayman',
        'selleckchem',
        'med_chem_express',
        'ncats',
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

            if idtype == 'chembl_id':
                my_ids.append(idval)

        if in_target_col:
            out.update(my_ids)
    return out

def load_dpi_to_moas(fn):
    from dtk.files import get_file_records
    from dtk.data import MultiMap
    recs = []
    for mol, prot, ev, dr in get_file_records(fn, keep_header=False):
        # We're intentionally ignoring direction here, we just want unique
        # target sets for now.
        recs.append([mol, prot])

    fwd = MultiMap(recs).fwd_map()
    moa2mol = [(frozenset(moa), mol) for mol, moa in fwd.items()]
    return MultiMap(moa2mol)


def run(output, full, clusters, adme, dpi):
    adme_set = load_adme_set(adme)
    clustered_set = load_clustered_set(clusters)

    from dtk.data import MultiMap
    from atomicwrites import atomic_write

    moa2mol_mm = load_dpi_to_moas(dpi)

    id2recs = {}

    seen_moas = set()

    mols_in = 0
    mols_out = 0
    extra_moa_mols = 0
    with atomic_write(output, overwrite=True) as f:
        recs = get_file_records(full, progress=True, keep_header=True)
        header = next(recs)
        f.write('\t'.join(header) + '\n')
        for molrecs in grouped(recs):
            mols_in += 1
            chemblid = molrecs[0][0]
            molattrs = [rec[1:] for rec in molrecs]
            moldata = MultiMap(molattrs).fwd_map()
            keep = False
            if int(list(moldata.get('max_phase', [0]))[0]) >= 1:
                keep = True
            elif chemblid in adme_set:
                keep = True
            elif chemblid in clustered_set:
                keep = True
            
            id2recs[chemblid] = molrecs


            if keep:
                mols_out += 1
                for molrec in molrecs:
                    f.write('\t'.join(molrec) + '\n')

                # It could be missing if it has no MoA.
                if chemblid in moa2mol_mm.rev_map():
                    moa = list(moa2mol_mm.rev_map()[chemblid])
                    assert len(moa) == 1, "There should be exactly 1 if it's in here at all."
                    moa = moa[0]
                    seen_moas.add(moa)

    
        for moa, mols in moa2mol_mm.fwd_map().items():
            if moa not in seen_moas:
                extra_moa_mols += 1
                mols_out += 1

                # Some of the ChEMBL molecules in the DPI file are not actually available
                # in chembl.full, probably because of molecule_type filtering.
                # Let's try to find one that we can use.
                avail_mols = [x for x in mols if x in id2recs]

                if not avail_mols:
                    logger.warning(f"No available molecules for MoA {moa} - {mols} all missing from chembl.full")
                    continue

                # Just pick the min chembl ID as an arbitrary tie break.
                exemplar_id = min(avail_mols)
                recs = id2recs[exemplar_id]
                for molrec in recs:
                    f.write('\t'.join(molrec) + '\n')

    logger.info(f'Kept {mols_out} of {mols_in} molecules, added {extra_moa_mols} missing MoA exemplars.') 

def main():
    arguments = argparse.ArgumentParser(description="Subsets chembl.full to pull out only keys of interest")

    arguments.add_argument('-o', '--output', required=True, help="Output file to write")
    arguments.add_argument("--full", required=True, help="Full attributes tsv to subset from")
    arguments.add_argument("--clusters", required=True, help="Cluster matching file")
    arguments.add_argument("--adme", required=True, help="ADME assays file")
    arguments.add_argument("--dpi", required=True, help="DPI to ensure we get exemplars of all unique MoAs")

    args = arguments.parse_args()
    from dtk.log_setup import setupLogging
    setupLogging()
    run(**vars(args))
    



if __name__ == "__main__":
    main()
