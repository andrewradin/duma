#!/usr/bin/env python3

from atomicwrites import atomic_write

import logging
logger = logging.getLogger(__name__)


# We want all our ids from here to start with BRD, but some don't.
# We also need to use these IDs  as references into the LINCS datafiles.
# When BRD is missing we add this prefix, which as appropriate other code can look
# for and strip.
MISSING_BRD_PREFIX = 'BRD-__'

def make_gene_to_uniprot(uniprot_version):
    from dtk.s3_cache import S3File
    from dtk.files import get_file_records
    ver = f'HUMAN_9606.{uniprot_version}'
    uni_s3f = S3File.get_versioned('uniprot',ver,'Uniprot_data')
    uni_s3f.fetch()
    out = {}
    for rec in get_file_records(uni_s3f.path(), keep_header=False):
        if rec[1] == 'Gene_Name' or rec[1] == 'Gene_Synonym':
            out[rec[2]] = rec[0]
    return out


def run(input, out_attr, out_dpi, uniprot_version):
    prop_map = {
            'cmap_name': 'canonical',
            'compound_aliases': 'synonym',
            'canonical_smiles': 'smiles_code',
            }
    seen = set()

    dpis = []
    logger.info("Making attr file")

    with atomic_write(out_attr, overwrite=True) as f:
        f.write('\t'.join(['lincs_id', 'attribute', 'value']) + '\n')
        from dtk.files import get_file_records
        header = None
        prev_id = None
        for recs in get_file_records(input, progress=True, keep_header=True, parse_type='tsv', sort=True):
            if header is None:
                header = recs
                continue
            row = dict(zip(header, recs))

            if row['canonical_smiles'] == '""' and row['target'] == '""':
                # Can't do much with these ones, just skip.
                continue

            id = row['pert_id']

            if not id.startswith('BRD'):
                id = MISSING_BRD_PREFIX + id

            if row['target'] != '""':
                dpis.append((id, row['target']))

            for row_name, attr_name in prop_map.items():
                val = row[row_name]
                if val and val != '""':
                    if id == prev_id and attr_name == 'canonical':
                        # Sometimes they add extra synonyms as extra lines, sometimes as aliases.
                        attr_name = 'synonym'
                    # There are duplicate attr lines due to having 1 target per line with
                    # everything else duplicated - skip over.
                    out = (id, attr_name, val)
                    if out in seen:
                        continue
                    seen.add(out)
                    if attr_name == 'canonical':
                        seen.add((id, 'synonym', val))
                    f.write('\t'.join(out) + '\n')

            prev_id = id

    logger.info("Making dpi file")
    g2u = make_gene_to_uniprot(uniprot_version)

    from dtk.data import MultiMap
    dpi_mm = MultiMap(dpis)
    with atomic_write(out_dpi, overwrite=True) as f:
        f.write('\t'.join(['lincs_id', 'uniprot_id', 'evidence', 'direction']) + '\n')
        for l_id, targets in dpi_mm.fwd_map().items():
            for target in targets:
                uniprot = g2u.get(target, None)
                if uniprot is None:
                    logger.info(f"Couldn't find uniprot for {target}")
                    continue
                row = [l_id, uniprot, '0.9', '0']
                f.write('\t'.join(row) + '\n')

    logger.info(f'Wrote dpi for {len(dpi_mm.fwd_map())} entries')

if __name__ == "__main__":
    import argparse
    arguments = argparse.ArgumentParser(description="Creates a drugset for lincs data.")

    arguments.add_argument('--out-attr', help="Attrs output file")
    arguments.add_argument('--out-dpi', help="Evidence output file")
    arguments.add_argument('-u', '--uniprot-version', help="Uniprot converter version")
    arguments.add_argument('-i', '--input',  help="Compound info file")
    args = arguments.parse_args()
    from dtk.log_setup import setupLogging
    setupLogging()

    run(**vars(args))

