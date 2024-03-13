#!/usr/bin/env python

from path_helper import PathHelper
import os
from dtk.files import get_file_records


# Usually this is run as:
#   ./build_test_data.py --outdir e2e_dataset/
#
# Given a set of drugs, we want to find a set of related proteins in include.
# We'll take e.g. ncats, find all direct & indirect prots, then filter
# out all our dataset (protein.tsv, dpi, ppi) to include only those prots
# This helps to create a dataset that runs very quickly.

# These correspond to the gwas dataset we're using.
EXTRA_PROTS = set([
        'Q15544',
        'Q6BDS2',
        'Q92625',
        ])
def make_matching_dir(out_dir):
    import subprocess
    import shutil
    from pathlib import Path
    from dtk.s3_cache import S3File
    ncats_evid = S3File.get_versioned('ncats', 'default.v1', role='evidence').path()
    ncats_attrs = S3File.get_versioned('ncats', 'full.v1', role='attributes').path()

    out_dir = Path(out_dir)

    merge_prefix = str(out_dir / 'matching.e2e_test.v1')
    merge_moa_prefix = str(out_dir / 'matching.e2e_test-moa.v1')

    merge_dpi_file = merge_prefix + '.dpimerge.tsv'
    merge_ingredients_file = merge_prefix + '.ingredients.tsv'
    merge_clusters_file = merge_prefix + '.clusters.tsv'
    merge_clusters_file2 = str(out_dir / 'matching.full.v1.clusters.tsv')
    merge_props_file = merge_prefix + '.props.tsv'

    merge_moa_dpi_file = merge_moa_prefix + '.dpimerge.tsv'
    moa_attrs_file = str(out_dir / 'moa.full.v1.attributes.tsv')

    with open(merge_ingredients_file, 'w') as f:
        f.write(os.path.basename(ncats_attrs) + '\n')

    # Not using any real clusters, but some things get angry if this file is empty, so put a useless line.
    with open(merge_clusters_file, 'w') as f:
        f.write("ncats_id\tNCATS-ABT-089")

    # Some test is pulling down this other matching file as well and confusing things, so just put this in place.
    with open(merge_clusters_file2, 'w') as f:
        f.write("ncats_id\tNCATS-ABT-089")

    shutil.copyfile(ncats_evid, merge_dpi_file)

    uniprot_fn = S3File.get_versioned('uniprot', 'HUMAN_9606.v1', role='Uniprot_data').path()

    DPI_TRANSFORM = os.path.join(PathHelper.databases, 'matching/dpi_transform.py')
    subprocess.check_call([
        DPI_TRANSFORM,
        '--rekey',
        '-u', uniprot_fn,
        '-v', 'v1',
        '-i', merge_dpi_file,
        '-o', merge_moa_dpi_file,
        '-a', moa_attrs_file,
    ])
    

    all_prots = set(x[1] for x in get_file_records(merge_dpi_file, keep_header=False))
    all_prots |= EXTRA_PROTS


    props_in_fn = os.path.join(PathHelper.databases, 'matching/prop_dictionary.tsv.master')
    shutil.copyfile(props_in_fn, merge_props_file)

    return all_prots


def write_filtered_ppi(prots, out_ppi):
    from dtk.s3_cache import S3File
    PPI_IN = S3File.get_versioned('string', 'default.v1', format='tsv').path()

    indir_prots = set(prots)

    with open(out_ppi, 'w') as f:
        header = True
        for row in get_file_records(PPI_IN, keep_header=True, select=(prots, None)):
            if header:
                header = False
                f.write('\t'.join(row) + '\n')
            if (row[0] in prots or row[1] in prots) and (float(row[2]) > 0.98 or row[1] in prots and row[0] in prots):
                indir_prots.add(row[1])
                # Make sure we have strong data for these select prots.  gwasig sigdif gets upset if there are no
                # PPI connections for any of the data.
                if row[0] in EXTRA_PROTS or row[1] in EXTRA_PROTS:
                    row[2] = '0.9'
                f.write('\t'.join(row) + '\n')
    
    from dtk.tsv_alt import SqliteSv
    SqliteSv.write_from_tsv(out_ppi.replace('.tsv', '.sqlsv'), out_ppi, [str, str, float, int])

    return indir_prots


def filter_file(fn_in, fn_out, dataset):
    with open(fn_out, 'w') as f:
        header = True
        for row in get_file_records(fn_in, keep_header=True):
            if header:
                header = False
                f.write('\t'.join(row) + '\n')
            if row[0] in dataset and not 'Ensembl' in row[1] and not 'Reactome' in row[1]:
                f.write('\t'.join(row) + '\n')

def filter_uniprots(out_dir, prots):
    prot_files = ['Uniprot_data', 'Protein_Entrez']
    for prot_file in prot_files:
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned('uniprot', 'HUMAN_9606.v1', role=prot_file)
        s3f.fetch()
        uniprot_fn = s3f.path()
        fn_out = os.path.join(out_dir, os.path.basename(uniprot_fn))
        filter_file(uniprot_fn, fn_out, prots)

def run():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--outdir", help="Where to write files")
    parser.add_argument("--dsname", help="Name for our mini files")
    args = parser.parse_args()

    dir_prots = make_matching_dir(args.outdir)
    print("Found %d direct targets" % len(dir_prots))
    indir_prots = write_filtered_ppi(prots=dir_prots, out_ppi=os.path.join(args.outdir, 'string.e2e_test.v1.tsv'))
    print("Found %d ind targets" % len(indir_prots))

    filter_uniprots(args.outdir, indir_prots)



if __name__ == "__main__":
    run()
