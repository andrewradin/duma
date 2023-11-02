#!/usr/bin/env python3

from atomicwrites import atomic_write

import logging
logger = logging.getLogger(__name__)

from tqdm import tqdm
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse
import numpy as np
import os

def process_chunk(chunk, sig_info, gene_info, treat_fn):
    chunk_ids = [x[1] for x in chunk]
    chunk_sample_ids = sig_info["sig_id"][sig_info["pert_id"].isin(chunk_ids)]

    if len(chunk_sample_ids) > gene_info.shape[0]:
        logger.error(f"Trying to pull {len(chunk_sample_ids)} for {chunk_ids} is probably going to OOM")
        logger.error("Try a smaller chunk size or just bump memory on machine and ignore this error")

    chunk_samples = parse(treat_fn, cid=chunk_sample_ids)

    out = []

    for idx, sample_pert_id in chunk:
        sample_sig_ids = sig_info["sig_id"][sig_info["pert_id"]==sample_pert_id]
        df = chunk_samples.data_df[sample_sig_ids]

        # TODO: Something better than mean for aggregating across cells/dosage/duration.
        # See notebook for filtering examples.
        # This is just a placeholder for now.
        expr = df.mean(axis=1).values
        out.append((idx, expr, df.index.tolist()))

    return out


def make_gene2prot(uniprot_ver):
    from dtk.s3_cache import S3File
    uni_s3f = S3File.get_versioned('uniprot','HUMAN_9606.'+uniprot_ver,'Uniprot_data')
    out = {}
    from dtk.files import get_file_records
    for uni, attr, val in get_file_records(uni_s3f.path(), keep_header=False):
        if attr in ['GeneID', 'Ensembl']:
            out[str(val)] = uni
    return out


def run(input_data_dir, input_attrs, out_expr, out_meta, chunk_size, uniprot_ver):
    gene2prot = make_gene2prot(uniprot_ver)
    logging.getLogger('cmap_logger').setLevel(logging.WARNING)
    attrs = pd.read_csv(input_attrs, sep="\t")
    ids = sorted(set(attrs['lincs_id'].values.tolist()))

    sig_fn = os.path.join(input_data_dir, 'siginfo_beta.txt')
    gene_fn = os.path.join(input_data_dir, 'geneinfo_beta.txt')
    treat_fn = os.path.join(input_data_dir, 'level5_beta_trt_cp_n720216x12328.gctx')

    logger.info("Loading sig info")
    sig_info = pd.read_csv(sig_fn, sep="\t")
    logger.info("Loading gene info")
    gene_info = pd.read_csv(gene_fn, sep="\t", index_col='gene_id')


    uni_col = []
    from collections import defaultdict
    missing_by_type = defaultdict(int)
    missing_by_fs = defaultdict(int)
    for gene_id, ensg, gene_type, feature_space in zip(gene_info.index, gene_info['ensembl_id'], gene_info['gene_type'], gene_info['feature_space']):
        gene_id = str(gene_id)
        if gene_id in gene2prot:
            uni_col.append(gene2prot[gene_id])
        elif ensg in gene2prot:
            uni_col.append(gene2prot[ensg])
        else:
            # There are a lot, this is noisy
            if gene_type == 'protein-coding':
                print(f"Unknown gene {gene_id} {ensg} {gene_type} {feature_space}")
            missing_by_type[gene_type] += 1
            missing_by_fs[feature_space] += 1
            uni_col.append('')
    logger.info(f"Missing uniprots by type: {missing_by_type}")
    logger.info(f"Missing uniprots by fs: {missing_by_fs}")
    gene_info['uniprot'] = uni_col

    indexed_ids = list(enumerate(ids))
    from dtk.parallel import chunker, pmap

    logger.info("Preallocating array")

    # We're using float16 here because the data is both huge and noisy, so we want to shrink it
    # and don't really need much precision.
    # float16 is ~3.3 decimal places of precision (and more exponent bits than we need)
    out_df = np.empty((gene_info.shape[0], len(indexed_ids)), dtype=np.float16)

    logger.info("Parsing in parallel chunks")


    # We can't parse the whole file all at once because we run out of memory.
    # Instead, we pull out chunks of ids and process those.
    # Bigger chunk sizes will work faster at the cost of more memory.
    # But if the chunk size is too large we hit a bug in the cmap parser and allocate
    # the entire array and run out of memory. (https://github.com/cmap/cmapPy/issues/71)
    chunks = chunker(indexed_ids, chunk_size=chunk_size)
    static_args = dict(
            sig_info=sig_info,
            gene_info=gene_info,
            treat_fn=treat_fn
            )

    gene_ids = None
    for chunk_output in pmap(process_chunk, chunks, static_args=static_args, progress=True):
        for idx, expr, chunk_gene_ids in chunk_output:
            out_df[:, idx] = expr
            if gene_ids is None:
                gene_ids = chunk_gene_ids
            assert gene_ids == chunk_gene_ids
     
    logger.info("Writing out expression data")
    with atomic_write(out_expr, overwrite=True, mode='wb') as f:
        np.savez_compressed(f, expr=out_df)

    logger.info("Writing out metadata")
    with atomic_write(out_meta, overwrite=True) as f:
        drug_meta = {
            'order': [x[1] for x in indexed_ids],
                }
        gene_meta = {
            'order': gene_ids,
            'info': gene_info.to_dict(),
            }
        meta = {
                'drugs': drug_meta,
                'genes': gene_meta,
                }
        
        import json
        f.write(json.dumps(meta, indent=2))



if __name__ == "__main__":
    import argparse
    arguments = argparse.ArgumentParser(description="Creates a drugset for lincs data.")

    arguments.add_argument('--out-expr', help="Expression Output file")
    arguments.add_argument('--out-meta', help="Metadata Output file")
    arguments.add_argument('--input-data-dir',  help="Should contain gctx file and metadatas")
    arguments.add_argument('--input-attrs', help="Attrs file with ids to parse")
    arguments.add_argument('--chunk-size', default=100, type=int, help="size of chunks to process")
    arguments.add_argument('-u', '--uniprot-ver', help="Uniprot version for converting")
    args = arguments.parse_args()
    from dtk.log_setup import setupLogging
    setupLogging()

    run(**vars(args))

