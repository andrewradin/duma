#!/usr/bin/env python3
import logging
from tqdm import tqdm
logger = logging.getLogger('parse_tractability')


def parse(in_fn, out_fn, gene2uni):
    import pandas as pd
    df = pd.read_csv(in_fn, sep='\t')

    # This datafile has a uniprot, but let's prefer our conversion.
    # Best is ens, then gene, then fallback to built-in uniprot accession.
    uniprots = [gene2uni.get(ens, gene2uni.get(gene, acc)) for gene, ens, acc in df[['symbol', 'ensembl_gene_id', 'accession']].itertuples(index=False)]
    df['uniprot'] = uniprots

    cols = df.columns.tolist()
    row_data = [x[1] for x in df.iterrows()]

    from dtk.tsv_alt import SqliteSv
    SqliteSv.write_from_data(out_fn, row_data, [str] * len(cols), header=cols, index=['uniprot'])

def make_gene_to_uniprot(unifile):
    from dtk.files import get_file_records
    out = {}
    for rec in get_file_records(unifile, keep_header=False):
        if rec[1] == 'Ensembl' or rec[1] == 'Gene_Name' or rec[1] == 'Gene_Synonym':
            out[rec[2]] = rec[0]
    return out

def run(input, output, uniprot_converter):
    logger.info(f"Processing {input} to {output}")
    gene2uni = make_gene_to_uniprot(uniprot_converter)

    parse(input, output, gene2uni)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse OpenTargets tractability')
    parser.add_argument('-i','--input', help = 'Raw otarg input file')
    parser.add_argument('-u', '--uniprot-converter')
    parser.add_argument('-o','--output', help = 'Where to write the output (.sqlsv)')
    args = parser.parse_args()
    from dtk.log_setup import setupLogging
    setupLogging()

    run(**vars(args))
