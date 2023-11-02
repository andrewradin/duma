#!/usr/bin/env python3

import logging
from tqdm import tqdm
logger = logging.getLogger('parse')

def make_gene_to_uniprot(unifile):
    from dtk.files import get_file_records
    out = {}
    for rec in get_file_records(unifile, keep_header=False):
        if rec[1] == 'Ensembl' or rec[1] == 'Gene_Name' or rec[1] == 'Gene_Synonym' or rec[1] == 'Ensembl_TRS':
            out[rec[2]] = rec[0]
    return out

def load_ppi_file(fn):
    from dtk.files import get_file_records
    from dtk.data import MultiMap
    pairs = []
    for rec in get_file_records(fn, keep_header=False, progress=True, parse_type='ssv'):
        org_num, p1 = rec[0].split('.')
        org_num, p2 = rec[1].split('.')
        if int(rec[2]) >= 900:
            pairs.append((p1, p2))
    return org_num, MultiMap(pairs).fwd_map()

def load_all_ppi(animal_ppis):
    orgnum_to_name = {
        '10090': 'mouse',
        '10116': 'rat',
        '9612': 'dog', # non-standard dog ID, best we've got in STRING
        '7955': 'zebrafish',
        '9606': 'human',
    }
    org_to_ppi = {} 
    for ppi_fn in animal_ppis:
        org_num, org_data = load_ppi_file(ppi_fn)
        org_name = orgnum_to_name[org_num]
        org_to_ppi[org_name] = org_data
       
    return org_to_ppi

def get_ppi_jac_score(ensp1, ppi_data1, ensp2, ppi_data2, ensp_transl):
    ps1 = ppi_data1.get(ensp1, set())
    ps2 = ppi_data2.get(ensp2, set())

    # Translate ps2 from its own organism to ps1's organism.
    trans_ps2 = {ensp_transl.get(p, p) for p in ps2}

    from dtk.similarity import calc_jaccard
    return calc_jaccard(ps1, trans_ps2)



def run(input, output, uniprot_converter, animal_ppis):
    from dtk.files import get_file_records

    organisms = ['mouse', 'rat', 'zebrafish', 'dog']
    gene2uni = make_gene_to_uniprot(uniprot_converter)

    ppi_data = load_all_ppi(animal_ppis)

    from collections import defaultdict
    out = defaultdict(set)
    out_ensp = {}

    ensp_transl = {}

    for rec in get_file_records(input, keep_header=False, progress=True):
        ensg, enst, gene_name, ensp = rec[:4]
        uniprot = gene2uni.get(ensg)
        if not uniprot:
            uniprot = gene2uni.get(enst)
            if not uniprot:
                uniprot = gene2uni.get(gene_name)
                if not uniprot:
                    logger.warning(f"Couldn't find a uniprot for {ensg}, {enst}, {gene_name}")
                    continue
        for i, organism in enumerate(organisms):
            # Organism data starts at the 5th column, with 3 columns per organism.
            start = 4+i*3
            end = 4+(i+1)*3
            org_gene_name, value, org_ensp = rec[start:end]
            # Sometimes value will be a blank if there is none.
            if value:
                # There are a lot of duplicates across transcripts for the same gene that have identical values.
                # Dedupe with a set.
                key = (uniprot, organism, org_gene_name, value)
                out[key].add(enst)
                out_ensp[key] = (ensp, org_ensp)
                ensp_transl[org_ensp] = ensp

    # Flatten out the multiple values.
    out_flat = []
    for k, transcripts in out.items():
        human_ensp, org_ensp = out_ensp[k]
        organism = k[1]
        ppi_score = get_ppi_jac_score(
            human_ensp,
            ppi_data['human'],
            org_ensp,
            ppi_data[organism],
            ensp_transl,
            )
        out_flat.append((*k, ','.join(transcripts), ppi_score))

    from dtk.tsv_alt import SqliteSv
    header = ['uniprot', 'organism', 'organism_gene', 'similarity(%)', 'transcripts', 'ppi_jacc']
    SqliteSv.write_from_data(output, out_flat, header=header, types=[str, str, str, float, str, float], index=['uniprot'])

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse data')
    parser.add_argument('-i','--input', help = 'Raw input file')
    parser.add_argument('-u', '--uniprot-converter')
    parser.add_argument('-o','--output', help = 'Where to write the output')
    parser.add_argument('animal_ppis', nargs='+', help = 'animal ppi files')
    args = parser.parse_args()
    from dtk.log_setup import setupLogging
    setupLogging()

    run(**vars(args))