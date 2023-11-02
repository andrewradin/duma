#!/usr/bin/env python3
import logging
from tqdm import tqdm
logger = logging.getLogger('parse_target_safety')

HEADER = ['uniprot', 'ref_links', 'ref_labels', 'safety_info', 'organs']

def parse_safety(safety):
    # These all have either a pmid or a link; just pick whichever is filled
    ref_pmids = [x['pmid'] or x['ref_link'] for x in safety['references']]
    ref_labels = [x['ref_label'] for x in safety['references']]
    safety.pop('references')
    
    liability = safety.pop('safety_liability')

    organ_systems = []
    for organ_data in safety.pop('organs_systems_affected'):
        organ_systems.append(organ_data['term_in_paper'])

    if safety:
        from pprint import pprint
        pprint(safety)
        assert False, 'Unexpected unparsed data'

    return '|'.join(ref_pmids), '|'.join(ref_labels), liability, '|'.join(organ_systems)

def make_gene_to_uniprot(unifile):
    from dtk.files import get_file_records
    out = {}
    for rec in get_file_records(unifile, keep_header=False):
        if rec[1] == 'Gene_Name' or rec[1] == 'Gene_Synonym':
            out[rec[2]] = rec[0]
    return out

def parse_adverse(ae):
    # These all have either a pmid or a link; just pick whichever is filled
    ref_pmids = [x['pmid'] or x['ref_link'] for x in ae['references']]
    ref_labels = [x['ref_label'] for x in ae['references']]
    ae.pop('references')

    all_effects = set()
    for dosing, effects in ae.pop('activation_effects').items():
        for effect in effects:
            all_effects.add('activate: ' + effect['term_in_paper'])

    for dosing, effects in ae.pop('inhibition_effects').items():
        for effect in effects:
            all_effects.add('inhibit: ' + effect['term_in_paper'])
    
    organ_systems = []
    for organ_data in ae.pop('organs_systems_affected'):
        organ_systems.append(organ_data['term_in_paper'])
    
    unspec = ae.pop('unspecified_interaction_effects')

    assert not unspec, f"Handle unspecified: {unspec}"

    assert ae == {}, f"Didn't handle all AEs: {ae}"

    return '|'.join(ref_pmids), '|'.join(ref_labels), '|'.join(sorted(all_effects)), '|'.join(organ_systems)
    


def parse_data(data):
    ae_data = data.pop('adverse_effects', [])
    logger.debug(f"Found {len(ae_data)} ae data to parse")
    for ae in ae_data:
        yield parse_adverse(ae)

    safety_data = data.pop('safety_risk_info', [])
    logger.debug(f"Found {len(safety_data)} safety data to parse")
    for safety in safety_data:
        yield parse_safety(safety)
    
    assert not data, f"Still didn't parse {data}"


def run(input, output, uniprot_converter):
    logger.info(f"Processing {input} to {output}")
    import json
    with open(input) as f:
        data = json.loads(f.read())
    
    gene2uni = make_gene_to_uniprot(uniprot_converter)

    from atomicwrites import atomic_write
    with atomic_write(output, overwrite=True) as out:
        out.write('\t'.join(HEADER) + '\n')
        for gene, gene_data in tqdm(data.items()):
            uniprot = gene2uni[gene]
            logger.debug(f"Doing {gene}, {uniprot}")

            for row_data in parse_data(gene_data):
                row = '\t'.join([uniprot, *row_data])
                out.write(f'{row}\n')

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse OpenTargets target safety data')
    parser.add_argument('-i','--input', help = 'Raw otarg input file')
    parser.add_argument('-u', '--uniprot-converter')
    parser.add_argument('-o','--output', help = 'Where to write the output (.tsv)')
    args = parser.parse_args()
    from dtk.log_setup import setupLogging
    setupLogging()

    run(**vars(args))
