#!/usr/bin/env python3
import os, django, sys, argparse, re
from collections import defaultdict
sys.path.insert(1,"../../web1")
from path_helper import PathHelper


# created 3.Mar.2016 - Aaron C Daugherty - twoXAR

# Extract select *C50 type data from the mess of chembl tables

def message(*objs):
    print(*objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def verboseOut(*objs):
    if args.verbose:
        print(*objs, file=sys.stderr)

def useage(*objs):
    print("USEAGE: ", *objs, file=sys.stderr)

def printOut(*objs):
    print(*objs, file=sys.stdout)

def bad_component_seq(cs):
    if str(cs.component_type) != 'PROTEIN':
        return 'component_type: ' + str(cs.component_type)
    if str(cs.tax_id) != '9606':
        return 'tax_id: ' + str(cs.tax_id)
    if str(cs.organism) != 'Homo sapiens':
        return 'organism: ' + str(cs.organism)
    if str(cs.db_source) != 'SWISS-PROT':
        return 'db_source: ' + str(cs.db_source)
    return False

def process_activities(measure_type, standard_value):
    direction = get_direction_from_measure_type(measure_type)
    final_value = unwind_vals(standard_value, measure_type)
    return final_value, direction

def unwind_vals(val, type):
    if type.startswith('p') or type.startswith('-Log ') or type.startswith('Log 1/'):
        return 10 ** (val * -1.0)
    return val

def get_direction_from_measure_type(measure_type):
    if measure_type in get_positive_direc_measures():
        return 1
    elif measure_type in get_negative_direc_measures():
        return -1
    elif measure_type in get_neutral_direc_measures() or measure_type in get_ki_measures():
        return 0

def get_single_entry(base_query):
    if base_query.count() > 1:
        return False
    return base_query.get()

def get_measurement_types():
    l1 = get_positive_direc_measures()
    l2 = get_negative_direc_measures()
    l3 = get_neutral_direc_measures()
    l4 = get_ki_measures()
    return [ i for sublist in [l1, l2, l3, l4] for i in sublist]
# I want to stay away from regex here b/c there is a set list of possibilities,
# and I defined them after looking at the chembl table
def get_ki_measures():
    return ['Ki', '-Log Ki',
            'pKi'
           ]

def get_positive_direc_measures():
    return ['EC50', 'Activity EC50',
            'pEC50', '-Log EC50',
            'ED50', 'Log 1/ED50',
            'Log 1/ED50', 'log(1/ED50)']

def get_negative_direc_measures():
    return ['IC50', 'pIC50',
            '-Log IC50', 'ID50',
            'Log 1/IC50', 'Log 1/ID50',
            'log(1/IC50)', 'log(1/ID50)'
           ]

def get_neutral_direc_measures():
    return ['XC50', 'AC50',
            'pXC50', 'pAC50',
            'AC_50']

if __name__=='__main__':
    #=================================================
    # Read in the arguments/define options
    #=================================================

    arguments = argparse.ArgumentParser(description="Extract potency data from chembl tables, C50 and Ki")
    arguments.add_argument('-v', "--verbose", action="store_true", help="Print out status reports")
    arguments.add_argument("c50_file", help="output file name")
    arguments.add_argument("ki_file", help="output file name")
    arguments.add_argument("raw_file", help="output file name")
    arguments.add_argument("chembl_version", help="version (e.g. chembl_23)")

    args = arguments.parse_args()

    import importlib
    ch = importlib.import_module(args.chembl_version+'_schema')

    useable_activity_types = get_measurement_types()
    valid_conf_scores=ch.ConfidenceScoreLookup.select().where(
            ch.ConfidenceScoreLookup.confidence_score >= 6
            )
    target_map={}
    q = ch.TargetComponents.select().join(ch.ComponentSequences).where(
            ch.ComponentSequences.tax_id == 9606
            )
    for row in q:
        error = bad_component_seq(row.component)
        if error:
            verboseOut('problem with',row.tid.tid,error)
            continue
        s = target_map.setdefault(row.tid.tid,set())
        s.add(str(row.component.accession))
    verboseOut(len(target_map),'targets found')
    verboseOut(len([
            x for x in target_map.values()
            if len(x) > 1
            ]),'with multiple uniprots')
    # single item list is queries is a workaround for fields that are
    # actually foreign keys
    acts=ch.Activities.select(
                    ch.Activities,
                    ch.Assays,
            ).join(ch.Assays).where(
                    (ch.Assays.relationship_type << ['D','H'])
                    & (ch.Assays.confidence_score << valid_conf_scores)
                    & (ch.Activities.standard_type << useable_activity_types)
                    & (ch.Activities.standard_value > 0)
                    & (ch.Activities.standard_units == 'nM')
            )
    act_count = acts.count()
    verboseOut('processing',act_count,'activity records')
    chembl_lookup=dict(
            ch.MoleculeDictionary.select(
                        ch.MoleculeDictionary.molregno,
                        ch.MoleculeDictionary.chembl_id,
                    ).tuples()
            )
    verboseOut('prefetched',len(chembl_lookup),'chembl ids')
    from atomicwrites import atomic_write
    with atomic_write(args.c50_file, overwrite=True) as c1, \
            atomic_write(args.ki_file, overwrite=True) as k1, \
            atomic_write(args.raw_file, overwrite=True) as raw:
        # write header
        shared_header = ['chembl_id', 'uniprot_id', 'direction']
        c50_header = "\t".join(shared_header[:2]+['C50']+[shared_header[-1]])
        ki_header = "\t".join(shared_header[:2]+['Ki']+[shared_header[-1]])
        c1.write(c50_header + "\n")
        k1.write(ki_header + "\n")

        raw_header = ['chembl_id', 'uniprot_id', 'type', 'relation', 'value', 'direction', 'assay_format']
        raw.write('\t'.join(raw_header) + "\n")
        src = acts.iterator()
        if args.verbose:
            from tqdm import tqdm
            src = tqdm(src,total=act_count)
        for r in src:
            if (str(r.standard_type) in useable_activity_types and
                    r.standard_value is not None and
                    r.standard_value > 0 and
                    r.standard_units == 'nM' and
                    r.standard_relation is not None and
                    (r.potential_duplicate is None
                        or r.potential_duplicate == 0) and
                    (r.data_validity_comment is None
                         or r.data_validity_comment == 'Manually validated'
                    )
                ):
                standard_value = float(r.standard_value)
                measure_type = str(r.standard_type)
                val, direc = process_activities(measure_type, standard_value)
                chembl_id = str(chembl_lookup[r.molregno_id])
            else:
                continue
            for uniprot in target_map.get(r.assay.tid_id,[]):
                ki = measure_type in get_ki_measures()
                type_str = 'ki' if ki else 'c50'
                relation = r.standard_relation
                # https://www.ebi.ac.uk/ols/ontologies/bao
                # We just want to distinguish cell vs non-cell.
                cell_ids = {
                    'BAO_0000219', # Cell based
                    'BAO_0000218', # Organism
                    'BAO_0000221', # Tissue
                    }
                assayformat = 'cell' if r.assay.bao_format_id in cell_ids else 'biochem'
                raw_to_write = "\t".join([chembl_id, uniprot, type_str, relation, str(val), str(direc), str(assayformat)]) + "\n"
                raw.write(raw_to_write)
                if r.standard_relation in ['=', '<', '<=']:
                    to_write = "\t".join([chembl_id, uniprot, str(val), str(direc)]) + "\n"
                    if ki:
                        k1.write(to_write)
                    else:
                        c1.write(to_write)
