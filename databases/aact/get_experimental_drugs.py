#!/usr/bin/env python3

from dtk import parse_aact
import re

def get_potential_drugnames(fname):
    from dtk.files import get_file_records
    return set(
            rec[0]
            for rec in get_file_records(fname)
            )

def get_drugname_to_nativekey(fname):
    from dtk.files import get_file_records
    from dtk.data import MultiMap
    return MultiMap(
            (rec[0], rec[1])
            for rec in get_file_records(fname)
            )

def extract_drugs_from_target(target,all_drugnames):
    '''Return list of cleaned-up drugnames for raw target string.'''
    # Rather than a bunch of ad-hoc rules like we use in FAERS,
    # look for known drugnames in the raw target string.
    target=target.lower()
    if target in all_drugnames:
        return [target]
    result=set()
    for word in re.split('[ +]+',target):
        if word in all_drugnames:
            result.add(word)
    return list(result)

def get_drug_map():
    result = {}
    fn = 'interventions.txt'
    header = None
    for fields in parse_aact.aact_file_records(fn):
        if not header:
            header = [x.upper() for x in fields]
            continue
        d = dict(zip(header,fields))
        if d['INTERVENTION_TYPE'] not in ('Drug','Biological'):
            continue
        target = d['NAME']
        # XXX Usually, NAME is the drug name and DESCRIPTION holds dosage
        # XXX and delivery info, but in some cases (e.g. NCT04073069) the
        # XXX name is just a trial-specific code, and the full drugname
        # XXX is only in the description. Maybe parse both?
        result.setdefault(d['NCT_ID'],set()).add(target)
    return result



def output_summary(drugnames_file, output_prefix):
    cond_lists = parse_aact.get_study_condition_lists()
    drug_map = get_drug_map()
    all_drugnames = get_potential_drugnames(drugnames_file)
    drug_lookup = get_drugname_to_nativekey(drugnames_file).fwd_map()
    fn = 'studies.txt'
    header = None
    out_fields = [
            'PHASE',
            'OVERALL_STATUS',
            'COMPLETION_DATE',
            ]
    print('\t'.join(
            ['STUDY','INTERVENTION','DISEASE','START_YEAR']
            +out_fields+['DRUGS']
            ))


    study_rows = []
    drug_study_rows = []
    disease_study_rows = []
    for fields in parse_aact.aact_file_records(fn):
        if not header:
            header = [x.upper() for x in fields]
            continue
        d = dict(zip(header,fields))
        start_date = d['START_DATE']
        start_year = start_date.split('-')[0] if start_date else ''
        study = d['NCT_ID']
        rest = [d[x] for x in out_fields]
        targets = drug_map.get(study)
        if not targets:
            continue
        
        study_rows.append((study, ' | '.join(targets), start_year, *rest))

        for target in targets:
            drugs = extract_drugs_from_target(target,all_drugnames)
            if not drugs:
                continue
            for drug in drugs:
                native_keys = drug_lookup.get(drug)
                for native in native_keys:
                    drug_study_rows.append((study, native, drug))

        # 'targets' is a set of all the raw intervention descriptions
        # from the study
        for disease in cond_lists.get(study,['NONE']):
            if disease == 'NONE':
                continue
            disease_study_rows.append((study, disease))
            for target in targets:
                drugs = extract_drugs_from_target(target,all_drugnames)
                if not drugs:
                    continue
                print('\t'.join(
                        [study,target,disease,start_year]+rest+drugs
                        ))
    study_out = output_prefix + '.studies.sqlsv'
    drug_study_out = output_prefix + '.drugs.sqlsv'
    disease_study_out = output_prefix + '.diseases.sqlsv'
    from dtk.tsv_alt import SqliteSv
    SqliteSv.write_from_data(study_out, set(study_rows),
            [str, str, str, str, str, str],
            header=[
                    'study',
                    'intervention',
                    'start_year',
                    'phase',
                    'status',
                    'completion_date',
                    ],
            index=['study'])
    SqliteSv.write_from_data(drug_study_out, set(drug_study_rows), [str, str, str], header=['study', 'native_key', 'drug'])
    SqliteSv.write_from_data(disease_study_out, set(disease_study_rows), [str, str], header=['study', 'disease'])



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('drugnames_file')
    parser.add_argument('--output-prefix', help='e.g. aact.v5')
    args = parser.parse_args()

    output_summary(args.drugnames_file, args.output_prefix)
