#!/usr/bin/env python

import os, sys
sys.path.insert(1,"../../web1")

def reader(fn):
    import csv
    return csv.reader(open(fn,'rb'),
            delimiter='|',
            quotechar='"',
            )

def extract_ingredients():
    ingredients = set()
    header=None
    for rec in reader('interventions.txt'):
        if not header:
            header=rec
            continue
        if len(rec) < 3:
            print rec
            raise RuntimeError()
        if rec[2] != 'Drug':
            continue
        ingredients.add(rec[3].lower())
    return ingredients

def show_raw(args):
    ingredients = extract_ingredients()
    if args.verbose:
        for name in sorted(ingredients):
            print name
    else:
        print len(ingredients)

def phases(args):
    from collections import Counter
    src = reader('studies.txt')
    header = src.next()
    phase_idx=header.index('phase')
    phases = Counter([rec[phase_idx] for rec in src])
    print phases

def nct_id_filtered_by_phase(valid_phases):
    src = reader('studies.txt')
    header = src.next()
    nct_idx=header.index('nct_id')
    phase_idx=header.index('phase')
    return set([
            rec[nct_idx]
            for rec in src
            if rec[phase_idx] in valid_phases
            ])

def nct_id_filtered_by_type(valid_types):
    src = reader('interventions.txt')
    header = src.next()
    nct_idx=header.index('nct_id')
    type_idx=header.index('intervention_type')
    return set([
            rec[nct_idx]
            for rec in src
            if rec[type_idx] in valid_types
            ])

def filtered_interventions(args):
    # get trials from phase 2 on
    valid_nct_ids = nct_id_filtered_by_phase(set([
            #'Phase 1/Phase 2',
            'Phase 2',
            'Phase 2/Phase 3',
            'Phase 3',
            'Phase 4',
            ]))
    # limit to drug interventions
    valid_nct_ids &= nct_id_filtered_by_type(set([
            'Drug',
            ]))
    src = reader('browse_interventions.txt')
    header = src.next()
    nct_idx=header.index('nct_id')
    mesh_idx=header.index('downcase_mesh_term')
    combo_suffix=' drug combination'
    result = set()
    for rec in src:
        if rec[nct_idx] not in valid_nct_ids:
            continue
        mesh = rec[mesh_idx]
        if mesh.endswith(combo_suffix):
            result |= set(mesh[:-len(combo_suffix)].split(', '))
        else:
            result.add(mesh)
    if args.verbose:
        for name in sorted(result):
            print name
    else:
        print len(result)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='utility')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('cmd')
    args = parser.parse_args()

    locals()[args.cmd](args)
