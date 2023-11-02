#!/usr/bin/env python3

def study_drug_pairs(fn):
    drug_index = None
    from dtk.files import get_file_records
    for rec in get_file_records(fn,keep_header=True):
        if drug_index is None:
            # process header
            drug_index = rec.index('DRUGS')
            continue
        for drug in rec[drug_index:]:
            yield(rec[0],drug)

def compare(base_fn,new_fn):
    from dtk.data import MultiMap
    base = MultiMap(study_drug_pairs(base_fn))
    new = MultiMap(study_drug_pairs(new_fn))
    all_keys = set(base.fwd_map().keys()) | set(new.fwd_map().keys())
    from collections import defaultdict
    diffs = defaultdict(set)
    for key in all_keys:
        bdrugs = base.fwd_map().get(key,set())
        ndrugs = new.fwd_map().get(key,set())
        if ndrugs != bdrugs:
            delta = (frozenset(bdrugs-ndrugs),frozenset(ndrugs-bdrugs))
            diffs[delta].add(key)
    for old,new in diffs:
        # XXX we can add more sophisticated filtering here to reduce the
        # XXX noise that needs to be manually filtered
        if not old:
            continue
        dropped = False
        for drug in old:
            if not any(drug in x for x in new):
                dropped = True
                break
        if not dropped:
            continue
        print(
            '|'.join(sorted(old)),
            '=>',
            '|'.join(sorted(new)),
            'in',len(diffs[(old,new)]),'studies',
            )

def outstats(fn,dump_drugs):
    studies = set()
    drugs = set()
    lines = 0
    diseases = set()
    from dtk.files import get_file_records
    for rec in get_file_records(fn,keep_header=False):
        lines += 1
        studies.add(rec[0])
        diseases.add(rec[2])
        for drug in rec[6:]:
            drugs.add(drug)
    print('%s: %d interventions; %d studies; %d diseases; %d drugs'%(
            fn,
            lines,
            len(studies),
            len(diseases),
            len(drugs),
            ))
    if dump_drugs:
        with open(dump_drugs,'w') as fh:
            for drug in sorted(drugs):
                fh.write(drug+'\n')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='stats')
    parser.add_argument('--dump-drugs')
    parser.add_argument('--cmp-to')
    parser.add_argument('matrix_file')
    args = parser.parse_args()

    if args.cmp_to:
        compare(args.cmp_to,args.matrix_file)
    else:
        outstats(args.matrix_file,args.dump_drugs)
