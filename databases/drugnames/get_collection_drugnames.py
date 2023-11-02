#!/usr/bin/env python3

def get_aact_used_drugnames(fname):
    drugnames = set()
    from dtk.files import get_file_records
    drug_index = None
    for rec in get_file_records(fname,keep_header=True):
        if drug_index is None:
            drug_index = rec.index('DRUGS')
            continue
        for part in rec[drug_index:]:
            drugnames.add(part)
    return drugnames

def get_collection_drugnames(fname,fact_checker):
    prop_names = set([
            'canonical',
            'synonym',
            # XXX The following could be enabled to match the behavior of
            # XXX name2cas, but they don't add any AACT matches, and make
            # XXX the file a lot more noisy.
            #'brand',
            #'mixture',
            ])
    drugnames = set()
    min_name_length=2
    key_name = None
    from dtk.files import get_file_records
    for rec in get_file_records(fname,keep_header=True):
        if key_name is None:
            # grab key name from header
            key_name = rec[0]
            continue
        if rec[1] not in prop_names:
            continue
        # re-write as fact checker expects
        rec[1] = 'name'
        rec[2] = rec[2].lower()
        if not fact_checker.check_fact([key_name]+rec):
            continue
        if len(rec[2]) < min_name_length:
            continue
        drugnames.add(f'{rec[2]}\t{rec[0]}')
    return drugnames

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--reverse',action='store_true')
    parser.add_argument('attr_file',nargs='+')
    args = parser.parse_args()

    if args.reverse:
        parser = get_aact_used_drugnames
    else:
        from path_helper import PathHelper
        from dtk.drug_clusters import FactChecker
        fact_checker = FactChecker(
                PathHelper.repos_root+'databases/matching/falsehoods.tsv'
                )
        parser = lambda x:get_collection_drugnames(x,fact_checker)
    src = set()
    for fn in args.attr_file:
        src |= parser(fn)
    max_punctuation = 5
    for name in src:
        if len([x for x in name if x in '(),-']) > max_punctuation:
            # this tosses iupac names, which are never used in AACT,
            # and make the output 3x longer
            continue
        print(name)
