#!/usr/bin/env python3

program_description='''\
Show tox filter stats.

To generate a suitable input file, run parseChEMBL2.py with --unfiltered-tox.
'''

def show_stats(fn,limit,matched_only):
    from dtk.files import get_file_records

    desc_idx = 1
    key_idx = 6
    from collections import Counter
    ctr = Counter()
    desc_by_key = {}
    for rec in get_file_records(fn,keep_header=False):
        key = rec[key_idx]
        ctr[key] += 1
        desc_by_key[key] = rec[desc_idx]
    from dtk.assays import chembl_tox_type
    recs = [
            [
                str(count),
                chembl_tox_type(desc_by_key[key]) or 'None',
                desc_by_key[key],
                ]
            for key,count in ctr.most_common()
            ]
    if matched_only:
        recs = [x for x in recs if x[1] != 'None']
    if limit:
        recs = recs[:limit]
    from dtk.text import print_table,wrap,split_multi_lines
    wrap(recs,2,60)
    print_table(split_multi_lines(recs))

################################################################################
# main
################################################################################
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=program_description,
            )
    parser.add_argument('--limit',
            type=int,
            default=100,
            )
    parser.add_argument('--show-all',
            action='store_true',
            )
    parser.add_argument('unfiltered',
            )
    args = parser.parse_args()

    show_stats(args.unfiltered,args.limit,not args.show_all)

