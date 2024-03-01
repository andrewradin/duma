#!/usr/bin/env python3

from dtk.lazy_loader import LazyLoader

def show_gd_suffixes(fn):
    from dtk.files import get_file_records
    dosages = []
    salts = []
    def extract_suffix(base):
        words = base.split()
        if len(words) == 1:
            return
        suffix = words[-1]
        if suffix == suffix.upper():
            dosages.append(suffix)
        else:
            salts.append(suffix.lower())
    prev = None
    for rec in get_file_records(fn,keep_header=False):
        if rec[1] == 'synonym':
            # overwrite canonical if synonym already extracted
            prev = rec[2]
        if rec[1] == 'canonical':
            if prev:
                # process previous, and set up for next time
                extract_suffix(prev)
            prev = rec[2]
    if prev:
        # process previous, and set up for next time
        extract_suffix(prev)
    from collections import Counter
    print('Dosages:')
    print([x for x in Counter(dosages).most_common() if x[1] > 2])
    print('Salts:')
    print([x for x in Counter(salts).most_common() if x[1] > 2])

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='''\
''',
            )
    parser.add_argument('attr_fn')
    args = parser.parse_args()
    show_gd_suffixes(args.attr_fn)
