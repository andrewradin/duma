#!/usr/bin/env python3

import collections

if __name__ == "__main__":
    import argparse
    from dtk.files import get_file_lines
    parser = argparse.ArgumentParser(
                    description='map indi names to meddra',
                    )
    parser.add_argument('meddra_ver')
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args=parser.parse_args()

    from dtk.meddra import IndiMapper
    d = IndiMapper(args.meddra_ver)
    from atomicwrites import atomic_write
    with atomic_write(args.outfile,overwrite=True) as out:
        c = collections.Counter()
        for line in get_file_lines(args.infile,progress=True):
            c.update(["lines"])
            # lines are supposed to consist of an event code and a drug name,
            # separated by a tab.  But some drug names contain embedded tabs,
            # and some records are just '\r\n'
            fields = line.strip().split('\t',2)
            if len(fields) != 3:
                print('skipping unexpected record:',fields)
                c.update(['skipped_lines'])
                continue
            for code,how in d.translate(fields[2].strip().lower()):
                if code:
                    name = d.code2name(code)
                    out.write("%s\t%s\t%s\n" % (fields[0],fields[1],name.lower()))
                c.update([how])
            if c['lines'] % 1000000 == 0:
                print(c)
    print(d.missed_names)
    print(c)
