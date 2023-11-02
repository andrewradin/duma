#!/usr/bin/env python3

import collections

if __name__ == "__main__":
    import argparse
    from dtk.files import get_file_lines
    parser = argparse.ArgumentParser(
                    description='map indi and drug names',
                    )
    parser.add_argument('--meddra-ver', required=True)
    parser.add_argument('--name2cas', required=True)
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args=parser.parse_args()

    from dtk.meddra import IndiMapper
    from map_drugs import DrugNameMapper
    d = IndiMapper(args.meddra_ver)
    dn = DrugNameMapper(args.name2cas)
    from atomicwrites import atomic_write
    with atomic_write(args.outfile,overwrite=True) as out:
        c = collections.Counter()
        for line in get_file_lines(args.infile,progress=True):
            c.update(["lines"])
            fields = line.strip('\r\n').split('\t')
            indiname = fields[2]
            drugname = fields[3]
            for indicode,indihow in d.translate(indiname.strip().lower()):
                if indicode:
                    mapped_indi_name = d.code2name(indicode)
                    for cas,drughow in dn.translate(drugname):
                        if cas:
                            out.write('\t'.join([fields[0],fields[1],mapped_indi_name,cas,*fields[4:]]) + '\n')
                        c.update(['dr-' + drughow])
                c.update(['in-' + indihow])
            if c['lines'] % 1000000 == 0:
                print(c)
    print(d.missed_names)
    print(c)
