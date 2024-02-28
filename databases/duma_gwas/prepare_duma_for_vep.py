#!/usr/bin/env python

def prep_grasp_for_vep(input_file, output_file):
    import gzip

    import sys
    try:
        from dtk.files import get_file_records
    except ImportError:
        sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
        from dtk.files import get_file_records

    fn = input_file
    parse_type = 'tsv'
    gen = get_file_records(fn, parse_type = parse_type,
                           keep_header = False)
    vep = set()
    for item in gen:
        try:
            allele = item[5].split(';')[0]
            vep.add(' '.join([item[2],
                              item[3],
                              '.',
                              'A' if allele != 'A' else 'C',
                              allele
                             ])
                    )
        except IndexError:
            continue

    with gzip.open(output_file, 'wb') as f:
        for item in vep:
            line = item + '\n'
            f.write(line.encode('utf-8'))

if __name__ == '__main__':
    import argparse
    arguments = argparse.ArgumentParser(description='''Parse prepped GWAS data 
                                                       to be ready for vep''')
    arguments.add_argument("i", help = 'Desired input file')
    arguments.add_argument("o", help = 'Desired output filename',
                           default = 'duma_vep.tsv.gz')
    args = arguments.parse_args()
    prep_grasp_for_vep(args.i, args.o)
