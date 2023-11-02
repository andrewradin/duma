#!/usr/bin/env python


def load_vep_locs(input_vep_file):
    from dtk.files import get_file_records
    fn = input_vep_file
    vep_gen = (x for x in get_file_records(fn, parse_type='tsv', progress=True) if x[0][0:2] != '##')
    header = next(vep_gen)
    header_inds = [header.index(x) for x in header]
    col2ind = dict(zip(header, header_inds))
    iloc = col2ind['Location']
    iallele = col2ind['Allele']

    out = set()
    for item in vep_gen:
        snp_loc = item[iloc] + ':' + item[iallele]
        out.add(snp_loc)
    return out


def output_missing(vep_locs, gwas_file, out_file):
    from dtk.files import get_file_records
    from atomicwrites import atomic_write
    missing = 0
    total = 0
    with atomic_write(out_file, overwrite=True) as out:
        for item in get_file_records(gwas_file, parse_type='tsv', progress=True):
            total += 1
            snp_loc = item[2] + ':' + item[3] + ':' + item[-2].split(';')[0]
            if snp_loc not in vep_locs:
                missing += 1
                out.write('\t'.join(item + [snp_loc]) + '\n')
    
    print(f"Missing {missing}/{total} ({missing*100/total:.2f}%)")


if __name__ == '__main__':
    import argparse
    import sys
    arguments = argparse.ArgumentParser(description='''Converts VEP output to our expected output format''')
    arguments.add_argument("--input-vep", help = 'Input VEP file')
    arguments.add_argument('--input-gwas', help = 'Input gwas file')
    arguments.add_argument('--output', help = 'Output unconverted snps')
    args = arguments.parse_args()
    vep_locs = load_vep_locs(args.input_vep)
    output_missing(vep_locs, args.input_gwas, args.output)