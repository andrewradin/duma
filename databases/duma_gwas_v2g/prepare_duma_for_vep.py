#!/usr/bin/env python

def prep_grasp_for_vep(input_file, output_file):
    import isal.igzip as gzip

    import sys
    from dtk.files import get_file_records

    fn = input_file
    parse_type = 'tsv'
    gen = get_file_records(fn, parse_type = parse_type,
                           keep_header = None, progress=True)
    
    # We are nominally constructing a minimalist VCF file here.
    # See https://www.internationalgenome.org/wiki/Analysis/vcf4.0
    # In particular, we're filling in:
    # - CHR#
    # - POS
    # - ID  (use the rsid if available, otherwise set to '.', the default missing value)
    # - REFerence base (we always set to 'A', unless our allele is already 'A' in which case we set to 'C'
    # - ALT non-reference allele
    # All other fields are left unset.

    vep = set()
    for item in gen:
        try:
            study, rsid, chrm, pos, pval, allele_maf = item[:6]
            allele = allele_maf.split(';')[0]
            vep.add(' '.join([chrm, 
                              pos,
                              f'rs{rsid}' if rsid and rsid != 'rs_tbd' else '.',
                              'A' if allele != 'A' else 'C',
                              allele
                             ])
                    )
        except IndexError:
            continue

    with gzip.open(output_file, 'wb') as f:
        from tqdm import tqdm
        for item in tqdm(vep):
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
