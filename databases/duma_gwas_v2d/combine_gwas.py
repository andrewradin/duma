#!/usr/bin/env python

# We output all the studies-snps that we see.
# If there are multiple copies of the same (study, snp), we take the first one we see.
import sys
from dtk.files import get_file_records
from dtk.gwas_filter import k_is_good

def run(snp_files, output):
    seen = set()
    from atomicwrites import atomic_write
    with atomic_write(output, overwrite=True) as out:
        for snp_file in snp_files:
            for frs in get_file_records(snp_file, keep_header=True, progress=True):
                if not k_is_good(frs[0]):
                    continue
                
                # Key on the study name, rsid, chrm and pos
                rec_key = tuple(frs[:4])
                if rec_key in seen:
                    continue
                    
                seen.add(rec_key)
                out.write("\t".join(frs) + '\n')

if __name__=='__main__':
    import argparse
    arguments = argparse.ArgumentParser(description="Combine snps data from different sources")
    arguments.add_argument("snps", nargs='*', help="SNP data files")
    arguments.add_argument("-o", "--output", help="merged SNP data file")
    args = arguments.parse_args()
    
    run(args.snps, args.output)


