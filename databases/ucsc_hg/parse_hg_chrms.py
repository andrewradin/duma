#!/usr/bin/env python3

if __name__=='__main__':
    import argparse
    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="Pull out only the relevant chroms")
    arguments.add_argument("raw_data", help="hg38.chrom.sizes")
    args = arguments.parse_args()

    with open(args.raw_data, 'r') as f:
        for l in f:
            frs = l.split("\t")
            chrm = frs[0].lstrip('chr')
            if len(chrm) <= 2:
                print(chrm+"\t"+frs[1].rstrip())
