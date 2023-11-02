#!/usr/bin/env python3

if __name__=='__main__':
    import argparse
    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="Aggregate transcripts")
    arguments.add_argument("raw_data", help="hg38.gene.sizes")
    args = arguments.parse_args()
    header = None
    current_tx = None
    lengths = []
    from statistics import median
    with open(args.raw_data, 'r') as f:
        for l in f:
            if header is None:
                header = l
                continue
            frs = l.split("\t")
            if current_tx is None:
                current_tx = frs[0]
                lengths.append(float(frs[1]))
            elif current_tx != frs[0]:
                print(current_tx + "\t" + str(median(lengths)))
                current_tx = frs[0]
                lengths = [float(frs[1])]
            else:
                lengths.append(float(frs[1]))
