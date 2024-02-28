#!/usr/bin/env python

from dtk.gwas_filter import gwas_filter

if __name__=='__main__':
    import argparse

    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="Filter GWAS SNP data")
    arguments.add_argument("-i", nargs='*', help="Input SNP data filename(s)")
    arguments.add_argument("-o", nargs='*', help="Output SNP data filename(s)")
    arguments.add_argument("-studies", nargs='*', help="Input Study filename(s)")
    arguments.add_argument("-log", help="log file name, default: log.tmp", default="log.tmp")
    args = arguments.parse_args()


    gf = gwas_filter(log=args.log)
    assert len(args.i) == len(args.o)
    for i in range(len(args.i)):
        gf.write_snps(args.i[i], args.o[i],args.studies[i])
    gf.report()
