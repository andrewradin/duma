#!/usr/bin/env python

if __name__=='__main__':
    import argparse,sys,subprocess
    from path_helper import PathHelper
    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="A convienience wrapper to manage chain file downloads")
    arguments.add_argument("X2Y", help="Which versions to map b/t")

    args = arguments.parse_args()

    if args.X2Y == '18to38':
        url = 'http://hgdownload.soe.ucsc.edu/goldenPath/hg18/liftOver/hg18ToHg38.over.chain.gz'
        dnld_file='hg18ToHg38.over.chain.gz'
        final_file='hg18ToHg38.over.chain.gz'
    elif args.X2Y == '19to38':
        url='http://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz'
        dnld_file='hg19ToHg38.over.chain.gz'
        final_file='hg19ToHg38.over.chain.gz'
    elif args.X2Y == '37to38':
        url='https://sourceforge.net/projects/crossmap/files/Ensembl_chain_files/homo_sapiens%28human%29/GRCh37_to_GRCh38.chain.gz/download'
        dnld_file='download'
        final_file='GRCh37_to_GRCh38.chain.gz'
    else:
        print("Unsupported chain type. Quitting.")
        sys.exit(1)

    subprocess.check_call(['wget', url])
    subprocess.check_call(['mv', dnld_file, PathHelper.storage+final_file])
