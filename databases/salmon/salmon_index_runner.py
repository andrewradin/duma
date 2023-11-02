#!/usr/bin/env python3

from path_helper import PathHelper

class runner:
    def __init__(self, kmer, outdir, cdna_file,v):
        import subprocess as sp
        import os
        salmon=os.path.join(PathHelper.website_root, 'R', 'RNAseq',f'Salmon{v}','bin','salmon')
        sp.check_call([salmon, 'index', '-t', cdna_file, '-i', outdir, '-k', kmer])

if __name__ == "__main__":
    import argparse, time
    import multiprocessing
    parser = argparse.ArgumentParser(description='Inserts standardized smiles into a create file')
    parser.add_argument("-k", "--kmer", help="kmer size")
    parser.add_argument("-o", "--output", help="Output tgz")
    parser.add_argument("-c", "--cdna", help="cdna file from Ensembl")
    parser.add_argument("-r", "--remote", help="Run on remote machine")
    parser.add_argument("-v", "--pkg_version", help="Validate the version of Salmon being run")
    args=parser.parse_args()

    if args.remote:
        import aws_op
        import time
        mch = aws_op.Machine.name_index[args.remote]
        remote_fn = f'/tmp/salmon_index.{time.time()}.in'
        remote_out_prefix = f'/tmp/salmon_index.{time.time()}'
        remote_temp_out = f'{remote_out_prefix}.out'
        remote_out_fn = f'{remote_temp_out}.tgz'
        mch.copy_to(args.cdna, remote_fn)
        mch.run_remote_cmd(f"2xar/twoxar-demo/databases/salmon/salmon_index_runner.py -k {args.kmer} -c {remote_fn} -o {remote_temp_out} -v {args.pkg_version}")
        mch.run_remote_cmd(f"tar -czvf {remote_out_fn} {remote_temp_out}")
        local_out_fn = './temp.tgz'
        mch.copy_from(remote_out_fn, local_out_fn)
        import os
        os.rename(local_out_fn, args.output)
    else:
        runner(args.kmer, args.output, args.cdna, args.pkg_version)
