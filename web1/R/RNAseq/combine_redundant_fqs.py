#!/usr/bin/env python

from __future__ import print_function
from builtins import range
import os

def get_fq_name(dir, pref, suffix = ''):
    return os.path.join(dir,
                        pref + suffix + '.fastq.gz'
                       )
def combine_gzipped(params):
    frs, odir = params
    prefix = frs[0]
    assert len(frs) == 2, "Additional adds should be comma separated"
    tmp_gz = os.path.join(odir, prefix+'_combining_tmp.gz')
    if os.path.isfile(get_fq_name(odir, frs[0])):
        suffixes = ['']
    elif os.path.isfile(get_fq_name(odir, frs[0], '_1')):
        suffixes = ['_1']
        if os.path.isfile(get_fq_name(odir, frs[0], '_2')):
            suffixes += ['_2']
    else:
        raise Exception(f"Couldn't find any files named like {get_fq_name(odir, frs[0])}")
    to_ret = []
    for suf in suffixes:
        ofile = get_fq_name(odir, frs[0], suf)
        assert os.path.isfile(ofile)
        to_add_accs = [get_fq_name(odir, x, suf) for x in frs[1].split(',')]
        to_ret.append(" ".join(to_add_accs + ['appended to', ofile]))
        cmd = ['cat', ofile]
        cmd += to_add_accs
        cmd += ['>', tmp_gz]
        subprocess.check_call(" ".join(cmd), shell = True)
        subprocess.check_call(" ".join(['rm'] + to_add_accs), shell = True)
        subprocess.check_call(" ".join(['mv', tmp_gz, ofile]), shell = True)
    return to_ret

if __name__ == "__main__":

    import argparse, subprocess
    from dtk.files import get_file_records

    parser = argparse.ArgumentParser(description='condense split SRRs')
    parser.add_argument("outdir", help="directory to put output in")
    parser.add_argument("--cores", type=int, default = 1, help = "number of cores to use")
    parser.add_argument("--record", type=str, default = 'None', help = "for processing just one record at a time")

    args=parser.parse_args()

    idir = args.outdir + '/../../'
    ifile = [os.path.join(idir,f) for f in os.listdir(idir) if f.endswith('_SRRsToCombine.tsv')]
    if len(ifile) == 0:
        print("Nothing to combine, looking in", idir)
    else:
        assert len(ifile) == 1
        all_lines = list(get_file_records(ifile[0], keep_header = None))
        if args.record != 'None':
            line = [l for l in all_lines if l[0] == args.record]
            assert len(line)==1, f'Single record ({args.record}) provided, but not found'
            all_lines = line
        n = len(all_lines)
        params = zip(all_lines,
                 [args.outdir] * n
                 )
        if args.cores > 1:
            import multiprocessing
            pool = multiprocessing.Pool(args.cores)
            res = pool.map(combine_gzipped, params)
        else:
            res = map(combine_gzipped, params)
        print(list(res))
