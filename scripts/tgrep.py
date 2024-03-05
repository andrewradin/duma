#!/usr/bin/env python

# A useful trick for finding python variables that are short or common words:
# tgrep.py '^[^#]*\<has\>' '*.py'

def make_filelist(pattern,startpath='.'):
    import os
    import fnmatch
    result = []
    skipdirs = [
            '.git',
            '.cache',
            'chembl_20_mysql',
            'chembl_23_mysql',
            'ensembl-vep',
            'input',
            'output',
            'data',
            'log',
            'stage_drugsets',
            'stage_dpi',
            'lib',
            'node_modules',
            '2020AA',
            ]
    skipdirs_noshow = [
            '__pycache__',
            ]
    skiptypes = [
            'zip',
            'gz',
            'npz',
            'log',
            'csv',
            'pyc',
            'tsv',
            'so',
            'swp',
            'soft',
            'txt',
            'RRF',
            'coverage',
            ]
    skipfiles = [
            'dry_AMD_case_only',
            'dry_AMD_control_only',
            ]
    for root, dirs, files in os.walk(startpath):
        for d in skipdirs+skipdirs_noshow:
            if d in dirs:
                if d not in skipdirs_noshow:
                    print('skipping',os.path.join(root,d))
                dirs.remove(d)
        for f in files:
            if f in skipfiles:
                continue
            if pattern:
                if not fnmatch.fnmatch(f,pattern):
                    continue
            else:
                if any([f.endswith('.'+x) for x in skiptypes]):
                    continue
            result.append(os.path.join(root,f))
    return result

def run_grep(pattern,filelist):
    import subprocess
    return subprocess.call(['grep',pattern]+filelist)

if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='tree grep utility')
    parser.add_argument('text_pattern')
    parser.add_argument('file_pattern',nargs='?')
    args = parser.parse_args()

    filelist = make_filelist(args.file_pattern)
    rc = run_grep(args.text_pattern,filelist)
    sys.exit(rc)
