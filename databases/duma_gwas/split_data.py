#!/usr/bin/env python

"""
Splits input TSV input 1000 output files randomly based on first column.
This means all lines with the same first column will end up in the same file.
"""


def split(outdir):
    import os
    import gzip
    import sys
    files = {}
    def get_file(key):
        if key not in files:
            path = os.path.join(outdir, key + '.gz')
            files[key] = gzip.open(path, mode='wt', compresslevel=1)
        return files[key]

    for line in sys.stdin:
        tabIdx = line.index('\t')
        key = str(hash(line[:tabIdx]) % 1000)

        of = get_file(key)
        of.write(line)


if __name__ == '__main__':
    import sys
    split(sys.argv[1])
