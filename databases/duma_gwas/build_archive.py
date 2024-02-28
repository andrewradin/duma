#!/usr/bin/env python3

"""
Builds a zip file from the lines in the input, grouping them by first column.
Assumes that data comes in grouped by that first column (i.e. sort first).
"""

def build(ofile, total_lines, prefix_length):
    import zipfile
    import sys
    import os
    from tqdm import tqdm

    with zipfile.ZipFile(ofile, mode='w', compression=zipfile.ZIP_DEFLATED) as out:
        cur_data = []
        cur_key = None

        def flush_entry(data, key):
            if not data:
                return

            # Note: Reader also has to do this.
            # /'s are illegal in filenames and create new directories in the tar.
            key = key.replace('/', '_')
            out.writestr(key, ''.join(data))

        for line in tqdm(sys.stdin, total=total_lines, smoothing=0):
            tab_idx = line.index('\t')
            key = line[:tab_idx]
            if prefix_length is not None:
                key = key[:prefix_length]
            if key != cur_key:
                flush_entry(cur_data, cur_key)
                cur_data = []
                cur_key = key
            cur_data.append(line)
        flush_entry(cur_data, cur_key)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-o", "--outfile",  required=True,  help="output (zip) file")
    parser.add_argument("-t", "--total-lines",  type=int, default=None, help="How many lines in the input (for progress)")
    parser.add_argument("-p", "--prefix-length",  type=int, default=None, help="How much of each column to use as index.  Useful if you have unique keys and want to group by shared prefixes.")
    from dtk.log_setup import addLoggingArgs, setupLogging
    addLoggingArgs(parser)
    args = parser.parse_args()
    setupLogging(args)
    build(args.outfile, args.total_lines, args.prefix_length)
