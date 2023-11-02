#!/usr/bin/python
from __future__ import print_function
import argparse
arguments = argparse.ArgumentParser(description="Convert stitch to drugbank. One ID per line.")
arguments.add_argument("-i", help="One stitch ID per line")
args = arguments.parse_args()
import napa_build as nb

with open(args.i, 'r') as f:
    for l in f:
        stitch = l.rstrip().split("\t")[0]
        dbid = nb.convert_to_dbid(stitch, 'stitch_id')
        if dbid:
            print("\t".join([dbid, stitch]))
