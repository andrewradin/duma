#!/usr/bin/python
from __future__ import print_function
import argparse
arguments = argparse.ArgumentParser(description="Convert DBID to stitch ID. One ID per line.")
arguments.add_argument("-i", help="One drugBank ID per line")
args = arguments.parse_args()
import napa_build as nb

pbcid_to_dbid = nb.get_pbcid_to_dbid(type='local')
convert={}
for p,d in pbcid_to_dbid.items():
    stitchID = 'CID' + p.zfill(9)
    convert[d] = stitchID

with open(args.i, 'r') as f:
    for l in f:
        dbid = l.rstrip().split("\t")[0]
        if dbid in list(convert.keys()):
            print("\t".join([dbid, convert[dbid]]))
