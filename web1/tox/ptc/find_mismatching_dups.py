#!/usr/bin/python

from __future__ import print_function
from collections import defaultdict

infile = '/home/ubuntu/2xar/ws/drugsets/rt.chembl.full.test.tsv'

all_entries = defaultdict(dict)
with open(infile, 'r') as f:
    print(f.readline().rstrip())
    for l in f:
        flds = l.rstrip().split("\t")
        if flds[0] in list(all_entries.keys()) and flds[1] in list(all_entries[flds[0]].keys()) and all_entries[flds[0]][flds[1]] != flds[2]:
            all_entries[flds[0]][flds[1]] = 'False'
        else:
            all_entries[flds[0]][flds[1]] = flds[2]


for k in all_entries.keys():
    for at,v in all_entries[k].items():
        print("\t".join([k,at,v]))
