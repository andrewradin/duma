#!/usr/bin/env python

import sys
sys.path.insert(1,"../../web1")
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
import django
django.setup()

import dtk.drug_clusters as dc

def check_relation_field(outList):
    if len(outList) == 4:
        return outList + ['.']
    return outList

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='rewrite dpi files with common keys')
    parser.add_argument('outdir')
    parser.add_argument('infile',nargs='+')
    args = parser.parse_args()

    # since the final DPI file needs to be keyed to a single
    # namespace, this script should create mappings for each
    # possible cluster into each possible keyspace; we can then
    # choose one or more of those to consolidate and push to the
    # platform

    keymap = {}
    all_namespaces = set()
    # { <from_pair>:[<to_pair>,<to_pair>,...],...}
    for line in open('dump.out'):
        pairs = dc.assemble_pairs(line.strip('\n').split('\t'))
        seen = set()
        target = []
        for pair in pairs:
            all_namespaces.add(pair[0])
            if pair[0] not in seen:
                seen.add(pair[0])
                target.append(pair)
        for pair in pairs:
            keymap[pair] = target

    outmap = {}
    for keytype in all_namespaces:
        outmap[keytype] = open(args.outdir+'/'+keytype,'w')

    for name in args.infile:
        stem = name.split('/')[-1]
        inp = open(name)
        key = None
        for line in inp:
            fields = line.strip('\n').split('\t')
            if key is None:
                key = fields[0]
                continue
            gk = (key,fields[0])
            if gk in keymap:
                for pair in keymap[gk]:
                    outList = check_relation_field([pair[1]]+fields[1:])
                    outmap[pair[0]].write('\t'.join(outList + [gk[0]])+'\n')
            else:
                outList = check_relation_field(fields)
                # list adding wasn't working for some reason, so I just nested the lists
                outmap[key].write('\t'.join([i for sl in [outList, [key]] for i in sl])+'\n')

