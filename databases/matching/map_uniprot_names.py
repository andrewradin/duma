#!/usr/bin/env python3

from __future__ import print_function
import sys
import os
import django_setup

from dtk.readtext import parse_delim

def convert(src,dst,uni_ver,show_detail):
    seen_header = False
    drugs = {}
    drugsout = set()
    inprots = set()
    changedprots = set()
    keptprots = set()
    droppedprots = set()
    outprots = set()
    from dtk.s3_cache import S3File
    uni_s3f = S3File.get_versioned('uniprot',uni_ver,'Uniprot_data')
    uni_s3f.fetch()
    protmap = {
            v:u
            for u,a,v in parse_delim(open(uni_s3f.path()))
            if a == 'Alt_uniprot' and u != v
            }
    valid_uniprots = set([u for u,a,v in parse_delim(open(uni_s3f.path()))])
    print(len(valid_uniprots),'valid uniprots')
    print(len(protmap),'alias mappings')
### There can be a variable number of fields for some c50 files
### but there are always the core 4
### so some special handling around that
    for tup in src:
        key,uniprot,ev,dirn = tup[:4]
        remain = list(tup[4:])
        if not seen_header:
            assert uniprot == 'uniprot_id'
            header_key = key
            seen_header = True
        else:
            d=drugs.setdefault(key,(set(),set()))
            inprots.add(uniprot)
            d[0].add(uniprot)
            if uniprot in protmap:
                changedprots.add(uniprot)
                uniprot = protmap[uniprot]
            elif uniprot in valid_uniprots:
                keptprots.add(uniprot)
            else:
                droppedprots.add(uniprot)
                continue
            d[1].add(uniprot)
            outprots.add(uniprot)
            drugsout.add(key)
        dst.write('\t'.join([key,uniprot,ev,dirn]+remain)+'\n')
    altered = {k:v for k,v in drugs.items() if len(v[0]) != len(v[1])}
    print(len(altered),'drugs with changed numbers of proteins')
    if show_detail:
        for k,v in altered.items():
            old = v[0]-v[1]
            new = set([protmap[x] for x in old if x in protmap])
            had = v[0]&new
            print('  ',k,'already had '+' '.join(had) if had else '')
            if len(new) < len(old):
                print('      ',' '.join(old),'->',' '.join(new))
            else:
                merged = [x for x in old if protmap[x] in had]
                print('      ','remapped',' '.join(merged))
    print(len(drugs),'drugs processed')
    print(len(drugsout),'drugs output')
    print(len(inprots),'input protein keys')
    print(len(keptprots),'distinct retained protein keys')
    print(len(changedprots),'distinct remapped protein keys')
    print(len(droppedprots),'distinct dropped protein keys')
    print(len(outprots),'output protein keys')

# XXX this is invoked in 3 places: chembl, bindingdb, and matching;
# XXX all should be updated to pass the new version parameter
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description='convert DPI/PPI to canonical uniprot names',
            )
    parser.add_argument('--detail',action='store_true')
    parser.add_argument('infile',type=open)
    parser.add_argument('outfile',type=argparse.FileType('w'))
    parser.add_argument('uniprot_version')
    args = parser.parse_args()

    convert(
            parse_delim(args.infile),
            args.outfile,
            args.uniprot_version,
            args.detail,
            )
