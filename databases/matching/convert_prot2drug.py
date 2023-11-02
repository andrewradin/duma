#!/usr/bin/env python

import sys
sys.path.insert(1,"../../web1")
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
import django
django.setup()

def convert_prot2drug_file(path,detail):
    filename = path.split('/')[-1]
    prefix = 'prot2drug_'
    suffix = '.csv'
    assert filename.startswith(prefix)
    assert filename.endswith(suffix)
    keyspace = filename[len(prefix):-len(suffix)]
    import csv
    with open(path,'rb') as inp:
        rdr = csv.reader(inp)
        header = None
        with open('dpi.'+keyspace+'.'+detail+'.tsv','w') as outp:
            cols = [keyspace+'_id'
                    ,'uniprot_id'
                    ,'evidence'
                    ,'direction'
                    ]
            outp.write('\t'.join(cols)+'\n')
            for row in rdr:
                if not header:
                    header = row
                    continue
                out = [row[2],row[0],row[4],row[5]]
                outp.write('\t'.join(out)+'\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='prot2drug conversion utility')
    parser.add_argument('--detail',default='default')
    parser.add_argument('file')
    args = parser.parse_args()

    convert_prot2drug_file(args.file,args.detail)
