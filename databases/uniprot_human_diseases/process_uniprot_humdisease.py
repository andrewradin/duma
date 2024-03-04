#!/usr/bin/python
from __future__ import print_function
import os, django, sys, argparse, re
from collections import defaultdict
sys.path.insert(1,"../../web1")
sys.path.insert(1,"../chembl")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from path_helper import PathHelper
from algorithms.exit_codes import ExitCoder

# A program to parse Stitch data to produce DPI and Attribute files

# A few subroutines to print out as I want

def message(*objs):
    print(*objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def error(*objs):
    print("ERROR: ", *objs, file=sys.stderr)
    sys.exit(-1)

def get_terms_to_attrs():
    terms_to_attrs = {'ID': 'canonical'
                      ,'AC': 'accession'
                      ,'AR': 'acronym'
                      ,'DE': 'definition'
                      ,'SY': 'synonym'
                      ,'KW': 'keyword'
                      }
    return terms_to_attrs

def check_line_types(line):
    temp = get_terms_to_attrs()
    acceptable_line_types = temp.keys()
    acceptable_line_types.append('//')
    acceptable_line_types.append('DR')
    for term in acceptable_line_types:
        if line.startswith(term):
            return True
    return False

def flush(attrs, o):
    if 'accession' not in attrs.keys():
        warning(attrs)
        sys.exit()
    acc = attrs['accession']
    o.write("\t".join([acc, 'canonical', attrs['canonical']]) + "\n")
    for k,v in attrs.items():
        if k == 'canonical' or k == 'accession':
            continue
        if type(v) == set or type(v) == list:
            for part in v:
                o.write("\t".join([acc, k, part]) + "\n")
        elif v is not None and v != 'None':
            o.write("\t".join([acc, k, v]) + "\n")
    return dict()

#=================================================
# Read in the arguments/define options
#=================================================

if __name__ == '__main__':
    out_filename='humdisease.out.tsv'
    parser = argparse.ArgumentParser(description='Stitch file conversion utility')
    parser.add_argument('--hd'
            ,help='input file from http://www.uniprot.org/docs/humdisease.txt; create %s' % out_filename
            )
    args = parser.parse_args()
    
    if not args.hd:
        error("Need uniprot file from http://www.uniprot.org/docs/humdisease.txt")
    
    # start with the DPI info, as we only want chemicals with DPI info
    attrs = {}
    terms_to_attrs = get_terms_to_attrs()
    terms_with_mult = ['SY', 'KW']
    with open(args.hd, 'r') as f:
        with open(out_filename, 'w') as o:
            o.write("\t".join(['uniprot_disease_accession', 'attribute', 'value']) + "\n")
            for line in f:
                if not check_line_types(line):
                    continue
                fields = line.rstrip().split('   ')
                if fields[0] == '//':
                     attrs = flush(attrs, o)
                elif fields[0] == 'DR':
                    flds = fields[1].split(";")
                    k = flds[0] + "_id"
                    if k in attrs.keys():
                        attrs[k].append(flds[1])
                    else:
                        attrs[k] = [flds[1]]
                elif fields[0] == 'DE':
                    k = terms_to_attrs[fields[0]]
                    if k in attrs.keys():
                        attrs[k] = attrs[k] + " " + fields[1]
                    else:
                        attrs[k] = fields[1]
                elif fields[0] in terms_with_mult:
                    k = terms_to_attrs[fields[0]]
                    if k in attrs.keys():
                        attrs[k].append(fields[1])
                    else:
                        attrs[k] = [fields[1]]
                else:
                    attrs[terms_to_attrs[fields[0]]] = fields[1]
        
    
