#!/usr/bin/python
from __future__ import print_function
import os, django, sys, argparse
sys.path.insert(1,"../../web1")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from path_helper import PathHelper
from algorithms.exit_codes import ExitCoder
from collections import defaultdict

# created 23.Jun.2016 - Aaron C Daugherty - twoXAR

# Aggregate an adr file up the meddra hierarchy

# TO DO:
# A few subroutines to print out as I want

def message(*objs):
    print(*objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def verboseOut(*objs):
    if verbose:
        print(*objs, file=sys.stderr)

def useage(*objs):
    print("USEAGE: ", *objs, file=sys.stderr)

def printOut(*objs):
    print(*objs, file=sys.stdout)

def condense_report(ddl, presAbs, prefix, header, ofile):
    new_ddl = defaultdict(lambda: defaultdict(list))
    if header is not None:
        with open(ofile, 'w') as f:
            if presAbs:
                third_col = 'presence'
            else:
                third_col = header[2]
            f.write("\t".join([header[0], header[1], third_col]) + "\n")
        temp = 1
    with open(ofile, 'a') as f:
        for term in ddl.keys():
            for drug in ddl[term].keys():
                evid = condense_evids(presAbs, ddl[term][drug])
                f.write("\t".join([drug, prefix + term, str(evid)]) +"\n")
                new_ddl[term][drug] = [evid]
    return new_ddl

def condense_evids(presAbs, evid_list):
    # first pass just report the max value
    return max(evid_list)

if __name__=='__main__':
    #=================================================
    # Read in the arguments/define options
    #=================================================
    exitCoder = ExitCoder()
    
    arguments = argparse.ArgumentParser(description="Aggregate an ADR file up the MedDRA hierarchy")
    
    arguments.add_argument("meddra", help="MedDRA file, e.g. meddra.v19.tsv")
    
    arguments.add_argument("adr", help="adr file, e.g. adr.sider.portion.tsv")
    
    args = arguments.parse_args()

    # return usage information if no argvs given

    if not args.meddra or not args.adr:
        arguments.print_help()
        sys.exit(exitCoder.encode('usageError'))
    
    adr_prefix = os.path.splitext(args.adr)[0]
    ofile = adr_prefix + '_aggregated.tsv'
    prefix_dic = { 'llt': 'pt_'
                , 'pt' : 'ht_'
                , 'hlt' : 'hgt_'
                , 'hlgt' : 'soc_'
               }
    
    lvl_dl = defaultdict(list)
    parent_dl = defaultdict(list)
    atrs_of_interst = ['parent_node', 'medra_level', 'pt_llts']
    with open(args.meddra, 'r') as f:
        for l in f:
            fields = l.strip().split("\t")
            if fields[1] not in atrs_of_interst:
                continue
            elif fields[1] == 'parent_node':
                parent_dl[fields[0]].append(fields[2])
            elif fields[1] == 'pt_llts':
                for llt in fields[2].split(","):
                    parent_dl[llt].append(fields[0])
                    lvl_dl['llt'].append(fields[0])
            elif fields[1] == 'medra_level':
                lvl_dl[fields[2]].append(fields[0])
                # because pts are also llts...
                if fields[2] == 'pt' :
                    lvl_dl['llt'].append(fields[0])
                    parent_dl[fields[0]].append(fields[0])

    child_drug_ddl = defaultdict(lambda: defaultdict(list))
    presAbs = False
    with open(args.adr, 'r') as f:
        header = None
        for l in f:
            fields = l.strip().split("\t")
            if not header:
                header = fields
                continue
            if len(fields) < 3:
                fields.append('1')
                presAbs = True
            child_drug_ddl[fields[1]][fields[0]].append(float(fields[2]))
    
    prev_d = child_drug_ddl
    for lvl in ['llt', 'pt', 'hlt', 'hlgt']:
        current_d = defaultdict(lambda: defaultdict(list))
        for child in list(set(lvl_dl[lvl])): # e.g. for each llt
            if lvl == 'llt':
                gen = (p for p in set(parent_dl[child]) if p not in lvl_dl['hlt'])
            elif lvl == 'pt':
                gen = (p for p in set(parent_dl[child]) if p in lvl_dl['hlt'])
            else:
                gen = (p for p in set(parent_dl[child]))
            for parent in gen: # for each parent node of that child node
                if child in prev_d.keys():
                    for drug in prev_d[child].keys():
                        current_d[parent][drug] = current_d[parent][drug] + prev_d[child][drug]
#        printOut(lvl, str(len(prev_d.keys())), str(len(current_d.keys())))
        prev_d = condense_report(current_d, presAbs, prefix_dic[lvl], header, ofile)
        header = None # after the first time through we don't want to print the header
