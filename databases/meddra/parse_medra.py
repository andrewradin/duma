#!/usr/bin/python
from __future__ import print_function
import os, django, sys, argparse
from collections import defaultdict
sys.path.insert(1,"../../web1")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from path_helper import PathHelper
from algorithms.exit_codes import ExitCoder

# created 2.Jun.2016 - Aaron C Daugherty - twoXAR

# Parse downloaded and unlocked MedDRA data

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

if __name__=='__main__':
    #=================================================
    # Read in the arguments/define options
    #=================================================
    exitCoder = ExitCoder()

    arguments = argparse.ArgumentParser(description="Parse MedDRA ascii files")

    arguments.add_argument("mdhier", help="mdhier.asc")
    arguments.add_argument("llt", help="llt.asc")
    arguments.add_argument("o", help="Outfile name")

    args = arguments.parse_args()

    # return usage information if no argvs given

    if not args.mdhier or not args.llt:
        arguments.print_help()
        sys.exit(exitCoder.encode('usageError'))

    atr_file = args.o
    pt_syns = defaultdict(list)
    pt_llts = defaultdict(list)
    with open(args.llt) as f:
        for line in f:
            fields = line.rstrip().split('$')
            pt_syns[fields[2]].append(fields[1].strip(","))
            pt_llts[fields[2]].append(fields[0])
    codes_seen = []
    with open(atr_file, 'w') as o:
        o.write("\t".join(['meddra_code', 'attribute', 'value']) + "\n")
        with open(args.mdhier) as f:
            for line in f:
                fields = line.rstrip().split('$')
                if fields[3] not in codes_seen:
                    o.write("\t".join([fields[3], 'adr_term', fields[7]]) + "\n")
                    o.write("\t".join([fields[3], 'medra_level', 'soc']) + "\n")
                    codes_seen.append(fields[3])
                if fields[2] not in codes_seen:
                    o.write("\t".join([fields[2], 'adr_term', fields[6]]) + "\n")
                    o.write("\t".join([fields[2], 'medra_level', 'hlgt']) + "\n")
                    codes_seen.append(fields[2])
                o.write("\t".join([fields[2], 'parent_node', fields[3]]) + "\n")
                if fields[1] not in codes_seen:
                    o.write("\t".join([fields[1], 'adr_term', fields[5]]) + "\n")
                    o.write("\t".join([fields[1], 'medra_level', 'hlt']) + "\n")
                    codes_seen.append(fields[1])
                o.write("\t".join([fields[1], 'parent_node', fields[2]]) + "\n")
                if fields[0] not in codes_seen:
                    o.write("\t".join([fields[0], 'adr_term', fields[4]]) + "\n")
                    o.write("\t".join([fields[0], 'medra_level', 'pt']) + "\n")
                    o.write("\t".join([fields[0], 'pt_pref_soc', fields[10]]) + "\n")
                    for x in set(pt_syns[fields[0]]):
                        o.write("\t".join([fields[0], 'synonym', x]) + "\n")
                    for x in set(pt_llts[fields[0]]):
                        o.write("\t".join([fields[0], 'pt_llts', x]) + "\n")
                    codes_seen.append(fields[0])
                o.write("\t".join([fields[0], 'parent_node', fields[1]]) + "\n")
