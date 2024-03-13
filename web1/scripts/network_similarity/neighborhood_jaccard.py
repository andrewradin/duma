#!/usr/bin/python

from __future__ import print_function
from builtins import range
from collections import defaultdict
import networkx as nx
import numpy as np
import sys, argparse, os, django
sys.path.insert(1,"../../web1")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from path_helper import PathHelper
from algorithms.exit_codes import ExitCoder

def message(*objs):
    print(*objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def verboseOut(*objs):
    if args.verbose:
        print(*objs, file=sys.stderr)

def useage(*objs):
    print("USEAGE: ", *objs, file=sys.stderr)

def printOut(*objs):
    print(*objs, file=sys.stdout)

def is_pickle(filehandle):
    file_name, file_extension = os.path.splitext(filehandle)
    if file_extension == '.pickle':
        return True
    elif file_extension == '.gz' and file_name.endswith('.pickle'):
        return True
    return False

def jac_nodes(g, n1, n2, m=0):
    if n1 == n2:
        return 1
    n1n = get_list_of_neighbors(g, n1)
    n2n = get_list_of_neighbors(g, n2)
    for i in range(m):
        n1n = expand_neighborhood(g, n1n)
        n2n = expand_neighborhood(g, n2n)
    return score_jac(n1n, n2n)

def score_jac(list1, list2):
    # get only the unique elements
    set1 = set(list1)
    set2 = set(list2)
    cn = set1 & set2
    denom = float(len(set1 | set2))
    if denom == 0:
        return 0
    numer = float(len(cn))
    return numer/denom

# this approach is faster b/c we allow repeats (those are filtered later), but could get big fast
def expand_neighborhood(g, nl):
    # need to do it this way so that we don't keep looping infinitely
    origin_length = len(nl)
    for i in range(origin_length):
        n=nl[i]
        nl.extend(get_list_of_neighbors(g, n))
    return nl

def get_list_of_neighbors(g, n):
    try:
        toreturn = list(nx.all_neighbors(g, n))
    # this happens when the node is not in the graph
    except nx.exception.NetworkXError:
        # this is the equivalent of failing silently as we up extending a list with an empty list
        toreturn = []
    return toreturn

def build_ppi_graph(ppi_filename, direction=True, min_ppi_evid = 0.0):
    if is_pickle(ppi_filename):
        return nx.read_gpickle(ppi_filename)
    if direction:
        new_graph = nx.DiGraph()
    else:
        new_graph = nx.Graph()
    with open(ppi_filename, 'r') as f:
        header = f.readline()
        for l in f:
            c = l.rstrip().split("\t")
            if float(c[2]) < min_ppi_evid or c[0] == "-" or c[1] == "-":
                continue
            if len(c) > 3:
                direc = int(c[3])
            else:
                direc = 0
            new_graph.add_edge(c[0], c[1], weight = float(c[2]), direction = direc)
    return new_graph

def jac_set(node_list1, node_list2, g, m=0):
    if len(node_list1) == 0 or len(node_list2) == 0:
        return 0
    # For each step, m, add neighboring nodes
    for i in range(m):
        node_list1 = expand_neighborhood(g, node_list1)
        node_list2 = expand_neighborhood(g, node_list2)
    # after stepping, scored jaccards
    return score_jac(node_list1, node_list2)

def get_dpi_dict(dpi_filename):
    dpi = defaultdict(dict)
    with open(dpi_filename, 'r') as f:
        for l in f:
            c = l.rstrip().split("\t")
            if c[0] in list(dpi.keys()) and c[1] in list(dpi[c[0]].keys()) and dpi[c[0]][c[1]] != c[2]:
                warning("An interaction was seen twice with varying confidence scores for " + c[0] + " and " + c[1] + ". Using the first score.")
                continue
            dpi[c[0]][c[1]] = c[2]
    return dpi

def useage_error(arguments):
    arguments.print_help()
    sys.exit(exitCoder.encode('usageError'))

def get_drugs_interest(infile, dpi):
    with open(infile, 'r') as f:
        toreturn = [l.rstrip() for l in f if l.rstrip() in list(dpi.keys())]
    return toreturn

def get_drugs_compare(infile, dpi):
    if infile is None:
        return list(dpi.keys())
    else:
        return get_drugs_interest(infile, dpi)

def write_dict_of_dict_to_file(dd, ofile):
    with open(ofile, 'w') as f:
        f.write("\n".join(["\t".join([k1, k2, str(v)]) for k1, subdict in dd.items() for k2, v in subdict.items()]) + "\n")


if __name__=='__main__':

    #=================================================
    # Read in the arguments/define options
    #=================================================
    exitCoder = ExitCoder()

    arguments = argparse.ArgumentParser(description="A batch method of comparing the similarity of targets for sets of drugs, either for direct or indirect targets.")

    arguments.add_argument("--dpi", help="DPI file to use. Protein IDs need to match the PPI file")
    
    arguments.add_argument("-i", help="Drugs of interest. IDs need to match the IDs in the provided DPI file")
    
    arguments.add_argument("-c", help="Drugs to compare to. If not provided will compare to all drugs in DPI file.")
    
    arguments.add_argument("--direct", action="store_true", help="Report the jaccard index of direct targets")
    
    arguments.add_argument("--indirect", action="store_true", help="Report the jaccard index of indirect targets (i.e. one protein-protein interaction away). Must also provide PPI file.")
    
    arguments.add_argument("--ppi", help="PPI file to use. Only necessary if --indirect is provided, otherwise ignored")    
    
    arguments.add_argument("-o", help="Output file prefix")
    
    arguments.add_argument("--verbose", action="store_true", help="Print out status reports")
    
    args = arguments.parse_args()
    
    # return usage information if no argvs given
    
    if not args.dpi or not args.i or not args.o or (not args.direct and not args.indirect) or (args.indirect and not args.ppi):
        useage_error(arguments)
    
    #=========================================================
    # settings
    #=========================================================
    
    ppi_filename = args.ppi
    dpi_filename = args.dpi
    if args.direct:
        direct_file = args.o + "_direct_traget_jaccard.tsv"
    if args.indirect:
        indirect_file = args.o + "_indirect_traget_jaccard.tsv"
    
    #=========================================================
    # main
    #=========================================================
    
    dpi = get_dpi_dict(dpi_filename)
    
    if args.indirect:
        g = build_ppi_graph(ppi_filename)
    
    drugsOI = get_drugs_interest(args.i, dpi)
    compareDrugs = get_drugs_compare(args.c, dpi)
    
    if args.direct:
        out_direct = defaultdict(dict)
    if args.indirect:
        out_indirect = defaultdict(dict)
    
    for drugOI in drugsOI:
        for compareDrug in dpi.keys():
            if args.direct:
                out_direct[drugOI][compareDrug] = jac_set(list(dpi[drugOI].keys()), list(dpi[compareDrug].keys()), None, m=0)
            if args.indirect:
                out_indirect[drugOI][compareDrug] = jac_set(list(dpi[drugOI].keys()), list(dpi[compareDrug].keys()), g, m=1)
    
    if args.direct:
        write_dict_of_dict_to_file(out_direct, direct_file)
    if args.indirect:
        write_dict_of_dict_to_file(out_indirect, indirect_file)

