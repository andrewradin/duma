#!/usr/bin/env python

from __future__ import print_function
from builtins import range
import os, sys
try:
    from path_helper import PathHelper, make_directory
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../..")
    from path_helper import PathHelper, make_directory

import networkx as nx
from collections import defaultdict
from algorithms.exit_codes import ExitCoder
import napa_build as nb

# created 23.Feb.2016 - Aaron C Daugherty - twoXAR
# An initial attempt to use a network approach to connect ADRs and attributes

# A few subroutines to print out as I want

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

def add_adrs_to_graph(g, adr_file, score_col):
    # first read in all drugs with ADRs
    verboseOut("Reading in ADR-attribute path data...")
    with open(adr_file, 'r') as f:
        header = f.readline().rstrip().split("\t")
        for l in f:
            fields = l.rstrip().split()
            atr_direc = fields[1].split("_")
            if score_col == 'all':
                score_types = header[2:]
            else:
                try:
                    score_types = header[int(score_col)]
                except:
                    warning("Issue with score_col. Expect this to be an integer >= 2, or the word 'all'. Quitting.")
                    sys.exit(exitCoder.encode('usageError'))
            edge_attrs = {score_types[ind] : float(fields[ind + 2]) for ind in range(len(score_types))}
            edge_attrs['direction'] = atr_direc[1]
# currently the direction is from attribute to ADR, but we could have it go both directions
            g.add_edge(a_prefix + atr_direc[0], adr_prefix + fields[0], edge_attrs)
#            g.add_edge(adr_prefix + fields[0], a_prefix + atr_direc[0], edge_attrs)
    return g, score_types

def set_drug_attr_wts(g, value, edge_attr_name='weight'):
    drug_prefix, adr_prefix, a_prefix = nb.establish_prefixes()
    attr_node_gen = (node for node in nx.nodes(g) if node.startswith(a_prefix))
    for a in attr_node_gen:
        nbr_gen = (n for n in nx.all_neighbors(g, a) if n.startswith(drug_prefix) or n.startswith(a_prefix))
        for n in nbr_gen:
            g[n][a][edge_attr_name] = value
    return g

def summary_report(g, prnt = True):
    drug_prefix, adr_prefix, a_prefix = nb.establish_prefixes()
    lines = []
    lines.append("total nodes:" + str(len(nx.nodes(g))))
    ds = []
    adrs = []
    attrs = []
    for i in nx.nodes(g):
        if i.startswith(drug_prefix):
            ds.append(i)
        elif i.startswith(adr_prefix):
            adrs.append(i)
        elif i.startswith(a_prefix):
            attrs.append(i)
    lines.append("drug nodes:" + str(len(ds)))
    lines.append("ADR nodes:" + str(len(adrs)))
    lines.append("attribute nodes:" + str(len(attrs)))
    if prnt:
        printOut("\n".join(lines))
    else:
        return lines

def find_paths(g, d_raw, indirect, score_types):
    drug_prefix, adr_prefix, a_prefix = nb.establish_prefixes()
    d = drug_prefix + d_raw
    if d not in nx.nodes(g):
        verboseOut("warning:", d_raw, " not in the provided drugs.")
        return None
    verboseOut("Connecting ", d_raw, " to side effects...")
    paths = defaultdict(dict)
    d_atr_gen = (a for a in nx.all_neighbors(g, d) if a.startswith(a_prefix))
    for a in d_atr_gen:
        a_d_w = g.edge[d][a]['weight']
        if indirect:
            a_atr_gen = (ia for ia in nx.all_neighbors(g,a) if ia.startswith(a_prefix))
            for ia in a_atr_gen:
                a_ia_w = g.edge[a][ia]['weight']
                ia_adr_gen = (adr for adr in nx.all_neighbors(g, ia) if adr.startswith(adr_prefix))
                for adr in ia_adr_gen:
                    # only count those side effects that come from this attribute going the same direction as this drug causes this attribute to go
                    if g.edge[d][a]['direction'] == g.edge[a][adr]['direction']:
                        weights = []
                        for w in score_types:
                            weights.append(g.edge[ia][adr][w] * a_d_w * a_ia_w)
                        paths[adr][a + "_" + g.edge[a][adr]['direction']] = weights
        else:
            a_adr_gen = (adr for adr in nx.all_neighbors(g, a) if adr.startswith(adr_prefix))
            for adr in a_adr_gen:
                # only count those side effects that come from this attribute going the same direction as this drug causes this attribute to go
                if int(g.edge[d][a]['direction']) == int(g.edge[a][adr]['direction']):
                    weights = []
                    for w in score_types:
                        weights.append(g.edge[a][adr][w] * a_d_w)
                    paths[adr][a + "_" + g.edge[a][adr]['direction']] = weights
    return paths

def calc_sums(paths, score_types):
    import statistics
    drug_prefix, adr_prefix, a_prefix = nb.establish_prefixes()
    meds = {}
    maxs = {}
    cnts = {}
    for s in paths.keys():
        adr = s.lstrip(adr_prefix)
        cnts[adr] = len(list(paths[s].values()))
        scores = defaultdict(list)
        for l in paths[s].values():
            if len(l) != len(score_types):
                warning("Something went wrong with the scores")
                sys.exit()
            for i in range(len(score_types)):
                scores[score_types[i]].append(l[i])
        meds[adr] = [statistics.median(scores[score]) for score in score_types]
        maxs[adr] = [max(scores[score]) for score in score_types]
    return cnts, meds, maxs

def write_results(out_dl, ofp, to_report, score_types):
    for adr in out_dl.keys():
        ofile = os.path.join(ofp, adr + '.tsv')
        with open(ofile, 'w') as f:
            header = nb.get_header(to_report)
            f.write("\t".join(header) + "\n")
            f.write("\n".join(out_dl[adr]) + "\n")

def update_adr_out_dict(drug, to_report, sd, current_d, path_summary_stat='median'):
    drug_prefix, adr_prefix, a_prefix = nb.establish_prefixes()
    if to_report == 'paths':
        for adr, sd2 in sd['paths'].items():
            for at, v in sd2.items():
                current_d[adr].append("\t".join([drug, adr.lstrip(adr_prefix), at.lstrip(a_prefix), str(v)]))
    elif to_report == 'ft_list':
        for adr in sd['paths'].keys():
            writeOut = [drug, adr.lstrip(adr_prefix)]
            csv_list = ",".join([at.lstrip(a_prefix) for at in sd['paths'][adr].keys()])
            current_d[adr].append("\t".join(writeOut + [csv_list]))
    else:
        if path_summary_stat == 'both':
            path_summary_stats = ['median', 'max']
        else:
            path_summary_stats = [path_summary_stat]
        for adr in sd[path_summary_stats[0]].keys():
            writeOut = [drug, adr, str(sd['counts'][adr])]
            for pss in path_summary_stats:
                for v in sd[pss][adr]:
                    writeOut.append(str(v))
            current_d[adr].append("\t".join(writeOut))
    return current_d

if __name__=='__main__':
    import argparse    
    #=================================================
    # Read in the arguments/define options
    #=================================================
    exitCoder = ExitCoder()
    
    arguments = argparse.ArgumentParser(description="Connect drugs to side effects, via attributes")
    
    arguments.add_argument("--adr", help="adr attribute path sums")
    
    arguments.add_argument("--score_col", default='all'
                            ,help="In the adr file, which 0-based columns contains " +
                              "the value to use as weight. All uses all columns 2 and " +
                              "above (default %(default)s)"
                           )
    
    arguments.add_argument("--d_a_type", help="What type of drug attibute are you connecting to ADRs:" +
                                                 " protein, molec_bits, structure_alerts")
    
    arguments.add_argument("--d_a_file", help="Drug - attribute file to use")
    
    arguments.add_argument("--pre_saved", help="Use a preconstructed graph")
    
    arguments.add_argument("-o", help="Output directory (will be made if non-existant)")
    
    arguments.add_argument("--indirect", action="store_true"
                            ,help="Include indrect targets (i.e one protein-protein interaction away)." +
                                   " Must also provide PPI file."
                           )
    
    arguments.add_argument("--ppi", help="PPI file to use. Only necessary if --indirect is provided, " +
                                           "otherwise ignored")
    
    arguments.add_argument("--wsid", type=int, default=49, help="Workspace ID. default %(default)s")

    arguments.add_argument("--no_weights", action="store_true"
                            ,help="For the predicted ADR - drug-attribute paths, " +
                                  "don't multiply the selected measures by the drug-attribute edge weight."
                          )
    
    arguments.add_argument("--pickle", action="store_true", help="Save the final graph")
    
    arguments.add_argument("--no_atr_adr_portion", action="store_true"
                            ,help="Do not report the portion of ADRs that an attribute is associated with")
    
    arguments.add_argument("--to_report", default='stats'
                            ,help="What to report for each adr: stats " +
                                    "(the median path value for all adr-attribute paths) OR" +
                                    " paths (each path from ADR to attribute) OR" +
                                    " ft_list: returns a CSV list of all attributes connected to the ADR" +
                                    " and the drug (default %(default)s)"
                           )
    
    arguments.add_argument("--path_summary_stat", default='median'
                             ,help="How to summarise scores for each predicted drug-ADR connected:" +
                                     " max, median, or both (default %(default)s)")
    
    arguments.add_argument("--verbose", action="store_true", help="Print out status reports")
    
    args = arguments.parse_args()
    
    # return usage information if no argvs given
    
    if not args.d_a_type and not args.adr or not args.o or not args.wsid or (not args.pre_saved and not args.d_a_file) or (args.indirect and not args.ppi):
        arguments.print_help()
        sys.exit(exitCoder.encode('usageError'))
    
    #======================================================================
    # main
    #======================================================================
    make_directory(args.o)
    drug_prefix, adr_prefix, a_prefix = nb.establish_prefixes()
    if args.pre_saved:
        g = nx.read_gpickle(args.pre_saved)
        # this is a complicated way to figure out what edge attributes/weights were recorded for the adr-attribute pairs
        score_types = []
        for node, neighbors in g.adjacency_iter():
            for nbr, edge_attr in neighbors.items():
                if (node.startswith(a_prefix) and nbr.startswith(adr_prefix)) or (node.startswith(adr_prefix) and nbr.startswith(a_prefix)):
                    score_types = list(edge_attr.keys())
                    if 'direction' in score_types:
                        score_types.remove('direction')
                    break
    else:
        from browse.models import WsAnnotation
        import django
        if not "DJANGO_SETTINGS_MODULE" in os.environ:
            os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
            django.setup()
        drugs_to_check = [str(rec.agent_id) for rec in WsAnnotation.objects.filter(ws_id=args.wsid)]
        #drugs_to_check = [line.strip().upper() for line in open(args.drugs_to_check, 'r')]
        verboseOut("Reading in drug-", args.d_a_type, "data...")
        g = nb.get_da_graph(args.d_a_type, args.d_a_file, args.wsid, args.pickle, args.o + '/')
        if args.d_a_type == 'protein' and args.indirect:
            g = nb.add_in_ppi(args.ppi, g, args.pickle, save_file = args.o + '/ppi.pickle.gz')
        # read in pre-made adr-atr path sums
        verboseOut("Adding adrs to graph...")
        g, score_types = add_adrs_to_graph(g, args.adr, args.score_col)
        if args.pickle:
            nx.write.gpickle(g, args.o + "/finalGraph.pickle.gz")
    
    if args.no_weights:
        verboseOut("Not including drug-attribute edge weights.")
        g = set_drug_attr_wts(g, 1)
    
    if args.verbose:
        summary_report(g)
    
    attrs_adrs = defaultdict(list)
    out_dl = defaultdict(list)
    for d in drugs_to_check:
        drug_adrs = {}
        drug_adrs['paths'] = find_paths(g, d, args.indirect, score_types)
        if drug_adrs['paths'] is not None:
            # now collapse the attributes, regardless of connecting drug
#            drug_adrs['counts'], drug_adrs['median'] = calc_sums(drug_adrs['paths'], score_types)
# I think we're better off to just report one value for each score type, but I initially wrote this to report both the median and max of each
            drug_adrs['counts'], drug_adrs['median'], drug_adrs['max'] = calc_sums(drug_adrs['paths'], score_types)
            out_dl = update_adr_out_dict(d, args.to_report, drug_adrs, out_dl, path_summary_stat=args.path_summary_stat)
            if not args.no_atr_adr_portion:
                for adr in drug_adrs['paths'].keys():
                    atrs = [at.lstrip(a_prefix) for at in drug_adrs['paths'][adr].keys()]
                    adr = adr.lstrip(adr_prefix)
                    for at in atrs:
                        attrs_adrs[at].append(adr)
        else:
            verboseOut("warning: No paths found for:", d)
    write_results(out_dl, args.o, args.to_report, score_types)
    if not args.no_atr_adr_portion:
        adr_cnt = len(list(out_dl.keys()))
        with open(args.o + '/atr_adr_portions.tsv', 'w') as f:
            for atr in attrs_adrs.keys():
                uniq_adr_cnt = len(set(attrs_adrs[atr]))
                f.write(atr + "\t" + str(float(uniq_adr_cnt) / adr_cnt) + "\n")
    verboseOut("Finished successfully")
