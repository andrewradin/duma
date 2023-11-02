#!/usr/bin/env python

from __future__ import print_function
from builtins import range
import os, sys
import django
if not "DJANGO_SETTINGS_MODULE" in os.environ:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    django.setup()

try:
    from path_helper import PathHelper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../..")
    from path_helper import PathHelper

import networkx as nx
from collections import defaultdict
try:
    import neighborhood_jaccard as nj
except ImportError:
    sys.path.insert(1,os.path.join(PathHelper.website_root, "scripts", "network_similarity"))
    import neighborhood_jaccard as nj

try:
    import run_eval_weka as rew
except ImportError:
    sys.path.insert(1,os.path.join(PathHelper.website_root, "ML"))
    import run_eval_weka as rew

from algorithms.exit_codes import ExitCoder
from browse.models import WsAnnotation

# created 23.Feb.2016 - Aaron C Daugherty - twoXAR
# An initial attempt to use a network approach to connect side effects and attributes

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

def establish_prefixes():
    return "d_", "adr_", "a_"

def convert_to_wsa(drug, wsid, current_type=None):
    if drug.startswith('CHEMBL') or current_type.lower() == 'chembl_id':
        if 'chembl_to_wsa' not in globals():
            get_chembl_to_wsa(wsid)
        try:
            return chembl_to_wsa[drug]
        except KeyError:
            return False
    elif current_type.lower() == 'stitch_id':
        pcid = convert_stitch_to_pbcid(drug)
        return convert_to_wsa(pcid, wsid, current_type='pubchem_cid')
    elif current_type.lower() == 'pubchem_cid':
        if 'pbcid_to_wsa' not in globals():
            get_pbcid_to_wsa(wsid)
        try:
            return pbcid_to_wsa[drug]
        except KeyError:
            return False
    else:
        warning("Unrecognized drug id type. Unable to convert")
        return False

def convert_stitch_to_pbcid(pbcid):
    no_prefix = pbcid[3:]
    return no_prefix.lstrip('0')
    
def get_pbcid_to_wsa(wsid, type='global'):
    if type=='global':
        global pbcid_to_wsa
    pbcid_to_wsa={}
    from drugs.models import Prop,Tag
    pc_prop=Prop.objects.get(name='pubchem_cid').id
    pc_map={x.drug_id:x.value for x in Tag.objects.filter(prop_id=pc_prop)}
    for rec in WsAnnotation.objects.filter(ws_id=wsid):
        try:
            i = rec.agent_id
            pbcid_to_wsa[pc_map[i]] = str(rec.agent_id)
        except KeyError:
            pass
    return pbcid_to_wsa

def get_chembl_to_wsa(wsid, type='global'):
    if type=='global':
        global chembl_to_wsa
    chembl_to_wsa={}
    from drugs.models import Prop,Tag
    chembl_prop=Prop.objects.get(name='m_chembl_id').id
    chembl_map={x.drug_id:x.value for x in Tag.objects.filter(prop_id=chembl_prop)}
    for rec in WsAnnotation.objects.filter(ws_id=wsid):
        try:
            i = rec.agent_id
            chembl_to_wsa[chembl_map[i]] = str(rec.agent_id)
        except KeyError:
            pass
    return chembl_to_wsa

def get_da_graph(d_a_type, d_a_file, wsid, pickle, prefix):
    save_file = prefix + "_" + d_a_type + '_dpi.pickle.gz'
    if d_a_type == 'protein':
        return get_dpi_graph(d_a_file, wsid, pickle, save_file = save_file)
    elif d_a_type == 'molec_bits':
        return get_mb_graph(d_a_file, pickle, wsid, save_file = save_file)
    elif  d_a_type == 'struc_alerts':
        return get_sa_graph(d_a_file, wsid, pickle, save_file = save_file)
    else:
        useage("Unrecognized d_a_type provided. Only support: protein, molec_bits, struc_alerts")
        sys.exit(exitCoder.encode('usageError'))

def get_sa_graph(sa_file, wsid, pickle, save_file = 'sa.pickle.gz'):
    if nj.is_pickle(sa_file):
        return nx.read_gpickle(sa_file)
    drug_prefix, adr_prefix, a_prefix = establish_prefixes()
    g = nx.DiGraph()
    with open(sa_file, 'r') as f:
        for l in f:
            fields = l.rstrip().split("\t")
            if not fields[1].startswith('sa_') or not bool(fields[2]):
                continue
            drug = convert_to_wsa(fields[0], wsid)
            if not drug:
                continue
            alert = fields[1].lstrip('sa_').replace("_","") # no _ allowed. We use those for adding info to the node name
            g.add_edge(drug_prefix + drug, a_prefix + alert, weight = 1.0, direction=0)
    if pickle:
        nx.write_gpickle(g, save_file)
    return g

def get_dpi_graph(dpi_filename, wsid, save, save_file='dpi.pickle.gz', direction=True):
    if nj.is_pickle(dpi_filename):
        return nx.read_gpickle(dpi_filename)
    drug_prefix, adr_prefix, a_prefix = establish_prefixes()
    if direction:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    from dtk.prot_map import DpiMapping,stringize_value_lists
    dpi_map_obj = DpiMapping(dpi_filename)
    dpi_map = stringize_value_lists(dpi_map_obj.get_drug_id_map())
    with open(dpi_map_obj.get_path(), 'r') as f:
        header = f.readline()
        for l in f:
            c = l.rstrip().split("\t")
            for d in dpi_map[c[0]]:
                if d in nx.nodes(g) and c[1] in nx.nodes(g) and (g[d][c[1]]['weight'] != c[2] or g[d][c[1]]['direction'] != c[3]):
                    warning("An interaction was seen twice with varying confidence scores or direction for " + d + " and " + c[1] + ". Using the first score and direction.")
                    continue
                    # to keep everything straight we include what the node is (side effect, drug, or attribute) in the name
                g.add_edge(drug_prefix + d, a_prefix + c[1].replace("_", ""), weight = float(c[2]), direction=int(c[3]))
    if save:
        nx.write_gpickle(g, save_file)
    return g

def get_mb_graph(mb_filename, save, wsid, save_file='mb.pickle.gz', to_avoid_zero = 0.1):
    if nj.is_pickle(mb_filename):
        return nx.read_gpickle(mb_filename)
    drug_prefix, adr_prefix, a_prefix = establish_prefixes()
    g = nx.DiGraph()
    subMolDrugDoD, allDrugs = mol_bits_file_to_dd(mb_filename, wsid)
    for subMol in subMolDrugDoD.keys():
        for drug in subMolDrugDoD[subMol].keys():
            count = subMolDrugDoD[subMol][drug]
            if drug in nx.nodes(g) and subMol in nx.nodes(g) and (g[drug][submol]['weight'] != int(count)):
#                warning("An interaction was seen twice with varying confidence scores for " + drug + " and " + subMol + ". Using the first score.")
                continue
                # to keep everything straight we include what the node is (side effect, drug, or attribute) in the name
            g.add_edge(drug_prefix + drug, a_prefix + subMol.replace("_", ""), weight = 1 - 1/(float(count) + to_avoid_zero), direction=0)
    if save:
        nx.write_gpickle(g, save_file)
    return g

def mol_bits_file_to_dd(file, wsid):
    res = defaultdict(dict)
    all_d = []
    wsa_to_agent_id = {str(wsa.id) : str(wsa.agent_id) for wsa in WsAnnotation.objects.filter(ws=wsid)}
    with open(file, 'r') as f:
        header = None
        for l in f:
            if not header:
                header = l.strip().split(",")[1:] # we don't want thte first one, it's just a drug_id header
                continue
            fields = l.strip().split(",")
            wsa = fields.pop(0)
            try:
                drug = wsa_to_agent_id[wsa]
            except KeyError:
                continue
            all_d.append(drug)
            gen = (i for i in range(len(fields)) if int(rew.convertTypes(fields[i])) != 0)
            for i in gen:
                res[header[i]][drug] = rew.convertTypes(fields[i])
    return res, all_d

def add_in_ppi(ppi_file, g, save, save_file='ppi.pickle.gz', direction=True, min_ppi_evid = 0.0):
    drug_prefix, adr_prefix, a_prefix = establish_prefixes()
    h = nj.build_ppi_graph(ppi_file, direction=direction, min_ppi_evid = min_ppi_evid)
    # to get this to play nicely with our graph I'll add the atr prefix
    relabel_dict = {n: a_prefix + n for n in nx.nodes(h)}
    nx.relabel_nodes(h, relabel_dict, copy=False)
    toReturn = nx.compose(g,h)
    if save:
        nx.write_gpickle(toReturn, save_file)
    return toReturn

def get_adr_data(adr_file, wsid, save, save_file='drugAdr.pickle.gz', min_drugs_per_adr = None):
    drug_prefix, adr_prefix, a_prefix = establish_prefixes()
    g = nx.DiGraph()
    reweight = False
    with open(adr_file, 'r') as f:
        data_types = f.readline().rstrip().split("\t")
        for line in f:
            fields = line.rstrip().split("\t")
            drug = convert_to_wsa(fields[0], wsid, current_type = data_types[0])
            if not drug:
                continue
            db_node = drug_prefix + drug
            adr_node = adr_prefix + fields[1].upper()
            if g.has_edge(db_node, adr_node):
                warning("Duplicate values found for:", drug, "and", fields[1], ". Using first value for now.")
                continue
            if len(fields) > 2:
                g.add_edge(db_node, adr_node, weight = float(fields[2]))
                if float(fields[2]) > 1.0:
                    reweight = True
            else:
                g.add_edge(db_node, adr_node, weight = 1.0)
    if min_drugs_per_adr:
        to_rm = []
        gen = (node for node in g if node.startswith(adr_prefix))
        for adr in gen:
            all_nbrs = nj.get_list_of_neighbors(g, adr)
            drug_cnt = len(set([i for i in all_nbrs if i.startswith(drug_prefix)]))
            if drug_cnt < min_drugs_per_adr:
                to_rm.append(adr)
        g.remove_nodes_from(adr)
    if reweight:
        warning("ADR edge weight above 1 found. Rescoring the edge weights (w) as: 1 - 1/w")
        for node, nbrsDict in g.adjacency_iter():
            gen = ((n, ea) for n, ea in nbrsDict.items() if 'weight' in ea)
            for nbr, edgeAttr in gen:
                if edgeAttr['weight'] > 0:
                    edgeAttr['weight'] = 1.0 - 1.0/float(edgeAttr['weight'])
                else:
                     warning(node, nbr, "had a weight of", edgeAttr['weight'])
    if save:
        nx.write_gpickle(g, save_file)
    return g

def add_adrs_to_graph(g, wsid, adrs_to_try, adr_file, save, save_file='napa_final.pickle.gz', min_drugs_per_adr = None):
    drug_prefix, adr_prefix, a_prefix = establish_prefixes()
    # first read in all drugs with side effects
    verboseOut("Reading in side effect prevelance data...")
    if nj.is_pickle(adr_file):
        h = nx.read_gpickle(adr_file)
    else:
        h = get_adr_data(adr_file, wsid, save, min_drugs_per_adr = min_drugs_per_adr)
    if adrs_to_try is None:
        adrs_to_try = [n for n in nx.nodes(h) if n.startswith(adr_prefix)]
    toReturn = nx.compose(g,h)
    if save:
        nx.write_gpickle(toReturn, save_file)
    return toReturn, adrs_to_try

# might be able to replace this with shortest paths from networkx
def find_paths(g, adr, indirect):
    drug_prefix, adr_prefix, a_prefix = establish_prefixes()
    if adr_prefix + adr not in nx.nodes(g):
        verboseOut("warning:", adr, " not in the provided side effects.")
        return None
    verboseOut("Connecting ", adr, " to attributes...")
    paths = defaultdict(dict)
    adr_drug_gen = (d for d in nx.all_neighbors(g, adr_prefix + adr) if d.startswith(drug_prefix))
    for d in adr_drug_gen:
        d_adr_w = g.edge[d][adr_prefix + adr]['weight']
        d_atr_gen = (a for a in nx.all_neighbors(g, d) if a.startswith(a_prefix))
        for a in d_atr_gen:
            a_d_w = g.edge[d][a]['weight']
            a_d_d = g.edge[d][a]['direction']
            if indirect:
                a_atr_gen = (ia for ia in nx.all_neighbors(g,a) if ia.startswith(a_prefix))
                for ia in a_atr_gen:
                    a_a_w = g.edge[a][ia]['weight']
                    a_ia_d = g.edge[a][ia]['direction']
                    final_direc = a_ia_d * a_d_d
                    paths[ia + "_" + str(final_direc)][a + "_" + d] = a_a_w * a_d_w * d_adr_w
            else:
                paths[a + "_" + str(a_d_d)][d] = a_d_w * d_adr_w
    return paths

# kind of a mess, but I'm keeping a fair amount of info in the attribute name here, so pull all of that out
def process_atr_direction(ad):
    temp = ad.split("_") # split the prefix and the direction
    return temp[1], "_".join(temp[0:2]), temp[2]

# to prune we're going to use a fishers exact test, to test for the significance of enrichment
def prune_insignificant_paths(paths, g, indirect_flag, bg, min_q, min_paths, min_or):
    from dtk.enrichment import mult_hyp_correct
    import scipy.stats as stats
    # paths is a defaultdict(dict) keyed on attributes, then drugs.
    # If indirect_flag is true, the drug key is directattribute_drug, where the attribute is the final (indirect) attribute
    # now that we are inlucding direction, we consider a attribute that is up-, down- or unknown-direction-regulated by any drug to be a distinct, virtual attribute
    # For each attribute-adr pair we need four things:
        # 1. The number of drugs linked to this adr, or in the case of indirect the number of drug-attribute pairs - cause adr
        # 2. The number of drugs linked to that attribute, or in the case of indirect the number of drug-attribute pairs - taret atr
        # 3. The intersection of 1 and 2
        # 4. The number of drugs in the DPI file, or in the case of indirect the number of drug-attribute pairs
    # the resulting contingency table looks like (where, in the case of indirect, d is replaced by d_p and p is ip (indirect attribute))
    #                          cause_adr
    #                       yes         no
    # target_atr   yes    (1&2)     2 - (1&2)
    #               no    1-(1&2)    bg - (1 | 2)
    drug_prefix, adr_prefix, a_prefix = establish_prefixes()
    path_atrs = [] # this is an ugly way to do this, but it'll work
    path_ps = []
    path_ors = []
    # drugs to adr is easy
    d2s = set([drug for atr_direction in paths.keys() for drug in paths[atr_direction].keys()]) # 1
    for ad in paths.keys():
        # kind of a mess, but I'm keeping a fair amount of info in the attribute name here, so pull all of that out
        atr, a, direc = process_atr_direction(ad)
        # d2p is #2 above
        if indirect_flag:
            aai = (datr for datr in nx.all_neighbors(g, a) if datr.startswith(a_prefix))
            d2as = []
            for datr in aai:
                d2as.append([datr + "_" + drug for drug in nx.all_neighbors(g, datr) 
                               if (drug.startswith(drug_prefix) 
                               and g.edge[drug][datr]['direction'] * g.edge[a][datr]['direction'] == int(direc))]
                            )
            d2a = set([i for l in d2as for i in l])
        else:
            d2a = set([d for d in nx.all_neighbors(g, a) if (d.startswith(drug_prefix) and g.edge[d][a]['direction'] == int(direc))])
        intersect = d2s & d2a
        only_d2s = d2s - intersect
        only_d2a = d2a - intersect
        only_bg = (bg - d2s) - d2a
# rarely were there enough, so I just defaulted to fisher's exact
#        if len(intersect) < 5 or len(only_d2s) < 5 or len(only_d2a) < 5 or len(only_bg) < 5:
#            warning("Too few samples found for ", atr, ". Using fisher's Exact test.")
        odds_ratio, pvalue = stats.fisher_exact([[len(intersect), len(only_d2a)], [len(only_d2s), len(only_bg)]])
#        message(odds_ratio, pvalue)
#        else:
#            chi2, pvalue, dof, ex = stats.chi2_contingency([[len(intersect), len(only_d2a)], [len(only_d2s), len(only_bg)]])
#            message(chi2, pvalue, dof, ex)
        path_ps.append(pvalue)
        path_ors.append(odds_ratio)
        path_atrs.append(ad)
    # now multiply hypothesis correct
    if len(path_atrs) > 1:
        path_qs = mult_hyp_correct(path_ps)
    else:
        path_qs = path_ps
    #
    pruned = defaultdict(dict)
    q = {}
    ors = {}
    cond_gen = (i for i in range(len(path_atrs)) 
                  if path_ors[i] > min_or 
                    and path_qs[i] <= min_q 
                    and len(paths[path_atrs[i]]) >= min_paths
                )
    for i in cond_gen:
        pruned[path_atrs[i]] = paths[path_atrs[i]]
        q[path_atrs[i].lstrip(a_prefix)] = 1.0 - path_qs[i]
        ors[path_atrs[i].lstrip(a_prefix)] = 1.0 -  1.0/path_ors[i]
    return pruned, ors, q

def get_background(g, indirect_flag):
    if indirect_flag:
        bg = set([d + '_' + a for a in nx.nodes(g) if a.startswith(a_prefix) for d in nx.all_neighbors(g, a) if d.startswith(drug_prefix)])
    else:
        bg = set([d for d in nx.nodes(g) if d.startswith(drug_prefix)])
    return bg

def calc_sums(paths, indirect, to_avoid_zero = 0.5):
    import statistics as stats
    drug_prefix, adr_prefix, a_prefix = establish_prefixes()
    wt_maxs = {}
    wt_meds = {}
    cnts = {}
    for ad in paths.keys():
        atr, a, direc = process_atr_direction(ad)
        cnt = 1.0 - 1.0/(len(list(paths[ad].values())) + to_avoid_zero)
        wts = [float(w) for w in paths[ad].values()]
        wt_maxs[atr + "_" + direc] = max(wts)
        wt_meds[atr + "_" + direc] = stats.median(wts)
        cnts[atr + "_" + direc] = cnt
    return wt_maxs, wt_meds, cnts

def get_out_filename(ofp, to_report):
    if to_report == 'paths' or to_report == 'ft_list':
        return ofp + "_" + to_report + ".tsv"
    elif to_report != 'stats':
        warning("Unrecognized to_report argument:", to_report, ". Defaulting to sums.")
    return ofp + "_path_stats.tsv"

def clear_outfile(ofp, to_report, attribute_name='attr'):
    outfile = get_out_filename(ofp, to_report)
    header = get_header(to_report)
    with open(outfile, 'w') as f:
        f.write("\t".join(header) + "\n")
    return True

def get_header(to_report):
    if to_report == 'paths':
        return ['ADR', 'uniprot_id', 'drug_id', 'score']
    elif to_report == 'ft_list':
        return ['drug_id', 'ADR', 'feature_list']
    else:
        return ['drug_id', 'ADR', 'max_wt', 'median_wt', 'path_counts', 'OR', 'q']

def write_results(adr, to_report, sd, ofp):
    if to_report == 'paths':
        outfile = get_out_filename(ofp, to_report)
        with open(outfile, 'a') as f:
            f.write("\n".join(["\t".join([adr, p, d, str(v)]) for p, sd2 in sd['paths'].items() for d, v in sd2.items()]) + "\n")
        return True
    outfile = get_out_filename(ofp, to_report)
    with open(outfile, 'a') as f:
        for p,v in sd['wt_max'].items():
            f.write("\t".join([adr, p, str(sd['wt_max'][p]), str(sd['wt_med'][p]), str(sd['path_counts'][p]), str(sd['OR'][p]), str(sd['q'][p]) ]) +"\n")

if __name__=='__main__':
    import argparse    
    #=================================================
    # Read in the arguments/define options
    #=================================================
    exitCoder = ExitCoder()
    
    arguments = argparse.ArgumentParser(description="Connect Side effects to attributes, via drugs")
    
    arguments.add_argument("--adr", help="adr prevalance for every drug. A TSV where each row is a drug, and each column is a adr prevalance.")
    
    arguments.add_argument("--d_a_type", help="What type of drug attibute are you connecting to ADRs: protein, molec_bits, structure_alerts")
    
    arguments.add_argument("--d_a_file", help="Input file to use, e.g. a DPI file")
    
    arguments.add_argument("--pre_saved", help="Final graph from a previous run. Providing this will result in only the calculations being repeated")
    
    arguments.add_argument("--adr_to_check", help="File of UMLS codes for adrs to connect to attributes")
    
    arguments.add_argument("-o", help="Output file prefix")
    
    arguments.add_argument("--indirect", action="store_true", help="Only applicable when proteins are being connected to ADRs. Include indrect targets (i.e one protein-protein interaction away). Must also provide PPI file.")

    arguments.add_argument("--ppi", help="PPI file to use. Only necessary if --indirect is provided, otherwise ignored")

    arguments.add_argument("--min_or", type = float, default=1, help="Minimum Odds ratio (fold enrichment) for the attribute ADR pair to report the path (default %(default)s)")    
    
    arguments.add_argument("--min_q", type = float, default=0.15, help="Minimum Fisher's exact BH-correct p-value for the attribute ADR pair to report the path (default %(default)s)")    
    
    arguments.add_argument("--min_drugs_per_adr", type = int, default=10, help="Minimum number of drugs which have an ADR for the ADR to be included (default %(default)s)")
    
    arguments.add_argument("--min_paths", type = int, default=1, help="Minimum number of attribute-ADR paths required to report the connection (default %(default)s)")
    
    arguments.add_argument("--to_report", default='stats', help="What to report for each adr: stats (all path stats for all attributes), paths (each path from ADR to attribute) (default %(default)s)")
    
    arguments.add_argument("--wsid", type=int, default=49, help="Workspace ID. default %(default)s")
    
    arguments.add_argument("--pickle", action="store_true", help="Pickle all graphs")
    
    arguments.add_argument("--verbose", action="store_true", help="Print out status reports")
            
    args = arguments.parse_args()
    
    # return usage information if no argvs given
    
    if not args.wsid or not args.d_a_type or (not args.pre_saved and (not args.d_a_file and not args.adr)) or not args.adr_to_check or not args.o or (args.indirect and (not args.ppi and not args.pre_saved)):
        arguments.print_help()
        sys.exit(exitCoder.encode('usageError'))
    
    #======================================================================
    # main
    #======================================================================
    clear_outfile(args.o, args.to_report) # later we append to this file, so I want to make sure it's empty/gone

    drug_prefix, adr_prefix, a_prefix = establish_prefixes()
    
    if args.adr_to_check != 'all':
        adrs_to_try = [line.strip().upper() for line in open(args.adr_to_check, 'r')]
    else:
        adrs_to_try = None
    if args.pre_saved:
        g = nx.read_gpickle(args.pre_saved)
        if adrs_to_try is None:
            adrs_to_try = [n for n in nx.nodes(g) if n.startswith(adr_prefix)]
    else:
        verboseOut("Reading in drug-", args.d_a_type, "data...")
        g = get_da_graph(args.d_a_type, args.d_a_file, args.wsid, args.pickle, args.o)
        
        final_save_file_prefix = args.o + "_NAPA_graph"
        if args.d_a_type == 'protein' and args.indirect:
            g = add_in_ppi(args.ppi, g, args.pickle, save_file = args.o + '_ppi.pickle.gz')
            final_save_file_prefix = final_save_file_prefix + "_indirect"
        
        verboseOut("Adding adrs to graph...")
        g, adrs_to_try = add_adrs_to_graph(g, args.wsid, adrs_to_try, args.adr, args.pickle, save_file = final_save_file_prefix + '_final.pickle.gz', min_drugs_per_adr = args.min_drugs_per_adr)
    
    # for testing purposes, I want to report all those drugs which are tied to at least one side effect
#    adrDs = []
#    alladrs = [i for i in nx.nodes(g) if i.startswith(adr_prefix)]
#    for adr in alladrs:
#        adrDs.append(set([d for d in nx.all_neighbors(g, adr) if d.startswith(drug_prefix)]))
#    with open('allRelevantDrugBanks_forTestingSA.txt', 'w') as f:
#        f.write("\n".join(list(set([i.lstrip(drug_prefix) for s in adrDs for i in s]))) +"\n")
    
    background = get_background(g, args.indirect)
    for adr in adrs_to_try:
        adr_atrs = {}
        adr = adr.lstrip(adr_prefix)
        adr_atrs['paths'] = find_paths(g, adr, args.indirect)
        if adr_atrs['paths'] is not None:
            adr_atrs['paths'], adr_atrs['OR'], adr_atrs['q'] = prune_insignificant_paths(adr_atrs['paths'], g, args.indirect, background, args.min_q, args.min_paths, args.min_or)
            # now collapadr the attributes, regardless of connecting drug
            adr_atrs['wt_max'], adr_atrs['wt_med'], adr_atrs['path_counts'] = calc_sums(adr_atrs['paths'], args.indirect)
            write_results(adr, args.to_report, adr_atrs, args.o)
    
    verboseOut("Finished successfully")
