#!/usr/bin/env python3

import sys
from path_helper import PathHelper

import os

from algorithms.exit_codes import ExitCoder
from dtk.prot_map import DpiMapping,PpiMapping

import networkx as nx
from dtk.files import get_file_records
from statistics import median
import time
from collections import defaultdict

debugging = False
ndrugs = None

class d2ps:
    def __init__(self,**kwargs):
        self.cores = kwargs.get('cores',None)
        self.dpi_file = kwargs.get('dpi',None)
        self.dpi_t = kwargs.get('dpi_t', DpiMapping.default_evidence)
        self.ppi_file = kwargs.get('ppi',None)
        self.ppi_t = kwargs.get('ppi_t', PpiMapping.default_evidence)
        self.gmt = kwargs.get('gmt',None)
        self.ind_ofile = kwargs.get('out',None)
        self.restart_prob = kwargs.get('restart_prob', 0.85)
        self.iters = kwargs.get('iterations', 100)
        self.methods = kwargs.get('methods',['enrich'])
        self._set_method_flags()
        self.drug_prefix, _, self.prot_prefix = establish_prefixes()
        self.ps_prefix = 'ps_'
        self.dpi_graph = kwargs.get('dpi_graph', None)
    def _set_method_flags(self):
        self.pr = 'protrank' in self.methods
        enr_methods = {'enrich', 'enrich_score', 'enrich_score_v2'}
        self.enr = bool(enr_methods & set(self.methods))
        self.strt = 'straight' in self.methods
    def run(self):
        self.setup_graphs()
        self.drug2ps()
        self.aggregate_scores()
        self.report_paths()
    def setup_graphs(self):
        if self.dpi_graph is not None:
            self.g = self.dpi_graph
        else:
            self.g = build_keyed_dpi_graph(self.dpi_file)
        self.add_ppi()
        self.add_ps()
        self.ps_nodes = [n for n in self.g if n.startswith(self.ps_prefix)]
        self.drug_nodes = [n for n in self.g if n.startswith(self.drug_prefix)]
    def add_ps(self, direction = True):
        if direction:
            new_graph = nx.DiGraph()
        else:
            new_graph = nx.Graph()
        for c in get_file_records(self.gmt, parse_type='tsv', keep_header=True):
            prots = c[1].split(",")
            for p in prots:
                new_graph.add_edge(self.ps_prefix + c[0], self.prot_prefix + p, weight = 1.0)
        self.g = nx.compose(self.g,new_graph)
    def add_ppi(self):
        from dtk.prot_map import PpiMapping
        ppi = PpiMapping(self.ppi_file)
        self.ppi_graph = build_ppi_graph(ppi,
                                         prot_prefix = self.prot_prefix,
                                         direction = True,
                                         min_ppi_evid = self.ppi_t
                                        )
        self.g = nx.compose(self.g,self.ppi_graph)

    def drug2ps(self):
        if not ndrugs:
            self.drug_nodes_to_use = self.drug_nodes
        else:
            self.drug_nodes_to_use = self.drug_nodes[0:ndrugs]
        self.n_iters = len(self.drug_nodes_to_use)
        if self.enr:
            ts = time.time()
            self._enrich()
            print("self._enrich took: " + str(time.time()-ts))
        if self.pr:
            ts = time.time()
            self._protrank()
            print("self._protrank took: " + str(time.time()-ts))
        if self.strt:
            ts = time.time()
            self._straight()
            print("self._straight took: " + str(time.time()-ts))
    def _straight(self):
        self.s_results={}
        for d in self.drug_nodes_to_use:
            target_nodes = [n
                             for n in nx.all_neighbors(self.g, d)
                             if n.startswith(self.prot_prefix)
                            ]
            self.s_results[d] = set((ps for n in target_nodes
                                      for ps in nx.all_neighbors(self.g, n)
                                      if ps.startswith(self.ps_prefix)
                                     ))
    def _protrank(self):
        self.target_nodes = {}
        self.drug_results = {}
# plat2144
### We do an iteration for each drug
### it takes longer to transport that data than it does to run the processing
### preferable would be a longer running process
### A completely different fix would be to avoid running this for drugs with identical DPI (just look it up from a previous run)
        params = zip(self.drug_nodes_to_use,
                     [self.g] * self.n_iters,
                     [self.ppi_graph] * self.n_iters,
                     [self.prot_prefix] * self.n_iters,
                     [self.ps_nodes] * self.n_iters,
                     [self.restart_prob] * self.n_iters,
                     [self.iters] * self.n_iters,
                     [True] * self.n_iters
                    )
        if debugging:
            pr_results = map(prot_rank, params)
        else:
            import multiprocessing
            pool = multiprocessing.Pool(self.cores)
            pr_results = pool.map(prot_rank,params)
        self.pr_results = {tup[0]:tup[1] for tup in pr_results}
    def _enrich(self):
        self.prot_set_connections()
        self.background()
        ps_ps = {self.ps_prefix+k:v for k,v in self.ps_indPs.items()}
        static_params = dict(
                prot_prefix=self.prot_prefix,
                drug_prefix=self.drug_prefix,
                ps_prefix=self.ps_prefix,
                g=self.g,
                ps_ps=ps_ps,
                bg_len=len(self.full_bg),
                min_q=0.01,
                base_node=None
                )
        from dtk.parallel import pmap
        out = pmap(
                calc_d_ps_paths,
                self.drug_nodes_to_use,
                static_args=static_params,
                num_cores=self.cores,
                fake_mp=debugging,
                )
        import tqdm
        results = tqdm.tqdm(out, total=self.n_iters, smoothing=0.1)
        self.e_results = {tup[0]:tup[1] for tup in results if tup[1] is not None}
        for tup in results:
            if tup[1] is not None:
                self.e_results[tup[0]] = tup[1]
    def prot_set_connections(self):
        self.ps_dirPs = {}
        self.ps_indPs = {}
        for psNode in self.ps_nodes:
            ps = psNode.lstrip(self.ps_prefix)
            self.ps_dirPs[ps] = set([n.lstrip(self.prot_prefix)
                                      for n in nx.all_neighbors(self.g, psNode)
                                      if n.startswith(self.prot_prefix)
                                     ])
# the order_and_combine_prots is that we don't have to worry
# about 1_2 being interpretted differently than 2_1
            self.ps_indPs[ps] = set([order_and_combine_prots(n.lstrip(self.prot_prefix),p)
                                      for p in self.ps_dirPs[ps]
                                      for n in nx.all_neighbors(self.g, self.prot_prefix+p)
                                      if n.startswith(self.prot_prefix)
                                    ])
    def background(self):
        self.dir_bg = set([n.lstrip(self.prot_prefix)
                            for n in self.g
                            if n.startswith(self.prot_prefix)
                          ])
        self.ind_bg = set([order_and_combine_prots(n.lstrip(self.prot_prefix), p)
                             for p in self.dir_bg
                             for n in nx.all_neighbors(self.g, self.prot_prefix + p)
                             if n.startswith(self.prot_prefix)
                          ])
        self.full_bg = self.dir_bg | self.ind_bg
    def aggregate_scores(self):
        self.drug_nodes_to_report = set()
        self.drug_nodes_to_report.update(list(self.pr_results.keys()) if self.pr else set())
        self.drug_nodes_to_report.update(list(self.e_results.keys()) if self.enr else set())
        self.drug_nodes_to_report.update(list(self.s_results.keys()) if self.strt else set())
    def report_paths(self):
        import gzip
        import numpy as np
        with gzip.open(self.ind_ofile + ".tsv.gz", 'wt') as f:
            score_orders = []
            pr_scores = []
            e_scores = []
            if self.strt:
                s_scores = ['directly_connected']
                score_orders += s_scores
            if self.pr:
                pr_scores = ['protrank_median', 'protrank_max', 'protrank_mean']
                score_orders += pr_scores
            if self.enr:
                e_scores = ['enrich_score', 'enrich_dir']
                score_orders += e_scores
            f.write("\t".join(['drug_id', 'pathway'] +
                              score_orders ) +
                    "\n")
            for dn in self.drug_nodes_to_report:
                for pn in self.ps_nodes:
                    scores = []
                    if self.strt:
                        if dn in self.s_results and pn in self.s_results[dn]:
                            scores += [1]
                        else:
                            scores += [0]
                    if self.pr:
                        if dn in self.pr_results and pn in self.pr_results[dn]:
                            d = self.pr_results[dn][pn]
                            for s in pr_scores:
                                scores.append(d[s])
                        else:
                            scores += [0.0]*len(pr_scores)
                    if self.enr:
                        if dn in self.e_results and pn in self.e_results[dn]:
                            starter = 1.0
                            for s in self.e_results[dn][pn]['scores']:
                                starter *= 1.0 - s
                            scores.append(1.0 - starter)
                            scores.append(np.sign(sum(self.e_results[dn][pn]['direcs'])))
                        else:
                            scores += [0.0] * len(e_scores)
                    import numpy as np
                    if 'enrich_score_v2' in self.methods:
                        nonzero = lambda x: any(np.array(x) > 0)
                    else:
                        # Reproduce old bug for a while for target importance.
                        nonzero = lambda x: np.sum(x) > 0
                    if scores and nonzero(scores):
                        out = [dn.lstrip(self.drug_prefix), pn.lstrip(self.ps_prefix)]
                        out += [str(x) for x in scores]
                        f.write("\t".join(out) + "\n")

def prot_rank(params):
    ref_node, g, ppi_g, prot_prefix, end_nodes, restart_prob, iters, use_en_wt  = params
# plat2144
### In run_page_rank it does somethign special when there are no targets, that should be assessed before we even get here (i.e. before this funciton is called)
### That will avoid running this expensive setup for a quick running process
### same for keyerrors?
    protrank_dict = run_page_rank([ref_node, g, ppi_g, prot_prefix, restart_prob, iters])
    en_scores = score_prot_rank(end_nodes, g, prot_prefix, use_en_wt, protrank_dict)
    return (ref_node, en_scores)

def score_prot_rank(end_nodes, g, prot_prefix, use_en_wt, protrank_dict):
    en_scores = {}
    for en in end_nodes:
        scores = []
        gen = (n for n in nx.all_neighbors(g,en) if n.startswith(prot_prefix))
        for n in gen:
            wt = 1.0
            try:
                if use_en_wt:
                    wt = g[en][n]['weight']
                scores.append(protrank_dict[n] * wt)
            except KeyError:
                scores.append(0.0)
        en_scores[en] = {'protrank_median': median(scores),
                         'protrank_mean': float(sum(scores))/len(scores),
                         'protrank_max': max(scores),
                        }
    return en_scores


def run_page_rank(params):
    """Legacy, just calls through to the below."""
    return run_page_rank2(*params)

def run_page_rank2(ref_node, g, ppi_g, prot_prefix, restart_prob, iterations):
    target_nodes = set([n
                         for n in nx.all_neighbors(g, ref_node)
                         if n.startswith(prot_prefix)
                        ])
    person_dict = {n:(g[ref_node][n]['weight']
                     if n in target_nodes
                     else 0.0)
                   for n in ppi_g
                  }
    # basically if there are no targets, don't bother
    if float(sum(person_dict.values())) == 0.0:
        protrank_dict = dict.fromkeys(ppi_g.nodes(), 0.0)
    else:
### This takes basically all of the time for the entire method
# scipy speeds things up ~7x, so the above should be reassessed
# note that the very small scores are slightly different with this different method, but only when looking 5-6 digits past the decimal point
        protrank_dict = nx.pagerank_scipy(ppi_g,
                alpha = float(restart_prob), # Chance of randomly restarting, 0.85 is default
                personalization = person_dict, # dictionary with a key for every graph node and nonzero personalization value for each node
                max_iter = iterations, # Maximum number of iterations in power method eigenvalue solver. default 100
                tol = 1e-6, # Error tolerance used to check convergence in power method solver. default 1e-6
                weight = 'weight', # Edge data key to use as weight. If None weights are set to 1
               )
    return protrank_dict

def build_ppi_graph(ppi_map, prot_prefix, direction=True, min_ppi_evid = 0.0):
    if direction:
        new_graph = nx.DiGraph()
    else:
        new_graph = nx.Graph()
    for c in ppi_map.get_data_records(min_evid=min_ppi_evid):
        if float(c[2]) < min_ppi_evid or c[0] == "-" or c[1] == "-":
            continue
        if len(c) > 3:
            direc = int(c[3])
        else:
            direc = 0
        new_graph.add_edge(prot_prefix+c[0],
                           prot_prefix+c[1],
                           weight = float(c[2]),
                           direction = direc
                           )
    return new_graph

def build_keyed_dpi_graph(dpi_filename, min_dpi = 0.5, direction=True, filter_keys=None):
    if filter_keys is not None:
        filter_keys = set(filter_keys)
    drug_prefix, _, prot_prefix = establish_prefixes()
    if direction:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    from dtk.prot_map import DpiMapping
    dpi_obj = DpiMapping(dpi_filename)
    for c in get_file_records(dpi_obj.get_path(), parse_type = "tsv", keep_header = False):
        if float(c[2]) < min_dpi:
            continue
        if filter_keys is not None and c[0] not in filter_keys:
            continue
        if (g.has_node(c[0]) and
            g.has_node(c[1]) and
            (g[c[0]][c[1]]['weight'] != c[2] or
             g[c[0]][c[1]]['direction'] != c[3]
            )
           ):
            sys.stderr.write("WARNING: An interaction was seen twice with varying confidence scores or direction for " + c[0] + " and " + c[1] + "\n")
            continue
            # to keep everything straight we include what the node is (side effect, drug, or attribute) in the name
        g.add_edge(drug_prefix + c[0],
                   prot_prefix + c[1].replace("_", ""),
                   weight = float(c[2]),
                   direction = int(c[3])
                  )
    return g



# this is almost identical to the one in the original connect script, except the keys are now with a prefix
def calc_d_ps_paths(dn, prot_prefix, drug_prefix, ps_prefix, g, ps_ps, bg_len, min_q, base_node):
    from functools import lru_cache
    @lru_cache(maxsize=2**16)
    def all_prefixed_neighbors(node, prefix):
        out = list(n for n in nx.all_neighbors(g, node) if n.startswith(prefix))
        return out
    indpaths = defaultdict(dict)
    inddirs = defaultdict(dict)
    if base_node:
        neighbor_prots_gen = (n for n in set(nx.all_neighbors(g, dn)) | set(nx.all_neighbors(g, base_node))
                               if n.startswith(prot_prefix)
                             )
    else:
        neighbor_prots_gen = (n for n in nx.all_neighbors(g, dn) if n.startswith(prot_prefix))
    p2d = set()
    for pn in neighbor_prots_gen:
        prot = pn.lstrip(prot_prefix)
        if base_node:
            try:
                base_wt = float(g.adj[base_node][pn]['weight'])
                base_dir = float(g.adj[base_node][pn]['direction'])
            except KeyError:
                base_wt = 0.0
                base_dir = 0
            try:
                d_wt = float(g.adj[dn][pn]['weight'])
                d_dir = float(g.adj[dn][pn]['direction'])
            except KeyError:
                d_wt = 0.0
                d_dir = 0
            dp_wt = min([base_wt + d_wt, 1.0])
            dp_dir = combine_dirs(d_dir, base_dir)
        else:
            dp_wt = float(g.adj[dn][pn]['weight'])
            dp_dir = float(g.adj[dn][pn]['direction'])
        # First we had the direct paths that go through a single protein
        dir_ps_gen = (n for n in nx.all_neighbors(g, pn) if n.startswith(ps_prefix))
        for ps in dir_ps_gen:
            indpaths[ps][pn] = dp_wt
            inddirs[ps][pn] = dp_dir
        p2d.add(pn)
### XXX Need to make sure the background also has the single prot paths
        ### Now add in the 2 steps
        ip_gen = (n for n in nx.all_neighbors(g, pn) if n.startswith(prot_prefix))
        for ipn in ip_gen:
            try:
                ind_dpp_wt = float(dp_wt * g.adj[ipn][pn]['weight']) # mult was much better than average
                ind_dpp_dir = combine_dirs(dp_dir, g.adj[ipn][pn]['direction'])
            except KeyError:
                continue
            pp_key = order_and_combine_prots(prot, ipn.lstrip(prot_prefix))
            final_gen = all_prefixed_neighbors(ipn, ps_prefix)
            for ps in final_gen:
# no need to take the direction or weight b/t proteins and protein sets
                indpaths[ps][pp_key] = ind_dpp_wt
                inddirs[ps][pp_key] = ind_dpp_dir
            p2d.add(pp_key)
    ts = time.time()
    results = calc_path_stats(ps_ps, prot_prefix, indpaths, inddirs, dn, bg_len, min_q, p2d)
    return (dn, results, time.time() - ts)

from functools import lru_cache
@lru_cache(maxsize=2**18)
def cached_fisher_pvalue(a, b, c, d):
    # this was faster than stats.fisher_exact
    from fisher import pvalue
    return pvalue(a, b, c, d).right_tail

def calc_path_stats(ps_ps, prot_prefix, indpath_scores, indpath_dirs, drug_node, bg_len, min_q, p2d):
    #from math import log
    # the first thing we're going to try is to assess the enrichment (if it's there) of the drug-protset connection
    # For that we'll need:
    # 1. The number of prots linked to this drug, or in the case of indirect the number of prot-prot pairs
    # 2. The number of prots linked to that protset, or in the case of indirect the number of prot-prot pairs
    # 3. The intersection of 1 and 2
    # 4. The number of prots in the graph, or in the case of indirect the number of prot-prot pairs
    ps_p = {}
    ps_dir = defaultdict(list)
    ps_scor = defaultdict(list)
    for ps in indpath_scores:
        just_ps = ps_ps[ps]
        jps_len = len(just_ps)
        p2dAndPs = just_ps & p2d
        only_p2d = p2d - p2dAndPs
        a = len(p2dAndPs)
        b = jps_len - a
        c = len(only_p2d)
        d = bg_len-(jps_len+c)
        ps_p[ps] = cached_fisher_pvalue(a,b,c,d)
    if len(list(indpath_scores.keys())) == 1 and list(ps_p.values())[0] <= min_q:
        return {ps: {'scores': list(list(indpath_scores.values())[0].values()),
                     'direcs': list(list(indpath_dirs.values())[0].values())
                    }
               }
    elif len(list(indpath_scores.keys())) > 1:
        try:
            from dtk.enrichment import mult_hyp_correct
            ps_q = dict(zip(list(ps_p.keys()), mult_hyp_correct(list(ps_p.values()))))
        except TypeError:
            print("Prot set error with " + ps_p)
            sys.exit(1)
        return {ps : {'scores': list(indpath_scores[ps].values()),
                       'direcs': list(indpath_dirs[ps].values())
                     }
                 for ps in indpath_scores.keys()
                 if ps_q[ps] <= min_q
               }
    return None


def establish_prefixes():
    return "d_", "adr_", "a_"

def order_and_combine_prots(n1, n2):
    return "_".join(sorted([n1, n2]))

def combine_dirs(dir1, dir2):
    import numpy as np
    return np.sign(dir1 + dir2)

if __name__=='__main__':

    import argparse
    #=================================================
    # Read in the arguments/define options
    #=================================================
    exitCoder = ExitCoder()

    arguments = argparse.ArgumentParser(description="Connect drugs to proteinSets (e.g. pathways")

    arguments.add_argument("--dpi", help="dpi name.")

    arguments.add_argument("--ppi", help="ppi name.")

    arguments.add_argument("--gmt",
                           default='reactome',
                           help="Pathway file name. default: %s(default)"
                          )

    arguments.add_argument("-o", help="Output file prefix. Default is dpi + ppi + gmt basename")

    arguments.add_argument("--min_dpi_evid",
                           type = float,
                           default = DpiMapping.default_evidence,
                           help="Minimum DPI evidence to consider the interaction. default: %f(default)"
                          )

    arguments.add_argument("--min_ppi_evid",
                           type = float,
                           default = PpiMapping.default_evidence,
                           help="Minimum PPI evidence to consider the interaction. default: %f(default)"
                          )

    arguments.add_argument("--restart_prob",
                           type = float,
                           default = 0.85,
                           help="In PageRank walk, the probablility of restarting. Smaller values will diffuse the scores further afield default: %f(default)"
                          )

    arguments.add_argument("--cores", type = int, default=None, help="Number of cores available")
    arguments.add_argument("--no_protrank", action='store_true', help="exclude protrank")
    arguments.add_argument("--no_enrich", action='store_true', help="exclude enrichment")

    args = arguments.parse_args()

    ts = time.time()
    from dtk.s3_cache import S3File
    s3f=S3File('glee',
            'annotated.pathways_%s.uniprot.gmt'%".".join([args.gmt,args.gmt])
            )
    s3f.fetch()
    if not args.o:
        args.o = "./"+ "_".join([args.dpi, args.ppi, args.gmt])
#    methods = []
#    if not args.no_protrank:
#        methods.append('protrank')
#    if not args.no_enrich:
#        methods.append('enrich')
#    if not methods:
#        print "Must use at least one of the possible methods"
#        sys.exit(1)
    run = d2ps(
               dpi = args.dpi,
               ppi = args.ppi,
               dpi_t = args.min_dpi_evid,
               ppi_t = args.min_ppi_evid,
               out = args.o,
               cores = args.cores,
               gmt = s3f.path(),
               restart_prob = args.restart_prob,
#               methods = methods
             )
    run.run()
    print("whole run took: " + str(time.time()-ts))
