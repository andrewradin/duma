#!/usr/bin/env python3

from __future__ import print_function
from builtins import range
import sys
# retrieve the path where this program is located, and
# use it to construct the path of the website_root directory
try:
    from path_helper import PathHelper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")
    from path_helper import PathHelper

import os

from algorithms.exit_codes import ExitCoder

# comd specific imports
import networkx as nx
try:
    import napa_build as nb
except ImportError:
    sys.path.insert(1,os.path.join(PathHelper.website_root, "ML"))
    sys.path.insert(1,os.path.join(PathHelper.website_root, "tox", "padre"))
    sys.path.insert(1,os.path.join(PathHelper.website_root, "scripts", "network_similarity"))
    import napa_build as nb
import time

class hndist:
    def __init__(self,**kwargs):
        self.cores = kwargs.get('cores',None)
        self.ws = kwargs.get('ws',None)
        self.g_pkld = kwargs.get('g_pkld',None)
        self.refset_file = kwargs.get('refset_file',None)
        self.reportset_file = kwargs.get('reportset_file',None)
        self.outdir = kwargs.get('outdir',None)
        self.tmp_pubdir = kwargs.get('tmp_pubdir',None)
        self.no_norm = kwargs.get('no_norm',None)
        # output files
        self.drug_score_file = self.outdir + "drug_distance_scores.tsv"
        self.prot_score_file = self.outdir + "prot_distance_scores.tsv"
        # published output files
        self.reportset_out = self.tmp_pubdir + 'reportset_stats.txt' 
    def run(self):
        ts = time.time()
        self.get_graph()
        print("self.get_graph took: " + str(time.time() - ts))
        ts = time.time()
        self.load_ref_set()
        print("self.load_ref_set took: " + str(time.time() - ts))
        ts = time.time()
        self.calc_dists()
        print("self.calc_dists took: " + str(time.time() - ts))
        ts = time.time()
        self.report_dists()
        print("self.report_dists took: " + str(time.time() - ts))
        ts = time.time()
        self.check_reportset()
        print("self.check_reportset took: " + str(time.time() - ts))
    def get_graph(self):
        self.g = nx.read_gpickle(self.g_pkld)
        self.drug_prefix, _, self.prot_prefix = nb.establish_prefixes()
        self.all_relevant_nodes = [n
                                   for n in self.g.nodes()
                                   if n.startswith(self.drug_prefix)
                                   or n.startswith(self.prot_prefix)
                                  ]
    def check_reportset(self):
        if self.reportset_file != 'None' and not os.path.isdir(self.reportset_file):
            self.load_report_set()
            self.calc_reportset_stats()
            self.report_reportset_stats()
    def report_reportset_stats(self):
        with open(self.reportset_out, 'w') as f:
            if self.no_norm:
                f.write("\t".join(['Set name', 'Median shortest path', 'Median distance FDR', 'Min shortest path', 'Min distance FDR']) + "\n")
            else:
                f.write("\t".join(['Set name', 'Log2 median distance', 'Median distance FDR', 'Log2 min distance', 'Min distance FDR']) + "\n")
            for tup in self.report_stats:
                f.write("\t".join([str(i) for i in tup])+ "\n")
    def calc_reportset_stats(self):
        from scipy.stats import mannwhitneyu
        from dtk.enrichment import mult_hyp_correct
        import operator
        from math import log
        import numpy as np
        ps = {}
        FCs = {}
        ts2 = time.time()
        for name,l in self.report_groups.items():
            medians = []
            mins = []
            for i in l:
                try:
                    medians.append(self.median_dist[self.prot_prefix + i])
                    mins.append(self.min_dist[self.prot_prefix + i])
                except KeyError:
                    pass
            prefix = self.prot_prefix
# I was trying to do this to make the report set also be drugs, but I don't currently need it to, 
# and it's causing more trouble than it's worth
#            except KeyError:
#                medians = [self.median_dist[self.drug_prefix + i] for i in l]
#                mins = [self.min_dist[self.drug_prefix + i] for i in l]
#                prefix = self.drug_prefix
            other_medians = []
            other_mins = []
            gen = (k for k in self.median_dist.keys() if k.startswith(prefix))
            for k in gen:
                n = k.lstrip(prefix)
                if n not in l:
                    other_medians.append(self.median_dist[k])
                    other_mins.append(self.min_dist[k])
            ps[name] = {'min': mannwhitneyu(mins, other_mins)[1]* 2.0, # to make it two tailed
                        'median': mannwhitneyu(medians, other_medians)[1]* 2.0
                       }
            if self.no_norm:
                FCs[name] = {'min' : float( np.median(np.array(mins))),
                             'median': float( np.median(np.array(medians)))
                            }
            else:
                FCs[name] = {'min' : log(float( np.median(np.array(mins)) / np.median(np.array(other_mins))),2),
                             'median': log(float( np.median(np.array(medians)) / np.median(np.array(other_medians))),2)
                            }
        print("main for loop took: " + str(time.time()-ts2))
        if len(list(ps.keys())) > 1:
            min_qs = mult_hyp_correct([ps[k]['min'] for k in ps.keys()])
            median_qs = mult_hyp_correct([ps[k]['median'] for k in ps.keys()])
        else:
            min_qs = list(ps.values())[0]['min']
            median_qs = list(ps.values())[0]['median']
        results = []
        for i in range(len(min_qs)):
            name = list(ps.keys())[i]
            sorter = np.sign(FCs[name]['median']) * (1-median_qs[i])
#            if FCs[name]['median'] > 0:
#                sorter = median_qs[i]
            results.append((sorter, (name, FCs[name]['median'], median_qs[i], FCs[name]['min'], min_qs[i])))
        results.sort(key=operator.itemgetter(0))
        self.report_stats = [tup[1] for tup in results]
    def load_report_set(self):
        if not os.path.isfile(self.reportset_file):
            from dtk.s3_cache import S3Bucket, S3File
            f=S3File(S3Bucket('glee'),os.path.basename(self.reportset_file))
            f.fetch()
        self.report_groups = {}
        with open(self.reportset_file, 'r') as f:
            for l in f:
                fields = l.rstrip("\n").split("\t")
                self.report_groups[fields[0]] = fields[1].split(",")
    def load_ref_set(self):
        with open(self.refset_file, 'r') as f:
            l = f.readline().strip()
            self.ref_set = l.split("\t")
    def report_dists(self):
        with open(self.drug_score_file, 'w') as f:
            f.write("\t".join(['wsa_id'
                               , 'medianDist'
                               , 'minDist'
                              ])
                     + "\n"
                   )
            gen = (d for d in self.all_relevant_nodes if d.startswith(self.drug_prefix))
            for dn in gen:
                d = dn.lstrip(self.drug_prefix)
                f.write("\t".join([str(d)
                                  , str(self.median_dist[dn])
                                  , str(self.min_dist[dn])
                                 ])
                          + "\n"
                         )
        with open(self.prot_score_file, 'w') as f:
            f.write("\t".join(['uniprot'
                               , 'medianDist'
                               , 'minDist'
                              ])
                     + "\n"
                   )
            gen = (p for p in self.all_relevant_nodes if p.startswith(self.prot_prefix))
            for pn in gen:
                p = pn.lstrip(self.prot_prefix)
                f.write("\t".join([str(p)
                                  , str(self.median_dist[pn])
                                  , str(self.min_dist[pn])
                                 ])
                          + "\n"
                         )
    def calc_dists(self):
        import numpy as np
        self.median_dist = {}
        self.min_dist = {}
        for n in self.all_relevant_nodes:
            all_dists = []
            gen = (rs for rs in self.ref_set if rs != n)
            for rs in gen:
                try:
                    all_dists.append(nx.shortest_path_length(self.g, rs, n))
                except (nx.exception.NetworkXError, nx.exception.NetworkXNoPath):
                    pass
            try:
                self.min_dist[n] = float(min(all_dists))
                self.median_dist[n] = float( np.median(np.array(all_dists))) # float(sum(all_dists)) / len(all_dists)
            except ValueError:
                self.min_dist[n] = 0.0
                self.median_dist[n] = 0.0

if __name__ == "__main__":
    # force unbuffered output
    sys.stdout = sys.stderr

    import argparse
    parser = argparse.ArgumentParser(description='run community detection')
    parser.add_argument("outdir", help="directory to put output in")
    parser.add_argument("tmp_pubdir", help="directory for plots")
    parser.add_argument("ref_set", help="tab-separated list of reference set")
    parser.add_argument("report_set", help="tab-separated list of the set to report distance for")
    parser.add_argument("g", help="Pickled NetworkX graph")
    parser.add_argument("w", help="workspace ID")
    parser.add_argument("cores", type=int, help="Number of cores alloted")
    parser.add_argument("--no_norm", action="store_true", help="Don't normalize distances to the graph average")

    args=parser.parse_args()

    if args.no_norm:
        no_norm = True
    else:
        no_norm = False

    ts = time.time()
    run = hndist(
              ws = args.w,
              g_pkld = args.g,
              refset_file = args.ref_set,
              reportset_file = args.report_set,
              outdir = args.outdir,
              tmp_pubdir = args.tmp_pubdir,
              cores = args.cores,
              no_norm = no_norm,
            )
    run.run()
    print("Took: " + str(time.time() - ts) + ' seconds')
