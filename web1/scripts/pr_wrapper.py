#!/usr/bin/env python3

from __future__ import print_function
from builtins import range
import sys
import logging
logger = logging.getLogger(__name__)
from path_helper import PathHelper

import os

from algorithms.exit_codes import ExitCoder

debugging = False

class prWrapper(object):
    def __init__(self,**kwargs):
        self.ref_nodes = kwargs.get('ref_nodes',None)
        self.g = kwargs.get('g',None)
        self.restart_prob = kwargs.get('restart_prob',None)
        self.iters = kwargs.get('iterations',100)
        self.cores = kwargs.get('cores',None)
        self.aggregate = kwargs.get('aggregate',None)
        self.end_nodes = kwargs.get('end_nodes', [])
        assert self.aggregate in ['drugs', 'proteins']
        self.final_pr_d = {}
    def run(self):
        self.setup()
        self.parallel_run()
    def parallel_run(self):
        self.pr_results = None
        self.get_pr_dicts()
        if self.aggregate == 'drugs':
            self.agg_drugs(self.end_nodes)
        elif self.aggregate == 'proteins':
            self.agg_prots()
    def setup(self):
        from scripts.connect_drugs_to_proteinSets import establish_prefixes
        self.drug_prefix, _, self.prot_prefix = establish_prefixes()
        self.ppi_graph = self.g.subgraph([n
                                for n in self.g
                                if n.startswith(self.prot_prefix)
                                ])
    def get_pr_dicts(self):
        from scripts.connect_drugs_to_proteinSets import run_page_rank2
        from dtk.parallel import pmap

        static_args = dict(
                g=self.g,
                ppi_g=self.ppi_graph,
                prot_prefix=self.prot_prefix,
                restart_prob=self.restart_prob,
                iterations=self.iters
                )
        logger.info("Running PR")
        self.pr_results = list(pmap(
                run_page_rank2,
                self.ref_nodes,
                static_args=static_args,
                num_cores=self.cores,
                fake_mp=self.cores==1 or debugging,
                progress='PageRank',
                ))
        logger.info("Done PR")
    def agg_prots(self):
        for sub_dict in self.pr_results:
            self.agg_single_prot_dict(sub_dict)
    def agg_single_prot_dict(self, sub_dict):
        for prot in sub_dict:
            # this takes the max across all
            self.final_pr_d[prot] = max([sub_dict[prot],
                                         self.final_pr_d[prot] if prot in self.final_pr_d else sub_dict[prot]
                                        ])
    def agg_drugs(self, end_nodes):
        assert len(self.ref_nodes) == len(self.pr_results)
        from dtk.parallel import pmap
        out = pmap(self.agg_single_drug_dict, self.pr_results, static_args={'end_nodes': end_nodes}, progress='pagerank agg')
        self.final_pr_d = {ref_node:output for ref_node,output in zip(self.ref_nodes, out)}

    def agg_single_drug_dict(self, d, end_nodes):
        from scripts.connect_drugs_to_proteinSets import score_prot_rank
        return score_prot_rank(end_nodes,
                               self.g,
                               self.prot_prefix,
                               True,
                               d
                              )

if __name__ == "__main__":
    # force unbuffered output
    sys.stdout = sys.stderr

    import argparse
    import time
    import pickle

    parser = argparse.ArgumentParser(description='Score drugs with prot_rank')
    parser.add_argument("ref_nodes_pickle", help="Pickled list of reference nodes")
    parser.add_argument("g", help="DPI/PPI networkx pickle")
    parser.add_argument("out_pickle", help="file to report results")
    parser.add_argument("restart_prob", type = float, help="restart probability for protRank")
    parser.add_argument("cores", type=int, help="Number of available cores")
    parser.add_argument("--aggregate", default='drugs', help="Should PR scores be aggregated by drug or by prot")

    args=parser.parse_args()

    ts = time.time()
    with open(args.g, 'rb') as handle:
        g = pickle.load(handle)
    with open(args.ref_nodes_pickle, 'rb') as handle:
        ref_nodes = pickle.load(handle)

    from scripts.connect_drugs_to_proteinSets import establish_prefixes
    drug_prefix, _, prot_prefix = establish_prefixes()
    end_nodes = [n for n in g if n.startswith(drug_prefix)]
    run = prWrapper(
              ref_nodes = ref_nodes,
              g = g,
              restart_prob = args.restart_prob,
              cores = args.cores,
              aggregate = args.aggregate,
              end_nodes = end_nodes,
            )
    run.run()
    with open(args.out_pickle, 'wb') as handle:
        pickle.dump(run.final_pr_d, handle, protocol=2)
    print("Took: " + str(time.time() - ts) + ' seconds')
