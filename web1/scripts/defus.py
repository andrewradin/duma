#!/usr/bin/env python3

import sys
import logging
verbose=False
logger = logging.getLogger(__name__)
from path_helper import PathHelper
import os

from algorithms.exit_codes import ExitCoder

debugging = True

class defus(object):
    def __init__(self,**kwargs):
        self.in_data = kwargs.get('in_data',None)
        self.smiles = kwargs.get('smiles',None)
        self.dpi_map = kwargs.get('dpi_map',None)
        self.settings = kwargs.get('settings',None)
        self.cores = kwargs.get('cores',1)
        self.pathways = kwargs.get('pathways')

        # Convert to baseline DPI, we deal in molecules here.
        from dtk.prot_map import DpiMapping
        self.dpi = DpiMapping(self.settings['dpi']).get_baseline_dpi().choice
        if verbose:
            logger.info(f"Defus using baseline dpi choice {self.dpi} derived from {self.settings['dpi']}")

    def run(self):
        self.setup()

        pr_wrapper = self._make_pr_wrapper()

        from dtk.parallel import chunker, pmap
        import psutil
        def process_batch(drugs, pr_wrapper):
            try:
                sys.stderr.write(f"Processing defus batch from proc {os.getpid()}\n")
                ms = self.calc_sims(drugs, pr_wrapper)
                return self.score(ms)
            except:
                sys.stderr.write("Exception during process_batch\n")
                import traceback as tb
                tb.print_exc()
                raise
            finally:
                sys.stderr.write(f"Memory used at end of process_batch: {psutil.virtual_memory().percent}%")
        # Process batches of drugs.
        # We want lots of chunks for parallelization and less memory overhead,
        # (Memory because unscored intermediate results store a separate score
        # for each (drug, ref_drug), which is a lot!)
        # but not too many because there's a fair amount of startup overhead
        # and repeated work on the ref-set per chunk.
        drug_batches = chunker(self.all_drugs, chunk_size=500)
        logger.info("Starting batch processing")
        self.all_scores = {}
        self.all_connections = {}
        static_args = dict(pr_wrapper = pr_wrapper)
        # TODO: Something here, presumably in pr_wrapper, is holding onto tons of memory between invocations.
        # Instead of solving it, we use maxtasksperchild to keep a handle on memory usage.
        for scores, connections in pmap(process_batch, drug_batches, static_args=static_args, num_cores=self.cores, maxtasksperchild=1, progress=True):
            self.all_scores.update(scores)
            self.all_connections.update(connections)
        logger.info("Done processing batches")

    def setup(self):
        self.starting_drugs = list(self.in_data.keys())
        self.all_drugs = list(self.smiles.keys())

    def _make_pr_wrapper(self):
        sts = self.settings
        logger.info("Setting up pr wrapper")
        from scripts.metasim import make_pr_wrapper
        pr_wrapper = make_pr_wrapper(
                dpi_file=self.dpi,
                dpi_t=sts['dpi_t'],
                ppi_file=sts['ppi'],
                ppi_t=sts['ppi_t'],
                rs_set=self.starting_drugs,
                dpi_wsa_map=self.dpi_map,
                cores=self.cores)
        pr_wrapper.run()
        logger.info("prwrapper setup complete")
        return pr_wrapper

    def calc_sims(self, drugs, pr_wrapper):
        from scripts.metasim import metasim
        drugs_or_refs = set(drugs) | set(self.starting_drugs)
        ms = metasim(refset = self.starting_drugs,
                          all_drugs = drugs,
                          dpi = self.dpi,
                          ppi = self.settings['ppi'],
                          dpi_t = self.settings['dpi_t'],
                          ppi_t = self.settings['ppi_t'],
                          smiles = [[self.smiles[k],k]
                                    for k in drugs_or_refs
                                    if self.smiles.get(k) is not None
                                   ],
                          dpi_map = self.dpi_map,
                          wts = self.settings['sim_wts'],
                          cores = 1, # We're parallelizing out here instead.
                          rdkit_use_bit_smiles_fp = self.settings['rdkit_use_bit_smiles_fp'],
                          prwrapper=pr_wrapper,
                          std_gene_list_set = self.settings['std_gene_list_set'],
                          d2ps_threshold = self.settings.get('d2ps_threshold', 0),
                          d2ps_method = self.settings.get('d2ps_method', None),
                         )
        ms.setup()
        ms.run()
        return ms
    def score(self, ms):
        all_scores = {}
        all_connections = {}
        for wsa,d in ms.all_scores.items():
            scores = {}
            rwsas = {}
            for rs_wsa, all_sims_d in d.items():
                for sim_type, sim in all_sims_d.items():
                    if sim < self.settings['min_sim'][sim_type]:
                        continue
                    new_score = self._single_score(rs_wsa, sim)
                    if sim_type not in scores or new_score > scores[sim_type]:
                        scores[sim_type] = new_score
                        rwsas[sim_type] = rs_wsa
            all_scores[wsa] = scores
            all_connections[wsa] = rwsas
        return all_scores, all_connections

    def _single_score(self, rs_wsa, sim):
### PREFERRED: it made the target sims very slightly worse, but the structure noticeably better than using just Sim
        return (1.0-self.in_data[rs_wsa][0]) * self.in_data[rs_wsa][1] * sim**2
### I prefered not to use just p or OR, so the following weren't investigated
### OR * sim
#                new_score = self.in_data[rs_wsa][1] * sim
### 1-p * sim
#                new_score = (1.0-self.in_data[rs_wsa][0]) * sim
### using -log10(p) seemed to value the FAERS signal a little too strongly
### so the scores werre much better but that;s b/c FAERS will always be good at
### finding KTs. That's not predictive though. More investigation is warranted however
#        from math import log
### -log10(p) * OR * sim
#        return -1.0 * log(self.in_data[rs_wsa][0],10) * self.in_data[rs_wsa][1] * sim
#        return -1.0 * log(self.in_data[rs_wsa][0],10) * self.in_data[rs_wsa][1] * sim**2
### -log10(p) * sim
#                new_score = (-1.0 * log(self.in_data[rs_wsa][0]),10) * sim

if __name__ == "__main__":
    from dtk.log_setup import setupLogging
    setupLogging()

    import argparse
    import time
    import pickle

    parser = argparse.ArgumentParser(description='Score drugs with prot_rank')
    parser.add_argument("input_pickle", help="Pickled list of drugs, their p-vals and their enrichments")
    parser.add_argument("smiles_pickle", help="Pickled list of all drugs in the WS and their smiles, if they have one")
    parser.add_argument("dpiMap_pickle", help="Pickled dict of dpi map")
    parser.add_argument("settings_pickle", help="Pickled dict of settings")
    parser.add_argument("out_file", help="file to report results")
    parser.add_argument("cores", type=int, help="Number of available cores")

    args=parser.parse_args()

    ts = time.time()
    with open(args.input_pickle, 'rb') as handle:
        in_data = pickle.load(handle)
    with open(args.settings_pickle, 'rb') as handle:
        settings = pickle.load(handle)
    with open(args.smiles_pickle, 'rb') as handle:
        smiles = pickle.load(handle)
    with open(args.dpiMap_pickle, 'rb') as handle:
        dpi_map = pickle.load(handle)
    run = defus(
              in_data = in_data,
              smiles = smiles,
              dpi_map = dpi_map,
              settings = settings,
              cores = int(args.cores),
            )
    run.run()


    logger.info("Writing out results, we have %d scores", len(run.all_scores))
    from dtk.prot_map import DpiMapping
    dpi = DpiMapping(settings['dpi'])
    codetype = dpi.mapping_type()

    header = [codetype]
    header += [x for t in settings['sim_wts'] for x in [t+'Score', t+'ConnectingDrug']]
    with open(args.out_file, 'w') as f:
        f.write("\t".join(header) + "\n")
        for wsa, score_type_d in run.all_scores.items():
             out = [wsa]
             for st in settings['sim_wts']:
                 if st in run.all_connections[wsa]:
                     v = run.all_connections[wsa][st]
                     n = score_type_d[st]
                 else:
                     v = '-'
                     n = 0.0
                 out += [n, v]
             f.write("\t".join([str(x) for x in out]) + "\n")
    print("Took: " + str(time.time() - ts) + ' seconds')
