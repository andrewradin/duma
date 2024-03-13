#!/usr/bin/env python3

import sys
from path_helper import PathHelper

import os
import dtk.rdkit_2019
import time
from functools import lru_cache
import logging
from tqdm import tqdm
import numpy as np
logger = logging.getLogger(__name__)


def compute_pathway_overlap(pw_to_score1, pw_to_score2):
    all_pw = set(pw_to_score1.keys() | pw_to_score2.keys())

    num = 0
    den = 0
    for pw in all_pw:
        score1 = pw_to_score1.get(pw, 0)
        score2 = pw_to_score2.get(pw, 0)
        if score1 > score2:
            num += score2
            den += score1
        else:
            num += score1
            den += score2

    if den == 0:
        return 0 

    return num / den


def _setup_dpi(dpi_file, dpi_t, wsas_of_interest=None, dpi_wsa_map=None):
        from dtk.prot_map import DpiMapping
        dpi_obj = DpiMapping(dpi_file)
        logger.info(f"Metasim using dpi file at {dpi_file}, {dpi_obj.choice}")
        dpi_rev_map = {}
        for k,l in dpi_wsa_map.items():
            for w in l:
                if wsas_of_interest and w not in wsas_of_interest:
                    continue
                if w not in dpi_rev_map:
                    dpi_rev_map[w] = []
                dpi_rev_map[w].append(k)
        if wsas_of_interest is None:
            wsas_of_interest = dpi_rev_map.keys()
        from dtk.files import get_file_records
        dpi_data={}
        for frs in get_file_records(dpi_obj.get_path(),
                                    keep_header = False):
            if float(frs[2]) < dpi_t:
                continue
            if frs[0] in dpi_data:
                dpi_data[frs[0]].append(frs[1])
            else:
                dpi_data[frs[0]] = [frs[1]]
        dpi_keys = {}
        for wsa in wsas_of_interest:
            l = dpi_rev_map.get(wsa, [])
            if len(l) != 1 and verbose:
                print("WARNING: Unable to find only one DPI key for", wsa, l)
            if not l:
                continue
            k_to_use = None
            best_cnt = 0
            dpi_to_use = []
            for x in l:
                dpi = dpi_data.get(x,[])
                cnt = len(dpi)
                if cnt > best_cnt:
                    k_to_use = x
                    best_cnt = cnt
                    dpi_to_use = dpi
            if best_cnt == 0:
                if verbose:
                    print('WARNING: no proteins found for', wsa, 'using any of the DPI keys. Skipping.')
                continue
            dpi_data[wsa] = dpi_to_use
            dpi_keys[wsa] = k_to_use
        return dpi_data, dpi_keys

def _create_graph(prot_prefix, dpi_file, ppi_map, ppi_t):
    import networkx as nx
    from scripts.connect_drugs_to_proteinSets import build_keyed_dpi_graph as bkdg, build_ppi_graph
    dpi_graph = bkdg(dpi_file)
    ppi_graph = build_ppi_graph(
            ppi_map,
            prot_prefix=prot_prefix,
            min_ppi_evid=ppi_t,
            )
    return nx.compose(dpi_graph, ppi_graph)

def _make_ref_nodes(drug_prefix, rs_set, nodes, dpi_keys):
    ref_nodes = []
    gen = (rs for rs in rs_set if rs in dpi_keys)
    for rs_wsa in gen:
        k = dpi_keys[rs_wsa]
        node = drug_prefix + k
        if node not in nodes:
            print("WARNING:", k, 'Not found in the DPI graph')
            print("It will be skipped")
            continue
        ref_nodes.append(node)
    return ref_nodes

def make_pr_wrapper(dpi_file, dpi_t, ppi_file, ppi_t, rs_set, dpi_wsa_map, cores):
    from scripts.connect_drugs_to_proteinSets import establish_prefixes as ep
    from scripts.pr_wrapper import prWrapper
    drug_prefix, _, prot_prefix = ep()
    from dtk.prot_map import PpiMapping
    ppi_map = PpiMapping(ppi_file)
    dpi_data, dpi_keys = _setup_dpi(dpi_file, dpi_t, dpi_wsa_map=dpi_wsa_map)
    g = _create_graph(prot_prefix, dpi_file, ppi_map, ppi_t)
    ref_nodes = _make_ref_nodes(drug_prefix, rs_set, g.nodes(), dpi_keys)
    pr = prWrapper(
                      ref_nodes = ref_nodes,
                      g = g,
                      restart_prob = 0.85,
                      cores = cores,
                      aggregate = 'drugs'
                     )
    return pr

verbose = False
class metasim(object):
    def __init__(self, std_gene_list_set=None, **kwargs):
        self.ws = kwargs.get('ws', None)
        # dpi and ppi info is required, so that we don't need to hassle
        # about dealing with workspace-specific defaults
        self.dpi = kwargs.get('dpi', None)
        self.dpi_t = kwargs.get('dpi_t', None)
        self.ppi = kwargs.get('ppi', None)
        self.ppi_t = kwargs.get('ppi_t', None)
        self.score_wts = kwargs.get('wts', None)
        self.min_sim = kwargs.get('min_sim', 0.0)
# these next few are usually only sent when this is being run on the worker machine
# and as a result the WS isn't available to build these
        self.cores = kwargs.get('cores', 1)
        self.rs_set = kwargs.get('refset', None)
        self.dpi_map = kwargs.get('dpi_map', None)
        self.smiles = kwargs.get('smiles', None)
        self.all_drugs = kwargs.get('all_drugs', None)
        self.use_bit_smiles_fp = kwargs.get('rdkit_use_bit_smiles_fp', None)
        self.d2ps_threshold = kwargs.get('d2ps_threshold', 0)
        self.d2ps_method = kwargs.get('d2ps_method', None)

        if self._check_wt('pathway'):
            from dtk.d2ps import D2ps
            self.d2ps = D2ps(ppi_id=self.ppi, ps_id=std_gene_list_set, method=self.d2ps_method)

        self.matches=[]
        if self.all_drugs is None:
            from browse.models import WsAnnotation
            self.all_drugs = [x.id for x in WsAnnotation.objects.filter(ws=self.ws.id)]

        self.all_drugs = set(self.all_drugs)
        self.drugs_and_refs = self.all_drugs | set(self.rs_set)
        self.pr_wrapper = kwargs.get('prwrapper', None)
        from scripts.connect_drugs_to_proteinSets import establish_prefixes as ep
        self.drug_prefix, _, self.prot_prefix = ep()
    def setup(self):
        if (self.score_wts.get('dirJac') or
            self.score_wts.get('indjac') or
            self.score_wts.get('prMax') or
            self.score_wts.get('pathway')
            ):
            self.prot_setup()
        self._setup_scores()
        self.struc_setup()
    def prot_setup(self):
        self._setup_dpi()
        if self.ppi:
            self._setup_ppi()
    def _setup_scores(self):
        if self.rs_set is None:
            from browse.models import WsAnnotation
            self.rs_set = [x.id for x in WsAnnotation.objects.filter(ws=self.ws.id)]
        # NOTE: Be careful about printing too much here, it can end up
        # generating crazy amounts of output in target importance.
        #print("Constructing all_scores with %d x %d dicts" % (len(self.rs_set), len(self.all_drugs)))
        self.all_scores = {w:{
                              k:{} for k in self.rs_set
                             }
                           for w in self.all_drugs
                          }
        self.matches = []
        if self.score_wts is None:
            # I currently prefer the target based scores and the structure based scores
            # to sum to the same value (currently 2)
            self.score_wts = {
                          'dirJac' : 0.5,
                          'indJac' : 0.5,
                          'prMax' : 1.0,
                          'rdkit' : 1.0,
                          'indigo' : 1.0,
                          'pathway': 1.0,
                         }
        if self.use_bit_smiles_fp:
            print("Turning off indigo, using bit smiles")
            self.score_wts['indigo'] = 0
        to_remove = [k for k,v in self.score_wts.items() if v <= 0]
        for k in to_remove:
            del self.score_wts[k]
        #print("Using score weights %s, removed %s" % (self.score_wts, to_remove))
        norm = sum(self.score_wts.values())
        self.score_wts = {k:v/norm for k,v in self.score_wts.items()}
    def _setup_ppi(self):
        from dtk.prot_map import PpiMapping
        ppi_obj = PpiMapping(self.ppi)
        self.ppi_map = ppi_obj
        self.ppi_data={}
        gen = (frs for frs in ppi_obj.get_data_records(min_evid=self.ppi_t))
        for frs in gen:
            if frs[0] in self.ppi_data:
                self.ppi_data[frs[0]][frs[1]] = float(frs[2])
            else:
                self.ppi_data[frs[0]] = {frs[1]:float(frs[2])}
    @lru_cache(maxsize=100000)
    def _get_ppi_subset_cached(self, l):
        to_ret = set()
        for p in l:
            to_ret.update(self.ppi_data.get(p, {}).keys())
        return frozenset(to_ret)
    def _get_ppi_subset(self, l):
        return self._get_ppi_subset_cached(frozenset(l))
    def _setup_dpi(self):
        if not self.dpi_map:
            assert self.ws
            from dtk.prot_map import DpiMapping
            dpi_obj = DpiMapping(self.dpi)
            self.dpi_map = dpi_obj.get_wsa_id_map(self.ws)

        self.dpi_data, self.dpi_keys = _setup_dpi(
                dpi_file=self.dpi,
                dpi_t=self.dpi_t,
                wsas_of_interest=self.drugs_and_refs,
                dpi_wsa_map=self.dpi_map,
                )

    def run(self):
        print('calculating individual similarities...')
        if self._check_wt('rdkit'):
            ts = time.time()
            self._run_rdkit()
            print('Running RDKit took:', time.time()-ts)
            self.molecSim.fp = {}
        if self._check_wt('indigo'):
            self.molecSim.lib = 'indigo'
            ts = time.time()
            self._run_indigo()
            print('Running Indigo took:', time.time()-ts)
        if self.molecSim:
            del self.molecSim
        if self._check_wt('prMax'):
            ts = time.time()
            self._run_prsim()
            print('Running PRsim took:', time.time()-ts)
        if self._check_wt('dirJac') or self._check_wt('indJac'):
            ts = time.time()
            self._run_jacsim()
            print('Running JacSim took:', time.time()-ts)
        if self._check_wt('pathway'):
            ts = time.time()
            self.run_pathway()
            print('Running pathway took:', time.time()-ts)

    def _check_wt(self, k):
         return k in self.score_wts and self.score_wts[k] > 0
    def _run_rdkit(self):
        self.molecSim.run()
        t = time.time()
        self._process_struc_scores('rdkit')
        print('processing scores for rdkit took:', time.time()-t)
    def _process_struc_scores(self,k):
        d = self.molecSim.get_results_dict()
        gen_list = [(wsa,l) for wsa,l in d.items() if l and wsa in self.all_drugs]
        inner_iters = [(i,rs_wsa) for i,rs_wsa in enumerate(self.molecSim.rs)]
        iters = len(gen_list)
        params = zip([inner_iters] * iters,
                     [k] * iters,
                     [self.all_scores[x[0]] for x in gen_list],
                     gen_list
                    )
        if self.cores > 1:
            import multiprocessing
            pool = multiprocessing.Pool(self.cores)
            results = pool.map(extract_struc_sims, params)
        else:
            results = map(extract_struc_sims, params)
        for tup in results:
             self.all_scores[tup[0]] = tup[1]
    def _run_indigo(self):
        self.molecSim.run()
        t = time.time()
        self._process_struc_scores('indigo')
        print('processing scores for indigo took:', time.time()-t)
    def struc_setup(self):
        if not self._check_wt('indigo') and not self._check_wt('rdkit'):
            self.molecSim = None
            return

        try:
            from similarity_csv import molecSim
        except ImportError:
            import sys
            sys.path.insert(1, PathHelper.website_root+"/../moleculeSimilarity/fingerprint")
            from similarity_csv import molecSim
        if self.smiles is None:
            self._load_smiles()
        self.molecSim = molecSim(
                            smiles = self.smiles,
                            refset_frs = [[x] for x in self.rs_set],
                            lib = 'rdkit',
                            map = False,
                            cores = self.cores,
                            use_bit_smiles_fp = self.use_bit_smiles_fp,
                        )

    def _load_smiles(self):
        from dtk.data import MultiMap
        from scripts.metasim import load_wsa_smiles_pairs
        mm = MultiMap(load_wsa_smiles_pairs(self.ws))
        self.smiles = []
        for k,v in MultiMap.flatten(mm.fwd_map()):
            self.smiles.append([v,k])
    def full_combine_scores(self):
        for k,d in self.all_scores.items():
            score = 0.0
### This approach gives an implicit zero if a sim method is missing
### and/or if a comparison/reference drug is missing for any or all methods
### That seems appropriate, but we just just note it
            for v in d.values():
                score += self._condense_comparisons(v)
            if score >= self.min_sim:
                self.matches.append([k,score])
    def combine_filter(self):
        ts = time.time()
        self.final_scores = {}
        for wsa,d in self.all_scores.items():
            self.final_scores[wsa] = {}
            for rs_wsa, inner_d in d.items():
                score = self._condense_comparisons(inner_d)
                if score >= self.min_sim:
                    if wsa != rs_wsa and verbose:
                        print(wsa, rs_wsa, inner_d)
                    self.final_scores[wsa][rs_wsa] = score
        print('combining took:', time.time()-ts)
    def _condense_comparisons(self, d):
        score = 0.0
        for score_name,wt in self.score_wts.items():
            score += d.get(score_name, 0.0)*wt
        return score
    def _run_prsim(self):
        if self.pr_wrapper:
            pr = self.pr_wrapper
        else:
            pr = make_pr_wrapper(
                    self.dpi,
                    self.dpi_t,
                    self.ppi,
                    self.ppi_t,
                    self.rs_set,
                    self.dpi_map,
                    self.cores,
                    )
            pr.run()
            self.pr_wrapper = pr

        end_nodes = [self.drug_prefix + self.dpi_keys[wsa] for wsa in self.drugs_and_refs if wsa in self.dpi_keys]
        pr.agg_drugs(end_nodes)

        gen = (rs for rs in self.rs_set if rs in self.dpi_keys)
        for rs_wsa in gen:
            rn = self.drug_prefix + self.dpi_keys[rs_wsa]
            if rn not in pr.final_pr_d:
                continue
### we need to normalize these score so that for each, the ref_node self comp is 1
### and down below we normalize for the number of ref nodes
### the end result is that all values should range b/t 1 and 0
            normalizer = pr.final_pr_d[rn][rn]['protrank_max']
            for k,d in pr.final_pr_d[rn].items():
                k_mod = k.lstrip(self.drug_prefix)
                if k_mod not in self.dpi_map:
                    # not all names in the dpi mapping are guaranteed to
                    # be in the workspace (and so, in the dpi_map)
                    continue
                for wsa in self.dpi_map[k_mod]:
                    if wsa not in self.all_drugs:
                        continue
                    if 'prMax' not in self.all_scores[wsa][rs_wsa]:
                        self.all_scores[wsa][rs_wsa]['prMax'] = 0.0
                    self.all_scores[wsa][rs_wsa]['prMax'] += (
                                                              d['protrank_max']/
                                                              normalizer
                                                             ) / len(self.dpi_map[k_mod])
### The last division (/ len(self.dpi_map[k_mod])) effectively takes the average when there are
### multiple DPI keys for a WSA
    def run_jacsim(self):
        self._run_jacsim()
    def _run_jacsim(self):
        relevant_rs_wsas = [rs_wsa
                            for rs_wsa in self.rs_set
                            if rs_wsa in self.dpi_data
                           ]
        for wsa, v in self.dpi_data.items():
            if wsa not in self.all_drugs:
                # the ref wsas are in dpi data even if not in all_drugs.
                continue
            dir_targets = set(v)
            indir_targets = self._get_ppi_subset(dir_targets)
            for rs_wsa in relevant_rs_wsas:
                sim = self._score_jac_sim(set(self.dpi_data[rs_wsa]), dir_targets, indir_targets)
                self.all_scores[wsa][rs_wsa]['dirJac'] = sim[0]
                self.all_scores[wsa][rs_wsa]['indJac'] = sim[1]
    def _score_jac_sim(self, targs, dir_prots, indir_prots):
        from dtk.similarity import calc_jaccard
        sim = [calc_jaccard(dir_prots, targs)]
        sim.append(calc_jaccard(dir_prots | indir_prots,
                                    self._get_ppi_subset(targs)
                                  ))
        return sim

    def run_pathway(self):
        from dtk.d2ps import MoA, D2ps
        # Generate MoAs for each wsa, rs_Wsa
        def targets_to_moa(targets):
            # We're ignoring evidence and direction here.
            return MoA([[target, 1, 0] for target in targets])
        relevant_rs_wsas = {rs_wsa
                            for rs_wsa in self.rs_set
                            if rs_wsa in self.dpi_data
                            }

        wsa2moa = {wsa: targets_to_moa(targets) for wsa, targets in self.dpi_data.items()
                    if wsa in self.all_drugs or wsa in relevant_rs_wsas}


        # Construct a d2ps
        d2ps = self.d2ps

        # Run update_for_moas
        d2ps.update_for_moas(wsa2moa.values())

        logger.info("Computing pathway overlap")

        moa2pw = {}

        # Use caching - we're operating on molecules here, so there's going to be
        # a lot of MoA overlap.
        @lru_cache(maxsize=100000)
        def moa_pathway_overlap(moa1, moa2):
            if moa1 not in moa2pw:
                moa2pw[moa1] = {pw.pathway:pw.score for pw in d2ps.get_moa_pathway_scores(moa1) if pw.score >= self.d2ps_threshold}
            if moa2 not in moa2pw:
                moa2pw[moa2] = {pw.pathway:pw.score for pw in d2ps.get_moa_pathway_scores(moa2) if pw.score >= self.d2ps_threshold}
            
            pws1 = moa2pw[moa1]
            pws2 = moa2pw[moa2]
            return compute_pathway_overlap(pws1, pws2)

        # Compute a pathway jaccard index.
        for wsa in self.all_drugs:
            if wsa not in self.dpi_data:
                continue
            wsa_moa = wsa2moa[wsa]
            for rs_wsa in relevant_rs_wsas:
                rs_moa = wsa2moa[rs_wsa]
                score = moa_pathway_overlap(wsa_moa, rs_moa)
                self.all_scores[wsa][rs_wsa]['pathway'] = score
        logger.info(f"Cache stats: {moa_pathway_overlap.cache_info()}")
            
    

def extract_struc_sims(params):
    inner_iters, k, d, tup = params
    wsa, l = tup
    for i,rs_wsa in inner_iters:
        d[rs_wsa][k] = tup[1][i]
    return (tup[0], d)

def load_wsa_smiles_pairs(ws, wsas=None):
    """Returns a list of (wsa, SMILES).

    Tries to provide std_smiles, but will fall back to smiles_code if needed.
    Note that in special cases the SMILES could be a list of bit smiles.
    """
    from drugs.models import Drug, Prop
    wsas = wsas or ws.wsannotation_set.all()
    agent_to_wsa  = dict((wsa.agent_id, wsa) for wsa in wsas)
    ver = ws.get_dpi_version()
    linked_agents = Drug.get_linked_agents_map(
            list(agent_to_wsa.keys()),
            version=ver,
            prop_type=Prop.prop_types.BLOB,
            )
    wsa_smiles_pairs = []
    for linked_agent_id, agents in linked_agents.items():
        had_any_smiles = False
        smiles_success = False
        for agent in agents:
            if agent.std_smiles:
                had_any_smiles = True
                from rdkit import Chem
                smiles = agent.std_smiles_or_bitfp()
                if isinstance(smiles, str):
                    if not smiles:
                        continue

                wsa = agent_to_wsa[linked_agent_id]
                wsa_smiles_pairs.append((wsa.id, smiles))
                smiles_success = True
                break
        if had_any_smiles and not smiles_success:
            print(("Failed to properly convert any SMILES for ", linked_agent_id))
    return wsa_smiles_pairs
