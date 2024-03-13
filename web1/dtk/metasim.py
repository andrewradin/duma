import logging
from dtk.subclass_registry import SubclassRegistry

logger = logging.getLogger(__name__)
    
from tools import Enum

class KeyType(SubclassRegistry):
    pass

class StructKey(KeyType):
    @classmethod
    def keys_for_agents(cls, agent_ids, dpi_choice):
        logger.info(f"Looking up struct keys for {len(agent_ids)} agents")
        from drugs.models import Blob, Drug
        from dtk.data import MultiMap
        from dtk.prot_map import DpiMapping
        from dtk.moa import make_moa_to_mol_agents
        dpi = DpiMapping(dpi_choice)
        thresh = DpiMapping.default_evidence

        agent_id_mm = Drug.matched_id_mm(agent_ids, dpi.version)

        all_agent_ids = agent_id_mm.rev_map()
        all_agent_smiles = MultiMap(Blob.objects.filter(prop__name='std_smiles', drug__in=all_agent_ids).values_list('drug_id', 'value'))

        out = []
        for agent_id in agent_ids:
            for linked_id in agent_id_mm.fwd_map()[agent_id]:
                for smiles in all_agent_smiles.fwd_map().get(linked_id, []):
                    out.append((agent_id, smiles))
        
        # MoAs get the keys of all mols with identical MoAs.
        logger.info("Getting moa to mol mapping")
        moa_to_mol_agents = make_moa_to_mol_agents(dpi_choice, thresh)

        moa_agents = moa_to_mol_agents.keys() & set(agent_ids)
        if moa_agents:
            moamol_agents = set()
            for moa_agent in moa_agents:
                moamol_agents.update(list(moa_to_mol_agents[moa_agent]))
            logger.info(f"{len(moa_agents)} moa agents, corresponding to {len(moamol_agents)} mols")

            if False:
                # NOTE: This is probably more correct, expanding out moa -> mols -> mol clusters, but 
                # takes basically forever.
                moamol_agent_id_mm = Drug.matched_id_mm(moamol_agents, dpi.version)
            else:
                moamol_agent_id_mm = MultiMap((x, x) for x in moamol_agents)

            all_moamol_agent_ids = moamol_agent_id_mm.rev_map().keys()
            all_moamol_agent_smiles = MultiMap(Blob.objects.filter(prop__name='std_smiles', drug__in=all_moamol_agent_ids).values_list('drug_id', 'value'))

            out = []
            for moa_agent_id in moa_agents:
                moa_smiles = set()
                for agent_id in moa_to_mol_agents[moa_agent_id]:
                    for linked_id in moamol_agent_id_mm.fwd_map().get(agent_id, []):
                        for smiles in all_moamol_agent_smiles.fwd_map().get(linked_id, []):
                            moa_smiles.add(smiles)
                for smiles in moa_smiles:
                    out.append((moa_agent_id, smiles))
        
        # Note that each agent could have multiple std_smiles, sometimes in the mol case,
        # but especially in the MoA case.
        out_mm = MultiMap(out)
        out_map = out_mm.fwd_map()
        logger.info(f"Outputting {len(out_mm.rev_map())} smiles corresponding to {len(out_map)} agents")
        return out_map

class TargetsKey(KeyType):
    @classmethod
    def keys_for_agents(cls, agent_ids, dpi_choice):
        # Don't need the id_mm here, dpi will handle any agent in a cluster.
        from dtk.prot_map import AgentTargetCache, DpiMapping, MultiAgentTargetCache
        dpi = DpiMapping(dpi_choice)
        atc = AgentTargetCache(mapping=dpi, dpi_thresh=DpiMapping.default_evidence, agent_ids=agent_ids)
        matc = MultiAgentTargetCache(atc)

        out = {}
        for agent_id in agent_ids:
            prots = []
            for drugkey, uniprot, ev, dr in matc.raw_info_for_agent(agent_id):
                prots.append(uniprot)
            prots = tuple(sorted(prots))
            out[agent_id] = prots
        return out

    @classmethod
    def keys_for_uniprots(cls, dpi_choice):
        from dtk.prot_map import DpiMapping
        dpi = DpiMapping(dpi_choice)
        prots = dpi.get_uniq_target()
        return {prot: (prot,) for prot in prots}

class SimType(SubclassRegistry):
    """
    Members:
        key_type: a KeyType indicating this algorithm's molecule representation (i.e. targets or structure)
        name: string for naming this in results and options.
    Methods:
        ref_precompute: Precompute anything that can be done based purely on the ref_keys
                        This is important for target importance efficiency, as it only needs to run once.
        setup_output_map: Additional precomputation including the full universe of keys to be considering
        similar_to_key: Returns similarity map for a given ref key, using the precomputed data.
    """

class DirectTargetSim:
    """
    Similarity by direct target.
    
    Computed as the jaccard score between the direct targets of the two drugs/MoAs.
    """

    key_type = TargetsKey
    name = 'dirJac'

    def ref_precompute(self, ref_to_key):
        return []
    
    def setup_output_map(self, agent_to_key, ref_to_key, ref_precompute):
        from collections import defaultdict
        from dtk.data import MultiMap
        protset_to_agent = MultiMap((frozenset(key),agent) for agent, key in agent_to_key.items()).fwd_map()
        prot_to_sets = defaultdict(set)
        for protset in protset_to_agent.keys():
            for prot in protset:
                prot_to_sets[prot].add(frozenset(protset))
        return [protset_to_agent, prot_to_sets]

    def similar_to_key(self, key, threshold, pre_cmp):
        from collections import defaultdict
        out = defaultdict(float)
        protset_to_agent, prot_to_sets = pre_cmp 
        seen = set()
        key = set(key)
        for prot in key:
            for protset2 in prot_to_sets[prot]:
                if protset2 in seen:
                    continue
                seen.add(protset2)
                sim_val = self.score(key, protset2)
                if sim_val > threshold:
                    for output_id in protset_to_agent[protset2]:
                        out[output_id] = max(out[output_id], sim_val)
        return out

    @classmethod
    def score(cls, protset1, protset2):
        from dtk.similarity import calc_jaccard
        return calc_jaccard(protset1, protset2)



def make_indir_protsets(protset, prot_ppi):
    # Include direct prots in the set for comparison.
    indirs = set(protset)
    for prot in protset:
        indirs |= prot_ppi.get(prot, set())
    return indirs

class IndirectTargetSim:
    """
    Similarity by indirect target.
    
    Computed as the jaccard score between the indirect targets of the two drugs/MoAs.
    """
    IndirTargSim='.indtargsim.npz'
    key_type = TargetsKey
    name = 'indJac'

    def __init__(self, ppi_choice):
        self.ppi_choice = ppi_choice

    def ref_precompute(self, ref_to_key):
        from dtk.data import MultiMap
        from dtk.prot_map import PpiMapping
        from dtk.parallel import pmap
        logging.info("Loading in ppi")
        ppi = PpiMapping(self.ppi_choice)
        ppis = []
        for prot1, prot2, evid, dr in ppi.get_data_records(min_evid=PpiMapping.default_evidence):
            ppis.append((prot1, prot2))
        prot_ppi = MultiMap(ppis).fwd_map()

        protsets = {frozenset(key) for key in ref_to_key.values()}
        protset_to_indirs = {}
        data = pmap(make_indir_protsets, protsets, static_args={'prot_ppi': prot_ppi})
        for protset, indir_protset in zip(protsets, data):
            protset_to_indirs[protset] = indir_protset

        return prot_ppi, protset_to_indirs

    def setup_output_map(self, agent_to_key, ref_to_key, ref_precompute):
        from dtk.data import MultiMap
        from dtk.parallel import pmap
        from collections import defaultdict
        
        prot_ppi, protset_to_indirs = ref_precompute
        protset_to_agents = MultiMap((frozenset(key),agent) for agent, key in agent_to_key.items()).fwd_map()
        protsets = protset_to_agents.keys()
        data = pmap(make_indir_protsets, protsets, static_args={'prot_ppi': prot_ppi})
        for protset, indir_protset in zip(protsets, data):
            protset_to_indirs[protset] = indir_protset
        
        return [prot_ppi, protset_to_agents, protset_to_indirs]

    def similar_to_key(self, key, threshold, pre_cmp):
        from collections import defaultdict
        from dtk.similarity import calc_jaccard
        out = defaultdict(float)
        prot_ppi, protset_to_agents, protset_to_indirs = pre_cmp

        ind_protset1 = protset_to_indirs[frozenset(key)]
        protsets = protset_to_agents.keys()

        for protset2 in protsets:
            ind_protset2 = protset_to_indirs[protset2]
            score_val = calc_jaccard(ind_protset1, ind_protset2) 
            if score_val > threshold:
                for output_id in protset_to_agents[protset2]:
                    out[output_id] = max(out[output_id], score_val)
        
        return out
        

class DiffusedTargetSim:
    """
    Similarity by diffusing reference drug targets.
    
    Drug A's targets are diffused along the PPI network and Drug B's similarity to A is computed
    as the maximum diffused value over any of Drug B's targets.

    Unlike most other similarities, this is asymmetric (similarity of A vs B != B vs A)
    """
    key_type = TargetsKey
    name = 'prMax'

    def __init__(self, ppi_choice):
        self.ppi_choice = ppi_choice
        from scripts.connect_drugs_to_proteinSets import establish_prefixes
        drug_prefix, _, prot_prefix = establish_prefixes()
        self.drug_prefix = drug_prefix
        self.prot_prefix = prot_prefix
    
    def _build_keyed_dpi_graph(self, ref_to_key):
        import networkx as nx
        g = nx.DiGraph()

        ref_nodes = []
        key_to_refnode = {}
        for agent_id, protset in ref_to_key.items():
            if not protset:
                continue
            agent_id = str(agent_id)
            agent_node_id = self.drug_prefix + agent_id
            key_to_refnode[protset] = agent_node_id
            ref_nodes.append(agent_node_id)
            for prot in protset:
                g.add_edge(agent_node_id,
                           self.prot_prefix + prot,
                           weight=1,
                           direction=1,
                        )

        return g, ref_nodes, key_to_refnode
    
    def ref_precompute(self, ref_to_key):
        from scripts.connect_drugs_to_proteinSets import build_ppi_graph
        from dtk.prot_map import PpiMapping
        dpi_graph, ref_nodes, key_to_refnode = self._build_keyed_dpi_graph(ref_to_key)
        ppi_graph = build_ppi_graph(
            PpiMapping(self.ppi_choice),
            prot_prefix=self.prot_prefix,
            min_ppi_evid=PpiMapping.default_evidence,
            )
        import networkx as nx
        g = nx.compose(dpi_graph, ppi_graph)
        from scripts.pr_wrapper import prWrapper
        pr = prWrapper(
                        ref_nodes = ref_nodes,
                        g = g,
                        restart_prob = 0.85,
                        aggregate = 'drugs'
                        )
        pr.run()
        return [pr, ref_nodes, key_to_refnode]

    def setup_output_map(self, agent_to_key, ref_to_key, ref_precompute):
        pr, ref_nodes, key_to_refnode = ref_precompute

        dpi_graph, end_nodes, _ = self._build_keyed_dpi_graph(agent_to_key)
        import networkx as nx
        pr.g = nx.compose(pr.g, dpi_graph)

        # Ideally we'd only agg on end_nodes, but the normalization relies on a self-score as well.
        # TODO: Does it actually matter?  Scaling everything by a constant shouldn't do much.
        pr.agg_drugs(ref_nodes + end_nodes)

        return [pr, key_to_refnode, set(end_nodes)]

    def similar_to_key(self, key, threshold, pre_cmp):
        from collections import defaultdict
        out = {}

        pr, key_to_refnode, end_nodes = pre_cmp

        if key not in key_to_refnode:
            return out

        rn = key_to_refnode[key]

        if rn not in pr.final_pr_d:
            return out

        normalizer = pr.final_pr_d[rn][rn]['protrank_max']
        for other_drug, score_dict in pr.final_pr_d[rn].items():
            if other_drug not in end_nodes:
                # There are cases where the ref nodes aren't part of end nodes
                # (e.g. uniprot dpi)
                continue
            other_drug = other_drug.lstrip(self.drug_prefix)
            try:
                other_drug = int(other_drug)
            except ValueError:
                # This happens when we have uniprot DPI here, where the 'agents' are just uniprots.
                pass
            score_val = score_dict['protrank_max'] / normalizer
            if score_val > threshold:
                out[other_drug] = score_val

        return out


def compute_pathway_overlap(pw_to_score1, pw_to_score2):
    overlap_pw = pw_to_score1.keys() & pw_to_score2.keys()
    if not overlap_pw:
        return 0

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

class PathwaySim:
    """
    Similarity by pathways.
    
    Computes a jaccard'esque metric on the D2PS pathway scores of the two molecules.
    """
    key_type = TargetsKey
    name = 'pathway'

    def __init__(self, ppi_choice, std_gene_list_set, d2ps_method, d2ps_threshold):
        self.ppi_choice = ppi_choice
        self.std_gene_list_set = std_gene_list_set
        self.d2ps_method = d2ps_method
        self.d2ps_threshold = d2ps_threshold

    def targets_to_moa(self, targets):
        from dtk.d2ps import MoA
        # We're ignoring evidence and direction here.
        return MoA([[target, 1, 0] for target in targets])

    def make_d2ps(self):
        from dtk.d2ps import D2ps
        return D2ps(ppi_id=self.ppi_choice, ps_id=self.std_gene_list_set, method=self.d2ps_method)

    def ref_precompute(self, ref_to_key):
        return []

    def setup_output_map(self, agent_to_key, ref_to_key, ref_precompute):
        d2ps = self.make_d2ps()
        from dtk.data import MultiMap
        protset_to_agent = MultiMap((frozenset(key),agent) for agent, key in agent_to_key.items()).fwd_map()

        protsets = list(protset_to_agent.keys())
        
        moas = [self.targets_to_moa(key) for key in protsets]
        logger.info(f"Checking d2ps for {len(moas)} ws MoAs")
        d2ps.update_for_moas(moas)

        ref_moas = [self.targets_to_moa(key) for key in ref_to_key.values()]
        logger.info(f"Checking d2ps for {len(ref_moas)} ref MoAs")
        d2ps.update_for_moas(ref_moas)

        from dtk.gene_sets import get_pathway_sets
        exclude_sets = get_pathway_sets(['type=cellular_component'], self.std_gene_list_set)
        excludes = set().union(*exclude_sets)
        logger.info(f"Have {len(excludes)} pws to exclude")

        logger.info("Extracting pathway signatures")
        def extract_pws(moa):
            d2ps = self.make_d2ps()
            return {pw.pathway:pw.score for pw in d2ps.get_moa_pathway_scores(moa) if pw.score >= self.d2ps_threshold and pw.pathway not in excludes}
        
        from dtk.parallel import pmap
        pws = list(pmap(extract_pws, moas, progress='pathway setup'))

        import numpy as np
        mean_pw_count = np.mean([len(x) for x in pws])
        logger.info(f"Moas have avg of {mean_pw_count} pathways above threshold {self.d2ps_threshold}")

        logger.info("Assembling data")
        key2pw = {}
        assert len(protset_to_agent) == len(pws)
        for key, pws in zip(protset_to_agent.keys(), pws):
            key = frozenset(key)
            key2pw[key] = pws

        return [key2pw, protset_to_agent, excludes]

    def similar_to_key(self, key, threshold, pre_cmp):
        key2pw, protset_to_agents, excludes = pre_cmp
        d2ps = self.make_d2ps()
        
        moa = self.targets_to_moa(key)

        cur_pws = {pw.pathway:pw.score for pw in d2ps.get_moa_pathway_scores(moa) if pw.score >= self.d2ps_threshold and pw.pathway not in excludes}

        from collections import defaultdict
        out = defaultdict(float)
        for protset, pws in key2pw.items():
            score = compute_pathway_overlap(cur_pws, pws)
            if score > threshold:
                for output_id in protset_to_agents[protset]:
                    out[output_id] = max(out[output_id], score)
        return out

class StructSim:
    """
    Similarity by structure.
    
    Tanimoto (jaccard) similarity between the Circular Morgan fingerprints of the two molecules.
    https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
    "Roughly equivalent to ECFP4 [fingerprints]"

    These are folded/hashed to FingerprintBits, which typically results in very similar, but sometimes slightly
    overestimated similarity values.

    Relies on a precomputed similarity database.

    TODO: Convert back to dice similarity like in original metasim?
    """
    key_type = StructKey
    MorganRadius=2
    FingerprintBits=1024
    name = 'rdkit'

    FpDbExt='.fpdb.h5'
    MetadataExt='.struct_metadata.json'

    @classmethod
    def precompute_to_files(cls, fn_prefix, attrs_files):
        db_fn = fn_prefix + cls.FpDbExt
        metadata_fn = fn_prefix + cls.MetadataExt

        logger.info("Loading in data")
        from dtk.files import get_file_records
        smiles_seen = set()
        smiles = []
        for attr_file in attrs_files:
            for molkey, name, val in get_file_records(attr_file, keep_header=False, select=(['std_smiles'], 1)):
                if val not in smiles_seen:
                    smiles_seen.add(val)
                    smiles.append(val)
                

        from FPSim2.io import create_db_file
        logger.info("Creating fingerprint database")
        from tqdm import tqdm
        tmp_db_fn = db_fn + '.tmp'
        create_db_file(
            ((smi, i) for i,smi in enumerate(tqdm(smiles))),
            tmp_db_fn,
            'Morgan', {'radius': cls.MorganRadius, 'nBits': cls.FingerprintBits}
            )

        import os
        os.replace(tmp_db_fn, db_fn)
        
        # For even faster lookups, FPSim makes it easy to compute a sparse similarity matrix at a given threshold.
        # Consider doing this if we find this to be a bottleneck.
        """
        logger.info("Precomputing sparse similarity matrix")
        from FPSim2 import FPSim2Engine
        fpe = FPSim2Engine(db_fn)
        import multiprocessing
        dist_matrix = fpe.symmetric_distance_matrix(min_thresh, n_workers=1 + multiprocessing.cpu_count()//2)
        sim_matrix = dist_matrix
        sim_matrix.data = 1.0 - dist_matrix.data
        sim_matrix.setdiag(1)
        from scipy import sparse
        sparse.save_npz(sim_fn, sim_matrix)
        """

        metadata = {
            'smiles': smiles,
        }
        import json
        with open(metadata_fn, 'w') as f:
            f.write(json.dumps(metadata))

    def __init__(self, choice, in_memory=True):
        from FPSim2 import FPSim2Engine
        import json

        fpdb_path, metadata_path = self.get_paths(choice)
        self.fpe = FPSim2Engine(fpdb_path, in_memory_fps=in_memory)
        with open(metadata_path) as f:
            self.metadata = json.loads(f.read())
        
        self.smiles = self.metadata['smiles']
        self.smiles_to_idx = {smiles:idx for idx, smiles in enumerate(self.smiles)}
    
    def get_paths(self, choice):
        from dtk.s3_cache import S3File
        fpdb_s3= S3File.get_versioned('similarity', choice, role='fpdb')
        fpdb_s3.fetch()

        metadata_s3= S3File.get_versioned('similarity', choice, role='struct_metadata')
        metadata_s3.fetch()
        return fpdb_s3.path(), metadata_s3.path()

    def similar_to_smiles(self, smiles, threshold):
        try:
            if self.fpe.in_memory_fps:
                idxs_and_sims = self.fpe.similarity(smiles, threshold)
            else:
                idxs_and_sims = self.fpe.on_disk_similarity(smiles, threshold)
            return [(self.smiles[idx], sim) for idx, sim in idxs_and_sims]
        except Exception as e:
            logger.warn(f"Error computing similarity for smiles {smiles}: {e}")
            return []

    def similar_to_key(self, key, threshold, pre_cmp):
        from collections import defaultdict
        import math
        out = defaultdict(lambda: (-math.inf, ''))
        for smiles in key:
            for sim_smiles, sim_val in self.similar_to_smiles(smiles, threshold):
                for output_id in pre_cmp.get(sim_smiles, []):
                    out[output_id] = max(out[output_id], (sim_val, sim_smiles))
        return dict(out)
    
    def ref_precompute(self, ref_to_key):
        return []

    def setup_output_map(self, agent_to_key, ref_to_key, ref_precompute):
        smiles_agents = []
        for agent, key in agent_to_key.items():
            for smiles in key:
                smiles_agents.append((smiles, agent))
        from dtk.data import MultiMap
        return MultiMap(smiles_agents).fwd_map()
        
class IndigoSim:
    """
    Similarity by structure.

    Similar concept to StructSim, but using the indigo package instead.
    I couldn't find any definitive documentation on the actual fingerprints being generated here, but
    the source is here (though it may not correspond to the version we're currently using):
    https://github.com/epam/Indigo/blob/master/core/indigo-core/molecule/src/molecule_fingerprint.cpp

    High level documentation here:
    https://lifescience.opensource.epam.com/indigo/api/index.html#fingerprints

    It appears to be combining a few different features into its hashes, including a tree-based enumeration
    of the molecule, as well as cataloging some arbitrary features of the molecules (e.g. >13C atoms, has P, has S, has rare atoms, etc.)

    """
    key_type = StructKey
    name = 'indigo'
    added_path = False
    def __init__(self):
        # TODO: Replace this with the pip package soon.
        # Just keeping as-is for now to reproduce existing results.
        if not self.added_path:
            from path_helper import PathHelper
            import sys
            import os
            sys.path.append(os.path.join(PathHelper.repos_root, 'moleculeSimilarity/fingerprint/indigo'))
            self.added_path=True
        import indigo
        self.indigo = indigo.Indigo()

    def similar_to_smiles(self, smiles, threshold, cmp_mols):
        import indigo
        out = []
        try:
            m1 = self.indigo.loadMolecule(smiles)
            fp1 = m1.fingerprint('sim')
            for sm, fp in cmp_mols:
                sim = self.indigo.similarity(fp1, fp, "tanimoto")
                if sim > threshold:
                    out.append((sm, sim))
        except indigo.IndigoException as e:
            logger.warn(f"Error computing similarity for smiles {smiles}: {e}")
        return out

    def similar_to_key(self, key, threshold, pre_cmp):
        key_to_outputs, smiles_to_fp = pre_cmp
        from collections import defaultdict
        import math
        out = defaultdict(lambda: (-math.inf, ''))
        for smiles in key:
            for sim_smiles, sim_val in self.similar_to_smiles(smiles, threshold, smiles_to_fp.items()):
                for output_id in key_to_outputs.get(sim_smiles, []):
                    out[output_id] = max(out[output_id], (sim_val, sim_smiles))
        return dict(out)
    
    def ref_precompute(self, ref_to_key):
        return []

    def setup_output_map(self, agent_to_key, ref_to_key, ref_precompute):
        smiles_agents = []
        smiles_to_fp = {}
        logger.info("Setting up indigo fingerprints")

        for agent, key in agent_to_key.items():
            for smiles in key:
                smiles_agents.append((smiles, agent))
                if smiles not in smiles_to_fp:
                    import indigo
                    try:
                        mol = self.indigo.loadMolecule(smiles)
                        fp = mol.fingerprint('sim')
                        smiles_to_fp[smiles] = fp
                    except indigo.IndigoException as e:
                        logger.warn(f"Error precomputing fingerprint for smiles {smiles}: {e}")

        logger.info("Done")
        from dtk.data import MultiMap
        return MultiMap(smiles_agents).fwd_map(), smiles_to_fp

def get_sim_methods():
    return [StructSim, DirectTargetSim, IndirectTargetSim, PathwaySim, IndigoSim, DiffusedTargetSim]

class MetaSim:
    def __init__(self, thresholds, sim_choice, ppi_choice, std_gene_list_set, d2ps_method, d2ps_threshold, in_memory=True):
        struct_sim = StructSim(sim_choice, in_memory=in_memory)
        dir_sim = DirectTargetSim()
        ind_sim = IndirectTargetSim(ppi_choice)
        pathway_sim = PathwaySim(ppi_choice, std_gene_list_set, d2ps_method, d2ps_threshold)
        indigo_sim = IndigoSim()
        pr_sim = DiffusedTargetSim(ppi_choice)
        self.thresholds = thresholds
        self.method_thresholds = {
            struct_sim: thresholds['rdkit'],
            dir_sim: thresholds['dirJac'],
            indigo_sim: thresholds['indigo'],
            ind_sim: thresholds['indJac'],
            pathway_sim: thresholds['pathway'],
            pr_sim: thresholds['prMax'],
        }

        # Remove anything >1.0.
        for name, thresh in list(self.method_thresholds.items()):
            if thresh > 1.0:
                logger.info(f"Removing method {name}, threshold is {thresh}")
                del self.method_thresholds[name]


    
    @classmethod
    def make_similarity_keys(cls, agents, methods, dpi_choice):
        from dtk.prot_map import DpiMapping
        dpi = DpiMapping(dpi_choice)
        key_types = set()
        for method in methods:
            key_types.add(method.key_type)
        
        keys_by_type = {}
        for key_type in key_types:
            if dpi.mapping_type() == 'uniprot':
                keys_by_type[key_type.__name__] = key_type.keys_for_uniprots(dpi_choice)
            else:
                keys_by_type[key_type.__name__] = key_type.keys_for_agents(agents, dpi_choice)
        
        return keys_by_type
    
    def precompute(self, ref_keys, methods):
        out = {}
        for method, threshold in self.method_thresholds.items():
            if method.name not in methods:
                continue
            key_type_name = method.key_type.__name__
            pre_cmp = method.ref_precompute(ref_keys[key_type_name])
            out[method.name] = pre_cmp
        return out
    
    def run_all_keys(self, ref_keys, agent_keys, methods, precomputed, cores):
        from dtk.parallel import pmap
        from dtk.features import SparseMatrixBuilder
        from collections import defaultdict
        out = defaultdict(lambda: SparseMatrixBuilder())
        out_extra = defaultdict(lambda: defaultdict(dict))
        for method, threshold in self.method_thresholds.items():
            if method.name not in methods:
                continue

            key_type_name = method.key_type.__name__
            keys = ref_keys[key_type_name].values()
            if method.name in precomputed:
                ref_pre_cmp = precomputed[method.name]
            else:
                ref_pre_cmp = method.ref_precompute(ref_keys[key_type_name])
            pre_cmp = method.setup_output_map(agent_keys[key_type_name], ref_keys[key_type_name], ref_pre_cmp)
            static_args = dict(threshold=threshold, pre_cmp=pre_cmp)
            data = list(pmap(method.similar_to_key, keys, static_args=static_args, num_cores=cores, progress=method.name))
            for ref_key, sim_results in zip(ref_keys[method.key_type.__name__].keys(), data):
                for ws_key, sim_val in sim_results.items():
                    if isinstance(sim_val, tuple):
                        extra = sim_val[1]
                        out_extra[method.name][ref_key][ws_key] = extra
                        sim_val = sim_val[0]
                    out[method.name].add(ref_key, ws_key, sim_val)
        
        import numpy as np
        return {key: val.to_wrapper(np.float32) for key, val in out.items()}, out_extra