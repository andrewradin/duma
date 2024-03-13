
import django_setup
from dtk.tsv_alt import SqliteSv
import os
import logging
from tqdm import tqdm
from collections import namedtuple
logger = logging.getLogger(__name__)

def moa_indirect(moa, ppi):
    from collections import defaultdict
    expanded_moa = defaultdict(float)
    for prot, evid, direc in moa:
        expanded_moa[prot] = max(expanded_moa[prot], evid)
        adj = ppi[prot]
        for adjprot, edge_evid in adj:
            score = evid * edge_evid
            expanded_moa[adjprot] = max(expanded_moa[adjprot], score)
    return [(prot, evid, 0) for prot, evid in expanded_moa.items()]


class MolToPathway:
    def __init__(self, geneset, ppi_choice, method):
        self.method = method
        from dtk.gene_sets import get_gene_set_file
        gmt_s3f=get_gene_set_file(geneset)
        gmt_s3f.fetch()
        self.pathway_prots = {}
        from dtk.files import get_file_records
        for c in get_file_records(gmt_s3f.path(), parse_type='tsv', keep_header=True):
            name = c[0]
            prots = c[1].split(",")
            self.pathway_prots[name] = set(prots)

        from dtk.prot_map import PpiMapping
        ppi = PpiMapping(ppi_choice)
        from collections import defaultdict
        all_ppi = defaultdict(list)
        for p1, p2, evid, direc in ppi.get_data_records(min_evid=0.9):
            all_ppi[p1].append((p2, evid))
        self.all_ppi = all_ppi


    def _moa_direct_portion(self, pathway):
        #moa_id, pwy, score, direction = c
        output = []
        targ_prots = self.pathway_prots[pathway]
        for moa in self.moas:
            evid_in = 0
            for prot, evid, direc in moa:
                if prot in targ_prots:
                    evid_in += evid
            score = evid_in / len(targ_prots)
            output.append([str(moa), pathway, score, 0])
        return output

    @classmethod
    def _moa_indirect_portion(cls, moa, pathway_prots, ppi):
        #moa_id, pwy, score, direction = c
        ind_moa = moa_indirect(moa, ppi)
        output = []
        for pathway, targ_prots in pathway_prots.items():
            evid_in = 0
            for prot, evid, direc in ind_moa:
                if prot in targ_prots:
                    evid_in += evid
            if evid_in > 0:
                # Don't output 0's, makes file huge.
                score = evid_in / len(targ_prots)
                output.append([str(moa), pathway, score, 0])
        return output

    @classmethod
    def _moa_ora(cls, moa, pathway_prots, ppi):
        ind_moa = moa_indirect(moa, ppi)
        output = []
        ind_prots = set([prot for prot, evid, direc in ind_moa])
        all_prots = set(ppi.keys())

        #        mol    !mol
        # path
        # !path

        from fisher import pvalue
        import numpy as np
        for pathway, path_prots in pathway_prots.items():
            path_mol = len(ind_prots & path_prots)
            path_notmol = len(path_prots - ind_prots)
            notpath_mol = len(ind_prots - path_prots)
            # This is much faster than actually constructing the sets and taking the length.
            notpath_notmol = len(all_prots) - len(path_prots) - len(ind_prots) + path_mol
            # We're only interested in the right tail (greater pathway/mol overlap than you'd expect by chance).
            pv = pvalue(path_mol, path_notmol, notpath_mol, notpath_notmol).right_tail
            if pv == 0:
                # The smallest floating point value larger than 0.
                pv = np.nextafter(0, 1)

            score = -np.log10(pv)

            if score > 2:
                output.append([str(moa), pathway, score, 0])
        return output

    def compute_indirect_portion(self, moas):
        from dtk.parallel import pmap
        moa_pwy_scores = []
        data_gen = pmap(self._moa_indirect_portion, moas, static_args={'pathway_prots': self.pathway_prots, 'ppi': self.all_ppi})

        for results in tqdm(data_gen, total=len(moas)):
            moa_pwy_scores.extend(results)
        return moa_pwy_scores

    def compute_ora(self, moas):
        from dtk.parallel import pmap
        moa_pwy_scores = []
        data_gen = pmap(self._moa_ora, moas, static_args={'pathway_prots': self.pathway_prots, 'ppi': self.all_ppi})

        for results in tqdm(data_gen, total=len(moas)):
            moa_pwy_scores.extend(results)
        return moa_pwy_scores

    def compute_indirect_binary(self, moas):
        por_scores = self.compute_indirect_portion(moas)
        import math
        return [(moa, pathway, math.ceil(evid), 0) for moa, pathway, evid, dr in por_scores]

    def compute(self, moas):
        if self.method == 'm2p_indirect_por':
            return self.compute_indirect_portion(moas)
        elif self.method == 'm2p_indirect_binary':
            return self.compute_indirect_binary(moas)
        elif self.method == 'm2p_ora':
            return self.compute_ora(moas)
        raise Exception("Unimplemented" + self.method)



def make_moa_dpi_graph(moas):
    import networkx as nx
    from scripts.connect_drugs_to_proteinSets import establish_prefixes
    drug_prefix, _, prot_prefix = establish_prefixes()
    g = nx.DiGraph()
    for moa in moas:
        moa_name = str(moa)
        for (prot, ev, direction) in moa:
            g.add_edge(drug_prefix + moa_name,
                       prot_prefix + prot.replace("_", ""),
                       weight = float(ev),
                       direction = int(direction),
                      )
    return g


def compute_d2ps(ppi, ps, moas, method):
    if method.startswith('m2p_'):
        m2p = MolToPathway(ps, ppi, method)
        return m2p.compute(moas)
    else:
        from dtk.s3_cache import S3File
        from dtk.gene_sets import get_gene_set_file
        gmt_s3f=get_gene_set_file(ps)
        gmt_s3f.fetch()
        from tempfile import NamedTemporaryFile
        from scripts.connect_drugs_to_proteinSets import d2ps
        moa_dpi_graph = make_moa_dpi_graph(moas)

        output = []
        with NamedTemporaryFile() as f:
            runner = d2ps(ppi=ppi, gmt=gmt_s3f.path(),
                          dpi_graph=moa_dpi_graph, out=f.name,
                          methods=[method])
            runner.run()

            from dtk.files import get_file_records
            # TODO: this isn't using the tempfile we created, oh well.
            for c in get_file_records(f.name + '.tsv.gz', keep_header=False):
                output.append(c)
                #moa_id, pwy, score, direction = c
        return output


class MoA(tuple):
    """((uniprot, evid, dir), ...)"""
    def __new__(cls, bindings_or_str):
        """Initialize from a json str, tuple/list or MoA object.

        Just a thin wrapper around a tuple.
        Since tuple is an immutable type we have to use the __new__ override.
        """
        def format_binding(binding):
            assert len(binding) == 3, f"Should be prot, ev, dir; got {binding}"
            return (binding[0], float(binding[1]), float(binding[2]))
        if isinstance(bindings_or_str, MoA):
            data = bindings_or_str
        elif isinstance(bindings_or_str, str):
            import json
            data = tuple(sorted(format_binding(x) for x in json.loads(bindings_or_str)))
        else:
            data = sorted(format_binding(binding) for binding in bindings_or_str)
        return super().__new__(cls, data)

    def __str__(self):
        import json
        return json.dumps(self)

class D2ps:
    internal_version=2
    default_method = 'm2p_indirect_por'
    known_methods = ['enrich_score', 'enrich_score_v2', 'm2p_indirect_por', 'm2p_indirect_binary', 'm2p_ora']
    enabled_method_choices = (
            ('enrich_score_v2', 'Enrich Score'),
            ('m2p_indirect_por', 'Indirect Portion'),
            ('m2p_indirect_binary', 'Indirect Binary'),
            ('m2p_ora', 'Over Representation'),
            )

    def __init__(self, ppi_id, ps_id, method=None, cachedir=None):
        method = method or self.default_method
        assert method in self.known_methods
        assert ppi_id and ps_id, f"Unexpected empty {ppi_id}, {ps_id}"
        from path_helper import PathHelper, make_directory
        cachedir = cachedir or PathHelper.d2ps_cache
        make_directory(cachedir)
        self.fn = os.path.join(cachedir, f'd2ps_{ppi_id}_{ps_id}_{method}.{self.internal_version}.sqlsv')
        logger.debug("Using d2ps cache file at %s", self.fn)

        self.ppi_id = ppi_id
        self.ps_id = ps_id
        self.method = method

        # Double-checked locking, it's very rare that the file doesn't
        # already exist.
        if not os.path.exists(self.fn):
            with self.db_lock():
                if not os.path.exists(self.fn):
                    self._write([])
        self.sv = SqliteSv(self.fn)

    def all_moas(self):
        records = self.sv.get_records(
                columns=['moa'],
                unique=True)
        return set(MoA(x[0]) for x in records)

    MoAFormat = namedtuple("MoAPathway", "moa pathway score direction")
    def get_moa_pathway_scores(self, moa):
        data = list(self.sv.get_records(filter__moa__eq=str(MoA(moa))))
        if len(data) != 1:
            raise Exception(f'No precomputed data for {moa}, you should call update first')
        moa, pathway_json = data[0]
        import json
        pathways = json.loads(pathway_json)
        return [self.MoAFormat(moa, pwname, float(score), float(dr)) for pwname, score, dr in pathways]
    def db_lock(self):
        from dtk.lock import FLock
        return FLock(self.fn+'.lock')
    def compute_lock(self):
        from dtk.lock import FLock
        return FLock(self.fn+'.compute.lock')
    def update_for_moas(self, moas):
        self._update_for_moas(moas)
    def _update_for_moas(self, moas):
        new_moas = set(MoA(x) for x in moas)
        with self.db_lock():
            missing_moas = new_moas - set(self.all_moas())
        if missing_moas:
            logger.info(f'Found {len(missing_moas)} new MoAs to compute d2ps for')
            from contextlib import ExitStack

            with ExitStack() as context:
                # Only lock if we're doing a huge recompute, usually means new version of pathways file,
                # want to prevent multiple depends from multiple jobs recomputing at the same time and running
                # out of memory.  Smaller than 500 and it's pretty fast to just compute it anyway.
                if len(missing_moas) > 500:
                    context.enter_context(self.compute_lock())
                    # Re-count what we need to do, in case it changed before we got the lock.
                    with self.db_lock():
                        missing_moas = new_moas - set(self.all_moas())

                logger.info(f'Computing d2ps for {len(missing_moas)} new MoAs')
                new_d2ps = compute_d2ps(self.ppi_id, self.ps_id, missing_moas, method=self.method)
                moa_tuples = [(moa, (pathway, float(score), float(direction))) for moa, pathway, score, direction in new_d2ps]
                # Clearing out things as we go in here, this is quite memory hungry and we end up repeating a lot.
                new_d2ps = None
                from dtk.data import MultiMap
                moa2paths = MultiMap(moa_tuples).fwd_map()
                moa_tuples = None

                # Done compute, now take the db lock before writing.
                context.enter_context(self.db_lock())

                # Recompute missing after holding the lock; this ensures we don't insert duplicates.
                still_missing_moas = new_moas - set(self.all_moas())
                new_records = []
                import json
                for moa in still_missing_moas:
                    moa_pathways = json.dumps(list(moa2paths.get(str(moa), [])))
                    new_records.append([str(moa), moa_pathways])
                moa2paths = None
                logger.info(f'Adding {len(new_records)} moa->pathways to cache')
                self.sv.insert(new_records)
                logger.info(f'MoAs inserted')

    def _write(self, records):
        header = ['moa', 'pathway_data']
        types = (str, str)
        SqliteSv.write_from_data(self.fn, records, types, header, index=['moa'])


