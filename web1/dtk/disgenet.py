import os
from path_helper import PathHelper

def score_cmap(disgenet_score, faers_score):
    from math import log
    return disgenet_score**2.0 * faers_score

class DisGeNet(object):
    def __init__(self, version_defaults, ontology='meddra'):
        self.ontology = ontology
        self.version_defaults = version_defaults
        self._load_data()
    def _get_file_path(self,role):
        file_class = 'disgenet'
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned(
                file_class,
                self.version_defaults[file_class],
                role,
                )
        s3f.fetch()
        return s3f.path()
    def _load_data(self):
        self.data = {}
        if self.ontology == 'meddra':
            self._load_converter()
        else:
            assert self.ontology == 'umls'
        path=self._get_file_path('curated_'+self.ontology)
        from dtk.files import get_file_records
        for frs in get_file_records(path, keep_header=False):
            k=self._process_k(frs[0])
            try:
                self.data[k][frs[1]] = float(frs[2])
            except KeyError:
                self.data[k] = {frs[1]:float(frs[2])}
    def _process_k(self, k):
        if self.ontology == 'meddra':
            return self.converter[k].lower()
        return k
    def _load_converter(self):
        names_file=self._get_file_path('disease_names')
        from dtk.files import get_file_records
        self.converter = {x[0]:x[1]
                          for x in get_file_records(
                                       names_file,
                                       parse_type = 'tsv',
                                       keep_header = False
                                   )
                        }
    def get_cmap_data(self,ordering,score_func=score_cmap):
        from dtk.files import safe_name
        cmap_data = {}
        for indi,score in ordering:
            k = safe_name(indi)
            cmap_data[k] = {}
            for p in self.data.get(indi,{}):
                cmap_data[k][p] = score_func(self.data[indi][p], score)
        return cmap_data
    

    def get_disease_sim(self, disease, other_diseases, ppi):
        from dtk.prot_map import PpiMapping
        from collections import defaultdict
        ppi_obj = PpiMapping(ppi)
        ppi_data=defaultdict(lambda: defaultdict(float))
        gen = (frs for frs in ppi_obj.get_data_records(min_evid=0.9))
        for frs in gen:
            ppi_data[frs[0]][frs[1]] = frs[2]
        
        def make_ind(targs):
            out = defaultdict(float)
            for targ, val in targs.items():
                out[targ] += val
                # Divide by N to reduce the imapact of promiscuous targets.
                N = len(ppi_data[targ])
                for ind_targ, ev in ppi_data[targ].items():
                    out[ind_targ] += ev * val / N
            return out
        
        def jacc_float(a, b):
            num = 0
            den = 0
            for k in a.keys() | b.keys():
                bv = b.get(k, 0)
                av = a.get(k, 0)
                num += min(av, bv)
                den += max(av, bv)
            return num / den

        from dtk.similarity import calc_jaccard
        all_diseases = list(other_diseases) + [disease]

        prim_targs = dict(self.data[disease])
        prim_ind_targs = make_ind(prim_targs)

        out = {}
        for dis in all_diseases:
            other = self.data.get(dis, None)
            if other is None:
                continue
            
            other_ind = make_ind(other)
            #jac = calc_jaccard(prim_targs, other.keys())
            #ind_jac = calc_jaccard(prim_ind_targs, other_ind)
            jac = jacc_float(prim_targs, other)
            ind_jac = jacc_float(prim_ind_targs, other_ind)

            out[dis] = jac, ind_jac
            
        return out


