from dtk.files import get_file_records


def score_mips(ph2g_score, d2ph_score):
    score_to_try = 1 # this didn't really change much, so I went with the more intuitive scoring method
    if score_to_try == 1: # multiply
        return ph2g_score * d2ph_score
    if score_to_try == 2: # mean
        return (ph2g_score + d2ph_score)/2.
    if score_to_try == 3: # square the pheno to gene
        return ph2g_score**2.0 * d2ph_score


class Monarch(object):
    def __init__(self, version_defaults, ws):
        self.version_defaults = version_defaults
        self.ws=ws
        self.evid_score_to_try = 1 # (1 or 2) When all others were held at 1, this had little impact and no consistent impact
        self.combine_score_to_try = 2 # (1-3) When all others were held at 1, this had little impact, but 2 was slightly better
    def _get_fp(self,role):
        file_class = 'monarch'
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned(
                file_class,
                self.version_defaults[file_class],
                role,
                )
        s3f.fetch()
        return s3f.path()

    def _set_score_evid(self):
        score_to_try =self.evid_score_to_try
# this evidence ontology is described here and I consulted this site to come up with these scores
# https://evidenceontology.org/
        me = MonarchEvid(self.version_defaults)
    # increasingly stringent
        if score_to_try == 1:
            score_d={
                     'author statement supported by traceable reference used in manual assertion':0.9,
                     'combinatorial evidence used in automatic assertion': 0.9,
                     'experimental evidence used in manual assertion': 0.7,
                     'sequencing assay evidence': 0.6,
                     'evidence used in automatic assertion': 0.5,
                     'inference from background scientific knowledge used in manual assertion': 0.5,
                     'imported manually asserted information used in automatic assertion': 0.3,
                     'imported automatically asserted information used in automatic assertion': 0.2,
                     'genomic context evidence': 0.1,
                     }
        elif score_to_try == 2:
            score_d={
                     'combinatorial evidence used in automatic assertion': 0.9,
                     'author statement supported by traceable reference used in manual assertion':0.8,
                     'experimental evidence used in manual assertion': 0.6,
                     'sequencing assay evidence': 0.5,
                     'evidence used in automatic assertion': 0.3,
                     'inference from background scientific knowledge used in manual assertion': 0.3,
                     'imported manually asserted information used in automatic assertion': 0.1,
                     'imported automatically asserted information used in automatic assertion': 0.1,
                     'genomic context evidence': 0.0,
                     }
        return {me.desc2code[k]:v for k,v in score_d.items()}

    def _handle_evid_scores(self,evidence_code, score_evid_d):
#XXX We should take into accoutn if there are multiple pieces of evidence (and likely) reward those
        all_ecs = evidence_code.split("|")
        all_scores=[]
        for ec in all_ecs:
            if ec in score_evid_d:
                all_scores.append(score_evid_d[ec])
            else:
# so far this only has happened when the ec is empty, but leaving it in here just in case
                print(f'Missing evidence code: {ec}')
        return max(all_scores+[0])

    def _set_score_func(self):
        score_to_try=self.combine_score_to_try
        if score_to_try == 1: # multiply
            def func(val1, val2):
                return val1 * val2
        elif score_to_try == 2: # average
            def func(val1, val2):
                return (val1 + val2) / 2.
        elif score_to_try == 3: # freq raised to 1-evidence
            def func(val1, val2):
                return val1**(1-val2)
        self.score_func = func

class MonarchEvid(object):
    def __init__(self, version_defaults):
        self.version_defaults = version_defaults
        file_class = 'monarch'
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned(
                file_class,
                self.version_defaults[file_class],
                'evidence',
                )
        s3f.fetch()
        RowType=None
        self.code2desc={}
        self.desc2code={}
        for fields in get_file_records(
                    s3f.path(),
                    keep_header=True
            ):
            if not RowType:
                from collections import namedtuple
                RowType=namedtuple('Monarch',fields)
                continue
            rec = RowType(*fields)
            self.desc2code[rec.evidence_label]=rec.evidence_code
            self.code2desc[rec.evidence_code]=rec.evidence_label


class MonarchGene(Monarch):
    def _get_file_path(self):
        return self._get_fp('gene')

    def get_mips_data(self,phenos):
        from dtk.files import get_file_records
        from collections import defaultdict
        # pheno: gene: score
        mips_data=defaultdict(lambda: defaultdict(float))
        self._setup_scoring()
        RowType=None
        patterns= phenos
        for fields in get_file_records(
                self._get_file_path(),
                keep_header=True,
                select=(patterns,1),
                ):
            if not RowType:
                from collections import namedtuple
                RowType=namedtuple('Monarch',fields)
                continue
            rec = RowType(*fields)
            score = self._score_ph2g_evidence(rec)
            if rec.pheno_id in mips_data:
                if rec.uniprot in mips_data[rec.pheno_id]:
                    # these duplicates seem to be arising from differing evidence codes, seems reasonable to just take the max
                    # Not sure why these aren't split up in the first place
                    score=max([mips_data[rec.pheno_id][rec.uniprot], score])
            mips_data[rec.pheno_id][rec.uniprot]=score
        return mips_data

    def _score_ph2g_evidence(self,rec):
        return self.score_func(self.score_relation_d[rec.relation_type],
                               self._handle_evid_scores(rec.evidence_code,
                                                        self.score_evid_d
                               ))

    def _setup_scoring(self):
        self.score_relation_d = self._set_score_relation()
        self.score_evid_d = self._set_score_evid()
        self._set_score_func()

    def _set_score_relation(self, score_to_try=1):
        if score_to_try == 1:
            score_d={
                     'has phenotype':1.0,
                     'contributes to condition':0.5
                     }
        return score_d

class MonarchDis(Monarch):
    def _get_file_path(self):
        return self._get_fp('disease')
    def get_mips_data(self, pattern_string=None):
        from dtk.files import get_file_records
        from collections import defaultdict
        # pheno: disease: score
        mips_data=defaultdict(lambda: defaultdict(float))
        phenoIDs_to_names={}
        self._setup_scoring()
        RowType=None
        if pattern_string is None:
            patterns=self.ws.get_disease_default('Monarch').split('|')
        else:
            patterns=pattern_string.split('|')
        for fields in get_file_records(
                self._get_file_path(),
                keep_header=True,
                select=(patterns,0),
                ):
            if not RowType:
                from collections import namedtuple
                RowType=namedtuple('Monarch',fields)
                continue
            rec = RowType(*fields)
            if rec.pheno_id in mips_data:
                assert rec.mondo_id not in mips_data[rec.pheno_id], "Duplicate disease, pheno pair"
            mips_data[rec.pheno_id][rec.mondo_id]=self._score_d2ph_evidence(rec)
            phenoIDs_to_names[rec.pheno_id]=rec.phenotype
# ordering format: [(phenoID1, score), (phenoID2, score)...]
        return [(pheno, max(mips_data[pheno].values())) for pheno in mips_data], phenoIDs_to_names

    def _score_d2ph_evidence(self,rec):
        if rec.frequency in self.score_freq_d:
            freq_score=self.score_freq_d[rec.frequency]
        else:
            freq_score=self.score_freq_d['aria_default']
        return self.score_func(freq_score, self._handle_evid_scores(rec.evidence_code, self.score_evid_d))

    def _setup_scoring(self):
# when holding all others at 1, this made essentially no change, though 1 was occasionally 0.001 better
        freq_score_to_try = 1
        self.score_freq_d = self._set_score_freq(freq_score_to_try)
        self.score_evid_d = self._set_score_evid()
        self._set_score_func()

    def _set_score_freq(self, score_to_try=1):
    # increasingly stringent
        if score_to_try == 1:
            score_d={'Obligate':1.0,
                     'Very frequent': 0.9,
                     'Frequent': 0.75,
                     'Occasional': 0.5,
                     'Very rare': 0.1,
# I.e. the score when there is no frequency listed
                     'aria_default':0.25
                     }
        elif score_to_try == 2:
            score_d={'Obligate':1.0,
                     'Very frequent': 0.9,
                     'Frequent': 0.75,
                     'Occasional': 0.5,
                     'Very rare': 0.1,
                     'aria_default':0
                     }
        elif score_to_try == 3:
            score_d={'Obligate':1.0,
                     'Very frequent': 0.8,
                     'Frequent': 0.5,
                     'Occasional': 0.3,
                     'Very rare': 0.1,
                     'aria_default':0
                     }
        return score_d




