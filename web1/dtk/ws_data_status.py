import numpy as np
from dtk.subclass_registry import SubclassRegistry
import os
import six
import logging
logger = logging.getLogger(__name__)

sig_mm = (200, 5000, 'log10')
omics_mm = (3,10, None)

class DataStatus(SubclassRegistry):
    """
    Different status types inherit from this.

    They are expected to provide a 'raw_scores' dict containing the raw score
    outputs, and a 'score_types' class method indicating which scores will be
    output and how to scale/normalize them.

    Additional user-visible messages can be appended to the messages list.
    Extra details about the scores can be appended to the details list.
    """
    ScoreType = 'score'
    def __init__(self, ws):
        self.ws = ws
        self.raw_scores = {}
        self.details = []
        self.messages = []
        self.setup()

    def scores(self):
        out = {}
        for score_name, minmax in self.score_types().items():
            out[score_name] = self._norm_score(self.raw_scores[score_name], minmax)
        return out

    @classmethod
    def _norm_score(cls, value, min_max_transform):
        minv, maxv, transform = min_max_transform
        known = ('log10', None, 'raw')
        assert transform in known, f'Unknown {transform}'

        if transform == 'raw':
            return value

        if transform == 'log10':
            minv = np.log10(minv)
            maxv = np.log10(maxv)
            value = np.log10(float(value))

        # Apply hackery to re-scale the 'sufficient' portion from 1 to 10 and
        # the 'insufficient' portion from 0 to 1.
        if value >= minv:
            out = np.clip((value - minv) / (maxv - minv), 0, 1)
            out *= 9
            out += 1
        else:
            out = value / minv
        return out

class Gwas(DataStatus):
    ScoreType = 'GWAS Datasets'
    @classmethod
    def score_types(cls):
        return {
                cls.ScoreType: omics_mm,
            }

    def setup(self):
        self.raw_scores = {
                self.ScoreType: len(self.ws.get_gwas_dataset_choices()),
                }


class GE(DataStatus):
    CC = 'Case/Control'
    miRNA = 'miRNA'
    @classmethod
    def score_types(cls):
        return {
                cls.CC: omics_mm,
                cls.miRNA: omics_mm,
            }

    def setup(self):
        self.raw_scores = {
                self.CC: float('nan'),
                self.miRNA: float('nan'),
                }
        from browse.models import TissueSet
        for ts in TissueSet.objects.filter(ws=self.ws):
            # XXX This maybe should be reorganized:
            # XXX - it assumes only CC and miRNA tissue sets should get scores
            # XXX   (and so only they should be used by default in a refresh
            # XXX   workflow)
            # XXX - it duplicates some of the logic in Tissue.ts_label,
            # XXX   although not the non-human part, so non-human tissue sets
            # XXX   are also disabled by default
            if ts.name == 'default' or ts.name == 'Case/Control':
                name = self.CC
            elif ts.name == self.miRNA:
                name = self.miRNA
            else:
                continue
            act_tiss_ids = set()
            for t in ts.tissue_set.all():
                over,_,_,_ = t.sig_result_counts()
                if over:
                    act_tiss_ids.add(t.id)
            self.raw_scores[name] = len(act_tiss_ids)

class OpenTargets(DataStatus):
    ScoreType = 'Integrated Target Data'
    @classmethod
    def score_types(cls):
        return {
                cls.ScoreType: sig_mm,
            }

    def setup(self):
        vdefaults = self.ws.get_versioned_file_defaults()
        self.ot_version = vdefaults['openTargets']
        from dtk.open_targets import OpenTargets
        self.ot = OpenTargets(self.ot_version)
        if self.get_key():
            self.extract_data()
            zipped_results = list(zip(self.score_names,
                                 self.score_name_non_zeros
                                ))
            zipped_results.sort(key = lambda t: t[1])
            # This is going to give 'overall' because it will sort highest.
            self.ot_val = zipped_results[-1][1]
        else:
            self.ot_val = float('nan')
        self.raw_scores = {
                self.ScoreType: self.ot_val
                }

    def get_key(self):
        names = self.ws.get_disease_default('OpenTargets').split(',')
        key_prefix = 'key:'
        self.keys = []
        for name in names:
            if name[:len(key_prefix)].lower() == key_prefix:
                key = name[len(key_prefix):]
                self.keys.append(key)
                if not self.ot.check_key(key):
                    self.messages += ["Key '%s' is not present in OpenTargets." % key]
                    return False
        if not self.keys:
            return False
        else:
            return True

    def extract_data(self):
        from collections import defaultdict
        # scan master file, looking for matches to disease
        d = defaultdict(lambda: defaultdict(float))
        for key in self.keys:
            cur = self.ot.get_disease_scores(key)
            for uniprot, name_to_value in six.iteritems(cur):
                for name, value in six.iteritems(name_to_value):
                    d[uniprot][name] = max(d[uniprot][name], value)
        # find all columns with at least one non-zero score
        populated = set()
        for score in d.values():
            populated |= set(score.keys())
        # write out any matches
        score_names = [
                'overall',
                'literature',
                'rna_expression',
                'somatic_mutation',
                'genetic_association',
                'known_drug',
                'animal_model',
                'affected_pathway'
                ]
        score_name_non_zeros = [0 for i in score_names]
        score_name_zeros_or_NAN = [0 for i in score_names]
        for k,v in six.iteritems(d):
            for i,score in enumerate(score_names):
                if score in v and v[score] > 0.:
                    score_name_non_zeros[i] += 1
                else:
                    score_name_zeros_or_NAN[i] += 1
        self.rec = d
        self.score_name_non_zeros = score_name_non_zeros
        self.score_name_zeros_or_NAN = score_name_zeros_or_NAN
        score_names = [
                'Open Targets Total (%s)'%self.ot_version,
                '--Literature',
                '--RNA Expression',
                '--Somatic Mutation',
                '--Genetic Association',
                '--Known Drug',
                '--Animal Model',
                '--Affected Pathway'
                ]
        self.score_names = score_names
        self.found = len(d)

class AvailableTargets(DataStatus):
    ScoreType = 'Available Targets'
    ScoreTypeRaw = 'Available Targets (Raw %)'
    @classmethod
    def score_types(cls):
        return {
                cls.ScoreType: (0.95, 1.0, None),
                cls.ScoreTypeRaw: (0, 1, 'raw'),
            }

    def setup(self):
        _, _, druggable_por = self.get_druggable_target_count()
        self.raw_scores = {
                self.ScoreType: druggable_por,
                self.ScoreTypeRaw: druggable_por * 100,
                }

    def get_druggable_target_count(self):
        from browse.models import ProtSet, Protein
        from browse.utils import get_dpi_or_sm_druggable_prots
        try:
            ws = self.ws
            all_prots = get_dpi_or_sm_druggable_prots()
            bad_prots = ws.get_uniprot_set('autops_wsunwanted')

            p2g = Protein.get_uniprot_gene_map(all_prots | bad_prots)

            bad_genes = set(p2g.get(p,p) for p in bad_prots)
            all_genes = set(p2g.get(p,p) for p in all_prots)
            druggable_target_count = len(all_genes - bad_genes)
            all_target_count = len(all_genes)
            por = druggable_target_count/ all_target_count

            if ws.get_nonnovel_ps_default() == ProtSet.default_nonnovel_protset:
                logger.warning(f'{ws.name} has no nonnovel protset setup')
                raise ValueError()

        except ValueError:
            druggable_target_count = None
            por = float('nan')
        return druggable_target_count, all_target_count, por

class DisGeNet(DataStatus):
    ScoreType = 'DisGeNET Values'
    @classmethod
    def score_types(cls):
        return {
                cls.ScoreType: (20, 500, 'log10'),
            }

    def setup(self):
        f_choices = self.ws.get_prev_job_choices('dgn')
        if f_choices:
            f_initial = f_choices[0][0]
            from runner.process_info import JobInfo
            cat = JobInfo.get_bound(self.ws,
                                    f_initial).get_data_catalog()
            self.dgn_val = len([x for x
                                  in cat.get_ordering('dgns', True)
                                  if x[1] != 0
                                 ])
        else:
            self.dgn_val = float('nan')
            self.messages += ["DGN data not available (If applicable, go to Run -> DisGeNet)"]

        self.raw_scores = {self.ScoreType: self.dgn_val}

class AGR(DataStatus):
    ScoreType = 'AGR Values'
    @classmethod
    def score_types(cls):
        return {
                cls.ScoreType: (20, 500, 'log10'),
            }

    def setup(self):
        f_choices = self.ws.get_prev_job_choices('agr')
        if f_choices:
            f_initial = f_choices[0][0]
            from runner.process_info import JobInfo
            cat = JobInfo.get_bound(self.ws,
                                    f_initial).get_data_catalog()
            self.agr_val = len([x for x
                                  in cat.get_ordering('agrs', True)
                                  if x[1] != 0
                                 ])
        else:
            self.agr_val = float('nan')
            self.messages += ["AGR data not available (If applicable, go to Run -> AGR)"]

        self.raw_scores = {self.ScoreType: self.agr_val}

class Monarch(DataStatus):
    ScoreType = 'Monarch Initiative Values'
    @classmethod
    def score_types(cls):
        return {
                cls.ScoreType: (20, 500, 'log10'),
            }

    def setup(self):
        f_choices = self.ws.get_prev_job_choices('misig')
        if f_choices:
            f_initial = f_choices[0][0]
            from runner.process_info import JobInfo
            cat = JobInfo.get_bound(self.ws,
                                    f_initial).get_data_catalog()
            self.misig_val = len([x for x
                                  in cat.get_ordering('misig', True)
                                  if x[1] != 0
                                 ])
        else:
            self.misig_val = float('nan')
            self.messages += ["Monarch Initiative data not available (If applicable, go to Run -> Phenotype -> MISig)"]
        self.raw_scores = {self.ScoreType: self.misig_val}

class TcgaMut(DataStatus):
    ScoreType = 'Tumor Mutation Sigs'

    @classmethod
    def score_types(cls):
        return {
                cls.ScoreType: sig_mm,
            }
    def setup(self):
        self.tcgam_val = float('nan')
        f_choices = self.ws.get_prev_job_choices('tcgamut')
        if f_choices:
            f_initial = f_choices[0][0]
            from runner.process_info import JobInfo
            cat = JobInfo.get_bound(
                      self.ws,f_initial).get_data_catalog()
            self.tcgam_val = len([x for x
                                  in cat.get_ordering('mutpor',
                                       True)
                                  if x[1] != 0
                                 ])
        else:
            self.messages += ["TCGA Data not avaliable (If applicable, go to Run->TCGAMut)"]

        self.raw_scores = {
                self.ScoreType: self.tcgam_val
                }



class Faers(DataStatus):
    ClinicalType = 'Clinical Values'
    CompleteClinicalType = 'Complete Clinical Values'

    @classmethod
    def score_types(cls):
        return {
                cls.ClinicalType: (500, 5000, 'log10'),
                cls.CompleteClinicalType: (200, 1500, 'log10'),
            }

    def get_latest_faers_job_id_by_cds(self,cds):
        # Like ws.get_prev_job_choices, but hard-wired to FAERS jobs,
        # and pulling only a particular CDS
        from runner.process_info import JobInfo
        ubi = JobInfo.get_unbound('faers')
        names = ubi.get_jobnames(self.ws)
        from runner.models import Process
        l = Process.objects.filter(
                name__in=names,
                status=Process.status_vals.SUCCEEDED,
                settings_json__contains=f'"cds": "{cds}.v',
                ).order_by('-id').values_list('id',flat=True)[:1]
        return l[0] if l else None

    def setup(self):
        self.faers_val = float('nan')
        self.target_f_v = float('nan')
        error = False
        # Note this is hardwired to the FAERS CDS. If we re-enable
        # other CDSs in the future it should be extended.
        faers_job_id = self.get_latest_faers_job_id_by_cds(cds='faers')
        if faers_job_id:
            from runner.process_info import JobInfo
            bji = JobInfo.get_bound(self.ws, faers_job_id)
            bji.fetch_lts_data()
            pkl_location = bji.important_stats
            error_message = "Data not avaliable (Go to Run->FAERS)"
            try:
                with open(pkl_location, 'rb') as handle:
                    import pickle
                    # Because these were pickled via cPickle in py2,
                    # the encoding is apparently latin1.
                    statsdict = pickle.load(handle, encoding='latin1')
            except IOError:
                statsdict = {}
                error = True
            self.faers_val = statsdict.get('total', float('nan'))
            self.details += [('FAERS Disease Total (most recent run',
                                 self.faers_val,
                                 True
                                )]

            self.details += [('FAERS Search Terms (most recent run)',
                                 ', '.join(statsdict.get('search_term', [])),
                                 False
                                )]
            self.target_f_v = statsdict.get('target_sum', float('nan'))
            self.details += [('FAERS Full Demographics (most recent run)',
                                 self.target_f_v, False
                                )]
        else:
            self.messages += ['No successful FAERS run detected']
            self.details += [('FAERS Disease Total',
                                 'No successful FAERS run detected',
                                  True
                                 )]

        if error:
            self.messages += [error_message]
        self.raw_scores = {
                self.ClinicalType: self.faers_val,
                self.CompleteClinicalType: self.target_f_v,
                }

    # this is currently unused
    def _extract_faers(self):
        from dtk.faers import ClinicalEventCounts
        self.indi_set = self.ws.get_disease_default('FAERS').split('|')
        cec=ClinicalEventCounts(self.ws.get_cds_default())
        try:
            # get portion data
            (
                    bg_ctr,
                    self.bg_total,
                    self.indi_ctr,
                    self.disease_total,
                    )=cec.get_drug_portions(self.indi_set)
        except ValueError:
            self.disease_total=float('nan')
            self.messages += ['No FAERS data for '+repr(self.indi_set)]


def get_score_status_types(include_avail_targets=True):
    types = []
    if include_avail_targets:
        types.append(('AvailableTargets', 'ScoreTypeRaw', None))
    # ClassName, ScoreAttrName, WeightName
    types += [
        ('DisGeNet', 'ScoreType', 'dgn'),
        ('AGR', 'ScoreType', 'agr'),
        ('TcgaMut', 'ScoreType', 'tcgamut'),
        ('Monarch', 'ScoreType', 'misig'),
        ('Faers', 'CompleteClinicalType', 'faers'),
        ('Gwas', 'ScoreType', 'gwas'),
        ('GE', 'CC', 'cc'),
        ('OpenTargets', 'ScoreType', 'otarg'),
    ]
    return types


def compute_suitabilities(weight_ws_ids, disp_ws_ids, include_avail_targets=True):
    from browse.models import Workspace
    types = get_score_status_types(include_avail_targets)
    keys = set([x[2] for x in types])

    from collections import defaultdict
    raw_scores = defaultdict(dict)
    norm_scores = defaultdict(dict)

    import dtk.retrospective as retro
    workspaces = retro.filter_workspaces(weight_ws_ids)
    workspaces = sorted(workspaces, key=retro.ws_review_date_key)
    weights = compute_suitability_weights(workspaces, keys)

    disp_workspaces = Workspace.objects.filter(pk__in=list(disp_ws_ids) + list(weight_ws_ids))

    # retrieve mesh disease data using global version default
    from browse.default_settings import mesh
    from dtk.mesh import MeshDiseases
    md = MeshDiseases(mesh.value(None))
    for ws in disp_workspaces:
        ws_total = 0
        for typename, scoreattrname, weightname in types:
            Class = DataStatus.lookup(typename)
            inst = Class(ws)
            scorename = getattr(inst, scoreattrname)
            norm_score = inst.scores()[scorename]
            norm_scores[ws][scorename] = norm_score
            raw_scores[ws][scorename] = inst.raw_scores[scorename]
            import math
### Note that log2'ing the scores, but not the weights, and then 2**'ing the product
### ends up giving a non-linear boost to the important scores.
### This is a heuristic anyhow, so it's fine for now, but worth considering in the future
            if not math.isnan(norm_score) and norm_score >= 1.:
                ws_total += weights[weightname] * math.log2(norm_score+1.)
            norm_scores['weights'][scorename] = weights[weightname]
        norm_scores[ws]['Combined'] = 2**ws_total - 1.
        link = ws.reverse('data_status')
        norm_scores[ws]['name'] = f'<a href="{link}">{ws.name}</a>'
        norm_scores[ws]['short_name'] = ws.get_short_name()
        # Try to find a disease name category. First, we need a mesh term
        # for the disease name. Start with the ClinicalTrials name, which
        # is MeSH-based. Note that the ClinicalTrials name may actually be
        # multiple pipe-separated names
        ct_names,dd = ws.get_disease_default(
                'ClinicalTrials',
                return_detail=True,
                )
        matches = [md.match(part) for part in ct_names.split('|')]
        terms = [term for term,how in matches if term is not None]
        if dd and not terms:
            # if that doesn't work, and it wasn't already a fallback name,
            # try falling back to the workspace name
            term,how = md.match(ws.name)
            terms = [term]
        if terms:
            # now, get any corresponding categories
            cats = set()
            for term in terms:
                cats |= md.base_name_categories[term]
            cat_label = ', '.join(sorted(cats))
        else:
            cat_label = 'No MeSH match'
        norm_scores[ws]['category'] = cat_label
        raw_scores[ws]['name'] = ws.name


    norm_scores['weights']['Combined'] = np.sum(list(norm_scores['weights'].values()))
    norm_scores['weights']['name'] = '* Weights'
    norm_scores['weights']['short_name'] = ''
    norm_scores['weights']['category'] = ''

    return types, raw_scores, norm_scores

def compute_suitability_weights(workspaces, keys):
    def is_defus_faers(out):
        return 'defus' in out or 'weightedzscore' in out or 'drtarget' in out or 'drstruct' in out or 'wzs' in out

    gwas_types = set(['gpath', 'esga', 'gwasig'])
    dgn_types = set(['dgn', 'dgns'])

    def grouper(x):
        if is_defus_faers(x[0]):
            return 'faers'
        if len(x) >= 2 and x[1] == 'otarg':
            return 'otarg'
        elif x[0] in gwas_types:
            return 'gwas'
        elif x[0] in dgn_types:
            return 'dgn'
        else:
            return x[0]

    from collections import defaultdict
    import dtk.retrospective as retro
    fns, cats = retro.make_category_funcs(
            workspaces,
            retro.unreplaced_selected_mols,
            lambda *args: retro.mol_score_imps(*args, score_imp_group_fn=grouper),
            )

    key_sums = defaultdict(int)
    for fn, cat in zip(fns, cats):
        if cat in keys:
            for ws in workspaces:
                key_sums[cat] += fn(ws)
    import numpy as np
    norm_factor = np.sum(list(key_sums.values()))
    for cat in list(key_sums.keys()):
        key_sums[cat] /= norm_factor

    return key_sums
