from __future__ import print_function
from builtins import range
from dtk.subclass_registry import SubclassRegistry
import math
import numpy as np

#-------------------------------------------------------------------------------
# Statistical tools
#-------------------------------------------------------------------------------
def dot(v1,v2):
    return sum([x*y for x,y in zip(v1,v2)])
def selfdot(v):
    return dot(v,v)
def znorm(v):
    from dtk.scaling import LinearScaler
    from dtk.num import avg_sd
    avg,sd = avg_sd(v)
    if sd == 0:
        sd = 1.0 # avoid divide-by-zero if all values are the same
    ls = LinearScaler(1.0/sd,avg)
    return [ls.scale(x) for x in v]
def convolve(vec,kernel,center_offset):
    denom = sum(kernel)
    if denom:
        kernel = [x/float(denom) for x in kernel]
    import numpy as np
    end_trim = len(kernel)-center_offset-1
    return np.convolve(vec,kernel)[center_offset:-end_trim]
def mult_hyp_correct(plist, mthd = 'fdr_bh'):
    plist = no_zero_pvalues(plist)
    if len(plist) < 2:
        return plist
    import statsmodels.stats.multitest as smm
    corrected = smm.multipletests(plist, method=mthd)
    return corrected[1]
def entropy(prob_vec):
    import scipy.stats
    return scipy.stats.entropy(prob_vec,base=2)
"""
if False:
 def entropy(prob_vec):
    # above is equivalent
    total = float(sum(prob_vec))
    ent = 0
    for p in prob_vec:
        if p > 0:
            p = p/total
            ent -= p * math.log(p,2)
    return ent
"""

def no_zero_pvalues(plist):
    import sys
    return [p if p else sys.float_info.min for p in plist]

class TieTracker:
    def __init__(self):
        self.hits = 0
        self.start = None
    def bump(self):
        self.hits += 1
    def flush(self,i):
        if self.hits:
            # output ranks clustered in middle of tie range
            skip = i - self.start - self.hits
            base = self.start + skip//2
            result = list(range(base,base+self.hits))
        else:
            result = []
        self.hits = 0
        self.start = i
        return result

def fill_wsa_ordering(ordering, ws=None, ws_wsas=None, dpi=None):
    """Fills in implicit 0's for unscored WSAs in an ordering.

    This is important when computing metrics that care about the total number
    of items being considered.

    This also ensures that all KTs actually have an entry in the ordering.

    You can specify the underlying universe of WSAs either by passing in
    a ws_wsas list (which must be a superset of the ids in ordering) or
    by passing in a workspace and dpi name. Note that if you're calling
    this many times, calculating the ws_wsas list once externally and passing
    it in to each call may be much faster.
    """
    if ws_wsas is None:
        ws_wsas = ws.get_default_wsas(dpi).values_list('id', flat=True)
    found_wsas = set(x[0] for x in ordering)

    addon_score = 0
    if len(ordering) > 0:
        # Some of our CMs output negative scores (gpbr)
        # If the smallest score is negative,
        # we will use that instead of 0 for missing values.
        addon_score = min(ordering[-1][1], addon_score)

    addon = []
    for wsa in ws_wsas:
        if wsa not in found_wsas:
            addon.append((wsa, addon_score))
    return ordering + addon

def get_tie_adjusted_ranks(scores, is_kt):
    tt = TieTracker()
    result = []
    last = None
    for i,(value, is_kt) in enumerate(zip(scores, is_kt)):
        if value != last:
            result += tt.flush(i)
        last = value
        if is_kt:
            tt.bump()
    result += tt.flush(len(scores))
    return result

def ws_dpi_condense_keys(wsa_ids, dpi_name=None, dpi_t=0):
    """

    dpi_name: dpi file; if None, ws default
    dpi_t: dpi threshold to be included in condense key.  default is 0, which
        is unusual - but for most condensing purposes let's distinguish even
        if the DPI value typically isn't used.
    """
    if not wsa_ids:
        return []
    from browse.models import WsAnnotation
    ws = WsAnnotation.objects.get(pk=next(iter(wsa_ids))).ws

    dpi_name = dpi_name or ws.get_dpi_default()

    # Being a bit careful here because we could have multiple bindings
    # for the same prot for the same WSA.
    # This shouldn't be too common, but can happen when the clustering
    # changes and doesn't match up with the current DPI file.
    from collections import defaultdict
    wsaid_to_prot_ev = defaultdict(lambda: defaultdict(float))
    wsaid_to_prot_dir = defaultdict(lambda: defaultdict(int))

    from dtk.prot_map import DpiMapping
    dpi = DpiMapping(dpi_name)
    dpikey_to_wsaid = dpi.get_wsa_id_map(ws)
    from dtk.files import get_file_records
    for dpi_key, uniprot, ev, dr in get_file_records(dpi.get_path(),keep_header=False):

        ev = float(ev)
        dr = float(dr)
        if ev < dpi_t:
            continue

        for wsa_id in dpikey_to_wsaid.get(dpi_key, []):
            # Take max evidence, consensus direction.
            wsaid_to_prot_ev[wsa_id][uniprot] = max(wsaid_to_prot_ev[wsa_id][uniprot], ev)
            wsaid_to_prot_dir[wsa_id][uniprot] += dr

    wsaid_to_bindings = defaultdict(list)

    for wsa_id, ev_dict in wsaid_to_prot_ev.items():
        dir_dict = wsaid_to_prot_dir[wsa_id]
        for prot in ev_dict:
            dr = np.sign(dir_dict[prot])
            wsaid_to_bindings[wsa_id].append((prot, ev_dict[prot], dr))


    condense_keys = [
            tuple(sorted(wsaid_to_bindings[wsa_id]))
            for wsa_id in wsa_ids
            ]

    return condense_keys

def make_dpi_group_mapping(wsa_ids, key_func=ws_dpi_condense_keys):
    condense_keys = key_func(wsa_ids)

    wsa_to_groupid = {}
    key_to_groupid = {}

    for wsa_id, key in zip(wsa_ids, condense_keys):
        groupid = key_to_groupid.get(key, None)
        if groupid is None:
            groupid = len(key_to_groupid)
            key_to_groupid[key] = groupid

        wsa_to_groupid[wsa_id] = groupid

    import six
    groupid_to_key = {v:k for k, v in six.iteritems(key_to_groupid)}

    return wsa_to_groupid, groupid_to_key

def condense_ordering_with_mapping(ordering, wsa_to_group, required_wsa_ids=set(), condense_options=None):
    """Creates a condensed ordering given an ordering and a group mapping.

    This will contain only 1 entry per (group,unique score)
    Any required wsa_ids will also be kept, regardless of grouping.
    """
    condense_options = condense_options or {}
    out_vec = []

    from collections import defaultdict
    key_to_scoregroup = dict()
    wsa_to_scoregroup = defaultdict(list)

    for score_entry in ordering:
        wsa_id = score_entry[0]
        if wsa_id in required_wsa_ids:
            key = wsa_id
        else:
            if condense_options.get('ignore_score'):
                key = (wsa_to_group.get(wsa_id), )
            else:
                # If you gave us a wsa_to_group mapping without this wsa in it,
                # then its group is 'None' and it will be keyed purely on score.
                key = (wsa_to_group.get(wsa_id), score_entry[1])

        if key in key_to_scoregroup:
            wsa_to_scoregroup[wsa_id] = key_to_scoregroup[key]
            wsa_to_scoregroup[wsa_id].append(wsa_id)
            continue

        key_to_scoregroup[key] = [wsa_id]
        wsa_to_scoregroup[wsa_id] = key_to_scoregroup[key]

        out_vec.append(score_entry)
    return out_vec, wsa_to_scoregroup

def condense_emi(emi, wsa_to_group=None, key_func=ws_dpi_condense_keys, condense_options=None):
    score_vec = emi.get_labeled_score_vector()
    kt_set = emi.get_kt_set()
    if wsa_to_group is None:
        wsa_ids = [x[0] for x in score_vec]
        wsa_to_group, _ = make_dpi_group_mapping(wsa_ids, key_func)
    condensed, _ = condense_ordering_with_mapping(score_vec, wsa_to_group, kt_set, condense_options=condense_options)
    return EMInput(condensed, kt_set)


#-------------------------------------------------------------------------------
# Enrichment Metric Input
#-------------------------------------------------------------------------------
class EMInput:
    '''Provides input pre-processing shared by multiple Enrichment Metrics.
    '''
    def __init__(self,score,kt_set,id_to_group_map=None,fill_from_ws=None,condense_options=None):
        if fill_from_ws:
            score = fill_wsa_ordering(score, ws=fill_from_ws)
        self._score = score
        self._kt_set = kt_set
        self._id_to_group_map = id_to_group_map
        self._condensed = None
        self._condense_options = condense_options
    def get_condensed_emi(self):
        if self._condensed is None:
            self._condensed = condense_emi(self, self._id_to_group_map, condense_options=self._condense_options)
        return self._condensed
    def get_labeled_score_vector(self):
        return list(self._score)
    def get_unlabeled_score_vector(self):
        return [x[1] for x in self._score]
    def get_kt_flag_vector(self):
        return [x[0] in self._kt_set for x in self._score]
    def n_scores(self):
        return len(self._score)
    def get_kt_set(self):
        return set(self._kt_set)
    def n_kts(self):
        return len(self._kt_set)
    def get_raw_ranks(self):
        return [
                i
                for i,x in enumerate(self._score)
                if x[0] in self._kt_set
                ]
    def get_tie_adjusted_ranks(self):
        is_kt = [label in self._kt_set for label, value in self._score]
        scores = [value for label, value in self._score]
        return get_tie_adjusted_ranks(scores, is_kt)
    def get_hit_flag_vector(self):
        return [x[0] in self._kt_set for x in self._score]
    def get_hit_cdf(self):
        cdf=[]
        seen = 0
        from dtk.scores import get_ranked_groups
        for ahead,tied in get_ranked_groups(self._score):
            hits = len(set(tied) & self._kt_set)
            if hits:
                misses = len(tied) - hits
                before = misses//2
                cdf += [seen] * before
                for i in range(hits):
                    seen += 1
                    cdf.append(seen)
                cdf += [seen] * (misses-before)
            else:
                cdf += [seen] * len(tied)
        return cdf

class MetricProcessor:
    """Efficiently computes Enrichment Metrics.

    Enrichment metrics values are cached - if you request one that
    has already been generated, it will be served from the cache.

    Otherwise, it will recompute it and store it in the cache.

    Computing multiple metrics for the same workspace re-uses the same
    condense mapping.
    Computing multiple metrics for the same job & code re-uses the same
    ordering/EMInput.
    """

    def __init__(self):
        self.ws_condense_mappings = {}
        self.ws_wsas = {}
        self.job_code_emis = {}
        from dtk.cache import Cacher
        self.cache = Cacher('enrichment_metric')
    def get_mapping(self, ws_id, dpi_override=None):
        from browse.models import WsAnnotation
        cache_key = (ws_id, dpi_override)
        if cache_key not in self.ws_condense_mappings:
            if dpi_override:
                key_func = lambda x: ws_dpi_condense_keys(x, dpi_name=dpi_override)
            else:
                key_func = ws_dpi_condense_keys
            wsa_ids = WsAnnotation.objects.filter(ws=ws_id).values_list('id', flat=True)
            condense_mapping, _ = make_dpi_group_mapping(wsa_ids, key_func=key_func)
            self.ws_condense_mappings[cache_key] = condense_mapping
        return self.ws_condense_mappings[cache_key]

    def get_ws_wsaids(self, ws_id):
        if ws_id not in self.ws_wsas:
            from browse.models import Workspace
            qs = Workspace.objects.get(pk=ws_id).get_default_wsas()
            wsa_ids = qs.values_list('id', flat=True)
            self.ws_wsas[ws_id] = list(wsa_ids)
        return self.ws_wsas[ws_id]

    def get_emi(self, bji, code, ktset='wseval', dpi_override=None, condensed=False, **kwargs):
        import json
        key = (bji.job.id, code, ktset, str(dpi_override), condensed, frozenset(sorted((k, json.dumps(v)) for k, v in kwargs.items())))
        if key not in self.job_code_emis:
            ordering = bji.get_data_catalog().get_ordering(code, desc=True)
            if isinstance(ktset, str):
                kts = bji.ws.get_wsa_id_set(ktset)
            else:
                kts = ktset
            
            if condensed:
                condense_mapping = self.get_mapping(bji.ws.id, dpi_override)
            else:
                condense_mapping = None
            wsaids = self.get_ws_wsaids(bji.ws.id)
            ordering = fill_wsa_ordering(ordering, ws_wsas=wsaids, dpi=dpi_override)

            emi = EMInput(
                    ordering,
                    kts,
                    id_to_group_map=condense_mapping,
                    condense_options=kwargs.get('condense_options'),
                    )
            self.job_code_emis[key] = emi
        return self.job_code_emis[key]

    def compute(self, metric, ws_or_id, job_or_id, code, ktset='wseval', dpi_override=None, **kwargs):
        """Computes enrichment metrics, returning a new or cached results.

        ktset can be the name of a drugset or a list of wsa ids.
        """
        if not isinstance(ktset, str):
            # Canonicalize the ktset for caching.
            ktset = tuple(sorted(ktset))
        from runner.process_info import JobInfo
        bji = JobInfo.get_bound(ws_or_id, job_or_id)
        cache_key = f'{bji.ws.id}-{bji.job.id}-{code}-{metric}-{ktset}-{dpi_override}'
        if kwargs:
            cache_key += str(sorted(kwargs.items()))
        def compute_fn():
            EMClass = EnrichmentMetric.lookup(metric)
            em = EMClass()
            condensed = getattr(em, 'condensed', False)
            em.evaluate(self.get_emi(bji, code, ktset=ktset, dpi_override=dpi_override, condensed=condensed, **kwargs))
            return em.rating
        return self.cache.check(cache_key, compute_fn)


#-------------------------------------------------------------------------------
# EnrichmentMetrics
#-------------------------------------------------------------------------------
class EnrichmentMetric(SubclassRegistry):
    # this can be overridden if it's not appropriate for some metrics
    def is_close(self,v):
        return abs(self.rating-v)/self.rating < 0.1
    @classmethod
    def label(cls):
        return cls.__name__
    # label is occasionally overridden, so I added a duplicate that should stay as is
    def name(self):
        return type(self).__name__
    def evaluate(self,emi):
        '''Set self.rating to a float evaluating passed-in EMInput.

        Should be overridden by derived class
        '''
        raise NotImplementedError('evaluate not overridden')

def sklearn_parms(emi):
    return (
        emi.get_kt_flag_vector(),
        emi.get_unlabeled_score_vector(),
        )

def CondensedEnrichmentMetric(base_metric_class):
    class CondensedEM(base_metric_class):
        condensed = True
        def evaluate(self, emi):
            super().evaluate(emi.get_condensed_emi())
    return CondensedEM


class AUR(EnrichmentMetric):
    '''Area under ROC curve
    '''
    @classmethod
    def label(cls):
        return "AUROC"
    def evaluate(self,emi):
        if sum(emi.get_kt_flag_vector()) == 0:
            self.rating = 0
            return
        from sklearn.metrics import roc_auc_score
        self.rating = roc_auc_score(*sklearn_parms(emi))
    def plot(self,emi):
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(*sklearn_parms(emi))
        self.evaluate(emi)
        from tools import sci_fmt
        from dtk.plot import scatter2d, annotations
        return scatter2d(
                'False Positive Rate',
                'True Positive Rate',
                zip(fpr,tpr),
                text=['@'+sci_fmt(t) for t in thresholds],
                title=f'{type(self).__name__} curve',
                refline=False,
                linestyle='lines',
                annotations=annotations('AUC-ROC: %.3f' % (self.rating))
                )

class AURCondensed(CondensedEnrichmentMetric(AUR)):
    @classmethod
    def label(cls):
        return 'AUR Cnd'

class APS(EnrichmentMetric):
    '''Average Precision Score - area under Precision/Recall curve
    '''
    @classmethod
    def label(cls):
        return "Average Precision Score"
    def evaluate(self,emi):
        if sum(emi.get_kt_flag_vector()) == 0:
            self.rating = 0
            return
        from sklearn.metrics import average_precision_score
        self.rating = average_precision_score(*sklearn_parms(emi))
    def plot(self,emi):
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(
                                                *sklearn_parms(emi)
                                                )
        self.evaluate(emi)
        # the above returns an extra 1 at the end of precision, and 0
        # at the end of recall; they're stripped below
        from tools import sci_fmt
        from dtk.plot import scatter2d, annotations
        return scatter2d(
                'Recall',
                'Precision',
                list(zip(recall,precision))[:-1],
                text=['@'+sci_fmt(t) for t in thresholds],
                title='Precision-Recall curve',
                refline=False,
                linestyle='lines+markers',
                annotations=annotations('Avg. precision: %.3f' % (self.rating))
                )

# XXX This class has a non-standard ctor, so it can't be used in a completely
# XXX generic way via a name lookup. It's currently only called in one place,
# XXX which doesn't rely on the dpi parameter having a default, so I removed
# XXX it to reduce coupling.
class DPI_bg_corr(EnrichmentMetric):
    '''Correlate a score to the DPI-only scores.
       A strong disease signature should overcome this starting point
    '''
    def __init__(self, emi, wsa_id_map, dpi):
        super(DPI_bg_corr, self).__init__()
        from dtk.s3_cache import S3MiscBucket,S3File
        s3f=S3File(S3MiscBucket(),'dpi.%s.bg.tsv'%dpi)
        s3f.fetch()
        from dtk.files import get_file_records
        bg_score = {}
        for frs in get_file_records(s3f.path()):
            for wsa in wsa_id_map.get(frs[0], []):
                bg_score[wsa] = float(frs[1])
        self.xy = {tup[0]:(tup[1],
                          bg_score.get(tup[0], 0)
                         )
                   for tup
                   in emi.get_labeled_score_vector()
                  }
    def _evaluate(self):
        from dtk.num import corr
        self.correlation = corr(list(self.xy.values()), method='spearman')
        self.rating = 1 - self.correlation
    def evaluate(self):
        self._evaluate()
    def plot(self, ws_id):
        from dtk.plot import scatter2d, annotations
        from browse.models import Workspace
        if not hasattr(self,'correlation'):
            self._evaluate()
        ws = Workspace.objects.get(pk=ws_id)
        name_map = ws.get_wsa2name_map()
        names=[name_map[wsa] for wsa in self.xy.keys()]
        return scatter2d(
                'CM score',
                'DPI Bg. Score',
                list(self.xy.values()),
                logscale='yaxis',
                title='Real vs Bg score',
                refline=True,
                text=names,
                ids=('drugpage',list(self.xy.keys())),
                annotations=annotations('Bg. correlation: %.3f' % (self.correlation))
                )
import scripts.febe as febe
class FEBE(EnrichmentMetric):
    '''Fishers Exact Based Enrichment - Identifies the rank that is most enriched in KTs and the corresponding -log10Pval
    '''
    def evaluate(self,emi):
        runner = febe.febe(
                   named_scores = emi.get_labeled_score_vector(),
                   names_of_interest = emi.get_kt_set()
                 )
        runner.run()
        self.rating = runner.final_score

class wFEBE(EnrichmentMetric):
    '''Weighted Fishers Exact Based Enrichment -
       an extension of FEBE that rewards low-rank and high-portion of the ref. set enrichments
    '''
    def evaluate(self,emi):
        runner = febe.wfebe(
                   named_scores = emi.get_labeled_score_vector(),
                   names_of_interest = emi.get_kt_set()
                 )
        runner.run()
        self.rating = runner.final_score


import dea
class DEA_ES(EnrichmentMetric):
    '''DEA with the down steps including score weighting. Currently only trust the ES
    '''
    def evaluate(self,emi):
        dea.verbose = False
        runner = dea.Runner(
                    fhf=None,
                    nPermuts=0,
                    score_list = emi.get_labeled_score_vector(),
                    set_of_interest = emi.get_kt_set(),
                    alpha = 0.01
                    )
        runner.verbose = False
        try:
            runner.run()
#            runner.run(emi.get_labeled_score_vector(),emi.get_kt_set(),0.01)
            self.rating = runner.es # es
        except ValueError:
            # uncomment the following line for debuging
            # raise
            self.rating = 0

class DEA_AREA(EnrichmentMetric):
    '''The classic...
    '''
    def evaluate(self,emi):
        dea.verbose = False
        self.runner = dea.Runner(
                    fhf=None,
                    nPermuts=0,
                    weight=1.5,
                    score_list = emi.get_labeled_score_vector(),
                    set_of_interest = emi.get_kt_set(),
                    alpha = 0.01
                    )
        self.runner.verbose = False
### Shouldn't need to do the log any more b/c we've updated the score to already be scaled so that bigger is better
        try:
            self.runner.run()
#            runner.run(emi.get_labeled_score_vector(),emi.get_kt_set(),0.01)
            self.rating = self.runner.poly_area
            #print 'score_total', self.runner.score_total
            #print 'len(self.final_vector)', len(self.runner.final_vector)
            #print 'sum(runner.set_matches)', sum(self.runner.set_matches)
        except ValueError:
            # uncomment the following line for debuging
            # raise
            self.rating = 0
### Shouldn't need to do the log any more b/c we've updated the score to already be scaled so that bigger is better
#        before = self.rating
#        self.rating = 1.0/math.log(self.rating)
#        print before,self.rating
        if False:
          try:
            print(self.runner.leading_edge_cnt)
          except AttributeError:
            print("No leading edge cnt")
    def plot(self,emi,dtc='wsa', ids=None):
        if not hasattr(self,'runner'):
            self.evaluate(emi)
        return self.runner.dea_plotly(dtc=dtc, protids=ids)

# XXX For the normalized versions to work, you need to calculate a background,
# XXX and that code requires padding the ordering with zero scores for some
# XXX background population, but there's not really a way to specify here
# XXX what population is appropriate.  Since these NES things are sort of
# XXX on their way out anyway, I'm disabling them here.
if False:
 class DEA_NES(EnrichmentMetric):
    '''The classic...
    '''
    def evaluate(self,emi):
        dea.verbose = False
        runner = dea.Runner(
                    fhf=None,
                    score_list = emi.get_labeled_score_vector(),
                    set_of_interest = emi.get_kt_set(),
                    alpha = 0.01
                    )
        runner.verbose = False
        runner.run()
#        runner.run(emi.get_labeled_score_vector(),emi.get_kt_set(),0.01)
        self.rating = runner.nes

import old_dea
class OLD_DEA_ES(EnrichmentMetric):
    '''The classic...
    '''
    def evaluate(self,emi):
        old_dea.verbose = False
        runner = old_dea.Runner(
                    fhf=None,
                    nPermuts=0,
                    )
        runner.verbose = False
        try:
            runner.run(emi.get_labeled_score_vector(),emi.get_kt_set(),0.01)
            self.rating = runner.realResults[0] # es
        except ValueError:
            # uncomment the following line for debuging
            # raise
            self.rating = 0

# XXX see comment above on DEA_NES
if False:
 class OLD_DEA_NES(EnrichmentMetric):
    '''The classic...
    '''
    def evaluate(self,emi):
        old_dea.verbose = False
        runner = old_dea.Runner(
                    fhf=None,
                    )
        runner.verbose = False
        runner.run(emi.get_labeled_score_vector(),emi.get_kt_set(),0.01)
        self.rating = runner.nes_list[0] # nes



def score_rating(ordering, kts, target_index):
    target_index = min(target_index, len(ordering)-1)
    targ_score = ordering[target_index][1]

    N = len(ordering)
    kt_wt = 0
    other_wt = 0
    for i, (id, score) in enumerate(ordering):
        if id in kts:
            kt_wt += min(1.0, score / targ_score)

    return kt_wt / len(kts)

class SigmaOfRank(EnrichmentMetric):
    '''sum of sigma of KT ranks
    '''
    width = 100
    center = 200
    def _score(self,rank):
        from dtk.num import sigma
        return sigma((self.center-rank)/(0.5*self.width))
    def evaluate(self,emi):
        ranks = emi.get_tie_adjusted_ranks()
        n_kts = emi.n_kts()
        self.rating = sum(map(self._score,ranks))/n_kts if n_kts else 0.

    @classmethod
    def tf_eval_func(cls):
        from functools import partial
        from dtk.rank_gradient import sigma_of_rank
        return partial(sigma_of_rank, sigmoid_width=cls.width, sigmoid_center=cls.center)


class SigmaOfRankCondensed(CondensedEnrichmentMetric(SigmaOfRank)):
    @classmethod
    def label(cls):
        return 'SoR Cnd'

class wFEBECondensed(CondensedEnrichmentMetric(wFEBE)):
    pass

class SigmaOfRank1000(EnrichmentMetric):
    '''sum of sigma of KT ranks
    Think of this as the expected fraction of KTs that we would discover.
    For each rank, it gives the probability of us noticing a KT at that rank.
    Sigma1000 has updated parameters to more accurately reflect the current
    review process where typically get to roughly the 1000'th drug.

    https://www.wolframalpha.com/input/?i=sigmoid((800+-x)+%2F+(0.5+*+200))
    '''
    width = 200 # ~ranks between 0.75 and 0.25
    center = 800 # rank of 0.5 score

    @classmethod
    def label(cls):
        return 'SoR1000'
    def _score(self,rank):
        from dtk.num import sigma
        return sigma((self.center-rank)/(0.5*self.width))
    def evaluate(self,emi):
        ranks = emi.get_tie_adjusted_ranks()
        n_kts = emi.n_kts()
        self.rating = sum(map(self._score,ranks))/n_kts if n_kts else 0.

    @classmethod
    def tf_eval_func(cls):
        from functools import partial
        from dtk.rank_gradient import sigma_of_rank
        return partial(sigma_of_rank, sigmoid_width=cls.width, sigmoid_center=cls.center)


class SigmaOfRank1000Condensed(CondensedEnrichmentMetric(SigmaOfRank1000)):
    @classmethod
    def label(cls):
        return 'SoR1000Cnd'


class ScoreRating(EnrichmentMetric):
    def evaluate(self,emi):
        self.rating = score_rating(emi.get_labeled_score_vector(),
                                   emi.get_kt_set(),
                                   target_index=800
                                   )

class RankRating(EnrichmentMetric):
    def evaluate(self, emi):
        ranks = emi.get_tie_adjusted_ranks()
        import numpy as np
        if len(ranks) == 0:
            self.rating = 0
        else:
            self.rating = -np.mean(ranks)

class SigmaOfRank1000Continuous(EnrichmentMetric):
    def __init__(self):
        self.width = 200 # ~ranks between 0.75 and 0.25
        self.center = 800 # rank of 0.5 score
    def _score(self,rank):
        from dtk.num import sigma
        return sigma((self.center-rank)/(0.5*self.width))

    def evaluate(self,emi):
        import numpy as np
        ranks = emi.get_tie_adjusted_ranks()
        rating1 = sum(map(self._score,ranks))/emi.n_kts()
        rating2 = score_rating(emi.get_labeled_score_vector(),
                               emi.get_kt_set(),
                               target_index=self.center
                               )
        self.rating = rating1 + 0.01 * rating2


class SigmaPortion1000(EnrichmentMetric):
    """Roughly the portion of top 1000 items covered by the kt set.

    Weighted by a sigmoid to smooth out the boundary.
    """

    width = 200
    center = 1000
    def evaluate(self, emi):
        numer = 0
        denom = 0
        from dtk.num import sigma
        for i, is_kt in enumerate(emi.get_kt_flag_vector()):
            x = sigma((self.center - i) / (0.5 * self.width))

            denom += x
            numer += is_kt * x

        self.rating = numer / denom

class SigmaPortionWeighted1000(EnrichmentMetric):
    """Roughly the portion of total score in top 1000 items covered by the kt set.

    i.e. SigPor1000 additionally weighted by the score of each item.
    """
    width = 200
    center = 1000
    def evaluate(self, emi):
        numer = 0
        denom = 0
        from dtk.num import sigma
        data = zip(emi.get_kt_flag_vector(), emi.get_unlabeled_score_vector())
        for i, (is_kt, score) in enumerate(data):
            x = sigma((self.center - i) / (0.5 * self.width)) * score

            denom += x
            numer += is_kt * x

        self.rating = numer / denom


class SigmaPortion100(SigmaPortion1000):
    center = 100
    width = 20

class SigmaPortionWeighted100(SigmaPortionWeighted1000):
    center = 100
    width = 20

if False:
 # fails test_no_kt_scores, and not worth fixing
 class EcdfPoly(EnrichmentMetric):
    '''size of KT ECDF Polygon
    '''
    def evaluate(self,emi):
        def area_proxy(rank):
            #return rank * 1.0/emi.n_kts()
            # XXX currently, we take the log of the rank to de-emphasize the
            # XXX scores of higher-ranked drugs; this may be a little too
            # XXX aggressive; maybe there's a way to use sigmoid scaling
            # XXX or something else here instead
            return math.log(1+rank) * 1.0/emi.n_kts()
        ranks = emi.get_tie_adjusted_ranks()
        area = sum(map(area_proxy,ranks))
        missed = emi.n_kts() - len(ranks)
        area += missed * area_proxy(emi.n_scores())
        perfect = sum(map(area_proxy,list(range(emi.n_kts()))))
        self.rating = perfect/area

if False:
 # fails test_no_kt_scores, and not worth fixing
 class DensityCorr(EnrichmentMetric):
    '''correlation of score with KT density.
    This is not a great metric in and of itself, but can be used to
    see score curve effects on things with similar scores on other metrics.
    '''
    def __init__(self):
        self.weights = [
                math.exp(-abs(i)/10.0)
                for i in range(-20,21)
                ]
    def evaluate(self,emi):
        hits = emi.get_hit_flag_vector()
        margin = len(self.weights)/2
        score_only = emi.get_unlabeled_score_vector()
        density = convolve(hits,self.weights,len(self.weights)/2)
        # correlation assumes mean 0 (this makes almost no difference)
        score_only = znorm(score_only)
        density = znorm(density)
        # calculate correlation
        xx = selfdot(score_only)
        yy = selfdot(density)
        xy = dot(density,score_only)
        #print xx,yy,xy
        self.rating = abs(xy)/math.sqrt(xx*yy)

