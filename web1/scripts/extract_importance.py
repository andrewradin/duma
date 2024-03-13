#!/usr/bin/env python3

import sys

from path_helper import PathHelper
import django_setup

import os

from browse.models import WsAnnotation

import logging
verbose=False
logger = logging.getLogger(__name__)
from runner.process_info import JobInfo

from dtk.subclass_registry import SubclassRegistry

class TargetImportanceDataLoader:
    """Fetches data used for computing target importance for a particular CM."""
    def __init__(self, ws_id, bji_or_id, wsa_id, target_id, cache, key_exclude=None, max_input_prots=None, **kwargs):
        self.ws_id = ws_id
        self.target_id = target_id
        self.cache = cache
        from runner.process_info import JobInfo
        if isinstance(bji_or_id, JobInfo):
            self.bji = bji_or_id
            self.job_id = self.bji.job.id
        else:
            self.job_id = bji_or_id
            self.bji = JobInfo.get_bound(ws_id,self.job_id)
        self.key_exclude = key_exclude or []
        from browse.models import WsAnnotation, Workspace
        self.default_dpi = Workspace.objects.get(pk = ws_id).get_dpi_default()
        if wsa_id is not None:
            self.drug_ws = WsAnnotation.objects.get(pk = wsa_id)
        self.parms = self.bji.parms
        self._max_input_prots = max_input_prots

    def fetch(self):
        """Returns a dict with all the information needed for this CM."""
        raise NotImplementedError()

    def predict_duration(self):
        """Outputs a number predicting how long this will take to run.

        The value is used for relative ordering when running in parallel.

        Tasks with larger outputs will be run first.  This doesn't have to
        be super accurate, but a rough approximation does much better than
        random in keeping utilization high.
        """
        from dtk.prot_map import DpiMapping
        cache_key = ('predict_duration', self.drug_ws.id)
        def compute():
            return len(DpiMapping(self.default_dpi).get_dpi_info(self.drug_ws.agent, min_evid=0.5))
        prediction = self.cache.get_or_compute(cache_key, compute)
        return prediction

# After finding the top N prots, we stop caring about peeling off
# the biggest each time.
DEFAULT_LOOSE_PEEL = 10

class TargetImportance(SubclassRegistry):
    # Derived class must implement:
    # - setup(data) - to do any setup, given the loader data
    # - get_score_set() - to retrieve one set of scores, possibly excluding
    #     a passed-in list of targets
    # - remaining_targets() - return remaining candidates for exclusion,
    #     after excluding an optional passed-in list of targets
    def __init__(self, wsa_id, data, cache, key_exclude=None, **kwargs):
        self.wsa_id = wsa_id
        self.cache = cache
        self.key_exclude = key_exclude or []
        self.loose_peel_after = kwargs.get('loose_peel_after', DEFAULT_LOOSE_PEEL)
        self.setup(data)
        self.scores_to_report = []
        from collections import defaultdict
        self.importance_scores = defaultdict(dict)

    def leave_one_out(self):
        self.score_col = None
        from collections import OrderedDict
        self.altered_scores = OrderedDict()
        self.orig_scores = self._name_scores(
                        self.get_score_set()
                        )
        for target in self.remaining_targets():
            self.altered_scores[target] = self._name_scores(
                            self.get_score_set(exclude=[target])
                            )
        self.scores_to_report = self.score_names

    def leave_one_in(self):
        self.score_col = None
        from collections import OrderedDict
        self.altered_scores = OrderedDict()
        all_targets = set(self.remaining_targets())
        for target in all_targets:
            # Only keep the one target.
            excludes = all_targets - set([target])
            self.altered_scores[target] = self._name_scores(
                            self.get_score_set(exclude=excludes)
                            )
        self.scores_to_report = self.score_names

    def _try_peel_group(self,score_col,group_size,prior_exclude,remaining):
        import itertools
        if (self.loose_peel_after is not None and
                len(prior_exclude) >= self.loose_peel_after):
            # If we've already peeled off the K largest things, now we're just
            # going to start peeling them off in arbitrary order, to speed up.
            remaining = [next(iter(remaining))]
        labeled_scores = [
                [target]+self.get_score_set(exclude=prior_exclude+target)
                for target in itertools.combinations(remaining,group_size)
                ]
        return self._general_try_peel_group(labeled_scores,
                                            score_col,
                                            group_size,
                                            prior_exclude,
                                            remaining)
    def _try_opposite_peel_group(self,score_col,group_size,prior_exclude,remaining):


        import itertools
        labeled_scores = [
                [target]+self.get_score_set(exclude=prior_exclude+tuple(set(remaining)-set(target)))
                for target in itertools.combinations(remaining,group_size)
                ]
        return self._general_try_peel_group(labeled_scores,
                                            score_col,
                                            group_size,
                                            prior_exclude,
                                            remaining)
    def _general_try_peel_group(self,labeled_scores,score_col,group_size,prior_exclude,remaining):
        labeled_scores.sort(key=lambda x:x[score_col+1])
        score_only = [x[score_col+1] for x in labeled_scores]
        n_tied=len([x for x in score_only if x == score_only[0]])
        log_label = ','.join(prior_exclude)
        max_log=100
        if len(log_label) > max_log:
            log_label = "..." + log_label[-max_log:]
        if False: # pragma: no cover
            # This logger line might be useful for debugging, but is very
            # spammy and will fill up the disk with the target importance CM.
            logger.info(
                    "%d prior %s group %d results %d n_tied %d max %f",
                    len(prior_exclude),
                    log_label,
                    group_size,
                    len(score_only),
                    n_tied,
                    score_only[0],
                    )
        return labeled_scores,n_tied
    def _next_peel_tier(self,score_col,prior_exclude,remaining):
        group_size=1
        while True:
            labeled_scores,n_tied = self._try_peel_group(
                        score_col,
                        group_size,
                        prior_exclude,
                        remaining,
                        )
            # n_tied will be 0 if there are no scores, 1 if there's a
            # single target removal that caused the most damage, and
            # 2 or more if multiple target removals tie for the most
            # amount of damage.
            if n_tied == 1:
                # found a single best (set of) targets; return
                return labeled_scores[0][0],labeled_scores[0][1:]
            if n_tied < len(labeled_scores):
                # some, but not all, cases tied; assume the union of the
                # tied targets are all equal
                targ_set = set()
                for x in labeled_scores[:n_tied]:
                    targ_set |= set(x[0])
                return targ_set,labeled_scores[0][1:]
            # all the targets tied; this can happen if the top tier
            # is a set of redundant targets, such that removing any
            # one will not affect the score.  In this case, we want
            # to try again with pairs of targets (or triples, etc.)
            # until we find a set that stands out.
            #
            # but before we do that, we'll try leave all BUT one out
            # this only make sense to do the first time.
            # If leaving one out gives the same score as leaving all but one out,
            # then we're never going to converge, so just bail.
            # This almost only happens when the original score was 0 (i.e. Null)
            if group_size == 1:
                op_labeled_scores,op_n_tied = self._try_opposite_peel_group(
                        score_col,
                        group_size,
                        prior_exclude,
                        remaining,
                        )
                if ([x[score_col+1]
                     for x in op_labeled_scores
                    ] == [
                     x[score_col+1]
                     for x in labeled_scores
                    ]
                    and n_tied == op_n_tied
                   ):
                    logger.warning('aborting because we will never converge')
                    # They're all equally good, drop the end score to 0 and attribute to all prots,
                    # which divides remaining weight equally.
                    targ_set = set()
                    for x in labeled_scores[:n_tied]:
                        targ_set |= set(x[0])
                    return targ_set,[0 for x in labeled_scores[0][1:]]
            # now continue on
            group_size += 1
            # Unfortunately, the number of combinations can grow rapidly
            # if there are lots of potential targets remaining.  So, we
            # implement some protection.  The original code used fixed cutoffs
            # for the number of iterations and the group size, but to be
            # effective if 'remaining' was large, these needed to be set
            # very low.
            #
            # Rather than a fixed cutoff, we pre-calculate the number of
            # combinations that will result from the next group size, and bail
            # if that's too large.  This lets us pursue larger group sizes
            # when the total number of targets is smaller.  If triggered,
            # the error return below will throw away any targets not yet
            # rated (so, some data will be missing).
            next_size = (
                (len(labeled_scores) * (len(remaining)-group_size))
                /group_size
                )
            if verbose:
                logger.info("Up group size to %d, next_size %d", group_size, next_size)
            if next_size > 100000:
                logger.warning('aborting to avoid iter size %d',next_size)
                # XXX for now, throw an exception rather than returning
                # XXX so we can get a feel for when/if this happens
                raise NotImplementedError('too many iterations to handle')
                return None,None
                # XXX if this turns out to be a problem, there are lots
                # XXX of additional things we could try to do to address it:
                # XXX - we could remove zero-/low-scoring targets from
                # XXX   the 'remaining' list, which reduces the number
                # XXX   of combinations.
                # XXX   Update on this idea: in test cases I've seen
                # XXX   this doesn't help b/c most of the scores are
                # XXX   currently tied at 0
                # XXX - since the worst case is when group_size is half
                # XXX   of len(remaining), we could try searching from the
                # XXX   other side (group_size = len(remaining)-1); if this
                # XXX   turns up a matching score, we can skip all the
                # XXX   iterations in the middle
                # note this last one is now done above to check either end
    def peel_one_off(self,score):
        from collections import OrderedDict
        self.altered_scores = OrderedDict()
        score_col = self.score_names.index(score)
        self.score_col = score_col
        self.orig_scores = self._name_scores(
                        self.get_score_set()
                        )
        if verbose:
            logger.info("starting peel_one_off wsa %s score %s orig %s",
                self.wsa_id,
                score,
                repr(self.orig_scores),
                )
        # peel off proteins for as long as we can
        prior_exclude = ()
        while True:
            # quit if no more targets
            remaining = self.remaining_targets(prior_exclude)
            if not remaining:
                break
            targets,scores = self._next_peel_tier(
                            score_col,
                            prior_exclude,
                            remaining,
                            )
            if targets == None:
                # we hit the combinatorial protection code; bail
                break
            # store targets on this tier
            prior_exclude = prior_exclude + tuple(targets)
            iteration_key = ','.join(prior_exclude)
            self.altered_scores[iteration_key] = self._name_scores(scores)
            # quit if we've reached zero score
            if not scores[score_col]:
                break
        self.scores_to_report.append(score)

    @classmethod
    def prep_scores(cls, importance_scores, scores_to_report, prefix=""):
        return {prefix+n:{p:importance_scores[p][n]
                          for p in importance_scores.keys()
                          if n in importance_scores[p]
                         }
                for n in scores_to_report
               }

    def _prep_scores(self, prefix=""):
        return TargetImportance.prep_scores(self.importance_scores, self.scores_to_report, prefix)

    def report_scores(self,ofile):
        self._prep_scores()
        with open(ofile, 'w') as f:
            f.write("\t".join(['key'] + self.score_names) + "\n")
            for p in self.importance_scores.keys():
                scores = [self.importance_scores[p][n] for n in self.scores_to_report]
                if any(scores):
                    f.write("\t".join([p] + [str(x) for x in scores]) + '\n')
    def score_importance(self, score_name = None, scoring_method = 'cumulative'):
        if score_name:
            self._score_peel(score_name, scoring_method)
        elif scoring_method == 'LOI':
            self._score_loi()
        else:
            self._score_loo()
    def _score_peel(self, score_name, method):
        orig = self.orig_scores[score_name]
        latest = orig
        contributions = {}
        seenK = []
        # The following will deliver scores in descending order,
        # since self.altered_scores is an OrderedDict
        for k in self.altered_scores:
            score = self.altered_scores[k][score_name]
            allK = k.split(",")
            assert allK[:len(seenK)] == seenK
            newK = allK[len(seenK):]
            seenK = allK
            if method == 'cumulative':
                v = float(latest - score)
                l = seenK
                v /= float(len(l))
            elif method == 'previous':
                v = latest
                l = newK
            for x in l:
                try:
                    contributions[x] += v
                except KeyError:
                    contributions[x] = v
            latest = score
        denom = float(sum(contributions.values()))
        tiers = {} # for logging only
        for k,v in contributions.items():
            val = v/denom if denom > 0 else 0.0
            # XXX for scores that can be negative, denom can be near zero;
            # XXX the line above guards against a divide-by-zero error
            # XXX (not necessarily in the best way), but a near-zero
            # XXX denom can create a huge val.  Limit it to 2, but preserve
            # XXX the sign.  This is temporary -- the case of signed scores
            # XXX needs to be completely re-thought.
            if abs(val)>2:
                val = 2 * val/abs(val)
            self.importance_scores[k][score_name] = val
            tiers.setdefault(val,set()).add(k)
        if False: # pragma: no cover
            # generate log
            # This is very spammy and will fill up the disk on worker, but
            # could still be useful for debugging if you want to enable it.
            for val in sorted(list(tiers.keys()),reverse=True):
                ks = tiers[val]
                if len(ks) > 5:
                    label = "%d targets" % len(ks)
                else:
                    label = ','.join(sorted(ks))
                logger.info('@ %f got %s',val,label)
    def _score_loo(self):
        from collections import defaultdict
        self.importance_scores = defaultdict(dict)
        for score_name in self.score_names:
            orig = self.orig_scores[score_name]
            for k in self.altered_scores.keys():
                # this is mostly for all zero cases, but I just generalized it for ease
                # I had to add the explicit orig == 0 for some of the direction scores
                if orig == 0.0 or orig == self.altered_scores[k][score_name]:
                    v = 0.0
                else:
                    v = float(orig - self.altered_scores[k][score_name]) / orig
                self.importance_scores[k][score_name] = v
    def _score_loi(self):
        # 'Max' scoring  replicates the peel-cumulative style behavior using
        # LOI per-prot scores.
        # Turning on because we're using this only for depend right now, which
        # does use max-style scoring.
        USE_MAX_SCORING = True

        from collections import defaultdict
        if USE_MAX_SCORING:
            from functools import partial
            # Using partial instead of lambda because it pickles.
            self.importance_scores = defaultdict(partial(defaultdict,float))
        else:
            self.importance_scores = defaultdict(dict)
        for score_name in self.score_names:
            denom = 1e-16
            if USE_MAX_SCORING:
                ordering = [[k, self.altered_scores[k][score_name]]
                             for k in self.altered_scores]
                ordering.sort(key=lambda x: -x[1])

                # If we have negative scores, shift everything to positive.
                if len(ordering) > 0 and ordering[-1][1] < 0:
                    for entry in ordering:
                        entry[1] += -ordering[-1][1]

                # Compute importance similarly to peel_cumulative, where each
                # 'section' of score gets attributed to all prots >= that value.
                for i in range(len(ordering)):
                    next_val = ordering[i+1][1] if i+1 < len(ordering) else 0
                    val = ordering[i][1] - next_val
                    adjust = val / (i+1)
                    denom += val
                    for prev in ordering[:i+1]:
                        self.importance_scores[prev[0]][score_name] += adjust

                for k in self.altered_scores.keys():
                    self.importance_scores[k][score_name] /= denom
                    if self.importance_scores[k][score_name] <= 1e-4:
                        del self.importance_scores[k][score_name]
            else:
                for k in self.altered_scores.keys():
                    v = float(self.altered_scores[k][score_name])
                    denom += v
                    self.importance_scores[k][score_name] = v
                # Use denom to normalize to 1.
                for k in self.altered_scores.keys():
                    self.importance_scores[k][score_name] /= denom
                    if self.importance_scores[k][score_name] <= 1e-4:
                        del self.importance_scores[k][score_name]

        # Convert from defaultdict back to dict for serialization, and drop any empties.
        self.importance_scores = {k:dict(v) for (k,v) in self.importance_scores.items() if v}

    def _name_scores(self, l):
        return dict(zip(self.score_names, l))

class FileWrapper(object):
    """Wraps input files that need to copied over for remote execution.

    All of these objects in the input data will be found, have their file
    marked for copying over to the remote host, and have the filename instance
    updated to point at the correct location for execution on the remote host.
    """
    def __init__(self, filename):
        self.filename = filename

    def __repr__(self):
        return "FileWrapper(%s)" % self.filename

class path_loader(TargetImportanceDataLoader):
    def fetch(self):
        settings = self.bji.job.settings()
        ts = self.bji.get_target_set()
        if ts is None:
            raise IOError('target set unavailable')
        target_key = self.bji.get_target_key(self.drug_ws)
        if target_key is None:
            raise IOError('target key unavailable')
        return {
            'settings': settings,
            'target': (FileWrapper(ts.filename), target_key)
        }

class path(TargetImportance):
    cm='path'
    loader=path_loader
    def _make_metric(self,ts,score):
        tup = self._return_metrics(score, ts)
        if len(tup) == 2:
            return tup
        k,w,col = tup
        from collections import namedtuple
        ScoreInfo=namedtuple('ScoreInfo','name rectype weights target_col')
        info = ScoreInfo(score,k,w,ts.columns[k].index(col))
        weights = [float(self.settings[x+'_w']) for x in info.weights.split()]
        from algorithms.bulk_pathsum import evidence_accumulator
        return (
                {k:info.target_col},
                evidence_accumulator(ts,{info.rectype:weights}),
                )
    def _return_metrics(self,score,ts):
        if score == 'direct':
            return (3, 't2p p2d', 't2p:protein')
        elif score == 'indirect':
            return (4, 't2p p2p p2d', 'p2p:prot1')
        elif score == 'direction':
            return self._make_direction_metric(ts, use_abs=False)
        elif score == 'absdir':
            return self._make_direction_metric(ts, use_abs=True)
        else:
            raise NotImplementedError()
    def _make_direction_metric(self,ts,use_abs):
        from algorithms.bulk_pathsum import direction_accumulator
        d={
            3:ts.colsets('protein',[3])[3][0],
            4:ts.colsets('prot1',[4])[4][0],
            }

        class AbsDirWrapper:
            def __init__(self,accum):
                self.accum = accum
            def __getattr__(self,attr):
                return getattr(self.accum,attr)
            def _final(self):
                return abs(self.accum._final())

        if use_abs:
            accum = AbsDirWrapper(direction_accumulator(ts,[3,4]))
        else:
            accum = direction_accumulator(ts, [3,4])
        return (d, accum)
    def setup(self, data):
        self.settings = data['settings']
        ts_file,target_name = data['target']
        from algorithms.pathsum2 import TargetSet
        ts = TargetSet()
        ts.load_from_file(ts_file.filename)
        if ts is None:
            raise IOError('target detail unavailable')
        self.target = ts.get_target(target_name)
        self._set_names()
        self.metrics=[self._make_metric(ts,x) for x in self.score_names]
        self._all_targets = set([
                row[target_col[k]]
                for target_col,m in self.metrics
                for k in m.score_map
                for row in self.target.paths.get(k,[])
                ])
        self._row_cache = {}
    def _set_names(self):
        self.score_names=['direct']
        if self.settings.get('p2p_file'):
            self.score_names.append('indirect')
        self.score_names.append('direction')
        self.score_names.append('absdir')
    def get_score_set(self,exclude=[]):
        exclude = set(exclude)
        result = []
        for target_col_map,m in self.metrics:
            # modified from Accumulator.score():
            m._reset()
            for k,v in m.score_map.items():
                target_col = target_col_map[k]
                cache_key = k
                row_data = self._row_cache.get(cache_key, None)
                if row_data is None:
                    row_data = []
                    for row in self.target.paths.get(k,[]):
                        vec = [float(row[col]) for col in v]
                        val = m._score(k,vec)
                        row_data.append((row, val))
                    self._row_cache[cache_key] = row_data

                for row, val in row_data:
                    if row[target_col] in exclude:
                        continue
                    m._accumulate(row,val)
            result.append( m._final() )
        return result

    def remaining_targets(self,exclude=[]):
        return self._all_targets - set(exclude)


class non_pathsum_path_data_loader(TargetImportanceDataLoader):
    def fetch(self):
        try:
            ts = self.bji.get_target_set()
            if ts is None:
                raise IOError('target set unavailable')
            target_key = self.bji.get_target_key(self.drug_ws)
            if target_key is None:
                raise IOError('target key unavailable')
        except self.bji.AmbiguousKeyError:
            if ts is not None:
                keys = [x
                        for x in self.bji.get_target_keys(self.drug_ws)
                        if x not in self.key_exclude
                        ]
                if len(keys) != 1:
                    raise
                target_key = keys.pop()
        if ts is None:
            raise IOError('target detail unavailable')
        settings = self.bji.job.settings()
        return {
            'settings': settings,
            'target': (FileWrapper(ts.filename), target_key)
        }


class non_pathsum_path(path):
    loader = non_pathsum_path_data_loader
    def _get_name(self):
        return self.cm
    def _set_names(self):
        self.score_names=[self.ds_name]
        if self.settings.get('p2p_file'):
            self.score_names.append(self.is_name)
    def _return_metrics(self, score, ts):
        if score == self.ds_name:
            return (3, 't2p p2d', 't2p:protein')
        elif score == self.is_name:
            return (4, 't2p p2p p2d', 'p2p:prot1')
        else:
            raise NotImplementedError()
    def setup(self, data):
        self.settings = data['settings']
        ts_file,target_name = data['target']
        from algorithms.pathsum2 import TargetSet
        ts = TargetSet()
        ts.load_from_file(ts_file.filename)
        self.target = ts.get_target(target_name)
        self._set_names()
        self.metrics=[self._make_metric(ts,x) for x in self.score_names]
        if self.target is None:
### Put an empty object in place to allow the 'normal' error handling take care of this
            from algorithms.pathsum2 import Target
            self.target = Target('')
        self._all_targets = set([
                row[target_col[k]]
                for target_col,m in self.metrics
                for k in m.score_map
                for row in self.target.paths.get(k,[])
                ])
        self._row_cache = {}

class gpath(non_pathsum_path):
    cm='gpath'
    ds_name = 'gds'
    is_name = 'gis'

class capp(non_pathsum_path):
    cm='capp'
    ds_name = 'capds'
    is_name = 'capis'


class codes_loader(TargetImportanceDataLoader):
    def fetch(self):
        drug_dpi = DrugDpi(self.parms['p2d_file'], self.parms['p2d_t'], self.drug_ws, self.cache)

        dpi_targets = set([x[1] for x in drug_dpi.dpi_info])
        score_dict = self._load_gesig(dpi_targets)

        return {
            'score_dict': score_dict,
            'dpi': drug_dpi
        }


    def _load_gesig(self, dpi_targets):
        from scripts.glee import get_gesig, open_infile
        from dtk.scores import JobCodeSelector
        info = self.bji
        ip_code = info.parms['input_score']

        # Most jobs will pull keys out of the same gesig data, cache it
        # to speed up load time.
        gesig_key = ('gesig', self.ws_id, ip_code)
        def compute_gesig():
            cat = JobCodeSelector.get_catalog(self.ws_id,ip_code)
            ordering = cat.get_ordering(ip_code,True)
            return get_gesig(ordering)
        lt = self.cache.get_or_compute(gesig_key, compute_gesig)

        score_dict = {
                x[0]: float(x[1]) for x in lt if x[0] in dpi_targets
                }

        return score_dict


class codes(TargetImportance):
    loader = codes_loader
    cm='codes'
    score_names=['absDir', 'codesMax'] # we don't use 'codesCor']
    def setup(self, data):
        dpi = data['dpi']
        dpi_info = dpi.get_dpi_info()
        from scripts.codes import get_PI_directional_evidence as gpde
        self.dpi_targets = {c[1]:gpde(c) for c in dpi_info}
        self._score_dict = data['score_dict']

    def get_score_set(self,exclude=[]):
        from scripts.codes import compare_sig_to_sig
        from dtk.data import dict_subtract

        # We skip correlation because it is expensive and it isn't one of
        # the scores we're using.
        to_ret = compare_sig_to_sig(
                        self._score_dict,
                        dict_subtract(self.dpi_targets, exclude),
                        union = False,
                        skip_cor = True
                        )
        if not to_ret:
            to_ret = [0.0] * len(self.score_names)
        to_ret[0] = abs(to_ret[0])
        return to_ret
    def remaining_targets(self,exclude=[]):
        all_targets = set(self.dpi_targets.keys())
        return all_targets - set(exclude)

class defus_loader(TargetImportanceDataLoader):
    def fetch(self):
        self._load_input_data()
        out = {
            'settings': self.settings,
            'in_data': self.in_data,
            'agent': self.drug_ws.agent_id,
            'job_id': self.job_id,
        }
        return out

    def _load_input_data(self):
        from algorithms.run_defus import gen_defus_settings

        cache_key = ('defus_inputs', self.job_id)
        def compute_defus_inputs():
            logger.info("Computing defus inputs for %s", self.job_id)
            import dtk.metasim as ms
            return self.bji.generate_input_data(sim_methods=[ms.DirectTargetSim, ms.IndirectTargetSim])

        self.in_data = self.cache.get_or_compute(cache_key, compute_defus_inputs)
        self.settings = gen_defus_settings(self.parms)


import time
class defus(TargetImportance):
    loader=defus_loader
    cm='defus'
    # prMax should be feasible to add too.
    # TODO: The issue with it, as written, is that we're already inside a parallel process,
    # so the actual pagerank precmp is super slow.
    score_names=['dirJacScore', 'indJacScore']
    def setup(self, data):
        # Directly transcribe the 'data' properties onto 'self'.
        # Most of the code was already written to handle this style.
        for k, v in data.items():
            setattr(self, k, v)
        self.dpi_targets = self.in_data['ws_sim_keys']['TargetsKey'][self.agent]
        from scripts.newdefus import make_metasim

        def precmp():
            ms = make_metasim(self.settings)
            precomputed = ms.precompute(
                ref_keys=self.in_data['ref_sim_keys'],
                methods=self.in_data['methods'],
            )
            return ms, precomputed
        self.ms, self.precmp = self.cache.get_or_compute(('defus_precmp', self.job_id), precmp)
        self.all_methods = self.in_data['methods']

    def get_score_set(self,exclude=[]):
        from scripts.newdefus import run_data

        targets = set(self.dpi_targets) - set(exclude)
        self.in_data['ws_sim_keys']['TargetsKey'] = {self.agent: targets}

        if self.score_col is not None:
            # Peel-one-off scores by a single score at a time (even though we return the whole score array).
            # If we're only looking at one score, we only need to generate the score of interest.
            self.in_data['methods'] = [self.all_methods[self.score_col]]

        out, _, _ = run_data(self.ms, self.in_data, precomputed=self.precmp, cores=1)
        scores = out['scores']
        
        out = [scores.get(self.agent, {}).get(score_name.split('Score')[0], 0)
               for score_name in self.score_names]
              
        return out

    def remaining_targets(self,exclude=[]):
        all_targets = set(self.dpi_targets)
        return all_targets - set(exclude)

class DrugDpi(object):
    """Wrapper around DPI for a single drug with specified threshold."""
    def __init__(self, dpi_handle, dpi_threshold, drug_ws, cache):
        self.dpi_handle = dpi_handle
        self.dpi_threshold = dpi_threshold
        self.dpi_keys = self._fetch_dpi_keys(drug_ws, cache)
        self.dpi_info = self._fetch_dpi_info(drug_ws, cache)

    def _fetch_dpi_keys(self, drug_ws, cache):
        cache_key = ('dpi_keys', self.dpi_handle, drug_ws.id)
        def compute():
            from dtk.prot_map import DpiMapping
            dpi = DpiMapping(self.dpi_handle)
            return dpi.get_dpi_keys(drug_ws.agent)
        return cache.get_or_compute(cache_key, compute)

    def _fetch_dpi_info(self, drug_ws, cache):
        cache_key = ('dpi_info', self.dpi_handle, drug_ws.id)
        def compute():
            from dtk.prot_map import DpiMapping
            dpi = DpiMapping(self.dpi_handle)
            dpi_info = dpi.get_dpi_info_for_keys(self.dpi_keys, self.dpi_threshold)
            # Save it as a tuple, because we can't pickle the inlined namedtuple.
            return [tuple(x) for x in dpi_info]
        return cache.get_or_compute(cache_key, compute)

    def get_dpi_info(self):
        return self.dpi_info


class esga_loader(TargetImportanceDataLoader):
    def fetch(self):
        self._load_input()
        self._get_dpi()
        out = {
            'pr_d_file': self.pr_d_file,
            'drug_prefix': self.drug_prefix,
            'prot_prefix': self.prot_prefix,
            'dpi': self.dpi
        }

        return out

    def _load_input(self):
        self.pr_d_file = FileWrapper(self.bji.out_pickle)


        from scripts.connect_drugs_to_proteinSets import establish_prefixes
        self.drug_prefix, _, self.prot_prefix = establish_prefixes()

    def _get_dpi(self):
        self.dpi = DrugDpi(self.parms['dpi_file'], float(self.parms['min_dpi']), self.drug_ws, self.cache)



class esga(TargetImportance):
    cm='esga'
    loader=esga_loader
    score_names=['prMax']
    def setup(self, data):
        # Directly transcribe the 'data' properties onto 'self'.
        # Most of the code was already written to handle this style.
        for k, v in data.items():
            setattr(self, k, v)

        def load_prd():
            from six.moves import cPickle as pickle
            with open(self.pr_d_file.filename, 'rb') as f:
                return pickle.loads(f.read())
        self.pr_d = self.cache.get_or_compute(('pr_d', self.pr_d_file.filename), load_prd)


        def load_graph():
            from scripts.connect_drugs_to_proteinSets import build_keyed_dpi_graph, establish_prefixes

            return build_keyed_dpi_graph(self.dpi.dpi_handle,
                                           min_dpi = self.dpi.dpi_threshold
                                          )
        cache_key = ('dpi_graph', self.dpi.dpi_handle, self.dpi.dpi_threshold)
        self.g = self.cache.get_or_compute(cache_key, load_graph)
        self._setup_dpi()

    def _setup_dpi(self):
        from scripts.codes import get_PI_directional_evidence as gpde
        from dtk.prot_map import DpiMapping
        self.dpi_targets = {
                c[1]:gpde(c)
                for c in self.dpi.get_dpi_info()
                }
        keys = [x
                for x in self.dpi.dpi_keys
                if x not in self.key_exclude
                    and self.drug_prefix+x in self.g
                ]
        if len(keys) > 1:
            from runner.process_info import JobInfo
            raise JobInfo.AmbiguousKeyError(
                    'ambiguous path key: '+' '.join(keys)
                    )
        self.nodes = [self.drug_prefix +  x
                      for x in keys
                     ]

    def get_score_set(self,exclude=[]):
        from scripts.connect_drugs_to_proteinSets import score_prot_rank
        from dtk.data import dict_subtract
        scores = score_prot_rank(self.nodes,
                                 self.g,
                                 self.prot_prefix,
                                 True,
                                 dict_subtract(self.pr_d,
                                               [self.prot_prefix+x for x in exclude]
                                 )
                                )
        assert len(scores) == 1
        return [list(scores.values())[0]['protrank_max']]
    def remaining_targets(self,exclude=[]):
        all_targets = set(self.dpi_targets.keys())
        return all_targets - set(exclude)


class depend_loader(TargetImportanceDataLoader):
    def fetch(self):
        self.setup()

        if 'd2ps_file' in self.parms:
            dpi_handle,ppi_handle,_ = self.d2ps.split('_')
        else:
            dpi_handle = self.parms['dpi_file']
            ppi_handle = self.parms['ppi_file']

        ws = self.ws
        dpi_t = ws.get_dpi_thresh_default()
        ppi_t = ws.get_ppi_thresh_default()

        return {
            'moa': self._get_moa(),
            'score_type': self.score_type,
            'glee_scores': self.glee_scores,
            'glee_directions': self.glee_directions,
            'dpi': DrugDpi(dpi_handle, dpi_t, self.drug_ws, self.cache),
            'ppi': PpiHandle(ppi_handle, ppi_t),
            'gmt': self.gmt,
            'needs_remap': True,
        }

    def setup(self):
        from browse.models import Workspace
        self.ws = Workspace.objects.get(pk=self.ws_id)

        from runner.process_info import JobInfo
        self.info = self.bji
        glee_bji = self.bji.get_gl_bji()
        self.gmt = glee_bji.parms['std_gene_list_set']
        ###
        # Prior to sprint110 (22 March 2017), DEEPEnD runs did not store these variables.
        # To work around this we will provide the most-common file and score as the default.
        # This will be true for everything except the combo base-drug runs,
        # but we haven't tested protein importance scoring with combo predicions anyways.
        ###
        ### we no longer store the d2ps file, but instead use the DPI, PPI to build the file name
        if 'd2ps_file' in self.info.parms:
            # depend stores a full pathname here, which is unhelpful if
            # we copy a database to a dev machine, for example; ignore
            # any path part (this will continue to work even if depend
            # is changed to store only the basename)
            self.d2ps = os.path.split(self.info.parms['d2ps_file'])[1]
            # PLAT-1562 switches to using the uncompressed d2ps file,
            # but older depend runs will have the filename of the compressed
            # version
            suffix = '.gz'
            if self.d2ps.endswith(suffix):
                self.d2ps = self.d2ps[:-len(suffix)]
            # PLAT-1716 stores this as basename only; reconstruct filename
            suffix = '.tsv'
            if not self.d2ps.endswith(suffix):
                self.d2ps += suffix
        else:
            self.d2ps = '_'.join([
                self.info.parms['dpi_file'],
                self.info.parms['ppi_file'],
                self.gmt,
                ])+'.tsv'

        self.score_type = self.info.parms['score_type']
        # The default when this wasn't a parameter was psmax.
        self.score_method = self.info.parms.get('score_method', 'psmax')
        self._get_glee()

    def _get_moa(self):
        cache_key = ('ws_moas', self.ws.id, self.info.parms['dpi_file'])
        def compute():
            logger.info("Computing ws MOAs for for %s, %s", self.ws_id, self.info.parms['dpi_file'])
            from dtk.enrichment import ws_dpi_condense_keys
            from dtk.prot_map import DpiMapping
            all_wsa_ids = WsAnnotation.objects.filter(ws=self.ws).values_list('id', flat=True)
            moas = ws_dpi_condense_keys(
                    all_wsa_ids,
                    dpi_name=self.info.parms['dpi_file'],
                    # For scoring purposes, we want to exclude DPIs below thresh.
                    dpi_t=self.ws.get_dpi_thresh_default(),
                    )
            from dtk.d2ps import MoA
            moas = [MoA(moa) for moa in moas]
            wsa2moa = dict(zip(all_wsa_ids, moas))
            logger.info("ws moas computed")
            return wsa2moa
        wsa2moa = self.cache.get_or_compute(cache_key, compute)
        return wsa2moa[self.drug_ws.id]

    def _get_glee(self):
        from scripts.depend import load_gl_data
        cache_key = ('glee', self.ws_id, self.bji.get_src_run(), self.parms['qv_thres'])
        def compute():
            glee_in = self.bji.gen_gl_input()
            return load_gl_data(in_src=glee_in)

        self.glee_scores, self.glee_directions = self.cache.get_or_compute(cache_key, compute)


class PpiHandle(object):
    def __init__(self, handle, threshold):
        self.ppi_handle = handle
        self.ppi_threshold = threshold

    def get_mapping(self):
        from dtk.prot_map import PpiMapping
        return PpiMapping(self.ppi_handle)

class PathsToProts(object):
    def __init__(self, drug_dpi, ppi_handle, gmt):
        self._drug_dpi = drug_dpi
        self._ppi_handle = ppi_handle
        self._gmt = gmt

    def convert_paths_to_prots(self, importance_scores, score_names, cache):
        path_data = self._get_gmt_file(cache)
        self._get_indirect_dpi(cache)
        new_targ_imp = {}
        for pathway, d in importance_scores.items():
            drug_prots_in_path = self._pathway_to_drugs_prots(path_data, pathway)
            for s in score_names:
                if not drug_prots_in_path:
                    # We just want to assign everything to the 'missing' prot.
                    drug_prots_in_path = {'missing':['missing']}
# We'll just split the pathway score evenly amongst the proteins in the pathway
                per_ind_val = d[s]/float(len(drug_prots_in_path))
                for ip,l in drug_prots_in_path.items():
                    split_val = per_ind_val/float(len(l))
                    for p in l:
                        if p not in new_targ_imp:
                            new_targ_imp[p] = {}
                        if s not in new_targ_imp[p]:
                            new_targ_imp[p][s] = 0.0
                        new_targ_imp[p][s] += split_val
        return new_targ_imp

    def _get_gmt_file(self, cache):
        from dtk.gene_sets import get_gene_set_file
        from dtk.files import get_file_records
        s3f = get_gene_set_file(self._gmt)
        path_data = {frs[0]:set(frs[1].split(','))
                          for frs in get_file_records(s3f.path(), parse_type = 'tsv')
                         }
        return path_data


    def _get_indirect_dpi(self, cache):
        from dtk.prot_map import DpiMapping, PpiMapping
        dpi_targets = set([
                c[1]
                for c in self._drug_dpi.get_dpi_info()
                ])
        ppi_t = self._ppi_handle.ppi_threshold
        def compute_indirect():
            ppi = self._ppi_handle.get_mapping()
            from collections import defaultdict
            indirect_targets = defaultdict(list)
            for ind in ppi.get_ppi_info_for_keys(dpi_targets,
                                                 ppi_t
                                                ):
                indirect_targets[ind.prot2].append(ind.prot1)
            return indirect_targets

        cache_key = ('ppi_indirect_targets', tuple(sorted(dpi_targets)), ppi_t)
        self.indirect_targets = cache.get_or_compute(cache_key, compute_indirect)
        self.ind_trg = set(self.indirect_targets.keys())
    def _pathway_to_drugs_prots(self, path_data, pathway):
            intersection = self.ind_trg & path_data[pathway]
            return {k:self.indirect_targets[k]
                    for k in intersection
                   }

class depend(TargetImportance):
    loader = depend_loader
    cm = 'depend'
    score_names = ['psScoreMax']
    def setup(self, data):
        # Directly transcribe the 'data' properties onto 'self'.
        # Most of the code was already written to handle this style.
        for k, v in data.items():
            setattr(self, k, v)

        self.score_method = data.get('score_method', 'psmax')

        from dtk.d2ps import D2ps
        d2ps = D2ps(self.ppi.ppi_handle, self.gmt, method=self.score_type)
        d2ps.update_for_moas([self.moa])
        scores = d2ps.get_moa_pathway_scores(self.moa)
        self.score_dict = {x.pathway:x.score for x in scores}
        self.dir_dict = {x.pathway:x.direction for x in scores}

    def get_score_set(self,exclude=[]):
        from scripts.depend import SCORE_METHODS
        calc = SCORE_METHODS[self.score_method]
        from dtk.data import dict_subtract
        exclude = set(exclude)
        sl,dl = calc(
                        dict_subtract(self.score_dict,exclude),
                        dict_subtract(self.dir_dict,exclude),
                        self.glee_scores,
                        self.glee_directions,
                       )
        return [sl,dl]
    def remaining_targets(self,exclude=[]):
        all_targets = set(self.score_dict.keys())
        return all_targets - set(exclude)


class sigdif_loader(TargetImportanceDataLoader):
    def fetch(self):
        gesig = self._load_sig()
        prot_links = list(gesig.keys())

        return {
            'parms': self.parms,
            'prot_links': prot_links,
            'gesig': gesig,
            'target_prot': self.target_id,
            'max_input_prots': self._max_input_prots,
        }

    def _load_sig(self):
        ip_code = self.parms['input_score']
        def compute_fn():
            from dtk.scores import JobCodeSelector
            cat = JobCodeSelector.get_catalog(self.ws_id,ip_code)
            ordering = cat.get_ordering(ip_code,True)
            return ordering
        cache_key = ('gesig', self.ws_id, ip_code)
        ordering = self.cache.get_or_compute(cache_key, compute_fn)

        return {
                x[0]: float(x[1]) for x in ordering if float(x[1])>0
                }

    def predict_duration(self):
        # Need to override this, default is based on WSA targets.
        return len(self._load_sig())


class sigdif(TargetImportance):
    loader = sigdif_loader
    cm='sigdif'
    score_names=['sigdif']
    def setup(self, data):
        self.parms = data['parms']
        self._gesig = data['gesig']
        self.sig_node = 'startingNode'
        self._target_prot = data['target_prot']

        # We can have a ton of targets, which would make this take forever
        # via peel, let's stop doing a full peel after we have the top.
        self.loose_peel_after = 5
        # Load in PPI data
        # Load in dis-sig
        # Compute the ppi graph
        from scripts.connect_drugs_to_proteinSets import build_ppi_graph
        self.prot_prefix = 'a_'
        from dtk.prot_map import PpiMapping
        ppi = PpiMapping(self.parms['ppi_file'])
        ppi_key = (self.parms['ppi_file'], self.parms['min_ppi'])
        def make_graph():
            logger.info("Building ppi graph")
            return build_ppi_graph(ppi,
                                   prot_prefix = self.prot_prefix,
                                   direction = False,
                                   min_ppi_evid = self.parms['min_ppi']
                                    )
        self.g = self.cache.get_or_compute(ppi_key, make_graph).copy()
        logger.info("Loading disease sig")

        prot_links = data['prot_links']
        prot_links = [x for x in prot_links if self.prot_prefix+x in self.g]
        self._prot_links = set(prot_links)

        N = len(self._prot_links)

        max_input_prots = data['max_input_prots']

        if max_input_prots is not None and N > max_input_prots:
            logger.info("WARNING: Bailing out too many input prots (%d)" % N)
            self._prot_links = set([])

        runs = N*(N+1) / 2
        if self.loose_peel_after and N > self.loose_peel_after:
            # Close enough
            runs = N*self.loose_peel_after +  N - self.loose_peel_after
        logger.info("Prot has %d links, will compute score ~%d times" % (N, runs))

        # Add start + dis-sig edges
        self._add_sig_data()

        logger.info("Building scipy graph")
        import networkx as nx
        self.gscipy = nx.to_scipy_sparse_matrix(self.g, nodelist=list(self.g), weight='weight', dtype=float)


        self.name_to_idx = {str(name):i for i, name in enumerate(self.g)}
        self._longest_exclude = 0

        if self.prot_prefix + self._target_prot not in self.g:
            logger.info("Missing %s from our PPI network, score will be 0" % self._target_prot)
            self._missing = True
        else:
            self._missing = False

        self._cache_key_prefix = ppi_key + (self.parms['input_score'],)
        logger.info("Setup complete")

    def _add_sig_data(self):
        for p,v in self._gesig.items():
            pn = self.prot_prefix + p
            if pn in self.g:
                #print 'adding edge for ' + p
                self.g.add_edge(self.sig_node, pn, weight = v)
            else:
                #print 'missing a node for ' + p
                pass

    def _remove_prot(self, prot):
        p = self.prot_prefix + prot
        edges = list(self.g.edges(p, data=True))
        self.g.remove_node(p)
        return edges

    def _add_back(self, removed):
        self.g.add_edges_from(removed)


    def get_score_set(self,exclude=[]):
        if self._missing:
            return [0]
        cache_key = self._cache_key_prefix + tuple(sorted(exclude))
        compute_fn = lambda: self._compute_pagerank(exclude)
        # This has been using up too much memory for data with huge prot #'s.
        if len(exclude) <= 1 and len(self._prot_links) < 100:
            protrank_dict = self.cache.get_or_compute(cache_key, compute_fn)
        else:
            # We're going to run out of memory if we try to cache all
            # combinations for peel.
            protrank_dict = compute_fn()
        return [protrank_dict[self.prot_prefix + self._target_prot]]

    def _compute_pagerank(self, exclude):
        # Make graph by removing links to excludes
        exclude = set(exclude)
        if len(exclude) > self._longest_exclude:
            #logger.info("Peeling %d" % len(exclude))
            self._longest_exclude = len(exclude)


        if False: # pragma: no cover
            # This is the sensible codepath, but takes 10x longer than the
            # other one, because coverting from networkx to scipy is very slow.
            removed = []
            for prot in exclude:
                removed.extend(self._remove_prot(prot))

            # Compute page rank for this graph
            from scripts.connect_drugs_to_proteinSets import run_page_rank
            protrank_dict = run_page_rank((self.sig_node,
                                        self.g,
                                        self.g,
                                        self.prot_prefix,
                                        self.parms['restart_prob'],
                                        self.parms['iterations'],
                                        ))

            self._add_back(removed)
        else:
            # In this codepath, we use some slightly modified versions of
            # things to be able to do the scipy conversion only once.
            # Should give identical output to the above codepath.
            from .pagerank import mat_without_nodes,pagerank_scipy, run_page_rank

            # With REMOVE_NODE, we remove the excluded prot from the network
            # entirely.
            # Without REMOVE_NODE, we are leaving the prot in the network, but
            # removing our initial weight from it.
            REMOVE_NODE = False
            if REMOVE_NODE:
                removed = []
                for prot in exclude:
                    removed.extend(self._remove_prot(prot))
                exclude_idxs = [self.name_to_idx[self.prot_prefix+e] for e in exclude]
                M = mat_without_nodes(self.gscipy, exclude_idxs)
            else:
                removed_w = {}
                for prot in exclude:
                    pn = self.prot_prefix + prot
                    removed_w[prot] = self.g[self.sig_node][pn]['weight']
                    self.g[self.sig_node][pn]['weight'] = 0
                    # We don't need to remove it from the scipy graph because
                    # the initialization weights come from the nx graph.
                M = self.gscipy

            protrank_dict = run_page_rank(self.sig_node,
                                        self.g,
                                        self.g,
                                        M,
                                        self.prot_prefix,
                                        self.parms['restart_prob'],
                                        self.parms['iterations'],
                                        )

            if REMOVE_NODE:
                self._add_back(removed)
            else:
                for prot in exclude:
                    pn = self.prot_prefix + prot
                    self.g[self.sig_node][pn]['weight'] = removed_w[prot]
        return protrank_dict

    def remaining_targets(self,exclude=[]):
        return self._prot_links - set(exclude)


if __name__ == "__main__":
    # force unbuffered output
    sys.stdout = sys.stderr

    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("wsid", help="ws_id")
    parser.add_argument("jobid", help="job id")
    parser.add_argument("cm", help="Job type (codes, depend, path)")
    parser.add_argument("wsa", help="The WSA for the drug of interest")
    parser.add_argument("--peel", help="rank by specified score; remove top target on each iteration")
    parser.add_argument("--loi", action='store_true', help="rank by specified score; remove top target on each iteration")
    parser.add_argument("-o", help="output file, default: <cm>_<jobid>_<wsa>_importance.tsv")
    from dtk.log_setup import addLoggingArgs, setupLogging
    addLoggingArgs(parser)
    args = parser.parse_args()
    setupLogging(args)
    if not args.o:
        args.o = "_".join([args.cm, args.jobid, args.wsa, 'importance.tsv'])

    CM_Class = TargetImportance.lookup(args.cm)
    CM_Loader = CM_Class.loader

    from dtk.target_importance import Cache
    cache = Cache()

    loader = CM_Loader(ws_id=args.wsid, wsa_id=args.wsa, bji_or_id=args.jobid, cache=cache, target_id=None)
    cm = CM_Class(ws_id=args.wsid, wsa_id=args.wsa, job_id=args.jobid, data=loader.fetch(), cache=cache)
    if args.peel:
        cm.peel_one_off(args.peel)
        method='cumulative'
    elif args.loi:
        cm.leave_one_in()
        method='LOI'
    else:
        cm.leave_one_out()
        method='LOO'
    cm.score_importance(scoring_method=method, score_name=args.peel)
    cm.report_scores(args.o)
    with open(args.o) as f:
        logger.info(f.read())
