#!/usr/bin/env python3


"""
NOTE:
    Featuresets can be added at will and experimented with for viewing purposes (AVAILABLE_FEATURESETS)
    Once they've been baked into a model, however, they shouldn't be modified unless you intend
    to retrain the model and stop using previous ones.
    (We could probably provide some simple version infrastructure here to enforce that if it becomes
    a common thing.)
"""

import django_setup
import numpy as np
import logging
logger = logging.getLogger(__name__)


class MLModel:
    def __init__(self, featuresets, model=None):
        self.featuresets = featuresets
        self.model = model

    @classmethod
    def from_file(self, fn):
        import json
        with open(fn) as f:
            data = json.loads(f.read())

            model_data = data['model']
            assert model_data['type'] == 'logreg', f'Unexpected model type {model_data["type"]}'
            from sklearn.linear_model import LogisticRegression
            import numpy as np
            model = LogisticRegression()
            model.classes_ = np.array([0, 1])
            model.coef_ = np.array(model_data['coef'])
            model.intercept_ = np.array(model_data['intercept'])

            featuresets_classes = []
            for featureset in data.get('featuresets', []):
                featuresets_classes.append(
                        globals()[featureset['name']]
                        )
            featuresets = instantiate_featuresets(featuresets_classes)
            logger.info("Loaded model with input featuresets: %s", featuresets)
        return MLModel(featuresets, model=model)

    def save(self, fn):
        featuresets = [{
            'name': fs.__class__.__name__,
            } for fs in self.featuresets]

        import json
        with open(fn, 'w') as f:
            # We're not pickling the raw model because of versioning issues with sklearn.
            # See https://scikit-learn.org/stable/modules/model_persistence.html
            data = {
                    'model': {
                        'type': 'logreg',
                        'coef': self.model.coef_.tolist(),
                        'intercept': self.model.intercept_.tolist(),
                    },
                    'featuresets': featuresets,
                }
            f.write(json.dumps(data))



    def train(self, feature_mat, label_mat):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(
                solver='lbfgs',
                random_state=0,
                class_weight={1: 1, 0: 1},
                max_iter=1e5,
                # C is inverse regularization strength.  sklearn defaults to
                # 1, but bumping it up higher here, our positive sample size
                # isn't huge relative to number of features so need to be
                # careful about overfitting.
                C=0.1
                )
        sample_weights = label_mat[:, 0] * 100 + 1
        self.model.fit(feature_mat, label_mat[:, 0], sample_weight=sample_weights)
        # Printing accuracy here, but it's not really a meaningful metric for
        # the way we're using this data.
        logger.info("Acc: %.8f", self.model.score(feature_mat, label_mat))
        logger.info("Coeff: %s", self.model.coef_)

    def global_importances(self):
        return self.model.coef_[0]

    def predict(self, feature_mat):
        return self.model.predict_proba(feature_mat)

def chunker(lst, k):
    step = len(lst) // k
    for i in range(k):
        yield lst[i*step:(i+1)*step]

def generate_eval_stats(model, wsa_source, workspaces, featuresets, evalsets):
    s = Selectability()
    probs = list(s.predict(model, wsa_source, workspaces, featuresets))
    wsas = [x[0] for x in probs]
    prob_vals = [x[1] for x in probs]

    fvs_featuresets = featuresets
    fvs_cols = sum([x.feature_names() for x in fvs_featuresets], [])
    fvs_cols = ['wsa_id'] + fvs_cols
    fvs = s.generate_data(wsa_source, workspaces, fvs_featuresets)
    fvs = [[wsa.id] + row.tolist() for wsa, row in zip(wsas, fvs)]

    # TODO: Separate these pieces out?
    eval_metrics = []
    for evalset in evalsets:
        labels = s.generate_data(wsa_source, workspaces, [evalset])
        pos_probs = [pred_prob for (pred_prob, y) in zip(prob_vals,labels) if y == 1]
        split_prob = min(pos_probs) if pos_probs else 0

        tp=0
        tn=0
        fp=0
        fn=0
        for pred_prob, y in zip(prob_vals, labels):
            pred = 1 if pred_prob >= split_prob else 0
            if pred == 1:
                if y == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if y == 0:
                    tn += 1
                else:
                    fn += 1
        logger.info("(%s) @%.3f tp=%d, tn=%d, fp=%d, fn=%d", evalset, split_prob, tp, tn, fp, fn)
        eval_data = {
                'split_prob': split_prob,
                'confusion': dict(tp=tp, tn=tn, fp=fp, fn=fn),
                'roc_curve': (prob_vals, labels),
                'evalset': evalset,
                }
        eval_metrics.append(eval_data)
    return {
        'eval_metrics': eval_metrics,
        'prob_vals': prob_vals,
        'wsas': wsas,
        'fvs': (fvs_cols, fvs),
        }

def instantiate_featuresets(classes):
    depname_to_inst = {}
    def instantiate(cls):
        depname = cls.__name__.lower()
        if depname in depname_to_inst:
            return depname_to_inst[depname]
        dep_insts = [instantiate(dep) for dep in cls.deps]
        inst = cls(*dep_insts)
        depname_to_inst[depname] = inst
        return inst

    out = [instantiate(cls) for cls in classes]
    return out




def cross_validate(wsa_source, workspaces, featuresets, labelset, evalsets, k=None):
    if k is None:
        k = len(workspaces)
    k = min(k, len(workspaces))

    import random
    workspaces = list(workspaces)
    random.shuffle(workspaces)

    s = Selectability()
    outs = []
    for test_ws in chunker(workspaces, k):
        train_ws = set(workspaces) - set(test_ws)
        model = s.train(wsa_source, train_ws, featuresets, labelset)
        outs.append(generate_eval_stats(model, wsa_source, test_ws, featuresets, evalsets))
    return outs


def plot_cross_validation(cross_valid_out):
    from dtk.plot import scatter2d, annotations
    plots = []
    for fold_out in cross_valid_out:
        fprs = []
        tprs = []
        thresholds = []
        class_idxs = []
        class_cfgs = []

        for i, eval_metric in enumerate(fold_out['eval_metrics']):
            evalset = eval_metric['evalset']
            class_cfgs.append((str(evalset), {}))
            probs, labels = eval_metric['roc_curve']
            from sklearn.metrics import roc_curve
            fpr, tpr, threshold = roc_curve(labels, probs)
            fprs.extend(fpr)
            tprs.extend(tpr)
            thresholds.extend(threshold)
            class_idxs.extend([i]*len(fpr))

        plot = scatter2d(
                'False Positive Rate',
                'True Positive Rate',
                zip(fprs,tprs),
                class_idx=class_idxs,
                classes=class_cfgs,
                text=['@%.2f'% t for t in thresholds],
                title='ROC curve',
                refline=False,
                linestyle='lines',
                )
        plots.append(plot)
    return plots


class Selectability:
    def train(self, wsa_source, workspaces, featuresets, labelset):
        logger.info("Generating data")
        # We force the refresh of our cache when training.
        # Much of our caching is only really valid in the context of
        # predictions, not training.
        data_mat = self.generate_data(wsa_source, workspaces, featuresets,
                                      force_refresh=True)
        logger.info("Generating labels")
        label_mat = self.generate_data(wsa_source, workspaces, [labelset],
                                       force_refresh=True)
        logger.info("Fitting")
        model = MLModel(featuresets)
        model.train(data_mat, label_mat)

        importances = model.global_importances()
        names = self.feature_names(featuresets)
        logger.info("Importances: %s", list(zip(names, importances)))

        logger.info("Done!")
        return model

    def predict(self, model, wsa_source, workspaces, featuresets, force_refresh=False):
        data_mat = self.generate_data(wsa_source, workspaces, featuresets, force_refresh=force_refresh)
        all_wsas = self.get_wsas(wsa_source, workspaces)
        return zip(all_wsas, [x[1] for x in model.predict(data_mat)])

    def get_wsas(self, wsa_source, workspaces):
        all_wsas = []
        for ws in workspaces:
            wsas = wsa_source.get_wsas(ws)
            all_wsas.extend(wsas)
        return all_wsas

    def feature_names(self, featuresets):
        feature_names = []
        for featureset in featuresets:
            feature_names += featureset.feature_names()
        return feature_names

    def generate_data(self, wsa_source, workspaces, featuresets, force_refresh=False, collect_errors=False):
        """
        If collect_errors, will catch exceptions for a featureset/ws and continue, returning both results
        and the exceptions.  This makes sense for feature viewing, but shouldn't be used in training unless
        the model can handle NaN values.
        """
        from browse.models import WsAnnotation
        feature_names = self.feature_names(featuresets)

        num_rows = 0
        for ws in workspaces:
            wsas = wsa_source.get_wsas(ws)
            num_rows += len(wsas)

        errors = []

        mat = np.empty((num_rows, len(feature_names)), dtype=float)
        start_row = 0
        for ws in workspaces:
            wsas = wsa_source.get_wsas(ws)
            end_row = start_row + len(wsas)

            col_start = 0
            for featureset in featuresets:
                col_end = col_start + len(featureset.feature_names())
                submat = mat[start_row:end_row,col_start:col_end]

                try:
                    submat[:] = featureset.make_mat(ws, wsas, force_refresh)
                except Exception as e:
                    if not collect_errors:
                        raise
                    else:
                        import traceback as tb
                        errors.append((ws, featureset, tb.format_exc()))
                        submat[:] = float('nan')

                col_start = col_end
            start_row = end_row

        if collect_errors:
            return mat, errors
        else:
            return mat


class FeatureSet:
    """

    We cache at a couple of levels.
    Per-instance caching is the fastest and is used for things like cross
    validation where we'd otherwise have to store or recompute the features.

    Otherwise, we also have a persistent cache so that e.g. the prescreen
    page can load faster.  For that cache we do care about invalidation.
    """

    deps = []
    def __init__(self):
        # We cache results to speed up cross-validation.
        # Shouldn't hold onto these features too long, they cache
        # a fair amount of data.
        self._cache = {}

    def cache_version(self, ws):
        """Returns a token that will be compared with the cache for validity.

        By default returns 0, indicating the cache never becomes invalid.
        None is treated as a special value to indicate never cache.
        """
        return 0

    @classmethod
    def name(cls):
        out = cls.__name__
        if out.endswith('FeatureSet'):
            return out[:-len('FeatureSet')]
    def feature_names(self):
        pass
    def available(self, ws):
        pass
    def fill(self, nparr, ws, wsas):
        pass
    def make_mat(self, ws, wsas, force_refresh=False):
        from hashlib import sha1
        def key_normalize_func(x):
            # We don't want to use the builtin hash method here because it will be inconsistant across sessions
            # for some data, due to salting.  Instead, let's just convert the whole thing to a json string.
            def encode(x):
                if isinstance(x, frozenset):
                    return sorted(x)
                else:
                    raise Exception(f"Unhandled type {type(x)} in {self.name()}")
            import json
            return json.dumps(x, sort_keys=True, default=encode)

        key = (ws.id, tuple(sorted(x.id for x in wsas)))
        if key not in self._cache:
            # Not in our in-memory cache, check our django cache.
            import time
            start = time.time()
            from django.core.cache import caches
            per_cache = caches['selectability']
            cache_version = self.cache_version(ws)
            per_hash = key_normalize_func((key, cache_version))
            per_key = f'selectability.{self.name()}.{per_hash}'
            # These keys can be very long (e.g. list of prots or wsas), created a short hashed version
            # for logging purposes so that they can be human-compared.
            short_key = sha1(per_key.encode('utf8')).hexdigest()
            logger.info("Looking for %s %s (cache took %.2f ms)", self.name(), short_key, (time.time() - start)*1000)
            if force_refresh or cache_version is None or per_key not in per_cache:
                # Not in there either (or uncacheable), regenerate
                logger.info("Generating cache entry for %s, %s, %s (%d)",
                    ws.id, type(self), short_key, len(self._cache))
                n_cols = len(self.feature_names())
                n_rows = len(wsas)
                mat = np.empty((n_rows, n_cols), dtype=float)
                self.fill(mat, ws, wsas)
                per_cache.set(per_key, mat)
            else:
                mat = per_cache.get(per_key)
            # Note that in testing our dummy cache never returns anything
            # even if we just put something into it.
            self._cache[key] = mat
        return self._cache[key]


class SelectivityFeatureSet(FeatureSet):
    @classmethod
    def feature_names(cls):
        return ['dpiSel', 'ppiSel']
    @classmethod
    def available(cls, ws):
        selectivity_jobs = ws.get_prev_job_choices('selectivity')
        return len(selectivity_jobs) > 0
    def fill(self, nparr, ws, wsas):
        from runner.process_info import JobInfo
        selectivity_jobs = ws.get_prev_job_choices('selectivity')
        bji = JobInfo.get_bound(ws, selectivity_jobs[0][0])
        cat = bji.get_data_catalog()
        fv = dict(cat.get_feature_vectors('dpiSel', 'ppiSel')[1])
        for nparr_row, wsa in zip(nparr, wsas):
            nparr_row[:] = fv.get(wsa.id, (0, 0))


class IndicationFeatureSet(FeatureSet):
    def __init__(self, min_indication):
        super().__init__()
        self.min_indication = min_indication

    def __str__(self):
        from browse.models import WsAnnotation
        label = WsAnnotation.indication_vals.get('label',self.min_indication)
        return 'Ind >= %s' % label

    def cache_version(self, ws):
        # We could try to detect whether any indications have changed within
        # our refset... but this computes fast enough that it's not worth
        # the effort for now, just always invalidate.
        return None

    @classmethod
    def feature_names(cls):
        return ['selected',]

    @classmethod
    def available(cls, ws):
        return True

    def fill(self, nparr, ws, wsas):
        from browse.models import WsAnnotation
        ivals = WsAnnotation.indication_vals
        targ_idx = WsAnnotation.discovery_order_index(self.min_indication)

        prefetch_wsas = list(WsAnnotation.objects.filter(ws=ws, pk__in=wsas).prefetch_related('dispositionaudit_set'))
        wsa2maxind = {}
        for wsa in prefetch_wsas:
            wsa2maxind[wsa.id] = wsa.max_discovery_indication()


        for nparr_row, wsa in zip(nparr, wsas):
            wsa_idx = wsa.discovery_order_index(wsa2maxind[wsa.id])
            is_replacement = wsa.replacement_for.count() > 0
            is_replaced = wsa.replacements.count() > 0
            # We don't want replacements, because typically they were hard
            # to find by our normal metrics.
            # e.g. often they didn't have the correct targets in that WS.
            # We'll track the original molecules they were found via; in
            # this case, if something was replaced and isn't itself a replacement,
            # then that's an original.
            nparr_row[:] = [1.0 if (wsa_idx >= targ_idx and not is_replacement) or is_replaced else 0.0]

        print("Good: %d / %d => %.8f" % (
            np.round(np.sum(nparr)), len(wsas), 1.0 - np.sum(nparr) / len(wsas)))

class IndicationGroupFeatureSet(FeatureSet):
    def __init__(self, code, unreplaced=False):
        super().__init__()
        self.code = code
        self.unreplaced=unreplaced
    def __str__(self):
        return self.code

    def cache_version(self, ws):
        # We could try to detect whether any indications have changed within
        # our refset... but this computes fast enough that it's not worth
        # the effort for now, just always invalidate.
        return None

    @classmethod
    def feature_names(cls):
        return ['selected',]

    @classmethod
    def available(cls, ws):
        return True

    def fill(self, nparr, ws, wsas):
        if self.unreplaced:
            from dtk.retrospective import ids_to_wsas,unreplaced_mols
            prefetch_wsas = set([x.id for x in unreplaced_mols(ids_to_wsas(ws.get_wsa_id_set(self.code)))])
        else:
            prefetch_wsas = ws.get_wsa_id_set(self.code)
        for nparr_row, wsa in zip(nparr, wsas):
            nparr_row[:] = [1.0 if wsa.id in prefetch_wsas else 0.0]
        print("Good: %d / %d => %.8f" % (
            np.round(np.sum(nparr)), len(wsas), 1.0 - np.sum(nparr) / len(wsas)))

class CMRankFeatureSet(FeatureSet):
    def __init__(self,jrc,jid_2_rnkr, wsa_2_jid):
        super().__init__()
        self.jrc=jrc
        self.jid_2_rnkr = jid_2_rnkr
        self.wsa_2_jid = wsa_2_jid
        self.ws_2_jid = {}
        for wsa,jid in self.wsa_2_jid.items():
            if wsa.ws.id not in self.ws_2_jid:
                self.ws_2_jid[wsa.ws.id] = set()
            self.ws_2_jid[wsa.ws.id].add(jid)
# XXX while i added caching to this, it doesn't really save much time in practice.
# XXX This is b/c the real lifting is done on the view-side to determine which
# XXX CMs to load and getting the Rankers used below
    def cache_version(self, ws):
        if ws.id in self.ws_2_jid:
            return frozenset(self.ws_2_jid[ws.id])
        return None
    def feature_names(self):
        return [self.jrc + ' rank',]

    @classmethod
    def available(cls, ws):
        return True

    def fill(self, nparr, ws, wsas):
        import numpy as np
        for nparr_row, wsa in zip(nparr, wsas):
            if wsa not in self.wsa_2_jid:
                print(f"{wsa.get_name(False)} in {wsa.ws.id} is missing {self.jrc}")
        # we're using NaNs to make it notably different from the distribution
        # but not skewing the scale
                v=np.nan
            else:
                r = self.jid_2_rnkr[self.wsa_2_jid[wsa]]
                v=r.get_pct(wsa.id)
            nparr_row[:] = [v]

class NonNovelTargetFeatureSet(FeatureSet):
    @classmethod
    def feature_names(cls):
        to_ret = cls.non_lbn_names()
        return to_ret
    @classmethod
    def non_lbn_names(cls):
        return ['nonNovel_dir_jac', 'nonNovel_ind_jac']
    @classmethod
    def available(cls, ws):
        return True
    def cache_version(self, ws):
        return frozenset(self.get_nonnovel_uniprots(ws))
    
    def get_nonnovel_uniprots(self, ws):
        from browse.default_settings import DiseaseNonNovelUniprotsSet
        ps = DiseaseNonNovelUniprotsSet.value(ws)
        non_novel_uniprots = ws.get_uniprot_set(ps)
        return non_novel_uniprots

    def fill(self, nparr, ws, wsas):
        non_lbn_scores = self._get_non_lbns(ws, wsas)
        for nparr_row, wsa in zip(nparr, wsas):
            non_lbns = non_lbn_scores[wsa]
            nparr_row[:] = non_lbns

    def _get_non_lbns(self, ws, wsas):
        from browse.default_settings import DpiDataset,DpiThreshold,PpiDataset,PpiThreshold
        from dtk.prot_map import PpiMapping,DpiMapping
        from dtk.similarity import calc_jaccard

        non_novel_uniprots = self.get_nonnovel_uniprots(ws)

        dpi=DpiMapping(DpiDataset.value(ws))
        dpi_t=DpiThreshold.value(ws)
        ppi=PpiMapping(PpiDataset.value(ws))
        ppi_t=PpiThreshold.value(ws)
        wsa2dpi = dpi.get_wsa2dpi_map(ws,
                                      [wsa.id for wsa in wsas],
                                       min_evid=dpi_t
                                      )
        indirs_d={}
        for rec in ppi.get_ppi_info_for_keys(set([p for s in wsa2dpi.values()
                                                  for p in s])
                                           , min_evid=ppi_t
                                           ):
            if rec.prot1 not in indirs_d:
                indirs_d[rec.prot1]=set()
            indirs_d[rec.prot1].add(rec.prot2)
        results = {}
        for wsa in wsas:
            if wsa.id not in wsa2dpi:
                results[wsa]=[0.0, 0.0]
                continue
            else:
                dir = calc_jaccard(set(wsa2dpi[wsa.id]),non_novel_uniprots)
            ind_unis = set()
            for p in wsa2dpi[wsa.id]:
                if p not in indirs_d:
                    continue
                ind_unis.update(set([ip for ip in indirs_d[p]]))
            ind = calc_jaccard(ind_unis,non_novel_uniprots)
            results[wsa]=[dir,ind]
        return results

class NoveltyFeatureSet(FeatureSet):
    @classmethod
    def feature_names(cls):
        to_ret = cls.lbn_names()
        return to_ret
    @classmethod
    def lbn_names(cls):
        return ['lbnOR', 'lbnP', 'disPorWDrug', 'drugPorWDis', 'targLogOdds', 'targPortion']
    @classmethod
    def available(cls, ws):
        lbn_jobs = ws.get_prev_job_choices('lbn')
        return len(lbn_jobs) > 0
    def fill(self, nparr, ws, wsas):
        from runner.process_info import JobInfo
        lbn_ml_jobs = ws.get_prev_job_choices('lbn')
        bji = JobInfo.get_bound(ws, lbn_ml_jobs[0][0])
        cat = bji.get_data_catalog()
        N = len(self.lbn_names())
        fv = dict(cat.get_feature_vectors(*self.lbn_names())[1])
        for nparr_row, wsa in zip(nparr, wsas):
            lbns = fv.get(wsa.id, (0,)*N)
            nparr_row[:] = list(lbns)

from browse.models import WsAnnotation

class UniquenessFeatureSet(FeatureSet):
    @classmethod
    def feature_names(cls):
        return [
            'reviewed_target_overlap',
            ]
    ivals = WsAnnotation.indication_vals
    good_inds = set([ivals.INITIAL_PREDICTION, ivals.REVIEWED_PREDICTION, ivals.HIT])

    def cache_version(self, ws):
        wsa_ids = frozenset(self.make_ref_wsas(ws, as_id=True))
        dpi = ws.get_dpi_default()
        return (wsa_ids, dpi)

    @classmethod
    def available(cls, ws):
        # Availability check is kind of slow, realistically not going to be
        # the blocking feature.
        return True
        if False:
            from browse.models import DispositionAudit
            return ws.id in DispositionAudit.objects.filter(
                    indication__in=cls.good_inds
                    ).values_list('wsa__ws', flat=True).distinct()

    @classmethod
    def make_ref_wsas(cls, ws, as_id):
        from browse.models import WsAnnotation, DispositionAudit
        wsa_ids = DispositionAudit.objects.filter(
                wsa__ws=ws,
                indication__in=cls.good_inds
                ).values_list('wsa', flat=True).distinct()
        if as_id:
            return wsa_ids
        ref_wsas = WsAnnotation.objects.filter(pk__in=wsa_ids)
        return ref_wsas
    @classmethod
    def make_target_to_ref(cls, ws):
        # Get the reference set
        # Include anything that was ever in our 'good_ind' set, even if it
        # is currently inactive or invivo.
        ref_wsas = cls.make_ref_wsas(ws, as_id=False)
        from collections import defaultdict
        target_to_ref = defaultdict(set)
        from dtk.prot_map import AgentTargetCache
        ref_atc = AgentTargetCache.atc_for_wsas(wsas=ref_wsas, ws=ws)
        for wsa in ref_wsas:
            for (key, uniprot, ev, direction) in ref_atc.raw_info_for_agent(wsa.agent_id):
                target_to_ref[uniprot].add(wsa.id)
        return target_to_ref

    def fill(self, nparr, ws, wsas):
        target_to_ref = self.make_target_to_ref(ws)

        from dtk.prot_map import AgentTargetCache
        atc = AgentTargetCache.atc_for_wsas(wsas=wsas, ws=ws)
        # For each wsa, check overlap with ref set
        # Score can be fraction of targets in/not in ref set (ignoring self)?
        for nparr_row, wsa in zip(nparr, wsas):
            non_overlapped_targets = 0
            overlapped_targets = 0
            me = set([wsa.id])
            for (key, uniprot, ev, direction) in atc.raw_info_for_agent(wsa.agent_id):
                ev = float(ev)
                ref = target_to_ref[uniprot] - me
                if len(ref) == 0:
                    non_overlapped_targets += ev
                else:
                    overlapped_targets += ev
            denom = non_overlapped_targets + overlapped_targets
            if denom == 0:
                score = 0
            else:
                score = non_overlapped_targets / denom
            nparr_row[:] = [score]


class TargetOverlapFeatureSet(FeatureSet):
    @classmethod
    def feature_names(cls):
        return [
            'nonnovel_target_overlap',
            'intolerable_target_overlap',
            ]
    @classmethod
    def available(cls, ws):
        return True

    def cache_version(self, ws):
        protsets = self.get_protsets(ws)
        key = tuple(frozenset(ps) for ps in protsets)
        return key

    def get_protsets(self, ws):
        protset_names = []
        # Everything should have an unwanted ps, even if it is bad.
        non_novel = ws.get_nonnovel_ps_default()
        if non_novel == 'autops_none':
            # TODO: Expose warning?
            logger.warning("No non-novel protset set, using DPI of known treatments")
            non_novel = 'ds_0.5_tts'
        protset_names.append(non_novel)

        protset_names.append(ws.get_intolerable_ps_default())

        protsets = [ws.get_uniprot_set(x) for x in protset_names]
        return protsets

    def fill(self, nparr, ws, wsas):
        protsets = self.get_protsets(ws)

        from browse.models import WsAnnotation

        from dtk.prot_map import AgentTargetCache
        atc = AgentTargetCache.atc_for_wsas(wsas=wsas, ws=ws)
        for nparr_row, wsa in zip(nparr, wsas):
            scores = []
            for protset in protsets:
                non_overlapped_targets = 0
                overlapped_targets = 0
                for (key, uniprot, ev, direction) in atc.raw_info_for_agent(wsa.agent_id):
                    ev = float(ev)
                    if uniprot in protset:
                        overlapped_targets += ev
                    else:
                        non_overlapped_targets += ev
                denom = non_overlapped_targets + overlapped_targets

                if denom == 0:
                    # We have no targets at all... probably that is bad,
                    # but technically it is neither non-novel nor intolerable.
                    score = 1
                else:
                    score = non_overlapped_targets / denom
                scores.append(score)

            nparr_row[:] = scores

class NameFeatureSet(FeatureSet):
    @classmethod
    def feature_names(cls):
        return ['named']
    @classmethod
    def available(cls, ws):
        return True
    def fill(self, nparr, ws, wsas):
        # If no one has spent the effort to name this molecule, chances are
        # it is pretty experimental.
        for nparr_row, wsa in zip(nparr, wsas):
            name = wsa.agent.canonical
            # TODO: Check linked agents?  Can be slow.
            unnamed = 'CHEMBL' in name or 'BDBM' in name
            nparr_row[:] = [0.0 if unnamed else 1.0]


class AvailabilityFeatureSet(FeatureSet):
    @classmethod
    def feature_names(cls):
        return ['available']
    @classmethod
    def available(cls, ws):
        return True
    def fill(self, nparr, ws, wsas):
        from dtk.comm_avail import wsa_comm_availability
        from browse.models import WsAnnotation
        wsa_qs = WsAnnotation.prefetch_agent_attributes(wsas)
        avails = wsa_comm_availability(ws, wsa_qs)

        for nparr_row, wsa, avail in zip(nparr, wsas, avails):
            nparr_row[:] = [1 if (avail.has_zinc or avail.has_cas or avail.has_commdb) else 0]

class ScoreDiversityFeatureSet(FeatureSet):
    """
    This feature has several problems at the moment.
    1) The demerits don't correlate well with the expected outcome
    2) It is extremely slow to compute. (not a blocker because of caching)
    3) Can't be computed for some of our older workspaces that didn't generate
       score weights
    4) Difficult to compute for molecules that were bulk-rejected, as we
       don't know which prescreen (and thus weights) it was based on.
       (currently not computed for them at all)
    """
    @classmethod
    def feature_names(cls):
        return ['single_source', 'pathways', 'knowndrug']

    @classmethod
    def available(cls, ws):
        return True

    def cache_version(self, ws):
        return None

    def fill(self, nparr, ws, wsas):
        from dtk.score_importance import get_score_importances

        logger.info("Fetching %d score importances for %s", len(wsas), ws)
        wsaid_to_score_importances = get_score_importances(wsas)
        from dtk.score_source import ScoreSource
        from collections import defaultdict
        for nparr_row, wsa in zip(nparr, wsas):
            by_source_type = defaultdict(float)
            pathway_sum = 0
            knowndrug_sum = 0

            if wsa.id not in wsaid_to_score_importances:
                # TODO: Figure out what to actually do for scores we can't
                # compute.
                nparr_row[:] = [2, 2, 2]
                continue

            for scoresrc, score in wsaid_to_score_importances[wsa.id].items():
                src = ScoreSource(scoresrc, [])
                if src.is_pathway():
                    pathway_sum += score
                if src.is_otarg() and src.otarg_source() == 'knowndrug':
                    knowndrug_sum += score
                by_source_type[src.source_type()[1]] += score

            singlesrc_sum = max(by_source_type.values())

            nparr_row[:] = [singlesrc_sum, pathway_sum, knowndrug_sum]

class TargetRejectedFeatureSet(FeatureSet):
    """
    Looks at how many times this molecules targets have already been marked
    INACTIVE via other molecules.

    NOTE: this feature isn't quite correct during training.  Ideally for each
    molecule it would only look at other molecules that had been marked before
    it.  That's a bit annoying & slow to do, so we're ignoring it for now.

    It does mean the classifier might overweight this feature; so we try to
    be a bit conservative in its formulation.

    """

    @classmethod
    def feature_names(cls):
        return ['target_rejected']

    def cache_version(self, ws):
        # Ideally we'd look up the targets for these WSAs in case you change
        # the DPI, but that's too slow with the # of WSAs.
        # TODO: Now that we have versioned DPIs, could just look at dpi name.
        return frozenset(self.make_ref_wsas(ws).values_list('id', flat=True))

    @classmethod
    def available(cls, ws):
        return True

    @classmethod
    def make_ref_wsas(cls, ws):
        from browse.models import WsAnnotation
        ref_wsas = WsAnnotation.objects.filter(
                ws=ws,
                indication=WsAnnotation.indication_vals.INACTIVE_PREDICTION,
                )
        return ref_wsas

    @classmethod
    def make_target_to_ref(cls, ws):
        """Find 'reference' targets (inactive predictions)"""
        # TODO: This is quite slow because we're computing the target set of
        # potentially thousands of mols that have been marked inactive, and we
        # need to recompute that each time you mark a new one inactive.
        # Ideally we'd do that incrementally here, but the caching system
        # for these scores doesn't support that - it would need to be custom.
        ref_wsas = cls.make_ref_wsas(ws)
        from collections import defaultdict
        target_to_ref = defaultdict(set)
        from dtk.prot_map import AgentTargetCache
        ref_atc = AgentTargetCache.atc_for_wsas(wsas=ref_wsas, ws=ws)
        for wsa in ref_wsas:
            for (key, uniprot, ev, direction) in ref_atc.raw_info_for_agent(wsa.agent_id):
                target_to_ref[uniprot].add(wsa.id)
        return target_to_ref

    def fill(self, nparr, ws, wsas):
        target_to_ref = self.make_target_to_ref(ws)

        wsa_to_idx = {wsa:idx for idx, wsa in enumerate(wsas)}

        # For each target, get list of molecules with that target.
        from dtk.prot_map import AgentTargetCache
        atc = AgentTargetCache.atc_for_wsas(wsas=wsas, ws=ws)

        for nparr_row, wsa in zip(nparr, wsas):
            targets = [x[1] for x in atc.raw_info_for_agent(wsa.agent_id)]
            if len(targets) == 0:
                score = 0.5
            else:
                num_rejects = [len(target_to_ref[t]) for t in targets]
                # Use the minimum over all its targets.
                eff_rejects = np.min(num_rejects)

                # Don't allow any effect unless it's been rejected multiple
                # times.
                if eff_rejects < 2:
                    eff_rejects = 0

                # Simple linear model with a cutoff.
                LIM_REJECTS = 10
                eff_rejects = min(eff_rejects, LIM_REJECTS)

                # We want 1.0 good, 0.0 bad.
                score = 1.0 - eff_rejects / LIM_REJECTS

            nparr_row[:] = [score]

class TargetExemplarFeatureSet(FeatureSet):
    deps = [SelectivityFeatureSet, NameFeatureSet, AvailabilityFeatureSet]

    def __init__(self, selectivity_fs, name_fs, avail_fs):
        super().__init__()
        self.selectivity_fs = selectivity_fs
        self.name_fs = name_fs
        self.avail_fs = avail_fs

    @classmethod
    def feature_names(cls):
        return ['best_selectivity', 'best_studied']

    @classmethod
    def available(cls, ws):
        return True

    def fill(self, nparr, ws, wsas):
        from dtk.enrichment import get_tie_adjusted_ranks
        # Get the selectivity scores for everything.
        sel_mat = self.selectivity_fs.make_mat(ws, wsas)
        name_mat = self.name_fs.make_mat(ws, wsas)
        avail_mat = self.avail_fs.make_mat(ws, wsas)

        study_mat = np.concatenate([name_mat, avail_mat], axis=1)

        wsa_to_idx = {wsa:idx for idx, wsa in enumerate(wsas)}

        # For each target, get list of molecules with that target.
        from dtk.prot_map import AgentTargetCache
        atc = AgentTargetCache.atc_for_wsas(wsas=wsas, ws=ws)
        targ_wsa_pairs = []
        for wsa in wsas:
            targets = [x[1] for x in atc.raw_info_for_agent(wsa.agent.id)]
            targ_wsa_pairs.extend([[t, wsa] for t in targets])
        from dtk.data import MultiMap
        targ_wsa_mm = MultiMap(targ_wsa_pairs)

        from collections import defaultdict
        def make_wsa_ranks(score_mat):
            wsa_ranks = defaultdict(list)
            for targ, targ_wsas in targ_wsa_mm.fwd_map().items():
                wsa_scores = [(score_mat[wsa_to_idx[wsa]].sum(), wsa) for wsa in targ_wsas]
                wsa_scores.sort(key=lambda x: x[0])

                targ_wsas = [x[1] for x in wsa_scores]
                scores = [x[0] for x in wsa_scores]

                ranks = get_tie_adjusted_ranks(scores, [True] * len(scores))
                ranks = 1.0 - np.array(ranks) / len(scores)
                for targ_wsa, rank in zip(targ_wsas, ranks):
                    wsa_ranks[targ_wsa.id].append(rank)
            return wsa_ranks

        sel_wsa_ranks = make_wsa_ranks(sel_mat)
        study_wsa_ranks = make_wsa_ranks(study_mat)

        for nparr_row, wsa in zip(nparr, wsas):
            sel_ranks, study_ranks = sel_wsa_ranks[wsa.id], study_wsa_ranks[wsa.id]
            sel_score = np.max(sel_ranks) if sel_ranks else 0.0
            study_score = np.max(study_ranks) if study_ranks else 0.0
            nparr_row[:] = [sel_score, study_score]





class DemeritFeatureSet(FeatureSet):
    """
    This feature has several problems at the moment.
    1) The demerits don't correlate well with the expected outcome
    2) It is extremely slow to compute. (not a blocker because of caching)
    3) Features are not applicable to any molecule that has only been marked
       in a single workspace, and there is not an obviously correct replacement.
    4) The metric is tightly coupled with how frequently a molecule shows up
       in our workspaces, which has its own biases that would ideally be
       captured separately.
    """
    TYPES = ['Data Quality', 'Ubiquitous', 'Unavailable', 'Tox']

    @classmethod
    def feature_names(cls):
        return cls.TYPES
    @classmethod
    def available(cls, ws):
        return True
    def cache_version(self, ws):
        # This actually changes any time a new demerit gets added in any
        # workspace... but we can ignore that for now.
        # For our purposes, we only care about changes that occur as you
        # review a given workspace.
        return 0
    def fill(self, nparr, ws, wsas):
        from browse.models import Demerit, WsAnnotation
        demerit_map = dict(Demerit.objects.filter(desc__in=self.TYPES).values_list('desc', 'id'))
        demerit_ids = [demerit_map[t] for t in self.TYPES]
        assert len(demerit_ids) == len(self.TYPES), "Missing a demerit"
        from collections import defaultdict

        for nparr_row, wsa in zip(nparr, wsas):
            relateds = WsAnnotation.objects.filter(indication__gt=0, agent=wsa.agent).exclude(pk=wsa.id)
            related_demerits = defaultdict(int)
            for related in relateds:
                for demerit in related.demerits():
                    related_demerits[demerit] += 1

            N = len(relateds)
            if N > 0:
                nparr_row[:] = [1.0 - related_demerits[dem]/N for dem in demerit_ids]
            else:
                # If this WSA wasn't reviewed in any other workspace, we don't
                # really have a meaningful metric for it.  1.0 isn't too bad.
                # TODO: Really should be inserting a NaN here and replacing
                # elsewhere so that classifiers which support missing data
                # can properly use this.
                nparr_row[:] = [1.0 for dem in demerit_ids]

            if False:
                for d_name, d_id in zip(self.TYPES, demerit_ids):
                    if d_id in related_demerits and wsa.indication >= 14:
                        print(f"{wsa.id} {wsa.ws} {wsa.agent.canonical} has {d_name}")

class VoteFeatureSet(FeatureSet):
    @classmethod
    def feature_names(cls):
        return ['portion_yes_votes']
    @classmethod
    def available(cls, ws):
        return True
    def fill(self, nparr, ws, wsas):
        from browse.models import Vote
        import numpy as np
        qs = Vote.objects.filter(drug__in=wsas)
        qs = qs.prefetch_related('election')
        vts_by_wsa = {wsa:[] for wsa in wsas}
        for x in qs:
            if x.recommended is not None:
                vts_by_wsa[x.drug].append(x.recommended)
        for nparr_row, wsa in zip(nparr, wsas):
            vts = vts_by_wsa[wsa]
            nparr_row[:] = sum(vts)/len(vts) if len(vts) else np.nan

class MoARankFeatureSet(FeatureSet):
    @classmethod
    def feature_names(cls):
        return ['moa_rank_score']
    @classmethod
    def available(cls, ws):
        return True
    def fill(self, nparr, ws, wsas):
        for nparr_row, wsa in zip(nparr, wsas):
            nparr_row[:] = [1]



class WsaTrainingSource:
    """Get training WSAs, anything that was reviewed in prescreen"""
    def get_wsas(self, ws):
        from browse.models import WsAnnotation
        UNCLASSIFIED = WsAnnotation.indication_vals.UNCLASSIFIED
        return WsAnnotation.objects.filter(ws=ws, indication__gt=UNCLASSIFIED)

class WsaGroupSource:
    """Get WSAs in the specified groups"""
    def __init__(self,codes,unreplaced=False):
        self.codes=codes
        self.unreplaced=unreplaced
    def get_wsas(self, ws):
        from browse.models import WsAnnotation
        wsa_ids = set()
        for code in self.codes:
            wsa_ids.update(ws.get_wsa_id_set(code))
        to_ret = WsAnnotation.objects.filter(pk__in=wsa_ids)
        if self.unreplaced:
            from dtk.retrospective import unreplaced_mols
            return unreplaced_mols(to_ret)
        return to_ret


class WsaWzsSource:
    """Gets WSAs from a specific WZS job, usually for predictions."""
    def __init__(self, wzs_jid, count, condensed=False):
        self.wzs_jid = wzs_jid
        self.count = count
        self.condensed = condensed

    def get_wsas(self, ws):
        from flagging.utils import get_target_wsa_ids
        wsa_ids = get_target_wsa_ids(ws, self.wzs_jid, 'wzs', 0, self.count, self.condensed)
        from browse.models import WsAnnotation
        return WsAnnotation.objects.filter(pk__in=wsa_ids)

class WsaIdSource:
    def __init__(self, wsas_or_ids):
        from browse.models import WsAnnotation
        # If it's an empty list provided, just convert to a qs for convenience (so the
        # filters below work).
        if not wsas_or_ids or not isinstance(wsas_or_ids[0], WsAnnotation):
            wsas_or_ids = WsAnnotation.objects.filter(pk__in=wsas_or_ids)
        self._wsas = wsas_or_ids

    def get_wsas(self, ws):
        return self._wsas.filter(ws_id=ws.id)

    def workspaces(self):
        return set([x.ws for x in self._wsas])


class WsaSourceFilter:
    def __init__(self, source, filter_fn):
        self._source = source
        self._filter_fn = filter_fn

    def get_wsas(self, ws):
        return (wsa for wsa in self._source.get_wsas(ws) if self._filter_fn(wsa))


# These featuresets are used as inputs to the selectability model by default.
# They should work across all (training) workspaces and should not have NaN values.
INPUT_FEATURESETS = [SelectivityFeatureSet, NoveltyFeatureSet, UniquenessFeatureSet, AvailabilityFeatureSet, TargetOverlapFeatureSet, NameFeatureSet, DemeritFeatureSet, TargetExemplarFeatureSet, TargetRejectedFeatureSet, NonNovelTargetFeatureSet]


# These additional featuresets can be used for viewing in the feature explorer page.
AVAILABLE_FEATURESETS = INPUT_FEATURESETS + [ScoreDiversityFeatureSet, CMRankFeatureSet, VoteFeatureSet]


def _run(fn_out):
    s = Selectability()
    from browse.models import Workspace
    workspaces = [Workspace.objects.get(pk=61), Workspace.objects.get(pk=43)]
    featuresets = [x() for x in INPUT_FEATURESETS]
    from browse.models import WsAnnotation
    ivals = WsAnnotation.indication_vals
    labelset = IndicationFeatureSet(min_indication=ivals.INITIAL_PREDICTION)
    evalsets = [
        IndicationFeatureSet(min_indication=ivals.INITIAL_PREDICTION),
        IndicationFeatureSet(min_indication=ivals.REVIEWED_PREDICTION),
        IndicationFeatureSet(min_indication=ivals.HIT),
        ]
    data = cross_validate(WsaTrainingSource(), workspaces, featuresets, labelset, evalsets, k=2)
    with open(fn_out, 'wb') as f:
        import pickle
        f.write(pickle.dumps(data))
    #s.train(WsaTrainingSource(), workspaces, featuresets, labelset)

def _plot(fn_in):
    import pickle
    with open(fn_in, 'rb') as f:
        data = pickle.loads(f.read())

    plots = plot_cross_validation(data)
    for i, plot in enumerate(plots):
        logger.info("Writing plot-%d.png", i)
        plot._build_thumbnail("plot-%d.plotly" % i, force=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--plot", help="Plot cross-validation output")
    parser.add_argument("--cross-validate", help="Run a cross-validation")
    args = parser.parse_args()
    if args.plot:
        _plot(args.plot)
    elif args.cross_validate:
        _run(args.cross_validate)
