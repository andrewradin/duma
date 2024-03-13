#!/usr/bin/env python3

import sys
import six
from path_helper import PathHelper,make_directory

import os
import django
import django_setup

from django import forms

from tools import ProgressWriter
from runner.process_info import JobInfo, StdJobInfo, StdJobForm
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager
import runner.data_catalog as dc
from dtk.prot_map import DpiMapping, PpiMapping
from dtk.cache import cached

import json
import logging
logger = logging.getLogger("algorithms.run_selectability")


@cached()
def get_wsa_uniprots(wsa_ids, dpi_choice):
    from dtk.prot_map import DpiMapping, AgentTargetCache
    from browse.models import WsAnnotation
    from drugs.models import Drug

    agent_ids = list(Drug.objects.filter(wsannotation__in=wsa_ids).distinct().values_list('id', flat=True))
    ref_atc = AgentTargetCache(
        agent_ids=agent_ids,
        mapping=DpiMapping(dpi_choice),
        dpi_thresh=DpiMapping.default_evidence,
    )

    uniprots = set()
    for agent_id in agent_ids:
        for (key, uniprot, ev, direction) in ref_atc.raw_info_for_agent(agent_id):
            uniprots.add(uniprot)
    return uniprots


def apply_heuristics(wsas_and_scores, ws, eff_jid):
    """Applies some score heuristics for the live selectability scores."""
    from browse.models import WsAnnotation
    from dtk.target_importance import get_target_importances
    from collections import defaultdict
    wsa_ids = [x[0].id for x in wsas_and_scores]

    ivals = WsAnnotation.indication_vals
    ref_wsa_ids = list(WsAnnotation.objects.filter(
        ws=ws,
        dispositionaudit__indication=ivals.INITIAL_PREDICTION,
    ).values_list('id', flat=True).distinct())

    ref_prots = get_wsa_uniprots(sorted(ref_wsa_ids), ws.get_dpi_default())
    nonnovel_prots = set(ws.get_uniprot_set(ws.get_nonnovel_ps_default()))

    wsa_ids = [wsa.id for wsa, score in wsas_and_scores]
    trg_imps = get_target_importances(ws.id, eff_jid, wsa_ids, cache_only=True)



    filter_stats = defaultdict(int)
    out = []
    for wsa, score in wsas_and_scores:
        prot2imp, errs = trg_imps[wsa.id]
        
        sum_reviewed = 0
        sum_nonnovel = 0
        for prot, imp in prot2imp.items():
            if prot in ref_prots:
                sum_reviewed += imp
            elif prot in nonnovel_prots:
                sum_nonnovel += imp
        
        num_targs = len(prot2imp)

        if sum_reviewed > 0.6:
            filter_stats['reviewed'] += 1
            score = -1.
        if sum_nonnovel > 0.6:
            filter_stats['nonnovel'] += 1
            score = -1.
        if sum_reviewed + sum_nonnovel > 0.6:
            filter_stats['review+nonnovel'] += 1
            score = -1.
        if num_targs > 10 and max(prot2imp.values(), default=0) < 0.6:
            filter_stats['targcount'] += 1
            score = -1.
        out.append((wsa, score))
    logger.info(f"Heuristics results for {len(wsa_ids)}: {dict(filter_stats)}")
    return out
        


class MyJobInfo(StdJobInfo):
    short_label = 'Selectability'
    page_label = 'Selectability'
    descr = 'Computes a score predicting how good of a selection this drug would be.'
    def make_job_form(self, ws, data):
        model_choices = ws.get_prev_job_choices('selectabilitymodel')
        ordering_choices = ws.get_prev_job_choices('wzs')

        class MyForm(StdJobForm):
            model_jid = forms.ChoiceField(
                    label='Selectability Model',
                    choices=model_choices,
                    )
            wzs_jid = forms.ChoiceField(
                    label='Ordering JID',
                    choices=ordering_choices,
                    )
            count = forms.IntegerField(
                    label='# to examine',
                    initial=800,
                    )
            condensed = forms.BooleanField(
                    label='Count via condensed',
                    initial=True,
                    required=False,
                    )
        return MyForm(ws, data)

    def get_data_code_groups(self):
        class DeferredScore:
            """A hack to batch up live score computations.

            It is much faster to compute scores in bulk, but the calc interface
            wants them one-by-one.  This class is ordered and convertable to
            a float, but doesn't actually compute its value until required,
            which allows us to collect the set of WSAs of interest beforehand.

            We're intentionally not providing ordering or arithmetic; the
            caller should float() these once they're collected (usually this
            will be data_catalog).
            """
            pending_wsas = set()
            computed = {}
            def __init__(self, wsa_id):
                self.pending_wsas.add(wsa_id)
                self._wsa_id = wsa_id
            def __float__(self):
                if self._wsa_id not in self.computed:
                    self.compute()
                # If a WSA is hidden, it won't get computed.
                return self.computed.get(self._wsa_id, 0.)
            def calc(self):
                return float(self)
            @classmethod
            def compute(cls):
                wsa_ids = cls.pending_wsas
                if not wsa_ids:
                    return
                logger.info("Computing %d", len(wsa_ids))
                try:
                    for wsa, score in self.compute_live_scores(list(wsa_ids)):
                        cls.computed[wsa.id] = score
                    cls.pending_wsas = set()
                except Exception as e:
                    # Explicitly print any exceptions here, they tend to get
                    # eaten somewhere up the stack.
                    import traceback
                    traceback.print_exc()
                    raise
                logger.info("Done")

        def calc_func(wsa_id):
            return DeferredScore(wsa_id)
        return [
                dc.CodeGroup('wsa',self._std_fetcher('outfile'),
                        # Need to include WSA explicitly as a code to use it
                        # as the key for calc.  But we can hide it.
                        dc.Code('wsa',label='wsa',efficacy=False,hidden=True,valtype='int'),
                        dc.Code('selectability',label='Selectability of drug',efficacy=False),
                        dc.Code('liveselectability', label='Recomputed on the fly',
                            calc=(calc_func, 'wsa')),
                        )
                ]

    def __init__(self,ws=None,job=None):
        super().__init__(ws=ws, job=job, src=__file__)
        if self.job:
            self.outfile = os.path.join(self.lts_abs_root, 'selectability.tsv')

    def run(self):
        self.make_std_dirs()
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "compute_scores",
                "finalize",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        logger.info("Computing scores")
        self.compute_scores()
        logger.info("Finalizing")
        p_wr.put("compute_scores","complete")
        self.finalize()
        p_wr.put("finalize","complete")

    def compute_live_scores(self, wsa_ids):
        model_jid = self.parms['model_jid']
        from runner.process_info import JobInfo
        bji = JobInfo.get_bound(self.ws, model_jid)
        try:
            model = bji.load_trained_model()
            from dtk.selectability import Selectability, WsaWzsSource, MLModel, WsaIdSource
            s = Selectability()
            workspaces = [self.ws]
            wsa_source = WsaIdSource(wsa_ids)
            featuresets = model.featuresets
            wsa_scores = s.predict(model, wsa_source, workspaces, featuresets)

            wsa_scores = apply_heuristics(list(wsa_scores), self.ws, self.parms.get('wzs_jid'))
        except FileNotFoundError:
            # This is a relic of switching model representations (due to sklearn version incompat) which
            # means some of our older selectabilitymodels can't be loaded.  Since we still have prescreens
            # keyed on them, we still want to be able to generate those scores, but since they're old it doesn't
            # really matter if they're recomputed vs baked in.
            logger.warn("Couldn't find selectability model for live scores, falling back to baked in scores")
            from browse.models import WsAnnotation
            wsas = {wsa.id:wsa for wsa in WsAnnotation.all_objects.filter(pk__in=wsa_ids)}
            lu = dict(self.get_data_catalog().get_ordering('selectability', True))
            wsa_scores = [(wsas[wsa_id], lu.get(wsa_id, 0)) for wsa_id in wsa_ids]
        return wsa_scores

    def wzs_jid(self):
        return self.parms['wzs_jid']

    def compute_scores(self):
        model_jid = self.parms['model_jid']
        logger.info(f"Computing scores for model {model_jid} in ws {self.ws.id}")
        from runner.process_info import JobInfo
        bji = JobInfo.get_bound(self.ws, model_jid)
        from dtk.selectability import Selectability, WsaWzsSource, MLModel
        model = bji.load_trained_model()
        s = Selectability()
        workspaces = [self.ws]
        wsa_source = WsaWzsSource(self.parms['wzs_jid'], self.parms['count'], self.parms['condensed'])
        wsa_scores = s.predict(model, wsa_source, workspaces, model.featuresets, force_refresh=True)

        from atomicwrites import atomic_write
        with atomic_write(self.outfile) as f:
            f.write('\t'.join(['wsa', 'selectability']) + '\n')
            for wsa, score in wsa_scores:
                row = [str(wsa.id), "%.4f" % score]
                f.write('\t'.join(row) + '\n')


if __name__ == "__main__":
    MyJobInfo.execute(logger)
