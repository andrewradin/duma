#!/usr/bin/env python3

import os
import django_setup
from django import forms

from tools import ProgressWriter
from runner.process_info import StdJobInfo, StdJobForm, JobInfo
import runner.data_catalog as dc

import logging
logger = logging.getLogger(__name__)


def extract_jids_and_codes(ws, jobs_and_codes):
    """Used for both the CM and the ProteinScoreView."""
    jids_and_codes = []
    for jid_code in jobs_and_codes:
        jid, code = jid_code.split('_')
        if code == 'refreshwf':
            ss = JobInfo.get_bound(ws, jid).get_scoreset()
            for name, ssjid in ss.job_type_to_id_map().items():
                ssbji = JobInfo.get_bound(ws, ssjid)
                dc = ssbji.get_data_catalog()
                for code in dc.get_codes('uniprot', ''):
                    jids_and_codes.append((ssjid, code))
        else:
            jids_and_codes.append((jid, code))
    return jids_and_codes

def wzs_weighted_uniprot_scores(ws, wzs_jid, jids_and_codes):
    """Applies wzs normalization and weighting to a set of uniprot scores.

    For each jid & code in
    
    Used for both the CM and the ProteinScoreView.
    """
    import dtk.features as feat
    attrs = []
    options = {'job_labels': {}}

    wzs_bji = JobInfo.get_bound(ws, wzs_jid)
    score_weights = wzs_bji.get_score_weights()
    weights = {k.lower():v for k, v in score_weights}
    weight_idxs = {k.lower():i for i, (k, v) in enumerate(score_weights)}
    names = []

    bjis = {}
    for jid, code in jids_and_codes:
        bji = JobInfo.get_bound(ws, jid)
        bjis[jid] = bji

    feature_idxs = []
    for jid, code in jids_and_codes:
        bji = bjis[jid]
        name = f'{bji.job.role}_{code}'
        if name.lower() not in weights:
            continue
        feature_idxs.append(weight_idxs[name.lower()])
        names.append(name)
        name_parts = name.split('_')
        dc_code = name_parts[-1]
        attrs.append(f'{jid}_{dc_code}')
        options['job_labels'][jid] = '_'.join(name_parts[:-1]) + str(jid)

    logger.info(f"Setting up fm with {options['job_labels']}")
    spec = feat.DCSpec(ws.id, attrs, **options)
    fm = feat.FMBase.load_from_recipe(spec)
    logger.info("Created fm with shape %s", fm.data.shape)
    uniprots = fm.sample_keys


    from scripts.wzs import make_agg_model
    agg_model = make_agg_model(wzs_bji.parms, fm=fm)
    weights = wzs_bji.get_full_weights(feature_idxs)
    normed = agg_model.post_norm(weights)
    # Normally we'd call agg_model.score here, but that sums over
    # all cols, and we want to keep the separate columns.
    scored = normed * agg_model.score_weights(weights)

    datas = []
    for col_idx in range(normed.shape[1]):
        def gen():
            for row_idx in range(normed.shape[0]):
                yield uniprots[row_idx], (scored[row_idx][col_idx],)
        datas.append(list(gen()))
    return datas, names

class MyJobInfo(StdJobInfo):
    descr= 'Generates a composite uniprot signature from other scores'
    short_label = 'Composite Sig'
    page_label = 'Composite Signature'

    def make_job_form(self, ws, data):

        wf_choices = ws.get_prev_job_choices('wf')
        uniprot_choices = []
        for jid, desc in wf_choices:
            from runner.process_info import JobInfo
            bji = JobInfo.get_bound(self.ws, jid)
            if 'RefreshFlow' not in bji.job.name:
                continue

            from dtk.prot_map import DpiMapping
            dpi = DpiMapping(bji.parms['p2d_file'])
            mapping_type = dpi.mapping_type()
            if mapping_type != 'uniprot':
                continue
                
            uniprot_choices.append((jid, desc))


        class MyForm(StdJobForm):
            wzs_jid=forms.ChoiceField(
                    label='WZS JobID',
                    choices = ws.get_prev_job_choices('wzs'),
                    required=True,
                 )
            uniprot_wf=forms.ChoiceField(
                    label='Uniprot WF',
                    choices=uniprot_choices,
                    required=True,
                 )
        return MyForm(ws, data)

    def get_data_code_groups(self):
        return [
                dc.CodeGroup('uniprot',self._std_fetcher('outfile'),
                        dc.Code('ev',label='Evidence'),
                        )
                ]

    def __init__(self,ws=None,job=None):
        super().__init__(ws=ws, job=job, src=__file__)
        if self.job:
            self.outfile = os.path.join(self.lts_abs_root, 'signature.tsv')

    def run(self):
        self.make_std_dirs()
        p_wr = ProgressWriter(self.progress, [
                "compute scores",
                "finalize",
                ])
        self.compute_scores()
        p_wr.put("compute scores","complete")
        self.finalize()
        p_wr.put("finalize","complete")
    
    def compute_scores(self):
        wf_code = f'{self.parms["uniprot_wf"]}_refreshwf'
        jids_and_codes = extract_jids_and_codes(self.ws, [wf_code])
        data, names = wzs_weighted_uniprot_scores(
            ws=self.ws,
            wzs_jid=self.parms['wzs_jid'],
            jids_and_codes=jids_and_codes,
            )
        
        uni_pairs = []
        for col in data:
            for uniprot, (ev,) in col:
                uni_pairs.append((uniprot, ev))


        from dtk.data import kvpairs_to_dict
        uniprot2scores = kvpairs_to_dict(uni_pairs)

        mapped_data = {}
        import numpy as np
        for uniprot, scores in uniprot2scores.items():
            score = np.mean(scores)
            mapped_data[uniprot] = score

        from atomicwrites import atomic_write
        with atomic_write(self.outfile) as f:
            f.write('\t'.join(['uniprot', 'ev']) + '\n')
            for prot, ev in sorted(mapped_data.items(), key=lambda x: (-x[1], x[0])):
                row = [prot, str(ev)]
                f.write('\t'.join(row) + '\n')

if __name__ == "__main__":
    MyJobInfo.execute(logger)
