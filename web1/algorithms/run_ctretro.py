#!/usr/bin/env python3


import os
import django_setup
from django import forms

from tools import ProgressWriter
from runner.process_info import StdJobInfo, StdJobForm, JobInfo
from reserve import ResourceManager
import runner.data_catalog as dc
from dtk.prot_map import DpiMapping, PpiMapping

import json
import logging
logger = logging.getLogger(__name__)

class MyJobInfo(StdJobInfo):
    descr= 'Archive data for Retrospective CT predictions'
    short_label = 'CT Retro'
    page_label = 'CT Retrospective Data'
    # Although this takes input from WZS jobs, it doesn't do it in a way
    # that affects the role name, so we don't need to define the upstream_jid
    # stuff.
    # Also, although these job runs act as a data source for the xws CT
    # Retrospective Predictions page, they return data via a custom
    # mechanism, and don't define any Data Catalog elements.

    def make_job_form(self, ws, data):
        from browse.models import ScoreSet
        # We only want scoresets that have all 3 WZS roles. The 3 separate
        # filter() calls are needed to allow different jobs to meet each
        # requirement.
        qs = ScoreSet.objects.filter(
                ws=ws,
                desc='RefreshFlow',
                scoresetjob__job_type='wzs-train',
                ).filter(
                scoresetjob__job_type='wzs-test',
                ).filter(
                scoresetjob__job_type='wzs',
                ).order_by('-id')
        from dtk.text import fmt_time
        class MyForm(StdJobForm):
            scoreset = forms.ChoiceField(
                    label='Source Scoreset',
                    choices = [(x.id,f'{x.id} {x.user}@{fmt_time(x.created)}')
                            for x in qs
                            ],
                    )

        return MyForm(ws, data)

    def __init__(self,ws=None,job=None):
        super().__init__(ws=ws, job=job, src=__file__)
        if self.job:
            self.moldata_fn = os.path.join(self.lts_abs_root, 'moldata.tsv')

    def run(self):
        self.make_std_dirs()
        self.run_steps([
                # this is so fast, don't bother with reserve
                ('gather data',self.gather_data),
                ('finalize',self.finalize),
                ])
    
    def gather_data(self):
        scoreset_id = int(self.parms['scoreset'])
        from dtk.ct_predictions import BulkFetcher, RetroMolData
        bulk_fetch = BulkFetcher(scoreset_id = scoreset_id)
        molecules = [
                RetroMolData(wsa_id=wsa_id,bulk_fetch=bulk_fetch)
                for wsa_id in bulk_fetch.all_wsa_ids
                ]
        # make sure all molecules have MOAs
        original = list(molecules)
        molecules = [x for x in molecules if x.moa_wsa]
        # write to TSV
        RetroMolData.write_tsv(self.moldata_fn,molecules)


if __name__ == "__main__":
    MyJobInfo.execute(logger)
