#!/usr/bin/env python3

import sys

from path_helper import PathHelper,make_directory

import os
import django
import django_setup

from django import forms

from tools import ProgressWriter
from runner.process_info import JobInfo
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager
import runner.data_catalog as dc
from dtk.prot_map import DpiMapping

import json
import logging
logger = logging.getLogger("algorithms.run_dnc_precompute")



class MyJobInfo(JobInfo):
    def make_job_form(self, ws, data):
        class MyForm(forms.Form):
            wsas_text = forms.CharField(
                    widget=forms.Textarea(),
                    label='WSAs to run (newline-separated)',
                    required=False,
                    initial=''
                    )

            def as_html(self):
                from django.utils.html import format_html
                return format_html('''
                        <table>{}</table>
                        '''
                        ,self.as_table()
                        )
            def as_dict(self):
                # this returns the settings_json for this form; it may differ
                # from what appears in the user interface; there are 2 use
                # cases:
                # - extracting default initial values from an unbound form
                #   for rendering the settings comparison column
                # - extracting user-supplied values from a bound form
                if self.is_bound:
                    src = self.cleaned_data
                else:
                    src = {fld.name:fld.field.initial for fld in self}
                p ={'ws_id':ws.id}
                for f in self:
                    key = f.name
                    value = src[key]
                    p[key] = value
                return p
            def from_json(self,init):
                p = json.loads(init)
                for f in self:
                    if f.name in p:
                        f.field.initial = p[f.name]
        return MyForm(data)

    def description_text(self):
        return '''
        Precomputes drugnote collections for drugs in a review round.
        '''
    def settings_defaults(self,ws):
        cfg=self.make_job_form(ws, None)
        return {
                'default':cfg.as_dict(),
                }
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        form = self.make_job_form(ws, None)
        if copy_job:
            form.from_json(copy_job.settings_json)
        return form.as_html()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        form = self.make_job_form(ws, post_data)
        if not form.is_valid():
            return (form.as_html(),None)
        settings = form.as_dict()
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(settings)
                            )
        next_url = ws.reverse('nav_progress',job_id)
        return (None, next_url)
    def role_label(self):
        return self._upstream_role_label()

    def __init__(self,ws=None,job=None):
        # base class init
        self.use_LTS=True
        super(MyJobInfo,self).__init__(
                    ws,
                    job,
                    __file__,
                    "DrugNoteCollection Precompute",
                    "DrugNoteCollection Precompute",
                    )
        # any base class overrides for unbound instances go here
        self.publinks = (
                )
        self.qc_plot_files = (
                )
        self.needs_sources = False
        self.ws = ws
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.ec = ExitCoder()

    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        make_directory(self.lts_abs_root)
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "run",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.run_precompute()
        p_wr.put("run","complete")

    def run_precompute(self):
        from algorithms.run_trgscrimp import parse_wsa_list
        from dtk.composer import DrugNoteCollection, EvidenceComposer, DrugNoteCache
        from browse.models import WsAnnotation, Prescreen

        for appendix in [True, False]:
            wsa_ids = parse_wsa_list(self.parms['wsas_text'])
            wsas = WsAnnotation.objects.filter(pk__in=wsa_ids)
            cache = DrugNoteCache
            ec = EvidenceComposer(self.ws)
            def run_wsa(wsa):
                pscr = wsa.marked_prescreen
                if not pscr:
                    logger.warning(f"WARNING: {wsa} has no pscr")
                logger.info(f"Running {wsa}")
                dnc=DrugNoteCollection(wsa,False)
                ec.extract_evidence(wsa,dnc,appendix=appendix)
                cache.store(wsa, dnc, appendix=appendix)
                for repl_wsa in wsa.replacement_for.all():
                    run_wsa(repl_wsa)

            for wsa in wsas:
                run_wsa(wsa)


if __name__ == "__main__":
    MyJobInfo.execute(logger)
