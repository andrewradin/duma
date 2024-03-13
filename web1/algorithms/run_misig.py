#!/usr/bin/env python3

import os
import sys
import django
if 'django.core' not in sys.modules:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    django.setup()

import logging
logger = logging.getLogger(__name__)

import json
from path_helper import PathHelper,make_directory
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo, StdJobInfo, StdJobForm

from django import forms

class MyJobInfo(StdJobInfo):
    short_label="MISig"
    page_label="Monarch Signature"
    descr='Extract genes assocaited with phenotypes associated with disease from the Monarch Initiative'
    def make_job_form(self,ws,data):
        # the workspace name is never the correct disease default for this
        # CM, so only populate the disease field if a default has been set
        (val,dd)=ws.get_disease_default('Monarch',return_detail=True)
        if dd is None:
            val=''
        class ConfigForm(StdJobForm):
            disease = forms.CharField(
                    max_length=2048,
                    initial=val,
                    widget=forms.Textarea
                    )
            def as_html(self):
                from dtk.html import join,link,tag_wrap,ulist
                return join(
                        tag_wrap('div',
                                join('''
                                        Disease terms should be set ''',
                                    link(
                                        'here',
                                        ws.reverse('nav_disease_names'),
                                        ),
                                ),
                                attr={'class':'well'},
                                ),
                        self.as_p(),
                        )
        return ConfigForm(ws,data)
    def __init__(self,ws=None,job=None):
        self.use_LTS=True
        # base class init
        super().__init__(ws, job, src=__file__)
        self.publinks = (
            (None, 'score_hist.plotly'),
        )
        self.qc_plot_files = (
                'score_hist.plotly',
                )
        # any base class overrides for unbound instances go here
        # job-specific properties
        if self.job:
            self.key_used_file = self.outdir +'key_used'
            self.score_fn = self.lts_abs_root+'scores.tsv'
            self.score_hist_plot = self.tmp_pubdir+"score_hist.plotly"
    def is_a_source(self):
        return True
    def get_data_code_groups(self):
        import runner.data_catalog as dc
        return [
                dc.CodeGroup('uniprot',self._std_fetcher('score_fn'),
                        dc.Code('misig',label='Monarch Signature',
                                fmt='%.2e'),
                        ),
                ]
    def run(self):
        self.make_std_dirs()
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "build signature data",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.load_data()
        p_wr.put("build signature data","complete")
        self._plot()
        self.finalize()
    def load_data(self):
        from dtk.monarch import MonarchDis, MonarchGene, score_mips
        from dtk.files import FileDestination
        MD=MonarchDis(self.ws.get_versioned_file_defaults(), self.ws)
        ordering, phenoIDs_to_names = MD.get_mips_data(self.parms['disease'])
        MG=MonarchGene(self.ws.get_versioned_file_defaults(), self.ws)
        mips_data = MG.get_mips_data(phenoIDs_to_names.keys())
        all_prot_score={}
        for pheno, score in ordering:
            for uniprot in mips_data[pheno]:
                if uniprot not in all_prot_score:
                    all_prot_score[uniprot]=[]
                all_prot_score[uniprot].append(score_mips(mips_data[pheno][uniprot], score))
        # now combine all scores for each uniprot
        self.values=[]
        with FileDestination(self.score_fn,['uniprot']+['misig']) as dest:
            import operator
            import functools
            for uniprot,scores in all_prot_score.items():
                combined = 1 - functools.reduce(
                        operator.mul,
                        [1-x for x in scores]
                        )
                self.values.append(combined)
                dest.append([uniprot,combined])
    def _plot(self):
        if not self.values:
            return
        from dtk.plot import smart_hist
        pp2 = smart_hist(self.values,
                      layout=
                          {'title': 'Protein evidence histogram',
                           'yaxis':{'title':'Protein count',
                              },
                           'xaxis':{'title':'MISig score'},
                          }
                      )
#        pp2._layout['annotations'] = [fig_legend([
#                                      'This plot quantifies the number of proteins with a score'
#                                      ,'in each OpenTargets score type. A minimum of ~20 proteins'
#                                      ,'should be present for a score to be used downstream.'
#                                      ], -0.175)]
        pp2.save(self.score_hist_plot)
    multi_uniprot_scores = True
    def add_workflow_parts(self,ws,parts):
        jobnames = self.get_jobnames(ws)
        assert len(jobnames) == 1
        jobname=jobnames[0]
        uji = self # simplify access inside MyWorkflowPart
        class MyWorkflowPart:
            label=uji.source_label(jobname)
            # WARNING TO ANYONE DOING COPY/PASTE to create a new CM:
            # This version of add_to_workflow() contains lot of complications
            # related to otarg, where it was copied from. It now needs to
            # stay here for backwards compatibility. If those complications
            # are relevant to your new CM, copy away, and be
            # sure that multi_uniprot_scores is set to True.
            # If not, you may want to work from the much simpler
            # version of add_to_workflow() in, e.g. gwasig.
            enabled_default=uji.data_status_ok(ws,'Monarch','Monarch Initiative Values')
            def add_to_workflow(self,wf):
                assert uji.multi_uniprot_scores
                wf.eff.hold('misig')
                cm_info = wf.std_wf.get_main_cm_info(uji.job_type)
                self.cm_info = cm_info
                cm_info.pre.add_pre_steps(wf)
                from dtk.workflow import MISigStep,LocalCode
                my_name='misig'
                self.misig_name = my_name
                MISigStep(wf,my_name,
                        disease=wf.ws.get_disease_default('Monarch'),
                        )
                LocalCode(wf,'misig_followup',
                        func=self._do_misig_followup,
                        inputs={my_name:True},
                        )
            def _do_misig_followup(self,my_step):
                wf = my_step._workflow
                from runner.process_info import JobInfo
                bji = JobInfo.get_bound(wf.ws, wf.step(self.misig_name).job_id)
                cat = bji.get_data_catalog()
                # get index of first job we're about to create
                start = len(wf._order)
                # add post-processing for each code except 'overall'
                for code in cat.get_codes('uniprot','score'):
                    self.cm_info.post.add_post_steps(
                            wf,
                            self.misig_name+':'+code+'_misig',
                            code,
                            )
                # schedule a completion step when all jobs are done
                from dtk.workflow import LocalCode
                LocalCode(wf,'misig_complete',
                        func=self._do_misig_complete,
                        inputs={k:True for k in wf._order[start:]},
                        )
            def _do_misig_complete(self,my_step):
                wf = my_step._workflow
                wf.eff.release('misig')
        parts.append(MyWorkflowPart())

if __name__ == "__main__":
    MyJobInfo.execute(logger)
