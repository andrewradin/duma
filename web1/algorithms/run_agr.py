#!/usr/bin/env python3

import os
import sys
import django
if 'django.core' not in sys.modules:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    django.setup()

import logging
logger = logging.getLogger("algorithms.run_agr")

import json
from path_helper import PathHelper,make_directory
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo, StdJobInfo, StdJobForm

from django import forms

class MyJobInfo(StdJobInfo):
    short_label="AGR"
    page_label="Alliance Genome Scores"
    descr='Extracts disease associated gene scores from Alliance Genome'
    def make_job_form(self,ws,data):
        # the workspace name is never the correct disease default for this
        # CM, so only populate the disease field if a default has been set
        (val,dd)=ws.get_disease_default('AGR',return_detail=True)
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
                                        For Disease, enter a pipe-separated
                                        list of DOIDs. To interactively
                                        search for IDs, click ''',
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
                        dc.Code('agrs',label='Alliance Genome score',
                                fmt='%.2e'),
                        ),
                ]
    def run(self):
        self.make_std_dirs()
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "fetch data",
                "get disease key(s)",
                "extract data",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.load_data()
        p_wr.put("fetch data","complete")
        if not self.get_key():
            p_wr.put("get disease key(s)","NO EXACT MATCH; SEE LOG FOR DETAILS")
            sys.exit(1)
        p_wr.put("get disease key(s)","complete (%s)"%", ".join(self.keys))
        self.extract_data()
        p_wr.put("extract data","scored %d proteins"%self.found)
        self._plot()
        self.finalize()
    def load_data(self):
        vdefaults=self.ws.get_versioned_file_defaults()
        from dtk.s3_cache import S3File
        file_class = 'agr'
        s3f = S3File.get_versioned(
                file_class,
                vdefaults[file_class],
                role='human',
                )
        s3f.fetch()
        from dtk.files import get_file_records
        src = get_file_records(s3f.path(), keep_header=True)
        from dtk.readtext import convert_records_using_header
        self.agr_recs = list(convert_records_using_header(src))
        self.valid_doids = set(x.DOID for x in self.agr_recs)
    def get_key(self):
        missing = []
        self.keys = set()
        for k in self.parms['disease'].split('|'):
            k = k.strip()
            if k in self.valid_doids:
                self.keys.add(k)
            else:
                missing.append(k)
        if missing:
            # at least one specified disease code was not an exact match
            fmt_missing = ", ".join(missing)
            print(f"""
Supplied code(s) '{fmt_missing}' didn't exactly match an Alliance Genome
disease. It may be that there are no genes associated with that code.
This can be verified using the look up link on the start page.
Otherwise, delete the code and try with any remaining codes.
""")
            return False
        else:
            with open(self.key_used_file, 'w') as f:
                f.write("\n".join(self.keys) + "\n")
        return True
    def extract_data(self):
        # The idea here is that each protein has multiple pieces of evidence
        # in AGR, and the chances that the protein is uninteresting is
        # the chance that all these pieces of evidence are misleading.
        # So we:
        # - assign a correctness probability to each evidence type
        # - calculate the overall incorrectness probability as the product
        #   of 1-p for each piece of evidence
        # - report a score of 1 - the overall incorrectness probability
        d = {}
        score_lookup={
                ("is_marker_for",
                    "expression pattern evidence",
                    "manual assertion"):0.9,
                ("is_implicated_in",
                    "assay evidence",
                    "manual assertion"):0.7,
                ("is_implicated_in",
                    "interaction evidence",
                    "manual assertion"):0.7,
                ("is_implicated_in",
                    "inference by association of genotype from phenotype",
                    "manual assertion"):0.7,
                ("is_implicated_in",
                    "mutant phenotype evidence",
                    "manual assertion"):0.7,
                ("biomarker_via_orthology",
                    "evidence",
                    "automatic assertion"):0.4,
                ("implicated_via_orthology",
                    "evidence",
                    "automatic assertion"):0.2,
                }
        for ar in self.agr_recs:
            if ar.DOID not in self.keys:
                continue
            score_inputs = tuple(
                    [ar.AssociationType]+
                    ar.EvidenceDescription.split(" used in ")
                    )
            score = score_lookup.get(score_inputs)
            if score is not None:
                d.setdefault(ar.Uniprot,[]).append(score)
        from dtk.files import get_file_records,FileDestination
        # write out any matches
        self.values = []
        with FileDestination(self.score_fn,['uniprot']+['agrs']) as dest:
            import operator
            import functools
            for uniprot,scores in d.items():
                combined = 1 - functools.reduce(
                        operator.mul,
                        [1-x for x in scores]
                        )
                self.values.append(combined)
                dest.append([uniprot,combined])
        self.found = len(d)
    def _plot(self):
        from dtk.plot import smart_hist
        pp2 = smart_hist(self.values,
                      layout=
                          {'title': 'Protein evidence histogram',
                           'yaxis':{'title':'Protein count',
                              },
                           'xaxis':{'title':'AGR score'},
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
            #enabled_default=uji.disease_name_set(ws,'AGR')
            enabled_default=uji.data_status_ok(ws,'AGR','AGR Values')
            # WARNING TO ANYONE DOING COPY/PASTE to create a new CM:
            # This version of add_to_workflow() contains lot of complications
            # related to otarg, where it was copied from. It now needs to
            # stay here for backwards compatibility. If those complications
            # are relevant to your new CM, copy away, and be
            # sure that multi_uniprot_scores is set to True.
            # If not, you may want to work from the much simpler
            # version of add_to_workflow() in, e.g. gwasig.
            def add_to_workflow(self,wf):
                assert uji.multi_uniprot_scores
                wf.eff.hold('agr')
                cm_info = wf.std_wf.get_main_cm_info(uji.job_type)
                self.cm_info = cm_info
                cm_info.pre.add_pre_steps(wf)
                from dtk.workflow import AGRStep,LocalCode
                my_name='agr'
                self.agr_name = my_name
                AGRStep(wf,my_name,
                        disease=wf.ws.get_disease_default('AGR'),
                        )
                LocalCode(wf,'agr_followup',
                        func=self._do_agr_followup,
                        inputs={my_name:True},
                        )
            def _do_agr_followup(self,my_step):
                wf = my_step._workflow
                from runner.process_info import JobInfo
                bji = JobInfo.get_bound(wf.ws, wf.step(self.agr_name).job_id)
                cat = bji.get_data_catalog()
                # get index of first job we're about to create
                start = len(wf._order)
                # add post-processing for each code except 'overall'
                for code in cat.get_codes('uniprot','score'):
                    self.cm_info.post.add_post_steps(
                            wf,
                            self.agr_name+':'+code+'_agr',
                            code,
                            )
                # schedule a completion step when all jobs are done
                from dtk.workflow import LocalCode
                LocalCode(wf,'agr_complete',
                        func=self._do_agr_complete,
                        inputs={k:True for k in wf._order[start:]},
                        )
            def _do_agr_complete(self,my_step):
                wf = my_step._workflow
                wf.eff.release('agr')
        parts.append(MyWorkflowPart())

if __name__ == "__main__":
    MyJobInfo.execute(logger)
