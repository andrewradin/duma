#!/usr/bin/env python3

import os
import sys
import django
if 'django.core' not in sys.modules:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    django.setup()

import logging
logger = logging.getLogger("algorithms.run_dgn")

import json
from path_helper import PathHelper,make_directory
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo, StdJobInfo, StdJobForm

from django import forms

class MyJobInfo(StdJobInfo):
    short_label="DisGeNet"
    page_label="DisGeNet Scores"
    descr='Extracts disease associated genes and their scores from DisGeNet'
    def make_job_form(self,ws,data):
        # the workspace name is never the correct disease default for this
        # CM, so only populate the disease field if a default has been set
        (val,dd)=ws.get_disease_default('DisGeNet',return_detail=True)
        if dd is None:
            val=''
        class ConfigForm(StdJobForm):
            disease = forms.CharField(
                    max_length=256,
                    initial=val,
                    )
            def as_html(self):
                from dtk.html import join,link,tag_wrap,ulist
                return join(
                        tag_wrap('div',
                                join('''
                                        For Disease, enter a comma-separated
                                        list of UMLS IDs. To interactively
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
# I haven't found a corresponding page yet
#            if os.path.isfile(self.key_used_file):
#                with open(self.key_used_file, 'r') as f:
#                    ks = [l.strip() for l in f]
#                self.otherlinks = [('OpenTargets page - '+k,
#                                    'https://www.targetvalidation.org/disease/%s/associations' %(k)
#                                   )
#                                   for k in ks
#                                  ]
    def is_a_source(self):
        return True
    def get_data_code_groups(self):
        import runner.data_catalog as dc
        return [
                dc.CodeGroup('uniprot',self._std_fetcher('score_fn'),
                        dc.Code('dgns',label='DisGeNet score', fmt='%.2e'),
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
### TODO extract data
        from dtk.disgenet import DisGeNet
        self.dgn = DisGeNet(
                self.ws.get_versioned_file_defaults(),
                ontology='umls',
                )
        p_wr.put("fetch data","complete")
        if not self.get_key():
            p_wr.put("get disease key(s)","NO EXACT MATCH; SEE LOG FOR DETAILS")
            sys.exit(1)
        p_wr.put("get disease key(s)","complete (%s)"%", ".join(self.keys))
        self.extract_data()
        p_wr.put("extract data","scored %d proteins"%self.found)
        self._plot()
        self.finalize()
    def get_key(self):
        keys = self.parms['disease'].split(',')
        missing = []
        self.keys = []
        for k in keys:
            k = k.strip()
            if k in self.dgn.data:
                self.keys.append(k)
            else:
                missing.append(k)
        if missing:
            # at least one specified disease code was not an exact match
            print("Supplied code(s) '%s' didn't exactly match a DisGeNet disease."%", ".join(missing))
            print('It may be that there are no genes associated with that code.')
            print('This can be verified using the look up link on the start page.')
            print('Otherwise, delete the code and try with any remaining codes.')
            return False
        else:
            with open(self.key_used_file, 'w') as f:
                f.write("\n".join(self.keys) + "\n")
        return True
    def extract_data(self):
        d = {}
        for k in self.keys:
            for p in self.dgn.data.get(k,{}):
                if p not in d:
                    d[p] = 0.
                ### take the max score seen across all keys
                ### we could take median, but max seems appropriate for sparse genetic data
                d[p] = max([d[p], self.dgn.data[k][p]])
        from dtk.files import get_file_records,FileDestination
        # write out any matches
        with FileDestination(self.score_fn,['uniprot']+['dgns']) as dest:
            for uniprot,score in d.items():
                dest.append([uniprot,score])
        self.rec = d
        self.found = len(d)
    def _plot(self):
        from dtk.plot import smart_hist
        pp2 = smart_hist(list(self.rec.values()),
                      layout=
                          {'title': 'Protein evidence histogram',
                           'yaxis':{'title':'Protein count',
                              },
                           'xaxis':{'title':'DisGeNet evidence score'},
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
            #enabled_default=uji.disease_name_set(ws,'DisGeNet')
            enabled_default=uji.data_status_ok(ws,'DisGeNet','DisGeNET Values')
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
                wf.eff.hold('dgn')
                cm_info = wf.std_wf.get_main_cm_info(uji.job_type)
                self.cm_info = cm_info
                cm_info.pre.add_pre_steps(wf)
                from dtk.workflow import DGNStep,LocalCode
                my_name='dgn'
                self.dgn_name = my_name
                DGNStep(wf,my_name,
                        disease=wf.ws.get_disease_default('DGN'),
                        )
                LocalCode(wf,'dgn_followup',
                        func=self._do_dgn_followup,
                        inputs={my_name:True},
                        )
            def _do_dgn_followup(self,my_step):
                wf = my_step._workflow
                from runner.process_info import JobInfo
                bji = JobInfo.get_bound(wf.ws, wf.step(self.dgn_name).job_id)
                cat = bji.get_data_catalog()
                # get index of first job we're about to create
                start = len(wf._order)
                # add post-processing for each code except 'overall'
                for code in cat.get_codes('uniprot','score'):
                    self.cm_info.post.add_post_steps(
                            wf,
                            self.dgn_name+':'+code+'_dgn',
                            code,
                            )
                # schedule a completion step when all jobs are done
                from dtk.workflow import LocalCode
                LocalCode(wf,'dgn_complete',
                        func=self._do_dgn_complete,
                        inputs={k:True for k in wf._order[start:]},
                        )
            def _do_dgn_complete(self,my_step):
                wf = my_step._workflow
                wf.eff.release('dgn')
        parts.append(MyWorkflowPart())

if __name__ == "__main__":
    MyJobInfo.execute(logger)
