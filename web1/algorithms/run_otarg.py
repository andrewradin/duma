#!/usr/bin/env python3

from __future__ import print_function
import sys
import six
# retrieve the path where this program is located, and
# use it to construct the path of the website_root directory
try:
    from path_helper import PathHelper,make_directory
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")
    from path_helper import PathHelper,make_directory

import os
import django
if 'django.core' not in sys.modules:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    django.setup()

import logging
logger = logging.getLogger("algorithms.run_otarg")

import json
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo

from django import forms

class MyJobInfo(JobInfo):
    def description_text(self):
        return '''
        Extracts several categories of protein signatures for a disease
        from the OpenTargets database.
        '''
    def get_config_form_class(self,ws):
        vdefaults=ws.get_versioned_file_defaults()
        file_class = 'openTargets'
        class ConfigForm(forms.Form):
            disease = forms.CharField(max_length=256)
            otarg_version = forms.ChoiceField(
                    choices=ws.get_versioned_file_choices(file_class),
                    initial=vdefaults[file_class],
                    )
            def as_dict(self):
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
        return ConfigForm
    def settings_defaults(self,ws):
        cfg=self.get_config_form_class(ws)()
        return {
                'default':cfg.as_dict(),
                }
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        ConfigForm = self.get_config_form_class(ws)
        if copy_job:
            form = ConfigForm(initial=copy_job.settings())
        else:
            disease_default = ws.get_disease_default('OpenTargets')
            form = ConfigForm(initial={'disease':disease_default})
        from dtk.html import join,link,tag_wrap,ulist
        return join(
                tag_wrap('div',
                        join(
                            'For Disease, enter one of:',
                            ulist([
                                "an OBO key, with the prefix 'key:'",
                                "an exact match for an OBO disease name",
                                "one or more search words",
                                ]),
                            'To interactively search for an OBO key, click',
                            link(
                                'here',
                                ws.reverse('nav_ontobrowse','OpenTargets'),
                                ),
                            ),
                        attr={'class':'well'},
                        ),
                form.as_p(),
                )
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        ConfigForm = self.get_config_form_class(ws)
        form = ConfigForm(post_data)
        if not form.is_valid():
            return (form.as_p(),None)
        settings = form.as_dict()
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(settings)
                            )
        next_url = ws.reverse('nav_progress',job_id)
        return (None, next_url)
    def __init__(self,ws=None,job=None):
        self.use_LTS=True
        # base class init
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "OpenTargets",
                "OpenTargets Scores",
                )
        self.publinks = (
            (None, 'final_wts.plotly'),
        )
        self.qc_plot_files = (
                'final_wts.plotly',
                )
        # any base class overrides for unbound instances go here
        # job-specific properties
        if self.job:
            self.key_used_file = self.outdir +'key_used'
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.score_fn = self.lts_abs_root+'scores.tsv'
            self.final_wt_plot = self.tmp_pubdir+"final_wts.plotly"
            if os.path.isfile(self.key_used_file):
                self.otherlinks = []
                with open(self.key_used_file, 'r') as f:
                    for l in f:
                        k = l.strip()
                        self.otherlinks.append(('OpenTargets page',
                                            'https://www.targetvalidation.org/disease/%s/associations' %(k)
                                           ))
    def is_a_source(self):
        return True
    def get_data_code_groups(self):
        import runner.data_catalog as dc
        return [
                dc.CodeGroup('uniprot',self._std_fetcher('score_fn'),
                        dc.Code('overall',label='Combined Score', fmt='%.2e', efficacy=False),
                        dc.Code('literature',label='Literature', fmt='%.2e'),
                        dc.Code('rnaexpression',label='RNA Expr', fmt='%.2e'),
                        dc.Code('somaticmutation',label='Somatic Mut', fmt='%.2e'),
                        dc.Code('geneticassociation',label='Gene Assoc', fmt='%.2e'),
                        dc.Code('knowndrug',label='Known Drug', fmt='%.2e'),
                        dc.Code('animalmodel',label='Animal Model', fmt='%.2e'),
                        dc.Code('affectedpathway',label='Pathway', fmt='%.2e')
                        ),
                ]
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        make_directory(self.lts_abs_root)
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "fetch data",
                "get disease key",
                "extract data",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        from dtk.open_targets import OpenTargets
        self.ot = OpenTargets(choice=self.parms['otarg_version'])
        p_wr.put("fetch data","complete")
        if not self.get_key():
            p_wr.put("get disease key","NO EXACT MATCH; SEE LOG FOR DETAILS")
            sys.exit(1)
        p_wr.put("get disease key","complete (%s)"%self.keys)
        self.extract_data()
        p_wr.put("extract data","scored %d proteins"%self.found)
        self._plot()
        self.finalize()
    def get_key(self):
        names = self.parms['disease'].split(',')
        self.keys = []
        for name in names:
            key_prefix = 'key:'
            if name[:len(key_prefix)].lower() == key_prefix:
                key = name[len(key_prefix):]
                self.keys.append(key)
                if not self.ot.check_key(key):
                    print()
                    print("Key '%s' is not present in OpenTargets." %key)
                    print("Try using the interactive key seach link.")
                    print()
                    return False
            else:
                key = self.ot.get_disease_key(name)
                self.keys.append(key)
            if not key:
                # specified disease name was not an exact match; present possible
                # matches in the log and exit with an error
                alternatives = self.ot.search_disease(name)
                from dtk.text import print_table
                print()
                print("Possible Matches:")
                print_table(alternatives)
                print()
                print("Supplied name '%s' didn't exactly match a disease."%name)
                print('Choose an exact match from the table above and try again.')
                print('If the disease name contains special characters, your')
                print('browser should be set to use Unicode encoding.')
                print()
                return False
        if not self.keys:
            print("No keys specified")
            return False

        with open(self.key_used_file, 'w') as f:
            f.write('\n'.join(self.keys))
        return True
    def extract_data(self):
        # scan master file, looking for matches to disease
        from collections import defaultdict
        d = defaultdict(lambda: defaultdict(float))
        for key in self.keys:
            cur = self.ot.get_disease_scores(key)
            for uniprot, name_to_value in six.iteritems(cur):
                for name, value in six.iteritems(name_to_value):
                    d[uniprot][name] = max(d[uniprot][name], value)

        # find all columns with at least one non-zero score
        populated = set()
        for score in d.values():
            populated |= set(score.keys())
        # write out any matches
        score_names = [
                'overall',
                'literature',
                'rna_expression',
                'somatic_mutation',
                'genetic_association',
                'known_drug',
                'animal_model',
                'affected_pathway'
                ]
        from dtk.files import get_file_records,FileDestination
        with FileDestination(self.score_fn,['uniprot']+[
                x.replace('_','')
                for x in score_names
                if x in populated
                ]) as dest:
            for uniprot in d:
                scores = d[uniprot]
                rec = [uniprot]+[
                        scores.get(score,0)
                        for score in score_names
                        if score in populated
                        ]
                dest.append(rec)
        score_name_non_zeros = [0 for i in score_names]
        score_name_zeros_or_NAN = [0 for i in score_names]

        for k,v in six.iteritems(d):
            for i,score in enumerate(score_names):
                if score in list(v.keys()) and v[score] > 0:
                    score_name_non_zeros[i] += 1
                else:
                    score_name_zeros_or_NAN[i] += 1
        self.rec = d
        self.score_name_non_zeros = score_name_non_zeros
        self.score_name_zeros_or_NAN = score_name_zeros_or_NAN
        self.score_names = score_names
        self.found = len(d)

    def _plot(self):
        from dtk.plot import PlotlyPlot, fig_legend
        max_name_len = max([len(i) for i in self.score_names])
        last_name_len = len(self.score_names[-1])
        import numpy as np; sort_idx = np.argsort(self.score_name_zeros_or_NAN)
        sorted_non_zeros = []
        sorted_name_zeros_or_NAN = []
        sorted_names =[]
        for i in sort_idx:
            sorted_non_zeros.append(self.score_name_non_zeros[i])
            sorted_name_zeros_or_NAN.append(self.score_name_zeros_or_NAN[i])
            sorted_names.append(self.score_names[i])
        pp2 = PlotlyPlot([
                dict(
                       x = sorted_non_zeros,
                       y = sorted_names,
                       orientation='h',
                       name='non-Zero and non-NaN',
                       type='bar'
                       ),
                dict(
                   x = sorted_name_zeros_or_NAN,
                   y = sorted_names,
                   orientation='h',
                   name='Zero or NaN',
                   type='bar'
               )
        ],
           dict(
               title='Matches by category',
               barmode='stack',
               yaxis=dict(tickangle=-30),
               xaxis=dict(title='Proteins'),
               height=(len(self.score_names)+8)*30,
               width=400 + max_name_len*6,
               margin=dict(
                      l=max_name_len*6,
                      r=30,
                      b=last_name_len*5+60,
                      t=30,
                      pad=4
                      )
           )
        )
        pp2._layout['annotations'] = [fig_legend([
                                      'This plot quantifies the number of proteins with a score'
                                      ,'in each OpenTargets score type. A minimum of ~20 proteins'
                                      ,'should be present for a score to be used downstream.'
                                      ], -0.175)]
        pp2.save(self.final_wt_plot)
    multi_uniprot_scores = True
    def add_workflow_parts(self,ws,parts):
        jobnames = self.get_jobnames(ws)
        assert len(jobnames) == 1
        jobname=jobnames[0]
        uji = self # simplify access inside MyWorkflowPart
        class MyWorkflowPart:
            label=uji.source_label(jobname)
            enabled_default=uji.data_status_ok(
                    ws,
                    'OpenTargets',
                    'Integrated Target Data',
                    )
            # WARNING TO ANYONE DOING COPY/PASTE to create a new CM:
            # This version of add_to_workflow() is complicated due to
            # the fact that otarg sources multiple protein scores,
            # and which outputs are available is unknown until after
            # the job is run. If applies to you, copy away, and be
            # sure that multi_uniprot_scores is set to True.
            # If not, you may want to work from the much simpler
            # version of add_to_workflow() in, e.g. gwasig.
            def add_to_workflow(self,wf):
                assert uji.multi_uniprot_scores
                wf.eff.hold('otarg')
                cm_info = wf.std_wf.get_main_cm_info(uji.job_type)
                self.cm_info = cm_info
                cm_info.pre.add_pre_steps(wf)
                from dtk.workflow import OpenTargStep,LocalCode
                my_name='otarg'
                self.otarg_name = my_name
                OpenTargStep(wf,my_name,
                        disease=wf.ws.get_disease_default('OpenTargets'),
                        )
                LocalCode(wf,'otarg_followup',
                        func=self._do_otarg_followup,
                        inputs={my_name:True},
                        )
            def _do_otarg_followup(self,my_step):
                wf = my_step._workflow
                from runner.process_info import JobInfo
                bji = JobInfo.get_bound(wf.ws, wf.step(self.otarg_name).job_id)
                cat = bji.get_data_catalog()

                nonzero_prot_counts = count_nonzero_prots(bji)

                # get index of first job we're about to create
                start = len(wf._order)
                # add post-processing for each code except 'overall'
                skip = set(['overall'])
                if not bji.parms.get('incl_knowndrug',False):
                    skip.add('knowndrug')
                for code in cat.get_codes('uniprot','score'):
                    if code in skip:
                        continue
                    if nonzero_prot_counts[code] < bji.parms.get('min_prots', 20):
                        logger.info(f"Skipping {code} with only {nonzero_prot_counts[code]} prots")
                        continue
                    self.cm_info.post.add_post_steps(
                            wf,
                            self.otarg_name+':'+code+'_otarg',
                            code,
                            )
                # schedule a completion step when all jobs are done
                from dtk.workflow import LocalCode
                LocalCode(wf,'otarg_complete',
                        func=self._do_otarg_complete,
                        inputs={k:True for k in wf._order[start:]},
                        )
            def _do_otarg_complete(self,my_step):
                wf = my_step._workflow
                wf.eff.release('otarg')
        parts.append(MyWorkflowPart())

def count_nonzero_prots(bji):
    cat = bji.get_data_catalog()
    codes = list(cat.get_codes('uniprot','score'))
    from collections import defaultdict
    out = defaultdict(int)
    for prot, scores in cat.get_feature_vectors(*codes)[1]:
        assert len(codes) == len(scores)
        for code, score in zip(codes, scores):
            if score > 0:
                out[code] += 1
    return out




if __name__ == "__main__":
    MyJobInfo.execute(logger)
