#!/usr/bin/env python3

from __future__ import print_function
import sys
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

from django import forms

from tools import ProgressWriter
from runner.process_info import JobInfo
import runner.data_catalog as dc
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager

import json
import logging
logger = logging.getLogger("algorithms.run_gwasig")

from collections import defaultdict
class ConfigForm(forms.Form):
    _subtype_name = "job_subtype"
    def __init__(self, ws, copy_job, *args, **kwargs):
        super(ConfigForm,self).__init__(*args, **kwargs)
        self.ws = ws
        # reload choices on each form load -- PPI
        # build individual threshold fields for each gwas dataset
        self.input_count = 0
        from dtk.gwas import gwas_code
        for choice in self.ws.get_gwas_dataset_choices():
            ds_name = gwas_code(choice[0])
            self.input_count += 1
            self.fields[ds_name] = forms.BooleanField(
                                label = choice[1],
                                initial = True,
                                required = False,
                                )
        self.fields['min_por'] = forms.FloatField(
                                 label = 'Min. portion of datasets a protein '+
                                         "must be detected in.",
                                 initial = 0.0
                                 )
        if copy_job:
            self.from_json(copy_job.settings_json)
    def as_html(self):
        from django.utils.html import format_html
        return format_html(
                u'''<div class="well">There are currently {} GWAS datasets
                in this workspace.<p><p>
                You can add more datasets
                <a href="{}">
                here</a>.<p><p>
                </div><p>{}
                '''
                ,   str(self.input_count)
                ,   self.ws.reverse('gwas_search')
                , self.as_p()
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
        p ={'ws_id':self.ws.id}
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

def combine_gwds_data(dd,min_tis):
    dl = defaultdict(list)
    for id in dd.keys():
        for uni in dd[id].keys():
            if uni == '-':
                continue
            try:
                val = dd[id][uni]
            except KeyError:
                val = 0.0
            dl[uni].append(val)
    if len(list(dl.keys())) == 0:
        raise RuntimeError("no significant proteins")
    signature={}
    gen = (i for i in dl.keys() if len(dl[i]) >= min_tis)
    from statistics import mean
    import numpy as np
    for uni in gen:
### Currently we are taking the mean as that what worked best in GESig, but we should check here
### We could also take the (weighted) average, max, or any other method.
### This approach also implicitly defaults to 0 if a protein is not detected in a given tissue.
        signature[uni] = [mean(dl[uni]),
                          len(dl[uni])
                          ]
    print(("signature calculated;",len(signature),"proteins"))
    return signature

class MyJobInfo(JobInfo):
    def description_text(self):
        return '''
        Creates a protein signature from GWAS data (analogously to what
        GESig does with Gene Expression data).
        '''
    def settings_defaults(self,ws):
        cfg=ConfigForm(ws,None)
        return {
                'default':cfg.as_dict(),
                }
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        form = ConfigForm(ws, copy_job)
        return form.as_html()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        form = ConfigForm(ws, None,post_data)
        if not form.is_valid():
            return (form.as_html(),None)
        settings = form.as_dict()
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(settings)
                            )
        next_url = ws.reverse('nav_progress',job_id)
        return (None, next_url)
    def __init__(self,ws=None,job=None):
        # base class init
        self.use_LTS=True
        super(MyJobInfo,self).__init__(
                    ws,
                    job,
                    __file__,
                    "GWASig",
                    "GWAS Signature",
                    )
        # any base class overrides for unbound instances go here
        self.publinks = (
                (None,'scatter.plotly'),
                (None,'histo.plotly'),
                )
        self.qc_plot_files = (
                'scatter.plotly',
                'histo.plotly',
                )
        # job-specific properties
        self.gwasig_node = 'gwasig'
        if self.job:
            self.log_prefix = self.job.name+':'
            self.debug("setup")
            self.ec = ExitCoder()
            # output files
            self.signature_file = self.lts_abs_root+"signature.tsv"
            # published output files
            self.scatter = self.tmp_pubdir+"scatter.plotly"
            self.histo = self.tmp_pubdir+"histo.plotly"
    def is_a_source(self):
        return True
    def get_data_code_groups(self):
        import runner.data_catalog as dc
        return [
                dc.CodeGroup('uniprot',self._std_fetcher('signature_file'),
                        dc.Code('ev',label='Evidence'),
                        dc.Code('gwascnt',label='GWAS Count'),
                        )
                ]
    def run(self):
        from collections import defaultdict
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        make_directory(self.lts_abs_root)
        p_wr = ProgressWriter(self.progress
                , [ "wait for resources"
                  , "setup"
                  , "distilling to a single signature"
                  , "plotting"
                  ]
                )
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.get_gwas_data()
        p_wr.put("setup","complete")
        self.single_distill()
        self.save_signature()
        p_wr.put("distilling to a single signature","complete")
        self.plot()
        p_wr.put("plotting","complete")
        self.finalize()
    def get_gwas_data(self):
        from dtk.gwas import scored_gwas,selected_gwas
        self.gwas_data = {
                key:scored_gwas(key, v2d_threshold=1.0, v2g_threshold=0.0, max_prot_assocs=None)
                for key in selected_gwas(self.parms)
                }
        if not self.gwas_data:
            print('Unable to find any GWAS data. Quitting.')
            sys.exit(self.ec.encode('unableToFindDataError'))
    def single_distill(self):
        try:
            self.signature = combine_gwds_data(
                                self.gwas_data,
                                int(round(
                self.parms['min_por'] * len(list(self.gwas_data.keys())
                ))),
                                )
        except RuntimeError:
            sys.exit(self.ec.encode('unexpectedDataFormat'))
    def save_signature(self):
        with open(self.signature_file, 'w') as f:
            header = ['uniprot', 'ev', 'gwascnt']
            f.write("\t".join(header) + "\n")
            print("\t".join(header))
            for uni in self.signature.keys():
                line = "\t".join([str(i) for i in [uni] + self.signature[uni]])
                f.write(line + "\n")
    def plot(self):
        from dtk.plot import scatter2d, Color, rand_jitter, smart_hist
        from math import log
        jittered = rand_jitter([x[1] for x in self.signature.values()])
        xys = zip(jittered,[x[0]for x in self.signature.values()])
        from browse.models import Protein
        prot_qs = Protein.objects.filter(uniprot__in=list(self.signature.keys()))
        uni_2_gene = { x.uniprot:x.gene for x in prot_qs }
        names = []
        for id in self.signature:
            try:
                names.append(uni_2_gene[id])
            except KeyError:
                 names.append(id)
        pp = scatter2d(
                'Number of datasets',
                'Evidence score',
                xys,
                title = 'Uniprot evidence vs number of datasets',
                text = names,
                ids = ('protpage', list(self.signature.keys())),
                refline = False,
                classes=[
                        ('',{'color':Color.default, 'opacity':0.5})
                        ],
                class_idx = [0] * len(list(self.signature.values())),
                bins = True
               )
        pp.save(self.scatter)
        pp = smart_hist([x[0] for x in self.signature.values()],
                        layout=
                              {'title': 'Protein evidence histogram',
                               'yaxis':{'title':'Protein count',
                                  },
                               'xaxis':{'title':'GWASig evidence score'},
                              }
                        )
        pp.save(self.histo)
    def add_workflow_parts(self,ws,parts):
        jobnames = self.get_jobnames(ws)
        assert len(jobnames) == 1
        jobname=jobnames[0]
        uji = self # simplify access inside MyWorkflowPart
        class MyWorkflowPart:
            label=uji.source_label(jobname)
            enabled_default=uji.data_status_ok(ws, 'Gwas', 'GWAS Datasets')
            def add_to_workflow(self,wf):
                cm_info = wf.std_wf.get_main_cm_info(uji.job_type)
                cm_info.pre.add_pre_steps(wf)
                from dtk.workflow import GWASigStep
                gwasig_name='gwasig'
                GWASigStep(wf,gwasig_name,
                        thresh_overrides={}
                        )
                cm_info.post.add_post_steps(wf,gwasig_name,'ev')
        parts.append(MyWorkflowPart())

if __name__ == "__main__":
    MyJobInfo.execute(logger)

