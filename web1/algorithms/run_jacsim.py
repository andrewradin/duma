#!/usr/bin/env python3

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

from django import forms

from tools import ProgressWriter
from runner.process_info import JobInfo
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager
import runner.data_catalog as dc
from dtk.prot_map import DpiMapping,PpiMapping

import json
import logging
logger = logging.getLogger("algorithms.run_jacsim")

class ConfigForm(forms.Form):
    drugset = forms.ChoiceField(
                label = 'DrugSet for comparison',
                initial='',
                choices=(('','None'),)
               )
    dpi_file = forms.ChoiceField(label='DPI dataset')
    dpi_t = forms.FloatField(label='Min DPI evidence')
    ppi_file = forms.ChoiceField(label='PPI dataset')
    ppi_t = forms.FloatField(label='Min PPI evidence')
    _subtype_name = "job_subtype"
    def __init__(self, ws, sources, copy_job, *args, **kwargs):
        super(ConfigForm,self).__init__(*args, **kwargs)
        self.ws = ws
        # reload choices on each form load -- first DPI...
        f = self.fields['dpi_file']
        f.choices = DpiMapping.choices(ws)
        f.initial = ws.get_dpi_default()
        self.fields['dpi_t'].initial = ws.get_dpi_thresh_default()
        # ... then PPI
        f = self.fields['ppi_file']
        f.choices = PpiMapping.choices()
        f.initial = ws.get_ppi_default()
        self.fields['ppi_t'].initial = ws.get_ppi_thresh_default()
        # ...then drugset
        f = self.fields['drugset']
        f.choices = self.ws.get_wsa_id_set_choices()
        f.initial = f.choices[0][0]
        if copy_job:
            self.from_json(copy_job.settings_json)
    def as_html(self):
        from django.utils.html import format_html
        return format_html('''
                <input name="{}" type="hidden" value="{}"/>
                <table>{}</table>
                '''
                ,self._subtype_name
                ,None
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
        p ={}
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

class MyJobInfo(JobInfo):
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        form = ConfigForm(ws, sources, copy_job)
        return form.as_html()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        form = ConfigForm(ws, sources, None,post_data)
        if not form.is_valid():
            return (form.as_html(),None)
        settings = form.as_dict()
        settings['ws_id'] = ws.id
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(settings)
                            )
        next_url = ws.reverse('nav_progress',job_id)
        return (None, next_url)
    def get_data_code_groups(self):
        return [
                dc.CodeGroup('wsa',self._std_fetcher('outfile'),
                        dc.Code('DirectJacSim',
                                    label='DirJac',
                                    fmt='%0.4f',
                                    ),
                        dc.Code('IndirectJacSim',
                                    label='IndirJac',
                                    fmt='%0.4f',
                                    ),
                        ),
                ]
    def __init__(self,ws=None,job=None):
        self.use_LTS=True
        # base class init
        super(MyJobInfo,self).__init__(
                    ws,
                    job,
                    __file__,
                    "JacSim",
                    "JacSim",
                    )
        # any base class overrides for unbound instances go here
        self.publinks = (
                )
        self.needs_sources = False
        # job-specific properties
        if self.job:
            self.base_drug_dgi = False
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.ec = ExitCoder()
            # output files
            self.outfile = os.path.join(self.lts_abs_root, 'jacsim.tsv')
            # published output files
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        make_directory(self.lts_abs_root)
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "setup",
                "calculating overlap",
                "wrap up",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.setup()
        p_wr.put("setup","complete")
        self.run_jacsim()
        p_wr.put("calculating overlap", 'complete')
        self.write_results()
        self.finalize()
        p_wr.put("wrap up","complete")
    def write_results(self):
        with open(self.outfile, 'w') as f:
            f.write("\t".join(['wsa', 'DirectJacSim', 'IndirectJacSim']) + "\n")
            for wsa,d in six.iteritems(self.ms.all_scores):
                if not d:
                    continue
                djac_score = 0.0
                ijac_score = 0.0
                for ref_set_drug,d in six.iteritems(self.ms.all_scores[wsa]):
                    if d:
                        djac_score += d['dirJac']
                        ijac_score += d['indJac']
                f.write("\t".join([str(x) for x in [wsa,
                                                    djac_score,
                                                    ijac_score
                                                   ]
                                   ]) + "\n")
    def run_jacsim(self):
        from scripts.metasim import metasim
        self.ms = metasim(drug_wss = self.basedrugs_ws,
                     dpi = self.parms['dpi_file'],
                     ws = self.ws,
                     dpi_t = self.parms['dpi_t'],
                     ppi = self.parms['ppi_file'],
                     ppi_t = self.parms['ppi_t']
                    )
        self.ms.prot_setup()
        self.ms.run_jacsim()
    def setup(self):
        from browse.models import WsAnnotation
        wsa_ids = self.ws.get_wsa_id_set(self.parms['drugset'])
        self.basedrugs_ws = list(WsAnnotation.objects.filter(ws_id = self.ws.id,
                                                        pk__in = wsa_ids)
                                 )

if __name__ == "__main__":
    MyJobInfo.execute(logger)
