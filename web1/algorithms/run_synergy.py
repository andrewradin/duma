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

import logging
logger = logging.getLogger("algorithms.run_example")

import json
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo

################################################################################
# Step-by-step plug in guide
# 
# Also see the comments in the JobInfo base class definition in
# runner/process_info.py.
################################################################################
# 8) once the framework is working, invoke your actual code from the run()
#     method
# 9) once your job runs, use self.publinks to expose links to any static output
#     files
# 10) call check_enrichment() and finalize() from inside run() to run DEA, and
#     to copy any published files into their final position
# 11) implement get_data_code_groups() to return score and feature vector
#     results

class ConfigForm(forms.Form):
    mono = forms.CharField(label='Monotherapy score')
    combo = forms.CharField(label='Combo therapy score')
    base = forms.CharField(label='Base drug id')
    def as_html(self):
        from django.utils.html import format_html
        return format_html(
                '''<div class="well">
                Input is rather crude at the moment.<p><p>
                {}<p><p>
                </div><p>{}
                '''
                ,   '''
                    Mono- and Combo score fields are in the form
                    <job_id>_<code> (for example 12345_direct).
                    Base drug id is a wsa id (the number at the end
                    of the URL when looking at the drug page).
                    '''
                , self.as_p()
                )

class MyJobInfo(JobInfo):
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        if copy_job:
            form = ConfigForm(copy_job.settings())
        else:
            form = ConfigForm()
        return form.as_html()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        form = ConfigForm(post_data)
        if not form.is_valid():
            return (form.as_html(),None)
        settings = dict(form.cleaned_data)
        settings['ws_id'] = ws.id
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(settings)
                            )
        next_url = ws.reverse('nav_progress',job_id)
        return (None, next_url)
    def __init__(self,ws=None,job=None):
        # base class init
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "Synergy",
                "Partner/Base Synergy",
                )
        # any base class overrides for unbound instances go here
        # TODO: self.publinks
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.fn_synergy = self.outdir+'synergy.tsv'
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "retrieve input scores",
                "calculate synergy",
                "check enrichment",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.retrieve_input()
        p_wr.put("retrieve input scores","complete")
        self.calculate_synergy()
        p_wr.put("calculate synergy","complete")
        self.check_enrichment()
        self.finalize()
        p_wr.put("check enrichment","complete")
    def retrieve_input(self):
        from runner.process_info import JobInfo
        score_parts = [self.parms[x].split('_') for x in ('mono','combo')]
        orderings = []
        for job_id,code in score_parts:
            bji = JobInfo.get_bound(self.ws,int(job_id))
            cat = bji.get_data_catalog()
            orderings.append(cat.get_ordering(code,True))
        self.mono = dict(orderings[0])
        self.combo = dict(orderings[1])
        self.base_score = self.mono[int(self.parms['base'])]
    def calculate_synergy(self):
        with open(self.fn_synergy,'w') as f:
            f.write('\t'.join(['drug_id','synergy'])+'\n')
            for key,c_score in six.iteritems(self.combo):
                thresh = min(self.mono.get(key,0),self.base_score)
                if c_score > thresh:
                    f.write('%d\t%f\n' % (key,c_score-thresh))
    def get_data_code_groups(self):
        import runner.data_catalog as dc
        return [
                dc.CodeGroup('wsa',self._my_fetcher,
                        dc.Code('synergy',label='Synergy',),
                        )
                ]
    def _my_fetcher(self,keyset):
        f = open(self.fn_synergy)
        import dtk.readtext as rt
        src = rt.parse_delim(f)
        header=next(src)
        return rt.dc_file_fetcher(keyset,src,
                key_mapper=int,
                key_idx=header.index('drug_id'),
                data_idxs=[
                        header.index('synergy'),
                        ],
                )

if __name__ == "__main__":
    MyJobInfo.execute(logger)
