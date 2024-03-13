#!/usr/bin/env python3

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

import logging
logger = logging.getLogger("algorithms.run_example")

import json
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo

class ConfigForm(forms.Form):
    score1 = forms.CharField(label='score 1')
    score2 = forms.CharField(label='score 2')
    rm_zeros = forms.BooleanField(label='Ignore mols w/0s in either score', required=False, initial=True)
    def as_html(self):
        from django.utils.html import format_html
        return format_html(
                '''<div class="well">
                {}
                </div><p>{}
                '''
                ,   '''
                    Score fields are in the form
                    <job_id>_<code> (for example 12345_direct).
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
                "Rank Delta",
                "Rank Delta",
                )
        # any base class overrides for unbound instances go here
        # TODO: self.publinks
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.fn_rankdel = self.outdir+'rankdel.tsv'
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "retrieve input scores",
                "calculate rank delta",
                "finalize",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.retrieve_input()
        p_wr.put("retrieve input scores","complete")
        self.calculate_rankdel()
        p_wr.put("calculate rank delta","complete")
        self.finalize()
        p_wr.put("finalize","complete")
    def retrieve_input(self):
        from runner.process_info import JobInfo
        score_parts = [self.parms[x].split('_') for x in ('score1','score2')]
        orderings = []
        self.to_ignore=set()
        for job_id,code in score_parts:
            bji = JobInfo.get_bound(self.ws,int(job_id))
            cat = bji.get_data_catalog()
            ord = cat.get_ordering(code,True)
            if self.parms['rm_zeros']:
                self.to_ignore.update(set([x[0] for x in ord if x[1] != 0.]))
            orderings.append(ord)
        from dtk.scores import Ranker
        self.base=Ranker(orderings[0])
        self.new=Ranker(orderings[1])
        if self.to_ignore:
            print('The following WSAs were ignored for having 0s in one of the scores:')
            print("\n".join([str(x) for x in self.to_ignore]))
    def calculate_rankdel(self):
        all_drugs = set(self.base.keys()) | set(self.new.keys())
        raw_score = []
        for key in all_drugs:
            if key in self.to_ignore:
                continue
            old_rank = self.base.get(key)
            new_rank = self.new.get(key)
            # the following seems backwards, but since a lower rank is
            # better, this gives the highest score to the most-improved
            delta = old_rank - new_rank
            raw_score.append( (key,delta) )
        # ZNorm this to put it more in range with other scores
        from dtk.scores import ZNorm
        zn = ZNorm(raw_score)
        with open(self.fn_rankdel,'w') as f:
            f.write('\t'.join(['drug_id','rankdel'])+'\n')
            for key in all_drugs:
                f.write('%d\t%f\n' % (key,zn.get(key)))
    def get_data_code_groups(self):
        import runner.data_catalog as dc
        return [
                dc.CodeGroup('wsa',self._my_fetcher,
                        dc.Code('rankdel',label='Rank Delta',),
                        )
                ]
    def _my_fetcher(self,keyset):
        f = open(self.fn_rankdel)
        import dtk.readtext as rt
        src = rt.parse_delim(f)
        header=next(src)
        return rt.dc_file_fetcher(keyset,src,
                key_mapper=int,
                key_idx=header.index('drug_id'),
                data_idxs=[
                        header.index('rankdel'),
                        ],
                )

if __name__ == "__main__":
    MyJobInfo.execute(logger)
