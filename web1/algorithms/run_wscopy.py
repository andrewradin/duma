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
    from_ws = forms.CharField(label='Source ws id')
    from_score = forms.CharField(label='Source score')
    def as_html(self):
        from django.utils.html import format_html
        return format_html(
                '''<div class="well">
                {}<p><p>
                </div><p>{}
                '''
                ,   '''
                    The Source score field is in the form
                    <job_id>_<code> (for example 12345_direct).

                    Or, the source score can be a path to a
                    file in the qi_data directory.
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
        self.use_LTS=True
        # base class init
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "WS Copy",
                "Copy Score from Workspace",
                )
        # any base class overrides for unbound instances go here
        # TODO: self.publinks
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.fn_output = self.lts_abs_root+'wscopy.tsv'
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        make_directory(self.lts_abs_root)
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "retrieve input",
                "remap score",
                "check enrichment",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        if '/' in self.parms['from_score']:
            self.import_from_file()
            p_wr.put("retrieve input","complete")
            p_wr.put("remap score","n/a")
        else:
            self.retrieve_input()
            p_wr.put("retrieve input","complete")
            self.map_into_workspace()
            p_wr.put("remap score","complete")
        self.check_enrichment()
        self.finalize()
        p_wr.put("check enrichment","complete")
    def import_from_file(self):
        path=os.path.join(
                PathHelper.repos_root,
                'experiments/quantifying_improvement/data',
                self.parms['from_score'],
                )
        from dtk.ext_label import Relabeler
        rl = Relabeler(self.ws.id,['drugbank'])
        from dtk.files import get_file_records
        with open(self.fn_output,'w') as f:
            f.write('\t'.join(['wsa','copy'])+'\n')
            for wsa_id,score in rl.read(path):
                f.write('%d\t%f\n' % (wsa_id,score))
    def retrieve_input(self):
        # get source score
        from browse.models import Workspace
        from_ws = Workspace.objects.get(pk=self.parms['from_ws'])
        job_id,code = self.parms['from_score'].split('_')
        from runner.process_info import JobInfo
        bji = JobInfo.get_bound(from_ws,int(job_id))
        cat = bji.get_data_catalog()
        self.from_score = cat.get_ordering(code,True)
        from browse.models import WsAnnotation
        # get source wsa to agent map
        self.src2agent = {
                wsa.id:wsa.agent_id
                for wsa in WsAnnotation.objects.filter(ws=from_ws)
                }
        # get agent to local wsa map
        self.agent2targ = {
                wsa.agent_id:wsa.id
                for wsa in WsAnnotation.objects.filter(ws=self.ws)
                }
    def map_into_workspace(self):
        with open(self.fn_output,'w') as f:
            f.write('\t'.join(['wsa','copy'])+'\n')
            for key,score in self.from_score:
                try:
                    key = self.src2agent[key]
                    key = self.agent2targ[key]
                    f.write('%d\t%f\n' % (key,score))
                except KeyError:
                    pass
    def get_data_code_groups(self):
        import runner.data_catalog as dc
        return [
                dc.CodeGroup('wsa',self._std_fetcher('fn_output'),
                        dc.Code('copy',label='Copy',),
                        )
                ]

if __name__ == "__main__":
    MyJobInfo.execute(logger)
