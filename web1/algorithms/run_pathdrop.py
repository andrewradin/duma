#!/usr/bin/env python3

from __future__ import print_function
from builtins import range
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
logger = logging.getLogger("algorithms.run_example")

import json
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo
import runner.data_catalog as dc
from django import forms

def config_form_class(job_type):
    parts = job_type.partition('drop')
    matching_path = parts[0]+parts[2]
    from runner.models import Process
    qs = Process.objects.filter(
                    name=matching_path,
                    exit_code=0,
                    ).order_by('-id')
    from dtk.text import fmt_time
    choices = []
    for p in qs:
        s = p.settings()
        try:
            choices.append( (p.id,'%d %s %s %s' %(
                    p.id,
                    fmt_time(p.completed),
                    s['p2d_file'],
                    s['p2p_file'],
                    )))
        except KeyError:
            pass
    # XXX copy job handling (see run_pathbg)
    class MyForm(forms.Form):
        pathjob = forms.ChoiceField(
                label='Settings from',
                choices=choices,
                )
        drop_pct = forms.IntegerField(
                initial=60,
                )
        iterations = forms.IntegerField(
                initial=20,
                )
    return MyForm

class MyJobInfo(JobInfo):
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        MyForm = config_form_class(job_type)
        form = MyForm()
        return form.as_p()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        MyForm = config_form_class(jobname)
        form = MyForm(post_data)
        if not form.is_valid():
            return form.as_p()
        p = dict(form.cleaned_data)
        from runner.models import Process
        pathjob = Process.objects.get(pk=p['pathjob'])
        ts_id,ws_id = self._parse_jobname(pathjob.name)
        p.update({
            'pathjob':pathjob.id,
            'ws_id':ws_id,
            'ts_id':ts_id,
            })
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(p)
                            )
        next_url = ws.reverse('nav_progress',job_id)
        return (None,next_url)
    def __init__(self,ws=None,job=None):
        # base class init
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "PathDrop",
                "Pathsum PPI Drop",
                )
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.fn_indirect_scores = self.outdir+'indirect_scores.tsv'
    def get_jobnames(self,ws):
        '''Return a list of all jobnames for this plugin in a workspace.'''
        return [
                self.get_jobname_for_tissue_set(ts)
                for ts in ws.get_tissue_sets()
                ]
    def get_jobname_for_tissue_set(self,ts):
        return self._format_jobname_for_tissue_set(self.job_type,ts)
    def get_data_code_groups(self):
        return [
                dc.CodeGroup('wsa',self._indir_fetcher,
                        dc.Code('indirdrop',label='Indirect Thinned',),
                        ),
                ]
    def _indir_fetcher(self,keyset):
        f = open(self.fn_indirect_scores)
        import dtk.readtext as rt
        src = rt.parse_delim(f)
        return rt.dc_file_fetcher(keyset,src,
                key_mapper=int,
                data_idxs=[1],
                )
    def _parse_jobname(self,jobname):
        fields = jobname.split('_')
        return (int(fields[1]),int(fields[2]))
    def source_label(self,jobname):
        ts_id,ws_id = self._parse_jobname(jobname)
        from browse.models import TissueSet
        ts = TissueSet.objects.get(pk=ts_id)
        return ts.ts_label()+' PathDrop'
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        from runner.models import Process
        self.pathjob = Process.objects.get(pk=self.job.settings()['pathjob'])
        self.pathjob_settings = self.pathjob.settings()
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "build input files",
                "get remote resources",
                "execute pathsums",
                "cleanup",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.reps = self.parms['iterations']
        self.build_inputs()
        p_wr.put("build input files","complete")
        got = self.rm.wait_for_resources(self.job.id,[0,(1,self.reps)])
        self.remote_cores = got[1]
        p_wr.put("get remote resources",
                "got %d remote cores" % self.remote_cores,
                )
        self.execute()
        p_wr.put("execute pathsums","complete")
        self.post_process()
        self.check_enrichment()
        self.finalize()
        p_wr.put("cleanup","complete")
    def build_inputs(self):
        from runner.models import Process
        settings = Process.objects.get(pk=self.parms['pathjob']).settings()
        worklist = []
        from algorithms.bulk_pathsum import PathsumWorkItem
        WorkItem=PathsumWorkItem
        for i in range(self.reps):
            wi = WorkItem()
            wi.serial = len(worklist)
            worklist.append(wi)
            wi.drop_ppi_pct = self.parms['drop_pct']
        WorkItem.pickle(self.indir,'worklist',worklist)
        from algorithms.run_path import get_tissue_ids_from_settings
        tissues = get_tissue_ids_from_settings(settings)
        from algorithms.run_path import get_tissue_settings_keys
        for tissue_id in tissues:
            ev_key, fc_key = get_tissue_settings_keys(tissue_id)
            WorkItem.build_tissue_file(
                    self.indir,
                    tissue_id,
                    settings[ev_key],
                    settings[fc_key],
                    )
        context = dict(settings,
                tissues=tissues,
                )
        WorkItem.pickle(self.indir,'context',context)
        # generate dpi mapping file
        WorkItem.build_dpi_map(self.indir,
                    settings['ws_id'],
                    settings['p2d_file'],
                    )
    def execute(self):
        import datetime
        start = datetime.datetime.now()
        print(str(start),self.reps,'parallel background runs started')
        pgm = PathHelper.website_root+'scripts/bulk_pathsum.py'
        local = False
        if local:
            import subprocess
            subprocess.check_call([pgm,
                        #'--cores', str(self.remote_cores),
                        self.indir,
                        self.outdir,
                        ])
        else:
            self.copy_input_to_remote()
            self.mch.check_remote_cmd(
                    'mkdir -p '+self.mch.get_remote_path(self.tmp_pubdir)
                    )
            self.mch.check_remote_cmd(
                    self.mch.get_remote_path(pgm)
                    +' --cores %d'%self.remote_cores
                    +' '+' '.join([
                            self.mch.get_remote_path(x)
                            for x in (self.indir,self.outdir)
                            ])
                    )
            self.copy_output_from_remote()
        end = datetime.datetime.now()
        print(str(end),'complete; elapsed',str(end-start))
    def post_process(self):
        from dtk.files import get_dir_file_names,get_file_records
        paths = [
            os.path.join(self.outdir,fn)
            for fn in get_dir_file_names(self.outdir)
            if fn.startswith('indirect')
            ]
        accum={}
        for path in paths:
            for key,score in [
                    (int(key),float(score))
                    for key,score in get_file_records(path,parse_type='tsv')
                    ]:
                accum.setdefault(key,[]).append(score)
        from dtk.num import median
        with open(self.fn_indirect_scores,'w') as f:
            for k,v in six.iteritems(accum):
                f.write('%d\t%f\n'%(k,median(v)))

if __name__ == "__main__":
    MyJobInfo.execute(logger)
