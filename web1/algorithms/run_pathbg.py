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

from django import forms

import json
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo
import runner.data_catalog as dc


class ConfigForm:
    def __init__(*args, **kwargs):
        # This used to copy ConfigForm from gpbr, but that doesn't work now that it has been ported
        # to new format.
        # To fix later if we ever resurrect this CM.
        pass

class MyJobInfo(JobInfo):
    cm_names=['path']
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        form = ConfigForm(ws, sources, copy_job, self.cm_names)
        return form.as_html()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        form = ConfigForm(ws, sources, None,self.cm_names,post_data)
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
    def __init__(self,ws=None,job=None):
        # base class init
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "PathBg",
                "Pathsum Background Removal",
                )
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.fn_direction_scores = self.outdir+'direction_scores.tsv'
            self.fn_direct_scores = self.outdir+'direct_scores.tsv'
            self.fn_indirect_scores = self.outdir+'indirect_scores.tsv'
            from algorithms.run_gpbr import pbr_plot_file_names
            self.publinks += pbr_plot_file_names()
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
                dc.CodeGroup('wsa',self._dir_fetcher,
                        dc.Code('directbgnormed',label='Direct Bg Normed',),
#                        dc.Code('directbg',label='Direct Background',
#                                efficacy=False,
#                                ),
#                        dc.Code('directbgpval',label='Direct Bg p-value',
#                                efficacy=False,
#                                ),
                        ),
                dc.CodeGroup('wsa',self._indir_fetcher,
                        dc.Code('indirectbgnormed',label='Indirect Bg Normed',),
#                        dc.Code('indirectbg',label='Indirect Background',
#                                efficacy=False,
#                                ),
#                        dc.Code('indirectbgpval',label='Indirect Bg p-value',
#                                efficacy=False,
#                                ),
                        ),
                dc.CodeGroup('wsa',self._direction_fetcher,
                        dc.Code('directionbgnormed',
                                label='Direction Bg Normed',
                                ),
#                        dc.Code('directionbg',label='Direction Background',
#                                efficacy=False,
#                                ),
#                        dc.Code('directionbgpval',label='Direction Bg p-value',
#                                efficacy=False,
#                                ),
                        ),
                ]
    def _my_fetcher(self,fn,keyset):
        f = open(fn)
        import dtk.readtext as rt
        src = rt.parse_delim(f)
        return rt.dc_file_fetcher(keyset,src,
                key_mapper=int,
                data_idxs=[2],
                )
    def _dir_fetcher(self,keyset):
        return self._my_fetcher(self.fn_direct_scores,keyset)
    def _indir_fetcher(self,keyset):
        return self._my_fetcher(self.fn_indirect_scores,keyset)
    def _direction_fetcher(self,keyset):
        return self._my_fetcher(self.fn_direction_scores,keyset)
    def _parse_jobname(self,jobname):
        fields = jobname.split('_')
        return (int(fields[1]),int(fields[2]))
    def source_label(self,jobname):
        ts_id,ws_id = self._parse_jobname(jobname)
        from browse.models import TissueSet
        ts = TissueSet.objects.get(pk=ts_id)
        return ts.ts_label()+' PathBg'
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        from runner.models import Process
        self.pathjob = Process.objects.get(pk=self.parms['pathjob'])
        self.pathjob_settings = self.pathjob.settings()
        self.reps = self.parms['iterations']
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
        self.build_inputs()
        p_wr.put("build input files","complete")
        got = self.rm.wait_for_resources(self.job.id,[0,(1,self.reps)])
        self.remote_cores = got[1]
        p_wr.put("get remote resources",
                "got %d remote cores" % self.remote_cores,
                )
        self.execute()
        p_wr.put("execute pathsums","complete")
        self.post_process_all()
        self.check_enrichment()
        self.finalize()
        p_wr.put("cleanup","complete")
    def post_process_all(self):
        self.post_process('direct')
        if self.pathjob_settings['p2p_file']:
            self.post_process('indirect')
        self.post_process('direction')
    def build_inputs(self):
        worklist = []
        from algorithms.bulk_pathsum import PathsumWorkItem
        WorkItem=PathsumWorkItem
        for i in range(self.reps):
            wi = WorkItem()
            wi.serial = len(worklist)
            worklist.append(wi)
        WorkItem.pickle(self.indir,'worklist',worklist)
        path_settings = self.pathjob_settings
        if path_settings['combo_with']:
            d = self.ws.get_combo_therapy_data(path_settings['combo_with'])
            from dtk.prot_map import DpiMapping
            dpi = DpiMapping(path_settings['p2d_file'])
            from algorithms.bulk_pathsum import get_combo_fixed
            path_settings['combo_fixed'] = get_combo_fixed(d,dpi)
        from algorithms.run_path import get_tissue_ids_from_settings
        tissues = get_tissue_ids_from_settings(path_settings)
        from algorithms.run_path import get_tissue_settings_keys
        for tissue_id in tissues:
            ev_key, fc_key = get_tissue_settings_keys(tissue_id)
            WorkItem.build_tissue_file(
                    self.indir,
                    tissue_id,
                    path_settings[ev_key],
                    path_settings[fc_key],
                    )
        from run_gpbr import _make_context
        context = _make_context(tissues, path_settings)
        WorkItem.pickle(self.indir,'context',context)
        # generate dpi mapping file
        WorkItem.build_dpi_map(self.indir,
                    int(path_settings['ws_id']),
                    path_settings['p2d_file'],
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
    def post_process(self,stem):
        print('post_process',stem)
        # read in 'real' pathsum scores
        from browse.models import Workspace
        ws = Workspace.objects.get(pk=self.parms['ws_id'])
        path_bji = JobInfo.get_bound(ws,self.parms['pathjob'])
        cat = path_bji.get_data_catalog()
        raw_scores = dict(cat.get_ordering(stem,True))
        # read in all background scores
        from algorithms.bulk_pathsum import PathsumBackground
        from dtk.files import get_dir_matching_paths
        import re
        pb = PathsumBackground(
                get_dir_matching_paths(self.outdir, stem+r'[0-9]+score$'),
                )
        print(len(pb.label2scores()),'scored drugs')
        print(pb.runs,'background score sets recovered')
        # calculate background, normed, and pval scores
        import dtk.num
        bg_scores = {}
        normed_scores = {}
        pval_scores = {}
        for label,score in six.iteritems(raw_scores):
            padded = pb.padded_score(label)
            avg = dtk.num.avg(padded)
            bg_scores[label] = avg
            normed_scores[label] = (score-avg)
            lt_score = len([x for x in padded if x < score])
            pval_scores[label] = float(lt_score)/len(padded)
        print('raw avg %f; sd %f' % dtk.num.avg_sd(list(raw_scores.values())))
        print('bg avg %f; sd %f' % dtk.num.avg_sd(list(bg_scores.values())))
        print('normed avg %f; sd %f' % dtk.num.avg_sd(list(normed_scores.values())))
        # generate distribution plot
        from dtk.plot import plot_vectors,PlotlyPlot,scatter2d
        fn = os.path.join(self.tmp_pubdir,stem+'_dist')
        pp = PlotlyPlot([
                dict(name='raw',
                        y=sorted(list(raw_scores.values()),reverse=True),
                        ),
                dict(name='bg',
                        y=sorted(list(bg_scores.values()),reverse=True),
                        ),
                dict(name='normed',
                        y=sorted(list(normed_scores.values()),reverse=True),
                        ),
                dict(name='pval',
                        y=sorted(list(pval_scores.values()),reverse=True),
                        ),
                ],
                dict(
                    title='%s score distributions'%(stem,),
                ))
        pp.save(fn+'.plotly',thumbnail=True)
        # generate synced score plots
        kts = ws.get_wsa_id_set(ws.eval_drugset)
        tmp = [
                (
                    raw_scores[label],
                    bg_scores[label],
                    normed_scores[label],
                    label in kts,
                )
                for label in raw_scores
                ]
        fn = os.path.join(self.tmp_pubdir,stem+'_raw_bg_scat')
        pp = scatter2d('raw','bg',
                [(x[0],x[1]) for x in tmp],
                class_idx=[1 if x[3] else 0 for x in tmp],
                classes=[
                        ('Unknown',{'color':'blue'}),
                        ('KT',{'color':'red'}),
                        ],
                title='%s raw vs bg'%(stem,),
                )
        pp.save(fn+'.plotly',thumbnail=True)
        fn = os.path.join(self.tmp_pubdir,stem+'_raw_normed_scat')
        pp = scatter2d('raw','normed',
                [(x[0],x[2]) for x in tmp],
                class_idx=[1 if x[3] else 0 for x in tmp],
                classes=[
                        ('Unknown',{'color':'blue'}),
                        ('KT',{'color':'red'}),
                        ],
                title='%s raw vs normed'%(stem,),
                refline=False,
                )
        pp.save(fn+'.plotly',thumbnail=True)
        traces = [
                ('raw',0),
                ('bg',1),
                ('normed',2),
                ]
        for sort,sort_idx in traces:
            tmp.sort(key=lambda x:x[sort_idx],reverse=True)
            fn = os.path.join(self.tmp_pubdir,stem+'_'+sort+'_dust')
            pp = PlotlyPlot(
                    [
                        dict(
                                name=label,
                                y=[y[idx] for y in tmp],
                                mode='markers',
                                marker={
                                        'size':2,
                                        'maxdisplayed':1000,
                                        },
                                )
                        for label,idx in traces
                    ],
                    {
                        'title':'%s %s-sorted dustplot'%(stem,sort),
                    },
                    )
            pp.save(fn+'.plotly',thumbnail=True)
        # write out bg_avg, normed and pval score file for catalog
        fn = getattr(self,'fn_'+stem+'_scores')
        with open(fn,'w') as f:
            for label in raw_scores:
                f.write('\t'.join([str(x) for x in (
                        label,
                        bg_scores[label],
                        normed_scores[label],
                        pval_scores[label],
                        )])+'\n')

if __name__ == "__main__":
    MyJobInfo.execute(logger)
