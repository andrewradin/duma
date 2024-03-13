#!/usr/bin/env python3

import sys
import six
from path_helper import PathHelper,make_directory

import os
import django
import django_setup

import logging
logger = logging.getLogger("algorithms.run_gpbr")

from django import forms

import json
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo, StdJobInfo, StdJobForm
import runner.data_catalog as dc
from algorithms.bulk_pathsum import BackgroundPathsumWorkItem

class GpbrInput:
    cm_names=['path', 'gpath', 'capp', 'mips']
    @classmethod
    def source_factory(cls,ws,job):
        bji = JobInfo.get_bound(ws,job)
        if bji.job_type == 'mips':
            return MipsGpbrInput(bji)
        if bji.job_type == 'capp':
            return CappGpbrInput(bji)
        if bji.job_type == 'gpath':
            return GpathGpbrInput(bji)
        if bji.job_type == 'path':
            return PathGpbrInput(bji)
        raise NotImplementedError("unsupported job type '%s'",bji.job_type)
    def __init__(self,bji):
        self.path_bji = bji
        self.tissue_list = []
    def settings(self):
        return dict(self.path_bji.parms)
    def make_context(self, path_settings, inner_cycles):
        from dtk.data import merge_dicts,dict_subset
        return merge_dicts(
                    {
                            'randomize':True,
                            'tissues':self.tissue_list,
                            'inner_cycles':inner_cycles,
                            },
                    dict_subset(path_settings,[
                            't2p_w',
                            'p2p_file',
                            'p2p_w',
                            'p2p_t',
                            'p2d_file',
                            'p2d_w',
                            'p2d_t',
                            'ws_id',
                            'combo_with',
                            'combo_type',
                            'combo_fixed',
                            'uniprot_flavor',
                            'uniprot_role',
                            ]),
                    )

class CappGpbrInput(GpbrInput):
    def score_names(self):
        result = [
                ['direct','capds'],
                ['indirect','capis'],
                ]
        if not self.path_bji.parms['p2p_file']:
            del(result[1])
        return result
    def build_tissue_files(self,indir):
        d, _, _ = self.path_bji.make_cmap_data()
        if not d:
            print('Unable to find any FAERS Co-morbidity data. Quitting.')
            from algorithms.exit_codes import ExitCoder
            ec = ExitCoder()
            sys.exit(ec.encode('unableToFindDataError'))
        for k in d:
            # XXX could skip output if d[k] is empty
            BackgroundPathsumWorkItem.build_nontissue_file(indir,k,d[k])
            self.tissue_list.append(k)

class MipsGpbrInput(GpbrInput):
    def score_names(self):
        result = [
                ['direct','mipsd'],
                ['indirect','mipsi'],
                ]
        if not self.path_bji.parms['p2p_file']:
            del(result[1])
        return result
    def build_tissue_files(self,indir):
        d, _ = self.path_bji.score_mi_data()

        if not d:
            print('Unable to find any Monarch phenotype data. Quitting.')
            from algorithms.exit_codes import ExitCoder
            ec = ExitCoder()
            sys.exit(ec.encode('unableToFindDataError'))

        for k in d:
            # XXX could skip output if d[k] is empty
            BackgroundPathsumWorkItem.build_nontissue_file(indir,k,d[k])
            self.tissue_list.append(k)

class GpathGpbrInput(GpbrInput):
    def score_names(self):
        result = [
                ['direct','gds'],
                ['indirect','gis'],
                ]
        if not self.path_bji.parms['p2p_file']:
            del(result[1])
        return result
    def build_tissue_files(self,indir):
        from algorithms.run_gpath import get_gwas_data
        d = get_gwas_data(self.path_bji.ws.id, self.path_bji.parms)
        for k in d:
            BackgroundPathsumWorkItem.build_nontissue_file(indir,k,d[k])
            self.tissue_list.append(k)

class PathGpbrInput(GpbrInput):
    def score_names(self):
        result = [
                ['direct','direct'],
                ['indirect','indirect'],
                ['direction','direction'],
                ]
        if not self.path_bji.parms['p2p_file']:
            del(result[1])
        return result
    def build_tissue_files(self,indir):
        path_settings = self.path_bji.parms
        from algorithms.run_path import get_tissue_ids_from_settings
        self.tissue_list = get_tissue_ids_from_settings(path_settings)
        from algorithms.run_path import get_tissue_settings_keys
        for tissue_id in self.tissue_list:
            ev_key, fc_key = get_tissue_settings_keys(tissue_id)
            BackgroundPathsumWorkItem.build_tissue_file(
                    indir,
                    tissue_id,
                    path_settings[ev_key],
                    path_settings[fc_key],
                    )

# XXX This could potentially be folded into Workspace.get_prev_job_choices,
# XXX if the latter handled multiple plugin names, and allowed for some
# XXX custom formatting in the dropdown. Its approach is more efficient,
# XXX as it does the workspace filtering on the database side.
def get_path_options(cm_names, ws_id):
    from django.db.models import Q
    my_filter_qs = Q()
    for s in cm_names:
        my_filter_qs = my_filter_qs | Q(name__startswith=s+'_')
    my_filter_qs = my_filter_qs & Q(exit_code=0)
    from runner.models import Process
    qs = Process.objects.filter(
                    my_filter_qs
                    ).order_by('-id')
    from dtk.text import fmt_time
    choices = []
    for p in qs:
        s = p.settings()
        try:
            if s["ws_id"] != ws_id:
                continue
            choices.append( (p.id,'%s %d %s %s %s' %(
                    p.job_type(),
                    p.id,
                    fmt_time(p.completed),
                    s['p2d_file'],
                    s['p2p_file'],
                    )))
        except KeyError:
            pass
    if choices:
        selected = choices[0][0]
    else:
        choices = [(('NA',0),'No pathsum-type jobs available')]
        selected = ' '
    return choices,selected
def pbr_plot_file_names(scores = ('direct','indirect','direction')):
    to_ret = []
    for score in scores:
        for label,fn_stem in (
                ('score distribution','_dist.plotly'),
                ('raw vs bg scatterplot','_raw_bg_scat.plotly'),
                ('raw vs normed scatterplot','_raw_normed_scat.plotly'),
                ('synced to raw','_raw_dust.plotly'),
                ('synced to bg','_bg_dust.plotly'),
                ('synced to normed','_normed_dust.plotly'),
               ):
            fn = score+fn_stem
            link = score.capitalize()+' '+label
            to_ret.append( (link,fn) )
    return to_ret

class MyJobInfo(StdJobInfo):
    descr = '''
        <b>gPBR</b> post-processes scores from CMs based on the pathsum
        algorithm to attempt to remove bias in the DPI datasets. It does
        this by calculating a 'background' score for each drug against
        a number of randomly-generated GE datasets, and subtracts that
        background from the original score to create a normalized score.
        '''
    short_label = "gPBR"
    page_label = "Generalized Pathsum Background Removal"
    upstream_jid = lambda cls, settings: (settings['pathjob'], None)
    
    def make_job_form(self, ws, data):
        choices, initial = get_path_options(GpbrInput.cm_names, ws.id)
        class MyForm(StdJobForm):
            pathjob = forms.ChoiceField(label='Path Job ID to use',
                                initial=initial,
                                choices=choices,
                                )
            iterations = forms.IntegerField(label='Background iterations',initial=5000)
            seed = forms.IntegerField(label='Random seed', initial=0, help_text='-1 for random seed')
        
        return MyForm(ws, data)

    def __init__(self,ws=None,job=None):
        super().__init__(ws, job, __file__)
        self.qc_plot_files = (
                'direct_raw_bg_scat.plotly',
                'indirect_raw_bg_scat.plotly',
                'direction_raw_bg_scat.plotly',
                )
        if self.job:
            self.fn_direction_scores = self.lts_abs_root+'direction_scores.tsv'
            self.fn_direct_scores = self.lts_abs_root+'direct_scores.tsv'
            self.fn_indirect_scores = self.lts_abs_root+'indirect_scores.tsv'
            self.publinks += pbr_plot_file_names()
    def dpi_codegroup_type(self):
        if not self.job:
            codetype = 'wsa'
        else:
            data_source = GpbrInput.source_factory(
                            self.ws,
                            self.parms['pathjob'],
                            )
            path_settings = data_source.settings()
            from dtk.prot_map import DpiMapping
            dpi_name = path_settings['p2d_file']
            dpi = DpiMapping(dpi_name)
            codetype = dpi.mapping_type()
        return codetype

    def get_data_code_groups(self):
        codetype = self.dpi_codegroup_type()
        return [
                dc.CodeGroup(codetype,self._std_fetcher('fn_direct_scores'),
                        dc.Code('directbgnormed',label='Direct Bg Normed',),
                        dc.Code('directbg',label='Direct Bg',
                                efficacy=False),
                        ),
                dc.CodeGroup(codetype,self._std_fetcher('fn_indirect_scores'),
                        dc.Code('indirectbgnormed',label='Indirect Bg Normed',),
                        dc.Code('indirectbg',label='Indirect Bg',
                                efficacy=False),
                        ),
                dc.CodeGroup(codetype,self._std_fetcher('fn_direction_scores'),
                        dc.Code('directionbgnormed',
                                label='Direction Bg Normed',
                                # Background on direction ends up as 0, so
                                # would just be double counting.
                                efficacy=False,
                                ),
                        dc.Code('directionbg',label='Direction Bg',
                                efficacy=False),
                        ),
                ]
    def pathsum_scale(self):
        # use the same scaling as the source job (which must be a pathsum-type
        # job, and so must define a pathsum_scale() method)
        src_bji = JobInfo.get_bound(self.ws,self.parms['pathjob'])
        return src_bji.pathsum_scale()
    def remove_workspace_scaling(self,code,ordering):
        if not code.endswith('normed'):
            return ordering
        s = self.pathsum_scale()
        return [(wsa,v/s) for wsa,v in ordering]
    def run(self):
        self.make_std_dirs()
        self.data_source = GpbrInput.source_factory(
                            self.ws,
                            self.parms['pathjob'],
                            )
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
        min_cycles_per_process = 25
        import math
        max_cores_wanted = int(math.ceil(self.reps/min_cycles_per_process))

        # Set the minimum cores to 1/4th of the max cores (though we cap
        # it to make sure it is satisfiable).
        min_cores_wanted = min(13, int(math.ceil(max_cores_wanted/4)))
        got = self.rm.wait_for_resources(self.job.id,[0,(min_cores_wanted,max_cores_wanted)])
        self.remote_cores = got[1]
        p_wr.put("get remote resources",
                "got %d remote cores" % self.remote_cores,
                )
        self.build_worklist()
        self.execute_pathsums()
        p_wr.put("execute pathsums","complete")
        self.post_process_all()
        # XXX In 2 of 4 test runs, DEA output files were damaged:
        # XXX - gpbr/29832/dea_directbgnormed_kts_ws_bootstrappingStats.txt
        # XXX - gpbr/29833/dea_indirectbgnormed_kts_ws_summaryStats.tsv
        # XXX - gpbr/29833/dea_indirectbgnormed_kts_ws_bootstrappingStats.tsv
        # XXX The first two were overwritten by the string:
        # XXX /home/carl/2xar/twoxar-demo/web1/dtk/plotly_png/package.json
        # XXX followed by a few nulls (3 in the first case, 16 in the second).
        # XXX The third was overwritten by:
        # XXX /home/carl/2xar/twoxar-demo/web1/browse/static/lib/plotly-20161003.min.j
        # XXX (with no trailing 's', or any other termination).
        # XXX The bad summaryStats file was removed, as it caused an exception
        # XXX when retrieving DEA results.
        # XXX These strings are only used in one place, to create symlinks for
        # XXX xvfb plotly rendering. This happens in post_process_all() above,
        # XXX and the 'real' content of those files is generated in
        # XXX check_enrichment() below, so it's not obvious how it happened.
        # XXX There's no difference in the logs compared with a subsequent
        # XXX 'good' run.  Keep an eye open for a recurrance.
        self.check_enrichment()
        self.finalize()
        p_wr.put("cleanup","complete")
    def post_process_all(self):
        for l in self.data_source.score_names():
            self.post_process(l[0],l[1])
    def build_worklist(self):
        worklist = []
        WorkItem=BackgroundPathsumWorkItem
        import math
        self.inner_cycles = int(math.ceil(self.reps/self.remote_cores))
        self.outer_cycles = int(math.ceil(self.reps/self.inner_cycles))
        print('running',self.reps,'repetitions on',self.remote_cores,'cores')
        print('%d inner cycles, %d outer cycles' % (
                self.inner_cycles,
                self.outer_cycles,
                ))
        for i in range(self.outer_cycles):
            wi = WorkItem()
            wi.serial = len(worklist)
            import random
            if self.parms['seed'] == -1:
                rng = random
            else:
                rng = random.Random(self.parms['seed'])
            offset = rng.randint(0, 1e10)
            wi.serial_offset = offset
            worklist.append(wi)
        WorkItem.pickle(self.indir,'worklist',worklist)
        path_settings = self.data_source.settings()
        vfd = self.ws.get_versioned_file_defaults()
        # These could move on to the config form if needed
        path_settings['uniprot_flavor'] = vfd['uniprot']
        path_settings['uniprot_role'] = 'Uniprot_data'
        if path_settings['combo_with']:
            d = self.ws.get_combo_therapy_data(path_settings['combo_with'])
            from dtk.prot_map import DpiMapping
            dpi = DpiMapping(path_settings['p2d_file'])
            from algorithms.bulk_pathsum import get_combo_fixed
            path_settings['combo_fixed'] = get_combo_fixed(d,dpi)
        self.data_source.build_tissue_files(self.indir)
        WorkItem.pickle(self.indir,'context',
                self.data_source.make_context(path_settings,self.inner_cycles)
                )
    def build_inputs(self):
        # generate dpi mapping file
        WorkItem=BackgroundPathsumWorkItem
        path_settings = self.data_source.settings()
        WorkItem.build_dpi_map(self.indir,
                    int(path_settings['ws_id']),
                    path_settings['p2d_file'],
                    )
    def execute_pathsums(self):
        import datetime
        start = datetime.datetime.now()
        print(str(start),self.reps,'parallel background runs started')

        from pathlib import Path
        opts = [
            '--cores', str(self.remote_cores),
            Path(self.indir),
            Path(self.outdir)
        ]
        self.run_remote_cmd('scripts/bulk_pathsum.py', opts, local=False)

        end = datetime.datetime.now()
        print(str(end),'complete; elapsed',str(end-start))
    def post_process(self,stem,code):
        print('post_process',stem,code)
        # read in 'real' pathsum scores
        from browse.models import Workspace
        ws = Workspace.objects.get(pk=self.parms['ws_id'])
        path_bji = JobInfo.get_bound(ws,self.parms['pathjob'])
        cat = path_bji.get_data_catalog()
        raw_scores = dict(cat.get_ordering(code,True))
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
        from dtk.plot import plot_vectors,PlotlyPlot,scatter2d,Color,fig_legend
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
        pp.save(fn+'.plotly')
        # generate synced score plots
        kts = ws.get_wsa_id_set(ws.eval_drugset)
        text = []
        tmp = []
        ids = []
        name_map = self.ws.get_wsa2name_map()
        for wsa in raw_scores.keys():
            tmp.append(
                (
                    raw_scores[wsa],
                    bg_scores[wsa],
                    normed_scores[wsa],
                    wsa in kts,
                )
             )
            text.append(name_map.get(wsa, wsa))
            ids.append(wsa)
        fn = os.path.join(self.tmp_pubdir,stem+'_raw_bg_scat')
        pp = scatter2d('raw','bg',
                [(x[0],x[1]) for x in tmp],
                class_idx=[1 if x[3] else 0 for x in tmp],
                classes=[
                        ('Unknown',{'color':Color.default, 'opacity':0.5}),
                        ('KT',{'color':Color.highlight, 'opacity':0.5}),
                        ],
                text=text,
                ids=('drugpage',ids),
                title='%s raw vs bg'%(stem,),
                bins = True
                )
        pp._layout['margin'] = dict(
                                     b=120
                                     )
        pp._layout['annotations'] = [fig_legend([
                                      "A drugs' score from the input PathSum-scored "
                                      +'run is plotted versus a background score '
                                      ,'derived using no disease-specific data. '
                                      +'Drugs below the reference line are ',
                                      'particularly promising as they have a higher '
                                      +'than expected score.'
                                     ],-0.25
                                     )]
        pp.save(fn+'.plotly')
        fn = os.path.join(self.tmp_pubdir,stem+'_raw_normed_scat')
        pp = scatter2d('raw','normed',
                [(x[0],x[2]) for x in tmp],
                class_idx=[1 if x[3] else 0 for x in tmp],
                classes=[
                        ('Unknown',{'color':Color.default, 'opacity':0.5}),
                        ('KT',{'color':Color.highlight, 'opacity':0.5}),
                        ],
                bins = True,
                title='%s raw vs normed'%(stem,),
                refline=False,
                )
        pp.save(fn+'.plotly')
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
            pp.save(fn+'.plotly')
        # write out bg_avg, normed and pval score file for catalog
        fn = getattr(self,'fn_'+stem+'_scores')
        with open(fn,'w') as f:
            codetype = self.dpi_codegroup_type()
            f.write('\t'.join([
                    codetype,
                    stem+'bg',
                    stem+'bgnormed',
                    stem+'pval',
                    ])+'\n')
            for label in raw_scores:
                f.write('\t'.join([str(x) for x in (
                        label,
                        bg_scores[label],
                        normed_scores[label],
                        pval_scores[label],
                        )])+'\n')

if __name__ == "__main__":
    MyJobInfo.execute(logger)
