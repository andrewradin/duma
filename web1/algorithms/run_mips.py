#!/usr/bin/env python3

import sys
from path_helper import PathHelper,make_directory

import os
import django
import django_setup

import logging
logger = logging.getLogger("algorithms.run_mips")

from django import forms

import json
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo
import runner.data_catalog as dc
from algorithms.exit_codes import ExitCoder
from browse.models import WsAnnotation
from dtk.files import get_file_records
from algorithms.run_gpath import plot


class ConfigForm(forms.Form):
    p2d_file = forms.ChoiceField(label='DPI dataset')
    p2d_t = forms.FloatField(label='Min DPI evidence')
    p2d_w = forms.FloatField(label='DPI weight',initial=1)
    p2p_file = forms.ChoiceField(label='PPI Dataset')
    p2p_t = forms.FloatField(label='Min PPI evidence')
    p2p_w = forms.FloatField(label='PPI weight',initial=1)
    t2p_w = forms.FloatField(label='T2P weight',initial=1)

    randomize = forms.BooleanField(initial=False
                    ,label='Randomize the gene names to generate a negative control dataset.'
                    ,required=False
                )
    combo_with = forms.ChoiceField(label='In combination with',initial=''
                      ,choices=(('','None'),)
                      ,required=False
                  )
    combo_type = forms.ChoiceField(label='Combo therapy algorithm'
                      ,choices=(
                               ('add','Add to Drug'),
                               ('sub','Subtract from Disease'),
                               )
                     ,required=False
                 )
    _subtype_name = "job_subtype"
    def __init__(self, ws, copy_job, *args, **kwargs):
        super(ConfigForm,self).__init__(*args, **kwargs)
        self.ws = ws
        self.input_count = 0
        # reload choices on each form load -- first DPI...
        f = self.fields['p2d_file']
        from dtk.prot_map import DpiMapping
        f.choices = DpiMapping.choices(ws)
        f.initial = ws.get_dpi_default()
        self.fields['p2d_t'].initial = ws.get_dpi_thresh_default()
        # ...then PPI
        f = self.fields['p2p_file']
        from dtk.prot_map import PpiMapping
        f.choices = list(PpiMapping.choices())
        f.initial = ws.get_ppi_default()
        self.fields['p2p_t'].initial = ws.get_ppi_thresh_default()
        # ...then combo therapies
        f = self.fields['combo_with']
        f.choices = [('','None')]+self.ws.get_combo_therapy_choices()
        f.initial = f.choices[0][0]
        if copy_job:
            self.from_json(copy_job.settings_json)
    def as_html(self):
        return self.as_p()
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

class MyJobInfo(JobInfo):
    def description_text(self):
        return '''
        <b>Monarch Initiative PathSum</b> scores disease to phenotype &
         gene to phenotype using the Pathsum algorithm.
        '''
    def settings_defaults(self,ws):
        form=ConfigForm(ws,None)
        return {
                'default':form.as_dict(),
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
                    "MIPS",
                    "Monarch Initiative PathSum",
                    )
        # any base class overrides for unbound instances go here
        self.publinks = (
                (None,'barpos.plotly',),
                )
        self.qc_plot_files = (
                'barpos.plotly',
                )
        self.needs_sources = False
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.ec = ExitCoder()
            # input files
            # all set by bulk pathsum
            # output files
            self.outfile = self.lts_abs_root+'mips.tsv'
            self.pathsum_detail = self.lts_abs_root+'path_detail.tsv'
            # Older jobs have this named as .tsv, newer have .tsv.gz.
            if not os.path.exists(self.pathsum_detail):
                self.pathsum_detail += '.gz'
            self.pathsum_detail_label = "Monarch"
            self.barpos = self.tmp_pubdir+"barpos.plotly"
            # published output files

    def get_data_code_groups(self):
        from math import log
        codes = [
            dc.Code('mipsd',label='Direct', fmt='%0.4f'),
            dc.Code('mipsi',label='Indirect', fmt='%0.4f'),
            dc.Code('mipskey',valtype='str',hidden=True),
            ]
        codetype = self.dpi_codegroup_type('p2d_file')
        return [
                dc.CodeGroup(codetype,self._std_fetcher('outfile'), *codes),
                ]
    def pathsum_scale(self):
        from algorithms.bulk_pathsum import extract_tissue_count_from_log
        return extract_tissue_count_from_log(self.job.id)
    def remove_workspace_scaling(self,code,ordering):
        if code == 'mipskey':
            return ordering
        s = self.pathsum_scale()
        return [(wsa,v/s) for wsa,v in ordering]
    def get_target_key(self,wsa):
        cat = self.get_data_catalog()
        try:
            val,_ = cat.get_cell('mipskey',wsa.id)
            return val
        except ValueError:
            return super(MyJobInfo,self).get_target_key(wsa)
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.lts_abs_root)
        make_directory(self.tmp_pubdir)
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "setup",
                "wait for remote resources",
                "score pathsums",
                "check enrichment",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.setup()
        p_wr.put("setup",'complete')
        got = self.rm.wait_for_resources(self.job.id,[
                                    0,
                                    self.remote_cores_wanted,
                                    ])
        self.remote_cores_got = got[1]
        p_wr.put("wait for remote resources"
                        ,"complete; got %d cores"%self.remote_cores_got
                        )
        self.run_remote()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("score pathsums","complete")
        self.report()
        self.check_enrichment()
        plot(self.plot_data,
                  self.barpos,
                  title='N prots per pheno',
                  xlabel='Number of proteins',
                  ylabel='Datasets')
        self.finalize()
        p_wr.put("check enrichment","complete")
    def run_remote(self):
        options = [
                  '--cores', str(self.remote_cores_got),
                  self.mch.get_remote_path(self.indir),
                  self.mch.get_remote_path(self.outdir)
                  ]
        print(('command options',options))
        self.copy_input_to_remote()
        self.make_remote_directories([
                                self.outdir,
                                self.tmp_pubdir,
                                ])
        rem_cmd = self.mch.get_remote_path(
                                    os.path.join(PathHelper.website_root,
                                                 "scripts",
                                                 "bulk_pathsum.py"
                                                )
                                    )
        self.mch.check_remote_cmd(' '.join([rem_cmd]+options))
        self.copy_output_from_remote()

    def _make_mi_data(self):
        from dtk.monarch import MonarchGene
        from dtk.monarch import MonarchDis
        MG = MonarchGene(self.ws.get_versioned_file_defaults(), self.ws)
        MD = MonarchDis(self.ws.get_versioned_file_defaults(), self.ws)
        ordering, phenoIDs_to_names=MD.get_mips_data()
        ph2g2score=MG.get_mips_data([x for x in phenoIDs_to_names])
        return ordering, ph2g2score, phenoIDs_to_names

    def score_mi_data(self):
        from dtk.monarch import score_mips
        ordering, ph2g2score, phenoIDs_to_names = self._make_mi_data()
        mi_data={}
        for pheno, p_score in ordering:
            if pheno not in ph2g2score:
                continue #implicit 0 for nonoverlaps
            mi_data[pheno]={}
            for prot in ph2g2score[pheno]:
                mi_data[pheno][prot]=score_mips(ph2g2score[pheno][prot], p_score)
        return mi_data, phenoIDs_to_names


    def setup(self):
        from algorithms.bulk_pathsum import PathsumWorkItem
        import numpy as np
        WorkItem = PathsumWorkItem
        wi = WorkItem()
        wi.serial = 0
        wi.detail_file=self.parms.get('detail_file', True)
        wi.compress_detail_file=True
        wi.show_stats=True
        wi.map_to_wsa=False
        worklist = [wi]
        WorkItem.pickle(self.indir,'worklist',worklist)
        mips_settings = self.job.settings()
        if mips_settings['combo_with']:
            d = self.ws.get_combo_therapy_data(mips_settings['combo_with'])
            from dtk.prot_map import DpiMapping
            dpi = DpiMapping(mips_settings['p2d_file'])
            from algorithms.bulk_pathsum import get_combo_fixed
            mips_settings['combo_fixed'] = get_combo_fixed(d,dpi)
        self.mi_data, phenoIDs_to_names = self.score_mi_data()
        self.plot_data = []

        for pheno,score_d in self.mi_data.items():
            WorkItem.build_nontissue_file(
                self.indir,
                pheno,
                score_d
                )
            num_prots = len(score_d)
            self.plot_data.append((phenoIDs_to_names[pheno], num_prots))
        from dtk.data import merge_dicts,dict_subset
        context = merge_dicts(
                        {
                         'tissues':self.mi_data,
                        },
                        dict_subset(
                                    mips_settings,
                                    [
                                     'randomize',
                                     'p2p_file',
                                     't2p_w',
                                     'p2p_w',
                                     'p2p_t',
                                     'p2d_file',
                                     'p2d_w',
                                     'p2d_t',
                                     'ws_id',
                                     'combo_with',
                                     'combo_type',
                                     'combo_fixed'
                                    ],
                                    )
                            )
        WorkItem.pickle(self.indir,'context',context)
        # generate dpi mapping file
        WorkItem.build_dpi_map(self.indir,
                               int(mips_settings['ws_id']),
                               mips_settings['p2d_file'],
                               )
        self._set_remote_cores_wanted()

    def _set_remote_cores_wanted(self):
        self.remote_cores_wanted=1
    def report(self):
        import shutil
        if os.path.exists(self.outdir+'path_detail0.tsv.gz'):
            shutil.move(self.outdir+'path_detail0.tsv.gz',self.pathsum_detail)
        self._load_scores()
        self._get_converter()
        with open(self.outfile, 'w') as f:
            score_map=dict(
                    direct='mipsd',
                    indirect='mipsi',
                    )
            codetype = self.dpi_codegroup_type('p2d_file')
            f.write("\t".join([codetype] + [
                    score_map.get(name,name)
                    for name in self.score_types
                    ]+['mipskey']) + "\n")
            used_wsa = set()
            priority_order = sorted(
                        list(self.scores.items()),
                        key=lambda x:x[1]['direct'],
                        reverse=True,
                        )
            for key,d in priority_order:
                try:
                    wsas = self.conv[key]
                except KeyError:
                    print('Unable to find WSA for', key)
                    continue
                for w in wsas:
                    if w in used_wsa:
                        self.info("skipping additional binding for wsa_id %d"
                            ": key %s; direct %s; indirect %s",
                            w,
                            key,
                            d['direct'],
                            d['indirect'],
                            )
                        continue
                    used_wsa.add(w)
                    out = [str(w)]
                    for st in self.score_types:
                        try:
                            out.append(d[st])
                        except KeyError:
                            out.append('0')
                    out.append(key)
                    f.write("\t".join(out) + "\n")
    def _load_scores(self):
# deleted 'direction' from below, b/c that's not relevant here
        self.score_types = ['direct', 'indirect']
        self.scores = {}
        for s in self.score_types:
            for frs in get_file_records(self.outdir +'/'+s+'0score',
                                        keep_header = True,
                                        parse_type = 'tsv'
                                        ):
                if frs[0] not in self.scores:
                    self.scores[frs[0]] = {}
                self.scores[frs[0]][s] = frs[1]
    def _get_converter(self):
        from dtk.prot_map import DpiMapping
        self.dpi = DpiMapping(self.parms['p2d_file'])
        self.conv = self.dpi.get_wsa_id_map(self.ws)

    def add_workflow_parts(self,ws,parts):
        jobnames = self.get_jobnames(ws)
        assert len(jobnames) == 1
        jobname=jobnames[0]
        uji = self # simplify access inside MyWorkflowPart
        class MyWorkflowPart:
            label=uji.source_label(jobname)
            enabled_default=uji.data_status_ok(ws, 'Monarch', 'Monarch Initiative Values')
            def add_to_workflow(self,wf):
                cm_info = wf.std_wf.get_main_cm_info(uji.job_type)
                cm_info.pre.add_pre_steps(wf)
                from dtk.workflow import mipsStep
                my_name='mips'
                mipsStep(wf,my_name,
                        )
                cm_info.post.add_post_steps(wf,my_name)
        parts.append(MyWorkflowPart())



if __name__ == "__main__":
    MyJobInfo.execute(logger)
