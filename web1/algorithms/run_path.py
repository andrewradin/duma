#!/usr/bin/env python3

import sys
from path_helper import PathHelper,make_directory

import os
import django
import django_setup

from django import forms

from browse.models import Tissue,WsAnnotation
from tools import ProgressWriter
from runner.process_info import JobInfo, StdJobInfo
import runner.data_catalog as dc
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager
from dtk.prot_map import DpiMapping,PpiMapping
from algorithms.run_gpath import plot

import json
import algorithms.pathsum2 as ps
import algorithms.bulk_pathsum as bps

import logging
logger = logging.getLogger("algorithms.run_pathsum")

def get_tissue_ids_from_settings(d):
    result = []
    import re
    for key in d:
        m = re.match(r't_([0-9]+)$',key)
        if m:
            result.append(int(m.group(1)))
    result.sort()
    return result

def get_tissue_settings_keys(t_id):
    return tuple((
            x % t_id
            for x in ('t_%d','t_%d_fc')
            ))

class ConfigForm(forms.Form):
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
    t2p_w = forms.FloatField(label='T2P weight',initial=1)
    p2d_file = forms.ChoiceField(label='DPI dataset')
    p2d_w = forms.FloatField(label='DPI weight',initial=1)
    p2d_t = forms.FloatField(label='Min DPI evidence')
    p2p_file = forms.ChoiceField(label='PPI dataset',required=False)
    p2p_w = forms.FloatField(label='PPI weight',initial=1)
    p2p_t = forms.FloatField(label='Min PPI evidence')
    _subtype_name = "job_subtype"
    def __init__(self, ts_id, copy_job, *args, **kwargs):
        if copy_job:
            ts_id = copy_job.name.split('_')[1]
        elif ts_id is None:
            # if both copy_job and ts_id are None, there must be POST data
            ts_id=args[0][self._subtype_name]
        super(ConfigForm,self).__init__(*args, **kwargs)
        self.ts_id = ts_id
        from browse.models import TissueSet
        ts = TissueSet.objects.get(pk=self.ts_id)
        # reload choices on each form load -- first DPI...
        f = self.fields['p2d_file']
        f.choices = DpiMapping.choices(ts.ws)
        f.initial = ts.ws.get_dpi_default()
        self.fields['p2d_t'].initial = ts.ws.get_dpi_thresh_default()
        # ...then PPI
        f = self.fields['p2p_file']
        f.choices = [('','None')]+list(PpiMapping.choices())
        f.initial = ts.ws.get_ppi_default()
        self.fields['p2p_t'].initial = ts.ws.get_ppi_thresh_default()
        # ...then combo therapies
        f = self.fields['combo_with']
        f.choices = [('','None')]+ts.ws.get_combo_therapy_choices()
        f.initial = f.choices[0][0]
        # build individual threshold fields for each tissue
        # XXX - we may also want to change the protein count display on
        # XXX   the tissue page to reflect the adjusted thresholds
        qs = Tissue.objects.filter(tissue_set_id=ts_id)
        for t in qs:
            _,_,_,total = t.sig_result_counts()
            if not total:
                continue
            ev_key, fc_key = get_tissue_settings_keys(t.pk)
            label_stem='%s (%d) %%s threshold' % (t.name,t.pk)
            self.fields[ev_key] = forms.FloatField(
                                label=label_stem%'ev',
                                initial=t.ev_cutoff,
                                )
            self.fields[fc_key] = forms.FloatField(
                                label=label_stem%'fc',
                                initial=t.fc_cutoff,
                                )
        if copy_job:
            self.from_json(copy_job.settings_json)
    def as_html(self):
        from django.utils.html import format_html
        return format_html(u'''
                <input name="{}" type="hidden" value="{}"/>
                <table>{}</table>
                '''
                ,self._subtype_name
                ,self.ts_id
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
        from browse.models import TissueSet
        tset = TissueSet.objects.get(pk=self.ts_id)
        p = {
            'ws_id':tset.ws_id,
            }
        # copy all non-tissue data into settings
        for f in self:
            key = f.name
            if key == 'iterate_over':
                continue
            if key.startswith('t_'):
                continue
            p[key] = src[key]
        # now copy all non-excluded tissue data
        for t_id in tset.tissue_ids():
            ev_key, fc_key = get_tissue_settings_keys(t_id)
            # tisues without any proteins are excluded when the form
            # is built, and so won't be present in src; skip silently
            try:
                value = src[ev_key]
            except KeyError:
                continue
            if value <= 1:
                p[ev_key] = value
                p[fc_key] = src[fc_key]
        return p
    def from_json(self,init):
        p = json.loads(init)
        if 'indr' in p:
            # convert from legacy format
            if p['indr']:
                p['p2p_file'] = 'drpias.default'
            else:
                p['p2p_file'] = ''
            del p['indr']
        for f in self:
            # if field is present in settings, copy it;
            # otherwise handle some special cases where they
            # may be missing
            if f.name in p:
                f.field.initial = p[f.name]
            elif f.name.startswith('t_'):
                if f.name.endswith('_fc'):
                    # assume legacy filtering (no FC threshold)
                    f.field.initial = 0
                else:
                    # assume a tissue with threshold > 1
                    f.field.initial = 2
            # else leave at default

class MyJobInfo(StdJobInfo):
    descr = '''
        <b>Pathsum</b> is the original DUMA method; it connects drugs to
        diseases via drug protein interaction data, and disease gene expression
        data.  It also has the option to include indirect interactions by
        adding protein-protein interactions.
        <i> Pathsum includes 2 distinct combo methods: add and subtract</i>
        '''
    short_label='Pathsum'
    page_label='Pathsum'

    def settings_defaults(self,ws):
        result = {}
        from browse.models import TissueSet
        for ts in TissueSet.objects.filter(ws=ws):
            form = ConfigForm(ts.id,None)
            result[ts.name] = form.as_dict()
        return result
    def source_label(self,jobname):
        ts_id,ws_id = self._parse_jobname(jobname)
        from browse.models import TissueSet
        ts = TissueSet.objects.get(pk=ts_id)
        return self.get_source_label_for_tissue_set(ts)
    def get_source_label_for_tissue_set(self,ts):
        return ts.ts_label()+' Pathsum'
    def build_role_code(self,jobname,settings_json):
        ts_id,_ = self._parse_jobname(jobname)
        return self._tissue_set_role_code(ts_id)
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        if copy_job or not job_type:
            ts_id = None
        else:
            ts_id = job_type.split('_')[1]
        form = ConfigForm(ts_id,copy_job)
        return form.as_html()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        form = ConfigForm(None,None,post_data)
        if not form.is_valid():
            return (form.as_html(),None)
        p = form.as_dict()
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(p)
                            )
        next_url = ws.reverse('nav_progress',job_id)
        return (None, next_url)
    def _parse_jobname(self,jobname):
        fields = jobname.split('_')
        return (int(fields[1]),int(fields[2]))
    def get_jobnames(self,ws):
        '''Return a list of all jobnames for this plugin in a workspace.'''
        return [
                self.get_jobname_for_tissue_set(ts)
                for ts in ws.get_tissue_sets()
                ]
    def get_jobname_for_tissue_set(self,ts):
        return self._format_jobname_for_tissue_set(self.job_type,ts)
    def out_of_date_info(self,job,jcc):
        ts_id,ws_id = self._parse_jobname(job.name)
        return self._out_of_date_from_tissue_set(job,ts_id)
    def ordering_jobnames(self,jobname,jcc=None):
        result = set()
        # wait for any sig jobs in this tissue set to complete
        ts_id,ws_id = self._parse_jobname(jobname)
        for t in Tissue.objects.filter(tissue_set_id=ts_id):
            result.add(t.get_sig_jobname())
        return result
    def __init__(self,ws=None,job=None):
        super().__init__(
                    ws=ws,
                    job=job,
                    src=__file__,
                    )
        # any base class overrides for unbound instances go here
        # job-specific properties
        self.publinks = [
                (None,'barpos.plotly'),
                ]
        self.qc_plot_files = (
                'barpos.plotly',
                )
        if self.job:
            # stash common ordered list of tissue ids here
            self.tissue_ids = get_tissue_ids_from_settings(self.parms)
            # flag that DPI info not loaded yet
            self.dpi = None
            self.outfile = self.lts_abs_root+'path_scores.tsv'
            self.pathsum_detail=self.lts_abs_root+'path_detail.tsv'
            # Older jobs have this named as .tsv, newer have .tsv.gz.
            if not os.path.exists(self.pathsum_detail):
                self.pathsum_detail += '.gz'
            self.pathsum_detail_label = "GE Tissue"
            self.barpos = self.tmp_pubdir+"barpos.plotly"
    def _load_dpi_map(self):
        if self.dpi:
            return
        from dtk.prot_map import DpiMapping
        self.dpi = DpiMapping(self.parms['p2d_file'])
        self.dpi_map = self.dpi.get_wsa_id_map(self.ws)
    def has_indirect(self): return bool(self.parms.get('p2p_file'))
    def get_data_code_groups(self):
        cand_codes = [
                dc.Code('direct',label='Direct'),
                dc.Code('indirect',label='Indirect'),
                dc.Code('direction',label='Direction', efficacy=False),
                dc.Code('absDir',label='Abs(direction)',
                                calc=(lambda x:abs(x),'direction'),
                                ),
                dc.Code('pathkey',valtype='str',hidden=True),
                        # this allows us to know which detail file records
                        # are associated with this drug
                ]
        codetype = self.dpi_codegroup_type('p2d_file')
        return [
                dc.CodeGroup(codetype,self._std_fetcher('outfile'), *cand_codes),
                ]
    def pathsum_scale(self):
        from algorithms.bulk_pathsum import extract_tissue_count_from_log
        return extract_tissue_count_from_log(self.job.id)
    def remove_workspace_scaling(self,code,ordering):
        if code == 'pathkey':
            return ordering
        s = self.pathsum_scale()
        return [(wsa,v/s) for wsa,v in ordering]
    def get_target_key(self,wsa):
        cat = self.get_data_catalog()
        val,_ = cat.get_cell('pathkey',wsa.id)
        return val
        # This is only called when trying to extract pathsum detail for
        # this job. In run_path, if the pathkey doesn't exist, then the
        # underlying record doesn't exist either, and there's no key we
        # can return that would find anything. So, just short-circuit the
        # process here.
        #if val:
        #    return val
        #return super(MyJobInfo,self).get_target_key(wsa)

    def _get_tis_names(self):
        self.tis_names = {}
        for tid in self.tissue_ids:
            t = Tissue.objects.get(id=tid)
            self.tis_names[tid] = str(tid) + " - " + t.concise_name()

    def run(self):
        self.make_std_dirs()
        # initialize progress reporting
        self.p_wr = ProgressWriter(self.progress , [
                'wait for resources',
                'setup',
                'build input files',
                'wait for remote resources',
                'run pathsum',
                'wait for local resources',
                'prepare score map',
                'candidate storage',
                'checking enrichment',
                ])
        print('loc', self.barpos)
        self.setup()
        self.build_inputs()
        self.run_remote()
        self.prep_score_map()
        self.save_scores()
        self.check_enrichment()
        self._get_tis_names()
        print('tissues', self.tis_names)
        for i, score in enumerate(self.tissue_ev_plot_data):
            nm = self.tis_names[int(score[0].split('_')[1])]
            self.tissue_ev_plot_data[i] = (nm, score[1])
        plot(self.tissue_ev_plot_data,
                  self.barpos,
                  title='Number of proteins in each dataset',
                  xlabel='Number of proteins',
                  ylabel='Datasets')
        self.finalize()
        self.p_wr.put('checking enrichment' ,"complete")
        return 0
    def setup(self):
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        self.p_wr.put("wait for resources","complete")
        self.net = ps.Network()
        self._load_dpi_map()
        if self.parms['combo_with']:
            # single fixed drug combo case; get the fixed drug data
            d = self.ws.get_combo_therapy_data(self.parms['combo_with'])
            self.combo_fixed = bps.get_combo_fixed(d,self.dpi)
            print('combo_fixed',self.combo_fixed)
        self.tissue_idx=0
        self.t2p_idx=1
        self.p2p_idx=2
        self.p2d_dir_idx=3
        self.p2d_indir_idx=4
        self.p2d_src_dest = [(self.t2p_idx,self.p2d_dir_idx)]
        max_bindings = sum(map(len,list(self.dpi_map.values())))
        max_keys = len(self.dpi_map)
        with open(self.ws.ws_dir()+"/pathsum_matchable_drugs.txt","w") as f:
            for v in self.dpi_map.values():
                for wsa_id in v:
                    f.write("%s\n" % wsa_id)
        summary = "%d potential targets; %d keys" % (max_bindings,max_keys)
        self.info(summary)
        self.p_wr.put("setup",summary)
    def build_inputs(self):
        worklist = []
        from algorithms.bulk_pathsum import PathsumWorkItem
        WorkItem=PathsumWorkItem
        wi = WorkItem()
        wi.serial = 0
        wi.detail_file=self.parms.get('detail_file', True)
        wi.compress_detail_file=True
        wi.show_stats=True
        wi.map_to_wsa=False
        worklist.append(wi)
        WorkItem.pickle(self.indir,'worklist',worklist)
        path_settings = self.job.settings()
        if path_settings['combo_with']:
            d = self.ws.get_combo_therapy_data(path_settings['combo_with'])
            from dtk.prot_map import DpiMapping
            dpi = DpiMapping(path_settings['p2d_file'])
            from algorithms.bulk_pathsum import get_combo_fixed
            path_settings['combo_fixed'] = get_combo_fixed(d,dpi)
        from algorithms.run_path import get_tissue_ids_from_settings
        tissues = get_tissue_ids_from_settings(path_settings)
        from algorithms.run_path import get_tissue_settings_keys
        self.tissue_ev_plot_data = []
        self.tissue_fc_plot_data = []
        for tissue_id in tissues:
            ev_key, fc_key = get_tissue_settings_keys(tissue_id)
            counts = WorkItem.build_tissue_file(
                    self.indir,
                    tissue_id,
                    path_settings[ev_key],
                    path_settings[fc_key],
                    counts=True
                    )
            self.tissue_ev_plot_data.append((ev_key, counts))
        from dtk.data import merge_dicts,dict_subset
        context = merge_dicts(
                    {
                            'tissues':tissues,
                            },
                    dict_subset(
                            path_settings,
                            [
                                'randomize',
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
                            ],
                            )
                    )
        WorkItem.pickle(self.indir,'context',context)
        # generate dpi mapping file
        WorkItem.build_dpi_map(self.indir,
                    int(path_settings['ws_id']),
                    path_settings['p2d_file'],
                    )
        self.p_wr.put('build input files' ,"complete")
    def run_remote(self):
        got = self.rm.wait_for_resources(self.job.id,[0,1])
        self.remote_cores = got[1]
        self.p_wr.put("wait for remote resources",
                "got %d remote cores" % self.remote_cores,
                )
        import datetime
        start = datetime.datetime.now()
        from pathlib import Path
        opts = [
            '--cores', str(self.remote_cores),
            Path(self.indir),
            Path(self.outdir),
        ]
        self.run_remote_cmd('scripts/bulk_pathsum.py', opts, local=False)
        end = datetime.datetime.now()
        print(str(end),'complete; elapsed',str(end-start))
        self.p_wr.put('run pathsum' ,"complete")
    def prep_score_map(self):
        self.rm.wait_for_resources(self.job.id,[1])
        self.p_wr.put("wait for local resources","complete")
        scores = []
        from dtk.readtext import parse_delim
        for score_name in 'direct indirect direction'.split():
            fn = os.path.join(self.outdir,'%s0score'%score_name)
            try:
                with open(fn) as f:
                    d = dict(parse_delim(f))
            except IOError:
                d = {}
            scores.append(d)
        all_keys = set()
        for d in scores:
            all_keys |= set(d.keys())
        self.priority_order = [
                (key,) + tuple(float(x.get(key,0)) for x in scores)
                for key in all_keys
                ]
        self.priority_order.sort(key=lambda x: x[1],reverse=True)
        self.p_wr.put("prepare score map"
            ,"complete"
            )
    def save_scores(self):
        total = 0
        used_wsa = set()
        ignored = 0
        with open(self.outfile, 'w') as f:
            codetype = self.dpi_codegroup_type('p2d_file')
            cols = [codetype,'direct','indirect','direction','pathkey']
            f.write('\t'.join(cols)+'\n')
            for path_key,direct,indirect,direction in self.priority_order:
                total += 1
                for wsa_id in self.dpi_map.get(path_key,[]):
                    if wsa_id in used_wsa:
                        # It's possible for a drug to match multiple keys in
                        # a dpi file. If both show up in the pathsum results,
                        # we only have a place for one score in the database.
                        # We keep the first one we see only, since that has
                        # the best direct score (due to priority ordering). But
                        # log all the detail here for what we discard.
                        ignored += 1
                        self.info("skipping additional binding for wsa_id %d"
                            ": key %s; direct %f; indirect %f, direction %f",
                            wsa_id,
                            path_key,
                            direct,
                            indirect,
                            direction,
                            )
                        continue
                    used_wsa.add(wsa_id)
                    f.write('\t'.join([str(x) for x in [
                            wsa_id,
                            direct,
                            indirect,
                            direction,
                            path_key,
                            ]])+'\n')
        self.info(
                "total of %d candidates processed (%d filtered)",
                total,
                ignored,
                )
        self.p_wr.put("candidate storage"
                ,"complete (%d filtered)" % ignored
                )
        import shutil
        if os.path.exists(self.outdir+'path_detail0.tsv.gz'):
            shutil.move(self.outdir+'path_detail0.tsv.gz',self.pathsum_detail)
    def add_workflow_parts(self,ws,parts,nonhuman=False):
        uji = self # simplify access inside MyWorkflowPart
        class MyWorkflowPart:
            def __init__(self,label,ts):
                self.label=label
                self.ts=ts
                self.enabled_default=uji.data_status_ok(ws,'GE',ts.ts_label())
            def add_to_workflow(self,wf):
                cm_info = wf.std_wf.get_main_cm_info(uji.job_type)
                cm_info.pre.add_pre_steps(wf)
                from dtk.workflow import PathStep
                my_name = uji._tissue_set_role_code(self.ts.id)
                PathStep(wf,my_name,
                        ts=self.ts,
                        thresh_overrides={}
                        )
                cm_info.post.add_post_steps(wf,my_name)
        from browse.models import Species
        for ts in ws.tissueset_set.all():
            is_human = ts.species == Species.HUMAN
            if ts.tissue_set.exists() and ts.num_valid_tissues() > 0 and is_human != nonhuman:
                parts.append(MyWorkflowPart(
                        self.get_source_label_for_tissue_set(ts),
                        ts,
                        ))

if __name__ == "__main__":
    MyJobInfo.execute(logger)
