#!/usr/bin/env python3

from builtins import range
import sys
from path_helper import PathHelper,make_directory

import os
import django
import django_setup

import logging
logger = logging.getLogger("algorithms.run_gpath")

from django import forms

import json
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo
import runner.data_catalog as dc
from algorithms.exit_codes import ExitCoder
from browse.models import WsAnnotation

class ConfigForm(forms.Form):
    p2d_file = forms.ChoiceField(label='DPI dataset')
    p2d_t = forms.FloatField(label='Min DPI evidence')
    p2d_w = forms.FloatField(label='DPI weight',initial=1)
    p2p_file = forms.ChoiceField(label='PPI Dataset')
    p2p_t = forms.FloatField(label='Min PPI evidence')
    p2p_w = forms.FloatField(label='PPI weight',initial=1)
    t2p_w = forms.FloatField(label='GWDS to Prot. weight',initial=1)
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
        f.choices = PpiMapping.choices()
        f.initial = ws.get_ppi_default()
        self.fields['p2p_t'].initial = ws.get_ppi_thresh_default()
        # build individual threshold fields for each gwas dataset
        from dtk.gwas import gwas_code

        self.fields['min_v2g'] = forms.FloatField(
                            label = "Min. v2g evid for SNPs->prot assocs",
                            initial = 0.1,
                            required = True,
                            )

        self.fields['max_ds_prots'] = forms.IntegerField(
                            label = "Maximum # of prot assocs per DS",
                            initial = 1000,
                            required = True,
                            help_text = "Each DS will take the top N scored prot associations",
                            )

        from browse.models import GwasDataset
        v2d = dict(GwasDataset.objects.filter(ws=ws).values_list('id', 'v2d_threshold'))

        for choice in self.ws.get_gwas_dataset_choices():
            ds_name = gwas_code(choice[0])
            self.input_count += 1
            self.fields[ds_name] = forms.FloatField(
                                label = "Min. v2d p-value for a SNP in " + ds_name,
                                initial = v2d[choice[0]],
                                required = True,
                                )
        # ...then combo therapies
        f = self.fields['combo_with']
        f.choices = [('','None')]+self.ws.get_combo_therapy_choices()
        f.initial = f.choices[0][0]
        if copy_job:
            self.from_json(copy_job.settings_json)
    def as_html(self):
        from django.utils.html import format_html
        return format_html(
                u'''<div class="well">There are currently {} GWAS datasets
                in this workspace.<p><p>
                You can add more datasets and configure default threshold values
                <a href="{}">
                here</a>.
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

class MyJobInfo(JobInfo):
    def description_text(self):
        return '''
        <b>gPath</b> generates drug scores by applying the pathsum algorithm
        to the collection of GWAS datasets in the workspace.
        '''
    def settings_defaults(self,ws):
        cfg = ConfigForm(ws, None)
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
                    "gPath",
                    "Genetics Pathsum",
                    )
        # any base class overrides for unbound instances go here
        self.publinks = (
                (None,'barpos.plotly'),
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
            self.outfile = self.lts_abs_root+'gpath.tsv'
            # published output files
            self.pathsum_detail = self.lts_abs_root+'path_detail.tsv'
            # Older jobs have this named as .tsv, newer have .tsv.gz.
            if not os.path.exists(self.pathsum_detail):
                self.pathsum_detail += '.gz'

            self.pathsum_detail_label = "GWAS Phenotype"
            self.barpos = self.tmp_pubdir+"barpos.plotly"
    def get_data_code_groups(self):
        from math import log
        codes = [
            dc.Code('gds',label='Direct', fmt='%0.4f'),
            dc.Code('gis',label='Indirect', fmt='%0.4f'),
            dc.Code('gpathkey',valtype='str',hidden=True),
            ]
        codetype = self.dpi_codegroup_type('p2d_file')
        return [
                dc.CodeGroup(codetype,self._std_fetcher('outfile'), *codes),
                ]
    def pathsum_scale(self):
        from algorithms.bulk_pathsum import extract_tissue_count_from_log
        return extract_tissue_count_from_log(self.job.id)
    def remove_workspace_scaling(self,code,ordering):
        if code == 'gpathkey':
            return ordering
        s = self.pathsum_scale()
        return [(wsa,v/s) for wsa,v in ordering]
    def get_target_key(self,wsa):
        cat = self.get_data_catalog()
        try:
            val,_ = cat.get_cell('gpathkey',wsa.id)
            return val
        except ValueError:
            return super(MyJobInfo,self).get_target_key(wsa)
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        make_directory(self.lts_abs_root)
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
        from dtk.gwas import selected_gwas
        all_gwds = len(self.ws.get_gwas_dataset_choices())
        used_gwds = len(selected_gwas(self.parms))
        if all_gwds > used_gwds:
            title = 'Number of proteins in each dataset excluding %i due to p-value threshold of zero' % (all_gwds-used_gwds)
        else:
            title = 'Number of proteins in each dataset'
        plot(self.gwds_plot_data,
                  self.barpos,
                  title=title,
                  xlabel='Number of proteins',
                  ylabel='Datasets',
                  pair=True)
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
    def setup(self):
        # don't actually need this converter until later, but it makes the DPI calls easier
        self._get_converter()
        from algorithms.bulk_pathsum import PathsumWorkItem
        WorkItem = PathsumWorkItem
        wi = WorkItem()
        wi.serial = 0
        wi.detail_file=self.parms.get('detail_file', True)
        wi.compress_detail_file=True
        wi.show_stats=True
        wi.map_to_wsa=False
        worklist = [wi]
        WorkItem.pickle(self.indir,'worklist',worklist)
        gpath_settings = self.job.settings()
        if gpath_settings['combo_with']:
            d = self.ws.get_combo_therapy_data(gpath_settings['combo_with'])
            from dtk.prot_map import DpiMapping
            dpi = DpiMapping(gpath_settings['p2d_file'])
            from algorithms.bulk_pathsum import get_combo_fixed
            gpath_settings['combo_fixed'] = get_combo_fixed(d,dpi)
        self.gwds_plot_data = ([],[])
        self.gwas_data, self.gwas_data_exclude = get_gwas_data(self.ws.id, self.parms,exclude=True)
        for gwds_id in self.gwas_data:
            WorkItem.build_nontissue_file(
                self.indir,
                gwds_id,
                self.gwas_data[gwds_id]
                )
            if gwds_id not in list(self.gwas_data_exclude.keys()):
                self.gwas_data_exclude[gwds_id] = {}
            self.gwds_plot_data[0].append((gwds_id, len(list(self.gwas_data[gwds_id].keys()))))
            self.gwds_plot_data[1].append((gwds_id, len(list(self.gwas_data_exclude[gwds_id].keys()))))
        from dtk.data import merge_dicts,dict_subset
        context = merge_dicts(
                        {
                         'tissues':self.gwas_data,
                        },
                        dict_subset(
                                    gpath_settings,
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
                               int(gpath_settings['ws_id']),
                               gpath_settings['p2d_file'],
                               )
        self._set_remote_cores_wanted()
    def _set_remote_cores_wanted(self):
        self.remote_cores_wanted=1
    def report(self):
        import shutil
        if os.path.exists(self.outdir+'path_detail0.tsv.gz'):
            shutil.move(self.outdir+'path_detail0.tsv.gz',self.pathsum_detail)
        self._load_scores()
        with open(self.outfile, 'w') as f:
            codetype = self.dpi_codegroup_type('p2d_file')
            score_map=dict(
                    direct='gds',
                    indirect='gis',
                    )
            f.write("\t".join([codetype] + [
                    score_map.get(name,name)
                    for name in self.score_types
                    ]+['gpathkey']) + "\n")
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
        from dtk.files import get_file_records
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
            enabled_default=uji.data_status_ok(ws, 'Gwas', 'GWAS Datasets')
            def add_to_workflow(self,wf):
                cm_info = wf.std_wf.get_main_cm_info(uji.job_type)
                cm_info.pre.add_pre_steps(wf)
                from dtk.workflow import gpathStep
                my_name='gpath'
                gpathStep(wf,my_name,
                        thresh_overrides={}
                        )
                cm_info.post.add_post_steps(wf,my_name)
        parts.append(MyWorkflowPart())

def plot(data, save_dir, title=None, xlabel=None,ylabel=None, pair=False):
    from dtk.plot import PlotlyPlot,fig_legend
    from dtk.text import limit
    if pair:
        data,data1 = data
    sorted_data = sorted(data, key=lambda tup: tup[-1])
    names = [limit(i) for i,j in sorted_data]
    value = [j for i,j in sorted_data]
    if pair:
        value_ex_unsort = [j for i,j in data1]
        names_ex_unsort = [i for i,j in data1]
        value_ex = []
        for n in names:
            if n in names_ex_unsort:
                idx = names_ex_unsort.index(n)
                value_ex.append(value_ex_unsort[idx])
        diff = set(names_ex_unsort).difference(set(names))
        for n in diff:
            idx = names_ex_unsort.index(n)
            names.append(n)
            value.append(0)
            value_ex.append(value_ex_unsort[idx])
        totals = [value[i] + value_ex[i] for i in range(len(value))]

        names, value, value_ex, total = zip(*sorted(zip(names, value, value_ex,
                   [value[i] + value_ex[i] for i in range(len(value))]),
                   key=lambda tup: tup[-1]))
    if len(names) > 0 :
        max_tick_lengths = max([len(i) for i in names])
        ht = (len(names)+8)*25
        if pair:
            pp1 = PlotlyPlot(
                    [dict(
                           type='bar',
                           y = names,
                           x = value,
                           orientation='h',
                           name='Included'
                           ),
                    dict(
                           type='bar',
                           y = names,
                           x = value_ex,
                           orientation='h',
                           name='Excluded'
                           )],
                    dict(
                           title=title,
                           barmode='stack',
                           xaxis= {'title':xlabel
                             },
                           yaxis={
                               'title':ylabel
                               },
                           height=ht,
                           margin=dict(
                                 l=max(30,int(max_tick_lengths*10)),
                                 b=120
                                 ),

               )
            )
            pp1._layout['annotations'] = [fig_legend([
                                          'The number of proteins from each dataset is plotted, noting which were',
                                          'included in PathSum scoring or not. PathSum does <b>best when there are',
                                          '1,000 or fewer proteins per dataset</b>, though up to 1,500 proteins is acceptable.'
                                         ],-2.4/len(names)
                                         )]
        else:
            pp1 = PlotlyPlot(
                        [dict(
                              type='bar',
                               y = names,
                               x = value,
                               orientation='h'
                               )],
                        dict(
                               title=title,
                               xaxis= {'title':xlabel
                                 },
                               yaxis={
                                   'title':ylabel
                                   },
                               height=ht,
                               margin=dict(
                                     l=max(30,int(max_tick_lengths*10)),
                                     b=120
                                     ),

                   )
                )
            pp1._layout['annotations'] = [fig_legend([
                                          'The number of proteins used from each dataset is plotted. PathSum'
                                          ,'scoring does <b>best when there are 1,000 or fewer proteins'
                                          ,'per dataset</b>, though up to 1,500 or so is acceptable.'
                                         ],-2.4/len(names)
                                         )]
    else:
        pp1 = PlotlyPlot(
                    [],
                    dict(
                           title='No tissue data to display',
               )
            )

    print('Saving...')
    pp1.save(save_dir)


def get_gwas_data(ws_id, parms, exclude=False):
    from dtk.gwas import scored_gwas,selected_gwas

    score_args = {}
    for key in selected_gwas(parms):
        score_args[key] = {
            'v2d_threshold': parms[key],
            'v2g_threshold': parms['min_v2g'],
            'max_prot_assocs': parms['max_ds_prots'],
        }

    gwas_data = {
                key:scored_gwas(key, **score_args[key])
                for key in selected_gwas(parms)
                }
    gwas_data_exclude = {
                key:scored_gwas(key, **score_args[key], exclude=exclude)
                for key in selected_gwas(parms)
                }
    if not gwas_data:
        print('Unable to find any GWAS data. Quitting.')
        ec = ExitCoder()
        sys.exit(ec.encode('unableToFindDataError'))
    if exclude:
        return gwas_data,gwas_data_exclude
    else:
        return gwas_data


if __name__ == "__main__":
    MyJobInfo.execute(logger)
