#!/usr/bin/env python3

import sys
from path_helper import PathHelper,make_directory

import os
import django
import django_setup

import logging
logger = logging.getLogger("algorithms.run_capp")

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

def make_otarg_prot_assoc(ws, dis_ordering, score_func):
    from dtk.mondo import Mondo
    from dtk.open_targets import OpenTargets
    import browse.default_settings as def_set
    from collections import defaultdict
    from dtk.files import safe_name


    logger.info("Loading up datasets")
    mondo = Mondo(def_set.mondo.value(ws=ws))
    otarg = OpenTargets(def_set.openTargets.value(ws=ws))
    logger.info("Generating mapping for disease names")
    dis_names = [x[0] for x in dis_ordering]
    mondo_lists = mondo.map_meddra_to_mondo(dis_names, ws=ws)
    out = {}
    all_otarg_keys = set()
    otarg_keys_list = []
    logger.info("Finding all opentarget keys")
    for mondos in mondo_lists:
        otarg_keys = otarg.keys_from_mondo_ids(mondo, mondos)
        if otarg_keys:
            otarg_keys = set.union(*otarg_keys)
            all_otarg_keys.update(otarg_keys)
        otarg_keys_list.append(otarg_keys)
    
    logger.info("Grabbing disease scores")
    otkey2scores = otarg.get_multiple_diseases_scores(all_otarg_keys)

    logger.info("Generating output")
    for mondos, (dis_name, dis_input_score), otarg_keys in zip(mondo_lists, dis_ordering, otarg_keys_list):
        dis_scores = defaultdict(float)
        for otarg_key in otarg_keys:
            scores = otkey2scores[otarg_key]
            for prot, ot_score in scores.items():
                cur_score = score_func(ot_score['overall'], dis_input_score)
                dis_scores[prot] = max(dis_scores[prot], cur_score)
        # DGN uses safe_name here, so have to do the same to make names line up.
        out[safe_name(dis_name)] = dis_scores
    return out
            


def extract_job_id(p):
    # PLAT-1716 changed faers_run from a path to a job id
    try:
        return int(p['faers_run'])
    except ValueError:
        parts = p['faers_run'].split('/')
        return int(parts[7])

class ConfigForm(forms.Form):
    faers_run = forms.ChoiceField(label='FAERS Job ID to use',initial=''
                        ,choices=(('','None'),)
                        )
    p2d_file = forms.ChoiceField(label='DPI dataset')
    p2d_t = forms.FloatField(label='Min DPI evidence')
    p2d_w = forms.FloatField(label='DPI weight',initial=1)
    p2p_file = forms.ChoiceField(label='PPI Dataset')
    p2p_t = forms.FloatField(label='Min PPI evidence')
    p2p_w = forms.FloatField(label='PPI weight',initial=1)
    t2p_w = forms.FloatField(label='Co-morbidity to Prot. weight',initial=1)
    odd = forms.FloatField(label='Odds ratio cutoff (absolute log2)',initial=1)
    pv = forms.IntegerField(label='Min. q-value exponent co-morb. assocation with disease (e.g. 10^X)'
                            ,initial=-5
                           )
    only_positive_oddsr = forms.BooleanField(label='Use only positive odds ratio indications', initial=True, required=False)
    score_oddsr = forms.BooleanField(label='Score using odds ratio rather than pvalue', initial=True, required=False)
    use_opentargets = forms.BooleanField(label='Link via OpenTargets', initial=False, required=False)
    use_disgenet = forms.BooleanField(label='Link via DisGeNet', initial=True, required=False)

    max_indi_prots = forms.IntegerField(
        label="Max Indi Prots",
        help_text='Maximum # of prots per source to link for an indication (0 for no limit)',
        initial=1000,
        )
    downscale_by_prots = forms.BooleanField(
        label='Downscale By # Prots',
        help_text='Divides indi scores by (log # of assoc prots per source) to prevent well-studied indis from overwhelming others',
        initial=False,
        required=False,
        )
    

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
        # and the FAERS lists
        f = self.fields['faers_run']
        f.choices = self.ws.get_prev_job_choices('faers')
        if not f.choices:
            f.choices = [(0,'No relevant FAERS runs available in this workspace')]
        f.initial = f.choices[0][0]
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
        # PLAT-1716 changed faers_run from a path to a job id
        p['faers_run'] = extract_job_id(p)
        for f in self:
            if f.name in p:
                f.field.initial = p[f.name]

class MyJobInfo(JobInfo):
    def description_text(self):
        return '''
        <b>CAPP</b> combines co-morbidity data with gene-disease from
        DisGeNet, and then converts the resulting protein signatures to
        drug scores using the Pathsum algorithm.
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
    def build_role_code(self,jobname,settings_json):
        import json
        d = json.loads(settings_json)
        job_id = extract_job_id(d)
        parts = self._upstream_role_code(job_id).split('_')

        if d.get('use_opentargets', False):
            src = 'otarg'
        else:
            src = 'dgn'
        
        parts = parts[:-1] + [src, parts[-1]]
        return '_'.join(parts)

    def role_label(self):
        # this doesn't use _upstream_role_label because of the odd way
        # FAERS role labels are constructed
        job_id = extract_job_id(self.job.settings())
        src_bji = self.get_bound(self.ws,job_id)
        if self.parms.get('use_opentargets', False):
            middle = 'OTarg'
        else:
            middle = 'DGN'
        parts = [src_bji.role_label(),middle,self.short_label]
        return ' '.join(parts)
    def out_of_date_info(self,job,jcc):
        job_id = extract_job_id(job.settings())
        return self._out_of_date_from_ids(job,[job_id],jcc)
    def __init__(self,ws=None,job=None):
        # base class init
        self.use_LTS=True
        super(MyJobInfo,self).__init__(
                    ws,
                    job,
                    __file__,
                    "CAPP",
                    "Co-morbidity Associated Protein Pathsum",
                    )
        # any base class overrides for unbound instances go here
        self.publinks = (
                (None,'scatter_prots.plotly',),
                (None,'scatter_sumev.plotly',),
                (None,'barpos.plotly',),
                (None,'barpos_ex.plotly',),
                )
        self.qc_plot_files = (
                'scatter_prots.plotly',
                'scatter_sumev.plotly',
                'barpos.plotly',
                'barpos_ex.plotly',
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
            self.outfile = self.lts_abs_root+'capp.tsv'
            self.pathsum_detail = self.lts_abs_root+'path_detail.tsv'
            # Older jobs have this named as .tsv, newer have .tsv.gz.
            if not os.path.exists(self.pathsum_detail):
                self.pathsum_detail += '.gz'
            self.pathsum_detail_label = "Co-morbidity"
            self.barpos = self.tmp_pubdir+"barpos.plotly"
            self.barpos_ex = self.tmp_pubdir+"barpos_ex.plotly"
            # published output files

            url = self.faers_table_url()
            self.otherlinks = [
                ('FAERS CAPP Data Table', url),
            ]
    def faers_table_url(self):
        faers_jid = extract_job_id(self.job.settings())
        url = f"{self.ws.reverse('faers_run_table')}?capp_jid={self.job.id}&jid={faers_jid}"
        return url
    def get_data_code_groups(self):
        from math import log
        codes = [
            dc.Code('capds',label='Direct', fmt='%0.4f'),
            dc.Code('capis',label='Indirect', fmt='%0.4f'),
            dc.Code('cappkey',valtype='str',hidden=True),
            ]
        codetype = self.dpi_codegroup_type('p2d_file')
        return [
                dc.CodeGroup(codetype,self._std_fetcher('outfile'), *codes),
                ]
    def pathsum_scale(self):
        from algorithms.bulk_pathsum import extract_tissue_count_from_log
        return extract_tissue_count_from_log(self.job.id)
    def remove_workspace_scaling(self,code,ordering):
        if code == 'cappkey':
            return ordering
        s = self.pathsum_scale()
        return [(wsa,v/s) for wsa,v in ordering]
    def get_target_key(self,wsa):
        cat = self.get_data_catalog()
        try:
            val,_ = cat.get_cell('cappkey',wsa.id)
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
        removed = 0
        self.cmap_plot_data_cleaned = []
        for i in self.cmap_plot_data:
            if i[1] == 0:
                removed = removed+1
            else:
                self.cmap_plot_data_cleaned.append(i[:2])
        self.cmap_plot_data_cleaned_exclude = []
        removed_thres = 0
        for i in self.cmap_plot_data_exclude:
            if i[1] == 0:
                removed_thres = removed_thres+1
            else:
                self.cmap_plot_data_cleaned_exclude.append(i[:2])

        if removed > 0:
            plot(self.cmap_plot_data_cleaned,
                      self.barpos,
                      title='Number of Proteins Per Co-morbidity excluding %i co-morbidity(ies) of 0' % (removed),
                      ylabel='Co-morbidity',
                      xlabel='Proteins')
        else:
            plot(self.cmap_plot_data_cleaned,
                      self.barpos,
                      title='Number of Proteins Per Co-morbidity',
                      ylabel='Co-morbidity',
                      xlabel='Proteins')

        if removed > 0:
            plot(self.cmap_plot_data_cleaned_exclude,
                      self.barpos_ex,
                      title=' Number of Proteins Per Excluded Co-morbidity excluding %i co-morbidity(ies) of 0' % (removed),
                      ylabel='Co-morbidity',
                      xlabel='Proteins')
        else:
            plot(self.cmap_plot_data_cleaned_exclude,
                      self.barpos_ex,
                      title='Number of Proteins Per Excluded Co-morbidity',
                      ylabel='Co-morbidity',
                      xlabel='Proteins')
        
        for y_name, y_title, y_suffix in [['num_prots', '# of Proteins', 'prots'], ['sum_ev', 'Evidence Sum (ProtEv**2 * DisInpScore)', 'sumev']]:
            traces = []
            for src, src_stats in self.cmap_stats.items():
                x, y, names = [], [], []
                for indi, indi_stats in src_stats.items():
                    x.append(indi_stats['input_score'])
                    # We're doing a log plot, so 0's don't show up, but I still want to see them.
                    # Display them as 0.1's.
                    y.append(indi_stats[y_name] or 0.1)
                    names.append(indi)
                traces.append({
                    'name': src,
                    'x': x,
                    'y': y,
                    'text': names,
                    'textposition': 'top center',
                    'type': 'scattergl',
                    'mode': 'markers+text',
                })
            from dtk.plot import fig_legend
            anno = [fig_legend(['Shows which comorbidities contributed the most to CAPP scores.<br>'
                         'Look out for questionable comorbidities, or too few comorbidities.<br>'
                         'There are usually a few spurious correlations, though.'
                                         ],-0.13
                                         )]
            layout = {
                'width': 640,
                'height': 640,
                'hovermode': 'closest',
                'xaxis': {'title': 'Disease Input Score (log2(OR) or -log10(Q))'},
                'yaxis': {'title': y_title, 'type': 'log'},
                'title': f'Linked Indication Stats ({y_suffix})',
                'margin': {'b': 120},
                'annotations': anno,
            }
            from dtk.plot import PlotlyPlot
            PlotlyPlot(traces, layout).save(self.tmp_pubdir + f'scatter_{y_suffix}.plotly')


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
    
    def make_cmap_data(self):
        import numpy as np
        from runner.process_info import JobInfo
        faers_bji = JobInfo.get_bound(self.ws,self.parms['faers_run'])
        cat = faers_bji.get_data_catalog()
        from dtk.disgenet import DisGeNet
        dgn = DisGeNet(self.ws.get_versioned_file_defaults())
        thres_ordering = []
        thres_excluded = []
        cat_coior = {k: v for k, v in cat.get_ordering('coior',True)}
        
        if self.parms.get('only_positive_oddsr', False):
            maybe_abs = lambda x: x
        else:
            maybe_abs = lambda x: np.abs(x)

        for i in cat.get_ordering('coiqv',True):
            match = (i[0], cat_coior[i[0]])

            # Depending on settings we'll score with either pvalue (i) or odds ratio (match).
            if self.parms.get('score_oddsr', False):
                entry = (match[0], np.log2(match[1]))
            else:
                entry = (i[0], -np.log10(i[1]))

            if i[1] < 10**self.parms['pv'] and maybe_abs(np.log2(match[1])) > self.parms['odd']:
                print('include', entry)
                thres_ordering.append(entry)
            else:
                print('exclude', entry)
                thres_excluded.append(entry)

        logger.info("Grabbing disease association data") 
        cmap_data, stats = self.make_cmap_data_from_ordering(dgn, thres_ordering)
        logger.info("Grabbing excluded disease association data") 
        cmap_exclude, _ = self.make_cmap_data_from_ordering(dgn, thres_excluded)
        if not cmap_data:
            raise Exception('Unable to find any FAERS Co-morbidity data.')
        return cmap_data, cmap_exclude, stats

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
        capp_settings = self.job.settings()
        if capp_settings['combo_with']:
            d = self.ws.get_combo_therapy_data(capp_settings['combo_with'])
            from dtk.prot_map import DpiMapping
            dpi = DpiMapping(capp_settings['p2d_file'])
            from algorithms.bulk_pathsum import get_combo_fixed
            capp_settings['combo_fixed'] = get_combo_fixed(d,dpi)
        self.cmap_data, self.cmap_exclude, self.cmap_stats = self.make_cmap_data()
        self.cmap_plot_data = []
        self.cmap_plot_data_exclude = []
        for meddra_term, score in self.cmap_data.items():
            WorkItem.build_nontissue_file(
                self.indir,
                meddra_term,
                self.cmap_data[meddra_term]
                )
            num_prots = len(self.cmap_data[meddra_term])
            self.cmap_plot_data.append((meddra_term, num_prots, score))
        for meddra_term, score in self.cmap_exclude.items():
            num_prots = len(self.cmap_exclude[meddra_term])
            self.cmap_plot_data_exclude.append((meddra_term, num_prots, score))
        from dtk.data import merge_dicts,dict_subset
        context = merge_dicts(
                        {
                         'tissues':self.cmap_data,
                        },
                        dict_subset(
                                    capp_settings,
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
                                     'combo_fixed'
                                    ],
                                    )
                            )
        WorkItem.pickle(self.indir,'context',context)
        # generate dpi mapping file
        WorkItem.build_dpi_map(self.indir,
                               int(capp_settings['ws_id']),
                               capp_settings['p2d_file'],
                               )
        self._set_remote_cores_wanted()
    def make_cmap_data_from_ordering(self, dgn, ordering):
        def merge_prot_assoc(d, new_data):
            for indi, prot_mapping in new_data.items():
                for prot, score in prot_mapping.items():
                    d[indi][prot] = max(d[indi][prot], score)
        
        def collect_stats(d):
            out = {}
            for indi, score in ordering:
                from dtk.files import safe_name
                indi_prot_ev = d.get(safe_name(indi), {})
                out[indi] = {
                    'num_prots': len(indi_prot_ev),
                    'sum_ev': sum(indi_prot_ev.values()),
                    'input_score': score,
                }
            return out
        
        def filter_and_scale(d):
            max_prots = self.parms.get('max_indi_prots')
            downscale = self.parms.get('downscale_by_prots')
            for indi, indi_stats in list(d.items()):
                if max_prots and len(indi_stats) > max_prots:
                    prot_ord = sorted(indi_stats.items(), key=lambda x:-x[1])
                    indi_stats = dict(prot_ord[:max_prots])
                    d[indi] = indi_stats
                

                if downscale:
                    import numpy as np
                    scaling_factor = np.log(len(indi_stats) + 1)
                    for prot in indi_stats:
                        indi_stats[prot] /= scaling_factor

        from collections import defaultdict
        # indication -> prot -> score
        prot_assoc = defaultdict(lambda: defaultdict(float))
        from dtk.disgenet import score_cmap
        stats = {}
        if self.parms.get('use_opentargets', False):
            otarg_p = make_otarg_prot_assoc(self.ws, ordering, score_func=score_cmap)
            filter_and_scale(otarg_p)
            stats['otarg'] = collect_stats(otarg_p)
            merge_prot_assoc(prot_assoc, otarg_p)

        if self.parms.get('use_disgenet', True): 
            dgn_p = dgn.get_cmap_data(ordering)
            filter_and_scale(dgn_p)
            stats['dgn'] = collect_stats(dgn_p)
            merge_prot_assoc(prot_assoc, dgn_p)

        # Convert back from defaultdicts to dicts to aid in pickling.
        out = {disease: {prot: score for prot, score in dis_scores.items()}
                for disease, dis_scores in prot_assoc.items()}
        return out, stats


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
                    direct='capds',
                    indirect='capis',
                    )
            codetype = self.dpi_codegroup_type('p2d_file')
            f.write("\t".join([codetype] + [
                    score_map.get(name,name)
                    for name in self.score_types
                    ]+['cappkey']) + "\n")
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
        self.score_types = ['direct', 'indirect', 'direction']
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
        uji = self # simplify access inside MyWorkflowPart
        class MyWorkflowPart:
            def __init__(self,label,cds):
                self.label=label
                self.cds=cds
                # Note only the FAERS cds has a data status
                self.enabled_default=uji.data_status_ok(
                        ws,
                        'Faers',
                        'Complete Clinical Values',
                        ) if self.cds.startswith('faers.v') else False
            def add_to_workflow(self,wf):
                cm_info = wf.std_wf.get_main_cm_info(uji.job_type)
                faers_name = cm_info.pre.add_pre_steps(wf,self.cds)
                assert faers_name
                from dtk.workflow import CappStep
                my_name_dgn = faers_name+'_dgn_'+uji.job_type
                CappStep(wf,my_name_dgn,
                        inputs={faers_name:True},
                        source='dgn',
                        )
                cm_info.post.add_post_steps(wf,my_name_dgn)

                my_name_ot = faers_name+'_otarg_'+uji.job_type
                CappStep(wf,my_name_ot,
                        inputs={faers_name:True},
                        source='otarg',
                        )
                cm_info.post.add_post_steps(wf,my_name_ot)
        for choice in ws.get_cds_choices():
            parts.append(MyWorkflowPart(
                    choice[1]+' '+self.short_label,
                    choice[0],
                    ))


if __name__ == "__main__":
    MyJobInfo.execute(logger)
