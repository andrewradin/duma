#!/usr/bin/env python3

from __future__ import print_function
from builtins import range
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

from browse.models import Tissue
from tools import ProgressWriter
from runner.process_info import JobInfo
import runner.data_catalog as dc
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager

import json
import logging
logger = logging.getLogger("algorithms.run_tsp")

from collections import defaultdict
class ConfigForm(forms.Form):
    prot_set = forms.ChoiceField(label='ProtSet',initial=''
                        ,choices=(('','None'),)
                        )

    ppi_file = forms.ChoiceField(label='PPI dataset',initial=''
                        ,choices=(('','None'),)
                        ,required=False
                        )
    ppi_thresh = forms.FloatField(label='Min PPI evidence',initial=0.5)
    npermuts = forms.IntegerField(label='Background permutations for enrichment',initial=10000)
    _subtype_name = "job_subtype"
    def __init__(self, ws, ts_id, copy_job, *args, **kwargs):
        if copy_job:
            ts_id = copy_job.name.split('_')[1]
        elif ts_id is None:
            # if both copy_job and ts_id are None, there must be POST data
            ts_id=args[0][self._subtype_name]
        super(ConfigForm,self).__init__(*args, **kwargs)
        self.ts_id = ts_id
        self.ws = ws
        # get all of the protein sets
        f = self.fields['prot_set']
        f.choices = self.ws.get_uniprot_set_choices()
        if not f.choices:
            f.choices = (
                    ('','No protein sets available'),
                    )
        f.initial = f.choices[0][0]
        f.label = 'ProtSet'
        # ...then PPI
        f = self.fields['ppi_file']
        from dtk.prot_map import PpiMapping
        f.choices = [('','None')]+list(PpiMapping.choices())
        f.initial = f.choices[0][0]
        if copy_job:
            self.from_json(copy_job.settings_json)
        # build individual threshold fields for each tissue
        qs = Tissue.objects.filter(tissue_set_id=ts_id)
        for t in qs:
            _,_,_,total = t.sig_result_counts()
            if not total:
                continue
            self.fields['t_'+str(t.pk)] = forms.BooleanField(
                                initial=True,
                                label=t.name,
                                required=False
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
        p = {
            'ws_id':TissueSet.objects.get(pk=self.ts_id).ws_id,
            }
        for f in self:
            key = f.name
            value = src[key]
            if key.startswith('t_') and value > 1:
                continue # exclude from dictionary
            p[key] = value
        return p
    def from_json(self,init):
        p = json.loads(init)
        for f in self:
            if f.name in p:
                f.field.initial = p[f.name]

class MyJobInfo(JobInfo):
    def settings_defaults(self,ws):
        result = {}
        from browse.models import TissueSet
        for ts in TissueSet.objects.filter(ws=ws):
            form = ConfigForm(ws,ts.id,None)
            result[ts.name] = form.as_dict()
        return result
    def source_label(self,jobname):
        ts_id,ws_id = self._parse_jobname(jobname)
        from browse.models import TissueSet
        ts = TissueSet.objects.get(pk=ts_id)
        return ts.ts_label()+' TissueSet Plots'
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        if copy_job or not job_type:
            ts_id = None
        else:
            ts_id = job_type.split('_')[1]
        form = ConfigForm(ws,ts_id,copy_job)
        return form.as_html()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        form = ConfigForm(ws,None,None,post_data)
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
        try:
            return (int(fields[1]),int(fields[2]))
        except ValueError as e:
            from future.utils import raise_with_traceback
            raise_with_traceback(ValueError(e.message + ' jobname is %s' % jobname))
    def __init__(self,ws=None,job=None):
        # base class init
        super(MyJobInfo,self).__init__(
                    ws,
                    job,
                    __file__,
                    "TSP",
                    "TissueSet plots",
                    )
        # any base class overrides for unbound instances go here
        self.publinks = (
                (None,'scatter.plotly'),
                (None,'tisevid.plotly'),
                (None,'tisfc.plotly'),
                (None,'protevid.plotly'),
                (None,'protfc.plotly'),
                (None,'heatmap.plotly'),
                (None,'full_heatmap.plotly'),
                )
        # job-specific properties
        if self.job:
            self.log_prefix = self.job.name+':'
            self.debug("setup")
            self.ec = ExitCoder()
            # stash common ordered list of tissue ids here
            self.tissue_ids = [
                    int(x[2:])
                    for x in self.parms
                    if x.startswith('t_') and self.parms[x]
                    ]
            self.tissue_ids.sort()
            # output files
            # published output files
            self.prot_evid = self.tmp_pubdir+"protevid.plotly"
            self.prot_fc = self.tmp_pubdir+"protfc.plotly"
            self.tis_evid = self.tmp_pubdir+"tisevid.plotly"
            self.tis_fc = self.tmp_pubdir+"tisfc.plotly"
            self.heatmap_plotly = self.tmp_pubdir+"heatmap.plotly"
            self.heatmap_plotly_full = self.tmp_pubdir+"full_heatmap.plotly"
            self.scatter = self.tmp_pubdir+"scatter.plotly"
    def get_jobnames(self,ws):
        '''Return a list of all jobnames for this plugin in a workspace.'''
        return [
                "%s_%d_%d" % (self.job_type,ts.id,ws.id)
                for ts in ws.get_tissue_sets()
                ]
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        p_wr = ProgressWriter(self.progress
                , [ "wait for resources"
                  , "loading protset"
                  , "loading tissue set"
                  , "analyzing enrichment"
                  , "plotting"
                  ]
                )
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.get_protset()
        p_wr.put("loading protset","complete - " + str(len(self.ps_prots)) + " proteins found")
        self.get_tissue_data()
        p_wr.put("loading tissue set","complete - " + str(len(list(self.tge_dd.keys()))) + " tissues found")
        self.check_enrich()
        p_wr.put("analyzing enrichment","complete")
        self.plot()
        p_wr.put("plotting","complete")
        self.finalize()
    def check_enrich(self):
        from scripts.glee import run_single_glee
        self._setup_for_glee()
        self.glee_results = map(run_single_glee, self.params)
        self.process_results()
    def process_results(self):
        import operator
        from dtk.enrichment import mult_hyp_correct
        self.glee_results.sort(key=operator.itemgetter(1,2), reverse=True)
        qs = mult_hyp_correct([out_tup[4] for out_tup in self.glee_results])
        self.gles = []
        for i in range(len(self.glee_results)):
            out_tup = self.glee_results[i]
            self.gles.append([int(out_tup[0])] + list(out_tup[3]) + [out_tup[4], qs[i]])
    def _setup_for_glee(self):
        niters = len(list(self.tge_dd.keys()))
        self.params = zip( [[(k,abs(v[0])) for k,v in self.tge_dd[tid].items()]
                            for tid in self.tge_dd.keys()
                           ]
                          ,[ str(tid)
                             + "\t"
                             + ",".join(self.ps_prots)
                             for tid in self.tge_dd.keys()] 
                          , [1.0] * niters
                          , [self.parms['npermuts']] * niters
                          , [0.05] * niters,
                         )
    def get_protset(self):
        self.ps_prots = [str(x) for x in self.ws.get_uniprot_set(self.parms['prot_set'])]
        if self.parms['ppi_file']:
            from dtk.prot_map import PpiMapping
            self.ppi = PpiMapping(self.parms['ppi_file'])
            indirects = set()
            with open(self.ppi.get_path(), 'r') as f:
                header = f.readline()
                for l in f:
                    c = l.rstrip().split("\t")
                    if float(c[2]) < self.parms['ppi_thresh']:
                        continue
                    if c[0] in self.ps_prots and c[1] != "-":
                        indirects.add(c[1])
                    elif c[1] in self.ps_prots and c[0] != "-":
                        indirects.add(c[0])
            self.ps_prots = list(indirects | set(self.ps_prots))
    def get_tissue_data(self):
        self._get_tis_names()
        self.tge_dd = {}
        self.pge_dd = {}
        for tid in self.tissue_ids:
            self.tge_dd[tid] = {}
            errors = set()
            t=Tissue.objects.get(pk=tid)
            for rec in t.sig_results(over_only=False):
                out = [float(rec.evidence) * float(rec.direction)]
                try:
                    out += [float(rec.fold_change) * float(rec.direction)]
                except TypeError:
                    errors.add("Found no Fold Change for " + str(tid) + ". Defaulting to 0.0.")
                    out += [0.0]
                self.tge_dd[tid][rec.uniprot] = out
                try:
                    self.pge_dd[rec.uniprot][self.tis_names[tid]] = out
                except KeyError:
                    self.pge_dd[rec.uniprot]={}
                    self.pge_dd[rec.uniprot][self.tis_names[tid]] = out
            # setup the defaults for the prots in the PS
            self._check_tid_for_ps(tid)
            if len(errors) > 0:
                print("\n".join(list(errors)))
    def _check_tid_for_ps(self, tid):
        remove = True
        for prot in self.ps_prots:
            if prot in list(self.tge_dd[tid].keys()):
                remove = False
            else:
                self.tge_dd[tid][prot] = [0.0, 0.0]
        if remove:
            print(self.tis_names[tid], "reported none of the proteins in the protSet, so we are removing that tissue")
            self.tge_dd.pop(tid, None)
    def _get_tis_names(self):
        self.tis_names = {}
        for tid in self.tissue_ids:
            t = Tissue.objects.get(id=tid)
            self.tis_names[tid] = str(tid) + " - " + t.concise_name()
    def plot(self):
        self._plot_enrichments()
        self._plot_strips()
        self._plot_heatmap()
    def _plot_enrichments(self):
        from dtk.plot import scatter2d
        from math import log
        import numpy as np
        tids = []
        names = []
        nProts = []
        portions = []
        xys = []
        for l in self.gles:
            tids.append(l[0])
            names.append(self.tis_names[l[0]])
            xys.append((-1.0 * log(float(l[-2]), 10) * np.sign(float(l[3]))
                        ,float(l[4])
                       ))
            nProts.append(int(l[1]))
            portions.append(float(l[2]))
        pp = scatter2d("Directional -Log10(p)",
                'Normalized Enrichment Score',
                xys,
                title = 'Significance vs Enrichment of GLEE results',
                text = ['<br>'.join([names[i]
                                    , "Protein number: " + str(nProts[i])
                                    , "Total set portion: " + str(round(portions[i],3))
                                    ])
                        for i in range(len(portions))
                       ],
                ids = ('sigpage', tids),
                refline = False,
                class_idx = [0] * len(xys), # filler
                classes=[('Unknown',
                          {
                           'color': portions
                           , 'opacity' : 0.5
                           , 'size' : [log(float(n), 2) + 6 for n in nProts] # adding to ensure the points are big enough to see
                           , 'showscale' : True
                           , 'colorbar' : {
                               'title' : 'Portion of proteinSet',
                               'len' : 0.25,
                               'yanchor' : 'bottom',
                               'y' : 0.9,
                              },
                          }
                        )],
                width = 800,
                height = 800,
                )
        pp._layout['annotations'] = [{
                                       'xref' : 'paper'
                                       , 'yref' : 'paper'
                                       , 'showarrow' : False
                                       , 'y' : -0.1
                                       , 'yanchor' : 'top'
                                       , 'x' : 0.5
                                       , 'xanchor' : 'center'
                                       , 'text' : '<br>'.join([
                                                   'This plot shows the enrichment (y>0) or depletion (y<0)'
                                                  +' of the proteinSet in each dataset as a function of the'
                                                  ,'signifiance of that enrichment. Each dot is a dataset.'
                                                  +' The size is determined by the number of proteins, and'
                                                  ,' the color is the portion of the proteinSet that number'
                                                  +" accounts for. Clicking a dot will load that datasets' sigProt page"
                                                 ])
                                      }]
        pp._layout['margin']=dict(
                              l=60,
                              r=30,
                              b=120,
                              t=30,
                              pad=4
                              )
        pp.save(self.scatter)

    def _plot_heatmap(self):
        from browse.utils import prep_heatmap
        from dtk.plot import plotly_heatmap
        import numpy as np
        plot_names, cor_mat = prep_heatmap(self._get_heatmap_data())
        pp = plotly_heatmap(np.array(cor_mat).astype(np.float)
                            , plot_names
                            , Title = 'Dataset correlation over protein set'
                            , color_bar_title = "Pearson's Rho"
                            )
        pp._layout['annotations'] = [{
                                       'xref' : 'paper'
                                       , 'yref' : 'paper'
                                       , 'showarrow' : False
                                       , 'y' : 0.1
                                       , 'yanchor' : 'top'
                                       , 'x' : 0.5
                                       , 'xanchor' : 'center'
                                       , 'text' : '<br>'.join([
                                                   "Using just the proteins from the proteinSet,"
                                                  +" Pearson's Rho was calculated for all"
                                                  ,'dataset pairs using the directional Evidence '
                                                  +'score. Those correlation values were'
                                                  ,'then clustered and plotted as a heatmap.'
                                                 ])
                                      }]
        pp.save(self.heatmap_plotly)
        plot_names, cor_mat = prep_heatmap(self._get_heatmap_data(all=True))
        pp = plotly_heatmap(np.array(cor_mat).astype(np.float)
                            , plot_names
                            , Title = 'Dataset correlation over all proteins'
                            , color_bar_title = "Pearson's Rho"
                           )
        pp._layout['annotations'] = [{
                                       'xref' : 'paper'
                                       , 'yref' : 'paper'
                                       , 'showarrow' : False
                                       , 'y' : 0.1
                                       , 'yanchor' : 'top'
                                       , 'x' : 0.5
                                       , 'xanchor' : 'center'
                                       , 'text' : '<br>'.join([
                                                   'Nearly identical to the "Dataset correlation over protein set"'
                                                  +" heatmap, but all"
                                                  ,'proteins detected in at least 80% of the datasets were used.'
                                                 ])
                                      }]
        pp.save(self.heatmap_plotly_full)
    def _get_heatmap_data(self, all = False):
        if all:
            return {prot:{tname: self.pge_dd[prot][tname][0]
                      for tname in self.pge_dd[prot].keys()
                     }
                for prot in self.pge_dd.keys()
               }
        return {prot:{self.tis_names[tid]: self.tge_dd[tid][prot][0]
                      for tid in self.tge_dd.keys()
                     }
                for prot in self.ps_prots
               }
    def _plot_strips(self):
        from dtk.plot import stripplot
        self._get_strip_data('by_set', 0)
        tis_evid = stripplot('Datasets', 'Evidence score',
                self._strip_data,
                title = 'Protein Evidence by dataset',
                click = 'protpage',
                )
        tis_evid._layout['annotations'] = [{
                                       'xref' : 'paper'
                                       , 'yref' : 'paper'
                                       , 'showarrow' : False
                                       , 'y' : -0.15
                                       , 'yanchor' : 'top'
                                       , 'x' : 0.5
                                       , 'xanchor' : 'center'
                                       , 'text' : '<br>'.join([
                                                   'Each dataset is its own column here (i.e. the X-axis),'
                                                  +' and each dot is a protein from the proteinSet.'
                                                  ,'The Y-axis in this case is the Evidence Score for that'
                                                  +' protein in that respective dataset. Clicking'
                                                  ,'a dot opens the protein page for that protein.'
                                                 ])
                                      }]
        tis_evid._layout['margin']=dict(
                              l=60,
                              r=30,
                              b=110,
                              t=30,
                              pad=4
                              )
        tis_evid.save(self.tis_evid)
        self._get_strip_data('by_set', 1)
        tis_fc = stripplot('Datasets', 'Log 2 FC',
                self._strip_data,
                title = 'Protein Log2 FC by dataset',
                click = 'protpage',
                )
        tis_fc._layout['annotations'] = [{
                                       'xref' : 'paper'
                                       , 'yref' : 'paper'
                                       , 'showarrow' : False
                                       , 'y' : -0.15
                                       , 'yanchor' : 'top'
                                       , 'x' : 0.5
                                       , 'xanchor' : 'center'
                                       , 'text' : '<br>'.join([
                                                   'This is very similar to "Protein Evidence by dataset"'
                                                  +' except that the'
                                                  ,'Y-axis is Log2(fold change).'
                                                 ])
                                      }]
        tis_fc._layout['margin']=dict(
                              l=60,
                              r=30,
                              b=110,
                              t=30,
                              pad=4
                              )
        tis_fc.save(self.tis_fc)
        self._get_strip_data('by_prot', 0)
        prot_evid = stripplot('Proteins', 'Evidence Score',
                self._strip_data,
                title = 'Protein Evidence by protein',
                click = 'sigpage',
                )
        prot_evid._layout['annotations'] = [{
                                       'xref' : 'paper'
                                       , 'yref' : 'paper'
                                       , 'showarrow' : False
                                       , 'y' : -0.15
                                       , 'yanchor' : 'top'
                                       , 'x' : 0.5
                                       , 'xanchor' : 'center'
                                       , 'text' : '<br>'.join([
                                                   'This is very similar to "Protein Evidence by dataset"'
                                                  +' except that the X-axis is organized'
                                                  ,'by proteins in the proteinSet,'
                                                  +" and as a result clicking a dot opens a page about the dataset."
                                                 ])
                                      }]
        prot_evid._layout['margin']=dict(
                              l=60,
                              r=30,
                              b=110,
                              t=30,
                              pad=4
                              )
        prot_evid.save(self.prot_evid)
        self._get_strip_data('by_prot', 1)
        prot_fc = stripplot('Proteins', 'Log 2 FC',
                self._strip_data,
                title = 'Protein Log2 FC by protein',
                click = 'sigpage',
                )
        prot_fc._layout['annotations'] = [{
                                       'xref' : 'paper'
                                       , 'yref' : 'paper'
                                       , 'showarrow' : False
                                       , 'y' : -0.15
                                       , 'yanchor' : 'top'
                                       , 'x' : 0.5
                                       , 'xanchor' : 'center'
                                       , 'text' : '<br>'.join([
                                                   'This is very similar to "Protein Evidence by protein"'
                                                  +' except that the'
                                                  ,'Y-axis Log2(fold change).'
                                                 ])
                                      }]
        prot_fc._layout['margin']=dict(
                              l=60,
                              r=30,
                              b=110,
                              t=30,
                              pad=4
                              )
        prot_fc.save(self.prot_fc)

    def _get_gene_names(self):
        from browse.models import Protein
        prot_qs = Protein.objects.filter(uniprot__in=self.ps_prots)
        self.uni2gene = { x.uniprot:x.gene for x in prot_qs }
        gen = (p for p in self.ps_prots if p not in self.uni2gene)
        for p in gen:
            self.uni2gene[p] = p
    def _get_strip_data(self, type, ind):
        from statistics import mean
        self._get_gene_names()
        if type == 'by_set':
            unsorted = [
                        [
                            (
                                abs(self.tge_dd[tid][prot][ind]),
                                self.uni2gene[prot] + "<br>" + self.tis_names[tid],
                                prot,
                            )
                            for prot in self.ps_prots
                        ]
                    for tid in self.tge_dd.keys()
                    ]
        elif type == 'by_prot':
            unsorted = [
                          [
                            (
                             abs(self.tge_dd[tid][prot][ind]),
                             self.tis_names[tid] + "<br>" + self.uni2gene[prot],
                             tid,
                            )
                            for tid in self.tge_dd.keys()
                          ]
                      for prot in self.ps_prots
                    ]
        means = []
        for l in unsorted:
            means.append(mean([t[0] for t in l]))
        self._strip_data = [x
                            for (y,x) in sorted(zip(means,unsorted)
                                               , key=lambda pair: pair[0]
                                               , reverse=True)
                           ]

if __name__ == "__main__":
    MyJobInfo.execute(logger)
