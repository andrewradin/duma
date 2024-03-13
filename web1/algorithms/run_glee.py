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

from browse.models import TissueSet,Tissue
from tools import ProgressWriter
from runner.process_info import JobInfo
import runner.data_catalog as dc
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager

import json
import logging
logger = logging.getLogger("algorithms.run_glee")

class ConfigForm(forms.Form):
    input_score = forms.ChoiceField(label='Input score',initial=''
                        ,choices=(('','None'),)
                        )
    weight = forms.FloatField(label='Weight',initial=3.0)
    nPermuts = forms.IntegerField(label='Background permutations',initial=100000)
    std_gene_list_set = forms.ChoiceField(label='Standard gene list set',initial=''
                        ,choices=(('','None'),)
                        , required=False
                        )
    fake_mp = forms.BooleanField(label='Disable multiprocessing',required=False)

    _subtype_name = "job_subtype"
    def __init__(self, ws, sources, copy_job, *args, **kwargs):
        super(ConfigForm,self).__init__(*args, **kwargs)
        self.ws = ws
        # do the same for input score
        f = self.fields['input_score']
        from dtk.scores import JobCodeSelector
        f.choices = self._get_input_scores(sources)
        f.initial = f.choices[0][0]
        # ...then std_gene_list_set
        f = self.fields['std_gene_list_set']
        f.choices = get_gene_list_set_choices()
        f.label = 'Standard Gene List Set'
        try:
            idx = [x[1] for x in f.choices].index('annotated.pathways_reactome')
        except ValueError:
            idx = 0
        f.initial = f.choices[idx][0]
        if copy_job:
            self.from_json(copy_job.settings_json)
    def _get_input_scores(self,sources):
        from dtk.scores import JobCodeSelector
        defaults = list(JobCodeSelector.get_choices(sources,'uniprot','score'))
        tissue_opts = list(get_tissueSets(self.ws))
        gwds_opts = list(get_gwds(self.ws))
        return defaults + tissue_opts + gwds_opts
    def as_html(self):
        from django.utils.html import format_html
        return format_html('''<div class="well">{}<p><p>
                For explanations on which standard gene list set you might want to use,
                you can <a href="http://software.broadinstitute.org/gsea/msigdb/collections.jsp">
                see a brief description here</a>.\n Alternatively, you can use the ProteinSets in this
                workspace by selecting 'WS_protSets' at the bottom of the Standard Gene List Set
                </div><p>{}
                <table>{}</table>'''
                ,""
                ,""
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
        Provided a list of protein sets, GLEE will identify which sets are
        most enriched, or depleted, in the provided gene expression data
        (preferably GESig output, but also can run on individual datasets
        from sig).

        This is most commonly used to convert a protein-based signature into a
        pathway-based signature.
        '''
    def settings_defaults(self,ws):
        # construct default with an empty source list, so it includes
        # only non-source-specific settings
        from dtk.scores import SourceList
        sl=SourceList(ws)
        cfg=ConfigForm(ws,sl,None)
        return {
                'default':cfg.as_dict(),
                }
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        form = ConfigForm(ws, sources, copy_job)
        return form.as_html()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        form = ConfigForm(ws, sources, None,post_data)
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
        src = d['input_score']
        if src.startswith('single'):
            return ''
        job_id,code = src.split('_')
        return self._upstream_role_code(job_id,code)
    def role_label(self):
        return self._upstream_role_label()
    def get_input_job_ids(self):
        src = self.job.settings()['input_score']
        if src.startswith('single'):
            return set()
        job_id,code = src.split('_')
        return set([int(job_id)])
    def out_of_date_info(self,job,jcc):
        try:
            src = job.settings()['input_score']
            if src.startswith('single'):
                return super(MyJobInfo,self).out_of_date_info(job,jcc)
            job_id,code = src.split('_')
        except KeyError:
            job_id = job.settings()['job_id']
        return self._out_of_date_from_ids(job,[job_id],jcc)
    def __init__(self,ws=None,job=None):
        self.use_LTS = True
        # base class init
        super(MyJobInfo,self).__init__(
                    ws,
                    job,
                    __file__,
                    "GLEE",
                    "Gene List Enrichment Evaluation",
                    )
        # any base class overrides for unbound instances go here
        self.needs_sources = True
        # job-specific properties
        self.publinks = [
                (None,'bar.plotly'),
                (None,'scatter.plotly'),
               ]
        self.qc_plot_files = (
                'scatter.plotly',
                'bar.plotly',
                )
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.ec = ExitCoder()
            # TODO: calculate paths to individual files
            # input files
            self.infile = os.path.join(self.indir, 'gesig.pickle')
            self.comd_gmt = os.path.join(self.indir, 'comd.gmt')
            # output files
            self.tmp_outfile = os.path.join(self.outdir, 'glee.tsv')
            self.outfile = os.path.join(self.lts_abs_root, 'glee.tsv')
            # published output files
            try:
                self.publinks += [('Consensus GLEE Results', 'glee.tsv.txt')]
                gen = (x for x in os.listdir(self.final_pubdir)
                        if x.endswith('_glee.tsv.txt')
                       )
                for x in gen:
                    self.publinks += [
                        (" ".join(x.split("_")[:-1] + ['GLEE', 'Results']),x),
                        (None, x.rstrip('tsv.txt') + '.plotly')
                        ]
            except OSError:
                pass
            self.scatter = os.path.join(self.tmp_pubdir, 'scatter.plotly')
            self.bar = os.path.join(self.tmp_pubdir, 'bar.plotly')
    def get_data_code_groups(self):
        return [
                dc.CodeGroup('uniprotset',self._std_fetcher('outfile'),
                        dc.Code('nProts'),
                        dc.Code('setPor'),
                        dc.Code('ES'),
                        dc.Code('NES'),
                        dc.Code('NESlower'),
                        dc.Code('NESupper'),
                        dc.Code('pvalue'),
                        dc.Code('qvalue'),
                        )
                ]
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.lts_abs_root)
        make_directory(self.tmp_pubdir)
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "loading signature",
                "wait for remote resources",
                "checking for enrichment of lists",
                "plotting results",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        status = self.setup()
        p_wr.put("loading signature","Signature includes " + status)
        if self.parms['fake_mp']:
            remote_request = 1
        else:
            remote_request = (5, self.gl_file_length)
        got = self.rm.wait_for_resources(self.job.id,[
                                    0,
                                    remote_request,
                                    ])
        self.remote_cores_got = got[1]
        p_wr.put("wait for remote resources"
                        ,"complete; got %d cores for %d items"%(
                                self.remote_cores_got,
                                self.gl_file_length,
                                )
                        )
        self.run_remote()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("checking for enrichment of lists", 'complete')
        if not self.multiple_input:
            self.plot()
            self.save_scores()
        else:
            self.plot_individ()
        p_wr.put("plotting results", 'complete')
        self.finalize()
    def save_scores(self):
        with open(self.outfile,'w') as outf:
            from dtk.files import get_file_records
            src = get_file_records(self.tmp_outfile)
            header = next(src)
            rewrite = {
                    'List_name':'uniprotset',
                    'NES_lower':'NESlower',
                    'NES_upper':'NESupper',
                    'p-value':'pvalue',
                    'q-value':'qvalue',
                    }
            header = [
                    rewrite.get(label,label)
                    for label in header
                    ]
            outf.write('\t'.join(header)+'\n')
            for rec in src:
                if len(rec) == len(header) - 1:
                    # if there's no data at all mapping to a pathway, GLEE
                    # correctly outputs 0 scores, but fails to output a
                    # q-value; fix that here
                    rec.append('1.0')
                outf.write('\t'.join(rec)+'\n')
    def plot(self):
        heat_scatter_plot(read_in_outdata(self.tmp_outfile), self.scatter)
        bar_plot(read_in_outdata(self.tmp_outfile),self.bar)
    def plot_individ(self):
        gen = ( x
                for x in os.listdir(self.outdir)
                if x.endswith('_glee.tsv')
               )
        for x in gen:
            heat_scatter_plot(read_in_outdata(os.path.join(self.outdir, x))
                              , os.path.join(self.tmp_pubdir, x.rstrip('tsv') + 'plotly')
                              , title = x.rstrip('_glee.tsv')
                              )
    def setup(self):
        from math import ceil
        self.get_gmt()
        return self.get_gesig()
    def get_gmt(self):
        from dtk.s3_cache import S3Bucket, S3File
        if self.parms['std_gene_list_set'] == 'WS_protSets':
            # get all protein sets, read them into a list and write them out like above
            self.assigns = {}
            protsets = list(self.ws.get_uniprot_set_choices())
            if len(protsets) == 0 :
                sys.stderr.write("WARNING: There are no protein sets in this workspace to use. Quitting.\n")
                sys.exit(self.ec.encode('usageError'))
            for tup in protsets:
                ps = [str(x) for x in self.ws.get_uniprot_set(tup[0])]
                if len(ps) > 0:
                    self.assigns[tup[1]] = ps
            self.write_gmt()
        else:
            self.gl_file = os.path.join(PathHelper.glee, self.parms['std_gene_list_set'])
            if not os.path.isfile(self.gl_file):
                b = S3Bucket('glee')
                s3f=S3File(b,self.parms['std_gene_list_set'])
                s3f.fetch()
        self.gl_file_length = file_len(self.gl_file)
    def write_gmt(self):
        with open(self.comd_gmt, 'w') as f:
            for k,l in self.assigns.items():
                f.write("\t".join([str(k), ",".join(l)]) + "\n")
        self.gl_file = self.comd_gmt
    def run_remote(self):
        options = [self.mch.get_remote_path(self.infile),
                    self.mch.get_remote_path(self.gl_file),
                    self.mch.get_remote_path(self.outdir),
                    self.mch.get_remote_path(self.tmp_pubdir),
                    str(self.parms['weight']),
                    str(self.parms['nPermuts']),
                    str(self.ws.id),
                    '0.01',
                    str(self.remote_cores_got),
                   ]
        if self.multiple_input:
            options += ['--multiple_input']
        if self.parms['fake_mp']:
            options += ['--fake-mp']
        print(('remote command options',options))
        self.copy_input_to_remote()
        self.make_remote_directories([
                                self.tmp_pubdir,
                                ])
        rem_cmd = self.mch.get_remote_path(
                                    PathHelper.website_root+"scripts/glee.py"
                                    )
        self.mch.check_remote_cmd(' '.join([rem_cmd]+options))
        self.copy_output_from_remote()
# this is just a temporary thing so we can see the results until we come up with something better
        try:
            gen = (f for f in os.listdir(self.outdir)
                     if os.path.isfile(os.path.join(self.outdir, f))
                       and f.endswith('glee.tsv')
                  )
            import subprocess
            for f in gen:
                pub_file = os.path.join(self.tmp_pubdir, f + '.txt')
                subprocess.check_call(['cp', os.path.join(self.outdir, f), pub_file])
        except OSError:
            pass
    def get_gesig(self):
        import pickle
        if self.parms['input_score'].startswith('single'):
            from dtk.files import safe_name
            self.gesig_out = {}
            if self.parms['input_score'].startswith('singleTissues'):
                name_parts = self.parms['input_score'].split("_")
                assert len(name_parts) == 3
                assert name_parts[1] in ['evid', 'fc']
                qs = Tissue.objects.filter(tissue_set_id=name_parts[-1])
                for t in qs:
                    tges = []
                    for rec in t.sig_results(over_only=False):
                        if name_parts[1] == 'evid':
                            tges.append((rec.uniprot,
                                         float(rec.evidence) * float(rec.direction)
                                        ))
                        elif name_parts[1] == 'fc':
                            tges.append((rec.uniprot,
                                         float(rec.fold_change) * float(rec.direction)
                                       ))
                    if len(tges) > 0:
                        self.gesig_out["_".join([safe_name(t.name),str(t.id)])] = tges
                term = "tissues"
            elif self.parms['input_score'].startswith('singleGWDS'):
                from dtk.gwas import scored_gwas,gwas_code
                for gwds in self.ws.get_gwas_dataset_qs():
                    ds_name = gwas_code(gwds.id)
                    n = safe_name(gwds.phenotype+'_'+ds_name)
                    self.gesig_out[n]=list(scored_gwas(ds_name).items())
                term = "GWDS"
            if len(list(self.gesig_out.keys())) == 0:
                sys.stderr.write("WARNING: No significant data were found. Quitting.\n")
                sys.exit(self.ec.encode('unableToFindDataError'))
            self.multiple_input = True
            status = str(len(list(self.gesig_out.keys()))) + " " + term
        else:
            ip_code = self.parms['input_score']
            from dtk.scores import JobCodeSelector
            cat = JobCodeSelector.get_catalog(self.ws,ip_code)
            self.gesig_out = cat.get_ordering(ip_code,True)
            self.multiple_input = False
            status = str(len(self.gesig_out)) + " proteins"
        with open(self.infile, 'wb') as handle:
            pickle.dump(self.gesig_out, handle)
        return status
def read_in_outdata(ofile):
    header = None
    with open(ofile, 'r') as f:
        for l in f:
            fields = l.rstrip("\n").split("\t")
            if not header:
                out_data = {c:[] for c in fields}
                header = fields
                continue
            for i,x in enumerate(header):
                try:
                    out_data[x].append(fields[i])
                except IndexError:
                    if x == 'q-value':
                        out_data[x].append(1)
                    else:
                        print((ofile, 'seems to be missing data.'))
    return out_data

def bar_plot(out_data,pp_name):
    from dtk.plot import PlotlyPlot, fig_legend
    combined = []
    min_por = 0.9
    for i in range(len(out_data['NES'])):
        name = out_data['List_name'][i] + ' (%s)' % (str(out_data['nProts'][i]))
        setPor = float(out_data['setPor'][i])
        if setPor < min_por:
            combined.append((name, setPor))
    combined_sorted = sorted(combined, key=lambda x: x[-1])[::-1]
    names = []
    setPors = []
    name_lens = []
    for i,j in combined_sorted:
        names.append(i)
        setPors.append(j)
        name_lens.append(len(i))
    max_name = max(name_lens) if name_lens else 1
    bar = dict(y = names,
           x = setPors,
           orientation='h',
           type='bar')
    margin = max_name * 8
    pp_bar = PlotlyPlot([bar],
               {
                'title':'Pathways with less than 90% of proteins',
                'width':600 + margin,
                'height':270+10*len(name_lens),
                'margin':dict(
                             l = 650,
                             b = 150
                             ),
               }
        )
    if name_lens:
        pp_bar._layout['annotations'] = [fig_legend([
                                 'This plot is intended to show the pathways which had few proteins '
                                 +'present in the provided score. The X-axis is '
                                 ,'the portion of proteins in that path which were present, the total  '
                                 +'number of proteins are also noted in the pathway'
                                 ,'name on the Y-axis. Pathways with few proteins present are '
                                 +'unlikely to have statistically supported association'
                                 ,'with the disease. That means <b>if many pathways'
                                 +'are this plot, think twice about using this GLEE run</b>.'
                                ],-2.4/len(name_lens))]
    pp_bar.save(pp_name)

def heat_scatter_plot(out_data, out_name, title = 'Significance vs Enrichment of GLEE results'):
    from dtk.plot import scatter2d, fig_legend
    from math import log
    nes = []
    xys = []
    nProts = []
    portions = []
    for i in range(len(out_data['NES'])):
        portions.append(float(out_data['setPor'][i]))
        nProts.append(out_data['nProts'][i])
        nes.append(float(out_data['NES'][i]))
        xys.append((-1.0 * log(float(out_data['q-value'][i]), 10),
                    nes[-1]
                   )
                  )
    pp = scatter2d('-Log10(FDR)',
            'Normalized Enrichment Score',
            xys,
            title = title,
            text = ['<br>'.join([out_data['List_name'][i]
                                , "Protein number: " + str(nProts[i])
                                , "Total set portion: " + str(round(portions[i],3))
                                ])
                    for i in range(len(portions))
                   ],
            refline = False,
            bins = True,
            class_idx = [0] * len(out_data['NES']), # filler
            classes=[('Unknown',
                      {
                       'color':portions
                       , 'opacity' : 0.4
                       , 'size' : [log(float(n)+8.0, 2) for n in nProts] # adding the 8.0 to ensure the points are big enough to see
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
    pp._layout['shapes']=[
                     {
                      'type': 'line',
                      'x0': -1.0 * log(0.05,10),
                      'y0': min(nes),
                      'x1': -1.0 * log(0.05,10),
                      'y1': max(nes),
                      'line': {
                         'color': 'red',
                         'dash': 'dot',
                        },
                     }
                    ]
    pp._layout['margin']=dict(b=130)
    pp._layout['annotations'] = [fig_legend([
                                 'This is a (sideways) volcano plot, where each dot '
                                 +'is a pathway. The X-axis is significance of association,'
                                 ,'and the Y-axis is basically the directional enrichment '
                                 +'(meaning that negative values are depleted at the top of'
                                 ,'of the score used. The vertical dashed red line is an FDR '
                                 +'of 0.05. The size of the dot corresponds to the number of'
                                 ,'proteins in the pathway while the color corresponds to the '
                                 +'portion of proteins found in the score provided.'
                                ],-0.11)]
    pp.save(out_name)
def file_len(fname):
    i = -1
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
def get_tissueSets(ws):
    tss = TissueSet.objects.filter(ws=ws)
    to_return = []
    for ts in tss:
        label = ts.ts_label()
        to_return.append(('singleTissues_evid_' + str(ts.id),
                          'Evidence for each tissue in ' + label))
        to_return.append(('singleTissues_fc_' + str(ts.id),
                          'Fold change for each tissue in ' + label))
    return to_return
def get_gwds(ws):
    return [('singleGWDS_pval_score', 'Scores for each GWAS dataset')]

def get_gmt_choices():
    all_gmt_files = get_gmt_files()
    # what we use, what we show
    return sorted([(f, ".".join(f.split(".")[0:2])) for f in all_gmt_files], key=lambda x: x[0])

def get_gmt_files():
    all_gmt_files=[]
    from dtk.s3_cache import S3Bucket, S3File
    b = S3Bucket('glee')
    for file in b.list():
        if file.endswith(".gmt"):
            s3f=S3File(b,file)
            s3f.fetch()
            all_gmt_files.append(file)
    if len(all_gmt_files) == 0:
        sys.stderr.write("Unable to find gmt files.\n")
        sys.exit(ExitCoder('unableToFindDataError'))
    return all_gmt_files

def get_gene_list_set_choices():
    standard_choices = get_gmt_choices()
    return [('', 'None')] + standard_choices + [('WS_protSets', 'WS_protSets')]

if __name__ == "__main__":
    MyJobInfo.execute(logger)
