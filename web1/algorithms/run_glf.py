#!/usr/bin/env python3

from builtins import range
import sys
from path_helper import PathHelper,make_directory

import os
import django
import django_setup

from django import forms

from browse.models import TissueSet,Tissue
from tools import ProgressWriter
from runner.process_info import StdJobInfo, StdJobForm
import runner.data_catalog as dc
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager

import json
import logging
logger = logging.getLogger("algorithms.run_glf")


class MyJobInfo(StdJobInfo):
    descr = '''
        Provided a list of protein sets, GLF will identify which sets are
        most enriched in the provided gene expression data
        (preferably Sig output, but also can run on individual datasets
        from sig).

        This is most commonly used to convert a protein-based signature into a
        pathway-based signature.
        '''
    short_label = "GLF"
    page_label = "Gene List FEBE"
    upstream_jid = lambda cls, settings: settings['input_score'].split('_')

    def make_job_form(self, ws, data):
        from dtk.scores import JobCodeSelector, SourceList
        from algorithms.run_glee import get_tissueSets,get_gwds
        from browse.default_settings import GeneSets
        from scripts.glf import GLF_METHODS
        sources = self.sources
        defaults = list(JobCodeSelector.get_choices(sources,'uniprot','score'))
        tissue_opts = list(get_tissueSets(ws))
        gwds_opts = list(get_gwds(ws))
        source_choices = defaults + tissue_opts + gwds_opts
        method_choices = [(x, x) for x in GLF_METHODS]

        class ConfigForm(StdJobForm):
            input_score = forms.ChoiceField(
                                label='Input score',
                                choices=source_choices,
                                )
            std_gene_list_set = forms.ChoiceField(
                                label='Standard gene list set',
                                initial=GeneSets.value(ws),
                                choices=GeneSets.choices(ws),
                                )
            method = forms.ChoiceField(
                                label='Method',
                                choices=method_choices,
                                initial=method_choices[0][0],
                                )
            fake_mp = forms.BooleanField(label='Disable multiprocessing',required=False)
        return ConfigForm(ws, data)

    def __init__(self,ws=None,job=None):
        super().__init__(
                    ws=ws,
                    job=job,
                    src=__file__,
                    )
        self.needs_sources = True
        # job-specific properties
        self.publinks = [
                (None,'scatter.plotly'),
               ]
        self.qc_plot_files = (
                'scatter.plotly',
                )
        if self.job:
            # input files
            self.infile = os.path.join(self.indir, 'sig.pickle')
            self.comd_gmt = os.path.join(self.indir, 'comd.gmt')
            # output files
            self.tmp_outfile = os.path.join(self.outdir, 'glf.tsv')
            self.outfile = os.path.join(self.lts_abs_root, 'glf.tsv')
            # published output files
            try:
                self.publinks += [('Consensus GLF Results', 'glf.tsv.txt')]
                gen = (x for x in os.listdir(self.final_pubdir)
                        if x.endswith('_glf.tsv.txt')
                       )
                for x in gen:
                    self.publinks += [
                        (" ".join(x.split("_")[:-1] + ['GLF', 'Results']),x),
                        (None, x.rstrip('tsv.txt') + '.plotly')
                        ]
            except OSError:
                pass
            self.scatter = os.path.join(self.tmp_pubdir, 'scatter.plotly')

    def get_data_code_groups(self):
        return [
                dc.CodeGroup('uniprotset',self._std_fetcher('outfile'),
                        dc.Code('nProts'),
                        dc.Code('setPor'),
                        dc.Code('febeQ'),
                        dc.Code('febeOR'),
                        dc.Code('peakInd'),
                        dc.Code('wFEBE'),
                        )
                ]
    def run(self):
        self.make_std_dirs()
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
            # want at least a few jobs per process... let's say 500.
            remote_request = (5, 5 + self.gl_file_length // 500)
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
            outf.write('\t'.join(header)+'\n')
            for rec in src:
                if len(rec) == len(header) - 1:
                    rec.append('0.0')
                outf.write('\t'.join(rec)+'\n')
    def plot(self):
        scatter_plot(read_in_outdata(self.tmp_outfile),self.scatter)
    def plot_individ(self):
        # XXX This plotting and individual approach was done with GLEE
        # XXX but we haven't been using it hardly at all
        # XXX and as such have not enabled it with GLF yet
        return False
        gen = ( x
                for x in os.listdir(self.outdir)
                if x.endswith('_glf.tsv')
               )
        for x in gen:
            from algortihms.run_glee import heat_scatter_plot
            heat_scatter_plot(read_in_outdata(os.path.join(self.outdir, x))
                              , os.path.join(self.tmp_pubdir, x.rstrip('tsv') + 'plotly')
                              , title = x.rstrip('_glf.tsv')
                              )
    def setup(self):
        from math import ceil
        self.get_gmt()
        return self.get_gesig()
    def get_gmt(self):
        from dtk.s3_cache import S3Bucket, S3File
        from algorithms.run_glee import file_len
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
            from dtk.gene_sets import get_gene_set_file
            self.gl_file = get_gene_set_file(self.parms['std_gene_list_set']).path()
        self.gl_file_length = file_len(self.gl_file)
    def write_gmt(self):
        with open(self.comd_gmt, 'w') as f:
            for k,l in self.assigns.items():
                f.write("\t".join([str(k), ",".join(l)]) + "\n")
        self.gl_file = self.comd_gmt
    def run_remote(self):
        from pathlib import Path
        options = [Path(self.infile),
                    Path(self.gl_file),
                    Path(self.outdir),
                    Path(self.tmp_pubdir),
                    str(self.ws.id),
                    str(self.remote_cores_got),
                   ]

        options += ['--method', self.parms['method']]
        if self.multiple_input:
            options += ['--multiple_input']
        if self.parms['fake_mp']:
            options += ['--fake-mp']
        print(('remote command options',options))
        self.run_remote_cmd('scripts/glf.py', options)

        # this is just a temporary thing so we can see the results until we come up with something better
        try:
            gen = (f for f in os.listdir(self.outdir)
                     if os.path.isfile(os.path.join(self.outdir, f))
                       and f.endswith('glf.tsv')
                  )
            import subprocess
            for f in gen:
                pub_file = os.path.join(self.tmp_pubdir, f + '.txt')
                subprocess.check_call(['cp', os.path.join(self.outdir, f), pub_file])
        except OSError:
            pass
    def get_gesig(self):
        import pickle
        from dtk.scores import check_and_fix_negs
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
                        self.gesig_out["_".join([safe_name(t.name),str(t.id)])] = check_and_fix_negs(tges)
                term = "tissues"
            elif self.parms['input_score'].startswith('singleGWDS'):
                from dtk.gwas import scored_gwas,gwas_code
                for gwds in self.ws.get_gwas_dataset_qs():
                    ds_name = gwas_code(gwds.id)
                    n = safe_name(gwds.phenotype+'_'+ds_name)
                    self.gesig_out[n]=check_and_fix_negs(list(scored_gwas(ds_name).items()))
                term = "GWDS"
            if len(list(self.gesig_out.keys())) == 0:
                sys.stderr.write("WARNING: No significant data were found. Quitting.\n")
                sys.exit(self.ec.encode('unableToFindDataError'))
            self.multiple_input = True
            status = f'{len(self.gesig_out)} {term}'
        else:
            ip_code = self.parms['input_score']
            from dtk.scores import JobCodeSelector
            cat = JobCodeSelector.get_catalog(self.ws,ip_code)
            self.gesig_out = check_and_fix_negs(cat.get_ordering(ip_code,True))
            self.multiple_input = False
            status = str(len(self.gesig_out)) + " proteins"
        with open(self.infile, 'wb') as handle:
            pickle.dump(self.gesig_out, handle)
        return status
    def _check_and_fix_negs(self,lt):
        negs = sum(1 for x in lt if x[1] < 0.)
        if negs > 0:
            print(f'{negs} of {len(lt)} total values were negative. Absolute value is being applied.')
            return [(x[0],abs(x[1])) for x in lt]
        return lt
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

# TODO it's probably worth having a second plot that includes the OR, this one just includes the q-value
def scatter_plot(out_data,pp_name):
    from dtk.plot import scatter2d, fig_legend, Color
    from math import log
    names = []
    xys = []
    nProts = []
    portions = []
    maxScore = 0.
    for i in range(len(out_data['wFEBE'])):
        nProt = float(out_data['nProts'][i])
        nProts.append(nProt)
        setPor = float(out_data['setPor'][i])
        portions.append(setPor)
        q = float(out_data['febeQ'][i])
        OR = float(out_data['febeOR'][i])
        score = float(out_data['wFEBE'][i])
        names.append('<br>'.join([out_data['uniprotset'][i],
                                 '(score: %.2f)' % (score),
                                 '(q: %.2e)' % (q),
                                 '(OR: %.2f)' % (OR),
                                 '(nProts: %d)' % (nProt),
                                 '(setPor: %.2f)' % (setPor),
                                ]))
        maxScore = max([maxScore,score])
        xys.append((score,(-1.*log(q,10))))
    pp = scatter2d('wFEBE score (significance of enrichment',
            '-Log10(FDR)',
            xys,
            title = 'GLF pathway enrichment',
            text = names,
            refline = False,
            bins = True,
            class_idx = [0] * len(out_data['wFEBE']), # filler
            classes=[('',
                      {
                       'color':portions
                       , 'opacity' : 0.4
# +3 is just to ensure a resonably large dot
                       , 'size' : [log(n+3, 2)*1.8 for n in nProts]
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
    y_line = -1.*log(0.05,10)
    pp._layout['shapes']=[
                     {
                      'type': 'line',
                      'x0': 0.,
                      'y0': y_line,
                      'x1': maxScore*1.04,
                      'y1': y_line,
                      'line': {
                         'color': 'red',
                         'dash': 'dot',
                        },
                     }
                    ]
    pp._layout['margin']=dict(b=130)
    pp._layout['annotations'] = [fig_legend([
                                 'This is similar to a (sideways) volcano plot, where each dot '
                                 +'is a pathway. The Y-axis is significance of association,'
                                 ,'and the X-axis is our enrichment score which takes odds ratio '
                                 +'into account.'
                                 ,'The vertical dashed red line is an FDR '
                                 +'of 0.05. The size of the dot corresponds to the number of'
                                 ,'proteins in the pathway while the color corresponds to the '
                                 +'portion of proteins found in the score provided.'
                                ],-0.11)]

    pp.save(pp_name)

if __name__ == "__main__":
    MyJobInfo.execute(logger)
