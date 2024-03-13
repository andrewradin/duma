#!/usr/bin/env python3

import sys
from path_helper import PathHelper,make_directory

import os
import django
import django_setup

from django import forms

from tools import ProgressWriter
from runner.process_info import JobInfo
import runner.data_catalog as dc
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager

import json
import logging
logger = logging.getLogger("algorithms.run_sigdif")

from collections import defaultdict

class ConfigForm(forms.Form):
    input_score = forms.ChoiceField(label='Input score',initial=''
                        )
    ppi_file = forms.ChoiceField(label='PPI Dataset',initial=''
                        )
    min_ppi = forms.FloatField(label='Min PPI evidence',
                               initial=0.5
                               )
    restart_prob = forms.FloatField(label='ProtRank restart probability',
                                    initial=0.85
                                   )
    iterations = forms.IntegerField(label='Maximial number of iterations for ProtRank to converge',
                                    initial=100
                                   )
    sub_bg = forms.BooleanField(label='Subtract PPI background from scores', required=False, initial=True)
    trsig_sub = forms.IntegerField(
                            initial=None,
                            label = 'Treatment-Response signature (job id) to remove from this one',
                            required = False,
                            )
    _subtype_name = "job_subtype"
    def __init__(self, ws, sources, copy_job, *args, **kwargs):
        super(ConfigForm,self).__init__(*args, **kwargs)
        self.ws = ws
        # reload choices on each form load -- PPI
        from dtk.prot_map import PpiMapping
        f = self.fields['ppi_file']
        f.choices = [('','None')]+list(PpiMapping.choices())
        f.initial = ws.get_ppi_default()
        self.fields['min_ppi'].initial = ws.get_ppi_thresh_default()
        # do the same for input score
        f = self.fields['input_score']
        from dtk.scores import JobCodeSelector
        f.choices = JobCodeSelector.get_choices(sources,'uniprot','score')
        f.initial = f.choices[0][0]
        if copy_job:
            self.from_json(copy_job.settings_json)
    def as_html(self):
        from django.utils.html import format_html
        return format_html(
                u'''<div class="well">
                Selecting a PPI dataset will result in the signature data
                being diffused into the PPI net.
                </div><p>{}
                '''
                ,self.as_p()
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
        Diffuses out the signature. Uses a random walk with restart to
        blend (diffuse) the starting signature out into the protein-
        protein network. It's an algorithmic way of giving some score
        to neighboring proteins.
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
        job_id,code = src.split('_')
        return self._upstream_role_code(job_id,code)
    def role_label(self):
        return self._upstream_role_label()
    def get_input_job_ids(self):
        src = self.job.settings()['input_score']
        job_id,code = src.split('_')
        return set([int(job_id)])
    def out_of_date_info(self,job,jcc):
        src = job.settings()['input_score']
        job_id,code = src.split('_')
        return self._out_of_date_from_ids(job,set([job_id]),jcc)
    def __init__(self,ws=None,job=None):
        # base class init
        self.use_LTS=True
        super(MyJobInfo,self).__init__(
                    ws,
                    job,
                    __file__,
                    "SigDiffuser",
                    "Signature Diffuser",
                    )
        # any base class overrides for unbound instances go here
        self.publinks = (
                (None,'scatter.plotly'),
                (None,'histo.plotly'),
                )
        self.qc_plot_files = (
                'scatter.plotly',
                'histo.plotly',
                )
        # job-specific properties
        self.needs_sources = True
        self.sig_node = 'startingNode'
        if self.job:
            self.log_prefix = self.job.name+':'
            self.debug("setup")
            self.ec = ExitCoder()
            # output files
            self.signature_file = self.lts_abs_root+"signature.tsv"
            # published output files
            self.scatter = self.tmp_pubdir+"scatter.plotly"
            self.histo = self.tmp_pubdir+"histo.plotly"
    def is_a_source(self):
        return True
    def get_data_code_groups(self):
        import runner.data_catalog as dc
        return [
                dc.CodeGroup('uniprot',self._std_fetcher('signature_file'),
                        dc.Code('difEv',label='Diffused Evidence', fmt='%.2e'),
                        )
                ]
    def run(self):
        from collections import defaultdict
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        make_directory(self.lts_abs_root)
        p_wr = ProgressWriter(self.progress
                , [ "wait for resources"
                  , "setup"
                  , "diffusing into PPI network"
                  , "subtracting PPI network background"
                  , "plotting"
                  ]
                )
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.get_data()
        p_wr.put("setup","complete")
        self.diffuse()
        p_wr.put("diffusing into PPI network", self.diffuse_msg)
        self.bg()
        if self.parms['trsig_sub']:
            self.sub_base_combo()
        p_wr.put("subtracting PPI network background", self.bg_msg)
        self.save_signature()
        self.plot()
        p_wr.put("plotting","complete")
        self.finalize()
    def get_data(self):
        from dtk.scores import JobCodeSelector,check_and_fix_negs
        ip_code = self.parms['input_score']
        cat = JobCodeSelector.get_catalog(self.ws,ip_code)
        ord = check_and_fix_negs(cat.get_ordering(ip_code,True))
        self.tmp_sig = {tup[0]:tup[1] for tup in ord}
    def bg(self):
        if not self.parms['sub_bg']:
            self.bg_msg = 'N/A'
        else:
            bg_g = self.base_g.copy()
            ks = list(bg_g.nodes())
            for pn in ks:
                bg_g.add_edge(self.sig_node, pn, weight = 1.)
            from scripts.connect_drugs_to_proteinSets import run_page_rank
            protrank_dict = run_page_rank((self.sig_node,
                                       bg_g,
                                       bg_g,
                                       self.prot_prefix,
                                       self.parms['restart_prob'],
                                       self.parms['iterations'],
                                      ))
            for pn,v in protrank_dict.items():
                if pn == self.sig_node:
                    continue
                uni = pn.lstrip(self.prot_prefix)
                # TODO: This prevents background removal from creating negative scores
                # I'm pretty sure the NetworkX implimentation of page rank fails with
                # signed scores anyway, but leaving this just in case
                assert self.signature[uni] >= 0, "This doesn't work for signed scores"
                self.signature[uni] -= min(v, self.signature[uni])
            self.bg_msg = 'complete'

    def get_tr_from_job(self, jobid):
        from runner.models import Process
        from runner.process_info import JobInfo
        p = Process.objects.get(pk=jobid)
        ws = p.settings()['ws_id']
        bji = JobInfo.get_bound(ws, jobid)
        return bji.signature_file

    def sub_base_combo(self):
        print("Removing base combo")
        trsig_file = self.get_tr_from_job(self.parms['trsig_sub'])
        tr_sig = load_tr_signature(trsig_file)

        for uni, val in self.signature.items():
            ev = tr_sig.get(uni)
            if not ev:
                continue
            self.signature[uni] -= min(ev, self.signature[uni])





    def diffuse(self):
        self.signature = {}
        self._build_net()
        from scripts.connect_drugs_to_proteinSets import run_page_rank
        protrank_dict = run_page_rank((self.sig_node,
                                       self.g,
                                       self.g,
                                       self.prot_prefix,
                                       self.parms['restart_prob'],
                                       self.parms['iterations'],
                                      ))
        for pn,v in protrank_dict.items():
            if pn == self.sig_node:
                continue
            uni = pn.lstrip(self.prot_prefix)
            self.signature[uni] = v
        self.diffuse_msg = 'complete'
    def _build_net(self):
        from scripts.connect_drugs_to_proteinSets import establish_prefixes
        _, _, self.prot_prefix = establish_prefixes()
        from dtk.prot_map import PpiMapping
        from scripts.connect_drugs_to_proteinSets import build_ppi_graph
        ppi = PpiMapping(self.parms['ppi_file'])
        self.base_g = build_ppi_graph(ppi,
                                    prot_prefix = self.prot_prefix,
                                    direction = False,
                                    min_ppi_evid = self.parms['min_ppi']
                                   )
        self._add_sig_data()
    def _add_sig_data(self):
        missing_nodes = []
        self.g = self.base_g.copy()
        for p,v in self.tmp_sig.items():
            pn = self.prot_prefix + p
            if pn in self.g:
                self.g.add_edge(self.sig_node, pn, weight = v)
            else:
                missing_nodes.append(p)
        print(f'Missing a node for {len(missing_nodes)} entries in the provided score:')
        print(", ".join(missing_nodes))
# This can be helpful for debugging, but just clutters the log otherwise
#        print(f'Here are the graph nodes: {list(self.g)}')
        assert len(missing_nodes) != len(self.tmp_sig), "None of the nodes in the provided score were also in the graph"
    def save_signature(self):
        with open(self.signature_file, 'w') as f:
            header = ['uniprot', 'difEv']
            f.write("\t".join(header) + "\n")
            for uni in self.signature.keys():
                if self.signature[uni]:
                    line = "\t".join([str(i) for i in [uni, self.signature[uni]]])
                    f.write(line + "\n")
    def plot(self):
        from dtk.plot import scatter2d, Color, smart_hist, fig_legend
        from math import log
        from browse.models import Protein
        xys = []
        for k in self.signature:
            prev = self.tmp_sig[k] if k in self.tmp_sig else 0
            xys.append((prev, self.signature[k]))
        prot_qs = Protein.objects.filter(uniprot__in=list(self.signature.keys()))
        uni_2_gene = { x.uniprot:x.gene for x in prot_qs }
        names = []
        for id in self.signature:
            try:
                names.append(uni_2_gene[id])
            except KeyError:
                 names.append(id)
        pp = scatter2d(
                'Original evidence score',
                'Diffused evidence score',
                xys,
                title = 'Starting vs diffused scores',
                text = names,
                ids = ('protpage', list(self.signature.keys())),
                refline = False,
                classes=[
                        ('',{'color':Color.default, 'opacity':0.5})
                        ],
                class_idx = [0] * len(list(self.signature.values())), # filler
                bins = True
               )
        pp._layout['annotations'] = [fig_legend([
                                     'Comparing the original and the PPI-diffused score. '
                                     +'The expectation is that the scores will be correlated,'
                                     ,'though some proteins should increase in relative score.'
                                     ], -0.3)]
        pp._layout['margin'] = dict(b=120)
        pp.save(self.scatter)
        pp = smart_hist(list(self.signature.values()), layout=
                        {'title': 'Protein evidence histogram',
                         'yaxis':{'title':'Protein count',
                             },
                         'xaxis':{'title':'Diffused Sig evidence score'},
                        }
                       )
        pp._layout['annotations'] = [fig_legend([
                                     'A histogram of protein scores after'
                                     +'diffusing the original score into a PPI net.'
                                     ,'This is mostly to interpret the sparsity of data.'
                                     ], -0.3)]
        pp._layout['margin'] = dict(b=120)
        pp.save(self.histo)


def load_tr_signature(fn):
    sig = {}
    header = None
    from dtk.files import get_file_records
    for rec in get_file_records(fn):
        if header is None:
            header = rec
            continue
        uniprot, ev = rec
        sig[uniprot] = float(ev)
    return sig


if __name__ == "__main__":
    MyJobInfo.execute(logger)
