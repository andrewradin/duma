#!/usr/bin/env python3

from __future__ import print_function
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

from browse.models import WsAnnotation
from tools import ProgressWriter
from runner.process_info import JobInfo
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager
import runner.data_catalog as dc

import json
import logging
logger = logging.getLogger("algorithms.run_prsim")


class ConfigForm(forms.Form):
    drugset = forms.ChoiceField(
                label = 'DrugSet for comparison',
                initial='',
                choices=(('','None'),)
               )
    dpi_file = forms.ChoiceField(label='DPI dataset')
    dpi_t = forms.FloatField(label='Min DPI evidence')
    ppi_file = forms.ChoiceField(label='PPI dataset',required=False)
    ppi_t = forms.FloatField(label='Min PPI evidence')
    restart_prob = forms.FloatField(label='ProtRank restart probability',initial=0.85)
    _subtype_name = "job_subtype"
    def __init__(self, ws, sources, copy_job, *args, **kwargs):
        super(ConfigForm,self).__init__(*args, **kwargs)
        self.ws = ws
        # reload choices on each form load -- first DPI...
        f = self.fields['dpi_file']
        from dtk.prot_map import DpiMapping
        f.choices = DpiMapping.choices(ws)
        f.initial = ws.get_dpi_default()
        self.fields['dpi_t'].initial = ws.get_dpi_thresh_default()
        # ...then PPI
        f = self.fields['ppi_file']
        from dtk.prot_map import PpiMapping
        f.choices = [('','None')]+list(PpiMapping.choices())
        f.initial = ws.get_ppi_default()
        self.fields['ppi_t'].initial = ws.get_ppi_thresh_default()
        # ...then drugset
        f = self.fields['drugset']
        f.choices = self.ws.get_wsa_id_set_choices()
        f.initial = f.choices[0][0]
        if copy_job:
            self.from_json(copy_job.settings_json)
    def as_html(self):
        from django.utils.html import format_html
        return format_html('''
                <input name="{}" type="hidden" value="{}"/>
                <table>{}</table>
                '''
                ,self._subtype_name
                ,None
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
        p ={}
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
        Like SigDiffuser, uses the same "diffuse through the network" 
        concept to figure out how similar two proteins are. Algorithm 
        inspired by PageRank.
        '''
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        form = ConfigForm(ws, sources, copy_job)
        return form.as_html()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        form = ConfigForm(ws, sources, None,post_data)
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
    def get_data_code_groups(self):
        return [
                dc.CodeGroup('wsa',self._std_fetcher('outfile'),
                        dc.Code('prSimMax',
                                    label='PRSim_max',
                                    fmt='%0.4f',
                                    ),
                        #dc.Code('prSimMed',
                        #            label='PRSim_median',
                        #            fmt='%0.4f',
                        #            ),
                        dc.Code('prSimMean',
                                    label='PRSim_mean',
                                    fmt='%0.4f',
                                    ),
                        ),
                ]
    def __init__(self,ws=None,job=None):
        # base class init
        self.use_LTS=True
        super(MyJobInfo,self).__init__(
                    ws,
                    job,
                    __file__,
                    "PRSim",
                    "Protein Rank Similarity",
                    )
        # any base class overrides for unbound instances go here
        self.publinks = (
                )
        self.needs_sources = False
        # job-specific properties
        if self.job:
            self.base_drug_dgi = False
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.ec = ExitCoder()
            self.remote_cores_wanted=1
           # input files
            self.g_pickle = os.path.join(self.indir, 'in.pickle')
            self.ref_nodes_pickle = os.path.join(self.indir, 'refnodes.pickle')
            # output files
            self.out_pickle = os.path.join(self.outdir, 'out.pickle')
            # published output files
            self.outfile = os.path.join(self.lts_abs_root, 'prsim.tsv')
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        make_directory(self.lts_abs_root)
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "setup",
                "wait for remote resources",
                "calculating distances",
                "check enrichment",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.setup()
        p_wr.put("setup","complete")
        got = self.rm.wait_for_resources(self.job.id,[
                                    0,
                                    self.remote_cores_wanted,
                                    ])
        self.remote_cores_got = got[1]
        p_wr.put("wait for remote resources"
                        ,"complete; got %d cores"%self.remote_cores_got
                        )
        self.run_protRankSim()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("calculating distances", 'complete')
        self.write_results()
        self.finalize()
        p_wr.put("check enrichment","complete")
    def write_results(self):
        self.aggregate_results()
        conv = self.dpi.get_wsa_id_map(self.ws)
        with open(self.outfile, 'w') as f:
            f.write("\t".join(['wsa']
                + [x[1] for x in self.score_types]
                ) + "\n")
            for key,d in self.scores.items():
                drug_key = key.lstrip(self.drug_prefix)
                try:
                    wsas = conv[drug_key]
                except KeyError:
                    print('Unable to find WSA for', drug_key)
                    continue
                for w in wsas:
                    out = [str(w)]
                    for st,_ in self.score_types:
                        out.append(str(d[st]))
                    f.write("\t".join(out) + "\n")
    def aggregate_results(self):
        self.score_types = [
                ('protrank_max','prSimMax'),
                ('protrank_median','prSimMed'),
                ('protrank_mean','prSimMean'),
                ]
        import pickle
        with open(self.out_pickle, 'rb') as handle:
            scores = pickle.load(handle)
        self.scores = None
        for sub_dict in scores.values():
            if self.scores is None:
                self.scores = sub_dict.copy()
                continue
            for k in sub_dict:
                for s,_ in self.score_types:
                    self.scores[k][s] += sub_dict[k][s]
    def run_protRankSim(self):
        options = [
                   self.mch.get_remote_path(self.ref_nodes_pickle),
                   self.mch.get_remote_path(self.g_pickle),
                   self.mch.get_remote_path(self.out_pickle),
                   str(self.parms['restart_prob']),
                   str(self.remote_cores_got)
                  ]
        print(('command options',options))
        self.copy_input_to_remote()
        self.make_remote_directories([
                                self.tmp_pubdir,
                                ])
        rem_cmd = self.mch.get_remote_path(
                                    os.path.join(PathHelper.website_root,
                                                 "scripts",
                                                 "pr_wrapper.py"
                                                )
                                    )
        self.mch.check_remote_cmd(' '.join([rem_cmd]+options))
        self.copy_output_from_remote()
    def setup(self):
        from scripts.connect_drugs_to_proteinSets import build_keyed_dpi_graph, establish_prefixes
        self.drug_prefix, _, self.prot_prefix = establish_prefixes()
        self.g = build_keyed_dpi_graph(self.parms['dpi_file'])
        self._add_ppi()
        self._save_g()
        self._load_drugs()
        self._save_drugs()
        self._set_remote_cores_wanted()
    def _save_g(self):
        import pickle
        with open(self.g_pickle, 'wb') as handle:
            pickle.dump(self.g, handle)
    def _load_drugs(self):
        from dtk.prot_map import DpiMapping
        self.dpi = DpiMapping(self.parms['dpi_file'])
        wsa_ids = self.ws.get_wsa_id_set(self.parms['drugset'])
        self.basedrugs_ws = list(WsAnnotation.objects.filter(ws_id = self.ws.id,
                                                             pk__in = wsa_ids)
                                 )
        self.basedrugs_keys = load_dpi_drug_nodes(self.basedrugs_ws,
                                                  self.dpi,
                                                  self.drug_prefix,
                                                  self.g.nodes()
                                                 )
    def _save_drugs(self):
        import pickle
        with open(self.ref_nodes_pickle, 'wb') as handle:
            pickle.dump(self.basedrugs_keys, handle)
    def _set_remote_cores_wanted(self):
        self.remote_cores_wanted=(1,len(self.basedrugs_keys))
    def _add_ppi(self):
        from dtk.prot_map import PpiMapping
        from scripts.connect_drugs_to_proteinSets import build_ppi_graph
        import networkx as nx
        ppi = PpiMapping(self.parms['ppi_file'])
        self.ppi_graph = build_ppi_graph(ppi,
                                         prot_prefix = self.prot_prefix,
                                         direction = False,
                                         min_ppi_evid = self.parms['ppi_t']
                                        )
        self.g = nx.compose(self.g,self.ppi_graph)

def load_dpi_drug_nodes(basedrugs_ws, dpi, drug_prefix, nodes):
    basedrugs_keys = []
    for x in basedrugs_ws:
        k = dpi.get_dpi_keys(x.agent)
        if len(k) != 1:
            print("WARNING: Unable to find only one DPI key for this WSA:", x, k)
            k_to_use = None
            best_cnt = 0
            for x in k:
                cnt = len(dpi.get_dpi_info_for_keys(x))
                if cnt > best_cnt:
                    k_to_use = x
                    best_cnt = cnt
            print("We are using the key with the most targets:", k_to_use)
            k[0] = k_to_use
        node = drug_prefix + k[0]
        if node not in nodes:
            print("WARNING:", k[0], 'Not found in the DPI graph')
            print("It will be skipped")
            continue
        basedrugs_keys.append(node)
    return basedrugs_keys

if __name__ == "__main__":
    MyJobInfo.execute(logger)
