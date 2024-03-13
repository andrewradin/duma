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

import logging
logger = logging.getLogger("algorithms.run_esga")

from django import forms

import json
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo
import runner.data_catalog as dc
from algorithms.exit_codes import ExitCoder
from browse.models import WsAnnotation

class ConfigForm(forms.Form):
    dpi_file = forms.ChoiceField(label='DPI dataset')
    min_dpi = forms.FloatField(label='Min DPI evidence')
    ppi_file = forms.ChoiceField(label='PPI Dataset')
    min_ppi = forms.FloatField(label='Min PPI evidence')
    restart_prob = forms.FloatField(label='ProtRank restart probability',
                                    initial=0.95
                                   )
    _subtype_name = "job_subtype"
    def __init__(self, ws, copy_job, *args, **kwargs):
        super(ConfigForm,self).__init__(*args, **kwargs)
        self.ws = ws
        self.input_count = 0
        # reload choices on each form load -- first DPI...
        f = self.fields['dpi_file']
        from dtk.prot_map import DpiMapping
        f.choices = DpiMapping.choices(ws)
        f.initial = ws.get_dpi_default()
        self.fields['min_dpi'].initial = ws.get_dpi_thresh_default()
        # ...then PPI
        f = self.fields['ppi_file']
        from dtk.prot_map import PpiMapping
        f.choices = list(PpiMapping.choices())
        f.initial = ws.get_ppi_default()
        self.fields['min_ppi'].initial = ws.get_ppi_thresh_default()
        # build individual threshold fields for each gwas dataset
        from dtk.gwas import gwas_code
        for choice in self.ws.get_gwas_dataset_choices():
            ds_name = gwas_code(choice[0])
            self.input_count += 1
            self.fields[ds_name] = forms.BooleanField(
                                label = choice[1],
                                initial = True,
                                required = False,
                                )
        if copy_job:
            self.from_json(copy_job.settings_json)
    def as_html(self):
        from django.utils.html import format_html
        return format_html(
                u'''<div class="well">There are currently {} GWAS datasets
                in this workspace.<p><p>
                You can add more datasets
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
        Given a number of phenotype datasets, gene scores are calculated
        and then diffused throught the PPI network (to compensate for the
        sparsity of the GWAS data). These protein scores are then converted
        to drug scores using a specified DPI network.
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
                    "ESGA",
                    "Exploring Subtle Gene Associations",
                    )
        # any base class overrides for unbound instances go here
        self.publinks = (
                )
        self.needs_sources = False
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.ec = ExitCoder()
            # input files
            self.g_pickle = os.path.join(self.indir, 'in.pickle')
            self.ref_nodes_pickle = os.path.join(self.indir, 'refnodes.pickle')
            # output files
            self.tmp_out_pickle = os.path.join(self.outdir, 'out.pickle')
            self.out_pickle = os.path.join(self.lts_abs_root, 'out.pickle')
            self.outfile = os.path.join(self.lts_abs_root, 'esga.tsv')
            # published output files
    def get_data_code_groups(self):
        codetype = self.dpi_codegroup_type('dpi_file')
        return [
                dc.CodeGroup(codetype,self._std_fetcher('outfile'),
                        dc.Code('prMax',label='pr_max', fmt='%0.4f'),
                        ),
                ]
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        make_directory(self.lts_abs_root)
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "setup",
                "wait for remote resources",
                "diffuse and score",
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
        self.diffuse()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("diffuse and score","complete")
        self.report()
        self.check_enrichment()
        self.finalize()
        p_wr.put("check enrichment","complete")
    def diffuse(self):
        options = [
                   self.mch.get_remote_path(self.ref_nodes_pickle),
                   self.mch.get_remote_path(self.g_pickle),
                   self.mch.get_remote_path(self.tmp_out_pickle),
                   str(self.parms['restart_prob']),
                   str(self.remote_cores_got),
                   '--aggregate', 'proteins'
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
        self._copy_out_pickle()
    def _copy_out_pickle(self):
        import shutil
        shutil.copyfile(self.tmp_out_pickle, self.out_pickle)
    def setup(self):
        self._get_gwas_data()
        self._build_net()
        self._save_g()
        self._save_gwas_nodes()
        self._set_remote_cores_wanted()
    def _save_gwas_nodes(self):
        import pickle
        with open(self.ref_nodes_pickle, 'wb') as handle:
            pickle.dump(list(self.connected_gwds), handle)
    def _set_remote_cores_wanted(self):
        self.remote_cores_wanted=(1,len(self.connected_gwds))
    def _save_g(self):
        import pickle
        with open(self.g_pickle, 'wb') as handle:
            pickle.dump(self.g, handle)
    def _get_gwas_data(self):
        from dtk.gwas import scored_gwas,selected_gwas
        self.gwas_data = {
                key:scored_gwas(key, v2d_threshold=1.0, v2g_threshold=0.0, max_prot_assocs=None)
                for key in selected_gwas(self.parms)
                }
        if not self.gwas_data:
            print('Unable to find any GWAS data. Quitting.')
            sys.exit(self.ec.encode('unableToFindDataError'))
    def _build_net(self):
        from scripts.connect_drugs_to_proteinSets import build_keyed_dpi_graph, establish_prefixes
        self.drug_prefix, _, self.prot_prefix = establish_prefixes()
        self.g = build_keyed_dpi_graph(self.parms['dpi_file'],
                                       min_dpi = self.parms['min_dpi']
                                      )
        self._add_ppi()
        self._add_gwas_data()
    def _add_gwas_data(self):
        self.connected_gwds = set()
        for gwds,d in self.gwas_data.items():
            for p,v in d.items():
                pn = self.prot_prefix + p
                if pn in self.g:
                    self.g.add_edge(gwds, pn, weight = v)
                    self.connected_gwds.add(gwds)
                else:
                    print('missing a node for ' + p)
    def _add_ppi(self):
        from dtk.prot_map import PpiMapping
        from scripts.connect_drugs_to_proteinSets import build_ppi_graph
        import networkx as nx
        ppi = PpiMapping(self.parms['ppi_file'])
        ppi_graph = build_ppi_graph(ppi,
                                    prot_prefix = self.prot_prefix,
                                    direction = True,
                                    min_ppi_evid = self.parms['min_ppi']
                                   )
        self.g = nx.compose(self.g,ppi_graph)
    def report(self):
        # this defines which data catalog code goes with which key
        # in the dicts in self.scores.values()
        self.score_name_map = [
                ('prMax','protrank_max'),
                ]
        self._load_scores()
        self._get_converter()
        with open(self.outfile, 'w') as f:
            codetype = self.dpi_codegroup_type('dpi_file')
            f.write("\t".join([codetype] + [
                    x for x,_ in self.score_name_map
                    ]) + "\n")
            for key,d in self.scores.items():
                drug_key = key.lstrip(self.drug_prefix)
                try:
                    wsas = self.conv[drug_key]
                except KeyError:
                    print('Unable to find WSA for', drug_key)
                    continue
                for w in wsas:
                    out = [str(w)]
                    for _,st in self.score_name_map:
                        out.append(str(d[st]))
                    f.write("\t".join(out) + "\n")
    def _load_scores(self):
        import pickle
        with open(self.tmp_out_pickle, 'rb') as handle:
            pr_d = pickle.load(handle)
        from scripts.connect_drugs_to_proteinSets import score_prot_rank
        self.scores = score_prot_rank([n for n in self.g
                                       if n.startswith(self.drug_prefix)
                                      ],
                                      self.g,
                                      self.prot_prefix,
                                      True,
                                      pr_d
                                     )
    def _get_converter(self):
        from dtk.prot_map import DpiMapping
        self.dpi = DpiMapping(self.parms['dpi_file'])
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
                from dtk.workflow import EsgaStep
                my_name='esga'
                EsgaStep(wf,my_name,
                        thresh_overrides={}
                        )
                cm_info.post.add_post_steps(wf,my_name)
        parts.append(MyWorkflowPart())

if __name__ == "__main__":
    MyJobInfo.execute(logger)
