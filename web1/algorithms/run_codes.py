#!/usr/bin/env python3

import sys
from path_helper import PathHelper,make_directory

import os
import django
import django_setup

from django import forms

from tools import ProgressWriter
from runner.process_info import StdJobInfo, StdJobForm
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager
import runner.data_catalog as dc
from dtk.prot_map import DpiMapping

import json
import logging
logger = logging.getLogger("algorithms.run_codes")

class MyJobInfo(StdJobInfo):
    descr = '''
        Originally written to work with drug-gene expression changes,
        this method uses output from GESig and DPI confidence scores to
        calculate drug efficacy scores.
        <i> CoDES includes a combo method.</i>
        '''
    short_label = 'CoDES'
    page_label = 'COrrecting Disease Expression Signatures'

    def make_job_form(self, ws, data):
        from dtk.prot_map import DpiMapping
        from dtk.scores import JobCodeSelector, SourceList
        sources = self.sources
        class ConfigForm(StdJobForm):
            combo_with = forms.ChoiceField(label='In combination with',initial=''
                                ,choices=(('','None'),)
                                ,required=False
                                )
            input_score = forms.ChoiceField(label='Input score',initial=''
                                ,choices=JobCodeSelector.get_choices(sources,'uniprot','score')
                                )
            p2d_file = forms.ChoiceField(
                label='DPI dataset',
                choices=DpiMapping.choices(ws),
                initial=ws.get_dpi_default(),
                )
            p2d_t = forms.FloatField(
                label='Min DPI evidence',
                initial=ws.get_dpi_thresh_default(),
            )
        return ConfigForm(ws, data)

    @classmethod
    def upstream_jid(cls,settings):
        try:
            src = settings['input_score']
        except KeyError:
            # legacy CODES job - gesig support only
            src = '%d_ev' % settings['job_id']
        job_id,code = src.split('_')
        return job_id, code
    def input_job_and_code(self):
        return self.upstream_jid(self.job.settings())
    def get_data_code_groups(self):
        codetype = self.dpi_codegroup_type('p2d_file')
        return [
                dc.CodeGroup(codetype,self._std_fetcher('outfile'),
                        dc.Code('negDir',
                                    label='neg. direction',
                                    fmt='%0.4f',
                                    hidden=True,
                                    ),
                        dc.Code('posDir',
                                    label='pos. direction',
                                    fmt='%0.4f',
# We flip the direction so that bigger is better
                                    calc=(lambda x:0-x,'negDir'),
                                    efficacy=False
                                    ),
                        dc.Code('absDir',
                                    label='Abs(direction)',
                                    fmt='%0.4f',
                                    calc=(lambda x:abs(x),'negDir'),
                                    ),
                        dc.Code('codesMax',
                                    label='max',
                                    fmt='%0.4f',
                                    ),
### The correaltions are really intended to work with Drug-Gene interaction data
### And currently we don't have that in a usable format
#                        dc.Code('posCor',
#                                    label='positive correlation',
#                                    fmt='%0.4f',
#                                    ),
#                        dc.Code('negCor',label='negative correlation',
#                                # lambda avoids operator.neg(0.0) => -0.0
#                                calc=(lambda x:0-x,'posCor'),
#                                fv=False,
#                                ),
                        ),
                ]
    def __init__(self,ws=None,job=None):
        super().__init__(ws=ws, job=job, src=__file__)
        # any base class overrides for unbound instances go here
        self.publinks = (
                (None,"hist.plotly"),
                )
        self.qc_plot_files = (
                'hist.plotly',
                )
        self.needs_sources = True
        # job-specific properties
        if self.job:
            self.base_drug_dgi = False
            self.remote_cores_wanted=1
            # input files
            self.infile = os.path.join(self.indir, 'in.pickle')
            self.dpimap_pickle = os.path.join(self.indir, 'dpimap.pickle')
            self.combod = os.path.join(self.indir, 'base_drug.pickle')
            # output files
            self.outfile = os.path.join(self.lts_abs_root, 'codes.tsv')
            self.hist = self.tmp_pubdir + "hist.plotly"
            # published output files
# this is just a temporary thing so we can see the results until we come up with something better
#            self.visible_results = self.tmp_pubdir + "codes.txt"
    def run(self):
        self.make_std_dirs()

        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "setup",
                "wait for remote resources",
                "comparing drug signatures to Sig",
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
        self.run_codes()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("comparing drug signatures to Sig", 'complete')
        self.move_output()
        self.check_enrichment()
        self.plot()
        self.finalize()
        p_wr.put("check enrichment","complete")

    def plot(self):
        from dtk.plot import smart_hist,fig_legend
        no_zero = [x for x in self.ordering_data if x]
        pp1 = smart_hist(no_zero, layout=
                        dict(
                           title = ('<b>Input protein scores</b><br>' +
                                    str(len(self.ordering_data) - len(no_zero)) +
                                    ' proteins not plotted, score = 0'
                                    )
                        )
               )
        pp1._layout['annotations'] = [fig_legend([
                                      'A histogram of the input scores. CoDES will work with '
                                      +'most score distributions, but was originally written',
                                      'with a 0 to 1 bounded normal distribution in mind. If '
                                      +'a distribution other than that is found more inspection',
                                      'may be warranted.'
                                      ],-0.1)]
        pp1.save(self.hist)
    def run_codes(self):
        local=False
        from pathlib import Path
        options = [Path(self.infile),
                    str(self.parms['p2d_t']),
                    Path(self.dpi.get_path()),
                    Path(self.dpimap_pickle),
                    Path(self.outdir),
                    Path(self.tmp_pubdir),
                    str(self.ws.id),
                   ]
        if self.combod:
            options += ['--base_drug', str(Path(self.combod))]
            if self.dpi.get_path().split(".")[-3] == 'broad' or self.base_drug_dgi:
                options += ['--bd_dgi']
        print(('command options',options))
        self.run_remote_cmd('scripts/codes.py', options, local=local)
    def move_output(self):
        import shutil
        shutil.move(self.outdir+'codes.tsv',self.outfile)
    def setup(self):
        import pickle
        from dtk.scores import JobCodeSelector
        ip_code = self.parms['input_score']
        cat = JobCodeSelector.get_catalog(self.ws,ip_code)
        ordering = cat.get_ordering(ip_code,True)
        self.ordering_data = [i[1] for i in ordering]
        with open(self.infile, 'wb') as handle:
            pickle.dump(ordering, handle)
        from dtk.prot_map import DpiMapping,stringize_value_lists
        self.dpi = DpiMapping(self.parms['p2d_file'])

        if self.dpi.is_combo():
            inputs = self.get_all_input_job_ids()
            from runner.process_info import JobInfo
            for jid in inputs:
                bji = JobInfo.get_bound(self.ws, jid)
                if bji.incorporates_combo_base():
                    logger.info(f"Found job {bji} which already accounts for combos, dropping DPI to non-combo")
                    self.dpi = self.dpi.get_noncombo_dpi()
                    break


        self.dpi_map = stringize_value_lists(self.dpi.get_wsa_id_map(self.ws))
        assert len(self.dpi_map) > 0
        with open(self.dpimap_pickle, 'wb') as handle:
            pickle.dump(self.dpi_map, handle)
        if self.parms['combo_with']:
            # single fixed drug combo case; get the fixed drug data
            combo_d = self.ws.get_combo_therapy_data(self.parms['combo_with'])
            try:
                path = os.path.join(str(self.ws.ws_dir()), 'combo', combo_d['GESig_file'])
                print ("Rather than using the DPI targets for the base drug,"
                       "we are using a GESig file."
                       )
                from dtk.readtext import parse_delim
                from numpy import sign
                with open(path, 'r') as f:
                    header = f.readline()
                    to_pickle = [(flds[0], abs(float(flds[1])), sign(float(flds[1])))
                                 for flds in parse_delim(f)
                                ]
                self.base_drug_dgi = True
            except KeyError:
                from browse.models import WsAnnotation
                wsa=WsAnnotation.objects.get(pk=combo_d['wsa'])
                l=self.dpi.get_dpi_info(wsa.agent)
                to_pickle = [(rec[1], rec.evidence, rec.direction) for rec in l]
            with open(self.combod, 'wb') as handle:
                pickle.dump(to_pickle, handle)
        else:
            self.combod = None

if __name__ == "__main__":
    MyJobInfo.execute(logger)
