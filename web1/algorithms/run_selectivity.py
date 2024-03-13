#!/usr/bin/env python3

import sys
import six
from path_helper import PathHelper,make_directory

import os
import django_setup
from django import forms

from tools import ProgressWriter
from runner.process_info import StdJobInfo, StdJobForm
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager
import runner.data_catalog as dc
from dtk.prot_map import DpiMapping, PpiMapping

import json
import logging
logger = logging.getLogger(__name__)


class MyJobInfo(StdJobInfo):
    short_label = 'Selectivity'
    page_label = 'Selectivity'
    descr = 'Computes a score indicating how selective a drug is.'

    def make_job_form(self, ws, data):
        class MyForm(StdJobForm):
            dpi_file = forms.ChoiceField(
                    label='DPI dataset',
                    choices=DpiMapping.choices(ws),
                    initial=ws.get_dpi_default(),
                    )
            ppi_file = forms.ChoiceField(
                    label='PPI Dataset',
                    choices=PpiMapping.choices(),
                    initial=ws.get_ppi_default(),
                    )
            ppi_threshold = forms.FloatField(
                    label='Min PPI evidence',
                    initial=ws.get_ppi_thresh_default()
                    )
            dpi_sum_sig_center = forms.FloatField(
                    label='DPI ev sum sigmoid center (score of 0.5)',
                    initial=10
                    )
            dpi_sum_sig_width = forms.FloatField(
                    label='DPI ev sum sigmoid width (0.25 to 0.75)',
                    initial=10
                    )
            ppi_sum_sig_center = forms.FloatField(
                    label='PPI ev sum sigmoid center (score of 0.5)',
                    initial=200
                    )
            ppi_sum_sig_width = forms.FloatField(
                    label='PPI ev sum sigmoid width (0.25 to 0.75)',
                    initial=200
                    )

        return MyForm(ws, data)

    def get_data_code_groups(self):
        return [
                dc.CodeGroup('wsa',self._std_fetcher('outfile'),
                        dc.Code('dpiSum',label='Sum of DPI weights',efficacy=False),
                        dc.Code('ppiSum',label='Sum of indirect weights',efficacy=False),
                        dc.Code('dpiSel',label='DPI Sel Classifier'),
                        dc.Code('ppiSel',label='PPI Sel Classifier'),
                        )
                ]

    def __init__(self,ws=None,job=None):
        super().__init__(ws=ws, job=job, src=__file__)
        if self.job:
            self.outfile = os.path.join(self.lts_abs_root, 'selectivity.tsv')

    def run(self):
        self.make_std_dirs()
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "setup",
                "compute scores",
                "finalize",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.setup()
        p_wr.put("setup","complete")
        self.compute_scores()
        p_wr.put("compute scores","complete")
        self.finalize()
        p_wr.put("finalize","complete")

    def get_wsa2keys(self, dpikeys, dpi):
        from dtk.data import MultiMap
        key2wsa_mm = MultiMap(dpi.get_key_wsa_pairs(ws=self.ws, keyset=dpikeys))
        return key2wsa_mm.rev_map()

    def _compute_dpi_sum(self, wsa):
        from collections import defaultdict
        seen = defaultdict(float)
        for key in self.wsa2keys[wsa]:
            protevs = self.d2pe.get(key, [])
            for prot, ev in protevs:
                ev = float(ev)
                seen[prot] = max(seen[prot], ev)
        evsum = sum(seen.values())
        return evsum

    def _compute_ppi_sum(self, wsa):
        from collections import defaultdict
        seen = defaultdict(float)
        for key in self.wsa2keys[wsa]:
            protevs = self.d2pe.get(key, [])
            for prot, ev in protevs:
                ev = float(ev)
                seen[prot] = max(seen[prot], ev)
        seen_direct = seen.copy()
        for prot, ev1 in six.iteritems(seen_direct):
            for iprot, ev2 in self.p2pe.get(prot, set()):
                if ev2 > self.parms['ppi_threshold']:
                    seen[iprot] = max(seen[iprot], ev1 * ev2)
        evsum = sum(seen.values())
        return evsum

    def _classify_metric(self, x, center, width):
        """
        Generates a score that can be used to classify
        The intuition here is that there's probably not much difference
        between 100 or 200 targets, and similarly between 2 or 3 - but there's
        an inflection point in the middle where we transition from good to bad.
        We use a sigmoid to accomplish this.  Then 1-x to make a high score
        indicate something good.

        We shift the sigmoid such that the center gives 0.5, and width roughly
        spans from 0.25 to 0.75.
        """
        # Special case for no targets.
        if x == 0:
            return 0
        from dtk.num import sigma
        return 1.0 - sigma((x - center) / (0.5 * width))

    def compute_scores(self):
        dpi_center = self.parms['dpi_sum_sig_center']
        dpi_width = self.parms['dpi_sum_sig_width']
        ppi_center = self.parms['ppi_sum_sig_center']
        ppi_width = self.parms['ppi_sum_sig_width']
        scores = [
                ('dpiSum', lambda x: self._compute_dpi_sum(x)),
                ('dpiSel', lambda x: self._classify_metric(
                    self._compute_dpi_sum(x), center=dpi_center, width=dpi_width)),
                ('ppiSum', lambda x: self._compute_ppi_sum(x)),
                ('ppiSel', lambda x: self._classify_metric(
                    self._compute_ppi_sum(x), center=ppi_center, width=ppi_width)),
                ]

        from atomicwrites import atomic_write
        with atomic_write(self.outfile) as f:
            f.write('\t'.join(['wsa'] + [x[0] for x in scores]) + '\n')
            for wsa in self.wsa2keys.keys():
                row = [str(wsa)]
                for label, fn in scores:
                    row += [str(fn(wsa))]
                f.write('\t'.join(row) + '\n')

    def setup(self):
        from dtk.prot_map import DpiMapping, PpiMapping
        from dtk.files import get_file_records
        dpim = DpiMapping(self.parms['dpi_file'])
        ppim = PpiMapping(self.parms['ppi_file'])
        def key_prot_pairs():
            for rec in get_file_records(dpim.get_path(), keep_header=False):
                yield rec[0], (rec[1], float(rec[2]))
        def prot_prot_pairs():
            for rec in ppim.get_data_records():
                yield rec[0], (rec[1], float(rec[2]))
        from dtk.data import MultiMap
        self.d2pe = MultiMap(key_prot_pairs()).fwd_map()
        self.p2pe = MultiMap(prot_prot_pairs()).fwd_map()
        self.wsa2keys = self.get_wsa2keys(self.d2pe.keys(), dpim)

if __name__ == "__main__":
    MyJobInfo.execute(logger)
