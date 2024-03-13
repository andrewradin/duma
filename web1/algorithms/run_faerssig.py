#!/usr/bin/env python3


import os
import django_setup
from django import forms

from tools import ProgressWriter
from runner.process_info import StdJobInfo, StdJobForm, JobInfo
from reserve import ResourceManager
import runner.data_catalog as dc
from dtk.prot_map import DpiMapping, PpiMapping

import json
import logging
logger = logging.getLogger(__name__)

def make_capp_sig(assoc_groups):
    from collections import defaultdict
    prot2evs = defaultdict(list)

    # CAPP doesn't 0-1 normalize scores; let's track the highest assoc score
    # and use that to normalize for the signature.
    max_assoc = 0
    for group in assoc_groups.values():
        for prot, assoc in group.items():
            prot2evs[prot].append(assoc)
            max_assoc = max(max_assoc, assoc)
    
    out = {}

    # Exit early if 0 rather than potentially generating NaNs.
    if max_assoc == 0:
        return out

    for prot, evs in prot2evs.items():
        # Mean ev score, with implicit 0 for anything it wasn't found in.
        out[prot] = (sum(evs) / len(assoc_groups)) / max_assoc
    
    return out
    

class MyJobInfo(StdJobInfo):
    descr= 'Generates FAERS-based signatures from FAERS/CAPP disease associations'
    short_label = 'FAERS Sig'
    page_label = 'FAERS Signature'
    upstream_jid = lambda cls, settings: (settings['capp_job'], None)

    def make_job_form(self, ws, data):
        capp_choices = ws.get_prev_job_choices('capp')

        class MyForm(StdJobForm):
            capp_job = forms.ChoiceField(
                    label='CAPP run',
                    choices=capp_choices,
                    )

        return MyForm(ws, data)

    def get_data_code_groups(self):
        return [
                dc.CodeGroup('uniprot',self._std_fetcher('outfile'),
                        dc.Code('ev',label='Evidence'),
                        )
                ]

    def __init__(self,ws=None,job=None):
        super().__init__(ws=ws, job=job, src=__file__)
        if self.job:
            self.outfile = os.path.join(self.lts_abs_root, 'signature.tsv')

    def run(self):
        self.make_std_dirs()
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "compute scores",
                "finalize",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.compute_scores()
        p_wr.put("compute scores","complete")
        self.finalize()
        p_wr.put("finalize","complete")
    
    def compute_scores(self):
        capp_jid = self.parms['capp_job']
        capp_bji = JobInfo.get_bound(self.ws, capp_jid)
        comorb_gene_assocs, _, _ = capp_bji.make_cmap_data()
        sig = make_capp_sig(comorb_gene_assocs)

        from atomicwrites import atomic_write
        with atomic_write(self.outfile) as f:
            f.write('\t'.join(['uniprot', 'ev']) + '\n')
            for prot, ev in sorted(sig.items(), key=lambda x: -x[1]):
                row = [prot, str(ev)]
                f.write('\t'.join(row) + '\n')

    def add_workflow_parts(self,ws,parts):
        jobnames = self.get_jobnames(ws)
        assert len(jobnames) == 1
        jobname=jobnames[0]
        uji = self # simplify access inside MyWorkflowPart
        class MyWorkflowPart:
            label=uji.source_label(jobname)
            # NOTE that both this CM and the FAERS DataStatus are hardwired to
            # the faers CDS. This could be extended in the future by creating
            # multiple statuses and multiple workflow parts. But for now,
            # rely on the single DataStatus.
            enabled_default=uji.data_status_ok(
                    ws,
                    'Faers',
                    'Complete Clinical Values',
                    )
            def add_to_workflow(self,wf):
                cm_info = wf.std_wf.get_main_cm_info(uji.job_type)
                cm_info.pre.add_pre_steps(wf)
                # In attempting to reverse engineer this, I think what's
                # going on is that this counts on the capp part to add the
                # necessary pre-steps, and just hooks on here for input
                # and ordering.
                # (PreFaersSig.add_pre_steps doesn't do anything)
                # The all-caps FAERS below forces it to pull from the
                # current version of the faers cds.
                from dtk.workflow import FaersSigStep
                prereq = 'FAERS_faers_otarg_capp'
                name= prereq + '_faerssig'
                FaersSigStep(wf,name,
                        inputs={prereq: True}
                        )
                cm_info.post.add_post_steps(wf,name,'ev')
        parts.append(MyWorkflowPart())

if __name__ == "__main__":
    MyJobInfo.execute(logger)
