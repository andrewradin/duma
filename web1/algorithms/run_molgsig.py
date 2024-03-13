#!/usr/bin/env python3

import os
import django_setup
from django import forms

from tools import ProgressWriter
from runner.process_info import StdJobInfo, StdJobForm, JobInfo
import runner.data_catalog as dc
from reserve import ResourceManager

import logging
logger = logging.getLogger(__name__)

class MyJobInfo(StdJobInfo):
    descr= 'Scores molecules via their gene signatures rather than their DPI targets'
    short_label = 'MolGSig'
    page_label = 'Mol Gene Signature'

    def make_job_form(self, ws, data):
        from dtk.scores import JobCodeSelector
        score_choices = JobCodeSelector.get_choices(self.sources,'uniprot','score')

        class MyForm(StdJobForm):
            input_score=forms.ChoiceField(
                    label='Input Score',
                    choices=score_choices,
                    required=True,
                 )
            local=forms.BooleanField(
                label='Run Local',
                initial=False,
                required=False,
            )
        return MyForm(ws, data)

    def get_data_code_groups(self):
        return [
                dc.CodeGroup('wsa',self._std_fetcher('outfile'),
                        dc.Code('negDir',label='Num Dir Align'),
                        dc.Code('sigEv',label='Evidence'),
                        dc.Code('sigCorr',label='Correlation'),
                        )
                ]

    def __init__(self,ws=None,job=None):
        super().__init__(ws=ws, job=job, src=__file__)
        self.needs_sources = True
        if self.job:
            self.infile = os.path.join(self.indir, 'in.pickle.gz')
            self.tmp_outfile = os.path.join(self.outdir, 'scores.tsv')
            self.outfile = os.path.join(self.lts_abs_root, 'wsa_scores.tsv')

    def run(self):
        self.make_std_dirs()

        self.run_steps([
            ('wait for resources', self.reserve_step([1])),
            ('setup', self.setup),
            ('wait for remote resources', self.reserve_step(lambda: [0, self.remote_cores_wanted])),
            ('compute scores', self.compute_scores),
            ('map to wsas', self.map_to_wsas),
            ('finalize', self.finalize),
        ])


    def setup(self):
        import pickle
        from dtk.scores import JobCodeSelector
        ip_code = self.parms['input_score']
        cat = JobCodeSelector.get_catalog(self.ws,ip_code)
        ordering = cat.get_ordering(ip_code,True)
        self.ordering_data = [i[1] for i in ordering]
        import gzip
        with gzip.open(self.infile, 'wb') as handle:
            pickle.dump(ordering, handle)

        self.remote_cores_wanted = (10, 32)
        
    
    def compute_scores(self):
        from pathlib import Path
        from browse.default_settings import lincs
        options = [
            '-i', Path(self.infile),
            '-o', Path(self.outdir),
            '-l', lincs.value(ws=self.ws),
            ]
        self.run_remote_cmd('scripts/molgsig.py', options, local=self.parms['local'])
    
    def _make_lincs_to_wsa(self):
        logger.info("Making lincs to wsa map")
        from browse.default_settings import DpiDataset
        from dtk.prot_map import DpiMapping
        version = DpiMapping(DpiDataset.value(ws=self.ws)).version

        from drugs.models import Drug
        from dtk.data import MultiMap
        lincs_key_id_mm = MultiMap(Drug.objects.filter(collection__name='lincs.full', tag__prop__name='lincs_id').values_list('tag__value', 'id'))
        lincs_ids = set(lincs_key_id_mm.rev_map().keys())
        id_mm = Drug.matched_id_mm(lincs_ids, version)
        logger.info(f"id_mm has {len(id_mm.fwd_map())} fwd and {len(id_mm.rev_map())} rev")

        from browse.models import WsAnnotation
        all_agents = id_mm.fwd_map().keys() | id_mm.rev_map().keys()
        wsas_qs = WsAnnotation.objects.filter(ws=self.ws, agent__in=id_mm.rev_map().keys())
        wsa_agent_mm = MultiMap(wsas_qs.values_list('id', 'agent_id'))

        logger.info(f"wsa_agent_mm has {len(wsa_agent_mm.fwd_map())} wsas and {len(wsa_agent_mm.rev_map())} agents")

        missing_wsa_count = 0

        out = {}
        for lincs_key, lincs_ids in lincs_key_id_mm.fwd_map().items():
            wsas = set()
            assert len(lincs_ids) == 1, f"Multiple lincs IDs for {lincs_key} ({lincs_ids})"
            for lincs_id in lincs_ids:
                linked_ids = id_mm.fwd_map().get(lincs_id, [])
                for linked_id in linked_ids:
                    wsas.update(wsa_agent_mm.rev_map().get(linked_id, []))
                
                if len(wsas) == 0:
                    logger.warn(f'No wsa found for {lincs_key} {lincs_id}')
                    missing_wsa_count += 1
                out[lincs_key] = wsas
        
        logger.info(f"Found wsas for {len(out)} lincs IDs") 
        logger.info(f"Missing wsas for {missing_wsa_count}")
        return out
    
    def map_to_wsas(self):
        lincs_to_wsa = self._make_lincs_to_wsa()

        wsa_scores = {}

        multi_wsas = set()

        header = None
        from dtk.files import get_file_records
        for rec in get_file_records(self.tmp_outfile, keep_header=True):
            if header is None:
                header = rec
                continue
            lincs_key = rec[0]
            skip = False
            import math
            for val in rec[1:]:
                try:
                    # Skip NaN, inf, etc.
                    assert math.isfinite(float(val))
                except:
                    logger.warn(f"Unexpected output of {rec}, skipping")
                    skip = True
                    break
            if skip:
                continue

            for wsa in lincs_to_wsa[lincs_key]:
                if wsa in wsa_scores:
                    multi_wsas.add(wsa)
                    #logger.warn(f"Multiple LINCS scores for wsa {wsa}, picking arbitrarily")
                wsa_scores[wsa] = rec[1:]
        logger.info(f"{len(multi_wsas)} wsas have multiple LINCs ids") 

        header[0] = 'wsa'
        with open(self.outfile, 'w') as f:
            f.write('\t'.join(header) + '\n')
            for wsa, rec in wsa_scores.items():
                full_rec = [str(wsa)] + rec
                f.write('\t'.join(full_rec) + '\n')

    def add_workflow_parts(self,ws,parts,nonhuman=False):
        uji = self # simplify access inside MyWorkflowPart
        class MyWorkflowPart:
            def __init__(self,label,ts,tissue_role):
                self.label=label
                self.ts=ts
                self.tissue_role=tissue_role
                self.enabled_default=False # still experimental
            def add_to_workflow(self,wf):
                cm_info = wf.std_wf.get_main_cm_info(uji.job_type)
                input_name = cm_info.pre.add_pre_steps(
                    wf,
                    tissue_role=self.tissue_role,
                    ts=self.ts,
                    )
                
                my_name = f'{input_name}_molgsig'
                assert my_name.startswith('cc')
                from dtk.workflow import MolGSigStep
                MolGSigStep(
                    wf,
                    my_name,
                    inputs={input_name:True},
                    )
                cm_info.post.add_post_steps(wf,my_name)
        from browse.models import Species
        for ts in ws.tissueset_set.all().order_by('id'):
            is_human = ts.species == Species.HUMAN
            if ts.tissue_set.exists() and ts.num_valid_tissues() > 0 and is_human != nonhuman:
                tissue_role = uji._tissue_set_role_code(ts.id).split('_')[0]
                parts.append(MyWorkflowPart(
                        f'{tissue_role} GESig MolGSig',
                        ts,
                        tissue_role,
                        ))


def load_mol_sigs(lincs_choice):
    """Loads in the mol sig data.

    NOTE: This is invoked from the script, but needs to be external to it for test/mocking purposes.
    """
    from dtk.s3_cache import S3File
    expr_s3 = S3File.get_versioned('lincs', lincs_choice, role='expression')
    expr_s3.fetch()

    metadata_s3 = S3File.get_versioned('lincs', lincs_choice, role='metadata')
    metadata_s3.fetch()

    logger.info("Loading signatures")
    import numpy as np
    # This comes in as (genes, mols), we transpose to (mols, genes) on return.
    data = np.load(expr_s3.path())['expr']

    logger.info(f"Loaded with shape {data.shape}")

    import json
    meta = json.loads(open(metadata_s3.path()).read())
    return data.T, meta

if __name__ == "__main__":
    MyJobInfo.execute(logger)
