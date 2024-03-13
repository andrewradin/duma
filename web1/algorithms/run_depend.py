#!/usr/bin/env python3

import sys
from path_helper import PathHelper,make_directory

import os
import django_setup
import django

from django import forms

from browse.models import WsAnnotation
from tools import ProgressWriter
from runner.process_info import StdJobInfo, StdJobForm, JobInfo
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager
import runner.data_catalog as dc

import json
import logging
logger = logging.getLogger("algorithms.run_depend")

import scripts.connect_drugs_to_proteinSets as d2ps

import subprocess


def gen_glee_input(ws_or_id, glee_job_id, qv_thres):
    glee_bji = JobInfo.get_bound(ws_or_id, glee_job_id)
    cat = glee_bji.get_data_catalog()
    from math import log
    import numpy as np
    _,gen = cat.get_feature_vectors('NES','qvalue')
    for key,(nes,qv) in gen:
        if nes == 0 or qv >= 10**qv_thres:
            continue
        yield (
                key,
                -1.0*log(qv,10)*log(abs(nes),10),
                np.sign(nes),
                )

def gen_glf_input(ws_or_id, glf_job_id, qv_thres):
    glf_bji = JobInfo.get_bound(ws_or_id, glf_job_id)
    return glf_input_from_bji(glf_bji,qv_thres)

def glf_input_from_bji(glf_bji,qv_thres):
    cat = glf_bji.get_data_catalog()
    from math import log
    import numpy as np
    _,gen = cat.get_feature_vectors('wFEBE','febeQ')
    for key,(score,qv) in gen:
        if qv > 10.**qv_thres:
            continue
        yield (
               key,
               score,
               0
              )

class MyJobInfo(StdJobInfo):
    descr = '''
        <b>DEEPEnD</b>
        (Drugs Exlcusively Encompassing Pathways ENriched in Disease)
        connects drugs to disease via pathways/protein-sets. The algorithm
        uses pre-calculated drug to pathway confidence scores. A disease
        is connected to pathways using the output of GLEE or GLF (you must choose one).
        The overlap of the drug and disease pathways are then scored to give a drug
        efficacy score. <i> DEEPEnD includes a combo method.</i>
        '''
    short_label = "DEEPEnD"
    page_label = "DEEPEnD"

    @classmethod
    def upstream_jid(cls, settings):
        src = settings.get('glf_run', '0')
        if src == '0':
            src = settings['glee_run']
        return src, None

    def make_job_form(self, ws, data):
        from dtk.prot_map import DpiMapping,PpiMapping
        from dtk.d2ps import D2ps
        from scripts.depend import SCORE_METHODS
        score_method_choices = [(x, x) for x in SCORE_METHODS]
        glf_choices = [('0','None')] + ws.get_prev_job_choices('glf')
        glee_choices = [('0','None')] + ws.get_prev_job_choices('glee')
        from dtk.gene_sets import pathway_exclude_choices
        pathway_exclude_choices = pathway_exclude_choices()

        class ConfigForm(StdJobForm):
            combo_with = forms.ChoiceField(label='In combination with',initial=''
                                ,choices=(('','None'),)
                                ,required=False
                                )
            glf_run = forms.ChoiceField(label='GLF Job ID to use (must choose this or GLEE, below)',initial=''
                                ,choices=glf_choices
                                ,required=False
                                )
            glee_run = forms.ChoiceField(label='GLEE Job ID to use (must choose this or GLF, above)',initial=''
                                ,choices=glee_choices
                                ,required=False
                                )
            dpi_file = forms.ChoiceField(
                label='DPI dataset',
                choices = DpiMapping.choices(ws),
                initial = ws.get_dpi_default(),
                )

            ppi_file = forms.ChoiceField(
                label='PPI dataset',
                choices = PpiMapping.choices(),
                initial = ws.get_ppi_default(),
                )
            score_type = forms.ChoiceField(
                label='Score type to use',
                choices = D2ps.enabled_method_choices,
                # NOTE: If you change this, you probably also want to change the
                # default in dtk.workflow.DependStep
                initial = D2ps.default_method,
                )
            score_method = forms.ChoiceField(
                label='Score method',
                choices=score_method_choices,
                initial=score_method_choices[0][0],
                )

            qv_thres = forms.FloatField(label='Q-value threshold exponent (e.g. 10^x)',initial=-2.)
            explicit_combo = forms.BooleanField(label='Explicit combo', initial=True)

            pathway_excludes = forms.MultipleChoiceField(
                        choices=pathway_exclude_choices,
                        initial=[pathway_exclude_choices[0][0]],
                        label='Pathways to exclude',
                        widget=forms.SelectMultiple(
                                attrs={'size':4}
                                ),
                        help_text="Any selected pathway groups will be removed from scoring"
                        )
        return ConfigForm(ws, data)

    def get_data_code_groups(self):
        codetype = self.dpi_codegroup_type('dpi_file')
        return [
                dc.CodeGroup(codetype,self._std_fetcher('outfile'),
                        dc.Code('psScoreMax',label='Max indirect evidence'),
                        )
                ]
    def __init__(self,ws=None,job=None):
        super(MyJobInfo,self).__init__(
                    ws=ws,
                    job=job,
                    src=__file__,
                    )
        # any base class overrides for unbound instances go here
        self.publinks = [
                ]
        # job-specific properties
        if self.job:
            # input files
            self.infile = os.path.join(self.indir, 'input.tsv')
            self.in_moas = os.path.join(self.indir, 'moas.json')
            self.tmp_results = os.path.join(self.outdir, 'depend.tmp')
            # output files
            self.outfile = os.path.join(self.lts_abs_root, 'depend.tsv')
            # published output files
    def run(self):
        self.make_std_dirs()
        self.drug_prefix, _, self.prot_prefix = d2ps.establish_prefixes()
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "setup",
                "wait for remote resources",
                "scoring",
                "check enrichment",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        status = self.setup()
        p_wr.put("setup", status)
        if status != 'complete':
            sys.exit(self.ec.encode('unexpectedDataFormat'))
        got = self.rm.wait_for_resources(self.job.id,[
                                    0,
                                    1,
                                    ])
        self.remote_cores_got = got[1]
        p_wr.put("wait for remote resources"
                        ,"complete; got %d cores"%self.remote_cores_got
                        )
        self.run_depend()
        self.rm.wait_for_resources(self.job.id,[1])
        self.report()
        p_wr.put("scoring", 'complete')
        self.check_enrichment()
        self.finalize()
        p_wr.put("check enrichment","complete")
    def report(self):
        from dtk.prot_map import DpiMapping
        from dtk.d2ps import MoA
        from dtk.files import get_file_records
        dpi = DpiMapping(self.parms['dpi_file'])
        header = None
        moa2wsa = self.wsa2moa.rev_map()
        with open(self.outfile, 'w') as f:
            for frs in get_file_records(self.tmp_results,
                                        parse_type='tsv',
                                        keep_header=True
                                       ):
                if not header:
                    frs[0] = dpi.mapping_type()
                    f.write("\t".join(frs) + "\n")
                    header = True
                    continue
                moa, score, direc = frs

                if float(score) != 0:
                    wsas = moa2wsa[MoA(moa)]
                    for wsa in wsas:
                        f.write("\t".join([str(wsa)] + frs[1:]) + "\n")

    def run_depend(self):
        from pathlib import Path
        local=False
        ps = self._get_pathways_id()
        options = ['--gl', Path(self.infile),
                   '--ppi', self.parms['ppi_file'],
                   '--ps', ps,
                   '--d2ps-method', self.parms['score_type'],
                   '--score-method', self.parms['score_method'],
                   '--moas', Path(self.in_moas),
                   '--cachedir', Path(PathHelper.d2ps_cache),
                   '-o', Path(self.tmp_results),
                  ]
        if self.dpi.is_combo() and self.parms['explicit_combo']:
            options += ['--combo', self.dpi.combo_name()]
        
        print(('command options',options))
        self.run_remote_cmd('scripts/depend.py', options, local=local)
    def get_src_run(self):
        return self.upstream_jid(self.job.settings())[0]
    def get_gl_bji(self):
        return JobInfo.get_bound(self.ws,self.get_src_run())
    def setup(self):
        from runner.process_info import JobInfo
        self.gl_bji = self.get_gl_bji()
        if self.parms['glf_run']!='0':
            assert self.parms['glee_run'] == '0'
            if not self._get_glf_input():
                print("WARNING: No scores passed the Q-value threshold")
                # This will output no useful scores, but shouldn't fail
                # because this can happen with sparse input data.
            self.in_type = 'GLF'
        elif self.parms['glee_run']!='0':
            if not self._get_glee_input():
                print("WARNING: No scores passed the Q-value threshold")
                # This will output no useful scores, but shouldn't fail
                # because this can happen with sparse input data.
            self.in_type = 'GLEE'
        else:
            return 'FAILED - must choose GLF or GLEE'
        self._setup_moas()
        return 'complete'
    def _setup_moas(self):
        from dtk.prot_map import DpiMapping
        dpi = DpiMapping(self.parms['dpi_file'])
        self.dpi = dpi

        if self.dpi.is_combo():
            inputs = self.get_all_input_job_ids()
            from runner.process_info import JobInfo
            for jid in inputs:
                bji = JobInfo.get_bound(self.ws, jid)
                if False and bji.incorporates_combo_base():
                    # NOTE: This is disabled for now, we get better results by double-incorporating here.
                    logger.info(f"Found job {bji} which already accounts for combos, dropping DPI to non-combo")
                    self.dpi = self.dpi.get_noncombo_dpi()
                    break

        if self.parms['explicit_combo']:
            base_dpi = dpi.get_noncombo_dpi()
        else:
            base_dpi = dpi
        logger.info(f"Using dpi {self.parms['dpi_file']} {base_dpi.choice}")
        if dpi.mapping_type() == 'wsa':
            wsas = self.ws.wsannotation_set.filter(agent__removed=False)
            wsa_ids = wsas.values_list('id', flat=True)
            from dtk.enrichment import ws_dpi_condense_keys
            from dtk.prot_map import DpiMapping
            moas = ws_dpi_condense_keys(
                    wsa_ids,
                    dpi_name=base_dpi.choice,
                    # For scoring purposes, we want to exclude DPIs below thresh.
                    dpi_t=self.ws.get_dpi_thresh_default(),
                    )
        else:
            assert dpi.mapping_type() == 'uniprot'
            wsa_ids = []
            moas = []
            from dtk.files import get_file_records
            for _, uniprot, ev, dr in get_file_records(dpi.get_path(),keep_header=False):
                wsa_ids.append(uniprot)
                moas.append(((uniprot, float(ev), float(dr)), ))

        from dtk.d2ps import MoA
        moas = [MoA(x) for x in moas]
        assert len(moas) == len(wsa_ids)
        from dtk.data import MultiMap
        self.wsa2moa = MultiMap(zip(wsa_ids, moas))
        self.moas = set(self.wsa2moa.rev_map().keys())
        with open(self.in_moas, 'w') as f:
            import json
            f.write(json.dumps(list(self.moas), indent=2))

    def gen_gl_input(self):
        if self.parms.get('glf_run', '0') != '0':
            fn = gen_glf_input
        elif self.parms.get('glee_run', '0') != '0':
            fn = gen_glee_input
        return fn(self.ws, self.get_src_run(), self.parms['qv_thres'])

    def _get_glf_input(self):
        from dtk.files import FileDestination
        dest = FileDestination(self.infile,
                header=['uniprotset','score','direction'],
                )
        found_score = False

        from dtk.gene_sets import get_pathway_sets

        exclude_sets = get_pathway_sets(self.parms.get('pathway_excludes', []), self._get_pathways_id())
        excludes = set().union(*exclude_sets)
        num_excluded = 0
        total = 0

        for data in gen_glf_input(self.ws, self.parms['glf_run'], self.parms['qv_thres']):
            total += 1
            if data[0] in excludes:
                num_excluded += 1
            else:
                dest.append(data)
                found_score = True
        logger.info(f"Excluded {num_excluded}/{total} significant pathway scores due to {self.parms.get('pathway_excludes')}")
        return found_score

    def _get_glee_input(self):
        from dtk.files import FileDestination
        dest = FileDestination(self.infile,
                header=['uniprotset','score','direction'],
                )
        found_score = False
        for data in gen_glee_input(self.ws, self.parms['glee_run'], self.parms['qv_thres']):
            dest.append(data)
            found_score = True
        return found_score
    def _get_pathways_id(self):
        try:
            gl_gmt = self.gl_bji.parms['std_gene_list_set']
            return gl_gmt
        except IndexError:
            print('Selected GLEE/GLF run has missing/non-standard gene list set')
            raise ValueError('Bad GLEE run')


if __name__ == "__main__":
    MyJobInfo.execute(logger)
