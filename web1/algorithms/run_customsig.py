#!/usr/bin/env python3

import os
import django_setup
from django import forms

from tools import ProgressWriter
from runner.process_info import StdJobInfo, StdJobForm 
import runner.data_catalog as dc

import logging
logger = logging.getLogger(__name__)

class ParseException(Exception):
    pass

def parse_score_json(score_text):
    try:
        import json
        data = json.loads(score_text)
    except json.JSONDecodeError as e:
        raise ParseException(str(e))

    if not isinstance(data, dict):
        raise ParseException(f"{data} is a {type(data)} instead of a dict")

    from browse.models import Protein
    u2g = Protein.get_uniprot_gene_map()
    g2u = Protein.get_gene_uniprot_map()
    canon_uniprots = set(Protein.objects.all().values_list('uniprot', flat=True))

    errors = []

    full_table = []
    mapped_data = {}
    for k, v in data.items():
        k = g2u.get(k, k)
        if k not in canon_uniprots:
            errors.append(f'Unmapped token "{k}" is not a gene or uniprot')
            continue
        mapped_data[k] = v
        full_table.append([k, u2g.get(k), v])

    if errors:
        raise ParseException('\n'.join(errors))

    return mapped_data, full_table

class MyJobInfo(StdJobInfo):
    descr= 'Generates custom user-defined protein signatures'
    short_label = 'Custom Sig'
    page_label = 'Custom Signature'
    extra_job_template = 'algorithms/customsig.html'

    def make_job_form(self, ws, data):
        from browse.models import Species
        class MyForm(StdJobForm):
            sig_json = forms.CharField(
                    widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
                    required=True,
                    label='Score Json',
                    help_text='Dict of {prot/gene: score}',
                    )
            description = forms.CharField(
                    widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
                    required=True,
                    label='Description',
                    help_text='Documentation of what this signature is and/or where it comes from',
                    )
            shortname = forms.CharField(
                required=True,
                label='Short Name',
                help_text='Identifier for selecting in dropdowns and such'
            )

            species = forms.ChoiceField(
                choices=Species.choices(),
            )
        return MyForm(ws, data)
    
    def short_name(self):
        if 'shortname' in self.parms:
            out = self.parms['shortname']
        else:
            # Earlier runs didn't have a shortname field.
            out = self.parms['description']
        
        if len(out) > 20:
            out = out[:17] + '..'
        return out

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
                "compute scores",
                "finalize",
                ])
        self.compute_scores()
        p_wr.put("compute scores","complete")
        self.finalize()
        p_wr.put("finalize","complete")
    
    def compute_scores(self):
        mapped_data, full_table = parse_score_json(self.parms['sig_json'])
        print("Scores:\n" + '\n'.join('\t'.join(str(v) for v in x) for x in full_table))

        from atomicwrites import atomic_write
        with atomic_write(self.outfile) as f:
            f.write('\t'.join(['uniprot', 'ev']) + '\n')
            for prot, ev in sorted(mapped_data.items(), key=lambda x: (-x[1], x[0])):
                row = [prot, str(ev)]
                f.write('\t'.join(row) + '\n')

    def add_workflow_parts(self,ws,parts, nonhuman=False):
        from dtk.workflow import WorkStep

        # Most worksteps are launching a background job; this one
        # is instead just a placeholder so that downstrema jobs know
        # which jobid to use as input.
        class CustomSigStep(WorkStep):
            def __init__(self, *args, **kwargs):
                self.job_id = kwargs.pop('job_id')
                super().__init__(*args, **kwargs)
                self.done()

        uji = self # simplify access inside MyWorkflowPart
        class MyWorkflowPart:
            def __init__(self,label,jid):
                self.label=label
                self.jid = jid
            def add_to_workflow(self,wf):
                cm_info = wf.std_wf.get_main_cm_info(uji.job_type)
                cm_info.pre.add_pre_steps(wf)
                my_name = f'customsig_{self.jid}'
                CustomSigStep(wf,my_name,job_id=self.jid)
                cm_info.post.add_post_steps(wf,my_name,'ev')
        from browse.models import Species
        from runner.process_info import JobInfo
        for choice in ws.get_prev_job_choices(self.job_type):
            bji = JobInfo.get_bound(ws, choice[0])
            species_idx = int(bji.parms['species'])
            is_human = (species_idx == Species.HUMAN)
            if is_human == nonhuman:
                continue
            species = Species.get('label', species_idx)
            label = f'CustomSig ({species}) {bji.short_name()}'
            parts.append(MyWorkflowPart(label, choice[0]))


if __name__ == "__main__":
    MyJobInfo.execute(logger)
