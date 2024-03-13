#!/usr/bin/env python3

from __future__ import print_function
import sys
import six
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
logger = logging.getLogger("algorithms.run_tcgamut")

from django import forms

import json
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo

def get_tcgamut_file_choices(ws_id):
    import re
    pattern = re.compile(r'tcgamut\.'+str(ws_id)+r'\.(.*)\.tsv$')
    from dtk.s3_cache import S3MiscBucket
    b = S3MiscBucket()
    choices = []
    for name in b.list(cache_ok=True):
        m = pattern.match(name)
        if m:
            choices.append((name,m.group(1)))
    return choices
def get_prot_gene_syns():
    from dtk.data import MultiMap
    from browse.models import ProteinAttribute
    mm=MultiMap(
            ProteinAttribute.objects.filter(
                    attr__name__in=['Gene_Synonym', 'GeneCards'],
                    ).values_list('val','prot__uniprot')
            )
    return mm.fwd_map()
def get_all_prots(gene, prot_gene_syns):
    from browse.models import Protein
    prot_qs=Protein.objects.filter(gene=gene)
    unis = list(prot_qs.values_list('uniprot',flat=True))
    unis += list(prot_gene_syns.get(gene, set()))
    return unis

class MyJobInfo(JobInfo):
    def description_text(self):
        return '''
        Imports mutation data for a disease fron the TCGA database.
        The dataset to be imported must be manually prepared and
        made available to the platform.
        '''
    def get_config_form_class(self,ws):
        choices = get_tcgamut_file_choices(ws.id)
        if choices:
            # rather than defaulting to the first file, force the user
            # to select a non-default choice from the list
            choices.insert(0,('','No file selected'))
        else:
            # lock out the run button and provide some indication of
            # what's going on
            choices.insert(0,('','No file available'))
        class ConfigForm(forms.Form):
            input_file = forms.ChoiceField(
                        label='Input File',
                        choices=choices,
                        )
            def as_dict(self):
                if self.is_bound:
                    src = self.cleaned_data
                else:
                    src = {fld.name:fld.field.initial for fld in self}
                p ={'ws_id':ws.id}
                for f in self:
                    key = f.name
                    value = src[key]
                    p[key] = value
                return p
        return ConfigForm
    def settings_defaults(self,ws):
        cfg=self.get_config_form_class(ws)()
        return {
                'default':cfg.as_dict(),
                }
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        ConfigForm = self.get_config_form_class(ws)
        if copy_job:
            form = ConfigForm(initial=copy_job.settings())
        else:
            form = ConfigForm()
        return form.as_p()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        ConfigForm = self.get_config_form_class(ws)
        form = ConfigForm(post_data)
        if not form.is_valid():
            return (form.as_p(),None)
        settings = form.as_dict()
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(settings)
                            )
        next_url = ws.reverse('nav_progress',job_id)
        return (None, next_url)
    def __init__(self,ws=None,job=None):
        self.use_LTS = True
        # base class init
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "TCGA Mutation",
                "TCGA Mutation Data Import",
                )
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.score_fn = self.lts_abs_root+'mutation_scores.tsv'
    def is_a_source(self):
        return True
    def get_data_code_groups(self):
        import runner.data_catalog as dc
        return [
                dc.CodeGroup('uniprot',self._std_fetcher('score_fn'),
                        dc.Code('mutpor',label='Mutation Portion', fmt='%.2e'
                                ),
                        dc.Code('mutbgpor',
                                label='Mutation Background Portion', fmt='%.2e',
                                ),
                        dc.Code('muten',label='Mutation Enrichment', fmt='%.2e',
                                calc=(lambda x,y:x/y,'mutpor','mutbgpor'),
                                ),
                        ),
                ]
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.lts_abs_root)
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "fetch data",
                "convert data",
                "cleanup",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        # fetch data
        from dtk.s3_cache import S3MiscBucket,S3File
        b = S3MiscBucket()
        f = S3File(b,self.parms['input_file'])
        f.fetch()
        p_wr.put("fetch data","complete")
        self.convert_data(f)
        p_wr.put("convert data","complete")
        self.finalize()
        p_wr.put("cleanup","complete")
    def convert_data(self,f):
        from dtk.files import get_file_records,FileDestination
        dest = FileDestination(self.score_fn,[
                'uniprot',
                'mutpor',
                'mutbgpor',
                ])
        src = get_file_records(f.path())
        header = next(src)
        gene_idx = header.index('Symbol')
        try:
            cohort_idx = header.index('# SSM Affected Cases in Cohort')
            total_idx = header.index('# SSM Affected Cases Across the GDC')
        except ValueError:
            # this means we're using an older data file (before mid 2019, or earlier)
            # the end of this line is dependent on the disease,
            # so just find the first instance that starts correctly (there's only ever one)
            cohort_idx = header.index([x for x in header
                                   if x.startswith('# Affected Cases in ')
                                  ][0]
                                 )
            total_idx = header.index('# Affected Cases Across the GDC')
        import re
        def cvt_fraction(s):
            m = re.match(r'([0-9,]+) / ([0-9,]+)',s)
            num = float(m.group(1).replace(',',''))
            denom = float(m.group(2).replace(',',''))
            return num/denom
        from dtk.hgChrms import get_prot_txn_sizes
        from browse.default_settings import ucsc_hg
        try:
            prot_txn_sizes = get_prot_txn_sizes(ucsc_hg.value(self.ws))
        except AssertionError as e:
            print("WARNING: You're using an outdated GWAS setting which also effects the genetics of this CM. Please update to >= v2")
            raise
        prot_gene_syns = get_prot_gene_syns()
        data = {}
        print('The following genes had no transcript length. The uniprots tried are list for each gene')
        for rec in src:
            unis = get_all_prots(rec[gene_idx], prot_gene_syns)
            ### The only genes I've found in this category are psuedogenes or other ncRNAs
            if len(unis) == 0:
                continue
            cohort_portion = cvt_fraction(rec[cohort_idx])
            # no need to keep going if this gene isn't mutated in this cancer
            if not cohort_portion:
                continue
            bg_portion = cvt_fraction(rec[total_idx])
            no_len = []
            for uniprot in unis:
                if uniprot not in prot_txn_sizes:
                    no_len.append(uniprot)
                    continue
                txn_len = prot_txn_sizes[uniprot]
                normd_cohort_por = cohort_portion/txn_len
                if uniprot not in data:
                    data[uniprot] = [normd_cohort_por, bg_portion/txn_len]
                if normd_cohort_por > data[uniprot][0]:
                    data[uniprot] = [normd_cohort_por, bg_portion/txn_len]
            if unis == no_len:
                print(rec[gene_idx],unis)
        for k,l in six.iteritems(data):
            dest.append([k]+l)
    def add_workflow_parts(self,ws,parts):
        uji = self # simplify access inside MyWorkflowPart
        class MyWorkflowPart:
            def __init__(self,label,input_file):
                self.label=label
                self.input_file=input_file
            def add_to_workflow(self,wf):
                cm_info = wf.std_wf.get_main_cm_info(uji.job_type)
                cm_info.pre.add_pre_steps(wf)
                from dtk.workflow import TcgaMutStep
                my_name = 'tcgamut'
                TcgaMutStep(wf,my_name,
                        input_file=self.input_file,
                        )
                cm_info.post.add_post_steps(wf,my_name,'mutpor')
        for choice in get_tcgamut_file_choices(ws.id):
            parts.append(MyWorkflowPart(
                    'TCGAMUT '+choice[1],
                    choice[0],
                    ))

if __name__ == "__main__":
    MyJobInfo.execute(logger)
