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

class MyJobInfo(StdJobInfo):
    descr= 'Does a background search of public Omics databases for disease-specific datasets'
    short_label = 'AE Search'
    page_label = 'Search Omics Repositories'

    def make_job_form(self, ws, data):
        from browse.models import AeSearch
        class MyForm(StdJobForm):
            search_term = forms.CharField(
                initial='"'+ws.get_disease_default('EFO')+'"'
                )
            mode = forms.ChoiceField(
                choices=AeSearch.mode_vals.choices(),
                )
            species = forms.ChoiceField(
                choices=AeSearch.species_vals.choices(),
                )
            def as_html(self):
                from django.utils.safestring import mark_safe
                from dtk.html import join
                return join(
                        super().as_html(),
mark_safe('''
<b style="color:darkred">
The suggested search term is optimized for ArrayExpress, but used<br>
with other sources. For best results try using synonyms as well.<br>
Use double quotes to indicate an exact match.
</b>
'''),
                        )
        return MyForm(ws, data)

    # Note that this CM doesn't return anything via the DataCatalog,
    # so there's no usage of LTS outside the logfile stuff.

    def __init__(self,ws=None,job=None):
        super().__init__(ws=ws, job=job, src=__file__, use_LTS=False)
        if self.job:
            pass

    def run(self):
        # None of these directories are needed
        #self.make_std_dirs()
        self.run_steps([
                ('wait for resources',self.reserve_step([1])),
                ('search',self.search),
                ('finalize',self.finalize),
                ])
    
    def search(self):
        p = self.job.settings()
        term = p['search_term'].strip()
        if not term:
            self.fatal('Search term must include non-blank characters')
        term = term.replace(' ','+')
        mode = p['mode']
        species = int(p['species'])
        from dtk.ae_search import do_ae_search
        msgs = do_ae_search(
            term=term,
            mode=mode,
            species=species,
            ws=self.ws,
            )
        for msg in msgs:
            self.info(msg)

if __name__ == "__main__":
    MyJobInfo.execute(logger)
