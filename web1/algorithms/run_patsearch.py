#!/usr/bin/env python3

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
logger = logging.getLogger(__name__)

from django import forms

import json
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo

class MyJobInfo(JobInfo):
    def description_text(self):
        return '''
        Runs a patent search
        '''
    def __init__(self,ws=None,job=None):
        # base class init
        self.use_LTS = True
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "Patent Search",
                "Patent Search",
                )
        if self.job:
            from patsearch.models import PatentSearch
            search = PatentSearch.objects.filter(job=self.job.id)
            if search:
                search_id = search[0].id
                self.otherlinks = [
                        ('Patent Search Results', ws.reverse('pats_summary', search_id))
                        ]
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        make_directory(self.lts_abs_root)

        query = self.parms['query']
        
        from patsearch.patent_search import PatentContentJob, PatentContentStorage
        storage = PatentContentStorage(self.lts_abs_root, job=self.job, ws=self.ws)
        runner = PatentContentJob(storage, query)

        steps = ['wait for resources'] + runner.steps() + ['finalize']
        p_wr = ProgressWriter(self.progress, steps)
        rm = ResourceManager()
        rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")

        runner.find_and_rank_patents(self.job.user, self.ws, self.job, p_wr)
        
        # If all our patent content already existed, we'll have an empty
        # directory, which we can't finalize.  Just write an empty file.
        with open(os.path.join(self.lts_abs_root, 'done'), 'w') as f:
            pass

        self.finalize()
        p_wr.put("finalize","complete")

    
if __name__ == "__main__":
    MyJobInfo.execute(logger)
