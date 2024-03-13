#!/usr/bin/env python3

import sys
from path_helper import PathHelper,make_directory
import os
import django

import logging
logger = logging.getLogger("algorithms.run_uphd")

from django import forms

import subprocess
import json
import time
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo
import runner.data_catalog as dc

class MyJobInfo(JobInfo):
    def __init__(self,ws=None,job=None):
        self.use_LTS=True
        # base class init
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "iDAP",
                "DisGeNet Disease associated proteins",
                )
        # any base class overrides for unbound instances go here
        # job-specific properties
        if self.job:
            self.dgn_filename = 'disGeNet.curated.tsv'
            self.log_prefix = str(self.ws)+":"
            self.accs = list(set([ x.strip()
                        for x in self.parms['search_term'].split(',')
                        ]))
            self._get_sources()
            self.debug("setup")
            # inputs
            self.uphd_dir = PathHelper.uphd
            self.file_suffix = ".txt"
            # outputs
            self.ofile = self.lts_abs_root+'uniprot_IDs.tsv'
            try:
                with open(self.ofile) as f:
                    uniprots = [x.strip().split('\t')[0] for x in f]
                # strip any blank lines (including empty file case)
                uniprots = [x for x in uniprots if x]
            except IOError:
                uniprots = []
            links = [self.ws.reverse("protein",u) for u in uniprots]
            self.uniprots = uniprots
            self.otherlinks = tuple(zip(uniprots, links))
    def _get_sources(self):
        self.sources = []
        ### Previous version of this code, UPHD, did not have these
        ### and loading old runs was causing an issue here
        try:
            if self.parms['gwas']:
                self.sources.append('GWASCAT')
            if self.parms['cvar']:
                self.sources.append('CLINVAR')
            if self.parms['uni']:
                self.sources.append('UNIPROT')
            if self.parms['ctd']:
                self.sources.append('CTD_human')
            if self.parms['onet']:
                self.sources.append('ORPHANET')
        except KeyError:
            pass
    def get_data_code_groups(self):
        return [
                dc.CodeGroup('uniprot',self._dap_fetcher,
                        dc.Code('dap',valtype='bool',label='Disease Proteins'),
                        ),
                ]
    def _dap_fetcher(self,keyset):
        for p in self.uniprots:
            yield (p,(True,))
