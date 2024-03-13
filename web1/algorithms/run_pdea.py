#!/usr/bin/env python3

import sys
from path_helper import PathHelper,make_directory
import os
import django

from django import forms

from tools import ProgressWriter
from runner.process_info import JobInfo
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager
from dtk.prot_map import DpiMapping

import json
import logging
logger = logging.getLogger("algorithms.run_pdea")

class MyJobInfo(JobInfo):
    def __init__(self,ws=None,job=None):
        # base class init
        super(MyJobInfo,self).__init__(
                    ws,
                    job,
                    __file__,
                    "Protein-DEA",
                    "pDEA",
                    )
        # any base class overrides for unbound instances go here
        self.publinks = (
                ('pDEA plot',"pdea_noscore_nods_nobg_DEAPlots.pdf"),
                )
        self.needs_sources = True
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.ec = ExitCoder()
            # input files
            self.uniprot_file = self.indir + 'uniprot.txt'
            # output files
            # published output files
            self.oplot = self.tmp_pubdir + 'pdea_noscore_nods_nobg_DEAPlots.pdf'
