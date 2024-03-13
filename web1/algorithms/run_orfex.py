#!/usr/bin/env python3

import sys

import os
import django

from django import forms

from browse.models import WsAnnotation
from tools import ProgressWriter
from runner.process_info import JobInfo
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager

import json
import logging
logger = logging.getLogger("algorithms.run_orfex")


# This is a stub for a job that used to exist, but is no longer used.
# We keep a definition around just because entries for it still exist in the
# runner.Process table and the system doesn't like being unable to find it.
class MyJobInfo(JobInfo):
    def __init__(self,ws=None,job=None):
        # base class init
        super(MyJobInfo,self).__init__(
                    ws,
                    job,
                    __file__,
                    "ORFEX",
                    "Over-Represented Feature EXtractor",
                    )
        # any base class overrides for unbound instances go here
        self.publinks = [
                ('Protein target count histogram','target_count_hist.png'),
                ('Enriched Targets',"enriched_targets.txt"),
                ('Enriched ATC level 2',"enriched_atc_lvl2.txt"),
                ('Enriched ATC level 3',"enriched_atc_lvl3.txt"),
                ]
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.ec = ExitCoder()
            # published output files
            self.ofile = self.outdir + "enriched_targets.txt"
            self.out_enrch_ts = self.tmp_pubdir + "enriched_targets.txt"
            self.out_enrch_atc2 = self.tmp_pubdir + "enriched_atc_lvl2.txt"
            self.out_enrch_atc3 = self.tmp_pubdir + "enriched_atc_lvl3.txt"
            self.out_hist = self.tmp_pubdir + 'target_count_hist.plotly'
            self.publinks.append((None,"target_count_hist.plotly"))
            try:
                with open(self.ofile) as f:
                    uniprots = [x.strip().split('\t')[0] for x in f]
            # strip any blank lines (including empty file case)
                uniprots = [x for x in uniprots if x and x != 'Uniprot']
            except IOError:
                uniprots = []
            links = [self.ws.reverse("protein",u) for u in uniprots]
            self.otherlinks = tuple(zip(uniprots, links))

def get_atc_cache():
    from drugs.models import Prop
    atc_prop = Prop.get('atc')
    atc_cache = {}
    for val in atc_prop.cls().objects.filter(prop=atc_prop):
        atc_cache.setdefault(val.drug_id,[]).append(str(val.value))
    return atc_cache
