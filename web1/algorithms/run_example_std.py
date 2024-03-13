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

################################################################################
# Step-by-step plug in guide
# 
# If you're doing something non-standard, you might be better off starting
# from the more general algorithms/run_example.py.
################################################################################
# 1) pick your plugin abbreviation (e.g. comd or struct)
# 2) copy example to make a run_<abbreviation>.py file
# 3) change the 2 strings with 'CM Name' below to match your CM name
# 4) modify the descr string below as appropriate
# 5) add abbreviation to level_names; verify it shows up on the Run dropdown
#     menu (under Other CMs)
# 6) from the Run dropdown menu, go to the job_start page
# 7) work on make_job_form(), refreshing the job_start page to check your work
#     Define what parameters the user will set 
#     If code changes do not make expected changes on UI, try restarting the
#     server.
#     This should allow you to start a job that will end up dying in run().
# 8) once the framework is working, remove the NotImplementedError exception.
#     and fill out the run_steps call
# 9) once your job runs, use self.publinks to expose links to any static output
#     files
# 10) implement get_data_code_groups() to return data
# 11) add CM to run menu
# 12) add CM to any workflow

class MyJobInfo(StdJobInfo):
    descr= 'Describe CM here' # XXX
    short_label = 'CM Name' # XXX
    page_label = 'Longer CM Name' # XXX
    # XXX Specify jobs that this job depends on for inputs.
    # XXX - if there's only one, provide a function that extracts it from
    # XXX   settings, like the lambda below:
    # XXX upstream_jid = lambda cls, settings: (settings['capp_job'], None)
    # XXX - if there are none, you can just leave this undefined
    # XXX - if there are multiple input jobs, you should probably use the
    # XXX   JobInfo class directly

    def make_job_form(self, ws, data):
        class MyForm(StdJobForm):
            pass # XXX define the form that configures the job

        return MyForm(ws, data)

    def get_data_code_groups(self):
        return [
                # XXX define the job outputs
                # XXX dc.CodeGroup('uniprot',self._std_fetcher('outfile'),
                # XXX         dc.Code('ev',label='Evidence'),
                # XXX         ),
                ]

    def __init__(self,ws=None,job=None):
        super().__init__(ws=ws, job=job, src=__file__)
        if self.job:
            pass # XXX define any job-dependent members (usually file paths)
            # XXX self.outfile = os.path.join(self.lts_abs_root, 'myfile.tsv')

    def run(self):
        self.make_std_dirs()
        # XXX alternatively, you can remove the run_steps call below,
        # XXX and code a typical run() method here,
        # XXX with explicit ProgressWriter and resource manager calls
        raise NotImplementedError()
        self.run_steps([
                # XXX add reserve_step parameter to the line below:
                # XXX ('wait for resources',self.reserve_step(parms_go_here)),
                # XXX define job steps
                # XXX Note that generally each definition is a tuple consisting
                # XXX of a label and a method (not a method call). If the call
                # XXX needs to be parameterized, you need to define a wrapper
                # XXX that takes the parameters, and returns a callable object.
                # XXX See StdJobInfo.reserve_step() as an example.
                ('finalize',self.finalize),
                ])
    
    # XXX add job steps referenced in the list above

if __name__ == "__main__":
    MyJobInfo.execute(logger)
