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
logger = logging.getLogger("algorithms.run_example")

import json
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo

################################################################################
# Step-by-step plug in guide
# 
# Also see the comments in the JobInfo base class definition in
# runner/process_info.py.
################################################################################
# 1) pick your plugin abbreviation (e.g. comd or struct)
# 2) copy example to make a run_<abbreviation>.py file
# 3) change the 2 strings around "Plugin Example Process" to your method name
#     (e.g. Community detection)
# 4) add abbreviation to level_names; verify it shows up on workflow page,
#     and in the Run dropdown menu (if the plugin manages runs of different
#     kinds that should be distinct (like TR vs CC pathsum), you'll also
#     need to provide an override for source_label()
# 5) from the Run dropdown menu, go to the job_start page
# 6) work on get_config_html(), refreshing the job_start page to check your work
#     Define what parameters the user will set 
#     If code changes do not make expected changes on UI, try restarting the
#     server.
# 7) try implementing handle_config_post(), testing it by hitting the 'Run'
#     button
# 8) once the framework is working, invoke your actual code from the run()
#     method
# 9) once your job runs, use self.publinks to expose links to any static output
#     files
# 10) call check_enrichment() and finalize() from inside run() to run DEA, and
#     to copy any published files into their final position
# 11) implement get_data_code_groups() to return score and feature vector
#     results



class MyJobInfo(JobInfo):
    # TODO: add to JobCrossChecker.level_names
    # TODO: def get_config_html(), handle_config_post()
    def __init__(self,ws=None,job=None):
        self.use_LTS=True
        # base class init
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "Example",
                "Plugin Example Process",
                )
        # any base class overrides for unbound instances go here
        # TODO: self.publinks
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            # TODO: calculate paths to individual files; note that is a
            # TODO: score file is calculated on the worker, it should
            # TODO: have both a temporary path (in the self.output
            # TODO: directory) and a permanent path (in the self.lts_abs_root
            # TODO: directory), and code called from run() should move the
            # TODO: file from one to the other, prior to calling finalize()
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        make_directory(self.lts_abs_root)
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "step 1",
                "step 2",
                # ...
                'check enrichment',
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        # TODO: call method to do step 1
        p_wr.put("step 1","complete")
        # TODO: call method to do step 2
        p_wr.put("step 2","complete")
        # ...
        self.check_enrichment()
        self.finalize()
        p_wr.put("check enrichment","complete")

if __name__ == "__main__":
    MyJobInfo.execute(logger)
