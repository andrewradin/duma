#!/usr/bin/env python3

import sys
import os

import django
import django_setup

from runner.models import Process
from path_helper import PathHelper

# Note that our default logging configuration sends stuff both to 'console'
# (which appears on stderr with an abbreviated timestamp) and to 'syslog'
# (which ends up in django.log with more extensive labeling). This process
# gets launched with stderr redirected to drive_background.log, so that
# file can be consulted to see any stderr output of the processes invoked,
# interspersed with the 'console' logs for approximate timing. You can
# consult django.log to see the same log messages with the complete date
# timestamp if necessary to remove ambiguities.
#
# Each subprocess spawned from here is a copy of runner/run_process.py,
# which itself spawns the job to be executed. run_process redirects the
# stderr of its subprocess into the appropriate bg_log.txt file for the
# job, so the only stderr output in drive_background.log is what's generated
# here or in run_process itself.

import logging
logger = logging.getLogger("runner.drive_background")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='background daemon')
    args=parser.parse_args()

    logger.info("starting")
    os.setpgrp()
    os.chdir(PathHelper.website_root)
    logger.info("cwd %s",os.getcwd())

    r = Process.start_all()
    logger.info("started %d; %d queued",r[0],r[1])
