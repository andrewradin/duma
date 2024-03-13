#!/usr/bin/env python3

from __future__ import print_function
import sys
try:
    from path_helper import PathHelper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")
    from path_helper import PathHelper

import os
import django

if not "DJANGO_SETTINGS_MODULE" in os.environ:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    django.setup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='produce text description of selected jobs on stdout'
            )
    parser.add_argument('--job_id')
    parser.add_argument('jobname',nargs='+')
    args=parser.parse_args()

    from runner.process_info import JobCrossChecker,JobInfo
    jcc=JobCrossChecker()

    import json
    for jobname in args.jobname:
        if args.job_id:
            from runner.models import Process
            p = Process.objects.get(pk=args.job_id)
            settings = p.settings()
        else:
            uji = JobInfo.get_unbound(jobname)
            try:
                latest = jcc.latest_jobs()[jobname]
                settings = latest.settings()
            except KeyError:
                settings = {}
        print(jobname,json.dumps(
                        settings,
                        separators=(',',':'),
                        sort_keys=True,
                        ))
