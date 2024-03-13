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
            description='queue runner jobs from a text description on stdin'
            )
    parser.add_argument('--queue-only',action='store_true')
    parser.add_argument('user_name')
    args=parser.parse_args()

    from runner.process_info import JobCrossChecker
    jcc=JobCrossChecker()
    for line in sys.stdin:
        jobname,settings = line.strip('\n').split(None,2)
        plugin = jobname.split('_')[0]
        jcc.queue_job(
                plugin,
                jobname,
                user=args.user_name,
                settings_json=settings,
                )
    if not args.queue_only:
        from runner.models import Process
        err = Process.prod_user_error()
        if err:
            print(err)
            print("Your work has been queued, but not started.")
            print("To start, run:")
            print("  sudo -u www-data queue_jobs - </dev/null")
            sys.exit(1)
        Process.drive_background()
