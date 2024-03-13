#!/usr/bin/env python3

from __future__ import print_function
import sys
import six
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

# This script processes an input file to queue a bunch of jobs that vary
# some parameter. The idea is to evaluate settings decisions across many
# workspaces to get less biased results. Doing this requires beginning
# with some globally shared settings, adding in ws-specific variations
# that are supposedly independent of the evaluation, and then adding in
# the variants under consideration. The detailed format of the input file
# is described in-line in the parsing code below. An initial version of
# the file which may be hand-tweaked is produced by the script
# build_xws_parm_eval_script.py.
#
# This process was never very heavily used. The conversion of the system
# from RunSet to ScoreSet models required changes. The script below has
# been edited to most likely work with ScoreSets, but it hasn't been
# tested. A view in the UI that supported examining results of these
# runs was removed in commit 0f2472eebd0b11857c7aa7cb7429fe1233763ea9,
# but could be reinstated and upgraded without too much trouble if this
# process comes back into favor.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='queue jobs for all ws/variant combinations'
            )
    parser.add_argument('--queue-only',action='store_true')
    parser.add_argument('user_name')
    args=parser.parse_args()

    # Config file is expected on stdin
    # - one record per line
    # - records can be commented out with '#' in first column
    def stdin_by_line():
        for line in sys.stdin:
            if line.startswith('#'):
                continue
            yield line.rstrip()
    inp=stdin_by_line()
    # First line is a description to be stored in the scoreset
    ss_desc = next(inp)
    # Second line is json for the common settings; this can be
    # obtained using the build_jobfile.py script
    import json
    settings_template = json.loads(next(inp))
    class DummyClass:
        pass
    # Next n lines (up to the first blank line) define the Component
    # Method parameter(s) to be varied.  These are in JSON, as overrides
    # or supplements to the common settings above. If the line contains
    # a tab, the JSON is preceeded by a description; otherwise, a
    # description is constructed summarizing the variant data.
    variant_list=[]
    for line in inp:
        if not line:
            break
        d=DummyClass()
        variant_list.append(d)
        if '\t' in line:
            parts = line.split('\t')
            assert len(parts) == 2
            d.desc = parts[0]
            d.settings=json.loads(parts[1])
        else:
            d.settings=json.loads(line)
            d.desc = ' '.join([
                    '%s=%s'%(k,str(d.settings[k]))
                    for k in sorted(d.settings)
                    ])
    # The remaining lines each describe a workspace to perform the
    # evaluation in.  The format of each line is a workspace id,
    # the relevant jobname within that workspace, and any workspace-specific
    # settings overrides expressed as JSON.
    # XXX This might be the next part to automate, as there are often complex
    # XXX workspace-specific settings (like per-tissue thresholds).  One
    # XXX option is an alternate line format with a workspace id and a
    # XXX 'flavor' code that builds a particular group of settings from the
    # XXX workspace configuration.
    from browse.models import Workspace,ScoreSet,ScoreSetJob
    ws_list=[]
    for line in inp:
        d=DummyClass()
        ws_list.append(d)
        ws_id,jobname,settings = line.split(' ')
        d.ws = Workspace.objects.get(pk=ws_id)
        d.jobname = jobname
        d.settings=json.loads(settings)
    # Now create scoresets and jobs
    from runner.process_info import JobCrossChecker
    jcc=JobCrossChecker()
    ss_ids=[]
    for ws_data in ws_list:
        ss=ScoreSet(
                ws=ws_data.ws,
                user=args.user_name,
                desc=ss_desc,
                )
        ss.save()
        ss_ids.append(ss.id)
        for variant_data in variant_list:
            settings=dict(settings_template)
            settings.update(ws_data.settings)
            if True:
                settings.update(variant_data.settings)
            else:
                # this was a temporary hack for plat2077, to allow the
                # variant data to adjust to which srcjobs are present
                # in which workspaces; it was left here in case it
                # might be needed again
                for k,v in six.iteritems(variant_data.settings):
                    if k.startswith('w_'):
                        stem = '_'.join(k.split('_')[1:-1])
                        if not 'srm_%s_srcjob'%stem in settings:
                            continue
                    settings[k] = v
            plugin = ws_data.jobname.split('_')[0]
            job_id = jcc.queue_job(
                    plugin,
                    ws_data.jobname,
                    user=args.user_name,
                    settings_json=json.dumps(settings),
                    )
            if job_id:
                ssj = ScoreSetJob(
                        scoreset=ss,
                        job_id=job_id,
                        job_type=variant_data.desc,
                        )
                ssj.save()
    print('ScoreSet ids:',','.join([str(x) for x in ss_ids]))
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
