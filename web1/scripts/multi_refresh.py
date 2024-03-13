#!/usr/bin/env python3

import sys
from path_helper import PathHelper

import os
import django

import django_setup

def warning(*args):
    print(*args,file=sys.stderr)

# Notes on integrating evaluation portion:
# - start with high-level eval script, as that will drive the interface to
#   the single-workspace compare portion
#   - this will need two sets of workspace-runset maps, so this information
#     should get saved in a JSON file rather than just printed
# - the initial cut will just focus on eval metrics applied to runset
#   scores
# - eventually there will be other evaluation components:
#   - some may be workspace-independent; for example, this may want to
#     incorporate databases/matching/diffstats.py, and extend it to
#     other databases besides drug collections; this doesn't fit well
#     into the current model because it needs access to both the old
#     and new files; this might require us to stash the signature
#     data for certain files along with each snapshot
#   - maybe this means each snapshot should be represented as a directory,
#     which can hold the JSON output from the refresh, and eventually
#     other data as well
# - alternatively, this script could output a JSON file with a fixed
#   name, or a name from the command line; that would maximize the
#   flexibility of the snapshot management process; similarly, the compare
#   process could take a pair of these JSON files and produce the
#   report output
# - final output will be:
#   - overall score change for each workspace (maybe summarized by
#     min, max, avg?)
#   - any outlier changes (more than n% up or down?) in individual scores

class RefreshQueuer:
    def __init__(self,user,resume_info):
        self.plugin = 'wf'
        self.user = user
        self.resume_info = resume_info
        from runner.process_info import JobInfo
        self.uji = JobInfo.get_unbound(self.plugin)
        from runner.process_info import JobCrossChecker
        self.jcc=JobCrossChecker()
        self.launches=[]
    def queue(self,ws_id):
        from browse.models import Workspace
        ws = Workspace.objects.get(pk=ws_id)
        # extract default settings for refresh workflow in this workspace
        jobnames = [
                x
                for x in self.uji.get_jobnames(ws)
                if x.endswith('RefreshFlow')
                ]
        assert len(jobnames) == 1
        FormClass=self.uji._get_form_class(jobnames[0])
        uform = FormClass()
        settings = {fld.name:fld.field.initial for fld in uform}
        # add non-default fields
        settings['user'] = self.user
        settings['ws_id'] = ws.id
        settings['eval_testtrain'] = True
        try:
            settings['resume_scoreset_id']=self.resume_info[ws.id]
        except KeyError:
            pass
        # queue job and collect job id
        import json
        job_id = self.jcc.queue_job(self.plugin,jobnames[0],
                user=self.user,
                settings_json=json.dumps(settings),
                )
        self.launches.append((ws_id,job_id))

def warn(msg=None):
    if msg:
        print(msg)
    answer = input('Proceed (y/n)? ')
    if answer.lower() in ('y','yes'):
        return
    import sys
    sys.exit(1)

def check_instance_type(hostname,want):
    from aws_op import Machine
    if hostname not in Machine.name_index:
        return
    try:
        mch = Machine.name_index[hostname]
    except KeyError:
        return # ignore if not an AWS instance
    i = mch.get_ec2_instance()
    got = i.instance_type
    # XXX We could load specs from the Machine.instance_properties table.
    # XXX This would let us provide more detail, and also bypass the warning
    # XXX if the current type exceeded the wanted type specs.
    if got != want:
        which = 'worker' if 'worker' in hostname else 'current'
        warn(f'Your {which} instance type is {got}. {want} is recommended.')

def check_disabled_parts(ws_ids):
    from browse.models import Workspace
    from workflows.refresh_workflow import StandardWorkflow
    ws_msgs = []
    for ws_id in ws_ids:
        ws = Workspace.objects.get(pk=ws_id)
        swf = StandardWorkflow(ws=ws)
        enabled = set(swf.get_refresh_part_initial())
        disabled = [
                l
                for i,l in swf.get_refresh_part_choices()
                if i not in enabled
                ]
        if disabled:
            ws_msgs.append(f'{ws.name}({ws.id}): {", ".join(disabled)}')
    if ws_msgs:
        sep = "\n "
        warn(f'''
The following refresh steps are disabled:
 {sep.join(ws_msgs)}
''')

def do_pre_checks(ws_ids):
    # verify machine sizes
    import os
    hostname = os.uname()[1]
    check_instance_type(hostname,'r6i.xlarge')
    worker = PathHelper.cfg('worker_machine_name')
    check_instance_type(worker,'m6a.16xlarge')
    check_disabled_parts(ws_ids)

normal_workspaces= [
     1, # T2D
     4, # SLE
     7, # Alzheimers
     11, # T1D
     12, # Parkinsons
     20, # Glaucoma
     21, # Sjogren Syndrome
     28, # ALS
     33, # IBS
     43, # IPF
     48, # Psoriasis
     52, # Psoriatic Arthritis
     53, # NASH
     61, # Ankylosing
     72, # Alopecia Areata
     73, # Scleroderma
     81, # NF1
     88, # Diabetic retin
     99, # Atopic Dermatitis
     105, # CKD
     115, # DM1
     119, # PSC
     158, # Liver Cirhosis
     326, # Endometriosis
     ]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='''queue workflow refresh jobs
            '''
            )
    parser.add_argument('--queue-only',action='store_true')
    parser.add_argument('--no-checks',action='store_true',
            help='Bypass interactive warnings',
            )
    parser.add_argument('--default-workspaces',action='store_true',
            help='''Run on normal workspaces used for multi-compare:
            %s
            '''%' '.join([str(x) for x in normal_workspaces]),
            )
    parser.add_argument('--result-json-file',
            help='''on the first run for a workspace, use this parameter
            to capture the RefreshWorkflow job id, in a format suitable
            for input to multi_compare.py
            ''',
            )
    parser.add_argument('--resume-json-file',
            help='''if a refresh doesn't run to completion, use this
            parameter to resume processing the same runset. The input
            file is in the same format as the result-json-file.
            Unused workspace ids are ignored. Workspace ids appearing
            on the command line but not in the file are run from the
            beginning.
            ''',
            )
    parser.add_argument('user_name')
    parser.add_argument('ws_id',nargs='*',type=int)
    args=parser.parse_args()
    # XXX One alternative to the above is, if a resume file is
    # XXX not specified, automatically generate a result file
    # XXX with a timestamped name. I didn't do this to avoid
    # XXX hard-coding a naming convention before we've discussed
    # XXX it.

    # validate input
    from django.contrib.auth.models import User
    if not User.objects.filter(username=args.user_name).exists():
        raise ValueError("'%s' is not a valid user name"%args.user_name)

    if args.default_workspaces:
        if args.ws_id:
            raise ValueError(
                "--default-workspaces and ws_ids can't be used together",
                )
        args.ws_id=normal_workspaces
    else:
        if not args.ws_id:
            raise ValueError(
                "either specify ws_ids or --default-workspaces",
                )

    if not args.no_checks:
        do_pre_checks(args.ws_id)

    if args.resume_json_file:
        from dtk.refresh_group import RefreshGroup
        rg = RefreshGroup(args.resume_json_file)
        resume_info = rg.resume_info()
    else:
        resume_info = {}

    rq = RefreshQueuer(args.user_name,resume_info)
    for ws_id in args.ws_id:
        rq.queue(ws_id)
        print(*rq.launches[-1])

    if args.result_json_file:
        import json
        json.dump(
                [{'ws_id':x[0],'wf_job':x[1]} for x in rq.launches],
                open(args.result_json_file,'w'),
                )

    if not args.queue_only:
        from runner.models import Process
        err = Process.prod_user_error()
        if err:
            warning(err)
            warning("Your work has been queued, but not started.")
            warning("To start, run:")
            warning("  sudo -u www-data scripts/queue_jobs.py - </dev/null")
            sys.exit(1)
        Process.drive_background()
