#!/usr/bin/env python3

import os
import django
import django_setup
import json

def get_default_settings(ws_id, plugin):
    from runner.process_info import JobInfo
    from browse.models import Workspace
    ws = Workspace.objects.get(pk=ws_id)
    uji = JobInfo.get_unbound(plugin)
    settings = uji.settings_defaults(ws).get('default', {})
    return settings

def get_latest_refresh_settings(ws_id, plugin):
    from runner.process_info import JobInfo
    from browse.models import Workspace, ScoreSet

    for ss in ScoreSet.objects.filter(ws=ws_id).order_by('-id'):
        for jobtype, jid in ss.job_type_to_id_map().items():
            if jobtype.endswith('_' + plugin) or jobtype == plugin:
                return JobInfo.get_bound(ws_id, jid).parms
    
    return None

def launch_job(settings, plugin, ws_id, jobnames_filter=None):
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile() as tf:
        tmp_out = tf.name
        run('qa', plugin, settings, tmp_out, [ws_id], jobnames_filter=jobnames_filter)

        with open(tmp_out) as f:
            import json
            jid = json.loads(f.read())[0]['job']
            print("Queued job ", jid)
            return jid

class JobWaiter:
    def __init__(self, halt_on_failure):
        from collections import defaultdict
        self._active_jids = defaultdict(list)
        self._halt_on_failure = halt_on_failure

    def add_job(self, jid, on_complete=None):
        self._active_jids[jid].append({
            'on_complete': on_complete,
        })

    def wait_for_all(self):
        completed = {}
        failed = {}
        from runner.models import Process
        import time
        from tqdm import tqdm
        with tqdm(total=len(self._active_jids),
                  desc="Queued Jobs") as progress:
            while self._active_jids:
                time.sleep(5)
                active_jids = dict(self._active_jids)
                for jid, datas in active_jids.items():
                    job = Process.objects.get(pk=jid)
                    sv = Process.status_vals
                    if job.status not in {sv.QUEUED, sv.RUNNING}:
                        del self._active_jids[jid]

                        if job.status == sv.SUCCEEDED:
                            for data in datas:
                                if data['on_complete']:
                                    data['on_complete']()
                            completed[jid] = datas 
                        elif job.status == sv.FAILED:
                            if self._halt_on_failure:
                                self.cancel_all()
                            failed[jid] = datas
                        else:
                            raise Exception("Unhandled case")
                
                # Update the total in case we've added new jobs in between.
                progress.total = len(completed) + len(failed) + len(self._active_jids)
                # This adds to the total, so subtract the current value (n) from it.
                progress.update(len(completed) + len(failed) - progress.n)
        return completed

    def cancel_all(self):
        pass


def wait_for_jid(jid):
    from runner.models import Process
    import time
    while True:
        job = Process.objects.get(pk=jid)
        sv = Process.status_vals
        if job.status == sv.QUEUED or job.status == sv.RUNNING:
            time.sleep(5)
        else:
            break
    print("Completed job ", jid)

class JobQueuer:
    def __init__(self,user,plugin):
        self.plugin = plugin
        self.user = user
        from runner.process_info import JobInfo
        self.uji = JobInfo.get_unbound(self.plugin)
        from runner.process_info import JobCrossChecker
        self.jcc=JobCrossChecker()
        self.launches=[]
    def queue(self,ws_id,override_settings,jobnames_filter=None):
        from browse.models import Workspace
        ws = Workspace.objects.get(pk=ws_id)
        jobnames = [ x for x in self.uji.get_jobnames(ws) ]
        if len(jobnames) != 1:
            print("Trying to narrow down jobname from", jobnames)
            jobnames = [x for x in jobnames if self.plugin in x]
            print("Narrowed down to ", jobnames)
        if jobnames_filter:
            jobnames = [jobnames_filter(jobnames)]
        assert len(jobnames) == 1
        print("Running with jobname", jobnames[0])
        defaults = self.uji.settings_defaults(ws)
        if jobnames[0] in defaults:
            settings = defaults.get(jobnames[0])
        else:
            settings = defaults.get('default', {})
        #print("Default settings = ", settings)
        # add non-default fields
        settings['user'] = self.user
        settings['ws_id'] = ws.id
        settings.update(override_settings)
        # queue job and collect job id
        import json
        job_id = self.jcc.queue_job(self.plugin,jobnames[0],
                user=self.user,
                settings_json=json.dumps(settings),
                )
        self.launches.append({'ws': ws_id, 'job': job_id})


def run(user_name, plugin, settings, output, ws_ids, drive_background=True, jobnames_filter=None):
    # validate input
    from django.contrib.auth.models import User
    if not User.objects.filter(username=user_name).exists():
        raise ValueError("'%s' is not a valid user name"%user_name)

    rq = JobQueuer(user_name,plugin)

    override_settings = {}
    if settings:
        if isinstance(settings, dict):
            override_settings = settings
        else:
            override_settings = json.loads(open(settings).read())

    for ws_id in ws_ids:
        rq.queue(ws_id, override_settings, jobnames_filter=jobnames_filter)
        print(*rq.launches[-1])

    if output:
        with open(output, 'w') as f:
            f.write(json.dumps(rq.launches))

    if drive_background:
        from runner.models import Process
        Process.drive_background()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='''queue jobs from the commandline'''
            )
    parser.add_argument('-u', '--user_name')
    parser.add_argument('-p', '--plugin', required=True, help='Job plugin to run')
    parser.add_argument('-s', '--settings', help='Json file with override settings')
    parser.add_argument('-o', '--output', help='Where to write the jobids')
    parser.add_argument('ws_ids',nargs='+',type=int)
    args=parser.parse_args()
    run(**vars(args))
