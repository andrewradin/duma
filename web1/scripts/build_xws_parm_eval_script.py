#!/usr/bin/env python3

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

class SettingsParser:
    def __init__(self,plugin,ws_list,role=None):
        self._plugin = plugin
        self._ws_info = []
        from runner.process_info import JobInfo
        uji = JobInfo.get_unbound(self._plugin)
        class DummyClass:
            pass
        from runner.models import Process
        for ws in ws_list:
            jobnames = uji.get_jobnames(ws)
            qs = Process.objects.filter(
                    status=Process.status_vals.SUCCEEDED,
                    name__in=jobnames
                    )
            if role:
                qs = qs.filter(role=role)
            if qs.exists():
                p = qs.order_by('-id')[0]
                d=DummyClass()
                self._ws_info.append(d)
                d.ws = ws
                d.jobname = p.name
                d.settings = p.settings()
    def parse(self,f):
        # accumulate all settings across all workspaces
        all_settings={}
        for d in self._ws_info:
            for k in d.settings:
                all_settings.setdefault(k,[]).append(d.settings[k])
        # assume that the workspace-dependent setting are either:
        # - those with a different value in every workspace
        # - those for which the key only appears once
        ws_specific_keys=set()
        self._common={}
        from collections import Counter
        report=[]
        for k,v in six.iteritems(all_settings):
            if len(v) == 1 or len(set(v)) == len(self._ws_info):
                ws_specific_keys.add(k)
            else:
                ctr=Counter(v)
                winner=ctr.most_common(1)[0]
                self._common[k] = winner[0]
                report.append((k,repr(winner[0]),winner[1],len(v)))
        # report on common value extraction
        report.sort(key=lambda x:x[2],reverse=True)
        for r in report:
            f.write("%s:%s (%d/%d)\n" % r)
        # store ws-specific values with each ws_info record
        for d in self._ws_info:
            d.ws_settings={}
            for k in d.settings:
                if k in ws_specific_keys:
                    d.ws_settings[k] = d.settings[k]
    def _json(self,o):
        import json
        return json.dumps(o, separators=(',',':'), sort_keys=True)
    def write(self,f):
        f.write('%s\n'%self._plugin)
        f.write('%s\n'%self._json(self._common))
        f.write('\n')
        for d in self._ws_info:
            f.write('%d %s %s\n'%(d.ws.id,d.jobname,self._json(d.ws_settings)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='produce a preliminary xws_parm_eval script'
            )
    parser.add_argument('--ws-list',default='1,5,6,7,18,20,25,34,35,50')
    parser.add_argument('--role')
    parser.add_argument('plugin')
    args=parser.parse_args()

    from browse.models import Workspace
    ws_list = [
        Workspace.objects.get(pk=ws_id)
        for ws_id in args.ws_list.split(',')
        ]
    sp = SettingsParser(args.plugin,ws_list,args.role)
    sp.parse(sys.stderr)
    sp.write(sys.stdout)
