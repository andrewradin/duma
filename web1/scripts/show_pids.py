#!/usr/bin/env python3

# This script retrieves information on active jobs (or inactive jobs that
# were incompletely cleaned up). Most of the code should move to dtk, so
# it can eventually support a web page display of this information.
# 
# This is meant to be used after job aborts to support manual cleanup,
# and build up automated error handling and recovery.

def get_pid_ppid_pairs():
    import subprocess
    result = subprocess.run(
            ['ps','-e','-o','pid=','-o','ppid='],
            check=True,
            capture_output=True,
            encoding='utf8',
            )
    for line in result.stdout.split('\n')[:-1]:
        pid,ppid = line.split()
        yield (int(pid),int(ppid))

from dtk.lazy_loader import LazyLoader
class PidInfo(LazyLoader):
    _kwargs=['pid','job_id']
    def _job_loader(self):
        from runner.models import Process
        return Process.objects.get(pk=self.job_id)
    def _rm_loader(self):
        from runner.models import Load
        try:
            return Load.objects.get(job=self.job_id)
        except Load.DoesNotExist:
            return None
    def _proc_status_loader(self):
        try:
            f=open(f'/proc/{self.pid}/status')
        except FileNotFoundError:
            return {}
        result = {}
        for line in f:
            key,val = line.split(':')
            val = val.strip()
            result[key] = val
        return result
    @classmethod
    def dangling_log_set(cls,shard):
        from path_helper import PathHelper
        repo_path = PathHelper.lts+'log'+str(shard)
        import subprocess
        result = subprocess.run(
                ['git','status','--porcelain'],
                cwd=repo_path,
                check=True,
                capture_output=True,
                encoding='utf8',
                )
        dangling = set()
        for line in result.stdout.split('\n')[:-1]:
            flags,path = line.split()
            parts = path.split('/')
            dangling.add(int(parts[1]))
        return dangling
    def _lts_log_loader(self):
        dangling = self.dangling_log_set(self.job_id%10)
        return 'unclean' if self.job_id in dangling else ''
    @classmethod
    def quick_add_lts_log(cls,pidlist):
        cache = {}
        for pid in pidlist:
            shard = pid.job_id%10
            if shard not in cache:
                cache[shard] = cls.dangling_log_set(shard)
            if pid.job_id in cache[shard]:
                pid.lts_log = 'unclean'
            else:
                pid.lts_log = ''
    @property
    def job_status(self):
        return self.job.status_vals.get('label',self.job.status)
    @property
    def job_name(self): return self.job.name
    @property
    def rm_want(self): return self.rm.want if self.rm else ''
    @property
    def rm_got(self): return self.rm.got if self.rm else ''
    @property
    def ShdPnd(self): return self.proc_status.get('ShdPnd','')
    @property
    def SigBlk(self): return self.proc_status.get('SigBlk','')
    @property
    def SigIgn(self): return self.proc_status.get('SigIgn','')
    @property
    def SigCgt(self): return self.proc_status.get('SigCgt','')
    @classmethod
    def from_pidfiles(cls):
        from path_helper import PathHelper
        from dtk.files import scan_dir
        result = []
        for item in scan_dir(PathHelper.pidfiles,output=lambda x:x):
            job_id = int(item.filename)
            try:
                pid = int(open(item.full_path).read().strip())
            except ValueError:
                print('skipping invalid pidfile',item.full_path)
                continue
            result.append(PidInfo(pid=pid,job_id=job_id))
        return result
    @classmethod
    def add_descendants(cls,pidlist):
        from dtk.data import MultiMap
        pid2ppid = MultiMap(get_pid_ppid_pairs())
        for pid_info in pidlist:
            pid_info.descendants = '\n'.join(
                    format_pid_tree(pid_info.pid,pid2ppid.rev_map())
                    )

def cmd_of_pid(pid):
    try:
        data=open(f'/proc/{pid}/cmdline').read()
    except FileNotFoundError:
        return ''
    parts = data.split('\0')
    if parts[0] == 'python3':
        parts=parts[1:]
    return parts[0].split('/')[-1]

def format_pid_tree(pid,ppid2pids):
    cmd = cmd_of_pid(pid)
    if not cmd:
        return [f'({str(pid)})']
    result = [str(pid)+' '+cmd]
    for child in ppid2pids.get(pid,set()):
        for x in format_pid_tree(child,ppid2pids):
            result.append('>'+x)
    return result
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='report on pidfile content',
            )
    parser.add_argument('-l',
            action='store_true',
            help='show list of column names and exit',
            )
    parser.add_argument('col',
            nargs='*',
            help='columns to display',
            )
    args=parser.parse_args()

    import os
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    import django
    django.setup()

    default_cols = ['pid','job_id',
            'job_name','job_status',
            'rm_want','rm_got',
            'descendants',
            'lts_log',
            ]
    other_cols = [
            'ShdPnd', 'SigBlk', 'SigIgn', 'SigCgt',
            ]
    if args.l:
        print(' '.join(sorted(default_cols+other_cols)))
        import sys
        sys.exit(0)
    cols = args.col or default_cols
    pids = PidInfo.from_pidfiles()
    print('found',len(pids),'pidfiles')
    PidInfo.add_descendants(pids)
    if 'lts_log' in cols:
        PidInfo.quick_add_lts_log(pids)
    rows = [cols]+[
            [str(getattr(pid,col)) for col in cols]
            for pid in pids
            ]
    from dtk.text import print_table,split_multi_lines,ljustify
    rows = split_multi_lines(rows)
    try:
        ljustify(rows,rows[0].index('descendants'))
    except ValueError:
        pass
    print_table(rows)
