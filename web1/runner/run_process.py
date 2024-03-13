#!/usr/bin/env python3

from __future__ import print_function
import sys
# retrieve the path where this program is located, and
# use it to construct the path of the website_root directory
sys.path.insert(0, sys.path[0]+"/..")

import os

import subprocess
import sys

class Operation:
    job_id_name = 'DUMA_JOB_ID'
    def __init__(self,abs_wd,job_id,abs_pidfile,logfile,cmd):
        self.job_id = job_id
        self.pidfile = abs_pidfile
        from runner.common import LogRepoInfo
        self.lri = LogRepoInfo(job_id)
        assert logfile == self.lri.log_path()
        self.logfile=logfile
        from path_helper import make_directory
        make_directory(os.path.dirname(self.logfile))
        self.cmd = cmd
        os.chdir(abs_wd)
        os.setpgrp()
        make_directory(os.path.dirname(abs_pidfile))
        with open(self.pidfile,'w') as f:
            f.write("%d\n" % (os.getpid(),))
    def syslog(self,s):
        subprocess.call([
                'logger',
                '-p','local0.info',
                '-t','job-'+self.job_id,
                s,
                ])
    def run(self):
        log = open(self.logfile,'w')
        null = open('/dev/null')
        import os
        env = dict(os.environ)
        env[self.job_id_name] = self.job_id
        try:
            return_code = subprocess.call(self.cmd
                                    ,shell=True
                                    ,stdin=null
                                    ,stdout=log
                                    ,stderr=subprocess.STDOUT
                                    ,env=env
                                    )
        # Note that stdout/stderr output generated below here (directly
        # or from the final LTS push) are sent to the fds set up by
        # Process.drive_background(): i.e. /var/log/drive_background.log
        except:
            # This should never happen (no matter what happens in the
            # subprocess, it only appears here as a return code, so
            # nothing should drive us into an error path). But this
            # was added just in case while trying to track down an odd error.
            self.syslog('got exception in subprocess.call')
            return_code = -1
        else:
            self.syslog('back from cmd with return_code %d'%return_code)
        try:
            self.lri.push_log()
            os.remove(self.pidfile)
        except Exception as ex:
            try:
                self.syslog("Exception %s" % ex)
                # We got an error on cleanup; if left uncaught, it would cause
                # the job to be marked as Failed (due to a non-zero return code)
                # but would leave no clue as to why in the job's logfile.
                #
                # Instead, send a slack notification with the failing job id,
                # and output the jobid and exception info to stdout, so they
                # get recorded along with any error output from LTS itself.
                msg = 'on %s cleanup error on job %s'%(
                        os.uname()[1],
                        self.job_id,
                        )
                self.syslog(msg)
                import traceback
                self.syslog(traceback.format_exc())
                print(msg)
                from path_helper import PathHelper
                slack = PathHelper.website_root+'scripts/slack_send.sh'
                subprocess.call([slack,msg+'; see drive_background.log'])
            except:
                # Something in the above handler block is failing.
                # Going to track it down, but until then, let's pass through
                # the return code.
                return return_code
        self.syslog("Returning %s"% return_code)
        return return_code
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='background process wrapper')
    parser.add_argument('abs_wd')
    parser.add_argument('job_id')
    parser.add_argument('abs_pidfile')
    parser.add_argument('log_file')
    parser.add_argument('cmd')
    parser.add_argument('args',metavar='arg',nargs='*')
    args=parser.parse_args()

    op = Operation(args.abs_wd
        , args.job_id
        , args.abs_pidfile
        , args.log_file
        , " ".join([args.cmd]+args.args)
        )
    import signal
    def handler(signum,frame):
        msg = "on %s pid %d (job %s) got signal %d" % (
                os.uname()[1],
                os.getpid(),
                op.job_id,
                signum
                )
        if False: # enable for testing aborts without spamming slack
            print(msg,file=sys.stderr)
            return
        from path_helper import PathHelper
        slack = PathHelper.website_root+'scripts/slack_send.sh'
        subprocess.call([slack,msg+'; see drive_background.log'])
    signal.signal(signal.SIGHUP,handler)
    exit(op.run())
