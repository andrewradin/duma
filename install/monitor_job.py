#!/usr/bin/env python

import sys
import subprocess
import os
exit_code = subprocess.call(sys.argv[1:])
if exit_code:
    msg='%s command failed on %s'%(sys.argv[1:],os.uname()[1])
    resp=subprocess.check_output([
            'curl',
            '-s',
            '-X','POST',
            '-H','Content-type: application/json',
            '--data','{"text":"%s"}'%msg,
            'https://hooks.slack.com/services/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'
            ])
