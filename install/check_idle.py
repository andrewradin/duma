#!/usr/bin/env python
# ex: set tabstop=4 expandtab:

import subprocess

# This script is meant to be invoked by cron, and to shut the system down
# if it's not being used.  Conceptually, the system is unused if all the
# following are satisfied:
# - it's been up some minimum amount of time (so it doesn't shut down
#   immediately after launch)
# - no background jobs are running
# - it's been some minimum amount of time since the last background job
#   completed
# - nobody is logged in, or any logged-in users have been idle for some
#   minimum amount of time
#
# The first can be determined by checking /proc/uptime.
# The second can be determined by ps (after filtering out system daemons,
# the infrastructure for this script, and any idle login shells).
# The third requires some kind of timestamp file explicitly touched by
# the background process.  Since we're doing this anyway, we could also
# touch the timestamp file at the start of the background run, and have it
# be a proxy for requirement 2.
# The last can be determined by checking the output of 'w'.

up_reasons = []
min_login_idle = 12 # hours
min_up = 5 # minutes
min_timestamp = 15 # minutes
timestamp='/tmp/timestamp'
renew=True

# This script also warns if the machine has been up for an extended period
# Usually that means something has gone wrong with this script. Typically
# that's a new service that needs to be whitelisted below.
#
# We start with warning if the machine has been up for more than a day, and
# will do 4x/day after that until it is restarted.
warn_uptime_file='/tmp/warn_uptime'
one_day_secs = 24 * 60 * 60

def get_uptime():
    with open('/proc/uptime', 'r') as f:
        return float(f.readline().split()[0])

def get_warn_uptime():
    try:
        with open(warn_uptime_file, 'r') as f:
            return float(f.readline().strip())
    except:
        return one_day_secs

def write_next_warning(uptime_amt):
    with open(warn_uptime_file, 'w') as f:
        f.write(str(uptime_amt))

def warn_about_uptime(uptime_amt):
    try:
        import platform
        import os
        hostname = platform.node()
        msg = 'Machine %s has been up for %.1f hours' % (hostname, uptime_amt / 60 / 60)
        scriptdir = os.path.dirname(os.path.abspath(__file__))
        slack_send = os.path.join(scriptdir, '..', 'web1', 'scripts', 'slack_send.sh')
        subprocess.check_call([slack_send, msg])
    except:
        import traceback as tb
        tb.print_exc()



uptime = get_uptime()
warn_uptime = get_warn_uptime()
print("Machine has been up for %.1f hours, next warning at %.1f hours" % (
    uptime / 60 / 60,
    warn_uptime / 60 / 60
    ))

if uptime > warn_uptime:
    warn_about_uptime(uptime)
    # Warn again in a bit.
    write_next_warning(uptime + one_day_secs // 4)
elif uptime < one_day_secs and warn_uptime > one_day_secs:
    # Reset the warning back to 1 day.
    write_next_warning(one_day_secs)


date = subprocess.check_output(['date'], encoding='utf8').strip()

import re
lines = subprocess.check_output(['ps','h'
        ,'-N','-u','root,syslog,daemon,messagebus,mysql,uuidd'
        ,'-o','user,cmd'], encoding='utf8').split('\n')
for line in lines:
    if line:
        if re.match("^ubuntu +ps ",line):
            continue
        if re.match("^ubuntu +/bin/sh -c /home/.*install/check_.*\.py.*",line):
            continue
        if re.match("^ubuntu +.*python(|3) /home/.*install/check_.*\.py.*",line):
            continue
        if re.match("^ubuntu +sshd:",line):
            continue
        if re.match("^ubuntu +-bash",line):
            continue
        if re.match("^ubuntu +/bin/bash$",line):
            continue
        if re.match("^ubuntu +SCREEN$",line):
            continue
        if re.match("^ubuntu +/lib/systemd/systemd --user$",line):
            continue
        if re.match("^ubuntu +\(sd-pam\)$",line):
            continue
        if re.match("^systemd\+ +/lib/systemd/systemd-timesyncd$",line):
            continue
        if re.match("^systemd\+ +/lib/systemd/systemd-networkd$",line):
            continue
        if re.match("^systemd\+ +/lib/systemd/systemd-resolved$",line):
            continue
        if re.match(".*neo4j.*",line):
            continue
        up_reasons.append("active process:"+line)
        if re.search("check_disk",line):
            # This runs all the time, if we reset the idle timestamp every time it does, we wouldn't shut down.
            renew=False
### from cron
#ubuntu   /bin/sh -c /home/ubuntu/2xar/twoxar-demo/install/check_idle.py >> /tmp/idle 2>&1
#ubuntu   python /home/ubuntu/2xar/twoxar-demo/install/check_idle.py
#ubuntu   ps h -N -u root,syslog,daemon,messagebus -o user,cmd
#
### from shell (idle checked below)
#ubuntu   sshd: ubuntu@pts/0
#ubuntu   -bash
#
### from screen
#ubuntu   SCREEN
#ubuntu   /bin/bash


#http://www.cyberciti.biz/faq/linux-unix-login-bash-shell-force-time-outs/

lines = subprocess.check_output(['w','-h'], encoding='utf8').split('\n')
# ['ubuntu   pts/0    108-221-17-89.li 06:40    4.00s  0.08s  0.00s w -h', '']
# options for idle field:
#   [# ]#.##s - seconds w/fraction
#   [# ]#:## - minutes and seconds
#   [# ]#:##m - hours and minutes
#   [# ]#days - days
if lines:
    for line in lines:
        if line:
            fields=line.split()
            #print fields[4]
            if fields[4].endswith('days'):
                continue # idle too long
            if fields[4].endswith('m'):
                subfields = fields[4].split(":")
                if int(subfields[0]) > min_login_idle:
                    continue # idle too long
            up_reasons.append("%s logged in from %s, idle %s" % (
                        fields[0],fields[2],fields[4]
                        ))

# if we're going to keep running for either of the first two reasons,
# touch the timestamp file so we get a grace period after the condition
# goes away
if up_reasons and renew:
    subprocess.call(['touch',timestamp])

try:
    lines = subprocess.check_output([
            'grep',
            'sshd:session.*closed',
            '/var/log/auth.log',
            ], encoding='utf8').split('\n')
    tstr = ' '.join(lines[-2].split()[0:3])
    import datetime
    now = datetime.datetime.now()
    chk_date = '%d %s'%(now.year,tstr)
    last_logout = datetime.datetime.strptime(chk_date,'%Y %b %d %H:%M:%S')
    if last_logout > now:
        last_logout = datetime.datetime(
                last_logout.year-1,
                last_logout.month,
                last_logout.day,
                last_logout.hour,
                last_logout.minute,
                last_logout.second,
                )
    mins_ago = int((now-last_logout).total_seconds())/60
    if mins_ago < min_timestamp:
        up_reasons.append("logout %s within %d minutes" % (tstr,mins_ago))
except:
    # This can fail if the line didn't show up in the log, which is fine.
    import traceback as tb
    tb.print_exc()

try:
    fields = subprocess.check_output(['cat','/proc/uptime'], encoding='utf8').split()
    uptime = float(fields[0])
    #print "uptime",uptime
    if uptime < min_up * 60:
        up_reasons.append("up time %f < %d minutes" % (uptime,min_up))
except:
    import traceback as tb
    tb.print_exc()

import os
import time
try:
    (mode, ino, dev, nlink
        , uid, gid, size
        , atime, mtime, ctime) = os.stat(timestamp)
    now = time.time()
    idle = now-mtime
    #print now,mtime,idle
    if idle < min_timestamp*60:
        up_reasons.append("idle time %f < %d minutes" % (idle,min_timestamp))
except:
    import traceback as tb
    tb.print_exc()

if up_reasons:
    print(date)
    print('\n'.join((' > %s' % x) for x in up_reasons))
else:
    print(date, "shutting down")
    subprocess.check_call(['sudo','halt','-p'])
