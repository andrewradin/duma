#!/usr/bin/env python3

from __future__ import print_function
import os, sys, getopt
from optparse import OptionParser
try:
    from path_helper import PathHelper,make_directory
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")
    from path_helper import PathHelper,make_directory

import os
import django

if not "DJANGO_SETTINGS_MODULE" in os.environ:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    django.setup()

from algorithms.exit_codes import ExitCoder
from browse.models import WsAnnotation

def useage(*objs):
    print("USEAGE: ", *objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def printOut(*objs):
    print(*objs, file=sys.stdout)


# get exit codes
exitCoder = ExitCoder()

opts = OptionParser()

usage = "usage: %prog [options] [input] Give this a WS ID and a file with canonical names and it will report the drugBank IDs."

opts = OptionParser(usage=usage)

opts.add_option("-i", help="Tab or space separated file of canonical names to convert to drugBank IDs")
opts.add_option("-w", help="ws_id")

options, arguments = opts.parse_args()

# return usage information if no argvs given

if len(sys.argv) < 2:

    opts.print_help()

    sys.exit(exitCoder.encode('usageError'))

ws_id = options.w

with open(options.i, 'r') as f:
    p_names = f.read().splitlines()
names = [x.strip().lower() for x in p_names]

allNames = {rec.agent.canonical.encode('utf-8').lower(): rec.agent.drugbank_id for rec in WsAnnotation.objects.filter(ws_id=ws_id)}
with open('canonicalNames.txt', 'w') as f:
    f.write("\n".join(list(allNames.keys())))

ids=[]
for d in names:
    if d in list(allNames.keys()):
        ids.append(allNames[d])
    else:
        warning(d, " not found in workspace")

printOut("\t".join(ids))
