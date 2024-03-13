#!/usr/bin/python

from __future__ import print_function
import os, django, sys, getopt
from optparse import OptionParser
try:
    from algorithms.exit_codes import ExitCoder
except ImportError:
    sys.path.insert(1,"../") # up one dir to web1
    from algorithms.exit_codes import ExitCoder
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
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

usage = "usage: %prog [options] [input] Give this a WS ID and a file with drug bank IDs and it will report the workSpace annotation IDs."

opts = OptionParser(usage=usage)

opts.add_option("-w", help="WorkSpace ID")
opts.add_option("-i", help="Tab or space separated file of drug bank IDs to convert to workSpace annotations")

options, arguments = opts.parse_args()

# return usage information if no argvs given

if len(sys.argv) < 2:

    opts.print_help()

    sys.exit(exitCoder.encode('usageError'))

ws_id = int(options.w)

with open(options.i, 'r') as f:
    drugBankIds = f.readline().rstrip().split()

allNames = {str(rec.agent.drugbank_id): rec.agent.canonical for rec in WsAnnotation.objects.filter(ws_id=ws_id)}
ids=[]
for d in drugBankIds:
    if d in list(allNames.keys()):
        ids.append(allNames[d])
    else:
        warning(d, " not found in workspace")

printOut("\n".join(ids))
