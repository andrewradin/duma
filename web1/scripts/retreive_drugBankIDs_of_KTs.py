#!/usr/bin/python

from __future__ import print_function
import os, django, sys, getopt
from optparse import OptionParser
try:
    import path_helper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")
    import path_helper
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from browse.models import WsAnnotation
from algorithms.exit_codes import ExitCoder

# get exit codes
exitCoder = ExitCoder()

opts = OptionParser()

usage = "usage: %prog [options] [input] Give this a WS ID and this will print out the Drug Bank IDs for the KTs."

opts = OptionParser(usage=usage)

opts.add_option("-w", help="WorkSpace ID")

options, arguments = opts.parse_args()

# return usage information if no argvs given

if not options.w:

    opts.print_help()

    sys.exit(exitCoder.encode('usageError'))

ws_id = int(options.w)

enum = WsAnnotation.indication_vals
kt_codes = [enum.FDA_TREATMENT, enum.KNOWN_TREATMENT]
qs2= WsAnnotation.objects.filter(ws_id = ws_id, indication__in = kt_codes)
kt_db_ids = [str(x.agent.drugbank_id) for x in qs2]
print("\t".join(kt_db_ids))

