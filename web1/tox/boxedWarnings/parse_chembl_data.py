#!/usr/bin/python
from __future__ import print_function
import os, sys, django
from optparse import OptionParser
sys.path.insert(1,"../../")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from algorithms.exit_codes import ExitCoder

if __name__=='__main__':
    exitCoder = ExitCoder()
    
    opts = OptionParser()
    
    usage = "usage: %prog [options] [input] Parse chembl attribute data for only those IDs with a max phase of 4"
    
    opts = OptionParser(usage=usage)
    
    opts.add_option("-i", help="List of chembl IDs to report")
    opts.add_option("-c", help="Full ChEMBL file to report/filter from")
    
    options, arguments = opts.parse_args()
    
    # return usage information if no argvs given
    
    if len(sys.argv) < 2:
        opts.print_help()
        sys.exit(exitCoder.encode('usageError'))
    
    #===============================================================================
    # main
    #===============================================================================
    with open(options.i, 'r') as f:
        ids = [line.rstrip() for line in f]
    
    with open(options.c, 'r') as f:
        for line in f:
            fields = line.rstrip().split()
            if fields[0] in ids:
                print(line.rstrip())
