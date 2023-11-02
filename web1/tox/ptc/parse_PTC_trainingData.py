#!/usr/bin/python
from __future__ import print_function
import os, sys, django, re
from collections import defaultdict
from optparse import OptionParser
try:
    from dtk.mol_struct import getCanonSmilesFromSmiles
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+'/../..')
    from dtk.mol_struct import getCanonSmilesFromSmiles
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from algorithms.exit_codes import ExitCoder

def writePTCAtrFile(outFile, d):
    with open(outFile, 'w') as f:
        f.write("PTC_id\tattribute\tvalue\n")
        for k,v in d.items():
            for k2, v in v.items():
                if v is not None and v != 'None':
                    f.write("\t".join([k, k2, v]) + "\n")


if __name__=='__main__':
    #==> training_corrected_toxResults.txt <==
    #TR000  MR=P, FR=N, MM=P, FM=P
    #TR001  MR=P, FR=P, MM=P, FM=P
    #TR002  MR=N, FR=N, MM=P, FM=P
    #
    exitCoder = ExitCoder()
    
    opts = OptionParser()
    
    usage = "usage: %prog [options] [input] Parse PTC training data into an attribtues file"
    
    opts = OptionParser(usage=usage)
    
    opts.add_option("-t", help="training_cas_name_tr.tsv")
    opts.add_option("-s", help="training_corrected_smiles.txt")
    opts.add_option("-r", help="training_corrected_toxResults.txt")
    opts.add_option("-o", help="output file")
    
    options, arguments = opts.parse_args()
    
    # return usage information if no argvs given
    
    if len(sys.argv) < 4:
        opts.print_help()
        sys.exit(exitCoder.encode('usageError'))
    
    #===============================================================================
    # harcoded Params
    #===============================================================================
    minPortion = 0.5 # what portion of the rats and mice need to show tox to be a positive
    #===============================================================================
    # main
    #===============================================================================
    d = defaultdict(dict)
    with open(options.t, 'r') as f:
        for line in f:
            fields = line.rstrip().split("\t")
            trid = re.sub("-", '', fields[2])
            d[trid]['cas']=fields[0]
            d[trid]['canonical']=fields[1]
    
    with open(options.s, 'r') as f:
        for line in f:
            fields = line.rstrip().split("\t")
            d[fields[0]]['smiles_code'] = getCanonSmilesFromSmiles(fields[1])
    
    #TR000  MR=P, FR=N, MM=P, FM=P
    positiveHits = ['P', 'CE', 'SE']
    with open(options.r, 'r') as f:
        for line in f:
            fields = line.rstrip().split()
            results = []
            poss = 0
            for field in fields[1:]:
                results.append(field.rstrip(",").split("=")[1])
                if results[-1] in positiveHits:
                    poss += 1
            if float(poss)/len(results) >= minPortion:
                d[fields[0]]['PTC_tox'] = 'True'
            else:
                d[fields[0]]['PTC_tox'] = 'False'
    
    writePTCAtrFile(options.o, d)
