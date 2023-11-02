#!/usr/bin/python
from __future__ import print_function
import os, sys, django, re
from collections import defaultdict
from optparse import OptionParser
sys.path.insert(1,"../../")
sys.path.insert(1,"../../../databases/chembl")
from parseChEMBL import getCanonSmilesFromSmiles
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from algorithms.exit_codes import ExitCoder
from parse_PTC_trainingData import writePTCAtrFile

#==> testing_results.tsv <==
#ID Name MR FR MM FM
#1 Acetaminophen - - + +
#2 Acebutolol - - - -
#3 Acetohexamide - - - -
#
#==> testing_smiles.tsv <==
#1 Acetaminophen 103-90-2 c1cc(O)ccc1NC(=O)C
#2 Acebutolol 37517-30-9 c1c(ccc(OCC(O)NC(C)C)c1C(=O)C)NC(=O)CCC
#3 Acetohexamide 968-81-0 N(C(=O)NC1CCCCC1)S(=O)(=O)c2ccc(cc2)C(=O)C
#5 Acrivastine 87848-99-5 c1cc(C=CC(=O)O)nc(c1)C(=CCN2CCCC2)c3ccc(C)cc3

if __name__=='__main__':
    exitCoder = ExitCoder()
    
    opts = OptionParser()
    
    usage = "usage: %prog [options] [input] Parse PTC training data into an attribtues file"
    
    opts = OptionParser(usage=usage)
    
    opts.add_option("-t", help="testing_smiles.tsv")
    opts.add_option("-o", help="output file")
    opts.add_option("-r", help="testing_results.tsv")
    
    options, arguments = opts.parse_args()
    
    # return usage information if no argvs given
    
    if len(sys.argv) < 3:
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
            trid = "TE" + '0'*(3-len(fields[0])) + fields[0]
            d[trid]['cas']=fields[2]
            d[trid]['canonical']=fields[1]
            d[trid]['smiles_code'] = getCanonSmilesFromSmiles(fields[3])
    
    with open(options.r, 'r') as f:
        for line in f:
            if not line[0].isdigit():
                continue
            fields = line.rstrip().split("\t")
            trid = "TE" + '0'*(3-len(fields[0])) + fields[0]
            results = fields[2:]
            poss = 0
            for r in results:
                if r == '+':
                    poss += 1
            if float(poss)/len(results) > minPortion:
                d[trid]['PTC_tox'] = 'True'
            else:
                d[trid]['PTC_tox'] = 'False'
    
    writePTCAtrFile(options.o, d)
