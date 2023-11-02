#!/usr/bin/python
from __future__ import print_function
import sys, getopt, os, re, django
from optparse import OptionParser
from collections import defaultdict
sys.path.insert(1,"../")
sys.path.insert(1, "../../")
import tox_funcs
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from path_helper import PathHelper
from algorithms.exit_codes import ExitCoder

# created 10.Feb.2016 - Aaron C Daugherty - twoXAR

# Written to match PTC drugs to ChEMBL drugs via smiles codes, or more precisely having a fingerprint similarity of 1, using RDKit

# TO DO:


def message(*objs):
    print(*objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def verboseOut(*objs):
    if options.verbose:
        print(*objs, file=sys.stderr)

def useage(*objs):
    print("USEAGE: ", *objs, file=sys.stderr)

def printOut(*objs):
    print(*objs, file=sys.stdout)

# read though the attributes file, and only pull out the lines with the smiles_codes
def parseAttributeFileWithSmiles(inFile):
    attributes = ['smiles_code', 'canonical']
    smiles = {}
    names = {}
    with open(inFile, 'r') as f:
        for line in f:
            fields = line.split("\t")
            if fields[1] not in attributes:
                continue
            if fields[1] == 'smiles_code':
                smiles[fields[0]] = fields[2]
            elif fields[1] == 'canonical':
                names[fields[0]] = fields[2].lower()
    return names, smiles

def getChemblSmiles(chemblFileName):
    chemblFile = tox_funcs.fetchDrugsetFile(chemblFileName)
    return parseAttributeFileWithSmiles(chemblFile)

# this takes the first match it finds, so if there are multiple that could be an issue
def matchStrings(firstD, secondD):
    matches = {}
    notMatched = []
    for id1,str1 in firstD.items():
        matched = False
        if str1 in list(secondD.values()):
            for id2,str2 in secondD.items():
                if str1 == str2:
                    matches[id1] = id2
                    matched = True
                    break
        if not matched:
            notMatched.append(id1)
    return matches, notMatched

if __name__=='__main__':
    #=================================================
    # Read in the arguments/define options
    #=================================================
    exitCoder = ExitCoder()
    
    opts = OptionParser()
    
    usage = "usage: %prog [options] [input] This will match PTC drugs to ChEMBL IDs via SMILES codes"
    
    opts = OptionParser(usage=usage)
    
    opts.add_option("-o", help="Output file.")
    
    opts.add_option("-i", help="PTC attribute file to match.")
    
    opts.add_option("--morgan_fingerprint_size", help="Radius of fingerprint.  2 is equivalent to ECFP 4, which is out default. [DEAFULT: 2]", type=int, default = 2)
    
    opts.add_option("-v", action="store_true", dest="verbose", help="Print out status reports")
    
    options, arguments = opts.parse_args()
    
    # return usage information if no argvs given
    
    if not options.o or not options.i:
        opts.print_help()
        sys.exit(exitCoder.encode('usageError'))
    
    
    #========================================================================================
    # currently hardcoded parameters
    #========================================================================================
    morganFingerprintSize = options.morgan_fingerprint_size # 2 is equivalent to ECFP 4
    chemblFileName = 'create.chembl.full.tsv'
    
    #========================================================================================
    # main
    #========================================================================================
    ptcNames, ptcSmiles = parseAttributeFileWithSmiles(options.i)
    verboseOut("Loaded", len(list(ptcNames.keys())), len(list(ptcSmiles.keys())), "molecules from PTC by canoncial name and smiles, respectively")
    chemblNames, chemblSmiles = getChemblSmiles(chemblFileName)
    verboseOut("Loaded", len(list(chemblNames.keys())), len(list(chemblSmiles.keys())), "molecules from ChEMBL by canonical name and smiles, respectively")
    
    # there aren't that many drugs in the PTC set, so I'll jsut brute force it
    matches, notMatched = matchStrings(ptcNames, chemblNames)
    verboseOut("Found matches using canonical names for ", len(matches), " of ", len(list(ptcNames.keys())))
    
    additionalMatches, stillNotMatched = matchStrings({k:v for k,v in ptcSmiles.items() if k not in list(matches.keys())}, chemblSmiles)
    verboseOut("Found matches using fingerprints for ", len(additionalMatches), " of ", len(notMatched), "that were not matched before, leaving ", len(stillNotMatched), "with no match")
    
    # combine the 2
    matches.update(additionalMatches)
    with open(options.o, 'w') as f:
        f.write("PTC_id\tattribute\tvalue\n")
        f.write("\n".join(["\t".join([k, "chembl_id", v]) for k,v in matches.items()]) + "\n")

