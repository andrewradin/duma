#!/usr/bin/env python3

# A constant-space parser for the ChEBI OBO format
# written by Aaron Daugherty, twoXAR, Novemeber 7, 2015

from collections import defaultdict
import argparse
import os
import sys
from optparse import OptionParser



#=================================================
# Define functions
#=================================================

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def useage(*objs):
    print("USEAGE: ", *objs, file=sys.stderr)
    
def printOut(*objs):
    print(*objs, file=sys.stdout)

def processTerm(chebiTerm):
    # this process an indiivudal term
    # In: an object representing a ChEBI term, replace single-element lists with their only member.
    # Returns: the modified object as a dictionary.
    
    ret = dict(chebiTerm) # Input is a defaultdict
    for key, value in ret.items():
        if len(value) == 1:
            ret[key] = value[0]
    return ret

def parseOBO(filename):
    # this process an OBO file from ChEBI
    # In: the file name.
    # Yields: the modified object as a dictionary.
    
    
    with open(filename, "r") as infile:
        currentTerm = None
        for line in infile:
            line = line.strip()
            if not line: continue #Skip empty
            if line == "[Term]":
                if currentTerm: yield processTerm(currentTerm)
                currentTerm = defaultdict(list)
            elif line == "[Typedef]":
                #Skip [Typedef sections]
                currentTerm = None
            else: #Not [Term]
                #Only process if we're inside a [Term] environment
                if currentTerm is None: continue
                key, sep, val = line.partition(":")
                currentTerm[key].append(val.strip())
        #Add last term
        if currentTerm is not None:
            yield processTerm(currentTerm)

if __name__ == "__main__":
    # define options
    
    opts = OptionParser()
    
    usage = "usage: %prog [options] [inputs] T"
    
    opts = OptionParser(usage=usage)
    
    opts.add_option("-f", help="<oboFile> input file")
    opts.add_option("-t", help="<starting term> CHEBI:<int>")
    
    options, arguments = opts.parse_args()
    
    # return usage information if no argvs given
    
    if len(sys.argv) < 2:
        
        opts.print_help()
        
        sys.exit()
    
    
    # we'll end up walking the tree of this ontology going only down from the provided term
    termsToSearchFor = [options.t]
    termsToReport = {}
    for searchTerm in termsToSearchFor: # this is a little risky b/c we'll be adding to this list and then iterating over those, but it seems to be working
        #warning("Searching for all childern of " + searchTerm)
        for term in parseOBO(options.f):
            if 'is_a' in list(term.keys()) :
                if term['is_a'] == searchTerm :
                    termsToReport[term['id']] = 1
                    termsToSearchFor.append(term['id']) # we might want to also include alt_id
    #
    # Now print out all the relevant child nodes and leaves
    for term in parseOBO(options.f):
        if term['id'] in list(termsToReport.keys()):
            for k,v in term.items():
                if not isinstance(v, str):
                    for val in v:
                        printOut(k + ":" + val)
                else:
                    printOut(k + ":" + v)
