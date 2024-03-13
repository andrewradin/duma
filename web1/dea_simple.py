#!/usr/bin/python
from __future__ import print_function
import sys, getopt, django, argparse, subprocess, os, random
import numpy as np
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from path_helper import PathHelper,make_directory
from algorithms.exit_codes import ExitCoder
import dea

# created 25.Jan.2016 - Aaron C Daugherty
# This takes a list of scores and runs a GSEA-like analysis

def useage(*objs):
    print("USEAGE: ", *objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)
    
def printOut(*objs):
    print(*objs, file=sys.stdout)

def retreiveAllDrugsFromFile (inputFile):
    l = []
    with open(inputFile, 'r') as f:
        for line in f:
            l.append(line.rstrip().split())
    # l is a list where each entry is a list of the drug ID, and the score
    # and because this is osrted and defines the BG we don't have to do anything else
    l.sort(key=lambda x: float(x[1]), reverse = True)
    return(l)

if __name__=='__main__':
    #=================================================
    # Read in the arguments/define options
    #=================================================
    # get exit codes
    exitCoder = ExitCoder()
    
    arguments = argparse.ArgumentParser(description="A general ranked list enrichment program.")
    
    arguments.add_argument("-i", help="input file with 2 columns: drug IDs and scores")
    
    arguments.add_argument("-l", help="Single line, white space separated file of drug IDs to check for enrichment.")
    
    arguments.add_argument("-p", type=int, default=100000, help="How many permutations to run for the background. (default %(default)s)")
    
    arguments.add_argument("-w", type=int, default=1, help="How to weight the score.(default %(default)s).")
    
    args = arguments.parse_args()
    
    # return usage information if no argvs given
    
    if not args.i or not args.l:
        arguments.print_help()
        sys.exit(exitCoder.encode('usageError'))
    
    #================================================================================
    ##### INPUTS AND OUTPUTS #####
    #================================================================================
    # name input and outputs
    wholeWorkspaceDescrip = str(os.path.splitext(os.path.basename(args.i))[0])
    drugSetDescrip = str(os.path.splitext(os.path.basename(args.l))[0])
    
    #================================================================================
    # Retreive the necessary data
    #================================================================================
    # get the proper background. In this cae that is all drugs in the input file
    # also set up the output filehandle depending on the background
    fileHandle = os.getcwd() + "/" + drugSetDescrip + "_enrichmentIn_" + wholeWorkspaceDescrip
    
    # the first thing we want to do is read in the input file
    # doesn't matter if they're strings, numbers, whatever, as long as they're unique
    # Only take into account the drugs in the background
    warning("Retreiving data")
    drugScoreList = retreiveAllDrugsFromFile(args.i)
    with open(args.l, 'r') as f:
        drugSet = f.readline().rstrip().split()
    
    #===========================================================================
    # Run the algorithm and write results
    #===========================================================================
    run = dea.Runner(
                weight = args.w,
                nPermuts = args.p,
                fileHandle=fileHandle,
                png=True,
                score_list = drugScoreList,
                set_oi = drugSet,
                alpha = 0.01
                )
    run.run()
