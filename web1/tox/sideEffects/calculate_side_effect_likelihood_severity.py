#!/usr/bin/python
from __future__ import print_function
import os, django, sys, argparse
import pandas as pd
from collections import defaultdict,Counter
sys.path.insert(1,"../..")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from path_helper import PathHelper
from algorithms.exit_codes import ExitCoder
import calculate_side_effect_prevalence_severity as cseps

# created 22.Feb.2016 - Aaron C Daugherty - twoXAR

# TO DO:
#  If files aren't present, call code to make them
#  Connect ULMS to severity

# this differes from the calculate_side_effect_severity_score.py in that it works with data from
# offsides, and as such rather than using side effect prevelance, works with the increased liklihood
# that a drug causes a side effect

# A few subroutines to print out as I want

def message(*objs):
    print(*objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def verboseOut(*objs):
    if args.verbose:
        print(*objs, file=sys.stderr)

def useage(*objs):
    print("USEAGE: ", *objs, file=sys.stderr)
    
def printOut(*objs):
    print(*objs, file=sys.stdout)

def get_side_effect_data(labelsFile):
    drugsWithLabels = defaultdict(dict)
    # read labels file
    with open(labelsFile, 'r') as f:
        header_to_skip = f.readline()
        for line in f:
            fields = line.rstrip().split("\t")
            roughPCID = fields.pop(0)
            pcid = roughPCID.lstrip("0") # the PCIDs have leading 0s, but our DB doesn't have that
            drugsWithLabels[pcid][fields[0]] = float(fields[1])
    return drugsWithLabels

def prep_se_df(drugsWithSEs):
    df = pd.DataFrame(list(drugsWithSEs.values()), index=list(drugsWithSEs.keys()))
    return df, df.columns

if __name__=='__main__':
    
    #=================================================
    # Read in the arguments/define options
    #=================================================
    exitCoder = ExitCoder()
    
    arguments = argparse.ArgumentParser(description="Calculate the total (increases in liklihood of a side effect X severity score for that side effect) for all significantly increased side-effects for a drug")
    
    arguments.add_argument("-p", help="TSV where each row is a side-effect drug combination")
    
    arguments.add_argument("-s", help="Full path of file with the severity scores for each side effect.")
    
    arguments.add_argument("-o", help="Output file prefix")
    
    arguments.add_argument("--verbose", action="store_true", help="Print out status reports")
            
    args = arguments.parse_args()
    
    # return usage information if no argvs given
    
    if len(sys.argv) < 3:
        arguments.print_help()
        sys.exit(exitCoder.encode('usageError'))
    
    ##### INPUTS AND OUTPUTS AND SETTINGS #####
    # now read in all drugs with side effects
    verboseOut("Reading in side effect prevelance data...")
    drugsWithSEs = get_side_effect_data(args.p)
    
    # this gives a dictionary of dinctionaries, keyed on the SE UMLS, then 'name' or 'score'
    verboseOut("Reading in side effect severity data...")
    sevScores = cseps.get_severity_scores(args.s)
    
    verboseOut("Calculating scores...")    
    df, sideEffects = prep_se_df(drugsWithSEs)
    real_scores = cseps.score_drugs(df, sideEffects, sevScores)
    
    verboseOut("Writing out results...")
#    report_scores(real_scores, args.detailedScores, args.o)
    cseps.report_scores(real_scores, args.o)
    
    verboseOut("Finished successfully")

