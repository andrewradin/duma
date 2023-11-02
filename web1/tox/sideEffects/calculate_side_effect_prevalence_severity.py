#!/usr/bin/python
from __future__ import print_function
from builtins import range
import os, django, sys, argparse
import pandas as pd
from collections import defaultdict,Counter
sys.path.insert(1,"../..")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from path_helper import PathHelper
from algorithms.exit_codes import ExitCoder

# created 16.Feb.2016 - Aaron C Daugherty - twoXAR

# TO DO:
#  If files aren't present, call code to make them
#  Connect ULMS to severity

# This is to predict the presence/absence of all side effects, individually, for all drugs we can
# This makes use of pre-parsed data from SIDER to label some portion of drugs as having the side effect or not.
# If a side effect isn't in the pre-parsed file we can't predict it.
# If a drug isn't in the pre-parsed file, it means it isn't in SIDER, and we don't know of any of its side effects.


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
    drugsWithLabels = defaultdict(list)
    # read labels file
    with open(labelsFile, 'r') as f:
        sideEffects = f.readline().rstrip().split("\t")
        #idType = sideEffects.pop(0)
        for line in f:
            fields = line.rstrip().split("\t")
            roughPCID = fields.pop(0)
            pcid = roughPCID.lstrip("0") # the PCIDs have leading 0s, but our DB doesn't have that
            drugsWithLabels[pcid] = [float(f) for f in fields]
    return drugsWithLabels, [s.upper() for s in sideEffects]

# read all of the severity data into an easy to access hash
# there are only 1-2k of these, so putting in memory isn't awful
def get_severity_scores(severityFile):
# data example:
# cardiac arrest  c0018790        1.00
# bone cancer metastatic  c0153690        0.98
    allSev = defaultdict(dict)
    with open(severityFile, 'r') as f:
        for line in f:
            fields = line.rstrip().split("\t")
            fields[1] = fields[1].upper()
            allSev[fields[1]]['name']= fields[0]
            allSev[fields[1]]['score']= float(fields[2])
#    verboseOut("Loaded ", len(allSev.keys()), " side effect severity scores")
    return allSev

def prep_se_df(drugsWithSEs, sideEffectOrder):
    df = pd.DataFrame(drugsWithSEs)
    df = df.transpose()
    df.columns = sideEffectOrder
    return df

def score_drugs(df, sideEffectOrder, sevs):
    sevScores = {}
    maxSevScore = sum([subdict['score'] for subdict in sevs.values()])
    for i in range(len(sideEffectOrder)):
        # only get a score for those side effects with a severiry score
        if sideEffectOrder[i] not in list(sevs.keys()):
            # get rid of this column b/c we don't have severity scores
            df.drop(sideEffectOrder[i], axis=1, inplace=True)
            continue
        # the end result is score where 100 is the equivalent of having 100% prevelance of all the side effects
        sevScores[sideEffectOrder[i]] = sevs[sideEffectOrder[i]]['score'] / maxSevScore * 100.0 
    for col in df.columns:
        df[col] *= sevScores[col]
    return df

#def report_scores(realScores, detailedScores, filePrefix):
def report_scores(realScores, filePrefix):
    realSums = realScores.sum(axis=1)
    realSums.to_csv(path=filePrefix + '_real_sums.tsv', sep="\t")
#    if detailedScores:
#        warning(realScores)
#        realScores.to_csv(path=filePrefix + '_real_detailed_scores.tsv', sep="\t", header=True)


if __name__=='__main__':
    
    #=================================================
    # Read in the arguments/define options
    #=================================================
    exitCoder = ExitCoder()
    
    arguments = argparse.ArgumentParser(description="Calculate the total side effect X severity score for a drug")
    
    arguments.add_argument("-p", help="SE prevalance for every drug. A TSV where each row is a drug, and each column is a SE prevalance.")
    
    arguments.add_argument("-s", help="Full path of file with the severity scores for each side effect.")
    
    arguments.add_argument("-o", help="Output file prefix")
    
# this doesn't seem to be working and it's less important currently
#    arguments.add_argument("--detailedScores", action="store_true", help="Print out how drug, side effect score matrix, not just a single score per drug")
    
    arguments.add_argument("--verbose", action="store_true", help="Print out status reports")
            
    args = arguments.parse_args()
    
    # return usage information if no argvs given
    
    if len(sys.argv) < 3:
        arguments.print_help()
        sys.exit(exitCoder.encode('usageError'))
    
    ##### INPUTS AND OUTPUTS AND SETTINGS #####
    # now read in all drugs with side effects
    verboseOut("Reading in side effect prevelance data...")
    drugsWithSEs, sideEffects = get_side_effect_data(args.p)
    
    # this gives a dictionary of dinctionaries, keyed on the SE UMLS, then 'name' or 'score'
    verboseOut("Reading in side effect severity data...")
    sevScores = get_severity_scores(args.s)
    
    verboseOut("Calculating scores...")    
    real_scores = score_drugs(prep_se_df(drugsWithSEs, sideEffects), sideEffects, sevScores)
    
    verboseOut("Writing out results...")
#    report_scores(real_scores, args.detailedScores, args.o)
    report_scores(real_scores, args.o)
        
    verboseOut("Finished successfully")

