#!/usr/bin/python
from __future__ import print_function
import sys, getopt, os, re, gzip, argparse
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from collections import defaultdict
sys.path.insert(1,"../../../../web1")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
import django
django.setup()
from path_helper import PathHelper
from algorithms.exit_codes import ExitCoder

# created 14.Jan.2016 - Aaron C Daugherty - twoXAR

# TO DO:
# Speed up - this isn't that slow, especially since we should only need to use it once, but the writing out takes forever
# Summary stats
# 

# IMPORTANT: For drug IDs we're currently using the pubchem IDs
# We could trace these to InChI via STITCH, at some point, but currently we are not.
# We are using PubChemIDs b/c the IDs used by SIDER (and STITCH) are systematically modified PubChemIDs.
# The one bit that is confusing is that SIDER/STITCH have 2 different versions of the PubChemID.
# As explained in the STITCH README:
''' 
Chemicals are derived from PubChem. As described in the STITCH paper,
we merge salt forms and isomers. However, since STITCH 3, isomeric
compounds can also be investigated separately. In the download files,
the following convention holds:

CID0... - this is a stereo-specific compound, and the suffix is the
PubChem compound id.

CID1... - this is a "flat" compound, i.e. with merged stereo-isomers
The suffix (without the leading "1") is the PubChem compound id.
'''
# In my limited searching I found that the stero-specific ID was most often a relavent drug
# and in some cases the flat and stereo were in fact the same PubChemID (ignoring the systematic changes)
#############################################################################################################
# Because of that I am, currently going with the stero-specific IDs. 
# That is column number 2 in the SIDER files I'm using.
# Note that the stereo-IDs are NOT in the third SIDER file, which I am NOT using.
#############################################################################################################

# This will download all of the sider files,
# then parse them to determine if each drug has each side effect.
# The final result is a tsv where each row is a drug.
# Column 1 is the ID for that drug (currently stereo-specific PubChem IDs).
# All other columns are side effects (represented by UMLS IDs): True or False
# There will be a header to provide the UMLS

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

def countPatternedFilesInDir(direc, pattern):
    allFiles = [f for f in listdir(direc) if isfile(join(direc, f))]
    actualNumber = 0
    for file in allFiles:
        if re.match(pattern, file):
            actualNumber = actualNumber + 1
    return(actualNumber)

# check for meddra files (from sider)
# if not there get the S3 file containing their webaddresses and download them
def checkForSiderFiles(downloadFileName, searchPattern):
    df = fetchToxFile(downloadFileName)
    expectedNumber = file_len(df) 
    actualNumber = countPatternedFilesInDir(PathHelper.tox, searchPattern)
    if expectedNumber != actualNumber :
        verboseOut("Did not find SIDER files. Downloading those now.")
        import urllib
        with open(df) as f:
            for line in f:
                fn = line.split("/")[-1:]
                filePath =  PathHelper.tox + str(fn[0]).rstrip()
                urllib.urlretrieve(line.rstrip(), filePath)
    actualNumber = countPatternedFilesInDir(PathHelper.tox, searchPattern)
    if expectedNumber == actualNumber:
        return True
    else :
        warning("SIDER files were not able to be accessed. Expected ", expectedNumber, ", but found ", actualNumber, ".  Try clearing out all meddra* in ", PathHelper.tox, " and re-trying")
        return False

def fetchToxFile(fileName):
    from dtk.s3_cache import S3Bucket, S3File
    f=S3File(S3Bucket('tox'),fileName)
    f.fetch()
    return PathHelper.tox + fileName

def file_len(fileName):
    with open(fileName) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def getUmlsFile(siderFilePrefix):
    import subprocess as sp
    # normally I would link together a bunch of Popen commands,
    # but in this case since I want to expand the * and it's several commands long
    cmd = "zcat " + PathHelper.tox + siderFilePrefix + "* | rev | cut -f 1-2 | rev | sort -u -f"
    p1 = sp.Popen(cmd, shell=True, stdout=sp.PIPE)
    return(p1.stdout.read().split("\n"))

def getUmlsAndPCID(fields, drugIDColumn, umlsIdCol = -2):
    pubChemId = fields[drugIDColumn - 1][3:] # remove the first 4 characters which are added by SIDER/STITCH to the pubChemID
    umlsId = fields[umlsIdCol]
    return(pubChemId, umlsId)

# In some cases there are a range of frequencies.
# This will decide on a single value.
# At this point I'm just taking the midpoint of min and max, and ignoring the description, but later we may want to be more sophisticated
def pickFreq(freqDesc, maxFreq, minFreq):
    return((float(maxFreq) + float(minFreq)) / 2.0)

def combineFreqs(frequencyList):
    return np.median(np.array(frequencyList))

def placeboNormalizeAll(seFreqFileName, drugSEPairs):
    freqs = defaultdict(lambda: defaultdict(list))
    drugsSeenInFreq = []
    currentDrug = None
    verboseOut("Reading in frequency data in order to filter out side effects that are worse in placebo matched controls as with the drug")
    with gzip.open(PathHelper.tox + seFreqFileName, 'r') as f:
        for line in f:
            fields = line.rstrip().split("\t")
            (pubChemId, umlsId) = getUmlsAndPCID(fields, drugIDColumn)
            # first deal with the first line issue
            if not currentDrug :
                currentDrug = pubChemId
            elif currentDrug != pubChemId :
                # we're done with this drug
                drugsSeenInFreq.append(currentDrug)
                # now that we're done with this drug, update the dict which is tracking whether a drug has a specific SE
                # and if the option is set, replace add the frequency as the end value
                drugSEPairs.loc[currentDrug] = placeboNormalize(freqs, drugSEPairs.loc[currentDrug], args.reportFreq)
                currentDrug = pubChemId
                if currentDrug in drugsSeenInFreq :
                    warning("SIDER Frequency file was expected to be sorted, but found drug, ", currentDrug, ", in 2 separate locations in the file. Quitting now.")
                    sys.exit(exitCoder.encode('unableToFindDataError'))
                # now reset freqs
                freqs = defaultdict(lambda: defaultdict(list))
            if not fields[3] or fields[3] == "":
                placeboFlag = "drug"
            elif fields[3] == "placebo":
                placeboFlag = "placebo"
            else:
                warning("Unexpected placebo flag found for ", currentDrug, " and ", umlsId, ": ", fields[3], ". Ignoring for now.\n", line)
            # take the mid point of the min or max frequency as the 1 frequency
            freqs[umlsId][placeboFlag].append(pickFreq(fields[4], fields[6], fields[5]))
    return drugSEPairs

# this filters out SEs that have just as high a frequency in the placebo as the drug
# it also replaces the counts of the SE (a temp place holder), with frequency information, if that option is set
def placeboNormalize(freqsDictDictList, drugSingleRowDF, reportFreqs):
    for se in freqsDictDictList.keys():
        if freqsDictDictList[se]['drug'] :
            drugFreq = combineFreqs(freqsDictDictList[se]['drug'])
            # if there is a placebo frequency, normalize
            if freqsDictDictList[se]['placebo'] :
                placeboFreq = combineFreqs(freqsDictDictList[se]['placebo'])
                # if we are reporting the actual frequencies
                if reportFreqs :
                    if placeboFreq >= drugFreq:
                        drugSingleRowDF.loc[se] = 0.0
                    else:
                        drugSingleRowDF.loc[se] = drugFreq - placeboFreq
                elif placeboFreq >= drugFreq :
                # otherwise keep the count, unless the placebo frequency is higher
                    drugSingleRowDF.loc[se] = 0.0
            elif reportFreqs:
            # if there is no placebo, just report the drugfrequency
                drugSingleRowDF.loc[se] = drugFreq
    return drugSingleRowDF

def getDrugSEPairs(gzippedFile, drugIDColumn, minSePortion):
    allSEs = {}
    drugSEPairs = defaultdict(dict)
    with gzip.open(gzippedFile, 'r') as f:
        verboseOut("Reading in total side effect data")
        for line in f:
            fields = line.rstrip().split("\t")
            (pubChemId, umlsId) = getUmlsAndPCID(fields, drugIDColumn)
            if umlsId in allSEs.keys():
                allSEs[umlsId] += 1
            else:
                allSEs[umlsId] = 1
    # rather than tracking the number of times this pair was seen, this is basically just noting it was seen at all
    # This will either turn into a True flag, or the frequency with which this drug was seen (which is 1 or below)
            drugSEPairs[pubChemId][umlsId] = foundSENumber
    # convert this to a pandas daaframe for ease of access and manipulation later
    drugSEPairs_df = pd.DataFrame(drugSEPairs.values(), index=drugSEPairs.keys())
    # filter out columns (SEs) w/too few drugs (i.e. super rare SEs)
    minDrugNum = round(len(drugSEPairs_df)*minSePortion, 0)
    before = len(drugSEPairs_df.columns)
    drugSEPairs_df.dropna(axis=1, thresh=minDrugNum, inplace=True)
    # unfortunately we need to filter 2 data sources separately
#    filteredSEs = {k: v for k, v in allSEs.items() if v >= minDrugNum}
    # this was causing issues, and I trust the df more
    filteredSEs = drugSEPairs_df.columns
    verboseOut("Removed ", str(before - len(drugSEPairs_df.columns)), " side effects. Leaving ", len(drugSEPairs_df.columns))
    # replace NaN with 0
    return filteredSEs, drugSEPairs_df.fillna(0)

def inferMissingFreqs(drugSEPairs):
    for colInd in xrange(len(drugSEPairs.columns)):
        column = drugSEPairs.ix[:,colInd]
        medianValueOfFreqs = findFillInVal(column)
        # now replace all values greater than 1 (they really should be 2 from above, but in case that chagnes, we'll make it general)
        column[column > 1] = medianValueOfFreqs
        drugSEPairs.ix[:,colInd] = column
    return drugSEPairs

# take a one column pandas DF, and returns the value that should replace those instances without frequency
def findFillInVal(column):
    # currently I am just taking the median of the frequencies
    # this big ugly line takes the median of column values that are not 0 or bigger than 1.
    # i.e. the frequencies from above.
    # this is done by masking the array (i.e. column) using 2 logicals
    maskedArray = np.ma.masked_where(reduce(np.logical_or, [column==0.0, column>1.0]), column)
    #if np.ma.MaskedArray.count(maskedArray) < 2:
    #    warning("Trying to infer frequency from 1 or fewer frequencies")
    return float(np.ma.median(maskedArray))

if __name__=='__main__':

    #=================================================
    # Read in the arguments/define options
    #=================================================
    exitCoder = ExitCoder()

    arguments = argparse.ArgumentParser(description="This will combine side effect files on the side effect name")
    
    arguments.add_argument("-o", help="Output file prefix. Depending on flags numberous files may be created; all will start with this")
    
    arguments.add_argument("--minSEPortion", type=float, default=0.005, help="Portion of drugs that must be in the minority to try to predict (default %(default)s)")
    
    arguments.add_argument("--stats", action="store_true", dest="printStats", help="Write summary stats to a separate file")
    
    arguments.add_argument("--freq", action="store_true", dest="reportFreq", help="Frequency of the side effect for the given drug is reported where possible")
    
    arguments.add_argument("--placebo", action="store_true", dest="placeboNorm", help="If possible include the placebo portion information. Only frequencies more prevalent than the paired placebo will be reported.")
    
    arguments.add_argument("--verbose", action="store_true", help="Print out status reports")
    
    args = arguments.parse_args()
    
    if not args.o:
        arguments.print_help()
        sys.exit(exitCoder.encode('usageError'))
    
    
    ##### INPUTS AND OUTPUTS #####
    # hardcoded variables, these code be changed to options, if necessary
    siderFilePrefix = "meddra"
    siderDownloadFileName = "siderToDownload.txt"
    allSeFileName = "meddra_all_se.tsv.gz"
    seFreqFileName = "meddra_freq.tsv.gz"
    drugIDColumn = 2 # this is the stereo-specific PubChemID
    foundSENumber = 2
    minSePortion = args.minSEPortion # if a feature is super rare, there isn't much use in including it as it isn't generalizable
    
    firstOutFile = args.o + "_presenceOfSideEffect.tsv"
    if args.reportFreq:
        freqOutFile = args.o + "_sideEffectFrequency.tsv"
    if args.placeboNorm :
        placeboNormOutFile = args.o + " _sideEffectFreq_placeboNormd.tsv"
    
    # check for meddra files (from sider)
    # if not there get the S3 file containing their webaddresses and download them
    if not checkForSiderFiles(siderDownloadFileName, siderFilePrefix):
        sys.exit(exitCoder.encode('unableToFindDataError'))
    
    # first just take the SE drug pairings as is
    # I've put them in a pandas dataframe: rows = pubChemIDs, column UMLS IDs
    # and a secondary dict with all SEs observed
    
    # I don't have an exact use just yet,
    # but I am going to track to the number of times each SE is seen
    allSEs, drugSEPairs = getDrugSEPairs(PathHelper.tox + allSeFileName, drugIDColumn, minSePortion)
    
    # these files are sorted by drugs, so we'll assume that's true, but include a log of drugs seen just in case
    if args.placeboNorm :
        firstOutFile = args.o + "_sideEffectFreq_placeboNormd.tsv"
        drugSEPairs = placeboNormalizeAll(seFreqFileName, drugSEPairs)
    
    # now that we've gleaned all of the frequency information we could, we try to fill in the gaps:
    # those situations where we know a drug has a SE, but no frequency information is given.
    # For the time being, we're doing this on a side effect by side effect basis
    if args.reportFreq:
        drugSEPairs = inferMissingFreqs(drugSEPairs)
    
    # now just write them out
    with open(firstOutFile, 'w') as out:
        verboseOut("Writing out results")
        # to support frequency reporting we actually keep the occurence of side effects as a number, but for reporting, I would rather have True/False
        if args.reportFreq:
            outLine = re.sub(' +', "\t", drugSEPairs.to_string())
        else:
            outLine = re.sub(' +', "\t", drugSEPairs.replace([0,foundSENumber], ['False', 'True']).to_string())
        out.write(outLine)
    
    verboseOut("Finished successfully")
