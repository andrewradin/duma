#!/usr/bin/python
from __future__ import print_function
import sys, getopt, os, re, gzip, argparse
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from collections import defaultdict
sys.path.insert(1,"../../web1")
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
def checkForSiderFiles(urls,searchPattern):
    expectedNumber = len(urls) 
    actualNumber = countPatternedFilesInDir(PathHelper.tox, searchPattern)
    if expectedNumber != actualNumber :
        verboseOut("Did not find SIDER files. Downloading those now.")
        import urllib
        for url in urls:
            fn = url.split("/")[-1:]
            filePath =  PathHelper.tox + fn[0]
            urllib.urlretrieve(url, filePath)
    actualNumber = countPatternedFilesInDir(PathHelper.tox, searchPattern)
    if expectedNumber == actualNumber:
        return True
    else :
        warning("SIDER files were not able to be accessed. Expected ", expectedNumber, ", but found ", actualNumber, ".  Try clearing out all meddra* in ", PathHelper.tox, " and re-trying")
        return False

def smart_append_dict(d, new_key, new_val):
    if new_key in d.keys():
        if new_val != d[new_key]:
            warning("Inconsistent umls naming:", d[new_key], new_val, ". Using the first instance seen.")
    else:
        d[new_key] = new_val
    return d

# In some cases there are a range of frequencies.
# This will decide on a single value.
# At this point I'm just taking the midpoint of min and max, and ignoring the description, but later we may want to be more sophisticated
def pickFreq(maxFreq, minFreq):
    return((float(maxFreq) + float(minFreq)) / 2.0)

def combineFreqs(frequencyList):
    return np.median(np.array(frequencyList))

if __name__=='__main__':

    #=================================================
    # Read in the arguments/define options
    #=================================================
    exitCoder = ExitCoder()

    arguments = argparse.ArgumentParser(description="This will combine side effect files on the side effect name")
    
    arguments.add_argument("--verbose", action="store_true", help="Print out status reports")
    
    args = arguments.parse_args()
    
#    if not args.o:
#        arguments.print_help()
#        sys.exit(exitCoder.encode('usageError'))
    
    
    ##### INPUTS AND OUTPUTS #####
    # hardcoded variables, these code be changed to options, if necessary
    siderFilePrefix = "meddra"
    allSeFileName = PathHelper.tox + "meddra_all_se.tsv.gz"
    seFreqFileName = PathHelper.tox + "meddra_freq.tsv.gz"
    drugIDColumn = 1 # this is the stereo-specific PubChemID, and zero indexed
    meddra_map = '../umls/umls_to_meddra.tsv'
    download_urls = (
    'http://sideeffects.embl.de/media/download/meddra_all_indications.tsv.gz',
    'http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz',
    'http://sideeffects.embl.de/media/download/meddra_freq.tsv.gz',
    )
    
    defaultOutFile = "adr.sider.default.tsv"
    portionOutFile = "adr.sider.portion.tsv"
    ratioOutFile = "adr.sider.odds_ratio.tsv"
    umlsOutFile = "tox.umls.sider.tsv"
    
    # read in mappings for UMLS ids
    umls2meddra = {}
    for line in open(meddra_map):
        rec = line.strip('\n').split('\t')
        if not rec:
            continue
        umls2meddra[rec[0]] = rec[1]
    # for recording bad mappings
    bad_umls = set()
    used_meddra = {}
    used_meddra_shown = set()
    def get_meddra(umls):
        if umls not in umls2meddra:
            bad_umls.add(umls)
            return None
        else:
            result = umls2meddra[umls]
            if result in used_meddra:
                if used_meddra[result] != umls:
                    err = '%s needed for %s but already used for %s' % (
                                    result,umls,used_meddra[result],
                                    )
                    if err not in used_meddra_shown:
                        #warning(err)
                        used_meddra_shown.add(err)
            else:
                used_meddra[result] = umls
            return result

    # check for meddra files (from sider)
    # if not there get the S3 file containing their webaddresses and download them
    if not checkForSiderFiles(download_urls,siderFilePrefix):
        sys.exit(exitCoder.encode('unableToFindDataError'))
    
    # First I'll go through the frequency data
    umls_dict = {}
    # rather than tracking the number of times this pair was seen, this is basically just noting it was seen at all
    # This will either turn into a True flag, or the frequency with which this drug was seen (which is 1 or below)
    with gzip.open(allSeFileName, 'r') as fi:
        with open(defaultOutFile, 'w') as fo:
            # header
            fo.write("\t".join(['stitch_id', 'meddra_id']) + "\n")
            verboseOut("Reading in total side effect data")
            current_drug = None
            ses = []
            for line in fi:
                fields = line.rstrip().split("\t")
                # we onl;y want to use the meddra prefered terms
                if fields[3] != 'PT':
                    continue
                if current_drug is None:
                    current_drug = fields[drugIDColumn]
                elif fields[drugIDColumn] != current_drug:
                    # report
                    for se in list(set(ses)):
                        meddra = get_meddra(se)
                        if meddra:
                            fo.write("\t".join([current_drug, meddra]) + "\n")
                    # clean out
                    current_drug = fields[drugIDColumn]
                    ses = []
                
                ses.append(fields[4])
    
                # generate umls - term dictionary
                umls_dict = smart_append_dict(umls_dict, fields[4], fields[5])
    
    # now go through frequency data
    drugSEPairs = defaultdict(list)
    with gzip.open(seFreqFileName, 'r') as f:
        with open(portionOutFile, 'w') as fp:
            # header
            fp.write("\t".join(['stitch_id', 'meddra_id', 'frequency']) + "\n")
            
            with open(ratioOutFile, 'w') as fr:
                # header
                fr.write("\t".join(['stitch_id', 'meddra_id', 'odds_ratio']) + "\n")
                verboseOut("Reading in side effect frequency data")
                current_drug = None
                ses = defaultdict(list)
                placebos = defaultdict(list)
                for line in f:
                    fields = line.rstrip().split("\t")
                    # we onl;y want to use the meddra prefered terms
                    if fields[7] != 'PT':
                        continue
                    if current_drug is None:
                        current_drug = fields[drugIDColumn]
                    elif fields[drugIDColumn] != current_drug:
                        # report
                        for se in ses.keys():
                            meddra = get_meddra(se)
                            if not meddra:
                                continue
                            freq = combineFreqs(ses[se])
                            fp.write("\t".join([current_drug, meddra, str(freq)]) + "\n")
                            if se in placebos.keys():
                                placebo_freq = combineFreqs(placebos[se])
                                if freq > placebo_freq:
                                    odds_ratio = freq / placebo_freq
                                    fr.write("\t".join([current_drug, meddra, str(odds_ratio)]) + "\n")
                        # clean out
                        current_drug = fields[drugIDColumn]
                        ses = defaultdict(list)
                        placebos = defaultdict(list)
                    if fields[3] == 'placebo':
                        placebos[fields[2]].append(pickFreq(fields[5], fields[6]))
                    else:
                        ses[fields[2]].append(pickFreq(fields[5], fields[6]))
                    
                    # generate umls - term dictionary
                    umls_dict = smart_append_dict(umls_dict, fields[8], fields[9])
    # now write out the term dictionary
    with open(umlsOutFile, 'w') as f:
        f.write("\t".join(['umls_id', 'attribute', 'value']) + "\n")
        for umls in umls_dict.keys():
            f.write("\t".join([umls, 'meddra_pt', umls_dict[umls]]) + "\n")
    verboseOut("skipped",len(bad_umls),"unmatched UMLS keys")
    verboseOut("found",len(used_meddra_shown),"distinct key clashes")
    verboseOut("Finished successfully")
