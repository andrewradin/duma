#!/usr/bin/python
from __future__ import print_function
import sys, getopt, os, re
from optparse import OptionParser
from os import listdir
from os.path import isfile, join
sys.path.insert(1,"../../web1")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
import django
django.setup()
from path_helper import PathHelper
from algorithms.exit_codes import ExitCoder

# created 13.Jan.2016 - Aaron C Daugherty - twoXAR
# A program to combine side effect data with UMLS codes and severity score data for those named side effects
# The final product being side effect names (from medera), UMLS codes for that side effect, the severity score for that side-effect

# A few subroutines to print out as I want

def message(*objs):
    print(*objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

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
# if not there get the S3 file containing their webaddresses and download htem
def checkForSiderFiles(downloadFileName, searchPattern):
    df = fetchToxFile(downloadFileName)
    expectedNumber = file_len(df) 
    actualNumber = countPatternedFilesInDir(PathHelper.tox, searchPattern)
    if expectedNumber != actualNumber :
        warning("Did not find SIDER files. Downloading those now.")
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

def getSeverityFile(fullSevFile):
    import subprocess as sp
    zcatProc = sp.Popen(["zcat", fullSevFile], stdout = sp.PIPE)
    tailProc = sp.Popen(["tail", "-n", "+2"], stdin = zcatProc.stdout, stdout = sp.PIPE)
    cutProc = sp.Popen(["cut", "-f", "1-2"], stdin = tailProc.stdout, stdout = sp.PIPE)
    return(cutProc.stdout.read().split("\n"))

def getUmlsFile(siderFilePrefix):
    import subprocess as sp
    # normally I would link together a bunch of Popen commands, 
    # but in this case since I want to expand the * and it's several commands long
    cmd = "zcat " + PathHelper.tox + siderFilePrefix + "* | rev | cut -f 1-2 | rev | sort -u -f"
    p1 = sp.Popen(cmd, shell=True, stdout=sp.PIPE)
    return(p1.stdout.read().split("\n"))


#=================================================
# Read in the arguments/define options
#=================================================
exitCoder = ExitCoder()

opts = OptionParser()

usage = "usage: %prog [options] [input] This will combine side effect files on the side effect name"

opts = OptionParser(usage=usage)

opts.add_option("-o", help="Name of the output file")

options, arguments = opts.parse_args()


# return usage information if no argvs given

if not options.o:
    opts.print_help()
    sys.exit(exitCoder.encode('usageError'))


##### INPUTS AND OUTPUTS #####
siderFilePrefix="meddra"
siderDownloadFileName = "siderToDownload.txt"
fullSevFileName = 'Gottlieb_supTable2_crowdSourcedSideEffectRankings.tsv.gz'
fullSevFile = fetchToxFile(fullSevFileName)

# check for meddra files (from sider)
# if not there get the S3 file containing their webaddresses and download htem
if not checkForSiderFiles(siderDownloadFileName, siderFilePrefix):
    sys.exit(exitCoder.encode('unableToFindDataError'))

# pull out all of the UMLS and meddra side effect name paris from the files
umlsFile = getUmlsFile(siderFilePrefix)
# pull out the relevant info from the severity file
sevFile = getSeverityFile(fullSevFile)

ses = {}
for line in umlsFile:
    splitLine = line.lower().split("\t")
    if len(splitLine) < 2 :
        break
    if splitLine[0] == "":
        splitLine[0] = "-" 
    if splitLine[1] in ses.keys():
        if splitLine[0] != ses[splitLine[1]]:
            warning( splitLine[0], " was seen multiple times, but with different UMLSs")
            sys.exit(exitCoder.encode('unexpectedDataFormat'))
    ses[splitLine[1]] = splitLine[0]

with open(options.o, 'w') as out:
    for line in sevFile:
        umls = "-"
        splitLine = line.lower().split("\t")
        if len(splitLine) < 2 :
            break
        if splitLine[0] in ses.keys() :
            umls = ses[splitLine[0]]
        out.write("\t".join((splitLine[0], umls, splitLine[1])) + "\n")
out.close()
