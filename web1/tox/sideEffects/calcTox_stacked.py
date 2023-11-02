#!/usr/bin/python
from __future__ import print_function
from builtins import range
import os, django, sys, arff, random, re #, time
from collections import defaultdict,Counter
from optparse import OptionParser
from copy import deepcopy
import numpy as np
sys.path.insert(1,"../../web1")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from browse.models import WsAnnotation
from path_helper import PathHelper,make_directory
from algorithms.exit_codes import ExitCoder

# created 18.Jan.2016 - Aaron C Daugherty - twoXAR

# TO DO:
#  Stack classifier and regression
#    just classifier isn't working
#  If files aren't present, call code to make them
#  

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
    if options.verbose:
        print(*objs, file=sys.stderr)

def useage(*objs):
    print("USEAGE: ", *objs, file=sys.stderr)
    
def printOut(*objs):
    print(*objs, file=sys.stdout)

# basically a glorified/complicated wrapper of weka plus builds the files weka needs for each SE
def makePreds(sideEffects, drugsWithSEs, mainAttributes, features, drugIDs, trainingPortion, forTesting, mlMethod, classification, cleanUp, baseFilePrefix):
    allPreds={}
    if not classification:
        # we ultimately want to run a regression model, but first we want to classify whether or not the drug has the side effect
        # to do that we'll alter the drugsWithSEs values to be T/F on whether or not the value is > 0.
        # and do an iterative call to this function. Then parse out any drugs predicted to be T. 
        #   We might need to play around with the cut off here. I think being more liberal than > 0.5 may pay off
        #   My rationale being that it's better to get all the drugs with some frequency and include some with 0s.
        #   That means this turns into a filtering of the drugs that obviously don't have the SE
        # then we'll subset drugsWithSEs to use only those predicted to have the SE, then we'll run regression on that
        #
        # first step is processing drugsWithSEs to T/F
        # luckily that's easy b/c bool(0) = False and bool of any float = True
        drugsWithSEs_binerized = {k: [bool(float(x)) for x in v] for k, v in drugsWithSEs.items()}
        # I should have the classifier method as an option
        allBinaryPreds = makePreds(sideEffects, drugsWithSEs_binerized, mainAttributes, features, drugIDs, trainingPortion, forTesting, 'naiveBayes', True, cleanUp, baseFilePrefix)
        verboseOut("Finished with initial binary classification before regression. Done to make regression easier")
        # now we parse through this list of strings (wsa<tab>True/False<tab>T prob), and only pull out those wsas predicted to have SE
        # then update drugsWithSEs and drugIDs
        # later we need to revamp this so that the same training data is used for each run, but for the time being to get this working, I won't worry about that
        minTrueProb = 0.5 # let's just go with the default
            
        
    for i in range(len(sideEffects)):
        se = checkSEPrevalance(sideEffects, i, list(drugsWithSEs.values()))
        if not se:
            continue
        
        # now process the data and print out arffs
        if classification:
            filePrefix = baseFilePrefix + "_binaryClassification"
            drugsToUse = list(drugsWithSEs.keys())
        else:
            filePrefix = baseFilePrefix + "_frequencyRegression"
            # We also need to edit the drug information for this SE
            # particularly we need to edit drugsWithSEs and drugIDs
            # hash keyed on wsa with T prob as value
            binaryPredictions = {drugPreds.split()[0]: float(drugPreds.split()[3]) for drugPreds in allBinaryPreds[se]}
            drugsToUse = []
            for wsa, tProb in binaryPredictions.items():
                if wsa in list(drugsWithSEs.keys()) and tProb >= minTrueProb:
                    drugsToUse.append(wsa)
            verboseOut("Of ", len(binaryPredictions), " drugs whose side effect class were predicted, ", len(drugsToUse), " are being used for regression")
            if len(drugsToUse) == 0:
                warning(list(binaryPredictions.items()))
        seFilePrefix = filePrefix + "_" + sideEffects[i]
        trainingArff = seFilePrefix + "_training.arff"
        
        seSpecificAttributes = buildSeSpecificAttributes(mainAttributes, se, classification)
        
        if forTesting:
            # for testing to see how we do:
            predictionArff = seFilePrefix + "_testing.arff"
        else:
            # just for predictions:
            predictionArff = seFilePrefix + "_allData.arff"
            outFile = seFilePrefix + "_allDrugPredictions.txt"
        
        trainingData, predictionData, predictionDrugIds = buildSeData(features, drugIDs, trainingPortion, drugsWithSEs, i, classification, drugsToUse, forTesting)
        
        writeOutArff(se, seSpecificAttributes, trainingData, trainingArff)
        writeOutArff(se, seSpecificAttributes, predictionData, predictionArff)
        
        # debugging
        if not classification:
            continue
        # now actually run weka
        wekaOut = runWeka(mlMethod, trainingArff, predictionArff, forTesting)
        
        if forTesting:
            stats = processWekaStats(wekaOut)
            stats.insert(0, sideEffects[i])
            allStats.append("\t".join(stats))
        else:
            if plotting:
                plottingPrefix = setupPlotsDir(options.ws_id, sideEffects[i])
            else:
                plottingPrefix = None
            
            # a requirement of this is that the predictionDrugIds are in the same order as the predictions in wekaOut. Luckily that is the deafult for Weka.
            allPreds[se] = parseWekaPredictions(wekaOut, predictionDrugIds, plotting, plottingPrefix)
            if options.writePreds:
                with open(outFile, 'w') as f:
                    f.write("\n".join(allPreds[se]) + "\n")
        if cleanUp:
            os.remove(trainingArff)
            os.remove(predictionArff)
        
# for debugging
#        if i > 1:
#            return allPreds
    if forTesting:
        with open(outFile, 'w') as f:
            f.write("\n".join(allStats) + "\n")
    
    return allPreds


def setupPlotsDir (ws_id, se):
    plotDir = PathHelper.tox + ws_id + "/predictionEvaluationPlots"
    make_directory(plotDir)
    return plotDir + "/" + ws_id + "_sePrediction_" + sideEffects[i]

def setupOutDir (ws_id):
    wsSpecificToxDir = PathHelper.tox + ws_id
    make_directory(wsSpecificToxDir)
    return wsSpecificToxDir + "/" + ws_id + "_sePrediction"

def getWSAtoPCID(wsId):
#    wsAnnotToPCID = {str(rec.agent.pubchem_cid): str(rec.id) for rec in WsAnnotation.objects.filter(ws_id=wsId)}
    # the above was taking too long to troubleshoot with, so I wrote things to a file
    convertFile = "wsAnnotToPCID.tsv"
#    with open(convertFile, 'w') as outF:
#        for k in wsAnnotToPCID:
#            outF.write("\t".join((k, wsAnnotToPCID[k])) + "\n")
#    #
    wsAnnotToPCID = {}
    with open(convertFile, 'r') as f:
        for line in f:
            fields = line.rstrip().split("\t")
            wsAnnotToPCID[fields[0]] = fields[1]
    return(wsAnnotToPCID)

def convertPCIDtoWSAWithSE(labelsFile, wsAnnotToPCID, includeNoLabels = False):
    drugsWithLabels = defaultdict(list)
    unableToConvert = []
    # read labels file
    with open(labelsFile, 'r') as f:
        sideEffects = f.readline().rstrip().split("\t")
        idType = sideEffects.pop(0)
        for line in f:
            fields = line.rstrip().split("\t")
            roughPCID = fields.pop(0)
            pcid = roughPCID.lstrip("0") # the PCIDs have leading 0s, but our DB doesn't have that
            if pcid in list(wsAnnotToPCID.keys()):
                drugsWithLabels[wsAnnotToPCID[pcid]] = fields
            else:
                # these are the drugs that we have labels for, but no work space annotations
                unableToConvert.append(pcid)
    verboseOut("Able to convert ", len(list(drugsWithLabels.keys())), " drugs from SIDER to workspace annotations")
    verboseOut("Unable to convert ", len(unableToConvert), " drugs to our WS annotation")
    return drugsWithLabels, sideEffects

def getFeatures(inFile, drugList, minFtPortion):
    features = [line.rstrip().split(",") for line in open(inFile, 'r')]
    bits = features.pop(0)
    # I want to do this for later, outside of this function, but further down in this function this causes an off by 1 issues I deal with
    idType = bits.pop(0)
#    # for speed of learning, we're going to filter the features, as a lot of them were single instances
    featureIndsToEliminate = []
    for colInd in range(len(features[0])):
        if colInd == 0: # the first column are names
            continue
        column = [sublist[colInd] for sublist in features]
        biggestPortion = getBiggestPortionInList(column)
        if 1 - biggestPortion < minFtPortion:
            # don't use this feature, it's not prevalent enough
            featureIndsToEliminate.append(colInd)
    verboseOut("Removed ", len(featureIndsToEliminate), " features b/c they were either ubiquitous or too rare (cut-off: ", minFtPortion, ")")
    # Note that you need to delete them in reverse order so that you don't throw off the subsequent indexes.
    for index in sorted(featureIndsToEliminate, reverse = True):
        for row in features:
            del row[index]
        del bits[index - 1] # I popped off the first entry in this list, so everything has to be shifted
    verboseOut("Able to retrieve features for ", len(features), " drugs")
    verboseOut("Using a total of ", len(features[0]), " features")
    # now for memory saving, I'm going to separate the features and make them single strings and drugIDs.
    drugIds = [row[0] for row in features]
    featureStrings = [",".join(row[1:]) for row in features]
    return featureStrings, drugIds, bits

def buildAttributes(bits):
    mainAttributes = []
    for bit in bits:
        mainAttributes.append((bit, ['True', 'False']))
    return mainAttributes

# the first thing we do is make sure there are equal enough instances of this side effect
def checkSEPrevalance(sideEffects, iter, sideEffectPresence):
    se = sideEffects[iter]
    seCol = [list[iter] for list in sideEffectPresence]
    biggestPortion = getBiggestPortionInList(seCol)
    if 1 - biggestPortion < minSePortion:
        verboseOut(se, " was not balanced enough to use. Skipping")
        return False
    return se

def buildSeSpecificAttributes(mainAttributes, se, classification):
    seSpecificAttributes = mainAttributes[:] # slicing is the fastest way to get a new list instead of just a pointer
    if classification:
        seSpecificAttributes.append((se, ['True', 'False']))
    else:
        seSpecificAttributes.append((se, 'NUMERIC'))
    return seSpecificAttributes

def buildSeData(features, wsas, trainingPortion, drugsData, colInd, classification, drugsToUse, forTesting=False):
    # making a local copy of the list keeps the original list from being edited
    data = features[:]
    # training index now needs to be limited to those drugs with labels
    # first pull out which drugs in features have labels, or were noted to be used for regression
    haveLabel = []
    for i in range(len(wsas)):
        wsa = wsas[i]
        if wsa in drugsToUse:
            haveLabel.append(i)
    haveLabelTrainingIndex = map(int, random.sample(list(range(len(haveLabel))), int(trainingPortion*len(haveLabel))))
    trainingIndex = [ haveLabel[i] for i in haveLabelTrainingIndex ]
    trainingData = []
    predictionData = []
    predictionWsas = []
    for j in range(len(data)):
        # now add the SE label/portion
        if wsas[j] in drugsToUse:
            data[j] = data[j] + "," + str(drugsData[wsas[j]][colInd])
        else: # otherwise this must be a drug without a label, or was decided not to be used for regression
              # so we don't know if it has the SE. The weka filler for missing data is ?
            data[j] = data[j] + "," + '?'
        # always build a training set
        if j in trainingIndex:
            trainingData.append(data[j])
        # if we're not building for testing, all data goes into the predictionData set
        if not forTesting:
            predictionData.append(data[j])
            predictionWsas.append(wsas[j])
        # if we are building for testing, only the data with a label/portion, but not in the training set goes in the predictionData set
        elif j not in trainingIndex and j in haveLabel:
            predictionData.append(data[j])
            predictionWsas.append(wsas[j])
        
    if classification:
        training, leftover = balanceBinarySet(trainingData)
    else:
        training = trainingData
    if forTesting:
        if classification:
            second, toIgnore = balanceBinarySet(predictionData)
        else:
            second = predictionData
        return training, second, False # if we're just looking at stats, I don't actually care about what the drugs are, at least not enough to hassle with it
    return training, predictionData, predictionWsas

# balance the training set so that there are an equal number of drugs with and without the side effect
# For testing pruposes, I'll do the same with the testing set to see how we are doing
def balanceBinarySet(data):
    labels = [row.split(",")[-1] for row in data]
    instanceTotals = Counter(labels)
    largestClass,largestClassAmount = instanceTotals.most_common(1)[0]
    smallestClass,smallestClassAmount = instanceTotals.most_common(2)[1]
    verboseOut("Downsampling ", largestClass, " from ", largestClassAmount, " to ", smallestClassAmount, " to create a balanced set")
    if smallestClassAmount == largestClassAmount:
        return data, None
    largestClassDownsampled = []
    removedLargestClass = []
    lcInds = [i for i, x in enumerate(labels) if x == largestClass]
    toKeepInds = map(int, random.sample(list(range(len(lcInds))), smallestClassAmount))
    toKeep=[]
    for ind in toKeepInds:
        toKeep.append(lcInds[ind])
    dataToKeep = []
    leftoverData = []
    for j in range(len(data)):
        if j in toKeep or j not in lcInds: # i.e. if this row is in the subset of the larger class we're keeping or in the smaller class
            dataToKeep.append(data[j])
        else:
            leftoverData.append(data[j])
    return dataToKeep, leftoverData

def writeOutArff(se, seSpecificAttributes, dataAsListOfStrings, file):
    verboseOut("writing to ", file)
    data = []
    for row in dataAsListOfStrings:
        data.append(row.split(","))
    out = {
          'description':'',
          'relation':se,
          'attributes':seSpecificAttributes,
          'data':data
          }
    with open(file, 'w') as f:
        f.write(arff.dumps(out))

def getBiggestPortionInList(l):
    c = Counter(l)
    return(float(c.most_common(1)[0][1]) / float(sum(c.values())))

def runWeka(method, trainingFile, predictionFile, testing = False):
    import subprocess
    pathCmd = "export CLASSPATH=" + PathHelper.weka+":."
    commonWekaCmds = " ".join(['java', '-Xmx6G']) # max memory given to the classifier # this should be parameterized somewhere
    # Not currently saving models, but we could: '-d', modelFile
    # the last two result in the predictions being reported (but not the stats)
    sharedCmdsList = ['-t', trainingFile, '-T', predictionFile]
    if not testing:
        for s in ['-p', '0', '-distribution']:
            sharedCmdsList.append(s)
    
    sharedCmds = " ".join(sharedCmdsList)
    
    if method == 'RF':
        commands = " ".join([commonWekaCmds
                   , 'weka.classifiers.trees.RandomForest'
                   , sharedCmds
                   , '-I', '10' # number of trees
                   , '-K', '0' # number of features selected for each tree, 0 defaults to log2(number of features) + 1
                   , '-S', '1' # S is the seed
                  ])
    elif method == 'RF_tune':
        commands = " ".join([commonWekaCmds
                   , 'weka.classifiers.meta.CVParameterSelection'
                   , sharedCmds
                   , '-P', '"K 1 21 2"' # param'ize the number of features b/t 1 and 21
                   , '-S', '1'
                   , '-W', 'weka.classifiers.trees.RandomForest'
                   , '--' # the commands below go to the -w classifier
                   , '-I', '10' # number of trees
                  ])
    elif method == 'decorate':
         commands = " ".join([commonWekaCmds
                   , 'weka.classifiers.meta.Decorate'
                   , sharedCmds
                   , '-E', '50'
                   , '-R', '1.0'
                   , '-S', '1'
                   , '-I', '50'
                   , '-W', 'weka.classifiers.trees.RandomForest'
                   , '--' # the commands below go to the -w classifier
                   , '-I', '10' # number of trees
                   , '-K', '0' # number of features selected for each tree, 0 defaults to log2(number of features) + 1
                  ])
    elif method == 'naiveBayes':
         commands = " ".join([commonWekaCmds
                   , 'weka.classifiers.bayes.NaiveBayes'
                   , sharedCmds
                  ])
    else :
        warning("Unrecognized ML method selected. Quiting.")
        sys.exit(exitCoder.encode('usageError'))
    
    p = subprocess.Popen(pathCmd + ' && ' + commands, shell = True, stdout = subprocess.PIPE)
    output = p.stdout.read()
    return(output)

def parseWekaPredictions(rawPredictions, predictionNames, plotting = False, plotFilePrefix = None):
# example of the data
#
#=== Predictions on test data ===
#
# inst#     actual  predicted error distribution
#     1     1:True    2:False   +   0.4,*0.6
#     2    2:False    2:False       0,*1
#     3    2:False    2:False       0.2,*0.8
    lines = getNonBlankLines(rawPredictions)
    outlines = []
    for line in lines:
        fields = line.rstrip().split()
        if fields[0] == "===" or re.match("inst", fields[0]) :
            continue
        # the order is drugIDName, predictions, actual, positive portion value
        # the prediction names HAVE to be in the same order as the predictions, but for weka that is the case
        # I'll just make them a single tab separated line
        outlines.append("\t".join([predictionNames.pop(0), fields[2].split(":")[1], fields[1].split(":")[1], re.sub("\*", "" , fields[-1].split(",")[0])]))
    if plotting:
        analyzeResultsAndPlot(outlines, plotFilePrefix)
    return outlines

def analyzeResultsAndPlot(resultsData, plotFilePrefix):
    # proces resultsData to format needed
    # hacky way to turn True sting into into via bool
    y_actual = [int(str2bool(row.split("\t")[2])) for row in resultsData]
    y_predScore = [float(row.split("\t")[3]) for row in resultsData]
    plotPRCurve(y_actual, y_predScore, plotFilePrefix)
    plotROC(y_actual, y_predScore, plotFilePrefix)

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def plotPRCurve(y_actual, y_predScore, plotFilePrefix):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    import matplotlib.pyplot as plt
    # Compute Precision-Recall and plot curve
    precision, recall, _ = precision_recall_curve(y_actual, y_predScore)
    average_precision = average_precision_score(y_actual, y_predScore)
    
    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision))
    plt.legend(loc="lower left")
    plt.savefig(plotFilePrefix + "_precisionRecallCurve.png")    
    plt.close()

def plotROC(y_actual, y_predScore, plotFilePrefix):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    # Compute Precision-Recall and plot curve
    fpr, tpr, _ = roc_curve(y_actual, y_predScore)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC
    plt.clf()
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(plotFilePrefix + "_ROC.png")    
    plt.close()

def processWekaStats(wekaStats):
    lines = getNonBlankLines(wekaStats)
    outFields = []
    for line in lines:
        fields = line.rstrip().split()
        if re.match("Correctly", fields[0]) or re.match("Incorrectly", fields[0]) :
            outFields.append(fields[-2])
    return outFields

def getNonBlankLines(longString):
    lines = longString.split("\n")
    # remove blank lines
    return([line for line in lines if line])

# read all of the severity data into an easy to access hash
# there are only 1-2k of these, so putting in memory isn't awful
def retreiveSeverityScores(severityFile):
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
    verboseOut("Loaded ", len(list(allSev.keys())), " side effect severity scores")
    return allSev

def scoreEveryDrug(allProcessedWekaPredictions, sevScores):
    noSevScore = []
    # a dictionary of lists, keyed on drugs, and the values the scores, in order of the SEs
    seRealSpecificScores = defaultdict(list)
    sePredSpecificScores = defaultdict(list)
    seOrder = []
    verboseOut("Weka predictions for: ", len(list(allProcessedWekaPredictions.keys())), " side effects.")
    for predSe in allProcessedWekaPredictions.keys():
        # only get a score for those side effects with a severiry score
        if predSe not in list(sevScores.keys()):
            noSevScore.append(predSe)
            continue
        # the pred score is keyed by the wsa and the value is the probability of that drug having that SE * the severity of that drug
        for r in allProcessedWekaPredictions[predSe]:
            row = r.split("\t")
            sePredSpecificScores[row[0]].append(float(row[3])*sevScores[predSe]['score'])
            realScore = 0
            if row[2] == 'True':
                realScore = 1
            seRealSpecificScores[row[0]].append(realScore*sevScores[predSe]['score'])
        seOrder.append(predSe)
    verboseOut("Unable to find prediction values for ", len(noSevScore), " side effects which have a severity score.")
    return sePredSpecificScores, seRealSpecificScores, seOrder

def reportScores(predictScores, realScores, sideEffectOrder, detailedScores, filePrefix):
    predictSums = { k: sum(v) for k,v in predictScores.items() }
    realSums = { k: sum(v) for k,v in realScores.items() }
    
    with open(filePrefix + '_predictSums.tsv', 'w') as f:
        f.write("\n".join([ str(k) + "\t" + str(v) for k,v in predictSums.items()]) + "\n")
    with open(filePrefix + '_realSums.tsv', 'w') as f:
        f.write("\n".join([ str(k) + "\t" + str(v) for k,v in realSums.items()]) + "\n")
    
    if detailedScores:
        with open(filePrefix + '_predictDetailedScores.tsv', 'w') as f:
            f.write("wsa\t" + "\t".join(sideEffectOrder)+"\n")
            for k,v in predictScores.items():
                f.write( str(k) + "\t" + "\t".join(  [str(i) for i in v ] ) + "\n")
        with open(filePrefix + '_realDetailedScores.tsv', 'w') as f:
            f.write("wsa\t" + "\t".join(sideEffectOrder) + "\n")
            for k,v in realScores.items():
                f.write( str(k) + "\t" + "\t".join(  [str(i) for i in v ] ) + "\n")


#=================================================
# Read in the arguments/define options
#=================================================
exitCoder = ExitCoder()

opts = OptionParser()

usage = "usage: %prog [options] [input] This will combine side effect files on the side effect name"

opts = OptionParser(usage=usage)

opts.add_option("-l", help="Labels of drugs that have SEs. A TSV where each row is a drug, and each column is a SE that will be predicited.")

opts.add_option("-f", help="Full path of file with the molecular sub structures (bits) to use as features.")

opts.add_option("-s", help="Full path of file with the severity scores for each side effect.")

opts.add_option("--ws_id", help="workspace ID")

opts.add_option("--regression_method", help="Method to predict frequency of side effect instead of presence/absence. Options: ")

opts.add_option("--class_method", help="Classification method. Options: RF (random forest), RF_tune (tune the number of features to use), decorate (running RF), naiveBayes")

opts.add_option("--training", help="Portion of drugs to use for training set [0-1, DEAFULT: 0.7]", default = '0.7')

opts.add_option("--minSEPortion", help="Portion of drugs that must be in the minority to try to predict [0-1, DEAFULT: 0.05]", default = '0.05')

opts.add_option("--minFtPortion", help="Portion of drugs that must have an attribute or feature in order to use the attribute for predictions [0-1, DEAFULT: 0.005]", default = '0.005')

opts.add_option("--testingStats", action="store_true", dest="testing", help="Rather than make predictions, report testing stats.")

opts.add_option("--plot", action="store_true", dest="plot", help="If making predictions, summary stats should be plotted.")

opts.add_option("--keepAll", action="store_true", dest="keepFiles", help="Keep all intermediate files")

opts.add_option("--writePreds", action="store_true", dest="writePreds", help="Write out predictions. One file per side effect.")

opts.add_option("--detailedScores", action="store_true", dest="detailedScores", help="Write out a score for each drug/side effect combination.")

opts.add_option("-v", action="store_true", dest="verbose", help="Print out status reports")

options, arguments = opts.parse_args()

# return usage information if no argvs given

if len(sys.argv) < 5:
    opts.print_help()
    sys.exit(exitCoder.encode('usageError'))


##### INPUTS AND OUTPUTS AND SETTINGS #####
wsId = int(options.ws_id)
molBitsFile = options.f
trainingPortion = float(options.training)
minSePortion = float(options.minSEPortion) # if the SE isn't at a minimal prevalance we're going to have a really hard time building a decent model
minFeaturePortion = float(options.minFtPortion) # if a feature is super rare, there isn't much use in including it as it isn't generalizable
if options.class_method and not options.regression_method:
    mlMethod = options.class_method
    classification = True
elif options.regression_method and not options.class_method:
    mlMethod = options.regression_method
    classification = False
else:
    warning("Must select regression or classification method, and not both")
    opts.print_help()
    sys.exit(exitCoder.encode('usageError'))

if options.keepFiles:
    cleanUp = False
else:
    cleanUp = True

if options.testing:
    forTesting = True
else:
    if options.plot:
        plotting = True
    else:
        plotting = False
    forTesting = False


#==================================================================================================
# Main
#==================================================================================================

# now read in all drugs for the ws and convert to the WSA with side effect labels
drugsWithSEs, sideEffects = convertPCIDtoWSAWithSE(options.l, getWSAtoPCID(wsId))

# retreive the features to be used (first pass, moleular bits)
features, drugIDs, bits = getFeatures(molBitsFile, list(drugsWithSEs.keys()), minFeaturePortion)

# build the majority of the arff, which is the molecular bits 
mainAttributes = buildAttributes(bits)

# add specific SE label to arff
# make training and testing sets 

filePrefix = setupOutDir(options.ws_id)

if forTesting:
    outFile = filePrefix + "_testingStats.txt"
    allStats = []

# keyed on the se
allProcessedWekaPredictions = makePreds(sideEffects, drugsWithSEs, mainAttributes, features, drugIDs, trainingPortion, forTesting, mlMethod, classification, cleanUp, filePrefix)

# this gives a dictionary of dinctionaries, keyed on the SE UMLS, then 'name' or 'score'
sevScores = retreiveSeverityScores(options.s)

# now I combine these 2, the result being 2 hashes keyed by the drug ID, and the value the severity score
# one is a predicted score, and the other, where possible is the empirical score
#predictScores, realScores, sideEffectOrder = scoreEveryDrug(allProcessedWekaPredictions, sevScores)
reportScores(scoreEveryDrug(allProcessedWekaPredictions, sevScores), options.detailedScores, filePrefix)

verboseOut("Finsihed successfully")
