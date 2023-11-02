#!/usr/bin/env python3
import os, sys

from path_helper import PathHelper

from collections import defaultdict,Counter

# created 10.Feb.2016 - Aaron C Daugherty - twoXAR
# commonly used defintions used to run weka and evaluate how it does

debugging = False

def report_multiple_attr_stats(attrs_selected, file):
    adr_atr_stats = {adr : core_report_attr_stats(attrs_selected[adr]) for adr in attrs_selected.keys()}
    with open(file, 'w') as f:
        f.write("\t".join(['ADR', 'attr,model_por']) + "\n")
        for adr in adr_atr_stats.keys():
            out = []
            for tup in adr_atr_stats[adr]['attr_cnt'].most_common():
                out.append(",".join([tup[0], str(float(tup[1]) / adr_atr_stats[adr]['model_cnt'])]))
            f.write(adr + "\t" + ';'.join(out) + "\n")

def report_attr_stats(attrs_selected, file):
    atr_stats = core_report_attr_stats(attrs_selected)
    with open(file, 'w') as f:
        f.write("\t".join(['Attribute', 'portion of all models']) + "\n")
        out = [ "\t".join([tup[0]
                           , str(float(tup[1]) / atr_stats['model_cnt'])
                          ])
               for tup
               in atr_stats['attr_cnt'].most_common()
              ]
        f.write("\n".join(out) + "\n")

def core_report_attr_stats(dol):
    atr_stats = {}
    all_attrs = []
    atr_stats['model_cnt'] = 0
    for sl in dol['slctd_attrs']:
        all_attrs += [atr for atr in sl]
        atr_stats['model_cnt'] += 1
    atr_stats['attr_cnt'] = Counter(all_attrs)
    return atr_stats

def process_pred_stats(prefix, dol_stats, dol_attrs):
    file = prefix + '_testingStats.txt'
    cvstats = False
    cv_conf = False
    d = {}
    attr_data = None
    with open(file, 'r') as f:
        lines = f.readlines()
    while len(lines) > 0:
        l = lines.pop(0)
        if l.startswith("Attribute Subset Evaluator"):
            attr_data = {}
            attr_data['intl_attr_cnt'] = str(l.rstrip().split()[-2])
        elif l.startswith("Selected attributes:"):
            attr_data['slctd_attr_cnt'] = str(l.rstrip().split()[-1])
            attr_data['slctd_attrs'] = []
            for i in range(int(attr_data['slctd_attr_cnt'])):
                attr_data['slctd_attrs'].append(lines.pop(0).strip())
        elif l.startswith("Out of bag error:"):
            d['OOB'] = float(l.rstrip().split()[-1])
        elif cvstats and l.startswith("Correctly Classified Instances"):
            d['Accuracy'] = float(l.rstrip().split()[-2])/100.0
        elif cvstats and l.startswith("Kappa"):
            d['Kappa'] = float(l.rstrip().split()[-1])
        elif cvstats and cv_conf and l.lstrip()[:1].isdigit():
            lst = l.lstrip().split()[0:2]
            fp = int(lst[0])
            tn = int(lst[1])
            all_neg = tn + fp
            d['TNR'] = float(tn) / all_neg
        elif cvstats and l.lstrip()[:1].isdigit():
            cv_conf = True
            lst = l.lstrip().split()[0:2]
            tp = int(lst[0])
            fn = int(lst[1])
            all_pos = tp + fn
            d['TPR'] = float(tp) /all_pos
        elif l.startswith("=== Stratified cross-validation ==="):
            cvstats = True
    if attr_data is not None:
        updated_attrs = update_dol(attr_data, dol_attrs)
    else:
        updated_attrs = None
    if not cv_conf:
        sys.stderr.write(file + ' did not contain info for cv_conf. That is suspicious. No CV stats will be reported\n')
        return None, updated_attrs
    d['F1'] = float(2.0*tp) / (2.0*tp + float(fp) + float(fn))
    updated_stats = update_dol(d, dol_stats)
    return updated_stats, updated_attrs

def update_dol(d, dol):
    for k in d.keys():
        if k not in list(dol.keys()):
            dol[k] = []
        dol[k].append(d[k])
    return dol

def plot_cv_stats_boxplot(data, out_file, plotly = False):
    vals = []
    ks = []
    for k in sorted(data):
        vals.append(data[k])
        ks.append(k)
    print(ks)
    print(vals)
    if plotly:
        from dtk.plot import PlotlyPlot
        pp = PlotlyPlot([
                    dict(type='box'
                          , y = vals[i]
                          , name = ks[i]
                          , boxpoints = 'all'
                          , jitter = 0.5
                          , boxmean = 'sd'
                          , marker = dict(size = 3, opacity = 0.5)
                         )
                    for i in range(len(ks))],
                    {'title':'Cross-validation stats'},
            )
        pp.save(out_file)
    else:
        import numpy as np
        import matplotlib.pyplot as plt
        fig = plt.figure()
        bp = plt.boxplot(vals, 1)
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,0.0,1.05))
        plt.xticks([i+1 for i in range(len(ks))], ks)
        # add median values for easy interpretation:
        for i,v in enumerate(vals):
            plt.text(i+0.7, 1, str(np.round(np.median(v), 3)))
        fig.savefig(out_file
                   , format = 'PNG'
                   )

def split_train_test(allDataD, allLabelsD, trainingPortion, balanced=True, min_samples = 1):
    import random
    from math import ceil

    training = defaultdict(dict)
    testing = defaultdict(dict)
    trainLabels = {}
    testLabels = {}
    
    if balanced:
        # randomly select from all labels to have a balanced set (equal number of postive and negative cases)
        drugsToUse, leftOvers = balance_binary_labels(allLabelsD)
        if not drugsToUse:
            return False, False, False, False, False
    else:
        drugsToUse = allLabelsD
        leftOvers = None
    # split the balanced set into positive and negative sets and then training and testing
    class1, class2 = list(set(drugsToUse.values()))[0:2]
    class1_ToUse = {}
    class2_ToUse = {}
    for k,v in drugsToUse.items():
        if v == class1:
            class1_ToUse[k] = v
        elif v == class2:
            class2_ToUse[k] = v
    trainingInds = map(int, random.sample(range(len(list(drugsToUse.keys()))//2), int(ceil(trainingPortion*len(list(drugsToUse.keys()))//2))))
    class1_trainingDrugs = []
    class2_trainingDrugs = []
    for i in trainingInds:
        class1_trainingDrugs.append(list(class1_ToUse.keys())[i])
        class2_trainingDrugs.append(list(class2_ToUse.keys())[i])
    assert len(class2_trainingDrugs) == len(class1_trainingDrugs)
    trainingDrugs = class1_trainingDrugs + class2_trainingDrugs
    for drug in drugsToUse.keys():
        if drug in trainingDrugs:
            training[drug] = allDataD[drug]
            trainLabels[drug] = allLabelsD[drug]
        else:
            testing[drug] = allDataD[drug]
            testLabels[drug] = allLabelsD[drug]
    if len(training) < min_samples:
        training, trainLabels, leftOvers = fillout_samples(training, trainLabels, allDataD, leftOvers, min_samples)
    return training, testing, trainLabels, testLabels, leftOvers

def fillout_samples(trainingData, trainingLabels, allDataD, leftOvers, min_samples):
    import random
    ks = random.sample(list(leftOvers.keys()), min_samples - len(trainingData))
    subset_D = {}
    subset_L = {}
    for k in ks:
        subset_D[k] = allDataD[k]
        subset_L[k] = leftOvers[k]
        del leftOvers[k]
    trainingData = merge_dicts(trainingData, subset_D)
    trainingLabels = merge_dicts(trainingLabels, subset_L)
    return trainingData, trainingLabels, leftOvers

def generateArffs(ftsToUse, dd, labels, labelName, file, label_type = None, arffName="", fillin = False, provided_attr_data=None):
    # build the header section
    if not provided_attr_data:
        atrTypes, atr_defaults, final_atrs = check_atr_types(ftsToUse, dd, labels, label_type)
    else:
        atrTypes = provided_attr_data[0]
        atr_defaults = provided_attr_data[1]
        final_atrs = provided_attr_data[2]
    atrs_and_label = final_atrs[:]
    atrs_and_label.append(labelName)
    attributes = [[atrs_and_label[i], atrTypes[i]] for i in range(len(atrTypes))]
    
    # build the important part: the data
    # while random, this gives us an order to keep the rows in
    drugIDs=list(labels.keys())
    # now build up the data, basically just pushing everything together
    dataList = []
    for drug in drugIDs:
        toAdd = []
        for atr in final_atrs: # the last one is the label, so that shouldn't be processed
            fillin_field = '?'
            if fillin:
                fillin_field = atr_defaults[atr]
            if dd[drug] is not None and atr in list(dd[drug].keys()):
                toAdd.append(dd[drug][atr])
            else:
                toAdd.append(fillin_field)
        # I was seeing some drugs with no attributes, so I'll do a quick check to ensure that not all of the fields are ?
        if toAdd.count('?') == len(toAdd): 
            continue
        dataList.append(",".join([str(i) for sublist in [toAdd, [labels[drug]]] for i in sublist]))
    
    write_arff(labelName, attributes, dataList, file, description=arffName)
    if not provided_attr_data:
        return [atrTypes, atr_defaults, final_atrs]

def check_atr_types(ftsToUse, dd_data, labels, label_type):
    ftTypes = []
    ftDefaults = {}
    finalFts = []
    for ft in ftsToUse:
        allFts = [dd_data[k][ft] for k in dd_data.keys() if dd_data[k] is not None and ft in list(dd_data[k].keys())]
        if len(allFts) == 0:
#            print "something went wrong with " + ft + ". No values were found."
            continue
        attr_type, default = getAttrType(allFts)
        if attr_type is None:
#            print "something went wrong with " + ft + " no non-missing values were found."
            continue
        elif not attr_type:
            allTypes = list(set([str(type(n)) for n in allFts]))
            msg = "Unable to identify attribute type of " + str(ft) + ". Found multiple types:" + ",".join(allTypes) + ". Features: "+"; ".join(allFts)
            raise RuntimeError(msg)
        ftTypes.append(attr_type)
        ftDefaults[ft] = default
        finalFts.append(ft)
    if label_type is not None:
        ftTypes.append(label_type)
    else:
        lab_type, temp = getAttrType(list(labels.values()))
        ftTypes.append(lab_type)
    return ftTypes, ftDefaults, finalFts

def getAttrType(ftList, try_to_convert=True):
    py_type = False
    for ft in ftList:
        if ft != '?':
            py_type = type(ft)
            break
    if not py_type: # i.e. we never found a feature that wasn't missing
        return None, None
    # verify that all of our types are the same
    if not all(isinstance(n, py_type) for n in [i for i in ftList if i != '?']):
        if try_to_convert:
            # try converting the atr
            conv_ft_list = [convertTypes(i) for i in ftList if i != '?' and isinstance(i, str)]
            return getAttrType(conv_ft_list, try_to_convert=False)
        else:
            return False, False
    # convert the py type to the type expected in an arff
    if py_type == int:
        return 'INTEGER', 0
    elif py_type == bool:
        return ['True', 'False'], False
    elif isinstance(py_type,str):
        return 'STRING', ""
    return 'REAL', 0.0

# I switched away from regex b/c they're error prone if you aren't careful and always slow
# the'problem' with the approach I used below is that a lot of things can be interpretted as a float.
# see: http://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python
# To avoid that I try everything besides float and string first, then with it narrowed down to those two, try float first
def convertTypes(element):
    boolStrings = ['true', 'false', 't', 'f', 'yes', 'no', 'y', 'n']
    if element.isdigit():
        return int(element)
    elif element.lower() in boolStrings:
        return str2bool(element)
    try:
        fltd = float(element)
        return fltd
    except ValueError:
        return str(element)

def readInBits(bitsFileName):
    d = defaultdict(list)
    with open(bitsFileName, 'r') as f:
        bits = f.readline().rstrip().split(",")
        idType = bits.pop(0)
        for line in f:
            fields=line.rstrip().split(",")
            d[fields[0]] = [int(i) for i in fields[1:]]
    return bits, d

def addFilteredFeatures(existingData, preFiltFtData, preFiltFtNames):
    for k in existingData.keys():
        for i in range(len(preFiltFtNames)):
            if k in list(preFiltFtData.keys()):
                existingData[k][preFiltFtNames[i]] = preFiltFtData[k][i]
            else:
                # generate ? the length of the preparsed data
                existingData[k][preFiltFtNames[i]] = '?'
    return existingData

def get_binary_label_counts(labels):
    instanceTotals = Counter(labels)
    try:
        largestClass,largestClassAmount = instanceTotals.most_common(1)[0]
        smallestClass,smallestClassAmount = instanceTotals.most_common(2)[1]
        return largestClass, largestClassAmount, smallestClass, smallestClassAmount
    except IndexError:
        return None, None, None, None

def balance_binary_labels(labelD_orig, verbose=False):
    import random
    # I didn't think python would do this, but the very original dictionary was getting modigied (2 function calls up)
    # to avoid that, I'm just creating a copy
    labelD = labelD_orig.copy()
    drugs = list(labelD.keys())
    labels = list(labelD.values())
    largestClass, largestClassAmount, smallestClass, smallestClassAmount = get_binary_label_counts(labels)
    if largestClass is None:
        return False, False
    if verbose or debugging:
        sys.stderr.write("Downsampling " + str(largestClass) + " from " + str(largestClassAmount) + " to " + str(smallestClassAmount) + " to create a balanced set.\n")
    if smallestClassAmount == largestClassAmount:
        print("smallestClassAmount == largestClassAmount:", smallestClassAmount, largestClassAmount) 
        return labelD, False
    lcInds = [i for i, x in enumerate(labels) if x == largestClass]
    to_rm = [lcInds[ind] for ind in random.sample(range(len(lcInds)), largestClassAmount - smallestClassAmount)]
    leftoverLabels = {}
    for ind in to_rm:
        leftoverLabels[drugs[ind]] = labels[ind]
        del labelD[drugs[ind]]
    assert len(list(labelD.keys())) == smallestClassAmount*2
    assert len(list(leftoverLabels.keys())) == largestClassAmount-smallestClassAmount
    return labelD, leftoverLabels

def balanceBinarySet(data, verbose=False):
    labels = [row.split(",")[-1] for row in data]
    instanceTotals = Counter(labels)
    largestClass,largestClassAmount = instanceTotals.most_common(1)[0]
    smallestClass,smallestClassAmount = instanceTotals.most_common(2)[1]
    if verbose or debugging:
        message("Downsampling ", largestClass, " from ", largestClassAmount, " to ", smallestClassAmount, " to create a balanced set")
    if smallestClassAmount == largestClassAmount:
        return data, None
    largestClassDownsampled = []
    removedLargestClass = []
    lcInds = [i for i, x in enumerate(labels) if x == largestClass]
    toKeepInds = map(int, random.sample(range(len(lcInds)), smallestClassAmount))
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

# this takes advantage of the arff package to set up the file, but just appends a csv to the arff b/c that's much faster and easier
def write_arff(relation, attributes, dataAsListOfStrings, file, description=""):
    import arff
    out = {
          'description':description,
          'relation':relation,
          'attributes':attributes,
 # I have to put something in data or it gets all upset, but it's super slow, so I give it the bare minimum and then filter below
          'data':attributes[0:1]
          }
    empty_arff = arff.dumps(out)
    with open(file, 'w') as f:
        # I added dummy filler data just to keep liac-arff happy, but I don't want to report that
        arffToReport=[]
        for line in empty_arff.split("\n"):
            arffToReport.append(line)
            if line.startswith("@DATA"):
                break
        # I don't know why, but liac-arff ends all files with 3 comments, so I'll just keep that
        f.write("\n".join([i for sublist in [arffToReport, dataAsListOfStrings, ["%"]*3] for i in sublist]) + "\n")

def getFirstPortionInList(l):
    c = Counter(l)
    return(float(list(c.values())[0]) / float(sum(c.values())))

def runWeka(method, trainingFile, predictionFile, testing = False, model_name=None, build = False, cost_list = None):
    import subprocess
    pathCmd = "export CLASSPATH=" + PathHelper.weka+":."
    commonWekaCmds = " ".join(['java', '-Xmx6G']) # max memory given to the classifier # this should be parameterized somewhere
    # Not currently saving models, but we could: '-d', modelFile
    # the last two result in the predictions being reported (but not the stats)
    sharedCmdsList = []
    if trainingFile is not None:
        sharedCmdsList += ['-t', trainingFile]
    if build:
        assert model_name,"Must provide model name if build = True"
        sharedCmdsList += ['-d', model_name]
    else:
        if model_name is not None:
            # kind of dumb way to do this, but if there is a model, this is the only time we don't use 
            sharedCmdsList = []
            sharedCmdsList += ['-l',  model_name]
        sharedCmdsList += ['-T', predictionFile]
    if not testing:
        sharedCmdsList += ['-p', '0', '-distribution']
    
    sharedCmds = " ".join(sharedCmdsList)
    
    if not build and model_name is not None:
        useModel = True
    else:
        useModel = False
    if method == 'RF':
        commandsList = [commonWekaCmds, 'weka.classifiers.trees.RandomForest', sharedCmds]
        if not useModel:
            commandsList = commandsList + ['-I', '10', '-K', '0', '-S', '1']
        commands = " ".join(commandsList)
                   #, '-I', '10' # number of trees
                   #, '-K', '0' # number of features selected for each tree, 0 defaults to log2(number of features) + 1
                   #, '-S', '1' # S is the seed
    elif method == 'RF_weight':
        commandsList = [commonWekaCmds, 'weka.classifiers.meta.CostSensitiveClassifier', sharedCmds]
        if not useModel:
            if cost_list:
                cost_mat = '"[' + str(cost_list[0]) + ' ' +  str(cost_list[1]) + '; ' + str(cost_list[2]) + ' ' +  str(cost_list[3]) + ']"'
            else:
                sys.stderr.write("WARNING: CostSensitiveClassifier expects a cost list. Quiting.\n")
            commandsList = commandsList + ['-cost-matrix', cost_mat, '-S', '1', '-W',
                                            'weka.classifiers.trees.RandomForest', '--',
                                            '-I', '10', '-K', '0'
                                           ]
        commands = " ".join(commandsList)
    elif method == 'attrSel_RF':
        commandsList = [commonWekaCmds, 'weka.classifiers.meta.AttributeSelectedClassifier', sharedCmds]
        if not useModel:
            commandsList = commandsList + ['-E', '"weka.attributeSelection.CfsSubsetEval -M"', '-S',
                                            '"weka.attributeSelection.BestFirst -D 1 -N 5"', '-W',
                                            'weka.classifiers.trees.RandomForest', '--', '-I', '10'
                                          ]
        commands = " ".join(commandsList)
    elif method == 'attrSel_lr':
        commandsList = [commonWekaCmds, 'weka.classifiers.meta.AttributeSelectedClassifier', sharedCmds]
        if not useModel:
            commandsList = commandsList + ['-E', '"weka.attributeSelection.CfsSubsetEval -M"', '-S',
                                            '"weka.attributeSelection.BestFirst -D 1 -N 5"', '-W',
                                            'weka.classifiers.functions.Logistic', '--', '-R', '1.0E-8', '-M', '-1'
                                          ]
        commands = " ".join(commandsList)
    elif method == 'logistic':
        commandsList = [commonWekaCmds, 'weka.classifiers.functions.Logistic', sharedCmds]
        if not useModel:
            commandsList = commandsList + [ '-R', '1.0E-8', '-M', '-1']
        commands = " ".join(commandsList)
    elif method == 'svm':
        commandsList = [commonWekaCmds, 'weka.classifiers.functions.SMO', sharedCmds]
        if not useModel:
            commandsList += '-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K'.split()
            commandsList.append('"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0"')
        commands = " ".join(commandsList)
    elif method == 'RF_tune':
        commandsList = [commonWekaCmds, 'weka.classifiers.meta.CVParameterSelection', sharedCmds]
        if not useModel:
            commandsList = commandsList + ['-P', '"K 1 21 2"', '-S', '1', '-W',
                                            'weka.classifiers.trees.RandomForest', '--', '-I', '10'
                                          ]
        commands = " ".join(commandsList)
#                   , '-P', '"K 1 21 2"' # param'ize the number of features b/t 1 and 21
    elif method == 'decorate':
        commandsList = [commonWekaCmds, 'weka.classifiers.meta.Decorate', sharedCmds]
        if not useModel:
            commandsList = commandsList + ['-E', '50', '-R', '1.0', '-S', '1', '-I', '50', '-W',
                                            'weka.classifiers.trees.RandomForest', '--', '-I',
                                            '10', '-K', '0'
                                          ]
        commands = " ".join(commandsList)
    elif method == 'naiveBayes':
        commandsList = [commonWekaCmds, 'weka.classifiers.bayes.NaiveBayes', sharedCmds]
        commands = " ".join(commandsList)
    else :
        raise RuntimeError("Unrecognized ML method selected")
    p = subprocess.Popen(pathCmd + ' && ' + commands, shell = True, stdout = subprocess.PIPE)
    output = p.stdout.read().decode('utf8')
    return(output)

def run_weka_infogain(arff):
    import subprocess
    pathCmd = "export CLASSPATH=" + PathHelper.weka+":."
    commonWekaCmds = " ".join(['java', '-Xmx6G']) # max memory given to the classifier # this should be parameterized somewhere
    commandsList = [commonWekaCmds, 'weka.attributeSelection.InfoGainAttributeEval', '-i', arff, '-s', 'weka.attributeSelection.Ranker']
    commands = " ".join(commandsList)
    p = subprocess.Popen(pathCmd + ' && ' + commands, shell = True, stdout = subprocess.PIPE)
    output = p.stdout.read().decode('utf8')
    return(output)


def condense_binary_weka_preds(l):
# example of the data in each list entry and what the output should look like
#
#=== Predictions on test data ===
#
# inst#     actual  predicted error distribution
#     1     1:True    2:False   +   0.4,*0.6
#     2    2:False    2:False       0,*1
#     3    2:False    2:False       0.2,*0.8
    actuals = {} # keyed by instance number
    firstclass_distributions = defaultdict(list) # keyed by instance number, the value is a list of the first class probability
    for output in l:
        for line in output.split("\n"):
            fields = line.rstrip().split()
            try:
                tmp = int(fields[0])
            except (ValueError, IndexError):
                continue
# I hope the except might be faster 
#            if fields[0] == "===" or fields[0].strip().startswith("inst") :
####
### I'm disabling this check to make things a little faster and b/c I know I don't see this error anymore
####
#            elif fields[0] in actuals.keys() and fields[1] != actuals[fields[0]]:
#                sys.stderr.write("WARNING: Inconsistent labels found for:" + line + "\n")
#                sys.exit(exitCoder.encode('usageError'))
#            else:
            actuals[fields[0]] = fields[1]
            firstclass_distributions[fields[0]].append(float(fields[-1].split(",")[0].replace('*', '')))
    output = []
    class_labels = list(set(actuals.values()))
    first_class_label = None
    second_class_label = None
    for c in class_labels:
        if c.startswith('1:'):
            first_class_label = c
        elif c.startswith('2:'):
            second_class_label = c
    if first_class_label and first_class_label == "1:?": # in this case we're working with predictions, and I only care about the distribution anyway, but I do need a filler
        second_class_label = '2:?' 
    elif not second_class_label or not first_class_label:
        sys.stderr.write("WARNING: 2 labels were not found:" + " ".join(class_labels) + ".\nAttempting to guess...\n")
        if not second_class_label and first_class_label == "1:True":
                second_class_label = "2:False"
        elif not first_class_label and second_class_label == "2:False":
                first_class_label = "1:True"
        else:
            sys.stderr.write("WARNING: Unable to guess labels. Giving up.\n")
            return None
    for inst in sorted(actuals):
        first_cls_avg = float(sum(firstclass_distributions[inst])) / len(firstclass_distributions[inst])
        second_cls_avg = 1.0 - first_cls_avg
        if first_cls_avg >= second_cls_avg:
            first_cls_avg = '*' + str(first_cls_avg)
            pred = first_class_label
        else:
            second_cls_avg = '*' + str(second_cls_avg)
            pred = second_class_label
        error = ""
        if pred != actuals[inst]:
            error = "+"
        dist = str(first_cls_avg) + "," + str(second_cls_avg)
        output.append(" ".join([inst, actuals[inst], pred, error, dist]))
    return "\n".join(output)

def analyze_report_weka(wekaOut, forTesting, plotting, predictionDrugIds, writePreds, outFilePrefix, write_stats = True, full_stats_report = True):
    if forTesting:
        if write_stats:
            outFile = outFilePrefix + "_testingStats.txt"
            stats = processWekaStats(wekaOut, full_stats_report)
            with open(outFile, 'w') as f:
                f.write(stats + "\n")
    else:
        if plotting:
            plottingPrefix = outFilePrefix + "_evalPlots"
        else:
            plottingPrefix = None
        # a requirement of this is that the predictionDrugIds are in the same order as the predictions in wekaOut. Luckily that is the deafult for Weka.
        allPreds, eval_stats = parse_weka_predictions(wekaOut, predictionDrugIds, plotting, plottingPrefix)
        if writePreds:
            outFile = outFilePrefix + "_predictions.txt"
            with open(outFile, 'w') as f:
                f.write("\n".join(["\t".join(sl) for sl in allPreds]) + "\n")
        return allPreds, eval_stats

def parse_weka_predictions(rawPredictions, predictionNames, plotting = False, plotFilePrefix = None):
# example of the data
#
#=== Predictions on test data ===
#
# inst#     actual  predicted error distribution
#     1     1:True    2:False   +   0.4,*0.6
#     2    2:False    2:False       0,*1
#     3    2:False    2:False       0.2,*0.8
    lines = get_non_blank_lines(rawPredictions)
    outlines = []
    for line in lines:
        fields = line.rstrip().split()
        if fields[0] == "===" or fields[0].strip().startswith("inst") :
            continue
        # the order is drugIDName, predictions, actual, positive portion value
        # the prediction names HAVE to be in the same order as the predictions, but for weka that is the case
        # I'll just make them a single tab separated line - I decided not to do that later so that I can return the predictions to the calling program, if wanted
        outlines.append([predictionNames.pop(0), fields[2].split(":")[1], fields[1].split(":")[1], fields[-1].split(",")[0].replace('*', '')])
    # placeholders so I have the right number of arguments to return, regadless
    if plotting:
        eval_stats = evaluateResultsAndPlot(["\t".join(sl) for sl in outlines], plotFilePrefix)
    else:
        eval_stats = None
    return outlines, eval_stats

def evaluateResultsAndPlot(resultsData, plotFilePrefix, prob_thresh = 0.5):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    # proces resultsData to format needed
    # hacky way to turn True sting into into via bool
#    y_actual = [int(str2bool(row.split("\t")[2])) for row in resultsData]
    y_actual = [int(str2bool(row.split("\t")[2])) for row in resultsData]
    y_predScore = [float(row.split("\t")[3]) for row in resultsData]
    y_pred = [1 if i >= prob_thresh else 0 for i in y_predScore]
    acc = accuracy_score(y_actual, y_pred)
    pr = precision_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred)
#    roc_auc = sklearn.metrics.roc_auc_score(y_actual, y_predScore)
    pr_auc = plotPRCurve(y_actual, y_predScore, plotFilePrefix)
    roc_auc = plotROC(y_actual, y_predScore, plotFilePrefix)
    return {'accuracy': acc, 'precision': pr, 'recall': recall
           , 'F1': f1, 'ROC-AUC': roc_auc, 'PR-AUC': pr_auc}

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def plotPRCurve(y_actual, y_predScore, plotFilePrefix):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    import matplotlib.pyplot as plt
    # Compute Precision-Recall and plot curve
    try:
        precision, recall, _ = precision_recall_curve(y_actual, y_predScore)
        average_precision = average_precision_score(y_actual, y_predScore)
    except ValueError:
        sys.stderr.write("WARNING: Unable to plot Precision Recall curve. Likely because only one class was in Y.\n")
        return False
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
    return average_precision

def plotROC(y_actual, y_predScore, plotFilePrefix):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    # Compute Precision-Recall and plot curve
    fpr, tpr, _ = roc_curve(y_actual, y_predScore)
    try:
        roc_auc = auc(fpr, tpr)
    except ValueError:
        sys.stderr.write("WARNING: Unable to plot ROC curve. Likely because only one class was in Y.\n")
        return False
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
    return roc_auc

def processWekaStats(wekaStats, full = False):
    import re
    lines = get_non_blank_lines(wekaStats)
    if full:
        return "\n".join(lines)
    outFields = []
    for line in lines:
        fields = line.rstrip().split()
        if re.match("Correctly", fields[0]) or re.match("Incorrectly", fields[0]) :
            outFields.append(fields[-2])
    return "\t".join(outFields)

def get_non_blank_lines(longString):
    lines = longString.split("\n")
    # remove blank lines
    return([line for line in lines if line])

def merge_dicts(dict1, dict2):
    if len(set(dict1.keys()) & set(dict2.keys())):
        sys.stderr.write("WARNING: Attempting to merge dictionaries with overlapping keys. This will lose data in the first dictionary.\n")
#        sys.stderr.write("Those keys are: " + "\t".join([str(i) for i in list(set(dict1.keys()) & set(dict2.keys()))]) + "\n")
    dict3 = dict1.copy()
    dict3.update(dict2)
    return dict3
