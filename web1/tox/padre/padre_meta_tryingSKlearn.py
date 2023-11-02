#!/usr/bin/python
from __future__ import print_function
import os, django, sys, argparse
from collections import defaultdict
#sys.path.insert(1,"../")
sys.path.insert(1,"../../")
sys.path.insert(1,"../../ML")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from path_helper import PathHelper
from algorithms.exit_codes import ExitCoder
import napa_build as nb
import run_eval_weka as rew
import run_eval_sklearn as res

# created 20.Apr.2016 - Aaron C Daugherty - twoXAR

# TO DO:
# report in a drug-centered way: all ADR probabilities for all drugs
#   Can then take this data into R and create average ROCs (for each drug) and overall all drugs all ADRs. Can also split it out by ADR
#      Might be able to do this in python

# This takes napa_predict output and generates ADR-specific feature matrices before running a classifier, via weka 

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

def get_just_filehandle(file):
    return os.path.splitext(os.path.basename(file))[0]

def get_atr_names(file_list, sep="\t"):
    atr_names = {}
    for file in file_list:
        filename = get_just_filehandle(file)
        with open(file, 'r') as f:
            header = f.readline().rstrip().split(sep)
            atr_names[filename] = [filename + "_" + i for i in header[2:]]
    return atr_names

def grep_out_lines(file, term):
    import subprocess as sp
    proc = sp.Popen(['grep', term, file], stdout=sp.PIPE)
    return proc.stdout.readlines()

def napa_file_to_dd(adr, file_list, drug_file, pred_file, atr_names, sep="\t"):
# these files should be in the following format:
# where all the fields after adr are features
#drug    adr     path_count      median_max_wt   median_median_wt        median_path_counts      median_OR       median_q
#DB00953 C0017152        5       0.051948051948  0.0181818181818 0.6     0.899420788979  0.931802366757
#
# in order to work with existing code we need some deep data structures. 
# keyed by ADR, the value is a dictionary keyed by drugID, and the value there are dictionaries keyed by the attribute name, and the final vlaue is for the matrix
    adr_data = defaultdict(dict)
    # it's easier to create empty versions than have multiple conditionals below
    pred_data = defaultdict(dict)
    drugs_to_pred = []
    if pred_file:
        # read in the drugs to predict
        drugs_to_pred = get_drug_set(pred_file)
    for file in file_list:
        filename = get_just_filehandle(file)
        for l in grep_out_lines(file, adr):
            fields = l.rstrip().split(sep)
            if adr != fields[1]:
                warning("Something went wrong with the grep_out:", adr, field[1])
            drug = fields[0]
            new_val = dict(zip(atr_names[filename], [rew.convertTypes(ft) for ft in fields[2:]]))
#            new_val = dict(zip(atr_names[filename], [ft for ft in fields[2:]]))
            if drug in drugs_to_pred:
                pred_data = update_dd(pred_data, drug, new_val)
            else:
                adr_data = update_dd(adr_data, drug, new_val)
    full_drug_set = get_drug_set(drug_file)
#    adr_data = fill_in_missing_drugs(list(full_drug_set - set(adr_data.keys()) - drugs_to_pred), adr_data)
#    if len(pred_data.keys()) > 0:
#        pred_data = fill_in_missing_drugs(list(drugs_to_pred - set(pred_data.keys())), pred_data)
    return adr_data, pred_data, drugs_to_pred

def fill_in_missing_drugs(tobeadded, dd):
    for d in tobeadded:
        dd[d] = None
    return dd

def update_dd(dd, drug, new_val):
    if drug in list(dd.keys()):
        dd[drug].update(new_val)
    else:
        dd[drug] = new_val
    return dd

def get_drug_set(drug_file):
    with open(drug_file, 'r') as f:
        full_drug_set = set(f.read().splitlines())
    return full_drug_set

def adr_file_to_dd(adr_file):
    dd = defaultdict(dict)
    with open(adr_file, 'r') as f:
        data_types = f.readline().rstrip().split("\t")
        for line in f:
            fields = line.rstrip().split("\t")
            drug = nb.convert_to_dbid(fields[0], current_type = data_types[0])
            if not drug:
                continue
# if we wanted to do more than binary, we could load in the weight(field[2]) here
# though we'd also have to adjust below when we add in the False 
            dd[fields[1].upper()][drug] = True
    return dd

def get_all_fts(drugAtr):
    return list(set([k for d in drugAtr.values() if d is not None for k in d.keys()]))

def add_predictions(drugAdrProb, predictions, adr):
    for sl in predictions:
        drugAdrProb[sl[0]][adr] = sl[3]
    return drugAdrProb

def report_final_preds(drugAdrProb, final_preds_file):
    with open(final_preds_file, 'w') as f:
        f.write("\t".join(["drug", "adr", "probability"]) + "\n")
        for drug in drugAdrProb.keys():
            for adr in drugAdrProb[drug].keys():
                f.write("\t".join([drug, adr, str(drugAdrProb[drug][adr])]) + "\n")

def report_aucs(d_roc, d_pr, file):
    with open(file, 'w') as f:
        f.write("\t".join(['ADR', 'ROC_AUC', 'Prec_Recall_AUC']) + "\n")
        for k in sorted(d_roc):
            f.write("\t".join([k, str(d_roc[k]), str(d_pr[k])]) + "\n")

def clean_up(l):
    for filename in l:
        try:
            os.remove(filename)
        except OSError:
            pass


if __name__=='__main__':
    #=================================================
    # Read in the arguments/define options
    #=================================================
    exitCoder = ExitCoder()
    
    arguments = argparse.ArgumentParser(description="Build models for predicting individual ADRs")
    
    arguments.add_argument("-i", nargs='+', help="All attribute files, e.g. multiple output from napa_predict")
    
    arguments.add_argument("-o", default="./", help="Prefix to put on all output files. DEAFULT: %(default)s")
    
    arguments.add_argument("--adrs", help="ADR file containing all drug-ADR pairings")
    
    arguments.add_argument("--predict", default=None, help="Use the model learned to predict ADRs for all drugs in this file")
    
    arguments.add_argument("--drugs", help="Original drug set used")
    
    arguments.add_argument("--trainingPortion", type = float, default=0.7, help="Portion of drugs to use for training set [0-1, DEAFULT: %(default)s]")
    
    arguments.add_argument("--no_test_eval", action="store_true", help="Do not run testing evaluation for each ADR")
    
    arguments.add_argument("--min_examples_predict", type = int, default=20, help="Minimum number of drugs with ADR to try to predict. DEAFULT: %(default)s]")
    
    arguments.add_argument("--min_examples_testEval", type = int, default=10, help="Minimum number of drugs that need to be in the test set to evaluate. DEAFULT: %(default)s]")
    
    arguments.add_argument("--ml_method", default='RF'
                     ,help="Machine learning method. Options: RF (random forest), RF_tune (tune the number of features to use), decorate (running RF), naiveBayes. DEAFULT: %(default)s")
    
    arguments.add_argument("--keep_all", action="store_true", help="Safe all ARFF files as well as individual ADR model files.")
    
    arguments.add_argument("--write_stats", action="store_true", help="Report stats, when possible")
    
    arguments.add_argument("--write_predictions", action="store_true", help="Report individual ADR predictions to separate files")
    
    arguments.add_argument("--verbose", action="store_true", help="Print out status reports")
    
    args = arguments.parse_args()
    
    
    # return usage information if no argvs given
    
    if not args.i or not args.adrs or not args.drugs:
        arguments.print_help()
        sys.exit(exitCoder.encode('usageError'))
    
    final_preds_file = args.o + "_final_predictions.tsv"
    auc_file = args.o + "_testSet_AUCs.tsv"
    
    #===============================================================================================================================
    # main
    #===============================================================================================================================
    drugAdrProb = defaultdict(dict)
    test_eval_stats = {}
    verboseOut("Retrieving labels...")
    adrDrugLabel = adr_file_to_dd(args.adrs)
    # now we work on each ADR
    atr_names = get_atr_names(args.i)
    for adr in adrDrugLabel.keys():
        verboseOut("Working on", adr, "...")
        # its an extra logical, but don't bother to load the data if we don't have enough true cases to even predict
        if len(list(adrDrugLabel[adr].keys())) <= args.min_examples_predict:
            verboseOut("Insufficient examples for", adr, ". Not evaluating or predicting")
            continue
        # set up
        prefix = args.o + adr
        # get data
        drugAtr, pred_drugAtr, drugs_to_pred = napa_file_to_dd(adr, args.i, args.drugs, args.predict, atr_names)
        allFts = get_all_fts(drugAtr)
        unbalancedLabels = {drug: True if drug in list(adrDrugLabel[adr].keys()) else False for drug in drugAtr.keys()}
        # We need to build training and testing cases. Currenlty we are also balancing the training and testing data here, but that could be easily changed
        trainingData, testingData, trainingLabels, testingLabels = rew.split_train_test(drugAtr, unbalancedLabels, args.trainingPortion, balanced=True)
        # QA
        if not trainingData or not testingData or not trainingLabels or not testingLabels:
            verboseOut("Unable to build training data. Skipping", adr)
            continue
        if len(list(testingLabels.keys())) <= args.min_examples_testEval:
            verboseOut("Insufficient examples for", adr, ". Not evaluating the testing model.")
        elif not args.no_test_eval:
            train_x, train_y, train_order = res.dd_to_xy(trainingData, trainingLabels)
            test_x, test_y, test_order = res.dd_to_xy(testingData, testingLabels)
            print(train_x)
            print(test_x)
            output = res.run_classifier(res.setup_clf(args.ml_method), train_x, train_y, test_x)
            test_eval_stats[adr] = res.eval_with_test_set(test_y, output, prefix)
            printOut(test_eval_stats)
        # now predict
        if args.predict and len(list(pred_drugAtr.keys())) > 0:
            # if there are no features the assumption is that this drug is not associated with the ADR, and that will be the default
            verboseOut("Beginning to predict...")
            combined_prefix = prefix + "_trainAndTestCombined"
            # combine the testing and training here to build the best model possible        
            combined_data = rew.merge_dicts(trainingData, testingData)
            combined_labels = rew.merge_dicts(trainingLabels, testingLabels)
            if len(list(combined_labels.keys())) < args.min_examples_predict * 2: # X2 is because this equal parts true/false
                continue
            combined_x, combined_y, combined_order = res.dd_to_xy(combined_data, combined_labels)
            pred_x, pred_order = res.dd_to_xy(pred_drugAtr)
            # retrieve the features used to build the model for these prediction drugs
            predictions = res.fit_eval_pred_classifier(res.setup_clf(args.ml_method), combined_x, combined_y, pred_x, combined_prefix) 
            printOut(predictions)
            sys.exit()
#
            # now use that model to predict on all prediction drugs (including those used to build the model, so they will have real and predicted values)
            weka_predict_out = rew.runWeka(args.ml_method, combined_arff, predict_arff, model_name=model_name)
            predictions, _, _ = rew.analyze_report_weka(weka_predict_out, False, False, list(pred_data.keys()), args.write_predictions, predict_prefix)
            drugAdrProb = add_predictions(drugAdrProb, predictions, adr)
        if not args.keep_all:
            clean_up(to_rm)
    # now go through each drug and report it's ADRs. If an ADR isn't in the drug dict, then it's presumed to have a predicted prob of 0
    verboseOut("Reporting predictions for every drug...")
    report_final_preds(drugAdrProb, final_preds_file)
    if not args.no_test_eval:
        report_aucs(roc_auc, pr_auc, auc_file)
    verboseOut("Finished successfully!")
