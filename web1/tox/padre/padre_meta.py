#!/usr/bin/python
from __future__ import print_function
from builtins import range
import os, sys, time
from collections import defaultdict
try:
    from path_helper import PathHelper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../..")
    from path_helper import PathHelper

try:
    import run_eval_weka as rew
except ImpoprtError:
    sys.path.insert(1,os.path.join(PathHelper.website_root, "ML"))
    import run_eval_weka as rew
from algorithms.exit_codes import ExitCoder
import napa_build as nb

# created 20.Apr.2016 - Aaron C Daugherty - twoXAR

# TO DO:
# report in a drug-centered way: all ADR probabilities for all drugs
#   Can then take this data into R and create average ROCs (for each drug) and overall all drugs all ADRs. Can also split it out by ADR
#      Might be able to do this in python

# This takes napa_predict output and generates ADR-specific feature matrices before running a classifier, via weka 

save_output = True # this results in writing drug files that have been re-keyed to wsas to be saved

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

def get_basedir(dir):
    return os.path.basename(os.path.normpath(dir))

def combine_in_files(i, s):
    if i and s:
        fs = i + s
        d = i
    elif i:
        fs = i
        d = i
    elif s:
        fs = s
        d = []
    return fs, d

def napa_file_to_dd(adr, dir_list, pred_file, ft_list, full_drug_set, sep="\t"):
# these files should be in the following format:
# where all the fields after adr are features
# drug    adr     path_count      median_max_wt   median_median_wt        median_path_counts      median_OR       median_q
# DB00953 C0017152        5       0.051948051948  0.0181818181818 0.6     0.899420788979  0.931802366757
#
# A ft_list file will look like this:
# drug_id ADR     feature_list
# DB00152 C0019080        O60779_0,Q9H3S4_0
#
# in order to work with existing code we need some deep data structures. 
# keyed by ADR, the value is a dictionary keyed by drugID,
# and the value there are dictionaries keyed by the attribute name,
# and the final vlaue is for the matrix
    adr_data = defaultdict(dict)
    # it's easier to create empty versions than have multiple conditionals below
    pred_data = defaultdict(dict)
    drugs_to_pred = set()
    if pred_file == 'all': # this means we want to predict all drugs in the WS
        pred_all = True
        drugs_to_pred = full_drug_set
    else:
        # read in the drugs to predict
        drugs_to_pred = get_drug_set(pred_file)
#    for file in file_list:
    for dir in dir_list:
        filename = os.path.join(dir, adr + '.tsv')
        basename = get_basedir(dir)
        try:
            with open(filename, 'r') as f:
                atr_names = None
                for l in f:
                    fields = l.rstrip().split(sep)
                    if atr_names is None:
                        atr_names = [basename + "_" + i for i in fields[2:]]
                        continue
                    drug = fields[0]
                    if dir in ft_list:
                        fts = fields[2].split(",")
                        new_val = dict(zip(fts, [rew.convertTypes('True')] * len(fts)))
                    else:
                        new_val = dict(zip(atr_names, [rew.convertTypes(ft) for ft in fields[2:]]))
            # because we are predicting all drugs in the WS, we have to double list the predict and adr_data
            # otherwise adr_data would be empty
                    if pred_all:
                        pred_data = update_dd(pred_data, drug, new_val)
                        adr_data = update_dd(adr_data, drug, new_val)
                    else:
                        if drug in drugs_to_pred:
                            pred_data = update_dd(pred_data, drug, new_val)
                        else:
                            adr_data = update_dd(adr_data, drug, new_val)
        except IOError:
            continue
    # see notes above about pred+all
    if pred_all:
        adr_data = fill_in_missing_drugs(list(full_drug_set - set(adr_data.keys())), adr_data)
    else:
        adr_data = fill_in_missing_drugs(list(full_drug_set - set(adr_data.keys()) - drugs_to_pred), adr_data)
    if drugs_to_pred and len(list(pred_data.keys())) > 0:
        pred_data = fill_in_missing_drugs(list(drugs_to_pred - set(pred_data.keys())), pred_data)
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

def get_ws_drug_set(wsid):
    from browse.models import WsAnnotation
    import django
    if not "DJANGO_SETTINGS_MODULE" in os.environ:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
        django.setup()
    ws_ds = set([str(rec.agent_id) for rec in WsAnnotation.objects.filter(ws_id=wsid)])
    if save_output:
        with open(str(wsid) + "_wsas.txt", 'w') as f:
            f.write("\n".join(ws_ds) + "\n")
    return ws_ds

def get_drug_set(drug_file):
    with open(drug_file, 'r') as f:
        full_drug_set = set(f.read().splitlines())
    return full_drug_set

def adr_file_to_dd(adr_file, wsid, rekeyed = False):
    dd = defaultdict(dict)
    with open(adr_file, 'r') as f:
        data_types = f.readline().rstrip().split("\t")
        for line in f:
            fields = line.rstrip().split("\t")
            if rekeyed:
                drug = fields[0]
            else:
                drug = nb.convert_to_wsa(fields[0], wsid, current_type = data_types[0])
                if not drug:
                    continue
# if we wanted to do more than binary, we could load in the weight(field[2]) here
# though we'd also have to adjust below when we add in the False 
            dd[fields[1].upper()][drug] = True
    if save_output:
        with open(str(wsid) + "wsas_" + os.path.basename(adr_file), 'w') as f:
            f.write("\t".join(['wsa', 'adr']) + "\n")
            for adr in dd.keys():
                for drug in dd[adr].keys():
                    f.write("\t".join([drug, adr]) + "\n")
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

def report_eval_stats(eval_stats, file):
    with open(file, 'w') as f:
        stats = list(set([k2 for k in eval_stats.keys() for k2 in eval_stats[k].keys()]))
        f.write("\t".join(['ADR'] + stats) + "\n")
        for adr in eval_stats.keys():
# this is to printout all of the iterations, presumably you'll want to combine these elsewher...R?
            out = [adr]
            for s in stats:
                try:
                    out.append(",".join([str(i) for i in eval_stats[adr][s]]))
                except KeyError:
                    out.append("NA")
            f.write("\t".join(out) + "\n")

def clean_up(l):
    for filename in l:
        try:
            os.remove(filename)
        except OSError:
            pass

def test_and_pred(params):
    ts = time.time()
    adr, drugLabels, args, in_dirs, is_indiv_ft, whole_drug_set, predtrain_stats_prefix = params
    verboselog=[]
    verboselog.append("Working on " + adr + "...")
    eval_stats = {}
    eval_stats['n'] = [str(len(list(drugLabels.keys())))] # the other stats may have multiple entries and will be in a list
    # its an extra logical, but don't bother to load the data if we don't have enough true cases to even predict
    if len(list(drugLabels.keys())) <= args.min_examples_predict:
        verboselog.append("Insufficient examples for " + adr + ". Not evaluating or predicting.")
        verboseOut("\n".join(verboselog))
        return (adr, None, None, None, None)
    ts = time.time()
    drugAtr, pred_drugAtr, drugs_to_pred = napa_file_to_dd(adr, in_dirs, args.predict, is_indiv_ft, whole_drug_set)
    verboselog.append(adr + " napa_file_to_dd took:" + str(time.time() - ts))
    allFts = get_all_fts(drugAtr)
    eval_stats['ft_cnt'] = [str(len(allFts))]
    unbalancedLabels = {drug: True if drug in list(drugLabels.keys()) else False for drug in drugAtr.keys()}
    if args.ml_method == 'RF_weight':
        cost_list = [0, len([v for v in unbalancedLabels.values() if v is False]), len([v for v in unbalancedLabels.values() if v is True]),0]
    else:
        cost_list = None
    all_weka_predict_out = []
    predtrain_stats = {}
    attrs_used = {}
    for outer_iter in range(args.outer_iters):
        # set up
        to_rm=[]
        prefix = args.o + adr + "_" + str(outer_iter)
        trainingArff = prefix + "_training.arff"
        testingArff = prefix + "_testing.arff"
        # get data
        # We need to build training and testing cases. Currenlty we are also balancing the training and testing data here, but that could be easily changed
        trainingData, testingData, trainingLabels, testingLabels, leftoverLabels = rew.split_train_test(drugAtr, unbalancedLabels, args.trainingPortion, balanced=args.balance)
        # QA
        if not trainingData or not testingData or not trainingLabels or not testingLabels:
            verboselog.append("Unable to build training data. Skipping" + adr)
            break
        if len(list(testingLabels.keys())) <= args.min_examples_testEval:
            verboselog.append("Insufficient examples for" + adr + ". Not evaluating the testing model.")
        elif not args.no_test_eval:
            all_weka_out = []
            inner_iter = 0
            while True:
                inner_iter += 1
                # set up for and run WEKA
                ft_data = rew.generateArffs(allFts, trainingData, trainingLabels, adr, trainingArff, arffName = adr + '_training', fillin=True)
                rew.generateArffs(allFts, testingData, testingLabels, adr, testingArff, arffName = adr + '_testing', fillin=True, provided_attr_data = ft_data)
                verboselog.append("Running weka in inner interation" + str(inner_iter) + "...")
                # This run is to see how we do on the testing training set
                # need to add this to a list
                all_weka_out.append(rew.runWeka(args.ml_method, trainingArff, testingArff, cost_list = cost_list))
                trainTestPrefix = prefix + "_trainTestEval"
                if not args.balance or inner_iter >= args.inner_iters:
                    break # if we're not using balanced sets we don't need to do multiple iterations (nothing will change)
                else:
                    # re run rew.split_train_test with just the training data and the leftover labels combined,
                    # but set the training portion to 1. 
                    # The result is the same samples of minority classes, but a different mix of the majority.
                    retrainLabels = rew.merge_dicts(trainingLabels, leftoverLabels)
                    trainingData, _, trainingLabels, _, leftoverLabels = rew.split_train_test(drugAtr, retrainLabels,
                                                                                               1, balanced=args.balance
                                                                                             )
            # now we need to be able to combine multiple predictions into one, and just evaluate that,
            # while also giving this function the format it expects
            verboselog.append("Analyzing weka results...")
            weka_out = rew.condense_binary_weka_preds(all_weka_out)
            if weka_out is not None:
                _, adr_eval_stats = rew.analyze_report_weka(weka_out, False, True, list(testingLabels.keys()), False, trainTestPrefix)
                eval_stats = rew.update_dol(adr_eval_stats, eval_stats)
            to_rm += [trainingArff, testingArff]
        # now predict
        if args.predict and len(list(pred_drugAtr.keys())) > 0:
            # if there are no features the assumption is that this drug is not associated with the ADR,
            # and that will be the default
            verboselog.append("Beginning to predict...")
            combined_prefix = prefix + "_trainAndTestCombined"
            combined_arff = combined_prefix + ".arff"
            predict_prefix = combined_prefix + "_toPredict"
            predict_arff = predict_prefix + ".arff"
            model_name = combined_prefix + ".model"
            # combine the testing and training here to build the best model possible      
            combined_data = rew.merge_dicts(trainingData, testingData)
            combined_labels = rew.merge_dicts(trainingLabels, testingLabels)
            if len(list(combined_labels.keys())) < args.min_examples_predict * 2: # X2 is because this equal parts true/false
                continue
            ft_data = rew.generateArffs(allFts, combined_data, combined_labels, adr,
                                         combined_arff, arffName = adr + '_combined', fillin=True
                                        )
            # retrieve the features used to build the model for these prediction drugs
            pred_data = pred_drugAtr
            pred_labels = {k: '?' for k in pred_data.keys()}
            ts = time.time()
            rew.generateArffs(allFts, pred_data, pred_labels, adr, predict_arff, label_type = ['True','False'],
                               arffName = adr + '_predictions', fillin=True, provided_attr_data = ft_data
                              )
            verboselog.append(adr + " rew.generateArffs(allFts 2, took:" + str(time.time() - ts))
            verboselog.append("Building weka model...")
            weka_model_out = rew.runWeka(args.ml_method, combined_arff, None, model_name=model_name, 
                                          testing = True, build = True, cost_list = cost_list
                                         )
            verboselog.append("Evaluating model...")
            rew.analyze_report_weka(weka_model_out, True, False, list(combined_data.keys()), False, combined_prefix,
                                     write_stats=args.pred_train_stats
                                    )
            if args.pred_train_stats:
                predtrain_stats, attrs_used = rew.process_pred_stats(combined_prefix, predtrain_stats, attrs_used)
                to_rm.append(combined_prefix + '_testingStats.txt')
            else:
                predtrain_stats = None
                attrs_used = None
            verboselog.append("Making predictions...")
            # now use that model to predict on all prediction drugs (including those used to build the model,
            # so they will have real and predicted values)
            ts = time.time()
            all_weka_predict_out.append(rew.runWeka(args.ml_method, combined_arff, predict_arff,
                                                     model_name=model_name, cost_list = cost_list
                                                    )
                                        )
            verboselog.append(adr + " Making predictions took:" + str(time.time() - ts))
            to_rm += [combined_arff, predict_arff, model_name]
        if not args.keep_all:
            clean_up(to_rm)
    predictions = None
    if args.predict and len(all_weka_predict_out) > 0:
        verboselog.append(adr + " processing preds")
        weka_predict_out = rew.condense_binary_weka_preds(all_weka_predict_out)
        if weka_predict_out is not None:
            predictions, _ = rew.analyze_report_weka(weka_predict_out, False, False, list(pred_data.keys()),
                                                      args.write_predictions, predict_prefix)
    if predtrain_stats is not None:
        verboselog.append(adr + " plotting CV stats")
        rew.plot_cv_stats_boxplot(predtrain_stats, "_".join(predtrain_stats_prefix, adr, 'boxplot.png'))
    verboseOut("\n".join(verboselog))
#    verboselog.append(adr + " took " + str(time.time()-ts) + "seconds")
    return (adr, predictions, predtrain_stats, eval_stats, attrs_used)

if __name__=='__main__':
    import argparse
    #=================================================
    # Read in the arguments/define options
    #=================================================
    exitCoder = ExitCoder()
    
    arguments = argparse.ArgumentParser(description="Build models for predicting individual ADRs")
    
    arguments.add_argument("--summary_fts", nargs='+'
                            ,help="A space separated list of summary feature files, e.g. multiple output from napa_predict")
    
    arguments.add_argument("--indiv_fts", nargs='+'
                            ,help="A space separated list of files that contain CSV list of features")
    
    arguments.add_argument("-o", default="./", help="Prefix to put on all output files. DEAFULT: %(default)s")
    
    arguments.add_argument("--adrs", help="ADR file containing all drug-ADR pairings")
    
    arguments.add_argument("--predict", default=None
                            ,help="Use the model learned to predict ADRs for all drugs in this file")
    
    arguments.add_argument("--wsid", type=int, default=49, help="Workspace ID. default %(default)s")

    arguments.add_argument("--ws_ds", help="File containing the whole WS drug set. OPTIONAL")
    
    arguments.add_argument("--trainingPortion", type = float, default=0.7
                            ,help="Portion of drugs to use for training set [0-1, DEAFULT: %(default)s]")
    
    arguments.add_argument("--outer_iters", type = int, default=1
                           ,help="How many times should the training/testing set be generated." +
                            " More than 1 will lead to prediction averaging across models DEAFULT: %(default)s")
    
    arguments.add_argument("--inner_iters", type = int, default=1
                            ,help="If balancing, how many times should the training majority class be be sampled." +
                             " The testing set will be unchanged. More than 1 will lead to prediction averaging " +
                             "across models DEAFULT: %(default)s")
    
    arguments.add_argument("--no_test_eval", action="store_true", help="Do not run testing evaluation for each ADR")
    
    arguments.add_argument("--ncores", type = int, default=20
                           ,help="How many cores can we use? DEAFULT: %(default)s]")

    arguments.add_argument("--min_examples_predict", type = int, default=10
                           ,help="Minimum number of drugs with ADR to try to predict. DEAFULT: %(default)s]")
    
    arguments.add_argument("--min_examples_testEval", type = int, default=10
                            ,help="Minimum number of drugs that need to be in the test set to evaluate. " +
                             "DEAFULT: %(default)s]")
    
    arguments.add_argument("--ml_method", default='RF'
                           ,help="Machine learning method. Options: RF (random forest)," +
                             " RF_tune (tune the number of features to use)," +
                             " attrSel_RF (select attributes before running RF)," +
                             " decorate (running RF), naiveBayes. DEAFULT: %(default)s")
    
    arguments.add_argument("--balance", action="store_true"
                            ,help="Training and testing sets have equal numbers of positive and negative instances")
    
    arguments.add_argument("--keep_all", action="store_true"
                            ,help="Save all ARFF files as well as individual ADR model files.")
    
    arguments.add_argument("--pred_train_stats", action="store_true"
                            ,help="Report prediction training stats, when possible")
    
    arguments.add_argument("--write_predictions", action="store_true"
                            ,help="Report individual ADR predictions to separate files")
    
    arguments.add_argument("--verbose", action="store_true", help="Print out status reports")
    
    args = arguments.parse_args()
    
    # return usage information if no argvs given
    if (not args.indiv_fts and not args.summary_fts) or not args.adrs or not args.wsid:
        arguments.print_help()
        sys.exit(exitCoder.encode('usageError'))

    ts = time.time()

    final_preds_file = args.o + "_final_predictions.tsv"
    eval_stats_file = args.o + "_testSet_eval_stats.tsv"
    predtrain_stats_prefix = args.o + "_cv_stats"
    predtrain_stats_file = predtrain_stats_prefix + ".tsv"
    attrs_selected_file = args.o + "_attrs_selected.tsv"
    
    #===============================================================================================================================
    # main
    #===============================================================================================================================
    drugAdrProb = defaultdict(dict)
    eval_stats = {}
    predtrain_stats = {}
    attrs_selected = {}
    verboseOut("Retrieving labels...")
    rekeyed = False
    if os.path.basename(args.adrs).startswith(str(args.wsid) + "wsas_"):
        rekeyed = True
    adrDrugLabel = adr_file_to_dd(args.adrs, args.wsid, rekeyed = rekeyed)
    indirs, is_indiv_list = combine_in_files(args.indiv_fts, args.summary_fts)
    n_iter = len(list(adrDrugLabel.keys()))
    if args.ws_ds:
        ws_ds = set([line.strip() for line in open(args.ws_ds, 'r')])
    else:
        ws_ds = get_ws_drug_set(args.wsid)
    params = zip(list(adrDrugLabel.keys()), list(adrDrugLabel.values())
                  ,[args] * n_iter
                  ,[indirs] * n_iter
                  ,[is_indiv_list] * n_iter
                  ,[ws_ds] * n_iter
                  ,[predtrain_stats_prefix] * n_iter,
                 )
    verboseOut("Set up took " + str(time.time() - ts) + " seconds")
# multiprocessing doesn't report the traceback if something goes wrong, so we're better off to loop as below.
#    results = []
#    for tup in params:
#        res = test_and_pred(tup)
#        results.append(res)
    import multiprocessing
    pool = multiprocessing.Pool(args.ncores)
    results = pool.map(test_and_pred, params)
    # results should be a list of tuples, in which each will be : adr, predictions, adr_predtrain_stats, adr_eval_stats, attrs_selected
    for tup in results:
        if tup[4]:
            attrs_selected[tup[0]] = tup[4]
        if tup[3]:
            eval_stats[tup[0]] = tup[3]
        if tup[2]:
            predtrain_stats[tup[0]] = tup[2]
        if tup[1]:
            drugAdrProb = add_predictions(drugAdrProb, tup[1], tup[0])
    # now go through each drug and report it's ADRs. If an ADR isn't in the drug dict,
    # then it's presumed to have a predicted prob of 0
    if args.predict:
        verboseOut("Reporting predictions for every drug...")
        report_final_preds(drugAdrProb, final_preds_file)
    if not args.no_test_eval:
        report_eval_stats(eval_stats, eval_stats_file)
    if args.pred_train_stats:
        report_eval_stats(predtrain_stats, predtrain_stats_file)
    if len(list(attrs_selected.keys())) > 0:
        rew.report_multiple_attr_stats(attrs_selected, attrs_selected_file)
    verboseOut("Finished successfully!")
