#!/usr/bin/python
from __future__ import print_function
import os, django, sys
from collections import defaultdict
import argparse
sys.path.insert(1,"../")
sys.path.insert(1,"../../")
sys.path.insert(1,"../../ML")
import run_eval_weka as rew
import tox_funcs as tf
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from path_helper import PathHelper
from algorithms.exit_codes import ExitCoder

# created 10.Feb.2016 - Aaron C Daugherty - twoXAR

# TO DO:
# be able to pass the features to use or not use as argument

# This is to predict whether or not a drug in the PTC set is toxic


# A few subroutines to print out as I want

def message(*objs):
    print(*objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def verboseOut(*objs):
    if verbose:
        print(*objs, file=sys.stderr)

def useage(*objs):
    print("USEAGE: ", *objs, file=sys.stderr)
    
def printOut(*objs):
    print(*objs, file=sys.stdout)

def getAllFts(file):
    initialData = tf.readInAttrFile(file)
    return tf.suplementWithChemblData(initialData, chemblFileName)

def getDrugsToPredict(file):
    drugsAndIdTypes = {}
    with open(file, 'r') as f:
        for line in f:
            fields = line.split("\t")
            drugsAndIdTypes[fields[0]] = fields[1]
    return drugsAndIdTypes

# for this case, we already have chembl, IDs, but in the future we may need to convert from some other ID type
def convertToChemblIds(drugAndIdType):
    return list(drugAndIdType.keys())

if __name__=='__main__':
    #=================================================
    # Read in the arguments/define options
    #=================================================
    exitCoder = ExitCoder()
    
    arguments = argparse.ArgumentParser(description="Predict PTC toxicity labels")
    
    arguments.add_argument("--training", help="PTC training attribute file")
    
    arguments.add_argument("--testing", help="PTC testing attribute file.")

    arguments.add_argument("--bits", help="Filtered molecular substructures (bits) file.")
    
    arguments.add_argument("--predict", default=None, help="Use the model learned from the PTC data to predict toc for all drugs in this attribute file")

    arguments.add_argument("--ml_method", default='naiveBayes'
                     ,help="Machine learning method. Options: RF (random forest), RF_tune (tune the number of features to use), decorate (running RF), naiveBayes [DEFAULT]")
    
    arguments.add_argument("--verbose", action="store_true", help="Print out status reports")
    
    args = arguments.parse_args()
    
    
    # return usage information if no argvs given
    
    if len(sys.argv) < 2:
        arguments.print_help()
        sys.exit(exitCoder.encode('usageError'))
    
    
    ##### INPUTS AND OUTPUTS AND SETTINGS #####
    
    chemblFileName = 'create.chembl.full.tsv'
    trainingArff = os.path.splitext(args.training)[0] + ".arff"
    testingArff = os.path.splitext(args.testing)[0] + ".arff"
    outFilePrefix = 'ptc_tox_predictions'
    # these will only get used if args.predict
    real_chembl_labels_file = "rt.chembl.tsv"
    combined_prefix = outFilePrefix + "_trainAndTestCombined"
    combined_arff = combined_prefix + ".arff"
    predict_prefix = combined_prefix + "_toPredict"
    predict_arff = predict_prefix + ".arff"
    model_name = combined_prefix + ".model"
    #
    mlMethod = args.ml_method
    if args.verbose:
        verbose = args.verbose
    else:
        verbose =  False
    #
    #  In the future it would be great if we passed these options.
    # That would be helpful for the main classifier as we could cutomize a run more easily
    #
    featuresToNotUse = ['sub_smiles', 'cas', 'PTC_tox', 'canonical'
                        , 'smiles_code', 'chembl_id', 'max_phase', 'blackbox_warning']
    ftsToUse = ['Kow__solubility', 'full_mwt', 'rule_of_3_pass', 'ALogP__hydrophobicity'
                , 'most_basic_pKa', 'num_lipinski_rule5_violations', 'wtd_qed__drug_likliness'
                , 'aromatic_rings', 'most_acidic_pKa', 'polar_surface_area', 'heavy_atoms'
                , 'hydrogen_bond_donors', 'num_rule_of_5_violations', 'hydrogen_bond_acceptors'
                , 'num_structural_alerts']
    #===============================================================================================================================
    # main
    #===============================================================================================================================
    verboseOut("Loading features...")    
    trainingData = getAllFts(args.training)
    testingData = getAllFts(args.testing)
    
    verboseOut("Loading molecule bits...")    
    allBitNames, allDrugBits = rew.readInBits(args.bits)
    
    fullTrainingData = rew.addFilteredFeatures(trainingData, allDrugBits, allBitNames)
    fullTestingData = rew.addFilteredFeatures(testingData, allDrugBits, allBitNames)
    
    verboseOut("Retrieving labels...")    
    # this ends up tossing any drugs that don't have a tox label. There's nothing we can do with them, so I don't see a problem with that
    unbalancedTrainingLabels = {drug: trainingData[drug]['PTC_tox'] for drug in trainingData.keys() if 'PTC_tox' in list(trainingData[drug].keys())}
    # This is balanced enough and there are few enough samples that I don't want to throw out any samples
    #trainingLabels, unusedTrainingLables = rew.balance_binary_labels(unbalancedTrainingLabels, verbose)
    trainingLabels = unbalancedTrainingLabels
    testingLabels = {drug: testingData[drug]['PTC_tox'] for drug in testingData.keys() if 'PTC_tox' in list(testingData[drug].keys())}

    verboseOut("Building arffs...")
    
    rew.generateArffs([item for sublist in [ftsToUse, allBitNames] for item in sublist], fullTrainingData, trainingLabels, 'ptc_tox', trainingArff, arffName='ptc_tox_training')
    rew.generateArffs([item for sublist in [ftsToUse, allBitNames] for item in sublist], fullTestingData, testingLabels, 'ptc_tox', testingArff, arffName='ptc_tox_testing')
    
    verboseOut("Running weka...")
    
#    This run is to see how we do on the testing training set
    wekaOut = rew.runWeka(args.ml_method, trainingArff, testingArff)
    
    verboseOut("Analyzing weka results...")
    
    trainTestPrefix = outFilePrefix + "_trainTestEval"
    rew.analyze_report_weka(wekaOut, False, True, list(testingLabels.keys()), False, trainTestPrefix)
    
    if args.predict:
        verboseOut("Preparing for predicting PTC toxicity probability...")
        # read in the drugs to predict
        drugsAndIdTypes = getDrugsToPredict(args.predict)
        # retrieve the features used to build the model for these prediction drugs
        pred_data = tf.get_pred_fts(drugsAndIdTypes, ftsToUse, allBitNames, chemblFileName)
        pred_labels = {k: '?' for k in pred_data.keys()}
        rew.generateArffs([item for sublist in [ftsToUse, allBitNames] for item in sublist], pred_data, pred_labels, 'ptc_tox', predict_arff, label_type = ['True','False'], arffName='ptc_tox_predictions')
        
        # combine the testing and training here to build the best model possible        
        combined_data = rew.merge_dicts(fullTrainingData, fullTestingData)
        combined_labels = rew.merge_dicts(trainingLabels, testingLabels)
        rew.generateArffs([item for sublist in [ftsToUse, allBitNames] for item in sublist], combined_data, combined_labels, 'ptc_tox', combined_arff, arffName='ptc_tox_combined')
        # I want to report the labels of all of the chembl drugs while we have everything combined
        with open(real_chembl_labels_file, 'w') as f:
            for d in combined_labels.keys():
                if 'chembl_id' in list(combined_data[d].keys()):
                    f.write("\t".join([combined_data[d]['chembl_id'], 'rt_ptc', str(combined_labels[d])]) + "\n")
#        # then pull in all provided (or ws) drugs, and determine their values for the given features
#        # the biggest challenge there will be the bits
#        rew.generateArffs([item for sublist in [ftsToUse, allBitNames] for item in sublist], predData, predLabels, 'ptc_tox', predict_arff, arffName='ptc_tox_combined_predict')
        # first build and save a model using the combined data.
        # we definitely want the testing stats
        
        verboseOut("Building weka model...")
    
        weka_model_out = rew.runWeka(args.ml_method, combined_arff, None, model_name=model_name, testing = True, build = True)
        
        verboseOut("Evaluating model...")
        
        rew.analyze_report_weka(weka_model_out, True, False, list(combined_data.keys()), False, combined_prefix)
        
        verboseOut("Making predictions...")
        
        # now use that model to predict on all prediction drugs (including those used to build the model, so they will have real and predicted values)
        weka_predict_out = rew.runWeka(args.ml_method, combined_arff, predict_arff, model_name=model_name)
        
        rew.analyze_report_weka(weka_predict_out, False, False, list(pred_data.keys()), True, predict_prefix)


