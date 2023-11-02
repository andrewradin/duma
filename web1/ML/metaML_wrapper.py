#!/usr/bin/env python3

from __future__ import print_function
from builtins import range
import sys
from path_helper import PathHelper,make_directory

import os

import run_eval_weka as rew

debugging = False

class metaML:
    def __init__(self,**kwargs):
        self.cores = kwargs.get('cores',None)
        self.ws = kwargs.get('ws',None)
        self.outdir = kwargs.get('outdir',None)
        self.tmp_pubdir = kwargs.get('tmp_pubdir',None)
        self.out_iters = kwargs.get('out_iters',None)
        self.in_iters = kwargs.get('in_iters',None)
        self.training_portion = kwargs.get('training_portion',None)
        self.method = kwargs.get('method',None)
        if self.method == 'decorate':
            self.min_samples = 56 # 50 is for Decorate, 6 is for 10-fold CV
        else:
            self.min_samples = 1 # this is basically just a filler
        self.infile_arff = kwargs.get('arff',None)
        self.infile_druglist = kwargs.get('druglist',None)
        self.cv_stats = {}
        self.attrs_used = {}
        # output files
        # published output files
    def run(self):
        self.read_arff_in()
        self.setup_dirs()
        if self.method == 'RF_weight':
            self.cost_list = [0,
                              len([v for v in self.drug_labels.values()
                                   if v == 'False']),
                              len([v for v in self.drug_labels.values()
                                   if v == 'True']),
                              0]
        else:
            self.cost_list = None
        for outer_iter in range(self.out_iters):
            self.balance_and_sample(outer_iter)
            self.setup_output_names(outer_iter)
            params = zip([self.data_to_use] * self.in_iters,
                          [self.labels_to_use] * self.in_iters,
                          [self.atr_header] * self.in_iters,
                          self.arff_names,
                          self.model_names,
                          [self.method] * self.in_iters,
                          [self.cost_list] * self.in_iters,
                          self.stats_file_names,
                          self.info_gain_file_names,
                          self.pred_file_names,
                          [self.infile_arff] * self.in_iters,
                          [self.min_samples] * self.in_iters,
                        )
            if debugging:
                map(inside_loop, params)
            else:
                import multiprocessing
                pool = multiprocessing.Pool(self.cores)
                pool.map(inside_loop_wrapper, params)
            for i in range(self.in_iters):
                self.cv_stats, self.attrs_used = rew.process_pred_stats(self.stats_file_names[i], self.cv_stats, self.attrs_used)
        #### Now run the various ouside scripts. Eventually I want to bring those in here, but one thing at a time
        self.run_outside_scripts()
        self.process_info_gain()
        rew.plot_cv_stats_boxplot(self.cv_stats, os.path.join(self.finalout_dir, 'cv_stats_boxplot.plotly'), plotly = True)
        try:
            rew.report_attr_stats(self.attrs_used, os.path.join(self.finalout_dir, 'attributes_used.txt'))
        except TypeError:
            pass

    def run_outside_scripts(self):
        import subprocess
        self.ml_dir = PathHelper.MLscripts
        all_stats_files = []
        subprocess.check_call([
                'Rscript',
                os.path.join(self.ml_dir, 'processPredictions.R'),
                self.preds_dir,
                self.infile_druglist,
                self.assign_dir,
                self.ml_dir,
                str(self.out_iters),
                ])
        from dtk.files import get_dir_file_names
        for f in get_dir_file_names(self.preds_dir):
            if f.endswith(".png") or f.endswith("s.csv"):
                subprocess.check_call([
                        'mv',
                        os.path.join(self.preds_dir, f),
                        os.path.join(self.finalout_dir, f),
                        ])
            elif f.endswith('testingStats.txt'):
                all_stats_files.append(os.path.join(self.preds_dir, f))
    def process_info_gain(self):
        from operator import itemgetter
        from collections import defaultdict
        from statistics import median
        sub_dirs = [os.path.join(self.info_gain_dir, name) for name in os.listdir(self.info_gain_dir)
                     if os.path.isdir(os.path.join(self.info_gain_dir, name))
                   ]
        total_file_num = 0
        all_info = defaultdict(list)
        for iter in sub_dirs:
            gen = (os.path.join(iter, file) for file in os.listdir(iter) if file.endswith(".txt"))
            for file in gen:
                total_file_num += 1
                with open(file, 'r') as f:
                    for line in f:
                        info_gain, name = line.rstrip("\n").split("\t")[0:2]
                        all_info[name].append(float(info_gain))
        stats = []
        for k in all_info:
            # fill in the missing values
            all_info[k] = all_info[k] + [0.0] * (total_file_num -  len(all_info[k]))
            stats.append((k, sum(all_info[k]) / len(all_info[k]), median(all_info[k]), max(all_info[k])))
        stats.sort(key = itemgetter(1,2, 3))
        self.plot_info_gain(all_info, stats)
    def plot_info_gain(self, all_info, stats):
        if stats:
            from dtk.plot import boxplot_stack
            pp = boxplot_stack(
                    [(all_info[k],k) for k,_,_,_ in stats],
                    title='Information Gain',
                    )
            pp.save(os.path.join(self.finalout_dir, "infoGainBoxplot.plotly"))
    # the goal of this is to read in a pre-made ARFF file and corresponding drug list (the same order as the arff data lines)
    # and get it into a format that split_train_test can use
    def balance_and_sample(self, outer_iter):
        training_fv, _, train_labels, test_labels, leftover_labels = rew.split_train_test(self.fv, self.drug_labels, self.training_portion, min_samples = self.min_samples)
        for x in [training_fv, train_labels, test_labels]:
            assert x
        maj_label = set(leftover_labels.values())
        assert len(maj_label) == 1
        maj_label = list(maj_label)[0]
        minority_inds = [k for k in test_labels.keys() if test_labels[k] != maj_label]
        min_label = set([v for v in train_labels.values() if v != maj_label])
        assert len(min_label) == 1
        self.current_assign_dir = os.path.join(self.assign_dir, str(outer_iter +1))
        with open(os.path.join(self.current_assign_dir, 'rowNumberOfTestTreatmentsInDF.txt'), 'w') as f:
        # the plus one is b/c this info is going to be used by R which is not 0-indexed, but 1-indexed
            f.write("\n".join([str(int(i) + 1) for i in minority_inds]) + "\n")
        # now combine the leftover data b/c this acctually gets re-run for each iteration
        self.labels_to_use = rew.merge_dicts(train_labels, leftover_labels)
        self.data_to_use = {k:self.fv[k] for k in self.labels_to_use.keys()}
    def read_arff_in(self):
        self.atr_header = []
        self.drugs_fv = []
        self.drugs_labels = []
        with open(self.infile_arff, 'r') as f:
            for l in (l.rstrip("\n") for l in f):
                if l == "" or l == "@DATA" or l.startswith("%"):
                    continue
                elif l.startswith("@"):
                    self.atr_header.append(l)
                else:
                    self.drugs_fv.append(l)
                    self.drugs_labels.append(l.split(",")[-1])
        # convert the data into dictionaries for use with the funciton, but while maintaining the ability to know the order
        self.drug_labels = dict(zip([i for i in range(len(self.drugs_labels))], [x for x in self.drugs_labels]))
        self.fv = dict(zip([i for i in range(len(self.drugs_fv))], [x for x in self.drugs_fv]))
    def setup_dirs(self):
        make_directory(self.outdir)
        self.assign_dir = os.path.join(self.outdir, 'assignments')
        make_directory(self.assign_dir)
        self.train_dir = os.path.join(self.outdir, 'toTrainOn')
        make_directory(self.train_dir)
        self.info_gain_dir = os.path.join(self.outdir, 'infoGain')
        make_directory(self.info_gain_dir)
        self.models_dir = os.path.join(self.outdir, 'models')
        make_directory(self.models_dir)
        self.preds_dir = os.path.join(self.outdir, 'predictions')
        make_directory(self.preds_dir)
        self.finalout_dir = os.path.join(self.outdir, 'PlotsAndFinalPredictions')
        make_directory(self.finalout_dir)
        for x in [self.assign_dir, self.train_dir, self.info_gain_dir, self.models_dir, self.preds_dir, self.finalout_dir]:
            for i in range(self.out_iters):
                make_directory(os.path.join(x, str(i+1)))
    def setup_output_names(self, out_iter):
        out_iter = str(out_iter + 1)
        self.arff_names = []
        self.model_names = []
        self.stats_file_names = []
        self.pred_file_names = []
        self.info_gain_file_names = []
        for i in range(self.in_iters):
            j = str(i+1)
            self.arff_names.append(os.path.join(self.train_dir, out_iter, j + '.arff'))
            self.model_names.append(os.path.join(self.models_dir, out_iter, j + '.model'))
            self.stats_file_names.append(os.path.join(self.preds_dir, "_".join([out_iter, j])))
            self.pred_file_names.append(os.path.join(self.preds_dir, out_iter, j + '.tsv'))
            self.info_gain_file_names.append(os.path.join(self.info_gain_dir, out_iter, j + '.txt'))

def inside_loop_wrapper(params):
    try:
        return inside_loop(params)
    except Exception as ex:
        print("GOT EXCEPTION IN SUBPROCESS")
        print('arff_name:',params[3])
        import traceback
        traceback.print_exc()
        import sys
        sys.stdout.flush()
        raise

def inside_loop(params):
    data, labels, arff_head, arff_name, model_name, method, cost_list, stats_file, infogain_file, pred_file, full_arff, min_samples = params
    # The result is the same samples of minority classes, but a different mix of the majority for each run of this function (even with the same input)
    trainingData, _, trainingLabels, _, _ = rew.split_train_test(data, labels, 1.0, min_samples = min_samples)
    # now that we have the training data we want, build the arffs
    # for this one the order doesn't matter (we just want the model)
    write_arff_with_predef_header(arff_head, list(trainingData.values()), arff_name)
    # then use that arff to build a model
    weka_model_out = rew.runWeka(method, arff_name, None, model_name=model_name, 
                                 testing = True, build = True, cost_list = cost_list
                                 )
    rew.analyze_report_weka(weka_model_out, True, False, list(data.keys()), False, stats_file, write_stats=True)
    # then make predictions
    preds = rew.runWeka(method, None, full_arff, model_name=model_name, cost_list = cost_list)
    process_preds(preds, pred_file)
    # Also need to run infoGain
    infogain = rew.run_weka_infogain(arff_name)
    # and need to process resulting files to be as expected
    process_infogain(infogain, infogain_file)
def process_infogain(input, file):
    still_header = True
    with open(file, 'w') as f:
        for l in input.split("\n"):
            if l.startswith('Ranked'):
                still_header = False
                continue
            if not still_header:
                if l == "":
                    break # end of list
                fields = l.strip().split()
                if float(fields[0]) == 0.0:
                    break # end of interesting values
                f.write("\t".join([fields[0], fields[2]]) + "\n")
def process_preds(preds, pred_file):
    lines = rew.get_non_blank_lines(preds)
    with open(pred_file, 'w') as f:
        for line in lines:
            fields = line.rstrip().split()
            if fields[0] == "===" or fields[0].strip().startswith("inst") :
                continue
            if fields[3] == "+":
                fields[3] = fields[4]
            f.write("\t".join([fields[0], fields[1], fields[3].replace('*','').split(",")[0]]) + "\n")

def write_arff_with_predef_header(header, data, filename):
    with open(filename, 'w') as f:
        f.write("\n".join(header + ["@DATA"] + data) + "\n")

if __name__ == "__main__":
    # force unbuffered output
    sys.stdout = sys.stderr
    from dtk.log_setup import setupLogging
    setupLogging()

    import argparse, time
    parser = argparse.ArgumentParser(description='run Weka for a meta efficacy method')
    parser.add_argument("druglist", help="ordered list of drugs")
    parser.add_argument("arff", help="premade ARFF")
    parser.add_argument("outdir", help="directory to put output in")
    parser.add_argument("tmp_pubdir", help="directory for plots")
    parser.add_argument("w", help="workspace ID")
    parser.add_argument("cores", type=int, help="Number of cores alloted")
    parser.add_argument("--training_portion", type = float, default=0.9
                            ,help="Portion of drugs to use for training set [0-1, DEAFULT: %(default)s]")
    parser.add_argument("--outer_iters", type = int, default=4
                           ,help="How many times should the training/testing set be generated." +
                            " More than 1 will lead to prediction averaging across models DEAFULT: %(default)s")
    parser.add_argument("--inner_iters", type = int, default=50
                            ,help="If balancing, how many times should the training majority class be be sampled." +
                             " The testing set will be unchanged. More than 1 will lead to prediction averaging " +
                             "across models DEAFULT: %(default)s")
    parser.add_argument("--ml_method", default='decorate'
                           ,help="Machine learning method. Options: RF (random forest)," +
                             " RF_tune (tune the number of features to use)," +
                             " attrSel_RF (select attributes before running RF)," +
                             " decorate (running RF), naiveBayes. DEAFULT: %(default)s")

    args=parser.parse_args()
    
    ts = time.time()
    run = metaML(
              ws = args.w,
              outdir = args.outdir,
              tmp_pubdir = args.tmp_pubdir,
              cores = args.cores,
              out_iters = args.outer_iters,
              in_iters = args.inner_iters,
              training_portion = args.training_portion,
              method = args.ml_method,
              arff = args.arff,
              druglist = args.druglist,
            )
    run.run()
    print("Took: " + str(time.time() - ts) + ' seconds')



