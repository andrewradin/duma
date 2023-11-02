#!/usr/bin/python
from builtins import range
import os, django, sys, re
from sklearn import *
from collections import defaultdict,Counter
sys.path.insert(1,"../")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from path_helper import PathHelper
import run_eval_weka as rew


def dd_to_xy(dd, labels_d = None):
    import pandas as pd
    x = pd.DataFrame(list(dd.values()), index=list(dd.keys()))
    if labels_d:
        y = order_labels(labels_d, list(x.index))
        return x.values, y, x.index
    return x.values, x.index

def order_labels(labels, row_names):
    import numpy
    return numpy.asarray([labels[i] for i in row_names])

def setup_clf(clf_name, n_estimators = 10, n_jobs = -1):
    if clf_name.upper() == 'RF' or clf_name.lower() == 'randomforest':
        return ensemble.RandomForestClassifier(n_estimators = n_estimators, n_jobs = n_jobs)
#    elif clf.name ==

def run_classifier(clf, x, y, test_x, positive_class = True):
    clf = clf.fit(x, y)
    output = clf.predict_proba(test_x)
    for i in range(len(clf.classes_)):
        if clf.classes_[i] == positive_class:
            return [a[i] for a in output]

def eval_with_test_set(y_actual, y_predScore, plotFilePrefix, score_types=['pr_auc', 'roc_auc']):
    stats = {}
    if 'pr_auc' in score_types:
        stats['pr_auc'] = rew.plotPRCurve(y_actual, y_predScore, plotFilePrefix)
    if 'roc_auc' in score_types:
        stats['roc_auc'] = rew.plotROC(y_actual, y_predScore, plotFilePrefix)
    return stats

def fit_eval_pred_classifier(clf, x, y, pred_x, prefix, score_types = ['roc_auc']):
    cv_stats_dict = my_cv_model(clf, x, y)
    return cv_stats_dict

# lifted and modified from here: http://stackoverflow.com/questions/23339523/sklearn-cross-validation-with-multiple-scores
def my_cv_model(estimator, x, y, n_folds = 10, stratify = True):
    if not stratify:
        cv_arg = cross_validation.KFold(y.size, n_folds)
    else:
        cv_arg = cross_validation.StratifiedKFold(y, n_folds)
    ys = get_true_and_pred_CV(estimator, x, y, n_folds, cv_arg)
    cv_acc = map(lambda tp: metrics.accuracy_score(tp[0], tp[1]), ys)
    cv_pr_weighted = map(lambda tp: metrics.precision_score(tp[0], tp[1], average='weighted'), ys)
    cv_rec_weighted = map(lambda tp: metrics.recall_score(tp[0], tp[1], average='weighted'), ys)
    cv_f1_weighted = map(lambda tp: metrics.f1_score(tp[0], tp[1], average='weighted'), ys)
    cv_rocauc_weighted = map(lambda tp: metrics.roc_auc_score(tp[0], tp[1], average='weighted'), ys)

    return {'CV accuracy': np.mean(cv_acc), 'CV precision_weighted': np.mean(cv_pr_weighted),
            'CV recall_weighted': np.mean(cv_rec_weighted), 'CV F1_weighted': np.mean(cv_f1_weighted),
             'CV ROC-AUC_weighted': np.mean(cv_rocauc_weighted)}

def get_true_and_pred_CV(estimator, X, y, n_folds, cv):
    ys = []
    for train_idx, valid_idx in cv:
        clf = estimator
        cur_pred = run_classifier(clf, X[train_idx], y[train_idx], X[valid_idx])
        ys.append((y[valid_idx], cur_pred))
    return ys    
