#!/usr/bin/python
import argparse, os, sys
from collections import defaultdict
arguments = argparse.ArgumentParser(description="Given a directory of validatoinStats (from meta_wrapper), aggregate the validation stats")
arguments.add_argument("-i", help="validationStats dir. In this dir should be only subdirectories for each iteration. Within each of those, only files with the probability cutoff. e.g.0.1_confusionMatrix.tsv")
args = arguments.parse_args()

#[1] "DB01050"
#Confusion Matrix and Statistics
#
#          Reference
#Prediction    0    1
#         0 3658   53
#         1   22    0
#
#               Accuracy : 0.9799
#                 95% CI : (0.9749, 0.9842)
#    No Information Rate : 0.9858
#    P-Value [Acc > NIR] : 0.998401
#
#                  Kappa : -0.0084
# Mcnemar's Test P-Value : 0.000532
#
#            Sensitivity : 0.000000
#            Specificity : 0.994022
#         Pos Pred Value : 0.000000
#         Neg Pred Value : 0.985718
#             Prevalence : 0.014198
#         Detection Rate : 0.000000
#   Detection Prevalence : 0.005893
#      Balanced Accuracy : 0.497011
#
#       'Positive' Class : 1
#
#[1] "DB00658"


balanced_accuracy = defaultdict(lambda: defaultdict(list))
accuracy = defaultdict(lambda: defaultdict(list))
positive_pred_value = defaultdict(lambda: defaultdict(list))
Neg_pred_value = defaultdict(lambda: defaultdict(list))
Sensitivity = defaultdict(lambda: defaultdict(list))
Specificity = defaultdict(lambda: defaultdict(list))
kappa = defaultdict(lambda: defaultdict(list))
pval = defaultdict(lambda: defaultdict(list))

dirs = os.listdir(args.i)
for iter in dirs:
    files = os.listdir(os.path.join(args.i,iter))
    for file in files:
        cutoff = file.split("_")[0]
        with open(os.path.join(args.i, iter, file), 'r') as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue
                l = line.strip().split()
                if l[0] =="Sensitivity":
                    Sensitivity[iter][cutoff].append(l[-1])
                elif l[0] =="Specificity":
                    Specificity[iter][cutoff].append(l[-1])
                elif l[0] =="Pos":
                    positive_pred_value[iter][cutoff].append(l[-1])
                elif l[0] =="Neg":
                    Neg_pred_value[iter][cutoff].append(l[-1])
                elif l[0] =="Balanced":
                    balanced_accuracy[iter][cutoff].append(l[-1])
                elif l[0] =="Kappa":
                    kappa[iter][cutoff].append(l[-1])
                elif l[0] =="Accuracy":
                    accuracy[iter][cutoff].append(l[-1])
                elif l[0] =="P-Value":
                    pval[iter][cutoff].append(l[-1])

for name in ['pval', 'accuracy', 'kappa', 'balanced_accuracy', 'Neg_pred_value', 'positive_pred_value', 'Specificity', 'Sensitivity']:
    dd = eval(name)
    with open(os.path.join(args.i, name+".tsv"), 'w') as f:
        f.write("\t".join(['cutoff', 'iteration', 'value']) + "\n")
        for i in sorted(dd):
            for c in sorted(dd[i]):
                for v in dd[i][c]:
                    f.write("\t".join([c, i, v]) + "\n")
