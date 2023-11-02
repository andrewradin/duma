#!/usr/bin/env python

import sys
try:
    from dtk.files import get_file_records
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
    from dtk.files import get_file_records
from collections import defaultdict

class scoreObj(object):
    def __init__(self):
        self.scores = defaultdict(dict)
    def update(self, gene, mir, score):
        try:
            current = self.scores[gene][mir]
        except KeyError:
            current = 0.0
        self.scores[gene][mir] = score if abs(score) > abs(current) else current
    def report(self):
        for gene in self.scores.keys():
            for mir, score in self.scores[gene].items():
                print("\t".join([gene, mir, str(score)]))

if __name__=='__main__':
    import argparse
    import operator
    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="Parse TargetScan default target predictions")
    arguments.add_argument("i", help="Predicted_Targets_Context_Scores.default_predictions.txt")
    args = arguments.parse_args()
    score_name = 'weighted context++ score percentile'
    header=[]
    results = scoreObj()
    for frs in get_file_records(args.i, parse_type='tsv'):
        if not header:
            header = frs
            continue
        if frs[header.index('Gene Tax ID')] != '9606':
            continue
### Using the percentile of the highest confidence predicted score
### This will collapse the upper end of the scores, but does result in 0-1 bounded scores
        score = float(frs[header.index(score_name)])/100.0
        if abs(score) > 0:
           # there is a trailing decimal, presumably for isoforms, which we don't care about
            gene = frs[header.index('Gene ID')].split('.')[0]
            results.update(gene,
                           frs[header.index('miRNA')],
                           score
                          )

    print("\t".join(['Ensembl', 'miRNA', 'score']))
    results.report()
