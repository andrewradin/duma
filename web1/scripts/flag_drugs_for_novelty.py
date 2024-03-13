#!/usr/bin/env python3


from __future__ import print_function
import sys
import six
try:
    import path_helper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")
    import path_helper

import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")

import django
from browse.models import Workspace
django.setup()

from flagging.utils import FlaggerBase

class NoveltyFlagger(FlaggerBase):
    def __init__(self, **kwargs):
        super(NoveltyFlagger,self).__init__(kwargs)
        self.alpha = kwargs.pop('alpha')
        self.thresh = kwargs.pop('thresh')
        assert not kwargs
    def flag_drugs(self):
        self.create_flag_set('Novelty')
        # calculate novelty score
        import algorithms.run_lbn
        novelty_dict = algorithms.run_lbn.calculate_novelty(
                        self.ws.name,
                        self.each_target_wsa(),
                        )
        from django.urls import reverse
        for wsa_id, results_dict in six.iteritems(novelty_dict):
            flag_category = 'Drug-Disease Novelty'
            log_odds = results_dict['log_odds']
            pvalue = results_dict['pvalue']
            disease_name = results_dict['disease_name']
            drug_name = results_dict['drug_name']
            detail_string = "Co-ocurrance of {0} and {1} on PubMed".format(drug_name, disease_name)
            pubmed_url = "https://www.ncbi.nlm.nih.gov/pubmed/?term={0}+AND+{1}".format(drug_name, disease_name)
            print(drug_name, pvalue, log_odds)
            if (pvalue < self.alpha) and (log_odds > self.thresh):
                # Drugs are flagged for questionable novelty if the fisher-exact test was significant and
                # the log odds ratio (logit) of finding a co-occurance of drug and disease is higher than zero.
                print('Drug flagged:', drug_name)
                self.create_flag(
                    wsa_id=wsa_id,
                    category=flag_category,
                    detail=detail_string,
                    href=pubmed_url
                    )
 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
             description = "flag drugs for novelty")
    parser.add_argument('--start', type=int, default=0, help="starting rank on scoreboard to begin testing for novelty")
    parser.add_argument('--count', type=int, default=200, help="total number of drugs to test in scoreboard")
    parser.add_argument('ws_id', type=int, help="workspace id")
    parser.add_argument('job_id', type=int, help="job id")
    parser.add_argument('score', help="score within job_id to rank drugs by")
    parser.add_argument('--thresh', type=float, default=-1.0, help='Minimum log odds ratio that results in a flag')
    parser.add_argument('--alpha', type=float, default=0.05, help="the fisher exact test significance level required to consider flagging for novelty concerns")
    args = parser.parse_args()

    from runner.models import Process
    err = Process.prod_user_error()
    if err:
        raise RuntimeError(err)

    flagger = NoveltyFlagger(
                ws_id=args.ws_id,
                job_id=args.job_id,
                start=args.start,
                count=args.count,
                score=args.score,
                alpha=args.alpha,
                thresh=args.thresh)
    flagger.flag_drugs()
