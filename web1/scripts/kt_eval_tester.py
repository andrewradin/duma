#!/usr/bin/env python3

from builtins import range
import sys
try:
    from path_helper import PathHelper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")

from algorithms.exit_codes import ExitCoder
import numpy as np
import random

class tester:
    def __init__(self,**kwargs):
        self.tests = kwargs.get('tests', ['all'])
        self.tail_portions = kwargs.get('tail_portions', [0,0.5])
        self.ndrugs = kwargs.get('ndrugs',10000)
        self.nkts = kwargs.get('nkts',50)
        self.top_portion = kwargs.get('top_portion',0.02)
        self.seed = kwargs.get('seed',666)
        self.get_exponent_range()
    def get_exponent_range(self):
        self.exponent_range = [0.025, 0.1, 0.4, 1.0, 2.5, 4.0]
    def run(self):
        self.get_test_list()
        for tail_por in self.tail_portions:
            self.set_consistent_kwargs(tail_por)
            for test in self.tests:
                # this expon range doesn't work if we don't want to use an exponential distribution
                for expon in self.exponent_range:
                    name = test + "_tailportion" + str(tail_por) + '_exp' + str(expon)
                    test_case = self.build_test(test)
                    self.good_score_kwargs['expon'] = expon
                    self.bad_score_kwargs['expon'] = expon
                    yield {name: test_case}
    def set_consistent_kwargs(self, tail_por):
        self.score_kwargs={'ndrugs': self.ndrugs,
                           'tail_length': int(tail_por * self.ndrugs)
                          }
        self.kt_kwargs={'ndrugs': self.ndrugs,
                        'nkts': self.nkts
                       }
    def build_test(self, test):
        self.set_kwargs(test)
        good_case = self.build_case(self.good_score_kwargs, self.good_kt_kwargs)
        bad_case = self.build_case(self.bad_score_kwargs, self.bad_kt_kwargs)
        return [good_case, bad_case]
    def get_test_list(self):
        if 'all' in self.tests:
            self.tests = self.all_tests
        else:
            for x in self.tests:
                assert x in self.all_tests
            self.tests = list(set(self.tests))
    def set_kwargs(self, test):
        self.reset_kwargs()
        if test == 'easy':
            # the defaults set in reset_kwargs are perfect
            # evenly distribute for the bad case
            self.bad_kt_kwargs['top_max'] = None # disable the enrichment at the top
            self.bad_kt_kwargs['remaining_kt_dist'] = 'even'
        elif test == 'some_v_all':
            # all the KTs are in the top by default
            # put some (half in this case) of the KTs in the top for the bad case
            self.bad_kt_kwargs['kt_portion_in_top'] = 0.5
            self.bad_kt_kwargs['top_min'] = self.good_kt_kwargs['top_min']
            self.bad_kt_kwargs['top_max'] = self.good_kt_kwargs['top_max']
            self.bad_kt_kwargs['remaining_kt_dist'] = 'even'
        elif test == 'some_v_none':
            # Now some (half) is the good case
            self.good_kt_kwargs['kt_portion_in_top'] = 0.5
            # the bad case is as for easy
            self.bad_kt_kwargs['top_max'] = None
            self.bad_kt_kwargs['remaining_kt_dist'] = 'even'
        elif test == 'KTtop_plateau':
            # same for the bad case, the KTs will stay the same,
            # but we'll change the score distribution to include a plateau
            self.bad_kt_kwargs['kt_portion_in_top'] = self.good_kt_kwargs['kt_portion_in_top']
            self.bad_kt_kwargs['top_min'] = self.good_kt_kwargs['top_min']
            self.bad_kt_kwargs['top_max'] = self.good_kt_kwargs['top_max']
            self.bad_score_kwargs['plateau_length'] = int(self.ndrugs * self.top_portion)
        elif test == 'KTmid_plateau':
            # Good in this case is KTs in the plateau, but not at the very highest scores
            # To do that we'll change the defintion of top to be shifted to the left
            self.good_kt_kwargs['top_min'] = int(self.ndrugs * self.top_portion)
            self.good_kt_kwargs['top_max'] = int(self.ndrugs * self.top_portion * 2)
            self.good_score_kwargs['plateau_length'] = int(self.ndrugs * self.top_portion * 2)
            # bad in this case is the same KT dist, but with no plateau
            self.bad_kt_kwargs['kt_portion_in_top'] = self.good_kt_kwargs['kt_portion_in_top']
            self.bad_kt_kwargs['top_min'] = self.good_kt_kwargs['top_min']
            self.bad_kt_kwargs['top_max'] = self.good_kt_kwargs['top_max']
        else:
            sys.stderr.write("Unrecognized test name. Quitting.")
            sys.exit(ExitCoder.encode('usageError'))
    def reset_kwargs(self):
        self.good_score_kwargs = self.score_kwargs.copy()
        self.good_kt_kwargs = self.kt_kwargs.copy()
        self.bad_score_kwargs = self.score_kwargs.copy()
        self.bad_kt_kwargs = self.kt_kwargs.copy()
        self.good_kt_kwargs['kt_portion_in_top'] = 1.0
        self.good_kt_kwargs['top_min'] = 1
        self.good_kt_kwargs['top_max'] = int(self.ndrugs * self.top_portion)
    def build_case(self, score_kwargs, kt_kwargs):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        scores = score_dist(**score_kwargs)
        scores.run()
        kts = kt_dist(**kt_kwargs)
        kts.run()
        keyed_scores = [(i,v)  for i,v in enumerate(scores.scores)]
        assert len(keyed_scores) >= max(kts.kt_inds)
        kt_keys = [keyed_scores[i][0] for i in kts.kt_inds]
        return (keyed_scores, kt_keys)

class score_dist:
    def __init__(self, **kwargs):
        self.ndrugs = kwargs.get('ndrugs',10000)
        self.expon = kwargs.get('expon',3)
        self.tail_length = kwargs.get('tail_length',0)
        self.plateau_length = kwargs.get('plateau_length',0)
        self.plateau_range = kwargs.get('plateau_range',0.01)
        self.nscored_drugs = self.ndrugs - self.tail_length
        self.max_score = 10000 # arbitrary number to kep the scores from getting so big that we can't square them
    def run(self):
        # randomly grab ndrugs numbers from 1-ndrugs as the input, with replacement.
        # this will mimic ties
        x = np.random.random_integers(1, self.nscored_drugs, size = self.nscored_drugs)
        self.scores = [float(i) ** self.expon for i in x]
        if self.plateau_length > 1:
            self.scores.sort(reverse=True)
            peak = max(x) ** self.expon
            end = peak * (1 - self.plateau_range)
            plateau = list(np.linspace(peak, end, self.plateau_length))
            self.scores[:self.plateau_length] = plateau
            assert len(self.scores) == self.nscored_drugs
        self.scores += [0]*self.tail_length
        self.scores.sort(reverse=True)
# since the scores don't actually matter (just their relative values), just take an easy value we know will keep the scors below the cap
        if max(self.scores) > self.max_score:
            denom = int(max(self.scores) / self.max_score) + 1 
            self.scores = [float(i/denom) for i in self.scores]

class kt_dist:
    def __init__(self, **kwargs):
        self.ndrugs = kwargs.get('ndrugs',None)
        self.nkts = kwargs.get('nkts',20)
        assert self.ndrugs > self.nkts
        self.top_min = kwargs.get('top_min',1)
        self.top_max = kwargs.get('top_max',500)
        self.kt_portion_in_top = kwargs.get('kt_portion_in_top',0.5)
        self.remaining_kt_dist = kwargs.get('remaining_kt_dist','even')
    def run(self):
        self.kt_inds = []
        if self.top_min and self.top_max:
            top_nkt = int(self.kt_portion_in_top * self.nkts)
            assert self.top_max - self.top_min >= top_nkt
            # the minus one is to get these to be indicies for 0-indexed lists
            self.kt_inds += random.sample(list(range(self.top_min - 1, self.top_max - 1)), top_nkt)
        remaining_kts = self.nkts - len(self.kt_inds)
        try:
            remaing_score_space = self.ndrugs - self.top_max
        except TypeError:
            remaing_score_space = self.ndrugs
            self.top_max = 0
        assert remaing_score_space > remaining_kts
        if self.remaining_kt_dist == 'even':
            self.kt_inds += [int(i) for i in np.linspace(self.top_max
                                                         , self.ndrugs - 1
                                                         , remaining_kts
                                                        )]
        elif self.remaining_kt_dist == 'random':
            self.kt_inds += random.sample(list(range(self.top_max, self.ndrugs - 1)), remaining_kts)
        elif self.remaining_kt_dist == 'end': 
            self.kt_inds += list(range(self.ndrugs-remaining_kts, self.ndrugs-1))
        else:
            sys.stderr.write("Unrecognized remaining_kt_dist. Support options are even or random. Quitting.\n")
            sys.exit(ExitCoder.encode('usageError'))

if __name__ == "__main__":
    # force unbuffered output
    sys.stdout = sys.stderr

    import argparse, time
    parser = argparse.ArgumentParser(description='run KT-based evaluation tester')
    args=parser.parse_args()

    ts = time.time()
    run = tester()
# run.run() returns a generator of dictionaries, where the key is the test name,
# and the value is a list of cases: good then bad.
# Each case is a list, where the first entry is a list of tuples: drug_id, score
# the other entry in the list is a list of drug_ids that correspond to the KTs
    import febe
    import score_plots
    for dict in run.run():
        good = febe.febe(named_scores =  list(dict.values())[0][0][0], names_of_interest = list(dict.values())[0][0][1])
        good.run()
        bad = febe.febe(named_scores =  list(dict.values())[0][1][0], names_of_interest = list(dict.values())[0][1][1])
        bad.run()
        plotter = score_plots.plot_scores(named_scores =  list(dict.values())[0][0][0], names_of_interest = list(dict.values())[0][0][1])
        plotter.run('good.png')
        plotter2 = score_plots.plot_scores(named_scores =  list(dict.values())[0][1][0], names_of_interest = list(dict.values())[0][1][1])
        plotter2.run('bad.png')
        sys.exit(1)
