#!/usr/bin/env python3

from __future__ import print_function
from builtins import range
import sys
try:
    from old_dea import findListMatches
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")
    from old_dea import findListMatches

import os
from algorithms.exit_codes import ExitCoder
from math import log

class febe:
    def __init__(self, **kwargs):
        self.named_scores = kwargs.get('named_scores',None)
        self.noi = kwargs.get('names_of_interest',None)
        self.filename = kwargs.get('filename', '/tmp/febe.png')
    def run(self):
        self.setup()
        self.check_enrichments()
    def report_results(self):
        self.plot_qvals()
        print("Initial scan results:")
        for i in range(len(self.inds)):
            print("\t".join([str(x) for x in [self.inds[i]+1, self.qvals[i], self.ors[i]]]))
    def plot_qvals(self):
        if not self.filename:
            return
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(self.inds, [ -1.0 * log(i, 10) for i in self.qvals], linewidth=2.0)
        plt.ylim([0, self.final_score * 1.05])
        plt.ylabel('-Log10(drug set enrichment FDR)')
        plt.xlabel('Drug rank')
        plt.title('Fishers Exact Based Evaluation')
        plt.savefig(self.filename)
        plt.close()
    def check_enrichments(self):
        self._get_enrichments()
        self._score_all()
    def _score_all(self):
        self.final_score = -1.0 * log(self.best_q, 10)
    def _get_enrichments(self):
        from dtk.enrichment import mult_hyp_correct
        self.ors = []
        ps = []
        for ind in self.inds:
            oddsr, p = self.single_enrich_check(ind)
            self.ors.append(oddsr)
            ps.append(p)
        if len(ps) == 0:
            self.best_q = 1.0
            self.peak_ind = len(self.named_scores)-1
            self.qvals = [1.0]*len(self.inds)
            self.best_or = 0.
        else:
            if len(ps) > 1:
                self.qvals = list(mult_hyp_correct(ps))
            else:
                self.qvals = ps
            # p values and q values should have the same ordering, but the
            # multi_hypothesis_correction may have driven many of the q values
            # to identical values, so use the p value to select the best idx.
            best_p = min(ps)
            best_p_idx = ps.index(best_p)
            self.best_q = min(self.qvals)
            self.peak_ind = self.inds[best_p_idx]
            self.best_or = self.ors[best_p_idx]
    def single_enrich_check(self, ind, inf_bump=0.1):
        import scipy.stats as stats
        from numpy import isfinite
        from dtk.num import get_non_zero_min
        match_in = sum(self.matches[:ind+1])
        match_out = self.total_matches - match_in
        non_in = ind+1 - match_in
        non_out = self.total_non - non_in
        oddsr,p = stats.fisher_exact([[match_in, non_in],
                                   [match_out, non_out]
                                  ],
                                   alternative = 'greater'
                                  )
        ### We don't want infite ORs because we use them for scoring
        ### the solution is to give every value a proportional bump up
        ### so that ratios, and therefore the OR stay similar
        ### To do this we add a set value (inf_bump) to the two smallest values
        ### that is 0 (which is how we got an infinite OR) and then smallest_val.
        ### Then we give the other two values a bump that is proportional to the bump
        ### that the smallest_val got
        if not isfinite(oddsr):
            l = [match_in, non_out, non_in, match_out]
            smallest_val = get_non_zero_min(l)
            sv_bump_portion = inf_bump/(smallest_val + inf_bump)
            for i,v in enumerate(l):
                if v <= smallest_val:
                    l[i] = v + inf_bump
                else:
                    l[i] = v * (1+sv_bump_portion)
            oddsr = (l[0] * l[1]) / (l[2] * l[3])
        return oddsr,p
    def setup(self):
        self.named_scores = list(reversed(sorted(self.named_scores, key = lambda x: float(x[1]))))
        self.count_matches()
        self.scores = [float(x[1]) for x in self.named_scores]
        self.get_inds()
    def count_matches(self):
        self.matches = findListMatches(self.named_scores, self.noi)
        self.total_matches = sum(self.matches)
        self.total_non = len(self.matches) - self.total_matches
    def get_inds(self):
        from dtk.enrichment import get_tie_adjusted_ranks
        scores = [score for name, score in self.named_scores]
        hit_inds = get_tie_adjusted_ranks(scores, self.matches)

        # The FE test counts how matches are before/after each index,
        # so we have to go back and make the matches array correspond to our
        # tie-adjusted ranks.
        hit_ind_set = set(hit_inds)
        self.matches = [1 if i in hit_ind_set else 0 for i in range(len(self.matches))]

        self.inds = list(set(hit_inds+[x-1 for x in hit_inds if x]))
        self.inds.sort(reverse=True)

class wfebe(febe):
    def _score_all(self):
        # the scoring combines the -Log10 of FEBE w/a weight
        # which is essentially how close to the left is the value
        # This emphasizes finding a lot of the ref set at the top of the rankings,
        # even if it doesn't have the best enrichment, or put the other way
        # tries to downplay large significant enrichments that aren't
        # at the top of the rankings.
        ### I also tried weighting on what portion of the reference set (e.g. KTs) are higher ranked,
        ### but that was essentially counteracting the other weight,
        ### and pushing maximal scores to the right
        ### I left the weight below, but commented it out

        self.wfebe_scores = [
                  (-1.0 * log(tup[1],10) *
                   (1.- float(tup[0])/len(self.named_scores))
                  )
                  for i,tup in enumerate(zip(self.inds, self.qvals))
                 ]
        self._finalize_score()
    def _finalize_score(self):
        if self.wfebe_scores:
            self.final_score = max(self.wfebe_scores)
            self.peak_ind = self.inds[self.wfebe_scores.index(self.final_score)]
        else:
            self.final_score = 0
            self.peak_ind = len(self.named_scores)-1

class glf(wfebe):
# inf_replacement was selected as it was a bit higher than any established ORs we have yet to see
# it is equivalent to an OR of ~125,000
    def _score_all(self, inf_replacement=20.):
        from numpy import isinf
        # updates wfebe only by adding the OR to the -log10(q) before weighting
        self.wfebe_scores = []
        for i,tup in enumerate(zip(self.inds, self.qvals, self.ors)):
            qScore = -1. * log(tup[1],10)
            if tup[2] == 0.:
                orScore = 0.
            elif isinf(tup[2]):
                orScore = inf_replacement
            else:
                orScore = log(tup[2],2)
            topWt = 1.- float(tup[0])/len(self.named_scores)
            self.wfebe_scores.append(orScore*topWt)
        self._finalize_score()


if __name__ == "__main__":
    # force unbuffered output
    sys.stdout = sys.stderr
