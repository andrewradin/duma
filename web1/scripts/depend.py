#!/usr/bin/env python3

import sys
from path_helper import PathHelper
import os

from algorithms.exit_codes import ExitCoder
import logging
logger = logging.getLogger("depend")


def psmax_score(drug_score_dict, drug_direc_dict, disease_score, disease_direction):
    gen = (k for k in drug_score_dict if k in disease_score)
# could also average, not multiply.
# I like multiply b/c it makes the drug-prot set scores more important
# (they're much smaller than the NES)
    all_scores = []
    all_dirs = []
    for protSet in gen:
        # multiplying is the current approach #a
        all_scores.append(disease_score[protSet] * drug_score_dict[protSet])
# TODO try this averaging approach, but only with #4 and 5 below - #b
#        all_scores.append((disease_score[protSet] + drug_score_dict[protSet]) / 2.0)
        all_dirs.append(drug_direc_dict[protSet] * disease_direction[protSet])
# could also report a median or max for the score
# The -1 for dir flips the direction so that it's more intuitive
# (pos means better, neg means worse)
    dir = sum(all_dirs) if all_dirs else 0.0
    return max(all_scores) if all_scores else 0.0, 0.0 if dir == 0.0 else dir * -1.0

import numpy as np
def psmean_score(drug_score_dict, drug_direc_dict, disease_score, disease_direction):
    gen = (k for k in drug_score_dict if k in disease_score)
    all_scores = []
    for protSet in gen:
        all_scores.append(disease_score[protSet] * drug_score_dict[protSet])
    return np.mean(all_scores) if all_scores else 0.0, 0.0

def pssoftmax_score(drug_score_dict, drug_direc_dict, disease_score, disease_direction):
    # This is logsumexp, not to be confused with the related softargmax which is also sometimes
    # just called softmax.
    # The key inequality here is: max(X) <= softmax(X) <= max(X) + log(N)
    # softmax(X) tends towards max(X) as non-max elements tend towards 0
    # softmax(X) tends towards max(X) + log(N) as non-max elements tend towards max(X)
    #
    # The log(N) term does make the absolute value of the score relative to log(N) important.
    gen = (k for k in drug_score_dict if k in disease_score)
    all_scores = []
    for protSet in gen:
        all_scores.append(disease_score[protSet] * drug_score_dict[protSet])
    #from scipy.special import logsumexp
    #return logsumexp(all_scores) if all_scores else 0.0, 0.0
    return np.log(np.sum(np.exp(all_scores)) - len(all_scores)) if all_scores else 0.0, 0.0

def pssmoothmax_score(drug_score_dict, drug_direc_dict, disease_score, disease_direction):
    gen = (k for k in drug_score_dict if k in disease_score)
    all_scores = []
    for protSet in gen:
        all_scores.append(disease_score[protSet] * drug_score_dict[protSet])
    if not all_scores:
        return 0.0, 0.0

    ALPHA = 2
    from scipy.special import softmax
    all_scores = np.array(all_scores)
    weighted = all_scores * softmax(ALPHA * all_scores)
    return np.sum(weighted), 0.0

def pssum_score(drug_score_dict, drug_direc_dict, disease_score, disease_direction):
    gen = (k for k in drug_score_dict if k in disease_score)
    all_scores = []
    for protSet in gen:
        all_scores.append(disease_score[protSet] * drug_score_dict[protSet])
    return np.sum(all_scores) if all_scores else 0.0, 0.0

def pscorr_score(drug_score_dict, drug_direc_dict, disease_score, disease_direction):
    pathways = drug_score_dict.keys() | disease_score.keys()

    a = []
    b = []
    for pathway in pathways:
        a.append(drug_score_dict.get(pathway, 0))
        b.append(disease_score.get(pathway, 0))

    from scipy.stats import spearmanr
    corr, pv = spearmanr(a, b)
    return (corr+1)/2, 0

def psgeodim_score(drug_score_dict, drug_direc_dict, disease_score, disease_direction):
    drug_ord = sorted(drug_score_dict.items(), key=lambda x: -x[1])

    total = 0
    R = 0.9
    for i, (pw, score) in enumerate(drug_ord):
        mult = R ** i
        total += score * disease_score.get(pw, 0) * mult

    N = len(drug_ord)
    norm = (1 - R ** N) / (1 - R)
    return total / norm, 0


SCORE_METHODS = {
        'psmax': psmax_score,
        'pssoftmax': pssoftmax_score,
        'pssmoothmax': pssmoothmax_score,
        'pssum': pssum_score,
        'psmean': psmean_score,
        #'pscorr': pscorr_score,
        #'psgeodim': psgeodim_score,
        }


class depend:
    def __init__(self, gl, outfile, ppi, ps, moas, cachedir, d2ps_method, score_method, combo, **kwargs):
        self.input = gl
        import json
        from dtk.d2ps import MoA
        with open(moas) as f:
            self.moas = [MoA(x) for x in json.loads(f.read())]
        self.ppi = ppi
        self.ps = ps
        self.output = []
        self.ofile = outfile
        self.cachedir = cachedir
        self.d2ps_method = d2ps_method
        self.combo = combo

        self.score_fn = SCORE_METHODS[score_method]

    def run(self):
        self.get_scores()
        self.check_d2ps_file()
        self.calculate_all_scores()

    def check_d2ps_file(self):
        from dtk.d2ps import D2ps
        self.d2ps = D2ps(self.ppi, self.ps, cachedir=self.cachedir, method=self.d2ps_method)
        self.d2ps.update_for_moas(self.moas)

    def get_scores(self):
        self.in_scores, self.in_directions = load_gl_data(self.input)

    def get_moa_ps_connections(self, moa):
        records = self.d2ps.get_moa_pathway_scores(moa)
        scores = {x.pathway:x.score for x in records}
        dirs = {x.pathway:x.direction for x in records}
        return scores, dirs

    def apply_combo(self, combo_scores, moa_ps_scores, combo_dirs, moa_ps_dirs):
        for moa in moa_ps_scores:
            if moa in combo_scores:
                updated_score = moa_ps_scores[moa] - combo_scores[moa]
                moa_ps_scores[moa] = max(0, updated_score)

    def calculate_all_scores(self):
        from dtk.files import get_file_records


        logger.info("Computing all scores")
        if self.combo:
            from dtk.combo import base_dpi_data
            from dtk.d2ps import MoA
            combo_dpi = base_dpi_data(self.combo)
            # Convert from {uniprot: (evid, dir)} to (uniprot, evid, dir)
            combo_moa = MoA((k, *v) for k, v in combo_dpi.items())
            self.d2ps.update_for_moas([combo_moa])
            combo_scores, combo_dirs = self.get_moa_ps_connections(combo_moa)
            logger.info(f"Setting up combo scores, base drug has {len(combo_scores)} pathway connections")
        else:
            combo_scores, combo_dirs = (None, None)


        with open(self.ofile, 'w') as o:
            header = ["moa", 'psScoreMax', 'direction']
            o.write("\t".join(header) + "\n")

            from tqdm import tqdm
            for moa in tqdm(self.moas):
                moa_ps_scores, moa_ps_dirs = self.get_moa_ps_connections(moa)
                if combo_scores:
                    self.apply_combo(combo_scores, moa_ps_scores, combo_dirs, moa_ps_dirs)
                if len(moa_ps_scores) > 0:
                    s, d = self.score_fn(moa_ps_scores,
                                        moa_ps_dirs,
                                        self.in_scores,
                                        self.in_directions
                                       )
                    lines = self.get_lines(s, d, moa)
                    self.output += lines

            o.write("\n".join(list(set(self.output))) + "\n")
    def get_lines(self, score, dir, d):
        try:
            return ["\t".join([str(i) for i in [d, score, dir]])]
        except AttributeError:
            return []

def load_gl_data(ifile=None, in_src=None):
    """Takes an input file or an iterable with the glee/glf data."""
    if ifile is not None:
        from dtk.files import get_file_records
        src = get_file_records(ifile)
        assert next(src) == ['uniprotset','score','direction']
    elif in_src is not None:
        src = in_src
    else:
        raise RuntimeError("Must specify either an input file or input source")

    glee_scores = {}
    glee_directions = {}
    for fields in src:
        glee_scores[fields[0]] = float(fields[1])
        glee_directions[fields[0]] = float(fields[2])
    return glee_scores, glee_directions

def get_dir_name(score_name):
# currently there is only one direction type, but there may be more in the future
    return 'enrich_dir'

if __name__ == "__main__":

    import argparse
    import time
    from dtk import log_setup
    parser = argparse.ArgumentParser(description='run DEEPEnD')
    parser.add_argument('--gl', help="glee/glf output")
    parser.add_argument('--moas', help="File with MoAs to compute")
    parser.add_argument('--ppi', help='PPI to use')
    parser.add_argument('--ps', help='Pathways to use')
    parser.add_argument('--d2ps-method', help='Method for drugs to pathways')
    parser.add_argument('--score-method', help='Method for scoring drugs across all pathways')
    parser.add_argument('--cachedir', default=None, help='Where to store d2ps cache')
    parser.add_argument('--combo', default=None, help='Name of the combo base drug to evaluate against')
    parser.add_argument('-o', '--outfile', help="file to report results")
    log_setup.addLoggingArgs(parser)

    args=parser.parse_args()

    log_setup.setupLogging(args)

    ts = time.time()
    run = depend(**vars(args))
    run.run()
    print("Took: " + str(time.time() - ts) + ' seconds')
