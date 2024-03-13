#!/usr/bin/env python3

from __future__ import print_function
from builtins import range
import sys
# retrieve the path where this program is located, and
# use it to construct the path of the website_root directory
try:
    from path_helper import PathHelper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")

import os

from algorithms.exit_codes import ExitCoder
from dea import bootstrap_enrichment

# glee specific imports
from collections import defaultdict
from dtk.enrichment import mult_hyp_correct
import numpy as np

class glee_runner:
    def __init__(self,**kwargs):
        self.multiple_input = kwargs.get('multiple_input',None)
        self.fake_mp = kwargs.get('fake_mp',None)
        self.cores = kwargs.get('cores',None)
        self.ws = kwargs.get('ws',None)
        self.weight = kwargs.get('wt',None)
        self.nPermuts = kwargs.get('nPermuts',None)
        self.alpha = kwargs.get('alpha',None)
        self.infile = kwargs.get('infile',None)
        self.gl_file = kwargs.get('glfile',None)
        self.outdir = kwargs.get('outdir',None)
        self.tmp_pubdir = kwargs.get('tmp_pubdir',None)
        self.relevant_gl_data = {}
        # output files
        # published output files
    def run(self):
        input = open_infile(self.infile)
        if self.multiple_input:
            self.total_tissues = len(list(input.keys()))
            self.all_gles = defaultdict(lambda : defaultdict(dict))
            for k,v in input.items():
                self.current_tis = k
                self.ofile = self.outdir + self.current_tis + "_glee.tsv"
                self.lt = v
                self.run_single()
            self.report_combined()
        else:
            self.lt = input
            self.ofile = self.outdir + "glee.tsv"
            self.run_single()
    def report_combined(self):
        import operator
        # combine all of the ofiles into one consensus
        self.combined_outfile = self.outdir + "glee.tsv"
        # self.all_gles is a 3-deep dictionary keyed by gene lists name, tissue name, and stat type
        out_combined_data = []
        for l in self.all_gles.keys():
            all_n = []
            all_nes = []
            all_directional_evid = []
            sig_pos = []
            sig_neg = []
            for t in self.all_gles[l].keys():
                all_n.append( self.all_gles[l][t]['n'] )
                all_nes.append( self.all_gles[l][t]['nes'] )
                all_directional_evid.append( np.sign(self.all_gles[l][t]['nes']) * (1.0 - self.all_gles[l][t]['q']) )
                if self.all_gles[l][t]['q'] <= 0.05:
                    assert self.all_gles[l][t]['nes'] != 0
                    if self.all_gles[l][t]['nes'] > 0:
                        sig_pos.append(t)
                    elif self.all_gles[l][t]['nes'] < 0:
                        sig_neg.append(t)
            all_nes += [0.0] * (self.total_tissues - len(all_nes))
            all_directional_evid += [0.0] * (self.total_tissues - len(all_directional_evid))
            mean_directional_evid = float( np.mean(np.array(all_directional_evid)))
            mean_direc_q = np.sign(mean_directional_evid) * (1.0 - abs(mean_directional_evid))
            out_combined_data.append((mean_directional_evid
                                      , l
                                      , mean_direc_q
                                      , float( np.mean(np.array(all_nes)))
                                      , float(sum(all_n)) / self.total_tissues
                                      , ", ".join(sig_pos)
                                      , ", ".join(sig_neg)
                                    ))
        out_combined_data.sort(key = operator.itemgetter(0), reverse = True)
        with open(self.combined_outfile, 'w') as f:
            f.write("\t".join(['Gene list', 'Mean directional FDR', 'Mean NES', 'Mean n prots', 'Signif. +NES', 'Signif. -NES']) + "\n")
            for tup in out_combined_data:
                f.write("\t".join([str(i) for i in tup[1:]]) + "\n")
    def run_single(self):
        self.score_list = get_gesig(self.lt)
        self.nonzero_score_unis = [x[0] for x in self.score_list if x[1]]
        self.check_enrich()
        self.report()
    def check_enrich(self):
        import operator
        if not self.relevant_gl_data:
            self._load_gl()
        n_iter = len(self.relevant_gl_data)
        params = zip([self.score_list] * n_iter,
                      list(self.relevant_gl_data.values()),
                      list(self.relevant_gl_data.keys()),
                      [self.weight] * n_iter,
                      [self.nPermuts] * n_iter,
                      [self.alpha] * n_iter,
                      )
        if self.fake_mp:
            results = list(map(run_single_glee, params))
        else:
            import multiprocessing
            pool = multiprocessing.Pool(self.cores)
            results = list(pool.map(run_single_glee, params))
        # now add the missing gene lists that had no overlap and thus have the default values
        results += [(x, 0, 0, ['0'] * 5, 1.0) for x in self.skipped_gl]
        # sort by p-val
        results.sort(key=operator.itemgetter(1,2), reverse=True)
        qs = mult_hyp_correct([out_tup[4] for out_tup in results])
        self.gles = []
        for i in range(len(results)):
            out_tup = results[i]
            self.gles.append("\t".join([out_tup[0]] + list(out_tup[3]) + [str(out_tup[4]), str(qs[i])]))
            if self.multiple_input:
                self.all_gles[out_tup[0]][self.current_tis]['n'] = float(out_tup[3][0])
                self.all_gles[out_tup[0]][self.current_tis]['setPor'] = float(out_tup[3][1])
                self.all_gles[out_tup[0]][self.current_tis]['nes'] = float(out_tup[3][3])
                self.all_gles[out_tup[0]][self.current_tis]['q'] = qs[i]
    def _load_gl(self):
        from dtk.files import get_file_records
        if not os.path.isfile(self.gl_file):
            from dtk.s3_cache import S3Bucket, S3File
            f=S3File(S3Bucket('glee'),os.path.basename(self.gl_file))
            f.fetch()
        all_gl = set()
        for frs in get_file_records(self.gl_file, parse_type = 'tsv'):
            unis = frs[1].split(',')
            all_gl.add(frs[0])
            if not set(unis).isdisjoint(self.nonzero_score_unis):
                self.relevant_gl_data[frs[0]] = unis
        self.skipped_gl = all_gl - set(self.relevant_gl_data.keys())
    def report(self):
        with open(self.ofile, 'w') as f:
            f.write("\t".join(['List_name', 'nProts', 'setPor', 'ES', 'NES', "NES_lower", "NES_upper", "p-value", "q-value"]) + "\n")
            f.write("\n".join(self.gles) + "\n")
def open_infile(infile):
    import pickle
    with open(infile, 'rb') as handle:
        lt = pickle.load(handle)
    return lt
def get_gesig(lt):
    return list(reversed(sorted(lt, key = lambda x: float(x[1]))))
def run_single_glee(params):
    score_list, unis, name, weight, nPermuts, alpha, = params
    relevant_unis = [x[0] for x in score_list if x[0] in unis]
    total_set_len = len(unis)
    glee_obj = glee(weight = weight,
                   nPermuts = nPermuts,
                   alpha = alpha,
                   score_list = score_list,
                   set_of_interest = relevant_unis,
                   )
    glee_obj.run()
    set_por = float(glee_obj.set_len) / total_set_len
    if glee_obj.pval == 0.0:
        glee_obj.pval = 1.0/nPermuts
    return (name
            , (1.0 - glee_obj.pval) * np.sign(glee_obj.nes)
            , glee_obj.nes
            , (str(glee_obj.set_len)
               , str(set_por)
               , str(glee_obj.es)
               , str(glee_obj.nes)
               , str(glee_obj.nes_l)
               , str(glee_obj.nes_u)
              )
            , glee_obj.pval
           )

class glee(bootstrap_enrichment):
    def run(self):
        self.get_non_zeros()
        self.calc_es()
        if self.weight and not self.matches_wtd_score_sum:
            self._report_bad_data()
        else:
            self.calc_nes()
    def score_fv(self):
        self.final_vector = list(self.set_matches * self.wtd_scores * self.matches_norm_factor
                                  - self.set_misses * self.misses_norm_factor
                                )
        up_bumps = [x for x in list(self.set_matches * self.wtd_scores * self.matches_norm_factor) if x != 0]
    def get_misses_norm(self):
        self.misses_norm_factor = 1.0/(self.total_len - self.set_len)
    def get_background(self):
        import random
        self.downBump = self.misses_norm_factor * -1.0
        r = random.Random()
        self.bg =[]
        for i in range(self.nPermuts):
            s = r.sample(range(self.total_len), self.set_len)
            self.non_vec_permut(s)
    def non_vec_permut(self, s):
        up_bumps = [
            (x, (self.wtd_scores[x] * self.matches_norm_factor))
            for x in s
            ]
        up_bumps = [x for x in up_bumps if x[1] > 0]
        up_bumps.sort(key=lambda x:x[0])
        self.bg.append(self.find_peak(up_bumps))
    def find_peak(self, up_bumps, peak = 0.0, val = 0.0, done = 0.0):
        while up_bumps:
            up_idx,up_val = up_bumps[0]
            val += (up_idx-done)*self.downBump
            if abs(val) > abs(peak):
                peak = val
            val += up_val
            done = up_idx+1
            del up_bumps[0]
            if abs(val) > abs(peak):
                peak = val
        return peak
    def find_nes(self):
        boot_signs = set([self.boot_med_sign, np.sign(self.bm_u), np.sign(self.bm_l)])
        if len(boot_signs) == 1 and self.es_sign == self.boot_med_sign:
            shift = self.to_avoid_zero
        elif abs(self.es - self.bm_u) > abs(self.es - self.bm_l):
            shift = abs(self.bm_u) * self.es_sign
        else:
            shift = (abs(self.bm_l) + abs(self.to_avoid_zero)) * self.es_sign
        shifted_bmu = self.bm_u + shift
        shifted_bml = self.bm_l + shift
        shifted_boot_med = self.boot_med + shift
        shifted_es = self.es + shift
        self.nes = float(shifted_es / shifted_boot_med) * self.es_sign
        bound1 = float(shifted_es / shifted_bmu) * self.es_sign
        bound2 = float(shifted_es / shifted_bml) * self.es_sign
        self.nes_l = min([bound1,bound2])
        self.nes_u = max([bound1,bound2])
    def find_p(self):
        if self.es_sign == -1.0:
            self.pval = float(sum(i < self.es for i in self.bg)) / self.nPermuts
        else:
            self.pval = float(sum(i > self.es for i in self.bg)) / self.nPermuts

if __name__ == "__main__":
    # force unbuffered output
    sys.stdout = sys.stderr

    import argparse, time
    parser = argparse.ArgumentParser(description='run GLEE')
    parser.add_argument("infile", help="Pickled list of tuples: Uniprot, score")
    parser.add_argument("glfile", help="Gene List file")
    parser.add_argument("outdir", help="directory to put output in")
    parser.add_argument("tmp_pubdir", help="directory for plots")
    parser.add_argument("wt", type=float, help="weight")
    parser.add_argument("nPermuts", type=int, help="Number of background permutations to do")
    parser.add_argument("w", help="workspace ID")
    parser.add_argument("alpha", type=float, help="alpha for CIs")
    parser.add_argument("cores", type=int, help="Number of cores alloted")
    parser.add_argument("--multiple_input", action="store_true", help="Is the input a dictionary of dictionaries")
    parser.add_argument("--fake-mp", action="store_true", help="Disable multiprocessing (may aid debugging)")

    args=parser.parse_args()

    ts = time.time()
    run = glee_runner(
              outdir = args.outdir,
              infile = args.infile,
              glfile = args.glfile,
              tmp_pubdir = args.tmp_pubdir,
              cores = args.cores,
              ws = args.w,
              alpha = args.alpha,
              nPermuts = args.nPermuts,
              wt = args.wt,
              multiple_input = args.multiple_input,
              fake_mp = args.fake_mp,
            )
    run.run()
    print("Took: " + str(time.time() - ts) + ' seconds')
