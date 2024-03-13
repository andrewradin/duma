#!/usr/bin/env python3

import sys
from path_helper import PathHelper

import os

from algorithms.exit_codes import ExitCoder
from dea import bootstrap_enrichment

from collections import defaultdict
from dtk.enrichment import mult_hyp_correct
import numpy as np

GLF_METHODS = [
        'wOR',
        'wFEBE',
]

class glf_runner:
    def __init__(self,**kwargs):
        self.multiple_input = kwargs.get('multiple_input',None)
        self.fake_mp = kwargs.get('fake_mp',None)
        self.cores = kwargs.get('cores',None)
        self.ws = kwargs.get('ws',None)
        self.infile = kwargs.get('infile',None)
        self.gl_file = kwargs.get('glfile',None)
        self.outdir = kwargs.get('outdir',None)
        self.tmp_pubdir = kwargs.get('tmp_pubdir',None)
        self.method = kwargs.get('method')
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
                self.ofile = os.path.join(self.outdir, self.current_tis + "_glf.tsv")
                self.lt = v
                self.run_single()
            self.report_combined()
        else:
            self.lt = input
            self.ofile = os.path.join(self.outdir, "glf.tsv")
            self.run_single()

## TODO
# XXX This was copied from the old GLEE code which supported running this CM on individual signatures
# XXX e.g. gene expression data sets or GWDSs
# XXX We haven't done that for some time, so I didn't take the time to update this for GLF
    def report_combined(self):
        import operator
        # combine all of the ofiles into one consensus
        self.combined_outfile = os.path.join(self.outdir, "glf.tsv")
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

        from dtk.parallel import pmap
        results = list(pmap(
                run_single_glf,
                self.relevant_gl_data.items(),
                static_args={
                    'score_list': self.score_list,
                    'method': self.method,
                    },
                fake_mp=self.fake_mp,
                num_cores=self.cores,
                ))
        # now add the missing gene lists that had no overlap and thus have the default values
        results += [(x, 0,0.,1.,0,0,0) for x in self.skipped_gl]
        # sort by wFEBE
        results.sort(key=operator.itemgetter(6), reverse=True)
        self.gles=[]
        for x in results:
            self.gles.append("\t".join([str(y) for y in x]))
    def _load_gl(self):
        from dtk.files import get_file_records
        if not os.path.isfile(self.gl_file):
            from dtk.s3_cache import S3Bucket, S3File
            f=S3File(S3Bucket('pathways'),os.path.basename(self.gl_file))
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
            f.write("\t".join(['uniprotset', 'nProts', 'setPor', 'febeQ', 'febeOR', 'peakInd', 'wFEBE']) + "\n")
            f.write("\n".join(self.gles) + "\n")
def open_infile(infile):
    import pickle
    with open(infile, 'rb') as handle:
        lt = pickle.load(handle)
    return lt
def get_gesig(lt):
    return list(reversed(sorted(lt, key = lambda x: float(x[1]))))
def run_single_glf(pathway_item, score_list, method):
    name, unis = pathway_item
    from scripts.febe import glf, wfebe
    relevant_unis = [x[0] for x in score_list if x[0] in unis]
    total_set_len = len(unis)
    MethodType = glf if method == 'wOR' else wfebe
    runner = MethodType(named_scores = score_list,
                        names_of_interest = unis
                         )
    runner.run()
    return (name,
            total_set_len,
            float(len(relevant_unis))/total_set_len,
            runner.best_q,
            runner.best_or,
            runner.peak_ind,
            runner.final_score,
          )

if __name__ == "__main__":
    # force unbuffered output
    sys.stdout = sys.stderr

    import argparse, time
    parser = argparse.ArgumentParser(description='run GLF')
    parser.add_argument("infile", help="Pickled list of tuples: Uniprot, score")
    parser.add_argument("glfile", help="Gene List file")
    parser.add_argument("outdir", help="directory to put output in")
    parser.add_argument("tmp_pubdir", help="directory for plots")
    parser.add_argument("w", help="workspace ID")
    parser.add_argument("cores", type=int, help="Number of cores alloted")
    parser.add_argument("--multiple_input", action="store_true", help="Is the input a dictionary of dictionaries")
    parser.add_argument("--fake-mp", action="store_true", help="Disable multiprocessing (may aid debugging)")
    parser.add_argument('--method', help='Method for scoring pathways from a signature')

    args=parser.parse_args()

    ts = time.time()
    run = glf_runner(
              outdir = args.outdir,
              infile = args.infile,
              glfile = args.glfile,
              tmp_pubdir = args.tmp_pubdir,
              cores = args.cores,
              ws = args.w,
              multiple_input = args.multiple_input,
              fake_mp = args.fake_mp,
              method = args.method,
            )
    run.run()
    print("Took: " + str(time.time() - ts) + ' seconds')
