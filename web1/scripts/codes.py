#!/usr/bin/env python3

import sys
from algorithms.exit_codes import ExitCoder

import os,pickle

verbose = False
class codes:
    def __init__(self,**kwargs):
        self.ws = kwargs.get('ws',None)
        self.infile = kwargs.get('infile',None)
        self.dpi_t = float(kwargs.get('dpi_t',None))
        self.dpi_file = kwargs.get('dpi_file',None)
# put in a check for the file and fetch from S3 if it's not there
        self.dpi_map_pickle = kwargs.get('dpi_map',None)
        with open(self.dpi_map_pickle, 'rb') as handle:
            self.dpi_map = pickle.load(handle)
        self.bad_keys = []
        self.current_id = None
        self.current_score_dict = {}
        self.output = []
        self.dir_cor_list = []
        self.outdir = kwargs.get('outdir',None)
        self.tmp_pubdir = kwargs.get('tmp_pubdir',None)
        self.base_drug = kwargs.get('base_drug',None)
        self.bd_dgi = kwargs.get('bd_dgi',False)
        # output files
        self.ofile = os.path.join(self.outdir, "codes.tsv")
        # published output files
        self.scatter = os.path.join(self.tmp_pubdir, "scatter.plotly")
    def run(self):
        from scripts.glee import get_gesig, open_infile
        lt = get_gesig(open_infile(self.infile))
        self.score_dict = {x[0]: float(x[1]) for x in lt}
        self.directional = any(x < 0 for x in self.score_dict.values())
        if not self.directional:
            print("No negative input values -- assuming non-directional")
        if self.base_drug:
            with open(self.base_drug, 'rb') as handle:
                l = pickle.load(handle)
            self.score_dict = combo_score(self.score_dict, l, dgi = self.bd_dgi)
        self.compare_to_sd()
    def flush_current_id(self):
        # analyze data
        score = compare_sig_to_sig(self.current_score_dict,
                                   self.score_dict,
                                   union = False
                                   )
        if not score:
            return
        if not self.directional:
            # drop negDir
            score = score[1:]
        self.output += self.get_lines(score)
        self.dpi_drugs_seen.append(self.current_id)
    def compare_to_sd(self):
        from dtk.files import get_file_records
        self.dpi_drugs_seen = []
        from dtk.prot_map import DpiMapping
        import os
        dpi_name = DpiMapping.label_from_filename(os.path.basename(self.dpi_file))
        dpi_map = DpiMapping(dpi_name)
        codetype = dpi_map.mapping_type()

        with open(self.ofile, 'w') as o:
            header = ['negDir', 'codesMax', "posCor"]
            if not self.directional:
                header = header[1:]
            header = [codetype] + header
            o.write("\t".join(header) + "\n")
            for c in get_file_records(dpi_map.get_path(), keep_header = False):
                if float(c[2]) < self.dpi_t:
                    continue
                if c[0] != self.current_id:
                    self.flush_current_id()
                    # now reset everything
                    self.current_score_dict = {}
                    self.current_id = c[0]
                    try:
                        self.current_mapped = self.dpi_map[self.current_id]
                    except KeyError:
                        self.bad_keys.append(self.current_id)
                        self.current_mapped = []
                        continue
    ### We only do this once b/c the data would be the same for each.
    ### Instead when reporting we just report the saem results for each d
                for d in self.current_mapped:
                    self.current_score_dict[c[1]] = get_PI_directional_evidence(c)
                    break
            self.flush_current_id()
            if not self.output:
                raise RuntimeError('No matching drugs found')
            o.write("\n".join(list(set(self.output))) + "\n")
        if len(self.dpi_drugs_seen) != len(set(self.dpi_drugs_seen)):
            print("ERROR: the dpi file may not be sorted by drug ID. There were "
                    + str(len(self.dpi_drugs_seen) - len(set(self.dpi_drugs_seen)))
                    + " repeats observed."
                 )
#            sys.exit(ExitCoder.encode('unexpectedDataFormat'))
    def get_lines(self, score):
        try:
            return ["\t".join([str(i) for i in [d] + score])
                    for d in self.current_mapped
                    if score and d
                   ]
        except AttributeError:
            return []

def combo_score(sd, l, dgi = False):
    all_scores = [sd[k] for k in sd.keys()]
    max_val = max([max(all_scores), abs(min(all_scores))])
    counter = 0
    for tup in l:
        key = tup[0]
        evid = tup[1]
        direc = tup[2]
        try:
            sd[key] = update_sd(sd[key], evid, int(direc), dgi = dgi, max = max_val)
            counter += 1
        except KeyError:
            pass
    print("Updated " + str(counter) + " proteins in the signature.")
    return sd

def update_sd(val, evid, direc, dgi = False, max = 1.0):
    import numpy as np
    signs = list(np.sign([val, direc]))
    if dgi:
# at some point we may want to do something more sophisticated than just adding the values, but for now...
        new_val = val + float(evid) * int(direc)
        if abs(new_val) > max:
            return max * signs[0]
        return new_val
    else:
        assert val != 0.0
        if sum(signs) == 0:
            return 0.0 # the drug counteracts the direction of the misexpression so set confidence to 0
        elif abs(sum(signs)) == 2:
            return max * signs[0] # the drug exacerbates the direction of the misexpression so set confidence to max
        else: # the drug has an unknown direction... not sure what's best here
            return val # currently we're defaulting to not moving the value since we don't know what to do with it

def compare_sig_to_sig(sd1, sd2, union = True, skip_cor = False):
    if len(list(sd2.keys())) == 0 or len(list(sd1.keys())) == 0:
        return False
    if union:
        all_prots = list(set([x for x in list(sd2.keys()) + list(sd1.keys())]))
    else:
        all_prots = list(sd1.keys())
    ordered_1 = []
    ordered_2 = []
    for p in all_prots:
        ordered_1.append(sd1.get(p, 0.0))
        ordered_2.append(sd2.get(p, 0.0))
    scores = sig_evid(ordered_1, ordered_2, all_prots, union = union)
    if not skip_cor:
        scores += sig_cor(ordered_1, ordered_2, all_prots, union = union)
    return scores

def sig_evid(ordered_1, ordered_2, all_prots, union = True):
    import numpy as np
### We're taking the product of each protein. We could also take the average.
    combined = np.multiply(np.array(ordered_1), np.array(ordered_2))
    absv = np.abs(combined)
    return [float(np.sum(np.sign(combined))),
            absv.max()
           ]

def sig_cor(ordered_1, ordered_2, all_prots, union = True):
    from scipy.stats.stats import spearmanr
    if len(all_prots) < 3:
        if verbose:
            print("skipping due to too few proteins")
        return [0]
    elif len(set(ordered_2)) == 1 or len(set(ordered_1)) == 1:
        if verbose:
            print("skipping due to too little variance")
        return [0]
    cor = spearmanr(ordered_1, ordered_2)[0]
# this was never happening, so I stopped filtering to speed up
#    if str(cor) == 'nan':
#        return False
    return [cor]

def get_PI_directional_evidence(c):
    if float(c[3]) != 0.0:
        return float(c[2]) * float(c[3])
### The assumption is that drugs tend to inhibit proteins, not activate
    return float(c[2]) * -1.0

if __name__ == "__main__":
    import time

    # force unbuffered output
    sys.stdout = sys.stderr

    import argparse
    parser = argparse.ArgumentParser(description='run CoDES')
    parser.add_argument("infile", help="Pickled list of tuples (Uniprot, score)")
    parser.add_argument("dpi_t", help="Minimum DPI evidence")
    parser.add_argument("dpi_file", help="Drug Gene Interaction file")
    parser.add_argument("dpi_map", help="Drug Gene Interaction map")
    parser.add_argument("outdir", help="directory to put output in")
    parser.add_argument("tmp_pubdir", help="directory for plots")
    parser.add_argument("w", help="workspace ID")
    parser.add_argument("--base_drug", default = False, help="OPTIONAL: What drug should be used as a base drug.")
    parser.add_argument("--bd_dgi", action='store_true', help="OPTIONAL: The base drug partner info is Drug Gene interaction data (e.g. from GESig).")

    args = parser.parse_args()
    if not args.bd_dgi:
        args.bd_dgi = False
    ts = time.time()
    run = codes(
              outdir = args.outdir,
              infile = args.infile,
              dpi_map = args.dpi_map,
              dpi_file = args.dpi_file,
              dpi_t = args.dpi_t,
              tmp_pubdir = args.tmp_pubdir,
#              cores = args.cores,
              ws = args.w,
              base_drug = args.base_drug,
              bd_dgi = args.bd_dgi,
            )
    run.run()
    print("Took: " + str(time.time() - ts) + ' seconds')
