#!/usr/bin/env python3

import logging
logger = logging.getLogger(__name__)

def compare_sig_to_sig(sd1, sd2, all_prots=None):
    if all_prots is None:
        all_prots = list(set([x for x in list(sd2.keys()) + list(sd1.keys())]))

    ordered_1 = [sd1.get(p, 0.0) for p in all_prots]
    ordered_2 = [sd2.get(p, 0.0) for p in all_prots]

    scores = sig_evid(ordered_1, ordered_2)
    scores += sig_cor(ordered_1, ordered_2)
    return scores

def sig_evid(ordered_1, ordered_2):
    import numpy as np
    combined = np.multiply(np.array(ordered_1), np.array(ordered_2))
    absv = np.abs(combined)
    return [float(np.sum(np.sign(combined))), absv.max()]

def sig_cor(ordered_1, ordered_2):
    from scipy.stats.stats import spearmanr
    if len(ordered_1) < 3:
        return [0]
    elif len(set(ordered_2)) == 1 or len(set(ordered_1)) == 1:
        return [0]
    cor = spearmanr(ordered_1, ordered_2)[0]
    return [cor]

def load_sig(infile):
    import pickle
    import gzip
    with gzip.open(infile, 'rb') as f:
        return dict(pickle.load(f))

def compare_sig(mol_sig, dis_sig, molsig_meta):
    sig_dict = {} 
    geneids = molsig_meta['genes']['order']
    assert len(geneids) == len(mol_sig), f'Had {len(geneids)} genes, mol_sig looks like {mol_sig.shape}'
    for geneid, val in zip(geneids, mol_sig):
        uni = molsig_meta['genes']['info']['uniprot'][geneid]
        if uni:
            sig_dict[uni] = val
    
    return compare_sig_to_sig(sig_dict, dis_sig, list(sig_dict.keys()))

def write_scores(outdir, id2scores):
    import os
    fn = os.path.join(outdir, 'scores.tsv')
    with open(fn, 'w') as f:
        header = ['id', 'negDir', 'sigEv', 'sigCorr']
        f.write('\t'.join(header) + '\n')
        for id, scores in id2scores.items():
            f.write('\t'.join([id] + [str(x) for x in scores]) + '\n')


def run(infile, outdir, lincs, **kwargs):
    sig = load_sig(infile)

    from algorithms.run_molgsig import load_mol_sigs
    mol_sigs, molsig_meta = load_mol_sigs(lincs)

    from dtk.parallel import pmap
    static_args = {
        'dis_sig': sig,
        'molsig_meta': molsig_meta,
    }
    logger.info("Doing signature comparison")

    scores = list(pmap(compare_sig, mol_sigs, static_args=static_args, progress=True))

    drug_ids = molsig_meta['drugs']['order']

    assert len(drug_ids) == len(scores)

    id2scores = dict(zip(drug_ids, scores))

    logger.info("Writing out scores")
    write_scores(outdir, id2scores)


if __name__ == "__main__":
    import time
    import argparse
    from dtk.log_setup import setupLogging, addLoggingArgs
    parser = argparse.ArgumentParser(description='run MolGSig')
    parser.add_argument('-i', "--infile", help="Pickled input signature (prot, ev)")
    parser.add_argument('-o', "--outdir", help="Where to write output")
    parser.add_argument('-l', "--lincs", help="Lincs version/choice")
    addLoggingArgs(parser)
    args = parser.parse_args()
    setupLogging(args)

    run(**vars(args))