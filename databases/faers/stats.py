#!/usr/bin/env python3

def make_datas(drug_fn, indi_col_fn, drug_col_fn):
    from scipy.sparse import load_npz

    drug_fm = load_npz(drug_fn)
    indis = set([x.strip() for x in open(indi_col_fn).readlines()])
    drugs = set([x.strip() for x in open(drug_col_fn).readlines()])
    num_samples = drug_fm.shape[0]


    return dict(indis=indis, drugs=drugs, num_samples=num_samples)

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=['faers.v{version}.drug_mat.npz', 'faers.v{version}.indi_cols.txt', 'faers.v{version}.drug_cols.txt'])