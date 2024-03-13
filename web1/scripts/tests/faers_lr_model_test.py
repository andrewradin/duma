import numpy as np
from scripts.faers_lr_model import LR, LRParams
from pytest import approx


X1 = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 1],
    [0, 0, 1],
    [0, 1, 0],
])
y1 = np.array([1, 1, 0, 0, 0])


def test_lr_model_basic():
    X, y = X1, y1    

    parms = LRParams(test_frac=0.5)
    logreg = LR(X, y, parms)
    lr, acc, wtd_acc = logreg.run()
    assert acc == 1.0
    assert wtd_acc == 1.0

def test_lr_model_elastic():
    X, y = X1, y1    

    parms = LRParams(test_frac=0.5, penalty='elasticnet')
    logreg = LR(X, y, parms)
    lr, acc, wtd_acc = logreg.run()
    assert acc == 1.0
    assert wtd_acc == 1.0

def test_lr_model_pvalues():
    X, y = X1, y1    

    parms = LRParams(test_frac=0.5)
    logreg = LR(X, y, parms)
    lr, acc, wtd_acc = logreg.run()
    se, z, p = logreg.compute_pvalues(lr)
    print("P-values:", p)
    assert len(p) == X.shape[1]

def test_lr_model_df():
    X, y = X1, y1    

    parms = LRParams(test_frac=0.5)
    logreg = LR(X, y, parms)
    lr, acc, wtd_acc = logreg.run()

    df = logreg.assemble_df(lr, ['feat1', 'feat2', 'feat3'])


def test_faers_lr(tmp_path):
    # This test takes too long to leave in for now, maybe revisit.
    if True:
        return

    indir = tmp_path / 'in'
    outdir = tmp_path / 'out'
    indir.mkdir()
    outdir.mkdir()
    from scripts.faers_lr_model import FAERS_LR
    flr = FAERS_LR('faers.v1', ['idiopathic pulmonary fibrosis'], indir, outdir,
        C=1.0,
        penalty='l1',
        class_weight=None,
        demo_covariates=['age'],
        method='normal'
    )
    flr.build_matrix()
    flr.fit_and_summarize()

def test_stratify_weights():
    from scipy import sparse
    A = sparse.csr_matrix([
        [1, 0, 1],
        [1, 0, 0],
        [1, 1, 1],
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
    ], dtype=bool)
    X = A[:, :2]
    y = np.reshape(A[:, 2].toarray(), (-1,))

    from scripts.faers_lr_model import stratify_weights

    weights = stratify_weights(X, y, [0, 1])

    # We have 3 strata ([1,0], [1,1], [0,0]) that have 1 case each.
    # Each should get 1/3rd of the weight, divided such that within
    # the strata cases and controls are balanced.

    # Normally this would equate to 6 samples (3 cases, 3 [effective] controls),
    # but one of the stratum is missing a control, so it gets dropped.

    assert sum(weights) == 6 - 1

    expected_weights = [
        1, # [1,0] case
        1, # [1,0] ctrl

        1, # [1,1] case

        0,   # [0,1] ctrl

        0.5,   # [0,0] ctrl
        0.5,   # [0,0] ctrl
        1,   # [0,0] case
    ]

    assert weights == expected_weights


    weights = stratify_weights(X, y, [0])

    # We have 2 strata ([0] and [1]) that have 1 and 2 cases respectively.
    # The [0] stratum should get 1/3rd of the weight, [1] gets 2/3rds.

    assert sum(weights) == approx(6)
    expected_weights = [
        1, # [1] case
        2, # [1] ctrl
        1, # [1] case

        1/3,   # [0] ctrl
        1/3,   # [0] ctrl
        1/3,   # [0] ctrl
        1,   # [0] case
    ]

    assert weights == expected_weights

def test_stratify_weights_missing_ctrl():
    from scipy import sparse
    A = sparse.csr_matrix([
        [1, 0, 1],
        [1, 0, 1],
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 1],
    ], dtype=bool)
    X = A[:, :2]
    y = np.reshape(A[:, 2].toarray(), (-1,))

    from scripts.faers_lr_model import stratify_weights

    weights = stratify_weights(X, y, [0, 1])

    # We have 2 strata ([1,0], [0,0]) that have 2 cases each.

    assert sum(weights) == approx(6)
    expected_weights = [
        1, # [1,0] case
        1, # [1,0] case

        2, # [0,0] ctrl
        1, # [0,0] case 
        1, # [0,0] case 
    ]

    assert weights == expected_weights

def test_stratified_cont_tables():
    from scipy import sparse
    A = sparse.csr_matrix([
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 0],

        [0, 1, 1],
        [0, 1, 1],
        [0, 0, 1],
        [0, 1, 0],
    ], dtype=bool)
    X = A[:, :2]
    y = np.reshape(A[:, 2].toarray(), (-1,))

    from scripts.faers_lr_model import stratified_cont_tables
    conts, _, _, _ = stratified_cont_tables(X, y, [0])

    assert conts.shape == (2, 2, 2, 2)

    # Note that the tables are ordered in appearance of that stratum
    # rather than the sorted behavior we used to get from np.unique.
    assert conts[1, 0].tolist() == [[0, 3], [0, 3]]
    assert conts[0, 0].tolist() == [[2, 0], [2, 0]]

    assert conts[1, 1].tolist() == [[2, 1], [3, 0]]
    assert conts[0, 1].tolist() == [[1, 1], [1, 1]]
