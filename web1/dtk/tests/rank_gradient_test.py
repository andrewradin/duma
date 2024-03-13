
def test_kt_ranks():
    from dtk.rank_gradient import kt_ranks 
    import numpy as np
    scores = np.array(sorted(range(300), reverse=True), dtype=float)
    kt_idxs = np.array([10, 100, 150])
    ranks = kt_ranks(scores, kt_idxs, 1e-9)
    print("Check", ranks)

def test_sigma_of_rank():
    from dtk.rank_gradient import sigma_of_rank
    import numpy as np
    scores = np.array(sorted(range(300), reverse=True), dtype=float)
    kt_idxs = np.array([99]) # 99'th index is rank 100
    out = sigma_of_rank(scores, kt_idxs, len(kt_idxs), 0.1, 5, 100)
    print("Check", out)

# TODO: Actually test for values, but this seems pretty sane
# TODO: Compare against EMINput for the non-tied case
# There are known differences with ties, but it should be close.
