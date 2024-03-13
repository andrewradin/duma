
import numpy as np

def fisher_exact(table, alternative='two-sided'):
    """Implements the scipy interface to fisher_exact but faster.

    Just wraps the Cython-implemented fisher package interface.

    NOTE: Actually just forwards to scipy.stats for now.
    The problem seems to be that the fisher package uses a large episilon
    of 1e-6 in their calculations, which ruins the precision below 1e-6.

    See https://github.com/brentp/fishers_exact_test/issues/27
    """

    if True:
        import scipy.stats as ss
        return ss.fisher_exact(table, alternative)

    from fisher import pvalue

    # Different naming conventions.
    alt_map = {
        'two-sided': 'two_tail',
        'less': 'left_tail',
        'greater': 'right_tail'
    }


    # This from the start of https://github.com/scipy/scipy/blob/master/scipy/stats/stats.py
    # fisher.  pvalue doesn't compute an oddsratio, so we do that explicitly here.
    c = np.asarray(table, dtype=np.int64)
    assert c.shape == (2, 2), "2x2 contingency table expected"

    if 0 in c.sum(axis=0) or 0 in c.sum(axis=1):
        oddsratio = np.nan
    elif c[1, 0] > 0 and c[0, 1] > 0:
        oddsratio = c[0, 0] * c[1, 1] / (c[1, 0] * c[0, 1])
    else:
        oddsratio = np.inf
    
    pv = pvalue(c[0,0], c[0, 1], c[1, 0], c[1, 1])
    out =  oddsratio, getattr(pv, alt_map[alternative])

    return out