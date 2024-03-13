import pytest
import numpy as np

def test_fisher():
    from dtk.stats import fisher_exact

    oddsr, pv = fisher_exact([[16, 3454], [34531, 1558379]])

    lpv = np.log10(pv)
    assert lpv > -18 and lpv < -15, "Make sure pvalue is at least within a couple orders of magnitude..."

    """Different implementations give different results.

    fisher.pvalue: 3.5e-6
    scipy.stats:   1.7e-16
    R:             2.2e-16
    > fisher.test(matrix(c(16, 3454, 34531, 1558379), nrow = 2))
    > p-value < 2.2e-16

    fisher.pvalue is way off, due to https://github.com/brentp/fishers_exact_test/issues/27
    """

