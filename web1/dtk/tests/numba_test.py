import pytest
import numpy as np
from numpy.testing import assert_array_equal


def test_unique_bool_rows():
    P0 = [0, 0, 0]
    P1 = [1, 0, 0]
    P2 = [0, 1, 0]
    P3 = [0, 0, 1]
    P4 = [1, 0, 1]
    P5 = [1, 1, 1]

    mat = np.array([
        P0, # 0
        P1, # 1
        P0, # 2
        P5, # 3
        P3, # 4
        P2, # 5
        P5, # 6
        P1, # 7
        P1, # 8
        ], dtype=np.bool)
    
    from dtk.numba import unique_bool_rows

    vals, counts, groups = unique_bool_rows(mat, return_counts=True, return_idx_groups=True)

    assert_array_equal(vals, [P0, P1, P5, P3, P2])
    expected_groups = [ 
        [0, 2],
        [1, 7, 8],
        [3, 6],
        [4],
        [5],
    ]
    for actual, expected in zip(groups, expected_groups):
        assert list(actual) == expected
        
    
    assert list(counts) == [2, 3, 2, 1, 1]
