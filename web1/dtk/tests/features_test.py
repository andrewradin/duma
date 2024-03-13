

import pytest

def test_sparse_matrix_wrapper():
    from dtk.features import SparseMatrixWrapper, SparseMatrixBuilder
    build = SparseMatrixBuilder()
    build.add('r1', 'c1', 1)
    build.add('r2', 'c2', 2)
    build.add('r3', 'c2', 3)
    build.add('r1', 'c3', 4)
    import numpy as np

    sp = build.to_wrapper(np.float32)

    assert sp['r1', 'c1'] == 1
    assert sp['r2', 'c2'] == 2
    assert sp['r3', 'c2'] == 3
    assert sp['r1', 'c3'] == 4
    assert sp['r3', 'c1'] == 0

    assert sp['r1'] == {'c1': 1, 'c3': 4}

