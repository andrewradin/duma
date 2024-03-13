import pytest

from dtk.arraystore import put_array, get_array, list_arrays
import numpy as np


def test_nparray(tmp_path):
    arr = np.array([1.1, 2.2, 0.9])

    fn = str(tmp_path/'mat.zarr')

    put_array(fn, 'arr1', arr)

    assert np.allclose(get_array(fn, 'arr1')[0],  arr)


def test_csrmat(tmp_path):
    from dtk.features import SparseMatrixBuilder

    sm = SparseMatrixBuilder()
    sm.add(0, 0, 0.1)
    sm.add(1, 1, 0.2)
    sm.add(2, 1, 0.3)


    fn = str(tmp_path/'mat.zarr')

    put_array(fn, 'sparse', sm.get_matrix(np.float32))
    loaded, meta = get_array(fn, 'sparse')

    assert np.allclose(loaded.todense(), sm.get_matrix(np.float32).todense())



def test_rounding(tmp_path):
    arr = np.array([0.111111, 0.222222, 0.333333])

    fn = str(tmp_path/'mat.zarr')

    put_array(fn, 'arr1', arr, quantize_digits=1)
    put_array(fn, 'arr2', arr, quantize_digits=3)


    arr1, _ = get_array(fn, 'arr1')
    assert list(arr) == pytest.approx(list(arr1), abs=0.1)
    assert list(arr) != pytest.approx(list(arr1), abs=0.001)

    arr2, _ = get_array(fn, 'arr2')
    assert list(arr) == pytest.approx(list(arr2), abs=0.1)
    assert list(arr) == pytest.approx(list(arr2), abs=0.001)


    bigger_arr = np.array([111.111, 2.22222, 0.00333333])
    put_array(fn, 'arr3', bigger_arr, quantize_digits=2)
    arr3, _ = get_array(fn, 'arr3')

    # With 2 digits of accuracy, should be within 10% of actual value, but will have
    # a large absolute error (for the 1st number, at least).
    assert list(arr3) != pytest.approx(list(bigger_arr), abs=0.1)
    assert list(arr3) == pytest.approx(list(bigger_arr), rel=0.1)


def test_metadata(tmp_path):
    arr = np.array([1.1, 2.2, 0.9])
    meta = {
        'simple': 2.0,
        'complex': {
            'nested': [1, 5, 'hi'],
        }
    }

    fn = str(tmp_path/'mat.zarr')

    put_array(fn, 'arr1', arr, metadata=meta)

    out_arr, out_meta = get_array(fn, 'arr1')

    assert out_meta == meta
    assert list(arr) == list(out_arr)