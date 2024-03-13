from numba import njit, jit
from numba.core import types
from numba.typed import Dict, List

import numpy as np

def sparse_unique_bool_rows(sparse_mat, unique_idxs, *args, **kwargs):
    """Used for stratifying a sparse matrix based on boolean columns.
    
    Similar to np.unique, but much faster (60s for np vs 0.5s for this)
    
    np.unique relies on sorting instead of hashing and probably the per-col
    boolean comparisons also hurt.
    """
    assert sparse_mat.dtype == np.bool
    arr = sparse_mat[:, unique_idxs].toarray()
    return unique_bool_rows(arr, *args, **kwargs)

def unique_bool_rows(arrmat, return_counts=False, return_idx_groups=False):
    assert arrmat.dtype == np.bool
    assert arrmat.shape[1] < 64, "Too many columns, won't encode into an int64"

    vals, counts, row_val_idxs = nb_unique_bool_rows(arrmat)

    out = [np.array(vals)]

    if return_counts:
        out.append(counts)

    if return_idx_groups:
        idxs = [[] for _ in range(len(vals))]
        for row_idx, val_idx in enumerate(row_val_idxs):
            idxs[val_idx].append(row_idx)
        
        out.append(idxs)
    
    
    return out


@njit
def bools_to_int64(bools):
    """If 64 becomes a limiting factor, could do some sort of bitflag/vector class."""
    assert len(bools) < 64
    cur = np.int64(0)
    for val in bools:
        cur <<= 1
        cur |= val
    return cur

@njit
def nb_unique_bool_rows(arrmat):
    """
    Perf note: Originally this function constructed the index groups and
    returned them, but it turned out to be faster to do that in raw python
    from the idxs array than to unpack and repack the numba list of lists.
    """
    hasher = Dict.empty(
        key_type=types.int64,
        value_type=types.int64,
    )
    vals = List()
    idxs = List.empty_list(types.int64)
    counts = List.empty_list(types.int64)

    for row_idx, row in enumerate(arrmat):
        assert len(row) < 64
        rowdata = bools_to_int64(row)
        if rowdata not in hasher:
            new_idx = len(hasher)
            hasher[rowdata] = new_idx
            vals.append(row)
            counts.append(0)

        entry = hasher[rowdata]
        idxs.append(entry)
        counts[entry] += 1
        
    return vals, np.asarray(counts), np.asarray(idxs)


@njit
def max_sum_row_inner(indptr, indices, data, tissue_mat, drug_idx, t_weight, out):
    """See bulk_pathsum.py"""
    col_s = indptr[drug_idx]
    col_e = indptr[drug_idx+1]
    if col_s  != col_e:
        col_idxs = indices[col_s:col_e]
        tissue_datas = tissue_mat[col_idxs]
        # Mask out any coldata where the tissue value is 0.
        # NOTE: Some tissue values are 0-1 (cc_path), some are 0 to anything (capp).
        # NOTE: Numba doesn't support bool or np.bool (which is apparently just an alias to builtin bool),
        # need to use the numpy-specific np.bool_ (https://github.com/numba/numba/issues/1311)
        col_datas = data[col_s:col_e] * tissue_datas.astype(np.bool_)
        val = np.max(col_datas + tissue_mat[col_idxs] * t_weight)
        out[drug_idx] += val

def accum_max_sum_row(drug_mat, tissue_mat, drug_idx, t_weight, out):
    return max_sum_row_inner(drug_mat.indptr, drug_mat.indices, drug_mat.data, tissue_mat, drug_idx, t_weight, out)