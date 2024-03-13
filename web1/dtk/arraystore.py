
"""
File format for storing multiple numpy arrays, along with optional metadata, in a single compressed file.

Some advantages over normal .npz files include:
- Can separately load individual arrays without reading entire file
- Can be read much faster than a full gzipped file, while having similar compression
- Can store json-encoded metadata alongside each array

The format also has some advantages around parallel processing that we aren't taking advantage of currently.
(Data is stored in chunks, so you can read chunks of the array without loading or decompressing the full thing).
"""


def put_array(fn, name, arr, metadata=None, dtype=None, compression_parms=None, quantize_digits=None):
    """
    Stores this array in the file with given name and metadata.
        arr: A numpy array or a csr_matrix (scipy.sparse) to store
        name: Identifier for this array.
        metadata: JSON-encodable metadata to store alongside this array.
        dtype: if specified, encode to this type (e.g. np.float32).
        quantize_digits: if specified, quantize floats to N digits, for better compression.
    """
    import zarr
    from numcodecs import Blosc, Quantize
    from contextlib import closing
    from scipy.sparse import csr_matrix
    import numpy as np
    # Blosc is a compression library designed for reading compressed data _faster_ than uncompressed.
    # (https://www.blosc.org/pages/blosc-in-depth/)
    # Probably not actually achieving that without a ton of cores, but certainly much faster than reading gzipped data.
    compression_parms = {'cname': 'zstd', 'clevel': 2} if compression_parms is None else compression_parms
    compressor = Blosc(**compression_parms)
    data_kwargs = {}
    if dtype is not None:
        data_kwargs = {'dtype': dtype}

    if quantize_digits is not None: 
        quantize = Quantize(quantize_digits, dtype=dtype or np.float32)
        data_kwargs['filters'] = [quantize]

    # Zarr supports a number of different backends for storing its underlying array chunks.
    # The default is to store them as multiple files in a directory, but we lose the convenience
    # of having a single file to pass around.
    # Use the sqlite backend, which stores the chunks as blobs in a sqlite file instead.
    with closing(zarr.SQLiteStore(fn)) as store:
        root = zarr.group(store=store)
        if isinstance(arr, csr_matrix):
            group = root.create_group(name, overwrite=True)
            group.array('sparse_data', arr.data, compressor=compressor, **data_kwargs)
            group.array('sparse_indices', arr.indices, compressor=compressor)
            group.array('sparse_indptr', arr.indptr, compressor=compressor)
            group.array('sparse_shape', arr.shape, compressor=compressor)
            if metadata:
                group.attrs['meta'] = metadata
        elif isinstance(arr, (np.ndarray, list, set, tuple)):
            stored_array = root.array(name, arr,  overwrite=True, **data_kwargs)
            if metadata:
                stored_array.attrs['meta'] = metadata
        else:
            raise Exception(f"Unexpected array type {type(arr)}")

def get_array(fn, name):
    """
    Retrieves the array with this name from the file.
    Will be either a numpy array or csr_matrix, depending on what was stored.

    Returns (array, metadata), where metadata can be None if none was specified.
    """
    import zarr
    from contextlib import closing
    from scipy.sparse import csr_matrix
    with closing(zarr.SQLiteStore(fn)) as store:
        root = zarr.group(store=store)
        data = root[name]
        if isinstance(data, zarr.Array):
            import numpy as np
            return np.array(data), data.attrs.get('meta', None)
        else:
            mat = csr_matrix(
                (data['sparse_data'], data['sparse_indices'], data['sparse_indptr']),
                shape=data['sparse_shape'],
                )
            return mat, data.attrs.get('meta', None)

def list_arrays(fn):
    """Lists the arrays available for reading from this file."""
    import zarr
    from contextlib import closing
    from scipy.sparse import csr_matrix
    with closing(zarr.SQLiteStore(fn)) as store:
        root = zarr.group(store=store)
        return list(root.keys())