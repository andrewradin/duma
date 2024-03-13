
import pytest
from dtk.parallel import *
def test_pmap_single():
    def add_one(x):
        return x + 1

    out = list(pmap(add_one, [2, 4, 5, 10]))
    assert out == [3, 5, 6, 11]


def test_pmap():
    data_arr = [10,20,30,40,50,60,70]

    static_args = dict(
            data_arr=data_arr
            )

    def add(a, b, data_arr):
        return data_arr[a] + data_arr[b]

    for progress in [False, True]:
        out = list(pmap(add, [0, 1, 2], [3, 4, 5], static_args=static_args, progress=progress))
        assert out == [50, 70, 90]

    out_fake = list(pmap(add, [0, 1, 2], [3, 4, 5], static_args=static_args, fake_mp=True))
    assert out == out_fake



def test_pmap_exc():
    def throws(x):
        if x == 2:
            raise Exception("Failed at ", x)
        return x
    with pytest.raises(Exception):
        list(pmap(throws, [0, 1, 2], num_cores=30))



def sum_series(x, to_add):
    def add(a, b):
        return a + b

    out = 0
    for val in pmap(add, range(x), static_args={'b': to_add}):
        out += val
    return out

def test_nested_pmap():

    tests = [1,2,3,4]
    to_add = 2
    expected = [2, 5, 9, 14]
    output = list(pmap(sum_series, tests, static_args={'to_add': to_add}))
    assert output == expected

def fibo(x):
    if x <= 1:
        return 1

    return sum(pmap(fibo, [x-1, x-2]))

def test_deep_nested_pmap():
    tests = [1,2,3,4,5,6]
    expected = [1, 2, 3, 5, 8, 13]
    output = list(pmap(fibo, tests))
    assert output == expected







def test_chunker():
    chunks = list(chunker([1,2,3,4,5,6], num_chunks=2))
    assert chunks == [[1,2,3], [4,5,6]]

    chunks = list(chunker([1,2,3,4,5,6], num_chunks=4))
    assert chunks == [[1], [2,3], [4], [5,6]]

    chunks = list(chunker([1,2,3,4,5,6], chunk_size=2))
    assert chunks == [[1,2], [3,4], [5,6]]

    chunks = list(chunker([1,2,3,4,5,6], chunk_size=4))
    # Note that these are chunks of size 3.
    # The chunker will rebalance rather than giving a tiny final chunk.
    assert chunks == [[1,2,3], [4,5,6]]

def test_chunker_auto():
    for list_sz in range(1, 30):
        for x in range(1, 30):
            inp = list(range(list_sz))
            chunks = list(chunker(inp, num_chunks=x))
            assert sum(chunks, []) == inp

            chunks = list(chunker(inp, chunk_size=x))
            assert sum(chunks, []) == inp
