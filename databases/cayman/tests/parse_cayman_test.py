# coding: utf8


from __future__ import division
from builtins import zip
from builtins import str
import os

def test_all(tmp_path):
    tmpfile = str(tmp_path/'out.tsv')

    base = os.path.dirname(__file__)
    path = os.path.join(base,'cayman_sample.sdf')

    import parse_cayman
    out = parse_cayman.run(path, tmpfile)

    with open(tmpfile, 'r') as f:
        actual = [x.strip().split('\t') for x in f.readlines()]

    expected_file = os.path.join(base, 'expected.tsv')
    with open(expected_file, 'r') as f:
        expected = [x.strip().split('\t') for x in f.readlines()]


    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a == e

