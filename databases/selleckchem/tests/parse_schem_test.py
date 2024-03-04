# coding: utf8


from __future__ import division
from builtins import zip
from builtins import str
import os

def test_all(tmp_path):
    tmpfile = str(tmp_path/'out.tsv')

    base = os.path.dirname(__file__)
    path1 = os.path.join(base,'schem_test_1.xlsx')
    path2 = os.path.join(base,'schem_test_2.xlsx')

    from selleckchem import parse_schem
    out = parse_schem.run([path1, path2], tmpfile)

    with open(tmpfile, 'r') as f:
        actual = [x.strip().split('\t') for x in f.readlines()]

    expected_file = os.path.join(base, 'expected.tsv')
    with open(expected_file, 'r') as f:
        expected = [x.strip().split('\t') for x in f.readlines()]

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a == e
    assert len(expected) > 0

TEST_DATA = [
    ("Basic", ("Basic", None)),
    ("WithSyn (Synonym)", ("WithSyn", "Synonym")),
    ("WithMultSyn (Syn1, Syn2)", ("WithMultSyn", "Syn1, Syn2")),
    ("hexa-(1,2)-flouride", ("hexa-(1,2)-flouride", None)),
    ("hexa-(1,2)-flouride (Syn)", ("hexa-(1,2)-flouride", "Syn"))
    ]

import pytest
@pytest.mark.parametrize("data", TEST_DATA)
def test_name_parse(data):
    from selleckchem.parse_schem import parse_name
    assert parse_name(data[0]) == data[1]
