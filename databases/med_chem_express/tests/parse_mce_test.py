# coding: utf8


from __future__ import division
from builtins import zip
from builtins import str
import os


def test_all(tmp_path):
    tmpfile = str(tmp_path/'out.tsv')
    from med_chem_express import parse_mce

    path = os.path.join(os.path.dirname(__file__), 'med_chem_test.xlsx')

    out = parse_mce.run(path, tmpfile)

    with open(tmpfile, 'r') as f:
        actual = [x.strip().split('\t') for x in f.readlines()]

    expected = [
         ['med_chem_express_id', 'attribute', 'value'],
         ['HY-100001', 'canonical', 'SKF-96365 (hydrochloride)'],
         ['HY-100001', 'cas', '130495-35-1'],
         ['HY-100006A', 'canonical', 'MRT68921 (hydrochloride)'],
         ['HY-100006A', 'cas', '2080306-21-2'],
         ['HY-100007', 'canonical', 'Vonoprazan'],
         ['HY-100007', 'cas', '881681-00-1'],
         ['HY-100007', 'synonym', 'TAK-438 (free base)'],
         ['HY-100009', 'canonical', 'Ufenamate'],
         ['HY-100009', 'cas', '67330-25-0'],
         ['HY-100009', 'synonym', 'Flufenamic acid butyl ester'],
         ['HY-100009', 'synonym', 'Butyl flufenamate'],
         ['HY-100012', 'canonical', 'CBR-5884'],
         ['HY-100012', 'cas', '681159-27-3'],
         ['HY-100015', 'canonical', 'Mivebresib'],
         ['HY-100015', 'synonym', 'ABBV-075'],
         ['HY-100016', 'canonical', 'AZD0156'],
         ['HY-100016', 'cas', '1821428-35-6'],
         ['HY-100217', 'canonical', 'D,L-3-Indolylglycine'],
         ['HY-100217', 'cas', '6747-15-5'],
         ['HY-100217', 'synonym', 'α-Amino-1H-indole-3-acetic acid'],
         ['HY-101797', 'canonical', 'Veralipride'],
         ['HY-101797', 'cas', '66644-81-3'],
         ['HY-101797', 'synonym', '(±)-Veralipride'],
         ['HY-101797', 'synonym', 'LIR166'],
         ]
    for a, e in zip(actual, expected):
        assert a == e
