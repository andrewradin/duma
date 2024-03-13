from __future__ import print_function

from path_helper import PathHelper
import pytest
import six
try:
    import similarity_csv
except ImportError:
    import sys
    sys.path.insert(1, PathHelper.website_root+"/../moleculeSimilarity/fingerprint")


from similarity_csv import get_rdkit_fingerprint, get_bit_smiles_bit_fp
def test_smiles_small():
    fp = get_bit_smiles_bit_fp("COc", 2)
    assert sum(fp.GetNonzeroElements().values()) == 1


def test_smiles_loop():
    fp = get_bit_smiles_bit_fp("C1(C)(n)C(C)C1", 2)
    assert sum(fp.GetNonzeroElements().values()) == 1

def test_smiles_tiny():
    t = {
        'mol': ['COO', 'mol']
    }
    x = 'mol'
    id, fp, allBits, bitFp = get_rdkit_fingerprint([t, x, 2, True, True])
    assert fp.GetNonzeroElements() == {}, "No radius2 fragments"

def test_bit_smiles_simple():
    t = {
        'mol': ['NCCOCCN', 'mol']
    }
    x = 'mol'
    id, fp, allBits, bitFp = get_rdkit_fingerprint([t, x, 2, True, True])
    print(("Hey ", fp.GetNonzeroElements()))
    print((allBits, bitFp))

    expected_bits = {
            'C(CN)OC': 2,
            'C(OC)CN': 2, # Same as prev, diff branch order
            'O(CC)CC': 1,
            'C(N)CO': 2,
            'C(CO)N': 2, # Same as prev, diff branch order
            }

    for bit_sm, expected_val in six.iteritems(expected_bits):
        bit = get_bit_smiles_bit_fp(bit_sm, 2)
        bit = list(bit.GetNonzeroElements().keys())[0]
        print("%s becomes %s" % (bit_sm, bit))
        assert fp[bit] == expected_val, "Should find each bit in it"

def test_bit_smiles_complex():
    smiles = "CC[C@H](C)C(NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(=O)O)NC(=O)CNC(=O)[C@H](CC(N)=O)NC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H]1CCCN1C(=O)[C@H](N)Cc1ccccc1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CC(C)C)C(=O)O"


    t = {
        'mol': [smiles, 'mol']
    }
    x = 'mol'
    id, fp, allBits, bitFp = get_rdkit_fingerprint([t, x, 2, True, True])
    print((allBits, bitFp))

    from collections import Counter
    expected_bits2 = Counter(allBits)
    expected_bits = bitFp

    for bit_sm, expected_val in six.iteritems(expected_bits):
        bit = get_bit_smiles_bit_fp(bit_sm, 2)
        bit = list(bit.GetNonzeroElements().keys())[0]
        print("%s becomes %s" % (bit_sm, bit))
        assert fp[bit] > 0, "Should find each bit in it"
        print("%s: %d vs %d" % (bit_sm, fp[bit], expected_val))


