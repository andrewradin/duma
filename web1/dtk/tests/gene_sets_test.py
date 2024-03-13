
import pytest

def test_make_dedupe_map():
    from dtk.gene_sets import make_dedupe_map, ROOT_ID
    protsets = {
        'PS1': {'P1', 'P2'},
        'PS2': {'P1', 'P2'},
        'PS3': {'P1'},
        'PS4': {'P2'},
        'PS5': {'P3'},
        'PS6': {'P2'},
        'PS7': {'P2'},
        ROOT_ID: set(),
    }

    hierarchy = {
        ROOT_ID: ['PS1', 'PS2'],
        'PS1': ['PS3', 'PS4', 'PS5', 'PS6'],
        'PS2': ['PS7'],
    }


    out = make_dedupe_map(protsets, hierarchy, siblings_only=True)
    expected = {x:x for x in protsets.keys()}
    expected['PS2'] = 'PS1'
    expected['PS6'] = 'PS4'
    assert out == expected

    out = make_dedupe_map(protsets, hierarchy, siblings_only=False)
    expected = {x:x for x in protsets.keys()}
    expected['PS2'] = 'PS1'
    expected['PS6'] = 'PS4'
    expected['PS7'] = 'PS4'
    assert out == expected
