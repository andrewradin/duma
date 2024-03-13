def test_unichem():
    from dtk.unichem import UniChem
    uc = UniChem()
    test_set = set([
                'BDBM91999',
                'BDBM50082132',
                ])
    d = uc.get_converter_dict('bindingdb','zinc','v1',key_subset=test_set)
    assert set(d.keys()) == test_set

    # these zinc ids are in both bindingdb and drugbank
    test_set = set([
                'ZINC000000000506',
                'ZINC000000000640',
                ])
    d = uc.get_converter_dict('zinc','bindingdb','v1',key_subset=test_set)
    assert set(d.keys()) == test_set
    d = uc.get_converter_dict('zinc','drugbank','v1',key_subset=test_set)
    assert set(d.keys()) == test_set
