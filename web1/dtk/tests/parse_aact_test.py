def test_aact_parse():
    testdata=[
            ['a|b|c',['a','b','c']],
            ['"a"|"b"|c',['a','b','c']],
            ['a|"b"|c',['a','b','c']],
            ['a|"b|d"|c',['a','b d','c']],
            ['a|"b',None],
            ['d"|c',['a','b d','c']],
            ['a|"b',None],
            ['x|y|z',None],
            ['d"|c',['a','b x y z d','c']],
    ]
    from dtk.parse_aact import yield_assembled_recs
    out=list(yield_assembled_recs(iter(x[0] for x in testdata)))
    exp = [x[1] for x in testdata if x[1]]
    assert out == exp

