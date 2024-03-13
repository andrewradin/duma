
def test_compare_dict():
    from dtk.text import compare_dict
    before = dict(
        unchanged='value1',
        removed='value2',
        modified='value3',
        )
    assert compare_dict('label1',before,before) == 'label1'
    after = dict(before)
    del after['removed']
    after['added'] = 'value4'
    after['modified'] = 'value5'
    assert compare_dict('label2',before,after) == \
            'label2 + (added:value4, modified:value3>>value5, ~removed)'
    before = dict(
        modified=None,
        )
    after = dict(before)
    after['modified'] = 'value6'
    assert compare_dict('label3',before,after) == \
            'label3 + (modified:None>>value6)'

def test_compare_set():
    from dtk.text import compare_set
    before = set(['keep','remove'])
    after = set(before)
    after.remove('remove')
    after.add('add')
    assert compare_set(before,after) == '+add-remove'

def test_compare_setlist():
    before = [
            ['same1','same2','same3'],
            ['remove1','remove2'],
            ['modify1','modify2','modify3'],
            ]
    after = [
            ['modify1','modify4','modify3'],
            ['add1'],
            ['same1','same2','same3'],
            ['another_add1'],
            ]
    from dtk.text import compare_setlist
    assert compare_setlist(before,after) == \
            ',{modify1,modify3+modify4-modify2}+{add1}+{another_add1}-{remove1,remove2}'

def test_compare_wzs_settings():
    from dtk.text import compare_wzs_settings
    # verify that text-only auto_contraints changes are stripped
    before = dict(auto_constraints='[["a","b"]]')
    after = dict(auto_constraints='[["b","a"]]')
    assert compare_wzs_settings('label',before,after) == 'label'

def test_dict_replace():
    from dtk.text import dict_replace
    to_replace = {
        'alpha': '<a>',
        'beta': '<b>',
        '()': '<e>',
        'a|b|c': '<f>',
        '[abc]': '<d>',
    }

    assert dict_replace(to_replace, 'Nothing to replace') == 'Nothing to replace'
    assert dict_replace(to_replace, 'alpha') == '<a>'
    assert dict_replace(to_replace, 'alphalpha') == '<a>lpha'
    assert dict_replace(to_replace, 'alphaalpha') == '<a><a>'
    assert dict_replace(to_replace, '(Check () all alpha beta alpha alpha)') == '(Check <e> all <a> <b> <a> <a>)'
    assert dict_replace(to_replace, 'a|b|c[abc]') == '<f><d>'
