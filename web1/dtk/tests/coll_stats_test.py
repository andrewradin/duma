import pytest

from dtk.coll_stats import CollStats

def test_membership(db):
    cs = CollStats()
    cs.add_keys('coll1','abcde')
    cs.add_keys('coll2','abcfg')
    cs.add_keys('coll3','cdfhij')
    # which collections hold a given key?
    key2coll_mm = cs.get_key2coll_mm()
    assert key2coll_mm.fwd_map()['a'] == set(['coll1','coll2'])
    # which keys are held by which combination of collections?
    collset2key_mm = cs.get_collset2key_mm()
    assert collset2key_mm.fwd_map()['coll1'] == set('e')
    assert collset2key_mm.fwd_map()['coll3'] == set('hij')
    assert collset2key_mm.fwd_map()['coll1 coll3'] == set('d')

def test_filtered_membership(db):
    cs = CollStats()
    cs.add_attrs('coll1',[
            'is_red',
            'is_yellow',
            ],[
            ('a',(True,False)),
            ('b',(True,True)),
            ('c',(True,False)),
            ('d',(True,True)),
            ])
    cs.add_attrs('coll2',[
            'is_yellow',
            'is_red',
            ],[
            ('a',(True,True)),
            ('b',(False,True)),
            ('c',(True,False)),
            ('d',(False,False)),
            ])
    # which collections supply 'red' members?
    collset2key_mm = cs.get_collset2key_mm(lambda x:x.is_red)
    assert len(collset2key_mm.fwd_map()['coll1']) == 2
    assert len(collset2key_mm.fwd_map()['coll1 coll2']) == 2
    assert 'coll2' not in collset2key_mm.fwd_map()

def test_basic_filtering(db):
    # CollStats keeps track of collections by name
    cs = CollStats()
    with pytest.raises(KeyError):
        cs.keys('coll1')
    # at a minimum, a collection is a set of keys, which can
    # be added to in batches from any iterable
    cs.add_keys('coll1',[])
    assert len(cs.keys('coll1')) == 0
    cs.add_keys('coll1','abcde')
    assert len(cs.keys('coll1')) == 5
    cs.add_keys('coll1','abcf')
    assert len(cs.keys('coll1')) == 6
    # collections can also have named attributes, which allow filtering
    cs.add_attrs('coll1',[
            'is_red',
            'is_yellow',
            ],[
            ('a',(True,False)),
            ('b',(True,True)),
            ('c',(True,False)),
            ('d',(True,True)),
            ])
    assert len(cs.filter('coll1',lambda x:x.is_red)) == 4
    assert cs.filter('coll1',lambda x:x.is_yellow) == set('bd')
    # attrs not explicitly defined for a key appear as None
    assert cs.filter('coll1',lambda x:x.is_yellow is None) == set('ef')
    # a second collection can have attributes with matching names
    # but distinct values
    cs.add_attrs('coll2',[
            'is_yellow',
            'is_red',
            ],[
            ('a',(True,True)),
            ('b',(False,True)),
            ('c',(True,False)),
            ('d',(False,False)),
            ])
    func = lambda x:x.is_yellow
    assert cs.filter('coll1',func) == set('bd')
    assert cs.filter('coll2',func) == set('ac')
    # global attributes are available when filtering any collection
    cs.add_global_attrs([
            'is_tall',
            'is_wide',
            ],[
            ('a',(True,True)),
            ('b',(True,False)),
            ])
    func = lambda x:x.is_yellow and x.is_tall
    assert cs.filter('coll1',func) == set('b')
    assert cs.filter('coll2',func) == set('a')
