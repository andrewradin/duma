import pytest

def test_is_single_value():
    from dtk.num import is_single_value
    # basic functionality
    assert is_single_value([1])
    assert is_single_value([1,1])
    assert not is_single_value([1,2])
    # verify one-shot iterator will raise an exception
    with pytest.raises(Exception):
        is_single_value(iter([1,1,1]))

def test_avg():
    from dtk.num import avg,avg_sd
    assert avg([1,2,3]) == 2
    # verify single-value lists return that value as the average, regardless
    # of floating point rounding
    special_list = [1.00000000000001]*10000
    assert avg(special_list) == special_list[0]
    assert avg_sd(special_list) == (special_list[0],0)

def test_znorm():
    from dtk.scores import ZNorm
    norm = ZNorm([(x,x) for x in range(1,10)])
    assert norm.get(5) == 0
    assert norm.get(8) > 1
    assert norm.get(7) < 1
    assert norm.get(3) == -norm.get(7)

def test_mmnorm():
    from dtk.scores import MMNorm
    norm = MMNorm([(x,x) for x in range(0,5)])
    assert norm.range() == (0,1)
    assert norm.get(0) == 0
    assert norm.get(4) == 1


def test_condense_scores():
    G = {
        'a': ['b', 'c', 'd'],
        'c': ['e'],
        'd': ['b'],
    }
    scores = {
        'a': 1,
        'b': 3,
        'c': 4,
        'd': 5,
        'e': 1,
    }

    import networkx as nx
    g = nx.DiGraph()
    for node, neighs in G.items():
        for x in neighs:
            g.add_edge(node, x)
            
    print(g.edges)
    from dtk.scores import condense_scores
    out, covers = condense_scores('a', g.predecessors, g.successors, scores.items())

    assert dict(out) == {'c': 4, 'd': 5}
    assert covers == {'c': {'e'}, 'd': {'a', 'b'}}



def test_ranker():
    ord1 = [
        ('a', 1),
        ('b', 2),
        ('c', 2)
    ]

    from dtk.scores import Ranker
    r = Ranker(ord1)
    assert r.get_pct('a') == pytest.approx(100 * 1/3)
    assert r.get_pct('b') == 100
    assert r.get_pct('c') == 100


    ord2 = [(chr(ord('a') + i), i) for i in range(10)]
    r = Ranker(ord2)
    assert r.get_pct('a') == 10
    assert r.get_pct('c') == 30
    assert r.get_pct('e') == 50
    assert r.get_pct('j') == 100


    assert r.get_pct('not in dataset') == 100

    assert r.get('not in dataset') == r.total + 1

    r = Ranker(ord2, none_if_missing=True)
    assert r.get('not in dataset') == None