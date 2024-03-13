from __future__ import print_function
from algorithms.pathsum2 import *

def test_pathsum2():
    t2p = [
        ['p11','t1',.001,0],
        ['p12','t1',.002,0],
        ['p13','t1',.004,0],
        ['px1','t1',.008,0],
        ['p21','t2',.016,0],
        ['px1','t2',.032,0],
        ]
    p2d = [
        ['drug','protein','evidence','direction'],
        ['d1','p11',.064,0],
        ['d1','py1',.128,0],
        ['d2','px1',.256,0],
        ['d2','p21',.512,0],
        ]
    p2p = [
        ['py1','p13',.222,0],
        ]
    ##
    # test 2-level mapping
    ##
    net = Network()
    net.run_mapping('t2p'
            ,t2p
            ,[(0,1)]
            ,auto_add=True
            ,header='protein tissue evidence direction'.split()
            )
    assert list(map(len,net.stack)) == [2,5]
    net.run_mapping('p2d'
            ,iter(p2d) # need iterator, not list, for header extraction
            ,[(1,2)]
            )
    assert list(map(len,net.stack)) == [2,5,2]
    ##
    # test direct score extraction
    ##
    ts = TargetSet()
    ts.load_from_network(net,(2,))
    p = ts.get_target('d1').paths
    assert len(p) == 1
    assert len(p[2]) == 1
    assert p[2][0] == ['t1', 'p11', 0.001, 0, 'd1', 0.064, 0]
    p = ts.get_target('d2').paths
    assert len(p) == 1
    assert len(p[2]) == 3
    assert p[2][0] == ['t1', 'px1', 0.008, 0, 'd2', 0.256, 0]
    assert p[2][1] == ['t2', 'px1', 0.032, 0, 'd2', 0.256, 0]
    assert p[2][2] == ['t2', 'p21', 0.016, 0, 'd2', 0.512, 0]

    ss = ScoreSet(ts,[
            Accumulator({2:[2,5]}), # direct
            ])
    assert near(ss.scores['d1'][0],(.001+.064)/2)
    assert near(ss.scores['d2'][0],(0.008+0.256)/2+(0.016+0.512)/2)
    ##
    # test 3-level mapping
    ##
    net.run_mapping('p2p'
            ,p2p
            ,[(1,3)]
            ,header='prot1 prot2 evidence direction'.split()
            )
    net.run_mapping('p2d'
            ,iter(p2d) # need iterator, not list, for header extraction
            ,[(3,4)]
            )
    assert list(map(len,net.stack)) == [2,5,2,1,1]
    ##
    # test indirect score extraction
    ##
    ts = TargetSet()
    ts.load_from_network(net,(4,))
    p = ts.get_target('d1').paths
    assert len(p) == 1
    assert len(p[4]) == 1
    assert p[4][0] == ['t1', 'p13', .004, 0, 'py1', .222, 0, 'd1', .128, 0]

    ss = ScoreSet(ts,[
            Accumulator({4:[2,5,8]}), # indirect
            ])
    assert near(ss.scores['d1'][0],(.004+.222+.128)/3)
    # if we get this far...
    print('all tests passed')
