import pytest

def test_rank_tracer():
    from dtk.score_pairs import RankTracer
    keys = 'abcdef'
    def make_rt(a_vals,b_vals):
        return RankTracer(
            x_ord=sorted(zip(keys,a_vals),key=lambda x:-x[1]),
            y_ord=sorted(zip(keys,b_vals),key=lambda x:-x[1]),
            )
    # three values, x & y are equal
    x_ord = list(zip(keys,[.8,.6,.2]))
    rt = make_rt(
            [.8,.6,.2],
            [.8,.6,.2],
            )
    rt.rank = 1
    assert [x[0] for x in rt.trace] == pytest.approx([0,1])
    assert [x[1] for x in rt.trace] == pytest.approx([.6,.6])
    # one crossing, descending
    rt = make_rt(
            [.8,.6,.2],
            [.5,.7,.2],
            )
    rt.rank = 1
    assert [x[0] for x in rt.trace] == pytest.approx([0,0.5,1])
    assert [x[1] for x in rt.trace] == pytest.approx([0.6,0.65,0.5])
    # two crossings, descending then ascending
    rt = make_rt(
            [.8,.6,.5,.2],
            [.8,.2,.5,.6],
            )
    rt.rank = 1
    assert [x[0] for x in rt.trace] == pytest.approx([0,.25,.75,1])
    assert [x[1] for x in rt.trace] == pytest.approx([.6,.5,.5,.6])
    # initial tie should pick middle slope
    rt = make_rt(
            [.5,.5,.5],
            [.8,.4,.2],
            )
    rt.rank = 1
    assert [x[0] for x in rt.trace] == pytest.approx([0,1])
    assert [x[1] for x in rt.trace] == pytest.approx([.5,.4])
    # crossing tie should pick closest slope
    rt = make_rt(
            [.7,.6,.5],
            [.3,.4,.5],
            )
    rt.rank = 2
    assert [x[0] for x in rt.trace] == pytest.approx([0,0.5,1])
    assert [x[1] for x in rt.trace] == pytest.approx([.5,.5,.4])
    # duplicate path should delay the appropriate number of crossings;
    # in both these cases there are 4 paths; paths C and D both run
    # from .5 to .5; path A runs from .7 to .3, crossing C and D at
    # 0.5; path B runs from .6 to .2, crossing C and D at .25. So,
    # the rankings are ABCD to 0.25, then ACDB to 0.5, the CDAB.
    rt = make_rt(
            [.7,.6,.5,.5],
            [.3,.2,.5,.5],
            )
    rt.rank = 2 # expect C,then D, then A
    assert [x[0] for x in rt.trace] == pytest.approx([0,0.25,0.5,1])
    assert [x[1] for x in rt.trace] == pytest.approx([.5,.5,.5,.3])
    rt = make_rt(
            [.7,.6,.5,.5],
            [.3,.2,.5,.5],
            )
    rt.rank = 1 # expect B, then C, then D
    assert [x[0] for x in rt.trace] == pytest.approx([0,0.25,.5,1])
    assert [x[1] for x in rt.trace] == pytest.approx([.6,.5,.5,.5])
    # more paths in the group shouldn't make a difference
    rt = make_rt(
            [.7,.6,.5,.5,.5,.5],
            [.3,.2,.5,.5,.5,.5],
            )
    rt.rank = 1 # expect B, then C, then D
    assert [x[0] for x in rt.trace] == pytest.approx([0,0.25,.5,1])
    assert [x[1] for x in rt.trace] == pytest.approx([.6,.5,.5,.5])

