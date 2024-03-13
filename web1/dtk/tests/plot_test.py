


import os
import pytest
import dtk.plot as dp


def test_convert():
    def conv(x):
        import json
        return json.loads(json.dumps(x, default=dp.convert))
    import numpy as np

    assert conv({'a': set([1,2])}) == {'a': [1,2]}, "Set becomes list"
    assert conv({'a': np.array([1,2])}) == {'a': [1,2]}, "nparray becomes list"
    assert conv({'a': np.int64(42)}) == {'a': 42}, "np.int64 becomes int"
    assert conv({'a': np.int64(2)**60}) == {'a': 2**60}, "np.int64 becomes int"


    # Make sure unsupported types raise.
    class Unsupported:
        pass

    with pytest.raises(TypeError):
        conv({'a': Unsupported()})

def test_scatter(tmp_path):
    out_fn = str(tmp_path / 'out')
    dp.scatterplot('x', 'y', [(1,2), (3,4), (5,6)], fn=out_fn)
    assert os.path.exists(out_fn + '.png')

def test_save_and_thumbnail(tmp_path):
    out_fn = str(tmp_path / 'out.plotly')
    out_png = str(tmp_path / 'out.png')
    pp = dp.scatter2d('x', 'y', [(1,2), (3,4), (5,6)], jitter=True)
    fig = pp.as_figure()
    assert pp != None
    assert fig != None


    pp.save(out_fn, thumbnail=True)
    assert os.path.exists(out_fn + '.gz')
    dp.PlotlyPlot.block_if_thumbnailing(out_png)
    assert os.path.exists(out_png)

