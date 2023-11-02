
import django
from mock import patch
from mock import MagicMock
import pytest

def do_test_GET(view, get_parms):
    from django.http import HttpRequest
    r = HttpRequest()
    r.method = 'GET'
    r.user = MagicMock()
    r.user.is_authenticated.return_value = True
    r.GET = get_parms

    view.args = None
    view.kwargs = {}
    view.message = lambda *args, **kwargs: None
    view.request = r
    view.dispatch()

class MockTissue:
    def __init__(self, name, sig_results=None, sig_qc_scores=None):
        self._name = name
        self._sig_results = sig_results
        self._sig_qc_scores = sig_qc_scores
    def concise_name(self):
        return self._name
    def sig_results(self, over_only=False):
        return self._sig_results
    def sig_qc_scores(self):
        return self._sig_qc_scores

from collections import namedtuple
SigResult = namedtuple('SigResult', 'uniprot evidence direction')
from ge.ts_views import TissueCorrView

import six
@pytest.mark.parametrize("corr_type", ["spearman", "pearson", "prot_overlap"])
@pytest.mark.parametrize("use_prot_cutoffs, use_direction", [("true", "true"), ('false', 'false')])
@patch.object(TissueCorrView, 'get_tissues')
def test_tissue_corr_view(get_tissues_fn, corr_type, use_prot_cutoffs, use_direction):

    view = TissueCorrView()
    params = {
        'tissue_set_id': 2,
        'corr_type': corr_type,
        'use_prot_cutoffs': use_prot_cutoffs,
        'use_direction': use_direction,
    }

    tissues = []
    get_tissues_fn.return_value = tissues
    do_test_GET(view, params)

    assert view.tissue_set_id == 2
    assert view.corr_type == corr_type



    tissues = [
            MockTissue('T1', [SigResult('P1', 0.9, 1), SigResult('P2', 0.8, 1)])
            ]
    get_tissues_fn.return_value = tissues
    do_test_GET(view, params)

    assert view.tissue_set_id == 2
    assert view.corr_type == corr_type



    tissues = [
            MockTissue('T1', [SigResult('P1', float('nan'), 1), SigResult('P2', float('inf'), 1)]),
            MockTissue('T2', [SigResult('P1', 0.9, 1), SigResult('P2', 0.8, 1)]),
            MockTissue('T3', [SigResult('P1', 0.9, 1), SigResult('P2', 0.8, 1)])
            ]
    get_tissues_fn.return_value = tissues
    do_test_GET(view, params)

    assert view.tissue_set_id == 2
    assert view.corr_type == corr_type
    assert view.corr_table != None


from ge.ts_views import TissueCorrView
@patch.object(TissueCorrView, 'get_tissues')
def test_tissue_corr_view_bad_type(get_tissues_fn):
    view = TissueCorrView()
    with pytest.raises(Exception):
        do_test_GET(view, {
            'tissue_set_id': 2,
            'corr_type': "not_real_type",
            'use_prot_cutoffs': use_prot_cutoffs,
            'use_direction': use_direction,
        })

from ge.ts_views import TissueSetAnalysisView
@patch.object(TissueSetAnalysisView, 'get_tissues')
def test_tissue_set_analysis(get_tissues_fn):
    view = TissueSetAnalysisView()
    params = {
        'tissue_set_id': 2,
    }

    tissues = []
    get_tissues_fn.return_value = tissues
    do_test_GET(view, params)

    assert view.tissue_set_id == 2

    scores = {
        'caseSampleScores': 0.9,
        'contSampleScores': 0.9,
        'concordScore': 0.9,
        'sigProbesScore': 0.9,
        'perfectProbesScore': 0.9,
        'caseCorScore': 0.9,
        'controlCorScore': 0.9,
        'consistDirScore': 0.9,
        'mappingScore': 0.9,
        'finalScore': 0.9,
    }
    scores2 = scores.copy()
    # Some scores are 'NA'
    scores2['concordScore'] = 'NA'
    # Some datasets have no scores at all.
    scores3 = {}

    tissues = [
            MockTissue('T1', sig_qc_scores=scores),
            ]
    get_tissues_fn.return_value = tissues
    do_test_GET(view, params)

    assert view.tissue_set_id == 2

    tissues = [
            MockTissue('T1', sig_qc_scores=scores),
            MockTissue('T2', sig_qc_scores=scores2),
            MockTissue('T3', sig_qc_scores=scores3),
            ]
    get_tissues_fn.return_value = tissues
    do_test_GET(view, params)

    assert view.tissue_set_id == 2

