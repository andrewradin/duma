
from builtins import range
import pytest
from pytest import approx

from dtk.enrichment import EMInput


def test_tie_adjusted_ranks():
    def run(scores, kt_set):
        return EMInput(score=scores, kt_set=kt_set).get_tie_adjusted_ranks()

    scores = [('d%d' % (10-i), i / 10.0) for i in range(10, 0, -1)]

    assert run(scores, ('d0',)) == [0]
    assert run(scores, ('d9',)) == [9]
    assert run(scores, ('d0','d7','d9')) == [0,7,9]

    vals = [10, 9, 9, 9, 8, 7, 7, 5, 5, 5]
    scores = [('d%d' % i, vals[i]) for i in range(0, 10)]
    assert run(scores, ('d0',)) == [0]
    assert run(scores, ('d1',)) == [2]
    assert run(scores, ('d2',)) == [2]
    assert run(scores, ('d3',)) == [2]
    assert run(scores, ('d1', 'd3')) == [1, 2]
    assert run(scores, ('d1', 'd2', 'd3')) == [1, 2, 3]
    assert run(scores, ('d1', 'd5')) == [2, 5]
    assert run(scores, ('d1', 'd5', 'd9')) == [2, 5, 8]

def make_run_enrichment(EMType):
    def run(scores, kt_set):
        emi = EMInput(score=scores, kt_set=kt_set)
        em = EMType()
        em.evaluate(emi)
        return em.rating
    return run

def test_sigma_of_rank():
    from dtk.enrichment import SigmaOfRank
    run = make_run_enrichment(SigmaOfRank)

    vals = [10, 9, 9, 9, 8, 7, 7, 5, 5, 5]
    scores = [('d%d' % i, vals[i]) for i in range(0, 10)]

    # This one isn't super interesting, just make sure it matches the known
    # existing value.  This will change if you modify any of the constants in
    # SigmaOfRank.
    assert run(scores, ('d0',)) == approx(0.982014)

def test_sigma_of_rank_ties():
    from dtk.enrichment import SigmaOfRank
    run = make_run_enrichment(SigmaOfRank)
    run_ties_test(run)


def test_febe():
    from dtk.enrichment import FEBE
    run = make_run_enrichment(FEBE)

    vals = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    scores = [('d%d' % i, vals[i]) for i in range(0, 10)]

    no_tie_scores = [run(scores, ('d%d' % i,)) for i in range(0, 10)]
    # Hard to interpret these numbers directly, mostly just checking they
    # aren't changing unexpectedly.
    assert no_tie_scores == approx([
        1.0,
        0.39794,
        0.22184875,
        .09691,
        0, 0, 0, 0, 0, 0
        ])

def test_febe_ties():
    from dtk.enrichment import FEBE
    run = make_run_enrichment(FEBE)
    run_ties_test(run)

def test_dea():
    from dtk.enrichment import DEA_AREA
    run = make_run_enrichment(DEA_AREA)

    vals = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    scores = [('d%d' % i, vals[i]) for i in range(0, 10)]

    no_tie_scores = [run(scores, ('d%d' % i,)) for i in range(0, 10)]
    # Hard to interpret these numbers directly, mostly just checking they
    # aren't changing unexpectedly.
    expected_scores = [
        17.3911883,
        17.3911883,
        15.840398,
        14.929017,
        14.2765175,
        13.767577,
        13.35066,
        12.997815,
        12.69185,
        12.421124
        ]
    for i, (actual, expected) in enumerate(zip(no_tie_scores, expected_scores)):
        assert actual == approx(expected), "Checking input %d" % i


def test_dea_ties():
    from dtk.enrichment import DEA_AREA
    run = make_run_enrichment(DEA_AREA)
    run_ties_test(run)

def run_ties_test(run):
    vals = [10, 9, 9, 9, 8, 7, 7, 5, 5, 5, 5]
    scores = [('d%d' % i, vals[i]) for i in range(0, 11)]

    # Check that we are handling ties properly, any ordering of ties should
    # be the same.
    assert run(scores, ('d1',)) == approx(run(scores, ('d2',)))
    assert run(scores, ('d1','d2')) == approx(run(scores, ('d2', 'd3')))
    assert run(scores, ('d1','d2')) == approx(run(scores, ('d1', 'd3')))
    assert run(scores, ('d5',)) == approx(run(scores, ('d6',)))
    assert run(scores, ('d7',)) == approx(run(scores, ('d9',)))
    assert run(scores, ('d7','d8','d10')) == approx(run(scores, ('d8', 'd9', 'd10')))



def test_dea_es_ties():
    from dtk.enrichment import DEA_ES
    run = make_run_enrichment(DEA_ES)
    run_ties_test(run)

def test_wfebe_ties():
    from dtk.enrichment import wFEBE
    run = make_run_enrichment(wFEBE)
    run_ties_test(run)


from dtk.tests import mock_dpi
from dtk.tests.ws_with_attrs import make_ws

ws_attrs = []
for i in range(1, 8):
    ws_attrs += [('DB0%d' % i, 'canonical', 'Drug%d' % i),
                 ('DB0%d' % i, 'm_dpimerge_id', 'DB0%d' % i)]

def test_condense_keys(make_ws, mock_dpi):
    ws = make_ws(ws_attrs)
    dpi = [
        ('dpimerge_id', 'uniprot_id', 'evidence', 'direction'),
        ('DB01', 'P01', '0.5', '0'),
        ('DB02', 'P01', '0.9', '0'),
        ('DB03', 'P01', '0.5', '0'),
        ('DB04', 'P01', '0.5', '0'),
        ('DB04', 'P02', '0.5', '1'),
        ('DB05', 'P01', '0.5', '0'),
        ('DB05', 'P02', '0.5', '1'),
        ]
    mock_dpi('fake_dpi', dpi)

    from dtk.enrichment import ws_dpi_condense_keys
    from browse.models import WsAnnotation

    wsas = WsAnnotation.objects.all()
    assert(len(wsas)) == 7

    keys = ws_dpi_condense_keys([x.id for x in wsas])
    assert len(keys) == 7
    assert keys[0] == (('P01', 0.5, 0), )
    assert keys[1] == (('P01', 0.9, 0), )
    assert keys[3] == (('P01', 0.5, 0), ('P02', 0.5, 1))

def test_condense_emi(make_ws, mock_dpi):
    ws = make_ws(ws_attrs)
    dpi = [
        ('dpimerge_id', 'uniprot_id', 'evidence', 'direction'),
        ('DB01', 'P01', '0.5', '0'),
        ('DB02', 'P01', '0.9', '0'),
        ('DB03', 'P01', '0.5', '0'),
        ('DB04', 'P01', '0.5', '0'),
        ('DB04', 'P02', '0.5', '1'),
        ('DB05', 'P01', '0.5', '0'),
        ('DB05', 'P02', '0.5', '1'),
        ]
    mock_dpi('fake_dpi', dpi)

    from dtk.enrichment import condense_emi, EMInput
    from browse.models import WsAnnotation

    wsas = WsAnnotation.objects.all()
    assert(len(wsas)) == 7

    ordering = [
            (wsas[i].id, 1)
            for i in range(len(wsas))
            ]
    emi1 = EMInput(ordering, set())
    cond1 = emi1.get_condensed_emi()
    # We expect to condense out 2 (same as 0), 4 (same as 3) and anything
    # after 5 (5 and above have no DPIs).
    assert cond1.get_labeled_score_vector() == [
            ordering[0],
            ordering[1],
            ordering[3],
            ordering[5]
            ]
