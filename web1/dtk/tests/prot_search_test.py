from __future__ import print_function

from builtins import range
from dtk.prot_search import parse_global_data, filter_global_data_duplicates, DrugNameReducer
import pytest


def test_reduce_drug_name():
    redu = DrugNameReducer.reduce_drug_name
    assert redu('something sodium') == 'something'
    assert redu('something sodium tablets') == 'something'
    assert redu('no matches') == 'no matches'

def test_filter_global_data_duplicates():
    data = """
        prednisone sodium phosphate	targetA
        prednisone sodium	targetA
        prednisone	targetA
        """
    expected = [{
        'title': 'prednisone',
        'full_name': 'prednisone sodium phosphate', # The first one wins.
        'targets': [{'name': 'targetA', 'ids': []}]
        }]
    actual = parse_global_data(data)
    assert actual == expected

    data = """
        prednisone sodium phosphate	targetA
        prednisone sodium	targetA
        """
    expected = [{
        'title': 'prednisone',
        'full_name': 'prednisone sodium phosphate', # The first one wins.
        'targets': [{'name': 'targetA', 'ids': []}]
        }]
    actual = parse_global_data(data)
    assert actual == expected

    data = """
        prednisone sodium phosphate	targetA
        prednisone sodium	targetB
        """
    expected = [{
        'title': 'prednisone sodium phosphate',
        'full_name': 'prednisone sodium phosphate',
        'targets': [{'name': 'targetA', 'ids': []}]
        }, {
        'title': 'prednisone sodium',
        'full_name': 'prednisone sodium',
        'targets': [{'name': 'targetB', 'ids': []}]
        }]
    actual = parse_global_data(data)
    assert actual == expected

    # This shows up in real data, sadly.  Make sure we keep both.
    data = """
        prednisone	targetA
        prednisone	targetB
        """
    expected = [{
        'title': 'prednisone',
        'full_name': 'prednisone',
        'targets': [{'name': 'targetA', 'ids': []}]
        }, {
        'title': 'prednisone',
        'full_name': 'prednisone',
        'targets': [{'name': 'targetB', 'ids': []}]
        }]
    actual = parse_global_data(data)
    assert actual == expected


def test_parse_global_data():
    data = """
            GS-6201	Adenosine Receptor A2b (ADORA2B)
            APCETH-201	Alpha 1 Antitrypsin (Alpha 1 Protease Inhibitor or SERPINA1)
            tipelukast	Arach (5 Lipoxygenase or ALOX5 or EC 1.13.11.34); Leukotriene Receptor; Phosphodiesterase 3 (PDE3 or EC 3.1.4.17)
           """
    
    result = parse_global_data(data) 

    expected = [{
            'title': 'GS-6201',
            'full_name': 'GS-6201',
            'targets': [{
                'name': 'Adenosine Receptor A2b',
                'ids': ['ADORA2B']
                }]
            }, {
            'title': 'APCETH-201',
            'full_name': 'APCETH-201',
            'targets': [{
                'name': 'Alpha 1 Antitrypsin',
                'ids': ['Alpha 1 Protease Inhibitor', 'SERPINA1']
                }]
            }, {
            'title': 'tipelukast',
            'full_name': 'tipelukast',
            'targets': [{
                'name': 'Arach',
                'ids': ['5 Lipoxygenase', 'ALOX5', 'EC 1.13.11.34']
                }, {
                'name': 'Leukotriene Receptor',
                'ids': []
                }, {
                'name': 'Phosphodiesterase 3',
                'ids': ['PDE3', 'EC 3.1.4.17']
                }]
            }]
            

    assert result == expected

@pytest.fixture
def search_setup():
    from browse.models import Protein, ProteinAttributeType, ProteinAttribute
    prots = ['P1', 'P2', 'P3', 'P4']
    alt_prots = [[], [], [], ['altp4', 'alt-p4']]
    genes = ['G1', '', 'G3', 'G4']
    names = ['prot1', 'prot2', 'prot3', 'prot4']
    alt_names = [[], ['alt_prot_2'], [], []]

    main_name_attr,new = ProteinAttributeType.objects.get_or_create(name="Protein_Name")
    alt_name_attr,new = ProteinAttributeType.objects.get_or_create(name="Alt_Protein_Name")
    alt_uni_attr,new = ProteinAttributeType.objects.get_or_create(name="Alt_uniprot")

    for i in range(len(prots)):
        p = Protein.objects.create(uniprot=prots[i], gene=genes[i])
        pa = ProteinAttribute.objects.create(
                prot=p,
                attr=main_name_attr,
                val=names[i])
        for alt_name in alt_names[i]:
            pa = ProteinAttribute.objects.create(
                    prot=p,
                    attr=alt_name_attr,
                    val=alt_name)
        for alt_uni in alt_prots[i]:
            pa = ProteinAttribute.objects.create(
                    prot=p,
                    attr=alt_uni_attr,
                    val=alt_uni)
    return prots

from dtk.prot_search import search_by_any
@pytest.mark.django_db
def test_search_by_uniprot(django_assert_num_queries, search_setup):
    with django_assert_num_queries(1) as c:
        out, hit_limit = search_by_any('P1', limit=1)
        assert not hit_limit
        assert len(out) == 1
        assert out[0].uniprot == 'P1'
    
    for q in c.captured_queries:
        print("Query: %s" % q['sql'])
        assert 'DISTINCT' in q['sql']


@pytest.mark.django_db
def test_search_limit(django_assert_num_queries, search_setup):
    with django_assert_num_queries(1) as c:
        out, hit_limit = search_by_any('prot', limit=3)
        assert hit_limit
        assert len(out) == 3


@pytest.mark.django_db
def test_search_name(django_assert_num_queries, search_setup):
    with django_assert_num_queries(1) as c:
        out, hit_limit = search_by_any('prot2', limit=1)
        assert not hit_limit
        assert len(out) == 1
        assert out[0].uniprot == 'P2'

@pytest.mark.django_db
def test_search_gene(django_assert_num_queries, search_setup):
    with django_assert_num_queries(1) as c:
        out, hit_limit = search_by_any('G3', limit=3)
        assert not hit_limit
        assert len(out) == 1
        assert out[0].uniprot == 'P3'

@pytest.mark.django_db
def test_search_alt_uni(django_assert_num_queries, search_setup):
    with django_assert_num_queries(1) as c:
        out, hit_limit = search_by_any('alt-p4', limit=3)
        assert not hit_limit
        assert len(out) == 1
        assert out[0].uniprot == 'P4'

@pytest.mark.django_db
def test_search_alt_name(django_assert_num_queries, search_setup):
    with django_assert_num_queries(1) as c:
        out, hit_limit = search_by_any('alt_prot_2', limit=3)
        assert not hit_limit
        assert len(out) == 1
        assert out[0].uniprot == 'P2'

@pytest.mark.django_db
def test_search_multiple(django_assert_num_queries, search_setup):
    prots = search_setup
    with django_assert_num_queries(1) as c:
        out, hit_limit = search_by_any('P', limit=None)
        assert not hit_limit
        assert len(out) == 4
        actual_uniprots = set([x.uniprot for x in out])
        assert actual_uniprots == set(prots)


