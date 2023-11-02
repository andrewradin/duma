from dtk.tests.tmpdir_path_helper import tmpdir_path_helper
from dtk.tests.std_vcr import before_record_request

from dtk.tests.end_to_end_test import shim_background_jobs_in_process, selenium, setup_users, login_selenium, wait_for_background_job, setup_lts_tmpdir
from mock import patch
import json
import pytest
import vcr

custom_vcr = vcr.VCR(
        before_record_request=before_record_request,
        match_on=['path', 'method']
        )

cassette_fn = 'patsearch/tests/data/pat_search3.yaml'

@patch('dtk.lts.LtsRepo')
@patch('patsearch.patent_search.MAX_SEARCH_RESULTS', 2)
@custom_vcr.use_cassette(cassette_fn, record_mode='once')
@pytest.mark.django_db(transaction=True)
def test_patent_search(lts_mock, django_user_model, client, tmpdir_path_helper):
    setup_lts_tmpdir()

    shim_background_jobs_in_process()
    setup_users(django_user_model, client)
    from browse.models import Workspace
    ws = Workspace.objects.create(name='sjogren')
    ws_id = ws.id

    url = '/pats/{}/search/'.format((ws_id))
    client.get(url).status_code == 200

    from patsearch.patent_search import BQ_TABLES
    assert BQ_TABLES[3][1].endswith('micro'), "Make sure we're testing against micro"

    drug1 = {
            'drug_terms': ['coumarin', u'nöt a match for anything'],
            'target_terms': [],
            'name': 'Coumarin'
        }
    drug2 = {
            'drug_terms': [],
            'target_terms': ['Phosphodiesterase Type 5 Inhibitor'],
            'name': 'Salt'
        }
    query = {
            'diseaseNames': [u'sjögren'],
            'drugList': [drug1, drug2],
            'tableNumber': '3',
        }
    opts = {
            'query': json.dumps(query),
            'search_btn': True,
        }
    client.post(url, opts).status_code == 200

    wait_for_background_job(expect_success=True, timeout_secs=60)

    from patsearch.models import PatentSearch, DrugDiseasePatentSearch
    assert len(PatentSearch.objects.all()) == 1
    ps_id = PatentSearch.objects.all()[0].id

    url = '/pats/{}/summary/{}/'.format((ws_id), (ps_id))
    resp = client.get(url)
    assert resp.status_code == 200

    assert 'Coumarin' in str(resp.content)
    assert 'Salt (targets)' in str(resp.content)


    dd_searches = DrugDiseasePatentSearch.objects.all()
    assert len(dd_searches) == 2
    dd_id = dd_searches[0].id

    url = '/pats/{}/resolve/{}/'.format((ws_id), (dd_id))
    resp = client.get(url)
    assert resp.status_code == 200
    
from dtk.tests.ws_with_attrs import ws_with_attrs

ws_attrs = [
        ("DB01", "canonical", "Levobupivacaine"),
        ] + [("CHEMBL%d" % i, "canonical", "Drug%d" % i) for i in range(20)]

@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
def test_search_drugs(ws_with_attrs, client, django_user_model):
    ws = ws_with_attrs
    setup_users(django_user_model, client)
    url = '/pats/{}/search_drugs/Levo/'.format((ws.id))
    resp = client.get(url)
    assert resp.status_code == 200
    content = json.loads(resp.content)
    assert len(content['names']) == 1
    assert content['names'][0]['name'] == 'Levobupivacaine'

    url = '/pats/{}/search_drugs/v/'.format((ws.id))
    resp = client.get(url)
    assert resp.status_code == 200
    content = json.loads(resp.content)
    assert len(content['names']) == 1
    assert content['names'][0]['name'] == 'Levobupivacaine'

    url = '/pats/{}/search_drugs/q/'.format((ws.id))
    resp = client.get(url)
    assert resp.status_code == 200
    content = json.loads(resp.content)
    assert len(content['names']) == 0

    url = '/pats/{}/search_drugs/Dru/'.format((ws.id))
    resp = client.get(url)
    assert resp.status_code == 200
    content = json.loads(resp.content)
    assert len(content['names']) == 10


@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
def test_get_drugset(ws_with_attrs, client, django_user_model):
    ws = ws_with_attrs
    setup_users(django_user_model, client)
    url = '/pats/{}/drugset/kts/'.format((ws.id))
    resp = client.get(url)
    assert resp.status_code == 200


ws_attrs = [
        ("DB00001", "canonical", "drug_can"),
        ("DB00001", "synonym", "drug_syn1"),
        ("DB00001", "synonym", "drug_syn2"),
        ("DB00001", "synonym", "drug_syn3"),
        ("DB00001", "brand", "drug_brand"),
        ]
@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
def test_get_drug(ws_with_attrs, client, django_user_model):
    ws = ws_with_attrs
    setup_users(django_user_model, client)
    from browse.models import WsAnnotation
    wsas = WsAnnotation.objects.all()
    assert len(wsas) == 1
    url = '/pats/{}/drug/{}/'.format((ws.id), (wsas[0].id))
    resp = client.get(url)
    assert resp.status_code == 200
    content = json.loads(resp.content)
    assert content['wsa'] == wsas[0].id
    assert content['name'] == 'drug_can'
    names = ['drug_can', 'drug_syn1', 'drug_syn2', 'drug_syn3', 'drug_brand']
    assert set(content['drug_terms']) == set(names)



from patsearch.models import *
@pytest.mark.django_db
def test_initial_resolution():

    from browse.models import Workspace, WsAnnotation
    ws1 = Workspace.objects.create(name='sjogren')
    ws2 = Workspace.objects.create(name='acne')

    from patsearch.patent_search import get_initial_resolution
    search1 = PatentSearch.objects.create(
            ws=ws1,
            user='me'
            )

    dd_search_1 = DrugDiseasePatentSearch.objects.create(
            patent_search=search1,
            query="query",
            drug_name="drug1",
            wsa=None,
            )

    patent_1 = Patent.objects.create(pub_id='US-001', title='pat-1') 
    patent_2 = Patent.objects.create(pub_id='US-002', title='pat-2') 
    patent_3 = Patent.objects.create(pub_id='US-003', title='pat-3') 

    rv = PatentSearchResult.resolution_vals
    assert get_initial_resolution(patent_1, ws1, None) == rv.UNRESOLVED,\
            "Defaults to unresolved"

    result_1 = PatentSearchResult.objects.create(
            search=dd_search_1,
            patent=patent_1,
            resolution=rv.IRRELEVANT_ALL,
            score=0,
            evidence="",
            )

    assert get_initial_resolution(patent_1, ws1, None) == rv.IRRELEVANT_ALL,\
            "Once one is irrelevant all, all should be"
    assert get_initial_resolution(patent_1, ws2, None) == rv.IRRELEVANT_ALL,\
            "Even in another workspace"

    result_1.resolution = rv.IRRELEVANT_DISEASE
    result_1.save()

    assert get_initial_resolution(patent_1, ws1, None) == rv.IRRELEVANT_DISEASE,\
            "Same workspace"
    assert get_initial_resolution(patent_1, ws2, None) == rv.UNRESOLVED,\
            "Different workspace, leave unresolved"


    result_1.resolution = rv.IRRELEVANT_DRUG
    result_1.save()

    wsa1 = WsAnnotation.objects.create(ws=ws1)
    dd_search_2 = DrugDiseasePatentSearch.objects.create(
            patent_search=search1,
            query="query",
            drug_name="drug1",
            wsa=wsa1,
            )

    result_2 = PatentSearchResult.objects.create(
            search=dd_search_2,
            patent=patent_1,
            resolution=rv.IRRELEVANT_DRUG,
            score=0,
            evidence="",
            )


    assert get_initial_resolution(patent_1, ws1, None) == rv.UNRESOLVED, \
            "No drugs set"

    assert get_initial_resolution(patent_1, ws1, wsa1) == rv.IRRELEVANT_DRUG, \
            "This patent has been marked irrelevant to this drug elsewhere"




@pytest.mark.django_db
def test_apply_resolution():

    from browse.models import Workspace, WsAnnotation
    ws1 = Workspace.objects.create(name='sjogren')
    ws2 = Workspace.objects.create(name='acne')
    wsa1 = WsAnnotation.objects.create(ws=ws1)

    from patsearch.patent_search import get_initial_resolution
    search1 = PatentSearch.objects.create(
            ws=ws1,
            user='me'
            )
    search2 = PatentSearch.objects.create(
            ws=ws2,
            user='me'
            )

    dd_search_1 = DrugDiseasePatentSearch.objects.create(
            patent_search=search1,
            query="query",
            drug_name="drug1",
            wsa=None,
            )
    dd_search_2 = DrugDiseasePatentSearch.objects.create(
            patent_search=search1,
            query="query",
            drug_name="drug1",
            wsa=wsa1,
            )
    dd_search_3 = DrugDiseasePatentSearch.objects.create(
            patent_search=search2,
            query="query",
            drug_name="drug1",
            wsa=wsa1,
            )
    dd_search_4 = DrugDiseasePatentSearch.objects.create(
            patent_search=search2,
            query="query",
            drug_name="drug1",
            )

    patent_1 = Patent.objects.create(pub_id='US-001', title='pat-1') 

    rv = PatentSearchResult.resolution_vals

    result_1 = PatentSearchResult.objects.create(
            search=dd_search_1,
            patent=patent_1,
            resolution=rv.UNRESOLVED,
            score=0,
            evidence="",
            )
    result_2 = PatentSearchResult.objects.create(
            search=dd_search_2,
            patent=patent_1,
            resolution=rv.UNRESOLVED,
            score=0,
            evidence="",
            )
    result_3 = PatentSearchResult.objects.create(
            search=dd_search_3,
            patent=patent_1,
            resolution=rv.UNRESOLVED,
            score=0,
            evidence="",
            )
    result_4 = PatentSearchResult.objects.create(
            search=dd_search_4,
            patent=patent_1,
            resolution=rv.UNRESOLVED,
            score=0,
            evidence="",
            )

    from patsearch.patent_search import apply_patent_resolution

    apply_patent_resolution(result_1.id, rv.IRRELEVANT_DRUG)
    result_2.refresh_from_db()
    assert result_2.resolution == rv.UNRESOLVED
    result_4.refresh_from_db()
    assert result_4.resolution == rv.UNRESOLVED

    apply_patent_resolution(result_1.id, rv.IRRELEVANT_DISEASE)
    result_2.refresh_from_db()
    assert result_2.resolution == rv.IRRELEVANT_DISEASE
    result_3.refresh_from_db()
    assert result_3.resolution == rv.UNRESOLVED
    result_4.refresh_from_db()
    assert result_4.resolution == rv.UNRESOLVED

    apply_patent_resolution(result_2.id, rv.IRRELEVANT_DRUG)
    result_3.refresh_from_db()
    assert result_3.resolution == rv.IRRELEVANT_DRUG
    result_4.refresh_from_db()
    assert result_4.resolution == rv.UNRESOLVED

    apply_patent_resolution(result_2.id, rv.IRRELEVANT_ALL)
    result_4.refresh_from_db()
    assert result_4.resolution == rv.IRRELEVANT_ALL





