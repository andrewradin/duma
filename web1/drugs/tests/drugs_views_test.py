
import pytest
from dtk.tests import auth_client, make_ws, mock_dpi
from dtk.tests.end_to_end_test import assert_good_response
from mock import patch

def test_chem_image(auth_client, make_ws):
    ws = make_ws([
        ('DB01', 'canonical', 'drug1'),
        ('DB01', 'smiles_code', 'CC(C(=O)NC(CC(=O)O)C(=O)COC1=C(C(=CC(=C1F)F)F)F)NC(=O)C(=O)NC2=CC=CC=C2C(C)(C)C'),
        ('DB02', 'canonical', 'drug2'),
    ])
    from drugs.models import Drug
    drug1 = Drug.objects.all()[0]
    drug2 = Drug.objects.all()[1]

    rsp = auth_client.get(f'/drugs/chem_image/{drug1.id}/?core=true&align=CCCC')
    assert_good_response(rsp)
    assert '<rect' in rsp.content.decode('utf8'), "Should be an SVG"

    rsp = auth_client.get(f'/drugs/chem_image/{drug2.id}/?core=true&align=CCCC')
    assert_good_response(rsp)
    assert 'svg xmlns' in rsp.content.decode('utf8'), "Should be an SVG"
    assert '<rect' not in rsp.content.decode('utf8'), "No smiles, should just be a blank svg"

def test_search_view(auth_client, make_ws, mock_dpi):
    mock_dpi('dpimerge.fake_dpi', [
        ('dpimerge_id', 'uniprot_id', 'evidence', 'direction'),
        ('DB01', 'P01', '1.0', '1.0'),
    ])
    ws = make_ws([
        ('DB01', 'canonical', 'drug1'),
    ])

    assert_good_response(auth_client.get(f'/drugs/search/'))
    
    assert_good_response(auth_client.post(f'/drugs/search/', {
        'search_btn': True,
        'ws_id': ws.id,
        'pattern': 'drug',
    }, follow=True))

@pytest.mark.django_db
def test_validate_smiles(make_ws):
    ws = make_ws([
        ('DB01', 'canonical', 'drug1'),
        ('DB01', 'std_smiles', 'CC(=O)O'),
        ('DB02', 'canonical', 'drug2'),
        ('DB02', 'std_smiles', 'OS(=O)'),
    ])
    from drugs.drug_edit import validate_smiles
    from drugs.models import Drug

    assert validate_smiles('COO') == {'matches': []}

    with pytest.raises(Exception):
        validate_smiles('not a real smiles')

    d1_id = Drug.objects.get(tag__prop__name='canonical', tag__value='drug1').id

    d1_info = {
        'id': d1_id,
        'collection__name': 'test_col.default',
        'tag__value': 'drug1',
    }

    assert validate_smiles('CC(=O)O') == {'matches': [d1_info]}
    assert validate_smiles('OC(=O)C') == {'matches': [d1_info]}