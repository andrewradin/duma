
import pytest

from dtk.tests import auth_client, ws_with_attrs, make_ws, mock_dpi
from dtk.tests.entrez_utils_test import ez_vcr

ws_attrs = []
for i in range(1, 6):
    ws_attrs += [('DB0%d' % i, 'canonical', 'Drug%d' % i),
                 ('DB0%d' % i, 'm_dpimerge_id', 'DB0%d' % i),
                 ('DB0%d' % i, 'std_smiles', 'COO'),
                 ]

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

def test_hitsel_drugset(auth_client, make_ws, mock_dpi):
    mock_dpi('fake_dpi', dpi)
    ws = make_ws(ws_attrs)

    assert auth_client.get(f'/mol/{ws.id}/hit_selection/').status_code == 200

    from browse.models import DrugSet
    ds = DrugSet.objects.create(name='Drugset', ws=ws)

    opts = {
            'ds': f'ds{ds.id}',
            'select_btn': True,
            }
    resp = auth_client.post(f'/mol/{ws.id}/hit_selection/', opts, follow=True)
    assert resp.status_code == 200
    assert len(resp.redirect_chain) == 1


# queries eutils.
@ez_vcr.use_cassette('dtk/testdata/test_hitsel_addmol.yaml', record_mode='once')
def test_hitsel_addmol(auth_client, make_ws, mock_dpi):
    mock_dpi('fake_dpi', dpi)
    ws = make_ws(ws_attrs)

    from browse.models import DrugSet, WsAnnotation
    ds = DrugSet.objects.create(name='Drugset', ws=ws)
    wsa = WsAnnotation.objects.all()[0]
    assert len(ds.drugs.all()) == 0

    opts = {
            'mols': f'[{wsa.id}]',
            'addmol_btn': True,
            }
    assert auth_client.post(f'/mol/{ws.id}/hit_selection/?ds=ds{ds.id}', opts, follow=True).status_code == 200
    assert len(ds.drugs.all()) == 1
    assert ds.drugs.all()[0] == wsa

    opts = {
            'wsa_id': f'{wsa.id}',
            'delwsa_btn': True,
            }
    assert auth_client.post(f'/mol/{ws.id}/hit_selection/?ds=ds{ds.id}', opts, follow=True).status_code == 200
    assert len(ds.drugs.all()) == 0

def test_hitsel_moa(auth_client, make_ws, mock_dpi):
    mock_dpi('fake_dpi', dpi)
    ws = make_ws(ws_attrs)

    from browse.models import DrugSet, WsAnnotation
    ds = DrugSet.objects.create(name='Drugset', ws=ws)

    opts = {
            'prots': f'[["P01", "G01"]]',
            'moa_btn': True,
            }
    assert auth_client.post(f'/mol/{ws.id}/hit_selection/?ds=ds{ds.id}', opts, follow=True).status_code == 200

# queries eutils.
@ez_vcr.use_cassette('dtk/testdata/test_hitsel_save.yaml', record_mode='once')
def test_hitsel_save(auth_client, make_ws, mock_dpi):
    mock_dpi('fake_dpi', dpi)
    ws = make_ws(ws_attrs)

    from browse.models import DrugSet, WsAnnotation
    ds = DrugSet.objects.create(name='Drugset', ws=ws)
    wsa = WsAnnotation.objects.all()[0]
    ds.drugs.add(wsa)


    opts = {
            'hitsel_note': f'Note for hitsel page',
            f'score_{wsa.id}_1': 1.5,
            f'note_{wsa.id}_1': "A note for this",
            'save_btn': True,
            }
    assert auth_client.post(f'/mol/{ws.id}/hit_selection/?ds=ds{ds.id}', opts, follow=True).status_code == 200

    assert auth_client.get(f'/mol/{ws.id}/hit_selection_report/?ds=ds{ds.id}').status_code == 200
    assert 'Note for hitsel page' in auth_client.get(f'/mol/{ws.id}/hit_selection_report/?ds=ds{ds.id}').content.decode('utf8')

    opts = {
            'note': f'Note for report page',
            'save_btn': True,
            }
    assert auth_client.post(f'/mol/{ws.id}/hit_selection_report/?ds=ds{ds.id}', opts, follow=True).status_code == 200
    assert 'Note for report page' in auth_client.get(f'/mol/{ws.id}/hit_selection_report/?ds=ds{ds.id}').content.decode('utf8')

