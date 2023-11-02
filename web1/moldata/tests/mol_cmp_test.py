import pytest

from dtk.tests import auth_client, ws_with_attrs, make_ws, mock_dpi
from dtk.tests.entrez_utils_test import ez_vcr

ws_attrs = []
for i in range(1, 6):
    ws_attrs += [('DB0%d' % i, 'canonical', 'Drug%d' % i),
                 ('DB0%d' % i, 'm_dpimerge_id', 'DB0%d' % i),
                 ('DB0%d' % i, 'std_smiles', 'COO'),
                 ]
@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
def test_mol_cmp_get(auth_client, ws_with_attrs):

    from browse.models import WsAnnotation
    ws = ws_with_attrs
    wsas = WsAnnotation.objects.all()

    assert auth_client.get(f'/mol/{ws.id}/molcmp/').status_code == 200, 'No parameters should work'
    assert auth_client.get(f'/mol/{ws.id}/molcmp/?search_wsa={wsas[0].id}').status_code == 200, 'search_wsa'

@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
def test_mol_cmp_search(auth_client, ws_with_attrs):
    from browse.models import WsAnnotation
    ws = ws_with_attrs
    wsas = WsAnnotation.objects.all()
    opts = {
            'search_btn': True,
            'search_wsa': wsas[0].id,
            }
    resp = auth_client.post(f'/mol/{ws.id}/molcmp/', opts, follow=True)
    assert resp.status_code == 200
    redirect_chain = resp.redirect_chain
    assert len(redirect_chain) == 1
    search_id = redirect_chain[0][0].split('=')[1]
    assert search_id == str(wsas[0].id)


def test_mol_cmp_prot_search(auth_client, make_ws, mock_dpi):
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
    ws = make_ws(ws_attrs)
    import json
    opts = {
            'protsearch_btn': True,
            'search_prots': json.dumps([['P01', 'GENE', '0']]),
            }
    resp = auth_client.post(f'/mol/{ws.id}/molcmp/', opts, follow=True)
    assert resp.status_code == 200


# queries eutils.
@ez_vcr.use_cassette('dtk/testdata/test_mol_cmp_drugset.yaml', record_mode='once')
def test_mol_cmp_drugset(auth_client, make_ws, mock_dpi):
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
    ws = make_ws(ws_attrs)
    import json

    from browse.models import DrugSet, WsAnnotation
    num_ds_before = DrugSet.objects.all().count()
    opts = {
            'create_btn': True,
            'name': 'My New Drugset',
            'description': "For MolCmp",
            'prots': '[["P01", "G01"]]',
            }
    resp = auth_client.post(f'/mol/{ws.id}/molcmp/', opts, follow=True)
    assert resp.status_code == 200
    num_ds_after = DrugSet.objects.all().count()
    assert num_ds_before + 1 == num_ds_after
    last_ds = DrugSet.objects.all()[num_ds_after-1]
    assert last_ds.name == 'My New Drugset'

    from drugs.models import Drug
    d = Drug.objects.all()[0]

    wsa1 = WsAnnotation.objects.create(ws=ws, agent=d)
    wsa2 = WsAnnotation.objects.create(ws=ws, agent=d)

    assert last_ds.drugs.count() == 0
    opts = {
            'add_btn': True,
            'ds': f'ds{last_ds.id}',
            f'check_{wsa1.id}': 'on',
            f'check_{wsa2.id}': 'off',
            }
    resp = auth_client.post(f'/mol/{ws.id}/molcmp/', opts, follow=True)
    assert resp.status_code == 200

    assert ws.get_wsa_id_set(f'ds{last_ds.id}') == {wsa1.id}
