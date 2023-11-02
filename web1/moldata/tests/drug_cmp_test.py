from builtins import range
import pytest

from dtk.tests import auth_client, ws_with_attrs

ws_attrs = []
for i in range(1, 6):
    ws_attrs += [('DB0%d' % i, 'canonical', 'Drug%d' % i),
                 ('DB0%d' % i, 'm_dpimerge_id', 'DB0%d' % i),
                 ('DB0%d' % i, 'std_smiles', 'COO'),
                 ]
@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
def test_drug_cmp_get(auth_client, ws_with_attrs):
    from browse.models import WsAnnotation
    ws = ws_with_attrs
    wsas = WsAnnotation.objects.all()
    assert auth_client.get('/mol/{}/drugcmp/?ids={}'.format((ws.id), (wsas[0].id))).status_code == 200

@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
def test_drug_cmp_search(auth_client, ws_with_attrs):
    from browse.models import WsAnnotation
    ws = ws_with_attrs
    wsas = WsAnnotation.objects.all()
    opts = {
            'search_btn': True,
            'search': 'Drug4'
            }
    resp = auth_client.post('/mol/{}/drugcmp/?ids={}'.format((ws.id), (wsas[0].id)), opts, follow=True)
    assert resp.status_code == 200
    redirect_chain = resp.redirect_chain
    assert len(redirect_chain) == 2
    ids_before = redirect_chain[0][0].split('=')[1].split(',')
    ids_after = redirect_chain[1][0].split('=')[1].split(',')
    assert len(ids_before) == 1, "Originally had 1 id"
    assert len(ids_after) == 2, "Should have added another from search"


from dtk.tests.kt_split_test import mock_dpi
import six
@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
def test_drug_cmp_search_sim(auth_client, ws_with_attrs, mock_dpi):
    ws = ws_with_attrs
    dpi = [
        ('dpimerge_id', 'uniprot_id', 'evidence', 'direction'),
        ('DB01', 'P01', '0.5', '0'),
        ('DB01', 'P02', '0.5', '0'),
        ('DB02', 'P02', '0.5', '0'),
        ('DB03', 'P02', '0.5', '0'),
        ('DB04', 'P02', '0.5', '0'),
        ('DB04', 'P04', '0.5', '0'),
        ('DB05', 'P04', '0.5', '0'),
        ]
    mock_dpi(ws.get_dpi_default(), dpi)
    from browse.models import WsAnnotation
    wsas = WsAnnotation.objects.all()
    opts = {
            'find_btn': True,
            'dpi': ws.get_dpi_default(),
            'dpi_t':ws.get_dpi_thresh_default(),
            'ppi': ws.get_ppi_default(),
            'ppi_t': ws.get_ppi_thresh_default(),
            'max': 3,
            'prsim': 1.0,
            'dirJac': 1.0,
            'indJac': 1.0,
            'indigo': 1.0,
            'rdkit': 1.0,
            }
    print("Sending", opts)
    resp = auth_client.post('/mol/{}/drugcmp/?ids={}'.format((ws.id), (wsas[0].id)), opts, follow=True)
    assert resp.status_code == 200
    redirect_chain = resp.redirect_chain
    print("Check out", resp.redirect_chain)
    ids_before = redirect_chain[0][0].split('=')[1].split(',')
    assert len(ids_before) == 1, "Originally had 1 id"
    ids_after = redirect_chain[1][0].split('=')[1].split(',')
    assert len(ids_after) == 3, "Should have added to max, lots of same"
