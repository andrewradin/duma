

from dtk.tests import make_ws, auth_client

def test_noneff_assay(db, make_ws, auth_client):
    ws_attrs = []
    i = 1

    # Note that CHEMBL98 (below) is a specific ID for Vorinostat which happens
    # to have a lot of assays.
    ws_attrs += [('DB0%d' % i, 'canonical', 'Drug%d' % i),
                ('DB0%d' % i, 'm_dpimerge_id', 'DB0%d' % i),
                ('DB0%d' % i, 'chembl_id', 'CHEMBL98')]
    

    ws = make_ws(ws_attrs)

    from browse.models import WsAnnotation
    wsa = WsAnnotation.objects.all()[0]

    url = f'/mol/{ws.id}/noneff_assays/{wsa.id}/'
    resp = auth_client.get(url)
    assert resp.status_code == 200

    # Just check for a couple of strings that should show up in there.
    # Really just exercising the code rather than testing for specific results.
    txt = resp.content
    assert b'Volume of distribution' in txt
    assert b'Thermodynamic solubility' in txt
