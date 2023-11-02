

from dtk.tests import auth_client, make_ws
from dtk.tests.end_to_end_test import assert_good_response

def test_audit_view(make_ws, auth_client):
    from browse.models import WsAnnotation, DispositionAudit
    ws_attrs = []
    for i in range(1, 8):
        ws_attrs += [('DB0%d' % i, 'canonical', 'Drug%d' % i),
                    ('DB0%d' % i, 'm_dpimerge_id', 'DB0%d' % i)]

    ws = make_ws(ws_attrs, name='Main Workspace')

    wsa = next(iter(WsAnnotation.objects.all()))

    # Check after nothing has happened to the molecule.
    # This page will be mostly blank, just make sure it doesn't crash.
    url = f'/mol/{ws.id}/dispositionaudit/{wsa.id}/'
    rsp = auth_client.get(url)
    assert_good_response(rsp)


    # Update the molecule and check again
    ivals = WsAnnotation.indication_vals
    wsa.update_indication(ivals.INITIAL_PREDICTION)

    rsp = auth_client.get(url)
    assert_good_response(rsp)
    assert 'Initial Prediction' in rsp.content.decode('utf8')
    assert 'Ignore' in rsp.content.decode('utf8')

    das = list(DispositionAudit.objects.all())
    assert len(das) == 1


    assert not das[0].ignore

    # Ignore the update and check again
    auth_client.post(url, {"ignore_btn": True, "da_id": das[0].id})

    das = list(DispositionAudit.objects.all())
    assert len(das) == 0

    das_with_ignore = list(DispositionAudit.all_objects.all())
    assert len(das_with_ignore) == 1
    assert das_with_ignore[0].ignore

    rsp = auth_client.get(url)
    assert_good_response(rsp)
    assert 'Initial Prediction' in rsp.content.decode('utf8')
    assert 'Unignore' in rsp.content.decode('utf8')

    # UnIgnore the update and check again
    auth_client.post(url, {"unignore_btn": True, "da_id": das_with_ignore[0].id})

    das = list(DispositionAudit.objects.all())
    assert len(das) == 1
