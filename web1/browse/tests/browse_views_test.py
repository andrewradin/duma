
from dtk.tests import auth_client, make_ws
from dtk.tests.end_to_end_test import assert_good_response
from mock import patch

def test_etl_history(auth_client):
    # Pick a single page with relatively small datasets.
    # We also override it such that we only generate statsfiles for a small # of versions.
    with patch('dtk.etl.get_last_published_version') as lpv:
        lpv.return_value = 3
        rsp = auth_client.get("/etl_history/duma/")
        assert_good_response(rsp)

def test_workflow_update(auth_client, make_ws):
    ws = make_ws([])
    assert_good_response(auth_client.get(f'/{ws.id}/workflow/'))
    import json

    from stages import WorkspaceVDefaults
    assert WorkspaceVDefaults(ws).completion() == 0.0
    assert_good_response(auth_client.post(f'/{ws.id}/workflow/', {
        'update_btn': True,
        'query': json.dumps({'name': 'WorkspaceVDefaults', 'status': 'complete'}),
    }))
    assert WorkspaceVDefaults(ws).completion() == 1.0


    from notes.models import Note
    assert Note.get(ws, 'disease_note', None) == ''
    assert_good_response(auth_client.post(f'/{ws.id}/workflow/', {
        'update_note_btn': True,
        'note': "test note data",
    }, follow=True))
    ws.refresh_from_db()
    assert Note.get(ws, 'disease_note', 'unit-test-username') == 'test note data'



