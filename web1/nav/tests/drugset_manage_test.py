

from builtins import range
import pytest
from mock import patch

from dtk.tests.end_to_end_test import setup_users
from dtk.tests.ws_with_attrs import ws_with_attrs

ws_attrs = []
for i in range(1, 6):
    ws_attrs += [('DB0%d' % i, 'canonical', 'Drug%d' % i)]

@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
def test_view(client, django_user_model, ws_with_attrs):
    from browse.models import WsAnnotation, Workspace
    setup_users(django_user_model, client)
    ws = Workspace.objects.all()[0]

    assert client.get('/cv/%d/ds/' % ws.id).status_code == 200


@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
def test_create(client, django_user_model, ws_with_attrs):
    from browse.models import WsAnnotation, Workspace
    setup_users(django_user_model, client)
    ws = Workspace.objects.all()[0]

    data = {
        'name': 'my_ds',
        'drugs': '\n'.join(["Drug1", "Drug3"]),
        'create_btn': True,
        }

    from browse.models import DrugSet
    assert len(DrugSet.objects.all()) == 0
    assert client.post('/cv/%d/ds/' % ws.id, data, follow=True).status_code == 200
    assert len(DrugSet.objects.all()) == 1
    ds = DrugSet.objects.all()[0]
    assert ds.name == 'my_ds'
    assert len(ds.drugs.all()) == 2

    # Now make sure we can still view it with our new drugset.
    assert client.get('/cv/%d/ds/' % ws.id).status_code == 200


@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
def test_preload(client, django_user_model, ws_with_attrs):
    from browse.models import WsAnnotation, Workspace
    setup_users(django_user_model, client)
    ws = Workspace.objects.all()[0]

    # Nothing is actually marked as a known treatment here, but most of the
    # code runs anyway.
    data = {
        'ind': '16',
        'preload_btn': True,
        }

    from browse.models import DrugSet
    assert len(DrugSet.objects.all()) == 0
    assert client.post('/cv/%d/ds/' % ws.id, data, follow=True).status_code == 200
    assert len(DrugSet.objects.all()) == 0

