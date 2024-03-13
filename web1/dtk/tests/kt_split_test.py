
from builtins import range
from mock import patch
from dtk.tests.ws_with_attrs import ws_with_attrs
from dtk.kt_split import *
import pytest

from dtk.tests import mock_dpi


ws_attrs = []
for i in range(1, 6):
    ws_attrs += [('DB0%d' % i, 'canonical', 'Drug%d' % i),
                 ('DB0%d' % i, 'm_dpimerge_id', 'DB0%d' % i)]
@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
@patch('browse.models.Workspace.get_wsa_id_set')
@patch('browse.models.Workspace.get_dpi_default')
def test_kt_split_all_cluster(get_dpi_default, get_wsa_id_set, ws_with_attrs, mock_dpi):
    from browse.models import WsAnnotation
    wsas = WsAnnotation.objects.all()
    get_wsa_id_set.return_value = set(wsas.values_list('id', flat=True))
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
    mock_dpi('dpimerge.dbank', dpi)
    get_dpi_default.return_value = 'dpimerge.dbank'

    # This set of drugs all clusters together, should still get 2 clusters, one will be empty
    split = get_split_drugset('split-test-kts', ws_with_attrs)


@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
@patch('browse.models.Workspace.get_wsa_id_set')
@patch('browse.models.Workspace.get_dpi_default')
def test_kt_split_2_cluster(get_dpi_default, get_wsa_id_set, ws_with_attrs, mock_dpi):
    from browse.models import WsAnnotation
    wsas = WsAnnotation.objects.all()
    get_wsa_id_set.return_value = set(wsas.values_list('id', flat=True))
    dpi = [
        ('dpimerge_id', 'uniprot_id', 'evidence', 'direction'),
        ('DB01', 'P01', '0.5', '0'),
        ('DB01', 'P02', '0.5', '0'),
        ('DB02', 'P02', '0.5', '0'),
        ('DB03', 'P02', '0.5', '0'),
        ('DB04', 'P02', '0.5', '0'),
        ('DB05', 'P04', '0.5', '0'),
        ]
    mock_dpi('dpimerge.dbank', dpi)
    get_dpi_default.return_value = 'dpimerge.dbank'

    ds_1 = get_split_drugset('split-train-kts', ws_with_attrs)
    ds_2 = get_split_drugset('split-test-kts', ws_with_attrs)
    agents_1 = [WsAnnotation.objects.get(pk=x).agent.get_key() for x in ds_1]
    agents_2 = [WsAnnotation.objects.get(pk=x).agent.get_key() for x in ds_2]
    assert sorted(agents_1) == ['DB01', 'DB02', 'DB03', 'DB04']
    assert agents_2 == ['DB05']
    

@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
@patch('browse.models.Workspace.get_wsa_id_set')
@patch('browse.models.Workspace.get_dpi_default')
def test_merge_cluster(get_dpi_default, get_wsa_id_set, ws_with_attrs, mock_dpi):
    from browse.models import WsAnnotation
    wsas = WsAnnotation.objects.all()
    get_wsa_id_set.return_value = set(wsas.values_list('id', flat=True))
    dpi = [
        ('dpimerge_id', 'uniprot_id', 'evidence', 'direction'),
        ('DB01', 'P01', '0.5', '0'),
        ('DB02', 'P02', '0.5', '0'),
        ('DB03', 'P03', '0.5', '0'),
        ('DB04', 'P04', '0.5', '0'),
        ('DB05', 'P05', '0.5', '0'),
        ]
    mock_dpi('dpimerge.dbank', dpi)
    get_dpi_default.return_value = 'dpimerge.dbank'

    ds_1 = get_split_drugset('split-train-kts', ws_with_attrs)
    ds_2 = get_split_drugset('split-test-kts', ws_with_attrs)
    agents_1 = [WsAnnotation.objects.get(pk=x).agent.get_key() for x in ds_1]
    agents_2 = [WsAnnotation.objects.get(pk=x).agent.get_key() for x in ds_2]
    assert sorted(agents_1) == ['DB02', 'DB04']
    assert sorted(agents_2) == ['DB01', 'DB03', 'DB05']

    ktsplit = KtSplit(ws_with_attrs, 'split-train-kts', 1)
    # Just make sure it doesn't crash
    ktsplit.clusters[0].dpi_table_str()

@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
@patch('browse.models.Workspace.get_wsa_id_set')
@patch('browse.models.Workspace.get_dpi_default')
def test_edit_split(get_dpi_default, get_wsa_id_set, ws_with_attrs, mock_dpi,
                    client, django_user_model):
    from .end_to_end_test import setup_users
    setup_users(django_user_model, client)

    from browse.models import WsAnnotation
    wsas = WsAnnotation.objects.all()
    get_wsa_id_set.return_value = set(wsas.values_list('id', flat=True))
    dpi = [
        ('dpimerge_id', 'uniprot_id', 'evidence', 'direction'),
        ('DB01', 'P01', '0.5', '0'),
        ('DB02', 'P02', '0.5', '0'),
        ('DB03', 'P03', '0.5', '0'),
        ('DB04', 'P04', '0.5', '0'),
        ('DB05', 'P05', '0.5', '0'),
        ]
    mock_dpi('dpimerge.dbank', dpi)
    get_dpi_default.return_value = 'dpimerge.dbank'

    wsid = ws_with_attrs.id
    url = '/%s/drugset/' % wsid
    assert client.get(url).status_code == 200
    kts_url = url + "?drugset=kts"
    assert client.get(kts_url).status_code == 200

    # Check that we start with expected clusters (see previous test).
    ds_1 = get_split_drugset('split-train-kts', ws_with_attrs)
    ds_2 = get_split_drugset('split-test-kts', ws_with_attrs)
    agents_1 = [WsAnnotation.objects.get(pk=x).agent.get_key() for x in ds_1]
    agents_2 = [WsAnnotation.objects.get(pk=x).agent.get_key() for x in ds_2]
    assert sorted(agents_1) == ['DB02', 'DB04']
    assert sorted(agents_2) == ['DB01', 'DB03', 'DB05']
    
    # Move a bunch around
    ds_1l = list(sorted(ds_1))
    ds_2l = list(sorted(ds_2))
    opts = {'change_test_train_btn': True,
            'train_%d' % ds_1l[0]: 'on',
            'train_%d' % ds_1l[1]: 'on',
            'test_%d' % ds_2l[1]: 'on',
            }
    assert client.post(kts_url, opts).status_code == 200

    # Check that we've gotten the expected new clusters.
    ds_1 = get_split_drugset('split-train-kts', ws_with_attrs)
    ds_2 = get_split_drugset('split-test-kts', ws_with_attrs)
    agents_1 = [WsAnnotation.objects.get(pk=x).agent.get_key() for x in ds_1]
    agents_2 = [WsAnnotation.objects.get(pk=x).agent.get_key() for x in ds_2]
    assert sorted(agents_1) == ['DB03']
    assert sorted(agents_2) == ['DB01', 'DB02', 'DB04', 'DB05']

    # Rerun autosplit, we should be back to our original
    opts = {'autosplit_btn': True, }
    assert client.post(kts_url, opts).status_code == 200
    ds_1 = get_split_drugset('split-train-kts', ws_with_attrs)
    ds_2 = get_split_drugset('split-test-kts', ws_with_attrs)
    agents_1 = [WsAnnotation.objects.get(pk=x).agent.get_key() for x in ds_1]
    agents_2 = [WsAnnotation.objects.get(pk=x).agent.get_key() for x in ds_2]
    assert sorted(agents_1) == ['DB02', 'DB04']
    assert sorted(agents_2) == ['DB01', 'DB03', 'DB05']

@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
@patch('browse.models.Workspace.get_wsa_id_set')
@patch('browse.models.Workspace.get_dpi_default')
def test_add_kt(get_dpi_default, get_wsa_id_set, ws_with_attrs, mock_dpi,
                    client, django_user_model):
    from .end_to_end_test import setup_users
    setup_users(django_user_model, client)

    from browse.models import WsAnnotation
    wsas = WsAnnotation.objects.all()
    get_wsa_id_set.return_value = set(wsas.values_list('id', flat=True)[:3])
    dpi = [
        ('dpimerge_id', 'uniprot_id', 'evidence', 'direction'),
        ('DB01', 'P01', '0.5', '0'),
        ('DB02', 'P02', '0.5', '0'),
        ('DB03', 'P03', '0.5', '0'),
        ('DB04', 'P01', '0.5', '0'),
        ('DB05', 'P05', '0.5', '0'),
        ]
    mock_dpi('dpimerge.dbank', dpi)
    get_dpi_default.return_value = 'dpimerge.dbank'

    # We start with 3 KTs, make sure they cluster.
    ds_1 = get_split_drugset('split-train-kts', ws_with_attrs)
    ds_2 = get_split_drugset('split-test-kts', ws_with_attrs)
    agents_1 = [WsAnnotation.objects.get(pk=x).agent.get_key() for x in ds_1]
    agents_2 = [WsAnnotation.objects.get(pk=x).agent.get_key() for x in ds_2]
    assert sorted(agents_1) == ['DB02']
    assert sorted(agents_2) == ['DB01', 'DB03']

    # Add in a new KT, we should redo all the clustering.
    get_wsa_id_set.return_value = set(wsas.values_list('id', flat=True)[:4])
    ds_1 = get_split_drugset('split-train-kts', ws_with_attrs)
    ds_2 = get_split_drugset('split-test-kts', ws_with_attrs)
    agents_1 = [WsAnnotation.objects.get(pk=x).agent.get_key() for x in ds_1]
    agents_2 = [WsAnnotation.objects.get(pk=x).agent.get_key() for x in ds_2]
    assert sorted(agents_1) == ['DB03']
    assert sorted(agents_2) == ['DB01', 'DB02', 'DB04']

    # Edit by hand, verify new KTs all go to test.
    kt = EditableKtSplit(ws_with_attrs, 'kts')
    # Move everything to test.
    kt.modify_split(wsas_to_test=ds_1, wsas_to_train=[])
    get_wsa_id_set.return_value = set(wsas.values_list('id', flat=True)[:5])
    ds_1 = get_split_drugset('split-train-kts', ws_with_attrs)
    ds_2 = get_split_drugset('split-test-kts', ws_with_attrs)
    agents_1 = [WsAnnotation.objects.get(pk=x).agent.get_key() for x in ds_1]
    agents_2 = [WsAnnotation.objects.get(pk=x).agent.get_key() for x in ds_2]
    assert sorted(agents_1) == []
    assert sorted(agents_2) == ['DB01', 'DB02', 'DB03', 'DB04', 'DB05']


def test_complement():
    compl = lambda x: parse_split_drugset_name(x).complement_drugset
    
    assert compl('split-test-blah') == 'split-train-blah'
    assert compl('split-train-blah') == 'split-test-blah'

