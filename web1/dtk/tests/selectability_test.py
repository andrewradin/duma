from dtk.tests.ws_with_attrs import make_ws
from dtk.tests import mock_dpi, mock_ppi, tmpdir_path_helper, auth_client
from dtk.tests.end_to_end_test import setup_lts_tmpdir, wait_for_background_job, shim_background_jobs_in_process, assert_file_unchanged, assert_good_response


import logging
logger = logging.getLogger(__name__)

# Need to setup at least two workspaces with a lot of different features.
# Test building model
# Test running model and generating a score
# Test viewing model data


def setup_workspace(ws):
    from browse.models import WsAnnotation
    wsas = ws.wsannotation_set.all()
    ivals = WsAnnotation.indication_vals
    # Flag some as hits, some as inactive.
    for i, wsa in enumerate(wsas):
        if i % 3 == 0:
            wsa.update_indication(ivals.INITIAL_PREDICTION)
        elif i % 3 == 1:
            wsa.update_indication(ivals.HIT)
        else:
            wsa.update_indication(ivals.INACTIVE_PREDICTION, demerits='1')

    wsas[0].update_indication(ivals.KNOWN_TREATMENT, href='http://twoxar.com')

    # Setup novelty
    # We need LBN data, which is somewhat trickier.
    from dtk.tests.mock_job import make_mock_job
    lbn_header = ('wsa', 'lbnOR', 'lbnP', 'disPorWDrug', 'drugPorWDis', 'targLogOdds', 'targPortion', )
    make_mock_job('lbn', ws,
            [lbn_header] + [(wsa.id,)*len(lbn_header) for wsa in wsas])

    # Setup selectivity
    # This runs real fast because ppi/dpi are mocked out to be tiny.
    from scripts.run_job import run
    run('unit-test-username', 'selectivity',settings=None, output=None, ws_ids=[ws.id])
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)

    # Assert that we're all setup.
    from dtk.selectability import INPUT_FEATURESETS
    for fs in INPUT_FEATURESETS:
        assert fs.available(ws), "All features should now be available"

from mock import patch
@patch('dtk.lts.LtsRepo')
def test_all(
        lts_repo,
        tmpdir_path_helper,
        make_ws,
        mock_dpi,
        mock_ppi,
        auth_client,
        live_server,
        ):
    setup_lts_tmpdir()
    shim_background_jobs_in_process()
    ws_attrs = []
    for i in range(1, 8):
        ws_attrs += [('DB0%d' % i, 'canonical', 'Drug%d' % i),
                    ('DB0%d' % i, 'm_dpimerge_id', 'DB0%d' % i)]

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
    ppi = [
        ('prot1', 'prot2', 'evidence', 'direction'),
        ('P01', 'P01', '0.5', '0'),
        ('P02', 'P02', '0.5', '0'),
        ('P01', 'P02', '0.1', '0'),
        ('P03', 'P02', '0.5', '0'),
        ('P09', 'P02', '1.0', '0'),
        ('P01', 'P04', '0.5', '0'),
        ('P08', 'P04', '0.9', '0'),
        ]
    mock_ppi('fake_ppi', ppi)

    for _ in range(3):
        ws = make_ws(ws_attrs)
        setup_workspace(ws)


    from browse.models import Demerit
    demerits = [
            'Data Quality',
            'Ubiquitous',
            'Tox',
            'Unavailable',
            ]
    for demerit in demerits:
        Demerit.objects.create(
                desc=demerit
                )

    logger.info("Running selectabilitymodel now")
    from scripts.run_job import run
    run('unit-test-username', 'selectabilitymodel',settings=None, output=None, ws_ids=[ws.id])
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)

    rsp = auth_client.get(f'/selectabilitymodel/{p_id}/', follow=True)
    assert_good_response(rsp)

    rsp = auth_client.get(f'/selectabilityfeatureplot/?featuresets=NameFeatureSet,CMRankFeatureSet&inds=tts&ws={ws.id}', follow=True)
    assert_good_response(rsp)

    from dtk.selectability import WsaWzsSource
    with patch.object(WsaWzsSource, 'get_wsas', autospec=True) as get_wsas:
        logger.info("Running selectability now")
        from browse.models import WsAnnotation
        get_wsas.return_value = WsAnnotation.objects.filter(ws_id=ws.id)
        run('unit-test-username', 'selectability',settings={"model_jid": p_id}, output=None, ws_ids=[ws.id])
        p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)
    


