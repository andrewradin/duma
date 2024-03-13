from builtins import range
import os
import pytest
from mock import patch, Mock, MagicMock

from dtk.tests import auth_client, make_ws, tmpdir_path_helper, ws_with_attrs, mock_remote_machine
from dtk.tests.end_to_end_test import setup_lts_tmpdir, wait_for_background_job, shim_background_jobs_in_process, assert_file_unchanged

def setup_fake_tissues(ws):
    # This ensures the default tissue sets have been created.
    ws.get_tissue_sets()

    from browse.models import Tissue, TissueSet, AeAccession, AeSearch, AeScore, AeDisposition
    from django.utils import timezone
    ts = TissueSet.objects.get(name='default', ws=ws)

    srch = AeSearch.objects.create(
        ws=ws,
        term='Psoriasis',
        version=AeSearch.LATEST_VERSION,
        when=timezone.now(),
    )

    def make_example(reject, common_text):
        idx = len(AeAccession.objects.all())
        geoID = f'T{idx}-WS{ws.id}'
        if not reject:
            # Rejects shouldn't have tissues, those that do get excluded from the model.
            t = Tissue.objects.create(
                geoID=geoID,
                name=geoID,
                tissue_set=ts,
                ws=ws,
            )

        acc1 = AeAccession.objects.create(
            ws=ws,
            geoID=geoID,
            title='A good title of words ' + common_text,
            desc='A happy description, also of words ' + common_text,
            experiment_type='A good experiment ' + common_text,
            num_samples=2,
        )

        AeDisposition.objects.create(
            accession=acc1,
            rejected=reject
        )

        AeScore.objects.create(
            search=srch,
            accession=acc1,
            score=1, # Doesn't matter, just need for association.
        )

        # Attrs can't all be identical or all be unique (otherwise filtered out)
        samples = [
            {'attr1': common_text, 'attr2': 'Another attribute'},
            {'attr1': common_text, 'attr2': 'Another attribute'},
            {'attr1': common_text + " but different", 'attr2': 'Another attribute but different'},
        ]

        from ge.models import GESamples
        gesamples = GESamples.objects.create(geoID=acc1.geoID)
        gesamples.attrs_json_gz = GESamples.compress_sample_attrs(samples)
        gesamples.save()

        from scripts.gesearch_model import make_entry
        de = make_entry(acc1, srch, None)
        assert ''.join(de.samples_vals) != ''
        assert ''.join(de.samples_header) != ''

    for i in range(10):
        make_example(reject='', common_text='Great successful example')
        make_example(reject='bad', common_text='Failure example')



@patch('dtk.lts.LtsRepo')
def test_run_gesearchmodel(lts_repo, # We write to this, so mock it out
                         tmpdir_path_helper, # Write out to a tmpdir
                         auth_client, # Make http requests (pre-auth'd)
                         make_ws, # We need a WS with drugs
                         live_server, # Allows cross-thread to see DB changes
                         mock_remote_machine, # Runs 'remote machine' code locally
                         ):
    # More convenient alias
    ws1 = make_ws([])
    ws2 = make_ws([])
    ws3 = make_ws([])
    ws4 = make_ws([])
    # Make sure LTS writes to the tmpdir
    setup_lts_tmpdir()

    # Make our background job run in-process (with all our mocks in place)
    shim_background_jobs_in_process()


    # Do a bunch of setup to get this to work.
    setup_fake_tissues(ws1)
    setup_fake_tissues(ws2)
    setup_fake_tissues(ws3)
    setup_fake_tissues(ws4)

    # Open up the job_start page, make sure it loads.
    url = f'/cv/{ws1.id}/job_start/gesearchmodel/'
    resp = auth_client.get(url)
    assert resp.status_code == 200

    # Parse the submit form out of the page.
    from lxml import html
    from lxml.html import submit_form
    dom = html.fromstring(resp.content)

    def http_fn(method, form_url, values):
        print(("Calling http", method, url, values))
        workspaces = [str(ws.id) for ws in [ws1, ws2, ws3, ws4]]
        parms = dict(values, run_btn=True, kfold='2', hyperopt_iters='0', workspaces=workspaces)
        print("Parms set to ", parms)
        assert method.lower() == 'post'
        assert form_url == None # This means same as origin page
        # lxml's form data doesn't include the button being pressed.
        rsp = auth_client.post(url, parms)
        assert rsp.status_code == 302, "Probably had a form error, didn't get a redirect"
    
    # Find the form we want.
    form = [x for x in dom.forms if x.get_element_by_id('id_kfold', None) != None]
    assert len(form) == 1
    # Submit it.
    submit_form(form[0], open_http=http_fn)
    
    # Wait until the job completes.
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)

    # Check that we got an output file.
    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws1.id, p_id)
    assert os.path.exists(bji.outfile)


    from dtk.ae_search import score_search
    from browse.models import AeSearch, AeAccession
    srch = AeSearch.objects.all()[0]
    acc = AeAccession.objects.all()[0]

    # Check that we can score with this model
    new_score_model = bji.load_trained_model()
    score_search(srch, [acc], new_score_model)

    # Check that we can score with hardcoded model as well.
    from scripts.gesearch_model import PreviousModel
    old_score_model = PreviousModel({})
    score_search(srch, [acc], old_score_model)
    
    # Check the gesearchmodel results page for this model.
    url = f'/ge/{ws1.id}/search_model/{bji.job.id}/'
    resp = auth_client.get(url)
    assert resp.status_code == 200
    
