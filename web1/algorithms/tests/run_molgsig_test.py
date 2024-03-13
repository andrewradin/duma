from builtins import range
import os
import pytest
from mock import patch, Mock, MagicMock

from dtk.tests import auth_client, mock_dpi, mock_ppi, tmpdir_path_helper, make_ws, make_score_job
from dtk.tests.mock_remote import local_stdjobinfo
from dtk.tests.end_to_end_test import setup_lts_tmpdir, wait_for_background_job, shim_background_jobs_in_process, assert_file_unchanged



@patch('dtk.lts.LtsRepo')
def test_run_selectivity(lts_repo, # We write to this, so mock it out
                         tmpdir_path_helper, # Write out to a tmpdir
                         auth_client, # Make http requests (pre-auth'd)
                         make_ws, # We need a WS with drugs
                         mock_dpi, # We need some fake DPI
                         mock_ppi, # We need some fake PPI
                         make_score_job, # Populate an input cc_gesig job
                         local_stdjobinfo, # Don't run on a real remote machine
                         live_server, # Allows cross-thread to see DB changes
                         ):

    # Setup our initial workspace attributes
    ws_attrs = []
    for i in range(1, 6):
        ws_attrs += [('DB0%d' % i, 'canonical', 'Drug%d' % i),
                    ('DB0%d' % i, 'm_dpimerge_id', 'DB0%d' % i),
                    ('DB0%d' % i, 'lincs_id', 'DB0%d' % i)]
    ws = make_ws(ws_attrs)


    # This depends on hardcoded lincs collection.
    from drugs.models import Collection
    coll = Collection.objects.all()[0]
    coll.name = 'lincs.full'
    coll.save()
    

    # Make sure LTS writes to the tmpdir
    setup_lts_tmpdir()

    # Make our background job run in-process (with all our mocks in place)
    shim_background_jobs_in_process()

    # Mock out DPI
    dpi = [
        ('dpimerge_id', 'uniprot_id', 'evidence', 'direction'),
        ('DB01', 'P01', '0.5', '0'),
        ('DB01', 'P02', '0.9', '0'),
        ('DB02', 'P02', '1.0', '0'),
        ('DB03', 'P02', '0.5', '0'),
        ('DB04', 'P02', '0.5', '0'),
        ('DB04', 'P04', '0.5', '0'),
        ('DB05', 'P04', '0.5', '0'),
        ]
    mock_dpi('fake_dpi', dpi)

    # Mock out PPI
    ppi = [
        ('uniprot1', 'uniprot2', 'evidence', 'direction'),
        ('P01', 'P01', '0.5', '0'),
        ('P02', 'P02', '0.5', '0'),
        ('P01', 'P02', '0.1', '0'),
        ('P03', 'P02', '0.5', '0'),
        ('P09', 'P02', '1.0', '0'),
        ('P01', 'P04', '0.5', '0'),
        ('P08', 'P04', '0.9', '0'),
        ]
    mock_ppi('fake_ppi', ppi)
    

    header = ['uniprot', 'ev', 'fold', 'tisscnt', 'avgDir']
    rows = [
        ('P01', '0.1', '0.2', '0', '0'),
        ('P02', '0.3', '0.4', '0', '0'),
        ('P04', '1.0', '-0.4', '0', '0'),
    ]


    input_jid = make_score_job(ws, 'gesig', 'cc_gesig', 'signature_file', header, rows)

    import numpy as np
    expr_data = np.array([
        [1., -1.],
        [2., 0.1],
        [0.2, 0.0],
    ])
    metadata = {
        'genes': {
            'order': ['G01', 'G04'],
            'info': {'uniprot':{ 'G01': 'P01', 'G04': 'P04'}}
        },
        'drugs': {
            'order': ['DB01', 'DB02', 'DB05'],
        }
    }


    # Just mock out the choices to point at our input job.
    with patch('dtk.scores.JobCodeSelector.get_choices', autospec=True) as get_choices, \
         patch('algorithms.run_molgsig.load_mol_sigs') as load_mol_sigs:

        get_choices.return_value = [(f'{input_jid}_ev', 'Test CC GESIG')]
        load_mol_sigs.return_value = (expr_data, metadata)

        # Open up the job_start page, make sure it loads.
        url = '/cv/{}/job_start/molgsig_{}/'.format((ws.id), (ws.id))
        resp = auth_client.get(url)
        assert resp.status_code == 200


        # Parse the submit form out of the page.
        from lxml import html
        from lxml.html import submit_form
        dom = html.fromstring(resp.content)

        def http_fn(method, form_url, values):
            print(("Calling http", method, url, values))
            assert method.lower() == 'post'
            assert form_url == None # This means same as origin page
            # lxml's form data doesn't include the button being pressed.
            auth_client.post(url, dict(values, run_btn=True))
        
        # Find the form we want.
        form = [x for x in dom.forms if x.get_element_by_id('id_input_score', None) != None]
        assert len(form) == 1

        # Submit it.
        submit_form(form[0], open_http=http_fn)
    
        # Wait until the job completes.
        p_id = wait_for_background_job(expect_success=True, timeout_secs=60)

    # Check that we got an output file.
    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws.id, p_id)
    assert os.path.exists(bji.outfile)

    assert_file_unchanged(bji.outfile, prefix='molgsig.basic')
