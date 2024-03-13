from builtins import range
import os
import pytest
from mock import patch, Mock, MagicMock

from dtk.tests import auth_client, mock_dpi, mock_ppi, tmpdir_path_helper, ws_with_attrs
from dtk.tests.end_to_end_test import setup_lts_tmpdir, wait_for_background_job, shim_background_jobs_in_process, assert_file_unchanged

# Setup our initial workspace attributes
ws_attrs = []
for i in range(1, 6):
    ws_attrs += [('DB0%d' % i, 'canonical', 'Drug%d' % i),
                 ('DB0%d' % i, 'm_dpimerge_id', 'DB0%d' % i)]


import six
@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
@patch('dtk.lts.LtsRepo')
def test_run_selectivity(lts_repo, # We write to this, so mock it out
                         tmpdir_path_helper, # Write out to a tmpdir
                         auth_client, # Make http requests (pre-auth'd)
                         ws_with_attrs, # We need a WS with drugs
                         mock_dpi, # We need some fake DPI
                         mock_ppi, # We need some fake PPI
                         live_server, # Allows cross-thread to see DB changes
                         ):
    # More convenient alias
    ws = ws_with_attrs

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
    

    # Open up the job_start page, make sure it loads.
    url = '/cv/{}/job_start/selectivity_{}/'.format((ws.id), (ws.id))
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
    form = [x for x in dom.forms if x.get_element_by_id('id_dpi_file', None) != None]
    assert len(form) == 1
    # Submit it.
    submit_form(form[0], open_http=http_fn)
    
    # Wait until the job completes.
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)

    # Check that we got an output file.
    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws.id, p_id)
    assert os.path.exists(bji.outfile)

    assert_file_unchanged(bji.outfile, prefix='seltest.')
