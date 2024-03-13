from builtins import range
import os
import pytest
from mock import patch, Mock, MagicMock

from dtk.tests import auth_client, tmpdir_path_helper, ws_with_attrs
from dtk.tests.end_to_end_test import setup_lts_tmpdir, wait_for_background_job, shim_background_jobs_in_process, assert_file_unchanged

# Setup our initial workspace attributes
ws_attrs = []
for i in range(1, 6):
    ws_attrs += [('DB0%d' % i, 'canonical', 'Drug%d' % i),
                 ('DB0%d' % i, 'm_dpimerge_id', 'DB0%d' % i)]

from dtk.s3_cache import S3MiscBucket

import six
@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
@patch.object(S3MiscBucket, 'list')
@patch('dtk.lts.LtsRepo')
def test_run_tcga(lts_repo, # We write to this, so mock it out
                  s3bucket_list,
                  tmpdir_path_helper, # Write out to a tmpdir
                  auth_client, # Make http requests (pre-auth'd)
                  ws_with_attrs, # We need a WS with drugs
                  live_server, # Allows cross-thread to see DB changes
                  ):


    # More convenient alias
    ws = ws_with_attrs

    # Needs to match this format to be picked up.
    tcga_fn = 'tcgamut.{}.mock.tsv'.format((ws.id))
    s3bucket_list.return_value = [tcga_fn]

    from dtk.s3_cache import S3File, S3MiscBucket
    s3_file = S3File(S3MiscBucket(),tcga_fn)
    s3_fpath = s3_file.path()
    assert 'tmp' in s3_fpath, "We should be in a tmpdir"
    with open(s3_fpath, 'w') as f:
        f.write("""Symbol	Name	Cytoband	Type	# Affected Cases in TCGA-DLBC	# Affected Cases Across the GDC	# Mutations	Annotations	Survival
NFKBIA	nuclear factor of kappa light polypeptide gene enhancer in B-cells inhibitor, alpha	14q13.2	protein_coding	5 / 37 (13.51%)	89 / 10,188	7		
""")

    # aliasing needs to be in protein db tables
    from browse.models import Protein
    prot = Protein(uniprot='P25963',gene='NFKBIA',uniprot_kb='IKBA')
    prot.save()

    # Make sure LTS writes to the tmpdir
    setup_lts_tmpdir()

    # Make our background job run in-process (with all our mocks in place)
    shim_background_jobs_in_process()
    

    # Open up the job_start page, make sure it loads.
    url = '/cv/{}/job_start/tcgamut_{}/'.format((ws.id), (ws.id))
    resp = auth_client.get(url)
    assert resp.status_code == 200

    opts = {
        'run_btn': True,
        'input_file': tcga_fn,
            }
    rsp = auth_client.post(url, opts)
    assert rsp.status_code == 302, \
            "Should redirect to progress page, unless there was an error"
    
    # Wait until the job completes.
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)

    # Check that we got an output file.
    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws.id, p_id)
    assert os.path.exists(bji.score_fn)
    print("Output scores", open(bji.score_fn).read())

    assert_file_unchanged(bji.score_fn, prefix='tcgatest.')

