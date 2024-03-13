from __future__ import print_function
# -*- coding: utf-8 -*-

import os
from dtk.tests import tmpdir_path_helper, make_ws, auth_client
from dtk.tests.end_to_end_test import setup_lts_tmpdir, wait_for_background_job, shim_background_jobs_in_process, assert_file_unchanged
from mock import patch

# TODO: Do this as a different version once we sort out workspace defaults.
ot_dummy_version=1
ot_dummy_version_choice='v%d'%ot_dummy_version
def ot_cache_path():
    file_class='openTargets'
    from dtk.s3_cache import S3Bucket
    s3b = S3Bucket(file_class)
    return s3b.cache_path
def ot_filename(role,fmt):
    file_class='openTargets'
    from dtk.files import VersionedFileName
    vfn=VersionedFileName(file_class=file_class)
    vfn.version=ot_dummy_version
    vfn.role=role
    vfn.format=fmt
    return os.path.join(ot_cache_path(),vfn.to_string())
def ot_data_filename():
    return ot_filename('data','tsv.gz')
def ot_names_filename():
    return ot_filename('names','tsv')

@patch('dtk.lts.LtsRepo')
def test_otarg(lts_repo, # We write to this, so mock it out
               tmpdir_path_helper, # Write out to a tmpdir
               auth_client, # Make http requests (pre-auth'd)
               make_ws,
               live_server, # Allows cross-thread to see DB changes
        ):
    ws = make_ws([])
    setup_lts_tmpdir()
    shim_background_jobs_in_process()

    from path_helper import PathHelper,make_directory
    make_directory(ot_cache_path())
    import gzip
    with gzip.open(ot_data_filename(), 'wt') as f:
        data = [
            ('disease_key', 'target_name', 'score_name', 'score_value'),
            ('EFO_001', 'P0001', 'literature', '0.1'),
            ('EFO_001', 'P0001', 'known_drug', '0.2'),
            ('EFO_001', 'P0002', 'animal_model', '0.3'),
            ('EFO_002', 'P0001', 'literature', '0.45'),
            ('EFO_002', 'P0002', 'animal_model', '0.15'),
            ('EFO_002', 'P0003', 'literature', '0.9'),
            ]
        f.write('\n'.join(['\t'.join(row) for row in data]))

    with open(ot_names_filename(), 'w') as f:
        data = [
            ('Skin Rash', 'EFO_001', '1'),
            ('Tiredness', 'EFO_002', '1'),
            ]
        f.write('\n'.join(['\t'.join(row) for row in data]))

    def run():
        url = '/cv/{}/job_start/otarg_{}/'.format(ws.id, ws.id)
        resp = auth_client.get(url)
        assert resp.status_code == 200

        # Parse the submit form out of the page.
        from lxml import html
        from lxml.html import submit_form
        dom = html.fromstring(resp.content)

        def http_fn(method, form_url, values):
            print("Calling http", method, url, values)
            assert method.lower() == 'post'
            assert form_url == None # This means same as origin page
            # lxml's form data doesn't include the button being pressed.
            resp = auth_client.post(url, dict(values, run_btn=True))
            assert resp.status_code == 302, "Should have forwarded to progress"

        
        # Find the form we want.
        form = [x for x in dom.forms if x.get_element_by_id('id_disease', None) != None]
        assert len(form) == 1
        # Submit it.
        submit_form(form[0], open_http=http_fn)
        
        # Wait until the job completes.
        p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)

        # Check that we got an output file.
        from runner.process_info import JobInfo
        bji = JobInfo.get_bound(ws.id, p_id)
        return bji


    ws.set_disease_default('OpenTargets', 'key:EFO_001', '')
    bji = run()
    assert os.path.exists(bji.score_fn)

    from dtk.files import get_file_records
    records = list(get_file_records(bji.score_fn, keep_header=False))
    # uniprot, lit, knowndrug, animal
    assert records[0] == ['P0001', '0.1', '0.2', '0']
    assert records[1] == ['P0002', '0', '0', '0.3']


    ws.set_disease_default('OpenTargets', 'key:EFO_001,Tiredness', '')
    bji = run()
    assert os.path.exists(bji.score_fn)
    from dtk.files import get_file_records
    records = list(get_file_records(bji.score_fn, keep_header=False))
    records.sort()
    # uniprot, lit, knowndrug, animal
    assert records[0] == ['P0001', '0.45', '0.2', '0']
    assert records[1] == ['P0002', '0', '0', '0.3']
    assert records[2] == ['P0003', '0.9', '0', '0']



def test_get_disease_key(tmpdir_path_helper):

    from path_helper import PathHelper,make_directory
    make_directory(ot_cache_path())

    assert PathHelper.s3_cache_root.startswith(PathHelper.get_config()['bigtmp'])

    # Write this file just so that we don't fetch the real one.
    import gzip
    with gzip.open(ot_data_filename(), 'wt') as f:
        data = [
            ]
        f.write('\n'.join(['\t'.join(row) for row in data]))

    with open(ot_names_filename(), 'wb') as f:
        print("Write data")
        data = [
            ('Skin Rash', 'EFO_001', '1'),
            ('Tiredness', 'EFO_002', '1'),
            (u'sjögren', 'EFO_003', '2'),
            ]
        f.write(('\n'.join(['\t'.join(row) for row in data])).encode('utf8'))
    print("Load data")
    from dtk.open_targets import OpenTargets
    ot = OpenTargets(ot_dummy_version_choice)
    assert ot.get_disease_key(u'sjögren') == 'EFO_003'
