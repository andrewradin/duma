from builtins import range
import os
import pytest
from mock import patch, Mock, MagicMock

from dtk.tests import auth_client, make_ws, tmpdir_path_helper, ws_with_attrs, mock_remote_machine, mock_dpi, mock_ppi, local_stdjobinfo, mock_pathways
from dtk.tests.end_to_end_test import setup_lts_tmpdir, wait_for_background_job, shim_background_jobs_in_process, assert_file_unchanged


def run_custom_sig(ws, auth_client):
    # Open up the job_start page, make sure it loads.
    url = f'/cv/{ws.id}/job_start/customsig_{ws.id}/'
    resp = auth_client.get(url)
    assert resp.status_code == 200

    # Parse the submit form out of the page.
    from lxml import html
    from lxml.html import submit_form
    from browse.models import Species
    dom = html.fromstring(resp.content)


    def http_fn(method, form_url, values):
        print(("Calling http", method, url, values))
        import json
        sig={"G01": 1.0}
        parms = dict(
            values,
            run_btn=True,
            sig_json=json.dumps(sig),
            description='A description',
            shortname='shortname',
            species=Species.MOUSE,
        )
        print("Parms set to ", parms)
        assert method.lower() == 'post'
        assert form_url == None # This means same as origin page
        # lxml's form data doesn't include the button being pressed.
        rsp = auth_client.post(url, parms)
        assert rsp.status_code == 302, "Probably had a form error, didn't get a redirect"
    
    # Find the form we want.
    form = [x for x in dom.forms if x.get_element_by_id('id_shortname', None) != None]
    assert len(form) == 1
    # Submit it.
    submit_form(form[0], open_http=http_fn)
    
    # Wait until the job completes.
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)

    # Check that we got an output file.
    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws.id, p_id)
    assert os.path.exists(bji.outfile)


def run_animal_model_flow(ws, auth_client):
    # Open up the job_start page, make sure it loads.
    url = f'/cv/{ws.id}/job_start/wf_{ws.id}_AnimalModelFlow/'
    resp = auth_client.get(url)
    assert resp.status_code == 200

    # Parse the submit form out of the page.
    from lxml import html
    from lxml.html import submit_form
    dom = html.fromstring(resp.content)

    def http_fn(method, form_url, values):
        print(("Calling http", method, url, values))
        animalmodel_parts = [v for k,v in values if k == 'animalmodel_parts']
        parms = dict(values, animalmodel_parts=animalmodel_parts, run_btn=True)
        print("Parms set to ", parms)
        assert method.lower() == 'post'
        assert form_url == None # This means same as origin page
        # lxml's form data doesn't include the button being pressed.
        rsp = auth_client.post(url, parms)
        assert rsp.status_code == 302, "Probably had a form error, didn't get a redirect"
    
    # Find the form we want.
    form = [x for x in dom.forms if x.get_element_by_id('id_p2d_file', None) != None]
    assert len(form) == 1

    from algorithms.run_depend import MyJobInfo as depend_job
    orig_depend_setup = depend_job.setup
    def depend_setup_wrapper(self):
        self.parms['qv_thres'] = 0
        return orig_depend_setup(self)

    from algorithms.run_gpbr import MyJobInfo as gpbr_job
    orig_gpbr_run = gpbr_job.run
    def gpbr_run_wrapper(self):
        self.parms['iterations'] = 2
        orig_gpbr_run(self)

    with patch('algorithms.run_depend.MyJobInfo.setup', side_effect=depend_setup_wrapper, autospec=True), \
         patch('algorithms.run_gpbr.MyJobInfo.run', side_effect=gpbr_run_wrapper, autospec=True):
        # Submit it.
        submit_form(form[0], open_http=http_fn)
        # Wait until the job completes.
        p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)

    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws, p_id)
    jt_to_id = bji.get_scoreset().job_type_to_id_map()

    print("Job type to id map: ", jt_to_id)

    customsig_jobs = {k:v for (k,v) in jt_to_id.items() if k.startswith('customsig')}
    assert len(customsig_jobs) == 8, "Expected custom sig+codes/glf/depend; sigdif+codes/glf/depend"

    path_jobs = {k:v for (k,v) in jt_to_id.items() if k.startswith('mousets_path')}
    assert len(path_jobs) == 2, "Expected path, gpbr"

    gesig_jobs = {k:v for (k,v) in jt_to_id.items() if k.startswith('mousets_gesig')}
    assert len(gesig_jobs) == 8, "Expected gesig codes/glf/depend; sigdif+codes/glf/depend"


    assert len(jt_to_id) == 18, "Extra jobs?"

def check_compare_page(ws, auth_client):
    url = f'/rvw/{ws.id}/animal_model_compare/'
    resp = auth_client.get(url)
    assert resp.status_code == 200

    # Parse the submit form out of the page.
    from lxml import html
    from lxml.html import submit_form
    dom = html.fromstring(resp.content)

    from browse.models import DrugSet, WsAnnotation
    ds = DrugSet.objects.create(
        ws=ws,
        name='drugs',
    )
    wsas = WsAnnotation.objects.filter(ws=ws)[:2]
    assert len(wsas) == 2
    ds.add_mols(wsas, 'qa')

    extra_parms = dict(
        ds=f"ds{ds.id}", 
        show_btn=True,
        )
    def http_fn(method, form_url, values):
        print(("Calling http", method, url, values))
        parms = dict(values, **extra_parms)
        print("Parms set to ", parms)
        assert method.lower() == 'post'
        assert form_url == None # This means same as origin page
        # lxml's form data doesn't include the button being pressed.
        rsp = auth_client.post(url, parms, follow=True)
        assert rsp.status_code == 200, "Error loading page"
    
    # Find the form we want.
    form = [x for x in dom.forms if x.get_element_by_id('id_top_n_prots', None) != None]
    assert len(form) == 1

    # Submit it.
    submit_form(form[0], open_http=http_fn)

def setup_fake_tissue(ws):
    from browse.models import Species, TissueSet, Tissue
    ts = TissueSet.objects.create(
        ws=ws,
        species=Species.MOUSE,
        name='MouseTS',
    )

    t = Tissue.objects.create(
        ws=ws,
        tissue_set=ts,
        geoID='FakeTissue',
        over_proteins=2,
        ev_cutoff=0,
        fc_cutoff=0,
        total_proteins=2,
    )

    from runner.models import Process
    from runner.process_info import JobInfo
    import json
    proc = Process.objects.create(
        name='sig',
        role='sig',
        rundir='',
        cmd='',
        status=Process.status_vals.SUCCEEDED,
        settings_json=json.dumps({
            'tissue_id': t.id,
        }),
    )
    bji = JobInfo.get_bound(ws, proc.id)
    from path_helper import make_directory
    make_directory(bji.outdir)
    make_directory(bji.lts_abs_root)
    data = [
        ['P01', '0.9', '1', '2'],
        ['P02', '0.99', '-1', '2'],
    ]
    with open(bji.fn_dcfile,'w') as f:
        f.write('uniprot\tev\tdir\tfold\n')
        for rec in data:
            f.write('\t'.join(rec) + '\n')


    t.sig_result_job_id=proc.id
    t.save()

    assert len(list(t.sig_results())) == 2
        


@patch('dtk.lts.LtsRepo')
def test_animalmodel(lts_repo, # We write to this, so mock it out
                         tmpdir_path_helper, # Write out to a tmpdir
                         auth_client, # Make http requests (pre-auth'd)
                         make_ws, # We need a WS with drugs
                         mock_dpi, # We need some fake DPI
                         mock_ppi, # We need some fake PPI
                         mock_pathways, # We need some fake pathways
                         live_server, # Allows cross-thread to see DB changes
                         mock_remote_machine, # Runs 'remote machine' code locally
                         local_stdjobinfo, # Same as above, but for stdjobinfo types
                         ):
    ws_attrs = []
    for i in range(1, 6):
        ws_attrs += [('DB0%d' % i, 'canonical', 'Drug%d' % i),
                    ('DB0%d' % i, 'm_dpimerge_id', 'DB0%d' % i)]
    ws = make_ws(ws_attrs)
    # Make sure LTS writes to the tmpdir
    setup_lts_tmpdir()

    # Make our background job run in-process (with all our mocks in place)
    shim_background_jobs_in_process()

    from browse.models import Protein
    Protein.objects.create(
        uniprot='P01',
        gene='G01',
    )

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
    mock_dpi('dpi.fake_dpi', dpi)

    # Mock out PPI
    ppi = [
        ('prot1', 'prot2', 'evidence', 'direction'),
        ('P01', 'P01', '0.9', '0'),
        ('P02', 'P02', '0.9', '0'),
        ('P01', 'P02', '0.9', '0'),
        ('P03', 'P02', '0.9', '0'),
        ('P09', 'P02', '1.0', '0'),
        ('P01', 'P04', '0.9', '0'),
        ('P08', 'P04', '0.9', '0'),
        ]
    mock_ppi('ppi.fake_ppi', ppi)

    mock_pathways([
        ['PWY1', ['P01', 'P02']],
        ['PWY2', ['P01', 'P04']],
        ['PWY3', ['P01']],
        ['PWY4', ['P01', 'P02', 'P04']],
        ['PWY5', [f'P{i:02}' for i in range(99)]],
    ])


    setup_fake_tissue(ws)
    run_custom_sig(ws, auth_client)
    run_animal_model_flow(ws, auth_client)
    check_compare_page(ws, auth_client)

    
