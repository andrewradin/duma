from __future__ import print_function
from builtins import range
import pytest
from django.urls import reverse
import django
import mock
from mock import patch
from mock import MagicMock
import json
import vcr
import os
import logging
import subprocess
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.remote_connection import LOGGER as sel_logger
from runner.process_info import JobInfo
from browse.models import WsAnnotation
import six

logger = logging.getLogger(__name__)

# Xvfb is very spammy with its debug logs, it logs via easyprocess, prevent it.
logging.getLogger("easyprocess").setLevel(logging.WARNING)
# Same with selenium
sel_logger.setLevel(logging.WARNING)
logging.getLogger("vcr").setLevel(logging.WARNING)

"""
This test aims to start from a clean database and test everything from there.

It uses a miniature dataset to try to keep timings reasonable.

NOTE: The mini test dataset is generated with the 'build_test_dataset.py' script in dtk/testdata/


In an ideal world, anything tested here would also have its own specific
test, but this is an easy way to get some comprehensive coverage.

Because iteration speed on a single giant test is awful, the test is broken
up into sequential parts.  We snapshot the state in between parts, allowing
them to be re-run independently without having to rerun all previous steps.

One downside of this, since the database is part of the snapshot, is that
schema changes only propagate to the first failure. After a schema change,
you need each part to complete successfully before the next part actually
gets tested in the correct environment.

Currently Out of Scope
- LTS (Mocked out via mock)
- Third-party website queries (Responses recorded replayed via vcrpy)
- Running work remote (everything is shimmed to run locally)


Two frameworks make this easiser.

mock: Replaces objects with MagicMocks, which (by default) return MagicMocks
for all instances and function calls.  You only need to implement the methods
that you care about.

vcrpy: Intercepts all http requests made by all the most popular http libraries.
If there is no cassette file, it will record all responses to a file.
If there is a cassette file, it will replay all responses from the file,
throwing an error if a request is made that it doesn't have a record for.
So, if a test response changes substantially, you can delete the existing
cassette file to force a re-record. For small changes, it may be worth
manually editing the cassette file. Note that when re-recording a file,
you should run the test a second time to verify that the playback works
as expected.


TODO: Even with a small dataset, we should try to make this as realistic.
Otherwise, we may hit unrealistic corner cases.
We should see if we can pick a small set of drugs, proteins, tissues, KTs, etc. that all fit together.


**DEBUGGING NOTES**
Some helpful hints if you're trying to debug a test failure here.
(Also see https://ariapharmaceuticals.atlassian.net/wiki/spaces/TD/pages/950403073/Automated+Testing)

- Each e2e test 'part' can be run in isolation, assuming that tests up to that point have run successfully
  - This relies on snapshotting that occurs at the end of each test
  - e.g. "pytest -vk 'check_workspace'" will run only that step, but relies on an existing 'import_drugs' snapshot

- There is a shared 'ws' cache directory re-used between tests - it can be useful to look at what's in there
  Sometimes deleting the cache can help with ephemeral state issues caused by old files sitting around in there
   (/tmp/teste2ecache or /mnt2/tmp/teste2ecache)

- Each part of the test snapshots - you can look at $TMPDIR/snapshots/check_glf_py/... to see the contents of:
  - The 'remote machine' ws directory
  - The 'LTS' directory'
  - The 'local ws' directory (minus non-ws-specific pieces that end up in the shared cache above)

- Each test part is running a fully functional django server
  - Port is random (check logs at the start, run with '-s' flag to see them in stdout)
  - Login is unit-test-username / unit-test-password
  - Insert a sleep at the point where you want it to pause, then spin up a browser and login and investigate

- For selenium debugging, you can use 'driver.screenshot("/tmp/blah1.png")' at any point in the test to have it save
  out a screenshot of what selenium sees in the browser

"""

username = "unit-test-username"
password = "unit-test-password"

from .std_vcr import std_vcr

from .tmpdir_path_helper import tmpdir_path_helper, s3cachedir
from .mock_remote import mock_remote_machine

from path_helper import PathHelper
TMPDIR = PathHelper.get_config()['bigtmp']

PPI_NAME = 'string.e2e_test.v1'

EXPECTED_MOLS = 85
EXPECTED_MOAS = 62
EXPECTED_DRUG_RECS = EXPECTED_MOLS + EXPECTED_MOAS

def get_only_ws():
    from browse.models import Workspace
    q = Workspace.objects.all()
    assert len(q) == 1
    return q[0]

def get_only_ws_id():
    return get_only_ws().id


def assert_file_unchanged(fn, expected_fn=None, prefix="", wsa_replace_cols=None):
    assert os.path.exists(fn), "Output file %s does not exist" % fn

    if not expected_fn:
        # We skip the job id because it could change, particularly if we add parts.
        # However, that does mean you'll need to specify an explicit name if
        # you want to validate multiple versions of the same job.
        parts = fn.split('/')
        fn_suffix = '_'.join([parts[-3], parts[-1]])
        expected_fn = 'dtk/testdata/ref/%s' % (prefix + fn_suffix, )

    # Apply any transforms to the input file to make it consistent.
    # We always sort it, and if wsa_replace is specified then we replace wsa_ids
    # with the agent name.

    def swap_wsa(wsa_id):
        if wsa_id == '-':
            # Special case for defus... if there are other cases, might
            # need to make this more generic.
            return wsa_id
        wsa = WsAnnotation.objects.get(pk=wsa_id)
        return wsa.agent.canonical

    from dtk.files import get_file_records
    rows = [row for row in get_file_records(fn)]

    # By default we will use replacement on columns labeled 'wsa'
    if wsa_replace_cols is None and rows and 'wsa' in rows[0]:
        wsa_replace_cols = [i for i, x in enumerate(rows[0]) if x == 'wsa']

    if wsa_replace_cols:
        for i in range(1, len(rows)):
            rows[i] = [swap_wsa(x) if colidx in wsa_replace_cols else x
                       for colidx, x in enumerate(rows[i])]

    for row in rows:
        for i, cell in enumerate(row):
            try:
                x = float(cell)
                # Use consistent float formatting to compare.
                row[i] = "%.6g" % x
            except:
                pass


    # Sort anything left
    rows = [rows[0]] + sorted(rows[1:])

    normalized_input_fn = fn + '.normalized'
    with open(normalized_input_fn, 'w') as f:
        f.write('\n'.join(['\t'.join(row) for row in rows]))


    if not os.path.exists(expected_fn):
        print("Expected file does not exist, using existing as expected")
        import shutil
        shutil.copyfile(normalized_input_fn, expected_fn)

    with open(normalized_input_fn) as f:
        new_content = f.readlines()
    with open(expected_fn) as f:
        expected_content = f.readlines()

    assert new_content == expected_content, \
            "File didn't match expected. \
            Run 'diff %s %s'. \
            This could be because of unstable outputs, or because of \
            explicit changes.  If this change is expected, just update the file. \
            " % (normalized_input_fn, expected_fn)


def assert_good_response(resp, expected_status=200, no_html_tag=True):
    if expected_status is not None:
        assert resp.status_code == expected_status

    if no_html_tag:
        # The intent here is to catch cases where we've failed to mark_safe our
        # HTML and it shows up as text on the page.
        # Unclear how valuable this is.
        from lxml import html, etree
        dom = html.fromstring(resp.content)
        text = dom.text_content()
        # Different ways of writing text'ified end tag />
        # Though lxml doesn't do a great job here, and tends to turn them
        # into real HTML instead of displaying them as text.
        bads = [
            # '/>',  # This one is used for tags in DataTable content.
            '/&gt',
            '/&#62',
            '/&#x3'
            ]
        for bad in bads:
            assert bad not in text

def setup_workspace(client, admin_client, driver, live_server, **kwargs):
    print("setup_workspace test")
    # Clear out any drugsets.  These are the most likely things to change
    # and affect tests (in particular create properties).
    from path_helper import PathHelper, make_directory
    ds_dir = PathHelper.drugsets
    if os.path.exists(ds_dir):
        assert ds_dir.startswith(TMPDIR), "Only delete thing in tmp"
        import shutil
        shutil.rmtree(ds_dir)
    make_directory(ds_dir)

    check_global_pages(client, admin_client, driver, live_server.url)
    ws_id = create_workspace(client, admin_client)
    check_empty_workspace_pages(client, ws_id)
    print("Done setup workspace test")

def import_drugs(client, **kwargs):
    print("import_drugs test")
    check_import_drugs(client)
    check_import_prots(client)
    print("Done import drugs test")

def check_workspace(client, **kwargs):
    print("Check workspace test")
    ws_id = get_only_ws_id()

    url = '/?all=1'.format()
    assert_good_response(client.get(url))

    check_prot_search(client, ws_id)
    check_prot_search_api(client)
    check_prot_page(client, ws_id)
    check_import_workspace_drugs(client, ws_id)
    check_search_page(client, ws_id)

def import_tissue_meta(client, driver, live_server, **kwargs):
    from algorithms.run_meta import MyJobInfo
    with patch.object(MyJobInfo, '_test_add_extra_settings', autospec=True) as extra_fn:
        def _test_add_extra_settings(self, sf):
            from path_helper import PathHelper,make_directory
            outdir = PathHelper.publish + self.tissue.geoID
            pubdir = self.mch.get_remote_path(outdir)
            sf.write('microarrayDir <- "%s"\n' % pubdir)
            sf.write('storageDir <- "%s"\n' % PathHelper.s3_cache_root)
            make_directory(pubdir)
        extra_fn.side_effect = _test_add_extra_settings

        ws_id = get_only_ws_id()
        live_prefix = live_server.url
        from browse.models import TissueSet, Tissue
        tissue_sets = TissueSet.objects.all()
        assert len(tissue_sets) == 2
        ts = tissue_sets[0]


        assert len(Tissue.objects.filter(tissue_set_id=ts.id)) == 0, \
            "We should start with no tissues in this tissueset"
        # GSE7451?
        opts = {
            'source': 0,
            'geo_id': 'GSE37985',
            'tissue': 'Test Tissue Data',
            'tissue_set': ts.id
            }
        assert client.post('/ge/{}/tissues/?op=new_tissue'.format((ws_id)), opts, follow=True).status_code == 200

        wait_for_background_job(expect_success=True, timeout_secs=60*30)

        assert len(Tissue.objects.filter(tissue_set_id=ts.id)) == 1, \
            "Our new tissue should show up"


def import_tissue_sig(client, driver, live_server, **kwargs):
    from aws_op import Machine
    from algorithms.run_sig import MyJobInfo
    with patch.object(MyJobInfo, '_test_add_extra_settings', autospec=True) as extra_fn:
        def _test_add_extra_settings(self, sf):
            from path_helper import PathHelper,make_directory
            outdir = PathHelper.publish + self.tissue.geoID
            # XXX versioned uniprot?
            unimap_fn = os.path.join(PathHelper.s3_cache_root, 'uniprot', 'uniprot.HUMAN_9606.v1.Protein_Entrez.tsv')
            pubdir = self.mch.get_remote_path(outdir)
            sf.write('microarrayDir <- "%s"\n' % pubdir)
            sf.write('storageDir <- "%s"\n' % PathHelper.s3_cache_root)
            # We're only providing conversions for a tiny fraction.
            sf.write('absoluteMinUniprotProportion <- 0.001\n')
            sf.write('EntrezToUniprotMap <- "%s"\n' % unimap_fn)
        extra_fn.side_effect = _test_add_extra_settings
        ws_id = get_only_ws_id()
        live_prefix = live_server.url
        from browse.models import TissueSet, Tissue
        tissue_sets = TissueSet.objects.all()
        assert len(tissue_sets) == 2
        ts = tissue_sets[0]
        t = Tissue.objects.filter(tissue_set_id=ts.id)[0]
        url = '/ge/{}/classify/{}/'.format((ws_id), (t.id))
        assert_good_response(client.get(url))
        driver.get('{}{}'.format((live_prefix), (url)))
        driver.find_elements_by_name('cci_radio_button_0')[1].click()
        driver.find_elements_by_name('cci_radio_button_1')[2].click()
        driver.find_elements_by_name('process_btn')[0].click()

        p_id = wait_for_background_job(expect_success=True, timeout_secs=60*30)
    

    ws_id = get_only_ws_id()
    bji = JobInfo.get_bound(ws_id, p_id)
    assert_file_unchanged(bji.fn_dcfile)

    from aws_op import Machine
    remote_publish = Machine.get_remote_path(PathHelper.publish)
    assert os.path.abspath(remote_publish).startswith(TMPDIR)
    import shutil
    print("Cleaning up some sig/meta data we don't need anymore")
    # This is fairly big, and if we don't delete it we'll be copying it
    # around to all future snapshots.
    shutil.rmtree(remote_publish)


def check_pathsum(client, driver, live_server, **kwargs):
    ws_id = get_only_ws_id()
    live_prefix = live_server.url
    from browse.models import TissueSet
    tissue_sets = TissueSet.objects.all()
    ts_id = tissue_sets[0].id
    url = '/cv/{}/job_start/path_{}_{}/'.format((ws_id), (ts_id), (ws_id))
    assert_good_response(client.get(url))


    driver.get('{}{}'.format((live_prefix), (url)))

    # Set the ev threshold way down so we get some actual scores.
    set_input_to(driver.find_element_by_name('t_1'), '0.1')

    driver.find_elements_by_class_name('btn-primary')[0].click()

    p_id = wait_for_background_job(expect_success=True, timeout_secs=30*60)

    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws_id, p_id)

    assert_file_unchanged(bji.outfile)


def check_tissue_views(client, **kwargs):
    with patch('dtk.ae_search.add_geo_to_run') as add_geo:
        # add_geo_to_run calls bigquery, and each bigquery call has a
        # GUID, so I couldn't (easily) get it to work with vcrpy. For
        # now, just disable that branch
        add_geo.return_value = []
        ws_id = get_only_ws_id()
        # Without force_fake_mp, pmap was throwing because it was trying to
        # pickle a module; I think this was somehow related to vcrpy, but
        # anyway, this fixes it.
        with patch('dtk.parallel.force_fake_mp', True):
            check_search_tissues(client, ws_id)
        get_tissue_views(client, ws_id)
        check_drug_views(client, ws_id)

def check_gesig(client, driver, live_server, **kwargs):
    from runner.process_info import JobInfo
    ws_id = get_only_ws_id()
    live_prefix = live_server.url
    from browse.models import TissueSet
    tissue_sets = TissueSet.objects.all()
    ts_id = tissue_sets[0].id
    url = '/cv/{}/job_start/gesig_{}_{}/'.format((ws_id), (ts_id), (ws_id))
    assert_good_response(client.get(url))

    # Test with randomize button
    # This seems to be broken, and isn't something we really care about,
    # so ignore for now.
    if False:
        from algorithms import run_gesig
        run_gesig.BG_ITERS = 2 # default is 100, which is slow
        driver.get('{}{}'.format((live_prefix), (url)))
        driver.find_element_by_id('id_randomize').click()
        driver.find_elements_by_class_name('btn-primary')[0].click()
        p_id = wait_for_background_job(expect_success=True, timeout_secs=30*60)
        bji = JobInfo.get_bound(ws_id, p_id)
        assert os.path.exists(bji.signature_file)


    # Test without randomize button (we do this second because this is the
    # one we actually want to use as inputs elsewhere)
    driver.get('{}{}'.format((live_prefix), (url)))
    driver.find_elements_by_class_name('btn-primary')[0].click()

    p_id = wait_for_background_job(expect_success=True, timeout_secs=30*60)
    bji = JobInfo.get_bound(ws_id, p_id)
    assert_file_unchanged(bji.signature_file)



def check_codes(client, driver, live_server, **kwargs):
    ws_id = get_only_ws_id()
    live_prefix = live_server.url
    url = '/cv/{}/job_start/codes_{}/'.format((ws_id), (ws_id))
    driver.get('{}{}'.format((live_prefix), (url)))
    driver.find_elements_by_class_name('btn-primary')[0].click()
    p_id = wait_for_background_job(expect_success=True, timeout_secs=30*60)
    bji = JobInfo.get_bound(ws_id, p_id)
    assert_file_unchanged(bji.outfile)

def check_gpbr(client, driver, live_server, **kwargs):
    ws_id = get_only_ws_id()
    live_prefix = live_server.url
    url = '/cv/{}/job_start/gpbr_{}/'.format((ws_id), (ws_id))
    driver.get('{}{}'.format((live_prefix), (url)))

    iterations_input = driver.find_element_by_id('id_iterations')
    iterations_input.clear()
    iterations_input.send_keys('2')
    driver.find_elements_by_class_name('btn-primary')[0].click()
    p_id = wait_for_background_job(expect_success=True, timeout_secs=30*60)

    bji = JobInfo.get_bound(ws_id, p_id)

    assert_file_unchanged(bji.fn_direct_scores)
    assert_file_unchanged(bji.fn_indirect_scores)

def set_input_to(input_el, value):
    input_el.clear()
    input_el.send_keys(str(value))

def check_glee(client, driver, live_server, **kwargs):
    ws_id = get_only_ws_id()
    live_prefix = live_server.url
    url = '/cv/{}/job_start/glee_{}/'.format((ws_id), (ws_id))
    driver.get('{}{}'.format((live_prefix), (url)))

    iterations_input = driver.find_element_by_id('id_nPermuts')
    iterations_input.clear()
    # Note that glee will fail on 1 or 2 iterations, because medianwithCI fails.
    iterations_input.send_keys('10')
    driver.find_element_by_id('id_fake_mp').click()
    driver.find_elements_by_class_name('btn-primary')[0].click()
    p_id = wait_for_background_job(expect_success=True, timeout_secs=30*60)

def check_glf(client, driver, live_server, **kwargs):
    ws_id = get_only_ws_id()
    live_prefix = live_server.url
    url = '/cv/{}/job_start/glf_{}/'.format((ws_id), (ws_id))
    driver.get('{}{}'.format((live_prefix), (url)))
    # It is slightly faster to use 1 core than 40 here for this test.
    driver.find_element_by_id('id_fake_mp').click()
    driver.find_elements_by_class_name('btn-primary')[0].click()
    p_id = wait_for_background_job(expect_success=True, timeout_secs=30*60)

    bji = JobInfo.get_bound(ws_id, p_id)
    assert_file_unchanged(bji.outfile)

def check_depend(client, driver, live_server, **kwargs):
    ws_id = get_only_ws_id()
    live_prefix = live_server.url
    url = '/cv/{}/job_start/depend_{}/'.format((ws_id), (ws_id))

    # Test GLEE -> Depend
    driver.get('{}{}'.format((live_prefix), (url)))
    glee_sel = Select(driver.find_element_by_id('id_glee_run'))
    # 0'th index is None, let's pick the first which should be our glee run.
    glee_sel.select_by_index(1)

    # Change the qval threshold, otherwise we have no inputs.
    set_input_to(driver.find_element_by_name('qv_thres'), '0')

    driver.find_elements_by_class_name('btn-primary')[0].click()
    p_id = wait_for_background_job(expect_success=True, timeout_secs=30*60)

    bji = JobInfo.get_bound(ws_id, p_id)
    # glee-based depend isn't stable, output changes every time.
    #assert_file_unchanged(bji.outfile)
    assert os.path.exists(bji.outfile)


    # Test GLF -> Depend
    driver.get('{}{}'.format((live_prefix), (url)))
    glf_sel = Select(driver.find_element_by_id('id_glf_run'))
    # 0'th index is None, let's pick the first which should be our glf run.
    glf_sel.select_by_index(1)

    # Change the qval threshold, otherwise we have no inputs.
    set_input_to(driver.find_element_by_name('qv_thres'), '0')

    driver.find_elements_by_class_name('btn-primary')[0].click()
    p_id = wait_for_background_job(expect_success=True, timeout_secs=30*60)

    bji = JobInfo.get_bound(ws_id, p_id)
    assert_file_unchanged(bji.outfile)

def check_gwas(client, driver, live_server, **kwargs):
    ws_id = get_only_ws_id()
    live_prefix = live_server.url
    url = '/{}/gwas_search/'.format((ws_id))
    driver.get('{}{}'.format((live_prefix), (url)))
    driver.find_element_by_name('terms').send_keys('lupus')
    print("Searching for disease")
    click_and_wait(driver, name='search_btn')
    print("Done, now import one")
    uncollapse_all(driver)
    select_btn = driver.find_elements_by_name('select_btn')[1]
    select_btn.click()
    WebDriverWait(driver, 180).until(EC.staleness_of(select_btn))
    print("Done GWAS")

def check_esga(client, driver, live_server, **kwargs):
    ws_id = get_only_ws_id()
    live_prefix = live_server.url
    url = '/cv/{}/job_start/esga_{}/'.format((ws_id), (ws_id))
    driver.get('{}{}'.format((live_prefix), (url)))


    select = Select(driver.find_element_by_id('id_ppi_file'))
    selected_option = select.first_selected_option
    assert selected_option.text == f'{PPI_NAME}'

    # Some nodes end up empty without this, which leads to it not converging.
    # I suspect it's an issue with the way we setup pagerank, not removing
    # initial nodes if they don't show up in the graph for personalization.
    set_input_to(driver.find_element_by_name('min_ppi'), "0.5")
    set_input_to(driver.find_element_by_name('restart_prob'), "0.5")

    # Lower the pagerank tolerance, our graph is probably too sparse to converge
    # nicely.
    driver.find_elements_by_class_name('btn-primary')[0].click()
    p_id = wait_for_background_job(expect_success=True, timeout_secs=30*60)

    bji = JobInfo.get_bound(ws_id, p_id)
    assert_file_unchanged(bji.outfile)

def check_gpath(client, driver, live_server, **kwargs):
    ws_id = get_only_ws_id()
    live_prefix = live_server.url
    url = '/cv/{}/job_start/gpath_{}/'.format((ws_id), (ws_id))
    driver.get('{}{}'.format((live_prefix), (url)))
    driver.find_elements_by_class_name('btn-primary')[0].click()
    p_id = wait_for_background_job(expect_success=True, timeout_secs=30*60)

    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws_id, p_id)
    assert_file_unchanged(bji.outfile)

def check_gwasig(client, driver, live_server, **kwargs):
    ws_id = get_only_ws_id()
    live_prefix = live_server.url
    url = '/cv/{}/job_start/gwasig_{}/'.format((ws_id), (ws_id))
    driver.get('{}{}'.format((live_prefix), (url)))
    driver.find_elements_by_class_name('btn-primary')[0].click()
    p_id = wait_for_background_job(expect_success=True, timeout_secs=30*60)

    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws_id, p_id)
    assert_file_unchanged(bji.signature_file)


def click_and_wait(driver, name=None, id=None, timeout=180):
    print("Clicking on %s and waiting for it" % name if name else id)
    el = driver.find_element_by_name(name) if name else driver.find_element_by_id(id)
    el.click()
    WebDriverWait(driver, 180).until(EC.staleness_of(el))
    print("Loaded")


def check_kt_search(client, driver, live_server, **kwargs):
    ws_id = get_only_ws_id()
    live_prefix = live_server.url
    url = '/kts/{}/search/'.format((ws_id))
    driver.get('{}{}'.format((live_prefix), (url)))
    print("CT search for " + driver.find_element_by_name('clinical_trials_search_term').get_attribute('value'))
    ct_search = driver.find_element_by_name('clinical_trials_search_term')
    ct_search.clear()
    ct_search.send_keys("Sjogren's Syndrome")
    search_btn = driver.find_element_by_name('search_btn')
    search_btn.click()
    print("Waiting for search to complete")
    WebDriverWait(driver, 180).until(EC.staleness_of(search_btn))
    print("Search completed, check summary")

    from ktsearch.models import KtSearchResult, KtSearch
    searches = KtSearch.objects.all()
    results = KtSearchResult.objects.all()
    assert len(searches) == 1
    assert len(results) == 26
    kts_id = searches[0].id

    url = '/kts/{}/summary/{}/'.format((ws_id), (kts_id))
    assert_good_response(client.get(url))

    url = '/kts/{}/name_resolve/{}/'.format((ws_id), (kts_id))
    assert_good_response(client.get(url))

    # We have none of these drugs in our workspace right now... oh well, let's
    # randomly match some up.
    driver.get('{}{}'.format((live_prefix), (url)))
    drug_name = driver.find_elements_by_partial_link_text('Clinical')[0].text
    print("Unmatching {}".format(drug_name))
    click_and_wait(driver, name="unmatch_btn")

    from drugs.models import DpiMergeKey, Drug

    WS_DRUGS = ["ABT-089", "ABT-288", "AZD1208"]
    for ws_drug in WS_DRUGS:
        drug_name = driver.find_elements_by_partial_link_text('Clinical')[0].text
        print("Setting {} to {}".format(drug_name, ws_drug))
        driver.find_element_by_id("id_search_term").clear()
        driver.find_element_by_id("id_search_term").send_keys(ws_drug)
        click_and_wait(driver, name='search_btn')
        click_and_wait(driver, name='match_btn')


    url = '/kts/{}/resolve/{}/'.format((ws_id), (kts_id))
    assert_good_response(client.get(url))

    print("Resolve some KTs")

    driver.get('{}{}'.format((live_prefix), (url)))
    drug_name = driver.find_element_by_partial_link_text('Clinical').text
    print("Updating {} to proposed".format(drug_name))
    click_and_wait(driver, name='update_btn')

    print("Ignoring something")
    drug_name = driver.find_element_by_partial_link_text('Clinical').text
    print("Leaving {} as-is".format(drug_name))
    click_and_wait(driver, name='ignore_btn')

    driver.get('{}{}'.format((live_prefix), (url)))
    drug_name = driver.find_element_by_partial_link_text('Clinical').text
    print("Updating {} to proposed".format(drug_name))
    click_and_wait(driver, name='update_btn')

def uncollapse_all(driver):
    driver.execute_script("$('.collapse').removeClass('collapse');")

def check_drug_eff(driver, live_server, **kwargs):
    ws_id = get_only_ws_id()
    live_prefix = live_server.url

    # Turns out we have no KTs (best is Phase 3), let's just set something.
    print("Update a drug to be a known treatment")
    from browse.models import WsAnnotation
    wsa = WsAnnotation.objects.filter(ws=ws_id)[0]
    url = '/mol/{}/annotate/{}/'.format((ws_id), (wsa.id))
    driver.get(live_prefix+url)
    uncollapse_all(driver) # Update indication is in a collapse section
    from selenium.webdriver.support.ui import Select
    ind_sel = Select(driver.find_element_by_name('indication'))
    ind_sel.select_by_visible_text("FDA Approved Treatment")
    set_input_to(driver.find_element_by_name('indication_href'), 'http://twoxar.com')
    click_and_wait(driver, name='indication_btn')
    from browse.models import Workspace
    ws = Workspace.objects.get(pk=ws_id)
    kts = ws.get_wsa_id_set(ws.eval_drugset)
    assert len(kts) > 0, "We should have more than 1 KT"

    print("Run drug eff job")
    url = '/cv/{}/job_start/fvs_wsa_efficacy_{}/'.format((ws_id), (ws_id))
    driver.get('{}{}'.format((live_prefix), (url)))

    path_direct = driver.find_element_by_name('feat_cc_path_direct')
    if path_direct.is_selected():
        path_direct.click()
    path_abs = driver.find_element_by_name('feat_cc_path_absDir')
    if path_abs.is_selected():
        path_abs.click()

    click_and_wait(driver, name='run_btn')
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)
    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws_id, p_id)
    assert os.path.exists(os.path.join(bji.lts_abs_root, 'feature_matrix.npz'))

def check_wzs(driver, live_server, **kwargs):
    ws_id = get_only_ws_id()
    live_prefix = live_server.url
    url = '/cv/{}/job_start/wzs_{}/'.format((ws_id), (ws_id))
    driver.get('{}{}'.format((live_prefix), (url)))
    set_input_to(driver.find_element_by_name('auto_iter'), 12)
    set_input_to(driver.find_element_by_name('auto_new_count'), 4)
    set_input_to(driver.find_element_by_name('auto_extra_count'), 3)
    click_and_wait(driver, name='run_btn')
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)
    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws_id, p_id)

    # These change wildly every run.
    assert os.path.exists(os.path.join(bji.lts_abs_root, 'wz_score.tsv'))
    assert os.path.exists(os.path.join(bji.lts_abs_root, 'weights.tsv'))

    # Now try again with SORCondensed, it does some different codepaths.
    driver.get('{}{}'.format((live_prefix), (url)))
    from selenium.webdriver.support.ui import Select
    rec_sel = Select(driver.find_element_by_name('auto_metric'))
    rec_sel.select_by_visible_text('SigmaOfRankCondensed')
    # Also switch agg method to wts, just to try something different.
    algo_sel = Select(driver.find_element_by_id('id_algo'))
    algo_sel.select_by_value('wts')
    set_input_to(driver.find_element_by_name('auto_iter'), 12)
    set_input_to(driver.find_element_by_name('auto_new_count'), 4)
    set_input_to(driver.find_element_by_name('auto_extra_count'), 3)
    click_and_wait(driver, name='run_btn')
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)
    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws_id, p_id)

    # These change wildly every run.
    assert os.path.exists(os.path.join(bji.lts_abs_root, 'wz_score.tsv'))
    assert os.path.exists(os.path.join(bji.lts_abs_root, 'weights.tsv'))




def check_trgscrimp(client, driver, live_server, **kwargs):
    ws = get_only_ws()
    wzs_id = ws.get_prev_job_choices('wzs')[0][0]
    live_prefix = live_server.url

    # Check the raw link prior to any job, should still work.
    from browse.models import WsAnnotation
    wsa = WsAnnotation.objects.all()[0]

    url = '/mol/{}/trg_scr_imp/{}/?wzs_jid={}&method=peel_cumulative'.format((ws.id), (wsa.id), (wzs_id))
    assert_good_response(client.get(url))

    # Now run the job.
    url = '/cv/{}/job_start/trgscrimp_{}/'.format((ws.id), (ws.id))
    driver.get('{}{}'.format((live_prefix), (url)))
    set_input_to(driver.find_element_by_name('count'), 5)
    click_and_wait(driver, name='run_btn')
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)
    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws, p_id)
    assert os.path.exists(os.path.join(bji.outdir, 'output.pickle.gz'))

def check_run_flag(client, driver, live_server, **kwargs):
    ws = get_only_ws()
    wzs_id = ws.get_prev_job_choices('wzs')[0][0]
    live_prefix = live_server.url

    url = '/cv/{}/job_start/flag_{}/'.format((ws.id), (ws.id))
    driver.get('{}{}'.format((live_prefix), (url)))
    driver.find_element_by_name('job_id').send_keys(str(wzs_id))
    driver.find_element_by_name('score').send_keys('wzs')
    driver.find_element_by_name('unwanted_targets_additional_uniprots').send_keys('P30613')
    click_and_wait(driver, name='run_btn')
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)
    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws, p_id)

def check_faers(client, driver, live_server, **kwargs):
    ws = get_only_ws()
    live_prefix = live_server.url
    from drugs.models import Drug

    faers_version = 'faers.v1'
    from dtk.faers import ClinicalEventCounts
    from scipy.sparse import load_npz, save_npz
    cec = ClinicalEventCounts(faers_version)
    print("Decimating our FAERS data to run quickly")
    for fm_name in ('indi', 'drug', 'demo', 'date'):
        npz_name = f'{fm_name}_mat.npz'
        fn = cec._get_path(faers_version, npz_name)
        assert fn.startswith(TMPDIR), "Why aren't we in the tmpdir?"
        npz = getattr(cec,f'_{fm_name}_fm')
        nrows = npz.shape[0]
        print("Found %s with %d rows" % (npz_name, nrows))
        MAX_LEN = 100000
        if nrows > MAX_LEN:
            print("Updating %s from %d to %d rows" % (fn, nrows, MAX_LEN))
            npz = npz[:MAX_LEN]
            save_npz(fn, npz)


    print("FAERS data decimated")

    # TODO: Construct a more conventional drugset so we don't have to do this.
    # FAERS isn't happy if there are 0 drugs with matching CAS #'s.
    # It also has to be a CAS with a bg_per_cas > 0.
    # We also need it to have an ATC code.
    Drug.objects.all()[0].set_prop("cas", "50-24-8")
    Drug.objects.all()[0].set_prop("atc", "N01AH05")

    url = '/cv/{}/job_start/faers_{}/'.format((ws.id), (ws.id))
    driver.get('{}{}'.format((live_prefix), (url)))
    driver.find_element_by_name('search_term').send_keys("sjogren's syndrome")
    click_and_wait(driver, name='run_btn')
    p_id = wait_for_background_job(expect_success=True, timeout_secs=5*60)
    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws, p_id)
    assert os.path.exists(bji.fn_enrichment)
    assert os.path.exists(bji.lr_enrichment)
    assert os.path.exists(bji.fn_coindications)

    assert_file_unchanged(bji.fn_enrichment)
    assert_file_unchanged(bji.lr_enrichment)
    assert_file_unchanged(bji.fn_coindications)


def check_capp(client, driver, live_server, **kwargs):
    ws = get_only_ws()
    live_prefix = live_server.url

    url = '/cv/{}/job_start/capp_{}/'.format((ws.id), (ws.id))
    driver.get('{}{}'.format((live_prefix), (url)))

    # If we don't set these parameters appropriately, we will exclude all
    # comorbidities and the job will fail.
    print("Configuring job")
    odd = driver.find_element_by_name('odd')
    odd.clear()
    odd.send_keys("0.1")
    pv = driver.find_element_by_name('pv')
    pv.clear()
    # This has to be an integer.
    pv.send_keys("1")
    print("Running job")

    click_and_wait(driver, name='run_btn')
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)
    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws, p_id)

    # Not consistent across machines
    assert_file_unchanged(bji.outfile)

def check_defus(client, driver, live_server, **kwargs):
    ws = get_only_ws()
    live_prefix = live_server.url

    url = '/cv/{}/job_start/defus_{}/'.format((ws.id), (ws.id))
    driver.get('{}{}'.format((live_prefix), (url)))
    click_and_wait(driver, name='run_btn')
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)
    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws, p_id)

    # Not consistent across machines
    assert_file_unchanged(bji.outfile, wsa_replace_cols=[0, 2, 4, 6, 8, 10])


def check_external(driver, live_server, **kwargs):
    ws = get_only_ws()
    wzs_id = ws.get_prev_job_choices('wzs')[0][0]
    live_prefix = live_server.url

    url = '/cv/{}/job_start/dgn_{}/'.format((ws.id), (ws.id))
    driver.get('{}{}'.format((live_prefix), (url)))
    driver.find_element_by_name('disease').clear()
    driver.find_element_by_name('disease').send_keys('C1527336')
    click_and_wait(driver, name='run_btn')
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)
    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws, p_id)
    assert_file_unchanged(bji.score_fn)

    url = '/cv/{}/job_start/otarg_{}/'.format((ws.id), (ws.id))
    driver.get('{}{}'.format((live_prefix), (url)))
    driver.find_element_by_name('disease').clear()
    driver.find_element_by_name('disease').send_keys('key:EFO_0000699')
    click_and_wait(driver, name='run_btn')
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)
    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws, p_id)
    assert_file_unchanged(bji.score_fn)

def check_lbn(driver, live_server, **kwargs):
    ws = get_only_ws()
    wzs_id = ws.get_prev_job_choices('wzs')[0][0]
    live_prefix = live_server.url

    # ML requires at least 6 examples.
    # Though this is only enforced when running directly, not when run
    # through the review workflow
    #
    # If you have fewer than 10 KTs, it will fail because it uses a training
    # portion of 0.9
    from flagging.utils import get_target_wsa_ids
    wsa_ids = get_target_wsa_ids(ws, wzs_id, 'wzs', 0, 10)
    for wsa in WsAnnotation.objects.filter(pk__in=wsa_ids):
        wsa.indication=WsAnnotation.indication_vals.KNOWN_TREATMENT
        wsa.save()

    url = '/cv/{}/job_start/lbn_{}/'.format((ws.id), (ws.id))
    driver.get('{}{}'.format((live_prefix), (url)))

    set_input_to(driver.find_element_by_name('job_id'), wzs_id)
    set_input_to(driver.find_element_by_name('score'), 'wzs')
    set_input_to(driver.find_element_by_name('count'), 23)

    with patch('dtk.entrez_utils.PubMedSearch') as pubmed_cons:
        pubmed = pubmed_cons.return_value
        pubmed.count_frequency.return_value = 5
        pubmed.size.return_value = 10000
        click_and_wait(driver, name='run_btn')
        p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)
        from runner.process_info import JobInfo
        bji = JobInfo.get_bound(ws, p_id)
        assert os.path.exists(bji.outfile)

def check_novelty_fvs(client, driver, live_server, **kwargs):
    ws = get_only_ws()
    live_prefix = live_server.url
    # Generate the novelty matrix using this.
    url = '/cv/{}/job_start/fvs_wsa_novelty_{}/'.format((ws.id), (ws.id))
    opts = {
            'feat_lbn_lbnOR': True,
            'run_btn': True,
            'training_set': 'kts',
            }
    # Not sure why, but selenium page often comes up without the right
    # sources configured.  Let's try just using the django client.
    assert client.post(url, opts).status_code == 302

    #driver.get(f'{live_prefix}{url}')
    #click_and_wait(driver, name='run_btn')
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)

def check_ml(driver, live_server, **kwargs):
    ws = get_only_ws()
    wzs_id = ws.get_prev_job_choices('wzs')[0][0]
    live_prefix = live_server.url


    url = '/cv/{}/job_start/ml_{}/'.format((ws.id), (ws.id))
    driver.get('{}{}'.format((live_prefix), (url)))

    set_input_to(driver.find_element_by_name('outer'), 2)
    set_input_to(driver.find_element_by_name('inner'), 2)

    click_and_wait(driver, name='run_btn')
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)
    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws, p_id)
    assert os.path.exists(bji.fn_scores)


def check_refresh_wf(driver, live_server, **kwargs):
    refresh_wf_main(driver, live_server, use_mols=False, test_train=True)

def check_refresh_wf_mols(driver, live_server, **kwargs):
    refresh_wf_main(driver, live_server, use_mols=True, test_train=False)

def refresh_wf_main(driver, live_server, use_mols, test_train):
    ws = get_only_ws()
    live_prefix = live_server.url

    # Save the threshold so that pathsum works, which lets gpbr work.
    from browse.models import Tissue
    t = Tissue.objects.all()[0]
    url = '/ge/{}/sigprot/{}/'.format((ws.id), (t.id))
    driver.get('{}{}'.format((live_prefix), (url)))
    set_input_to(driver.find_element_by_name('ev_cutoff'), '0.1')
    click_and_wait(driver, name='save_btn')
        

    # Set the faers name so that faers works
    ws.set_disease_default('FAERS', "sjogren's syndrome", 'username')
    ws.set_disease_default('OpenTargets', "Sjogren syndrome", 'username')
    ws.set_disease_default('DisGeNet', "C1800706", 'username')
    ws.set_disease_default('AGR', 'DOID:12894', 'username')
    ws.set_disease_default('Monarch', 'MONDO:0010030', 'username')

    # Hook in for our custom glee settings.
    from algorithms.run_glee import MyJobInfo as glee_job
    orig_glee_run = glee_job.run
    def run_wrapper(self):
        self.parms['fake_mp'] = True
        self.parms['nPermuts'] = 10
        orig_glee_run(self)

    # Hook in for our custom depend settings.
    from algorithms.run_depend import MyJobInfo as depend_job
    orig_depend_setup = depend_job.setup
    def depend_setup_wrapper(self):
        self.parms['qv_thres'] = 0
        return orig_depend_setup(self)

    # Hook in for our custom capp settings.
    # We need to modify the form itself, so that it propagates to the actual
    # job settings, so that gpbr picks up on these settings, or gpbr will later fail.
    from algorithms.run_capp import MyJobInfo as capp_job
    from algorithms.run_capp import ConfigForm as CappConfig
    CappConfig.base_fields['pv'].initial = 1
    CappConfig.base_fields['odd'].initial = 0.1

    orig_capp_setup = capp_job.setup
    def capp_setup_wrapper(self):
        # Redundant with above.
        #self.parms['odd'] = 0.1
        #self.parms['pv'] = 1
        orig_capp_setup(self)

    # Hook in for our custom gpbr settings.
    from algorithms.run_gpbr import MyJobInfo as gpbr_job
    orig_gpbr_run = gpbr_job.run
    def gpbr_run_wrapper(self):
        self.parms['iterations'] = 2
        orig_gpbr_run(self)

    # Hook in for our custom esga settings.
    from algorithms.run_esga import MyJobInfo as esga_job
    orig_esga_run = esga_job.run
    def esga_run_wrapper(self):
        self.parms['min_ppi'] = 0.5
        self.parms['restart_prob'] = 0.5
        orig_esga_run(self)

    # Setup ppi threshold to let esga work
    ws.ppi_thresh_default = 0.5
    ws.save()

    # Hook in for our custom wzs settings.
    from algorithms.run_wzs import MyJobInfo as wzs_job
    orig_wzs_run = wzs_job.run
    def wzs_run_wrapper(self):
        self.parms['auto_iter'] = 14
        self.parms['auto_new_count'] = 4
        self.parms['auto_extra_count'] = 3
        orig_wzs_run(self)

    # disable DataStatus checking, so our small datasets will run
    def dummy_data_status_ok(uji,ws,s1,s2):
        return True

    with patch('algorithms.run_glee.MyJobInfo.run', side_effect=run_wrapper, autospec=True), \
         patch('algorithms.run_capp.MyJobInfo.setup', side_effect=capp_setup_wrapper, autospec=True), \
         patch('algorithms.run_depend.MyJobInfo.setup', side_effect=depend_setup_wrapper, autospec=True), \
         patch('algorithms.run_gpbr.MyJobInfo.run', side_effect=gpbr_run_wrapper, autospec=True), \
         patch('algorithms.run_esga.MyJobInfo.run', side_effect=esga_run_wrapper, autospec=True), \
         patch('algorithms.run_wzs.MyJobInfo.run', side_effect=wzs_run_wrapper, autospec=True), \
         patch('runner.process_info.JobInfo.data_status_ok', side_effect=dummy_data_status_ok, autospec=True):
        url = '/cv/{}/job_start/wf_{}_RefreshFlow/'.format((ws.id), (ws.id))
        driver.get('{}{}'.format((live_prefix), (url)))

        split_drugset_name = 'split-train-' + ws.eval_drugset
        from dtk.kt_split import get_split_drugset
        get_split_drugset(split_drugset_name, ws)

        # The code below has two ways of clicking.  CTRL+Click is supposed
        # to be the way to unselect one in a multiselect, but it seems like
        # normal click is what does it in selenium/firefox/Xvfb.

        # gPath GPBR fails.
        gpath_el = driver.find_element_by_xpath("//select[@name='refresh_parts']/option[text()='gPath']")
        #ActionChains(driver).key_down(Keys.CONTROL).click(gpath_el).key_up(Keys.CONTROL).perform()
        gpath_el.click()

        # Test-train is now on by default, and runs WZS jobs on both the
        # test and the train dataset (simulating cross-fold validation)
        num_expected_wzs = 3
        if not test_train:
            # Turn off test/train wzs.
            testtrain_el = driver.find_element_by_name('eval_testtrain')
            testtrain_el.click()
            num_expected_wzs = 1


        # Default is MoAs, so only modify if we're using mols.
        if use_mols:
            # non-moa version of DPI
            sel = Select(driver.find_element_by_name('p2d_file'))
            sel.select_by_value('e2e_test.v1')
            # non-moa version of KTs
            sel = Select(driver.find_element_by_name('eval_ds'))
            sel.select_by_value('tts')

        num_wzs_jobs_before = len(ws.get_prev_job_choices('wzs'))

        click_and_wait(driver, name='run_btn')
        # I wouldn't normally expect this to take 10 minutes, but the first time
        # you run it on a fresh machine or after clearing /tmp it might.
        p_id = wait_for_background_job(expect_success=True, timeout_secs=10*60, driver=driver)

        num_wzs_jobs_after = len(ws.get_prev_job_choices('wzs'))
        # We run both the normal and split test/train wzs jobs.
        assert num_wzs_jobs_after == num_wzs_jobs_before + num_expected_wzs

        wzs_id = ws.get_prev_job_choices('wzs')[0][0]
        wzs_bji = JobInfo.get_bound(ws, wzs_id)
        score_weights, sources = wzs_bji.get_score_weights_and_sources()
        score_weights_dict = dict(score_weights)
        print("Score weights:", score_weights_dict)
        dgn_scores = [x for x in score_weights_dict.keys() if 'dgn' in x]
        print("Found DGN scores: ", dgn_scores)
        assert len(dgn_scores) > 0, "No DGN score in %s" % list(score_weights_dict.keys())

        knowndrug_codes_scores = [x for x in score_weights_dict.keys()
                                    if 'knowndrug_otarg_codes' in x]
        print("Found knowndrugs codes scores: ", knowndrug_codes_scores)
        assert len(knowndrug_codes_scores) == 0, \
                "Found unwanted kd codes score in %s" % list(score_weights_dict.keys())

def check_misc(client, driver, live_server, **kwargs):
    # A place to check various things of interest once we have a 'working'
    # workspace that has run workflows and such.

    ws = get_only_ws()
    live_prefix = live_server.url

    # Check the xws_refresh page, needs a button click so pages doesn't visit.
    ps_id = ws.get_prev_job_choices('path')[0][0]
    url = '/cv/{}/xws_cmp/?metric=SigmaOfRank&score={}_direct&ds=kts'.format((ws.id), (ps_id))
    assert_good_response(client.get(url))

    # Not sure why the ?all page isn't visited by check_pages, but the recalc
    # button wouldn't be tested anyway.
    url = '/cv/ws_cmp/?all=1'.format()
    assert_good_response(client.get(url))
    assert_good_response(client.post(url, {'recalc_btn': True}))

    # Page behind a button.
    url = '/cv/{}/ps_cmp/'.format((ws.id))
    driver.get('{}{}'.format((live_prefix), (url)))
    click_and_wait(driver, name='display_btn')

    # Page behind a button.
    url = '/cv/{}/score_cmp/wsa/?y={}_direct&x={}_indirect&ds=tts'.format((ws.id), (ps_id), (ps_id))
    assert_good_response(client.get(url))

    url = '/cv/{}/scoreplot/wsa/?ds=tts&score={}_direct&raise_exc=True'.format((ws.id), (ps_id))
    assert_good_response(client.get(url))

    # Check indirect trgscrimp
    from browse.models import WsAnnotation
    wsa = WsAnnotation.objects.filter(ws=ws)[0]
    url = '/mol/{}/ind_trg_imp/{}/'.format((ws.id), (wsa.id))
    assert_good_response(client.get(url))
    driver.get('{}{}'.format((live_prefix), (url)))
    # Switch to LOO, it's faster and nothing else tests it.
    from selenium.webdriver.support.ui import Select
    rec_sel = Select(driver.find_element_by_name('method'))
    rec_sel.select_by_visible_text('Leave one out')
    driver.find_element_by_name('use_cache').click()
    click_and_wait(driver, name='calc_btn')


    # Check that if you load the FVS from the refresh workflow, DGN is
    # selected by default.
    # This was an issue due to naming differences when dgn was run via
    # refresh.
    # We're picking the 2nd most recent, because the most recent is novelty.
    fvs_id = ws.get_prev_job_choices('fvs')[1][0]
    url = '/cv/{}/job_start/fvs_wsa_efficacy_{}/{}/'.format((ws.id), (ws.id), (fvs_id))
    driver.get('{}{}'.format((live_prefix), (url)))
    # Click reload default, sometimes the right scores aren't showing up
    # otherwise (but only as part of the full test), possibly session related
    click_and_wait(driver, name='dflt_btn')
    dgn_codes1 = driver.find_elements_by_name('feat_dgn_codes_codesMax')
    dgn_codes2 = driver.find_elements_by_name('feat_dgns_dgn_codes_codesMax')

    els = dgn_codes1 + dgn_codes2
    assert len(els) == 1, "No DGN elements showed up"

    assert els[0].is_selected(), "DGN wasn't selected by default"

    # Suitability page is behind a button.
    url = f'/suitability/?disp_ws={ws.id}&ws={ws.id}'
    assert_good_response(client.get(url))

    # Pathways page behind a button and select.
    glf_id = ws.get_prev_job_choices('glf')[1][0]
    url = f'/{ws.id}/pathways/?glf_jobs={glf_id}'
    assert_good_response(client.get(url))

    # Proteins page behind a button and select.
    sigdif_id = ws.get_prev_job_choices('sigdif')[1][0]
    url = f'/{ws.id}/proteinscores/?jobs_and_codes={sigdif_id}_difEv'
    rsp = client.get(url)
    assert_good_response(rsp)
    assert_good_dyn_content(rsp, client=client, num_expected=5)

    # Hit clustering page behind a button.
    pathway_id = ws.get_prev_job_choices('glf')[0][0]
    prot_id = ws.get_prev_job_choices('gesig')[0][0]
    url = f'/rvw/{ws.id}/hitclusters/?apply_dis_score=False&ds=tts&path_sig_id={pathway_id}&prot_sig_id={prot_id}_ev&ps=&top_n_prots=500&top_n_pws=200'
    assert_good_response(client.get(url))



def check_review_wf(driver, live_server, **kwargs):
    ws = get_only_ws()
    wzs_id = ws.get_prev_job_choices('wzs')[0][0]
    live_prefix = live_server.url

    url = '/cv/{}/job_start/wf_{}_ReviewFlow/'.format((ws.id), (ws.id))
    driver.get('{}{}'.format((live_prefix), (url)))

    # Disable parts we don't want to test
    for part in [
            'Selectability', # no tooling built for testing
            'Single Protein Scores', # redundant with refresh_wf
            ]:
        xpath = f"//select[@name='review_parts']/option[text()='{part}']"
        option_el = driver.find_element_by_xpath(xpath)
        option_el.click()

    # These are no longer needed, prepopulated.
    #set_input_to(driver.find_element_by_name('ordering_jid'), wzs_id)
    #set_input_to(driver.find_element_by_name('ordering_score'), 'wzs')

    set_input_to(driver.find_element_by_name('flag_count'), 5)


    num_trgscrimp_jobs_before = len(ws.get_prev_job_choices('trgscrimp'))
    num_ml_jobs_before = len(ws.get_prev_job_choices('ml'))

    # Hook in for our custom settings.
    from algorithms.run_lbn import MyJobInfo as lbn_job
    orig_lbn_run = lbn_job.run_lbn
    def lbn_run_wrapper(self):
        self.parms['count'] = 23
        orig_lbn_run(self)

    from algorithms.run_ml import MyJobInfo as ml_job
    orig_ml_run = ml_job.run
    def ml_run_wrapper(self):
        self.parms['inner'] = 2
        self.parms['outer'] = 2
        orig_ml_run(self)

    with patch('algorithms.run_lbn.MyJobInfo.run_lbn', side_effect=lbn_run_wrapper, autospec=True), \
         patch('algorithms.run_ml.MyJobInfo.run', side_effect=ml_run_wrapper, autospec=True), \
         patch('dtk.entrez_utils.PubMedSearch') as pubmed_cons:
        pubmed_cons.return_value.count_frequency.return_value = 5
        pubmed_cons.return_value.size.return_value = 10000

        click_and_wait(driver, name='run_btn')
        p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)

    num_trgscrimp_jobs_after = len(ws.get_prev_job_choices('trgscrimp'))
    num_ml_jobs_after = len(ws.get_prev_job_choices('ml'))

    assert num_trgscrimp_jobs_after == num_trgscrimp_jobs_before + 1, "Expected 1 more trgscrimp"
    assert num_ml_jobs_after == num_ml_jobs_before + 1, "Expected 1 more ml"

def check_cand_wf(driver, live_server, **kwargs):
    ws = get_only_ws()
    live_prefix = live_server.url

    url = '/cv/{}/job_start/wf_{}_CandidateFlow/'.format((ws.id), (ws.id))
    driver.get('{}{}'.format((live_prefix), (url)))

    num_dnprecompute_before = len(ws.get_prev_job_choices('dnprecompute'))
    num_trgscrimp_jobs_before = len(ws.get_prev_job_choices('trgscrimp'))
    click_and_wait(driver, name='run_btn')
    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)

    num_dnprecompute_after = len(ws.get_prev_job_choices('dnprecompute'))
    num_trgscrimp_jobs_after = len(ws.get_prev_job_choices('trgscrimp'))
    assert num_dnprecompute_after == num_dnprecompute_before + 1, "Expected 1 more dnprecompute"
    assert num_trgscrimp_jobs_after == num_trgscrimp_jobs_before + 1, "Expected 1 more trgscrimp"

def check_prescreen(driver, live_server, **kwargs):
    ws = get_only_ws()
    wzs_id = ws.get_prev_job_choices('wzs')[0][0]
    live_prefix = live_server.url

    url = '/cv/{}/prescreen_list/'.format((ws.id))
    driver.get(live_prefix+url)
    driver.find_element_by_name('name').send_keys('Test Screen')
    # The default is now selectability scores, which we don't have here.
    driver.find_element_by_name('primary_score').send_keys(f'{wzs_id}_wzs')

    from browse.models import Prescreen
    assert len(Prescreen.objects.all()) == 0
    click_and_wait(driver, name='add_btn')
    assert len(Prescreen.objects.all()) == 1

    pscr_id = Prescreen.objects.all()[0].id

    url = '/cv/{}/scoreboard/?prescreen_id={}'.format((ws.id), (pscr_id))
    driver.get(live_prefix+url)

    click_and_wait(driver, name='ct_shortcut_btn')
    # Why is this missing? Because this piece is configured in
    # the workspace.
    #click_and_wait(driver, name='demerit_shortcut_2_btn')
    click_and_wait(driver, name='ct_shortcut_btn')

    url = '/{}/review/'.format((ws.id))
    driver.get(live_prefix+url)

def check_review_round(driver, client, live_server, **kwargs):
    ws = get_only_ws()
    wzs_id = ws.get_prev_job_choices('wzs')[0][0]
    live_prefix = live_server.url

    # Create the reviewers group, and add the user to it.
    from django.contrib.auth.models import User, Permission, Group
    reviewers_group = Group.objects.create(name='reviewers')
    user = User.objects.get(username=username)
    user.groups.add(reviewers_group)
    user.save()

    # We shouldn't start with any elections.
    from browse.models import Election
    assert len(Election.objects.all()) == 0

    # Create our first election.
    url = '/rvw/{}/election/0/?flavor=pass1'.format((ws.id))
    assert_good_response(client.get(url))
    driver.get(live_prefix+url)
    driver.find_element_by_name('due_date').send_keys('01/01/2999')
    click_and_wait(driver, name='save_btn')

    # Make sure it has been created.
    assert len(Election.objects.all()) == 1
    elec = Election.objects.all()[0]

    # Visit the election page, and visit our first drug.
    url = '/rvw/{}/election/{}/'.format((ws.id), (elec.id))

    from browse.models import WsAnnotation
    enum = WsAnnotation.indication_vals
    drugs = WsAnnotation.objects.filter(
                ws=ws,
                indication=enum.INITIAL_PREDICTION)
    assert len(drugs) > 0, "Need candidates for this to work"
    drug_votes = ['No'] * len(drugs)
    drug_votes[0] = 'Yes'
    drug_names = [drug.agent.canonical for drug in drugs]
    print("Let's vote!", list(zip(drug_names, drug_votes)))


    for drug_substr, vote in zip(drug_names, drug_votes):
        driver.get(live_prefix+url)
        drug_link = driver.find_element_by_partial_link_text(drug_substr)
        drug_url = drug_link.get_attribute('href')

        driver.get(drug_url)

        # Make a recommendation on the drug.
        uncollapse_all(driver)
        from selenium.webdriver.support.ui import Select
        rec_sel = Select(driver.find_element_by_name('recommended'))
        rec_sel.select_by_visible_text(vote)
        click_and_wait(driver, name='vote_btn')

    driver.get(live_prefix+url)

    num_reviewed_drugs = lambda: len(WsAnnotation.objects.filter(indication=enum.REVIEWED_PREDICTION))

    assert num_reviewed_drugs() == 0, "Start with no reviewed drugs"
    primary_btn = driver.find_elements_by_name('primary_shortcut_btn')[0]
    primary_btn.click()
    WebDriverWait(driver, 180).until(EC.staleness_of(primary_btn))
    assert num_reviewed_drugs() == 1, "After button should have 1 reviewed drugs"


def assert_good_dyn_content(rsp, client, num_expected):
    """Checks all dynamically loaded divs on page and makes sure their URLs load correctly"""
    hrefs = []
    from lxml import html, etree
    dom = html.fromstring(rsp.content)
    for e in dom.iter('div'):
        # We use a custom attribute to indicate the URL for dynamically
        # expandable sections.
        content_url = e.get('content_url')
        if content_url:
            hrefs.append(content_url)

    if num_expected is not None:
        assert len(hrefs) == num_expected

    for href in hrefs:
        dyn_rsp = client.get(href, follow=True)
        assert_good_response(dyn_rsp)

def check_pages(driver, client, live_server, **kwargs):
    # Now that we have our workspace all setup with jobs and scores,
    # let's just go and open all the pages we expect to work.
    # We'll just use everything you can reach from the workflow page as a
    # baseline..
    ws = get_only_ws()
    live_prefix = live_server.url

    url = '/?all=1'
    consultant_url = '/consultant/'

    hrefs = set([url])
    from collections import defaultdict
    stem_count = defaultdict(int)
    new_hrefs = [url, consultant_url]

    refs = defaultdict(set)

    to_skip = [
            '/publish/', # Don't need to test static files
            '/logout/', # Let's not logout
            '/score_metrics/', # Not sure why broken here
            '/gwas_qc/', # Missing hg38.chrom.sizes.tsv ?
            '/static/', # Missing /static/disGeNet_disease_names.txt XXX remove?
            '/ind_trg_imp/', # Broken due to non-full prot set?
            '/trgscrimp/', # This is slow and we test it elsewhere.
            '/matching/', # This is slow and we test it elsewhere.
            '/admin/', # django admin stuff, don't bother
            '/dashboard/', # static file redirect, uses different attr for content
            '/pstatic/', # static file redirect, uses different attr for content
            '/etl_history/', # these end up downloading all versions of all data, tested separately
        ]

    MAX_PER_STEM = 5

    def add_hrefs(rsp, from_url):
        def add_href(href):
            if not href or not href.startswith('/'):
                return
            skips = [True for x in to_skip if (x in href)]
            if not skips:
                uniq_href = href
                if uniq_href not in hrefs:
                    hrefs.add(uniq_href)

                    parts = href.split('/')
                    if parts[1] == str(ws.id):
                        stem = '/'.join(parts[:3])
                    else:
                        stem = '/'.join(parts[:4])

                    stem_count[stem] += 1
                    if stem_count[stem] > MAX_PER_STEM:
                        return
                    new_hrefs.append(href)
                    refs[href].add(from_url)

        from lxml import html, etree
        dom = html.fromstring(rsp.content)
        for e in dom.iter('a'):
            href = e.get('href')
            add_href(href)

        for e in dom.iter('div'):
            # We use a custom attribute to indicate the URL for dynamically
            # expandable sections.
            content_url = e.get('content_url')
            if content_url:
                add_href(content_url)

    times = []
    i = 0
    while i < len(new_hrefs):
        href = new_hrefs[i]
        local_href = href
        print("Try out %s (%d / %d) [%s]" % (local_href, i, len(new_hrefs), refs[href]))
        import time
        start = time.time()
        rsp = client.get(local_href, follow=True)
        duration = time.time() - start
        times.append([duration, local_href])
        assert_good_response(rsp)

        add_hrefs(rsp, href)
        i += 1

    times.sort(key=lambda x: -x[0])
    print("Slowest pages:")
    for entry in times[:30]:
        print(f'- {entry}')
# The test is broken up into these sequential steps.
# You can filter to run only the desired steps if needed
# e.g. pytest -k "check_pathsum or check_gesig"
# This requires that the previous steps have successfully completed before and
# created a snapshot.
TEST_PARTS = [
    setup_workspace,
    import_drugs,
    check_workspace,
    import_tissue_meta,
    import_tissue_sig,
    check_tissue_views,
    check_pathsum,
    check_gesig,
    check_codes,
    check_gpbr,
    check_glee,
    check_glf,
    check_depend,
    check_gwas,
    check_esga,
    check_gpath,
    check_gwasig,
    check_faers,
    check_capp,
    check_kt_search,
    check_drug_eff,
    check_wzs,
    check_trgscrimp,
    # We run defus after trgscrimp because the defus trgimp is still quite
    # slow.  It doesn't normally run because it gets aggregated in the workflow.
    check_defus,
    check_run_flag,
    check_external,
    check_lbn,
    check_novelty_fvs,
    check_ml,
    check_refresh_wf,
    check_review_wf,
    check_prescreen,
    check_review_round,
    check_cand_wf,
    check_pages,
    check_misc,
    check_refresh_wf_mols,
    ]

def get_snapshot_dir(fn):
    name = fn.__name__
    if six.PY3:
        name += "_py3"
    return os.path.join(TMPDIR, 'snapshots/{}'.format(name))


# We want to share our s3 cache across the test.
# We don't want to re-use the real one, because we might stick some test
# data into there.
s3_cachedir = os.path.join(TMPDIR, 'teste2ecache/')

ph_opts = {'S3_CACHE_DIR': s3_cachedir}


@pytest.mark.django_db(reset_sequences=True, transaction=True)
@patch("dtk.lts.LtsRepo")
@patch("plotly.io.write_image", MagicMock()) # png writing is slow!
@patch("dtk.prot_map.PpiMapping.preferred", PPI_NAME)
@pytest.mark.parametrize('tmpdir_path_helper', [ph_opts], indirect=True)
@pytest.mark.parametrize('test_part', TEST_PARTS)
def test_e2e(lts_repo, tmpdir_path_helper, mock_remote_machine,
        client, django_user_model, admin_client, tmpdir,
        live_server, selenium, test_part):
    test_part_name = test_part.__name__
    print("Running e2e test on %s, server lives at %s" % (test_part_name, live_server.url))


    # If this isn't the very first step, we should load the state snapshot
    idx = TEST_PARTS.index(test_part)

    prev_part = TEST_PARTS[idx-1] if idx > 0 else None
    if prev_part:
        load_snapshot(get_snapshot_dir(prev_part))

    # Make sure all our mocks are in place and we're logged in.
    setup_environment()
    setup_users(django_user_model, client)
    driver = selenium
    login_selenium(driver, live_server.url)

    cassette_name = test_part_name + ('' if six.PY2 else 'py3')
    # Run the test part.
    with std_vcr.use_cassette('dtk/testdata/e2e_{}_tape.yaml'.format(cassette_name)):
        test_part(client=client, admin_client=admin_client, live_server=live_server, driver=driver)

    # Save the state now that we're done.
    snapshot_to(get_snapshot_dir(test_part))


def setup_users(django_user_model, client):
    if len(django_user_model.objects.filter(username=username)) == 0:
        user_obj = django_user_model.objects.create_superuser(
                username=username,
                password=password,
                email="not-an-email")
    else:
        user_obj = django_user_model.objects.filter(username=username)[0]


    assert user_obj.is_staff

    mark_user_access_known(user_obj)
    admins = django_user_model.objects.filter(username='admin')
    for admin_user in admins:
        mark_user_access_known(admin_user)

    client.login(username=username, password=password)


def check_global_pages(client, admin_client, driver, live_prefix):
    # Check that we can visit / and /upload before anything is setup.
    assert_good_response(client.get('/'))
    assert_good_response(client.get('/upload/'))

    # Check that we have a working admin client.
    assert_good_response(admin_client.get('/admin/'))

    driver.get('{}/'.format((live_prefix)))
    nav_els = driver.find_elements_by_class_name('nav')
    assert len(nav_els) > 0, "Why does our page have no nav elements?"
    driver.get('{}/upload/'.format((live_prefix)))

def create_workspace(client, admin_client):
    from browse.models import Workspace

    workspaces = Workspace.objects.all()
    assert len(workspaces) == 0

    # We have no workspaces, make sure this fails.
    with pytest.raises(Exception):
        client.get('/1/workflow/')

    print("Creating workspace")

    # Create a workspace
    create_ws_opts = {
            'name': 'sjogren',
            }
    rq = admin_client.post('/admin/browse/workspace/add/', create_ws_opts, follow=True)
    assert rq.status_code == 200
    workspaces = Workspace.objects.all()
    assert len(workspaces) == 1
    ws = workspaces[0]

    import browse.default_settings as ds
    ds.PpiDataset.set(ws, PPI_NAME, 'test')
    ds.DpiDataset.set(ws, 'e2e_test-moa.v1', 'test')
    # Also set global default, a couple pages use this.
    ds.DpiDataset.set(None, 'e2e_test-moa.v1', 'test')
    ds.EvalDrugset.set(ws, 'moa-tts', 'test')

    return ws.id

def check_empty_workspace_pages(client, ws_id):
    print("Checking pages in a newly created workspace")

    from browse.models import Tissue, TissueSet

    assert len(TissueSet.objects.all()) == 0
    # Visiting this page for the first time creates a default & miRNA tissueset for us.
    assert client.get('/{}/workflow/'.format((ws_id))).status_code == 200
    assert len(TissueSet.objects.all()) == 2

    assert client.get('/ge/{}/tissues/'.format((ws_id))).status_code == 200
    assert client.get('/cv/{}/disease_names/'.format((ws_id))).status_code == 200
    assert client.get('/ge/{}/ae_search/'.format((ws_id))).status_code == 200
    assert client.get('/cv/{}/scoreboard/'.format((ws_id))).status_code == 200
    assert client.get('/rvw/{}/review/'.format((ws_id))).status_code == 200


def check_import_drugs(client):
    print("Importing drug properties")
    # Refresh properties first, or we won't be able to load any drugs in.
    opts = {
            'prop_refresh_btn': True
        }
    assert client.post('/upload/', opts, follow=True).status_code == 200

    from drugs.models import Drug
    num_loaded_drugs = len(Drug.objects.all())
    assert num_loaded_drugs == 0, "How are there drugs before we've imported any?"

    print("Importing ncats")
    # Load in a small drug set.
    opts = {
            'versioned_drug_upload_btn': True
        }

    from drugs.tools import CollectionUploadStatus
    cols = [
        'ncats.full',
        'moa.full',
    ]
    with patch.object(CollectionUploadStatus, 'maintained_collections', cols):
        assert client.post('/upload/', opts, follow=True).status_code == 200

    # Check that we have drugs in the DB now.
    print("Verifying import")
    from drugs.models import Drug
    num_loaded_drugs = len(Drug.objects.all())
    # If you change ncats, this number could change.
    # All we really care about is that it is small but non-zero.
    assert num_loaded_drugs == EXPECTED_DRUG_RECS


    from drugs.models import DpiMergeKey, UploadAudit
    assert len(DpiMergeKey.objects.all()) > 0, "DpiMergeKey imported"

    num_keys = len(set(DpiMergeKey.objects.all().values_list('dpimerge_key')))
    num_drugs = len(set(DpiMergeKey.objects.all().values_list('drug_id')))

    assert num_keys == num_drugs, "No clustering, should have same # of keys and drugs, a non-test-cluster file may have imported"

    assert len(UploadAudit.objects.all()) > 0
    for ua in UploadAudit.objects.all():
        assert ua.ok == True


def check_import_prots(client):
    print("Importing proteins")
    from browse.models import Protein
    assert len(Protein.objects.all()) == 0
    opts = {
            'prot_refresh_btn': True,
            'prot_file':'HUMAN_9606.v1',
            }
    assert client.post('/upload/', opts, follow=True).status_code == 200

    Protein.objects.get(uniprot='P03372')

    # This appears to be an alt-uniprot of P03372.  However, in the production
    # DB it appears to be a real Protein object, despite it not showing up
    # as such in the HUMAN_9606_Uniprot_data file.
    # Possibly it existed in an older version and our retention logic kept it.
    # I've manually added it to the uniprot data for this test for now.
    # One of the drug reports we generate expects this prot to exist, so
    # let's make sure it does.
    Protein.objects.get(uniprot='Q9UBT1')


def check_prot_search(client, ws_id):
    print("Checking protein search")
    assert client.get('/{}/prot_search/'.format((ws_id))).status_code == 200
    search_opts = {'search': 'CNR2'}
    rq = client.post('/{}/prot_search/'.format((ws_id)), search_opts)
    assert rq.status_code == 200

def check_search_page(client, ws_id):
    print("Checking search")
    assert client.get('/rvw/{}/review/?search=AZD'.format((ws_id))).status_code == 200

def check_prot_search_api(client):
    rq = client.get('/api/prot_search/?search=FAKE_PROT')
    assert rq.status_code == 200
    assert json.loads(rq.content) == {'matches': [], 'reached_limit': False}

    rq = client.get('/api/prot_search/?search=CNR2')
    assert rq.status_code == 200
    expected = [
            {'gene': 'CNR2', 'uniprot': 'P34972', 'name': 'Some Name'},
            ]
    output = json.loads(rq.content)
    output['matches'].sort()
    assert output == {'matches': sorted(expected), 'reached_limit': False}

def check_prot_page(client, ws_id):
    assert client.get('/{}/protein/P34972/'.format((ws_id))).status_code == 200
    assert client.get('/{}/prot_detail/P34972/'.format((ws_id))).status_code == 200


def check_search_tissues(client, ws_id):
    search = "Sjogren syndrome"
    mode = "CC"
    from browse.models import AeSearch
    opts = {
        'run_btn': True,
        'search_term': '"sjogren syndrome"',
        'mode': 0,
        'species': AeSearch.species_vals.human,
        }

    # Make sure the first ae_list doesn't exist yet.
    with pytest.raises(Exception):
        client.get('/ge/{}/ae_list/1/'.format((ws_id)))

    # Do the search via CM
    url = f'/cv/{ws_id}/job_start/aesearch_{ws_id}/'
    resp = client.get(url)
    assert resp.status_code == 200 # make sure job start page loads
    # run search
    assert client.post(url, opts, follow=True).status_code == 200
    wait_for_background_job(expect_success=True, timeout_secs=60*30)

    # verify output
    searches = AeSearch.objects.all()
    assert len(searches) == 1
    search_id = searches[0].id

    # Now we have one.
    # TODO: Log output here suggests that we try to run Weka and fail.
    # However, that failure doesn't seem to get surfaced right now.
    # Maybe it's visible in the page HTML somehow, but might want to investigate.
    # Also, can fix by fetching AE_CC_search.model from s3 to our test ws.
    assert client.get('/ge/{}/ae_list/{}/'.format((ws_id), (search_id))).status_code == 200


def check_import_workspace_drugs(client, ws_id):
    # Check that this page doesn't work yet, no drugs in the workspace.
    with pytest.raises(Exception):
        client.get('/mol/{}/annotate/1/'.format((ws_id)))

    assert client.get('/cv/{}/col2/'.format((ws_id))).status_code == 200

    from browse.default_settings import DpiDataset
    from dtk.prot_map import DpiMapping
    dpi_name = DpiDataset.value(ws=ws_id)
    dpi = DpiMapping(dpi_name)
    assert not dpi.legacy, f"Workspace should not be using legacy DPI - {dpi_name}"

    from drugs.tools import CollectionUploadStatus

    from drugs.models import Collection
    collections = Collection.objects.all().values_list('name', flat=True)
    assert set(collections) == {'moa.full', 'ncats.full'}, "We should have only ncats and moa collections"
    opts = {
        'imports_btn': True,
        }
    for col in Collection.objects.all():
        opts[f'col{col.id}'] = True

    from drugs.models import UploadAudit
    from nav.views import Col2View
    from browse.models import Workspace

    assert client.post('/cv/{}/col2/'.format((ws_id)), opts, follow=True).status_code == 200

    wsas = WsAnnotation.objects.filter(ws_id=ws_id)
    assert len(wsas) == EXPECTED_DRUG_RECS,  "Should have imported all the drugs"
    wsa_id = wsas[0].id

    # Now that we've imported drugs, this page should work.
    assert client.get('/mol/{}/annotate/{}/'.format((ws_id), (wsa_id))).status_code == 200

    print("Checking matching page")
    # Let's also check the matching page
    assert client.get('/cv/{}/matching/{}/'.format((ws_id), (wsa_id))).status_code == 200



def get_tissue_views(client, ws_id):
    from browse.models import Tissue, TissueSet
    tissue_sets = TissueSet.objects.all()
    assert len(tissue_sets) == 2
    ts_id = tissue_sets[0].id
    assert client.get('/ge/{}/tissues/'.format((ws_id))).status_code == 200
    assert client.get('/ge/{}/tissue_set/{}/'.format((ws_id), (ts_id))).status_code == 200
    assert client.get('/ge/{}/tissue_stats/{}/'.format((ws_id), (ts_id))).status_code == 200
    assert client.get('/ge/{}/tissue_corr/?tissue_set_id={}'.format((ws_id), (ts_id))).status_code == 200
    assert client.get('/ge/{}/tissue_set_analysis/?tissue_set_id={}'.format((ws_id), (ts_id))).status_code == 200

def check_drug_views(client, ws_id):
    from browse.models import WsAnnotation
    wsas = WsAnnotation.objects.filter(ws_id=ws_id)
    wsa_id = wsas[0].id

    assert client.get('/mol/{}/annotate/{}/'.format((ws_id), (wsa_id))).status_code == 200



def mark_user_access_known(user):
    """Marks the user access as known, so that the IPMiddleware doesn't spam slack."""
    from browse.models import UserAccess
    hosts = ["127.0.0.1", 'localhost']
    types = ['unverified', 'normal']
    for host in hosts:
        for access_type in types:
            UserAccess.objects.get_or_create(
                    user = user,
                    host = host,
                    access = access_type,
                    )

from queue import Queue
job_thread_queue = Queue()
background_exceptions = []

def wait_for_background_job(expect_success, timeout_secs=10, driver=None, launch_timeout_secs=5):
    if driver:
        # Usually we'll be at a progress page, go to a blank page instead to avoid
        # wasting lots of time on refreshing progress.
        driver.get('about:blank')
    # This will throw if it times out.
    job_thread, p_id = job_thread_queue.get(timeout=launch_timeout_secs)

    try:
        # This will not throw if it times out, have to check is_alive.
        job_thread.join(timeout=timeout_secs)

        assert not job_thread.is_alive(), "If this is still alive, we timed out"

        if expect_success:
            if len(background_exceptions) > 0:
                raise background_exceptions.pop()
            from runner.models import Process
            assert Process.objects.get(id=p_id).status == Process.status_vals.SUCCEEDED

        else:
            assert len(background_exceptions) > 0
    finally:
        # This can happen in workflow-based tests, which like to spawn a
        # bunch of subjobs (which we aren't explicitly waiting on)
        # Also, if a test fails, we want to make sure it doesn't impact later.
        while not job_thread_queue.empty():
            print("Clearing out status of other background jobs")
            job_thread_queue.get()
        while background_exceptions:
            e = background_exceptions.pop()
            print("Clearing exception ", e)

    return p_id

def setup_environment():
    from dtk.lts import LtsRepo
    assert isinstance(LtsRepo, MagicMock), "LTS should be mocked out here"

    from path_helper import PathHelper
    ws = PathHelper.s3_cache_root
    assert ws.startswith(s3_cachedir), "This setup is intended for fake WS, not " + ws
    assert PathHelper.dpi.startswith(s3_cachedir), "DPI should point to shared" + PathHelper.dpi

    testdata_dir = 'dtk/testdata/e2e_dataset/'

    all_ftypes = set()
    for fn in os.listdir(testdata_dir):
        ftype = fn.split('.')[0]
        all_ftypes.add(ftype)

        src = os.path.join(testdata_dir, fn)
        dst = os.path.join(ws, ftype, fn)

        from path_helper import make_directory
        make_directory(os.path.join(ws, ftype))
        import shutil
        shutil.copyfile(src, dst)

    import gzip
    import json
    # Hardcode this file for now, maybe have some test data later.
    for dest in (
            'uniprot/uniprot.HUMAN_9606.v1.Protein_Names.json.gz',
            ):
        with gzip.open(os.path.join(ws, dest), 'w') as f:
            data = [
                    {'uniprots': ["P14416", "A0A1Y8EK52"],
                        'full_name': "D(2) dopamine receptor",
                        'alt_names': ['altname1', 'altname2']
                    }, {
                        'uniprots': ["Q5T211", "Q5T210"],
                        'full_name': 'Nicastrin',
                        'alt_names': ['APH2']
                    }, {
                        'uniprots': ["P34972"],
                        'full_name': 'Some Name',
                        'alt_names': ['APH99']
                    }
                    ]
            f.write(json.dumps(data).encode('utf8'))

    # Fill in listcache
    for ftype in all_ftypes:
        fdir = os.path.join(ws, ftype)
        with open(os.path.join(fdir, '__list_cache'), 'w') as f:
            paths = [x for x in os.listdir(fdir) if x.startswith(ftype)]
            f.write('\n'.join(paths))

    setup_lts_tmpdir()
    shim_background_jobs_in_process()

def setup_lts_tmpdir():
    from dtk.lts import LtsRepo
    from path_helper import PathHelper
    assert isinstance(LtsRepo, MagicMock), "This assumes we have a mock"
    LtsRepo.return_value.path.return_value = PathHelper.lts
    LtsRepo.get.return_value = LtsRepo.return_value

    lts_path = PathHelper.lts

    # Double-check that our mocks worked.
    repo = LtsRepo("A", "B")
    assert repo.path() == lts_path
    assert repo == LtsRepo.get("A", "B")
    assert LtsRepo.get("A", "B").path() == lts_path


def shim_background_jobs_in_process():
    """
    This hooks into the internals of the job running mechanisms to avoid
    running jobs in separate processes and instead run them in-process.
    This ensures we're still pointing at test databases / paths.

    We're effectively replacing run_process.py and drive_background.py with
    in-process versions.
    """

    # This has the potential to break selenium for the test, which uses
    # time.sleep to wait for the driver to spin up.
    #time_sleep_mock = mock.patch('time.sleep').start()

    background_driver_mock = patch('runner.models.Process.drive_background').start()
    background_wrapper_mock = patch('runner.models.background_wrapper').start()


    def test_drive_background():
        print("Looking for any background jobs to invoke")
        from runner.models import Process
        Process.start_all()
        return 'override'
    background_driver_mock.side_effect = test_drive_background


    def test_background_wrapper(p):
        # NOTE: Process.start_all() runs this on a new thread.
        print("Running a queued background job in-process: %s" % p)
        try:
            import threading
            job_thread_queue.put((threading.current_thread(), p.id))

            # Create the log directory.
            # TODO: Unify this with run_process.py, who we are replacing.
            from runner.common import LogRepoInfo
            lri = LogRepoInfo(p.id)
            logfile = p.logfile()
            assert logfile == lri.log_path()
            from path_helper import make_directory
            make_directory(os.path.dirname(logfile))

            import json
            settings = json.loads(p.settings_json)
            if 'ws_id' in settings:
                ws_id = settings['ws_id']
            elif 'tissue_id' in settings:
                from browse.models import Tissue
                tissue_id = settings['tissue_id']
                t = Tissue.objects.get(pk=tissue_id)
                ws_id = t.ws.id
            else:
                raise Exception("Don't know how to get a wsid for this job")
            from runner.process_info import JobInfo
            bji = JobInfo.get_bound(ws_id, p)
            bji.run()
            logger.info(f"Completed {p.id}, marking as successful")

            Process.stop(p.id, 0)
        except Exception as e:
            import traceback
            logger.error(f"Failed in thread for {p.id}, passing along to other thread: {traceback.format_exc()}")
            background_exceptions.append(Exception(traceback.format_exc()))
            Process.stop(p.id, 1)
    background_wrapper_mock.side_effect = test_background_wrapper

    # Double-check that our drive_background mock is in-place.
    from runner.models import Process
    assert Process.drive_background() == 'override'

    # Make sure that jobs can request resources.
    from reserve import ResourceManager
    from aws_op import Machine
    import multiprocessing
    # By default this method is hitting EC2 to check cpu count - let's just
    # get it locally for this test.
    patch.object(Machine, 'num_cores', lambda x: 10)

    ResourceManager().set_totals()

def snapshot_to(to_dir):
    print("Starting snapshot to ", to_dir)
    from path_helper import PathHelper, make_directory
    make_directory(to_dir)

    if False:
        db_file = os.path.join(to_dir, 'db.json')
        from django.core.management import call_command
        with open(db_file, 'wb') as f:
            call_command('dumpdata', stdout=f)
    else:
        db_file = os.path.join(to_dir, 'db.sql')
        with open(db_file, 'wb') as f:
            subprocess.check_call(['mysqldump', '-u', 'root', 'test_web1'], stdout=f)

    phdir = os.path.join(to_dir, 'ph')
    import shutil
    assert PathHelper.install_root.startswith(TMPDIR), \
            "Careful not to snapshot your real 2xar dir, that's big (%s)" % PathHelper.install_root
    if os.path.exists(phdir):
        assert phdir.startswith(TMPDIR), "You sure you want to delete %s?" % phdir
        shutil.rmtree(phdir)
    shutil.copytree(PathHelper.install_root, phdir, symlinks=True)

    # NOTE: I'd prefer not to have to 'image' the remotemch into the snapshot,
    # but it seems that certain things (e.g. meta & sig) assume they were run
    # on the same remote worker.
    to_remotedir = os.path.join(to_dir, 'remote_mch')
    from_remotedir = os.path.join(PathHelper.install_root, '../../remote_mch')
    if os.path.exists(to_remotedir):
        assert to_remotedir.startswith(TMPDIR), "You sure you want to delete %s?" % to_remotedir
        shutil.rmtree(to_remotedir)
    shutil.copytree(from_remotedir, to_remotedir, symlinks=True)
    print("Done snapshot to ", to_dir)

def load_snapshot(from_dir):
    print("Loading db snapshot")

    # Support either way of snapshotting, but prefer the .sql file because
    # it is faster.
    sql_file = os.path.join(from_dir, 'db.sql')
    if os.path.exists(sql_file):
        print("Loading from raw SQL file")
        with open(sql_file, 'r') as f:
            subprocess.check_call(['mysql', '-u', 'root', 'test_web1'], stdin=f)
    else:
        from django.db import connection
        from django.core.management import call_command
        db_file = os.path.join(from_dir, 'db.json')
        if not os.path.exists(db_file):
            pytest.skip("Skipping test, no snapshot data, previous part failed")
            raise Exception("Skipping test, no snapshot data")
        call_command('loaddata',  db_file)


    from path_helper import PathHelper
    assert PathHelper.install_root.startswith(TMPDIR), \
        "Careful not to overwrite your real 2xar dir (%s)" % PathHelper.install_root
    phdir = os.path.join(from_dir, 'ph')
    import shutil
    print("Loading snapshot from %s to %s" % (phdir, PathHelper.install_root))
    shutil.rmtree(PathHelper.install_root)
    shutil.copytree(phdir, PathHelper.install_root, symlinks=True)

    from_remotedir = os.path.join(from_dir, 'remote_mch')
    if os.path.exists(from_remotedir):
        to_remotedir = os.path.join(PathHelper.install_root, '../../remote_mch')
        print("Loading snapshot from %s to %s" % (from_remotedir, to_remotedir))
        # This version of copy_tree allows you to copy onto an existing dir.
        # This allows us to modify the remotemch more easily during iteration.
        from distutils.dir_util import copy_tree
        #shutil.rmtree(to_remotedir)
        copy_tree(from_remotedir, to_remotedir)

@pytest.fixture(scope='session')
def selenium():
    from pyvirtualdisplay import Display
    fb = Display(visible=0, size=(1600, 1200))
    fb.start()
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options

    options = Options()
    options.headless = True
    driver = webdriver.Firefox(options=options)

    yield driver

    driver.quit()
    fb.stop()


def login_selenium(driver, live_server_url):
    """
    Sometimes we POST/GET with the django test client and sometimes with selenium.
    Selenium is more convenient for posting complicated forms with lots of
    default values, and tests the actual user flow more accurately, but can be
    a bit brittle.
    """


    driver.get(live_server_url + '/account/login/')
    username_input = driver.find_element_by_name('auth-username')
    username_input.send_keys(username)
    password_input = driver.find_element_by_name('auth-password')
    password_input.send_keys(password)
    driver.find_elements_by_class_name('btn-primary')[0].click()
