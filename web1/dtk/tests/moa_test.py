import pytest
from mock import patch, MagicMock
from dtk.tests.tmpdir_path_helper import tmpdir_path_helper
from dtk.tests.end_to_end_test import setup_lts_tmpdir, wait_for_background_job, shim_background_jobs_in_process, assert_file_unchanged, assert_good_response
from dtk.tests.mock_remote import mock_remote_machine
from dtk.tests import make_ws, auth_client
from pathlib import Path
import os
import logging
logger = logging.getLogger(__name__)

MOLDPI = 'matching.testdpi.v1.dpimerge.tsv'
MOADPI = 'matching.testdpi-moa.v1.dpimerge.tsv'

# XXX These tests all take a few seconds, mostly because any of the default settings stuff ends up
# XXX making a whole bunch of AWS connections to list out buckets.
# XXX It's a bit tricky to mock those out, because they expect valid defaults for some.

@pytest.fixture()
def setup_fixture(tmpdir_path_helper, make_ws, tmpdir):
    # TODO: This should really be extracted into a more generic make_versioned_moa_ws utility, for
    # re-use in other tests.
    from dtk.prot_map import DpiMapping
    orig_get_path = DpiMapping.get_path
    with patch.object(DpiMapping, 'get_path', autospec=True) as dpi_get_path:
        from path_helper import PathHelper 
        TMPDIR = PathHelper.get_config()['bigtmp']

        # NOTE: Only write matching-expected files into this directory, otherwise
        # might hit asserts that get angry about unexpected files in there.
        # Use the tmpdir for other files.
        dpidir = Path(PathHelper.s3_cache_root) / 'matching'
        print("Using dpi path at ", dpidir)

        def get_path(self):
            if self.choice.startswith('cstm_'):
                logger.info(f"Falling back to original getpath for {self.choice}")
                return orig_get_path(self)
            out = str(dpidir / f'matching.{self.choice}.dpimerge.tsv')
            print(f"Returning dpi path of {out} for {self.choice}")
            return out
        dpi_get_path.side_effect = get_path

        fns = [MOLDPI, MOADPI]
        assert str(dpidir).startswith(TMPDIR)
        dpidir.mkdir(parents=True, exist_ok=True)
        with open(dpidir / '__list_cache', 'w') as f:
            f.write('\n'.join(fns))

        mols = [
            ['drugbank_id', 'uniprot_id', 'ev', 'dir'],
            ['DB001', 'P001', '0.5', '-1'],
            ['DB002', 'P003', '0.9', '0'],
            ['DB002', 'P001', '0.5', '1'],
            ['DB003', 'P001', '0.5', '-1'],
        ]

        u2g = [
            ['prot', 'name', 'val'],
            ['P001', 'Gene_Name', 'G001'],
            ['P002', 'Gene_Name', 'G002'],
            ['P003', 'Gene_Name', 'G003'],
        ]

        def write_tsv(fn, data):
            with open(fn, 'w') as f:
                for line in data:
                    f.write('\t'.join(line) + '\n')
        
        write_tsv(dpidir / MOLDPI, mols)

        write_tsv(tmpdir / 'u2g.tsv', u2g)

        transform_pgm = Path(PathHelper.databases) / 'matching' / 'dpi_transform.py'
        from subprocess import check_call
        check_call([
            transform_pgm,
            '-i', dpidir / MOLDPI,
            '-o', dpidir / MOADPI,
            '-v', 'v1',
            '-a', tmpdir / 'moa.full.tsv',
            '-u', tmpdir / 'u2g.tsv',
            '--rekey',
            ])
        #print("Check out ")
        #print(open(dpidir / MOADPI).read())

        
        # Import these molecules as drugs and wsas.
        ws = make_ws([
            ('DB001', 'canonical', 'Drug 1'),
            ('DB002', 'canonical', 'Drug 2'),
            ('DB003', 'canonical', 'Drug 3'),
        ])



        from drugs.models import Collection, UploadAudit, DpiMergeKey
        from dtk.files import get_file_records

        # Create dpimergekey records for everything.
        DpiMergeKey.fill_missing_dpimerge('1')

        moa_coll = Collection.objects.create(name='moa.full', key_name='moa_id')
        moa_coll.fast_create_from_records(
            get_file_records(str(tmpdir / 'moa.full.tsv')),
            'moa.full.v1.tsv',
            'moa.full',
            'moa_id'
        )
        UploadAudit.objects.create(filename='moa.full.v1.tsv', ok=True)

        from browse.default_settings import DpiDataset
        DpiDataset.set(ws, 'testdpi-moa.v1', 'testuser')

        ws.import_collection(moa_coll, 'test')
        yield ws


def test_setup_fixture(setup_fixture):
    # The setup fixture is quite complex, let's first check that is has set things up nicely.
    ws = setup_fixture

    from browse.models import WsAnnotation
    from drugs.models import Drug
    assert WsAnnotation.objects.all().count() == 5, "3 mols, 2 moas"
    assert Drug.objects.all().count() == 5, "3 mols, 2 moas"

def test_moa_variant():
    from dtk.moa import moa_dpi_variant
    assert moa_dpi_variant('DNChBX_ki.v16') == 'DNChBX_ki-moa.v16'
    assert moa_dpi_variant('DNChBX_ki-moa.v16') == 'DNChBX_ki-moa.v16'
    assert moa_dpi_variant('many.parts.of.dots') == 'many-moa.parts.of.dots'


def test_wsa_to_moa_agent():
    pass

def test_wsa_to_moa_wsa(setup_fixture):
    ws = setup_fixture
    from browse.models import WsAnnotation
    from dtk.moa import make_wsa_to_moa_wsa
    mol1 = WsAnnotation.objects.filter(ws=ws, agent__tag__value='DB001').distinct()[0]

    wsa2moawsa = make_wsa_to_moa_wsa([mol1.id], pick_canonical=True)
    assert len(wsa2moawsa) == 1
    
    moa1_id = wsa2moawsa[mol1.id]
    moa1 = WsAnnotation.objects.get(pk=moa1_id)

    # DB001 has only 1 prot, P001 with direction -1.
    assert moa1.agent.canonical == 'G001-'



    from dtk.prot_map import AgentTargetCache, DpiMapping
    from browse.default_settings import DpiDataset
    moa_dpi = DpiMapping(DpiDataset.value(ws=ws))
    base_dpi = moa_dpi.get_baseline_dpi()

    atc_mol = AgentTargetCache.atc_for_wsas(wsas=[mol1], dpi_mapping=base_dpi, dpi_thresh=0.5)
    atc_moa = AgentTargetCache.atc_for_wsas(wsas=[moa1], dpi_mapping=moa_dpi, dpi_thresh=0.5)

    assert len(atc_moa.all_prots) == 1
    assert atc_mol.all_prots == atc_moa.all_prots




def test_update_moa_indications(setup_fixture):
    ws = setup_fixture
    from browse.models import WsAnnotation
    iv = WsAnnotation.indication_vals
    from dtk.moa import make_wsa_to_moa_wsa, update_moa_indications
    mol1 = WsAnnotation.objects.filter(ws=ws, agent__tag__value='DB001').distinct()[0]

    wsa2moawsa = make_wsa_to_moa_wsa([mol1.id], pick_canonical=True)
    assert len(wsa2moawsa) == 1
    
    moa1_id = wsa2moawsa[mol1.id]
    moa1 = WsAnnotation.objects.get(pk=moa1_id)
    assert moa1.indication == iv.UNCLASSIFIED

    mol1.update_indication(iv.INITIAL_PREDICTION, user='test')
    update_moa_indications(ws)

    moa1.refresh_from_db()
    assert moa1.indication == iv.REVIEWED_AS_MOLECULE



# Since defus runs on a mock remote, we need to make sure
# we override this so that it reads the right dpi files and such.
# Not using the same directory as e2e cache, just to avoid any chance
# of conflicting with each other.
from path_helper import PathHelper
TMPDIR = PathHelper.get_config()['bigtmp']
s3_cachedir = os.path.join(TMPDIR, 'test_moa_cache/')
ph_opts = {'S3_CACHE_DIR': s3_cachedir}


# live_server is required for getting the background job stuff working, somehow that makes
# the database view consistent cross-process.
@patch('dtk.lts.LtsRepo')
@pytest.mark.parametrize('tmpdir_path_helper', [ph_opts], indirect=True)
def test_moa_defus(lts_repo, tmpdir_path_helper, setup_fixture, auth_client, live_server, mock_remote_machine):
    """
    This is a fairly barebones defus MoA test, but exercises the infrastructure.
    The main limitation is that there are no actual SMILES and no real FAERS scores, so everything
    just gets a score of 0.
    """
    ws = setup_fixture
    setup_lts_tmpdir()
    shim_background_jobs_in_process()

    from browse.models import WsAnnotation
    from runner.models import Process
    from runner.process_info import JobInfo
    faers_proc = Process.objects.create(
        name='faers',
        role='faers',
        status=Process.status_vals.SUCCEEDED,
    )

    from dtk.prot_map import DpiMapping
    dpi = DpiMapping(ws.get_dpi_default())
    moa_filter = {'agent__collection__name': 'moa.full'}
    assert dpi.get_dpi_type() == 'moa'
    assert len(WsAnnotation.objects.filter(ws=ws, **moa_filter)) == 2

    bji = JobInfo.get_bound(ws, faers_proc)
    import os
    os.makedirs(os.path.dirname(bji.lr_enrichment))
    lr_results = {
        wsa.id: [1e-99, 10.0, 1]
        for wsa in WsAnnotation.objects.exclude(agent__collection__name='moa.full')
        }
    print(f"Starting with {len(lr_results)} lr results")
    with open(bji.lr_enrichment, 'w') as f:
        f.write("\t".join(["wsa",
                            "lrpvalue",
                            "lrenrichment",
                            "lrdir"
                            ]) + "\n")
        for k,l in lr_results.items():
            f.write("\t".join([str(x)
                                for x in [k] + l
                                ]) + "\n")



    from scripts.run_job import run, get_default_settings
    settings = get_default_settings(ws.id, 'defus')
    settings['faers_run'] =  faers_proc.id
    settings['ws_id'] = ws.id

    run('unit-test-username', 'defus',settings=settings, output=None, ws_ids=[ws.id])

    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)

    defus_bji = JobInfo.get_bound(ws, p_id)
    outfile = defus_bji.outfile

    from dtk.files import get_file_records
    recs = list(get_file_records(outfile, keep_header=False))

    assert len(recs) == 2, "Should be 2 MoAs"
    moa_rec_ids = set([int(x[0]) for x in recs])

    from browse.models import WsAnnotation
    moa_wsa_ids = set(WsAnnotation.objects.filter(ws=ws, agent__collection__name='moa.full').values_list('id', flat=True))

    assert moa_rec_ids == moa_wsa_ids

    outsims_fn = defus_bji.outsims
    from dtk.arraystore import get_array
    indjac_arr, meta = get_array(outsims_fn, 'indJac')

    from dtk.features import SparseMatrixWrapper
    mat = SparseMatrixWrapper(indjac_arr, meta['row_names'], meta['col_names'])
    assert set(mat[mat.row_names[0]].values()) == { 1.0,  0.5}
    


from wsadmin.tests.views_test import FakeDpiRemote

@patch('wsadmin.custom_dpi.CustomDpiRemote', FakeDpiRemote)
@patch('dtk.lts.LtsRepo', MagicMock())
@pytest.mark.parametrize('tmpdir_path_helper', [ph_opts], indirect=True)
def test_moa_defus_customdpi(tmpdir_path_helper, setup_fixture, auth_client, live_server, mock_remote_machine):
    ws = setup_fixture
    setup_lts_tmpdir()
    shim_background_jobs_in_process()

    from runner.models import Process
    from runner.process_info import JobInfo
    from wsadmin.custom_dpi import create_custom_dpi, CustomDpiModes, custom_dpi_path
    from wsadmin.models import CustomDpi

    from browse.models import ProtSet, Protein
    ps = ProtSet.objects.create(
            ws=ws,
            name='Test PS',
        )
    Protein.objects.create(uniprot='P001', gene='G001')
    Protein.objects.create(uniprot='P002', gene='G002')
    a_prot = Protein.objects.all()[0]

    ps.proteins.add(a_prot)


    create_custom_dpi(
        base_dpi='testdpi-moa.v1',
        protset=f'ps{ps.id}',
        mode=CustomDpiModes.SUBTRACT,
        name='',
        descr='',
        ws=ws,
        user=''
    )
    custom_dpi = CustomDpi.objects.all()[0]
    print("Custom dpi is ", custom_dpi.uid)
    from browse.default_settings import DpiDataset
    DpiDataset.set(ws, custom_dpi.uid, 'testuser')

    # Test out that the custom path works.
    moa_path = custom_dpi_path(custom_dpi.uid)
    # Test out that the non-moa version of the custom path works.
    non_moa_path = custom_dpi_path(custom_dpi.uid.replace('-moa', ''))
    print(f"Custom DPI Paths: {moa_path}  {non_moa_path}")

    faers_proc = Process.objects.create(
        name='faers',
        role='faers',
        status=Process.status_vals.SUCCEEDED,
    )

    bji = JobInfo.get_bound(ws, faers_proc)
    import os
    os.makedirs(os.path.dirname(bji.lr_enrichment))
    from browse.models import WsAnnotation
    lr_results = {
        wsa.id: [1e-99, 10.0, 1]
        for wsa in WsAnnotation.objects.exclude(agent__collection__name='moa.full')
    }
    with open(bji.lr_enrichment, 'w') as f:
        f.write("\t".join(["wsa",
                            "lrpvalue",
                            "lrenrichment",
                            "lrdir"
                            ]) + "\n")
        for k,l in lr_results.items():
            f.write("\t".join([str(x)
                                for x in [k] + l
                                ]) + "\n")



    from scripts.run_job import run, get_default_settings
    settings = get_default_settings(ws.id, 'defus')
    settings['faers_run'] =  faers_proc.id
    settings['ws_id'] = ws.id
    settings['p2d_file'] = custom_dpi.uid

    run('unit-test-username', 'defus',settings=settings, output=None, ws_ids=[ws.id])

    p_id = wait_for_background_job(expect_success=True, timeout_secs=3*60)

    defus_bji = JobInfo.get_bound(ws, p_id)
    outfile = defus_bji.outfile

    from dtk.files import get_file_records
    recs = list(get_file_records(outfile, keep_header=False))

    # One of our MoAs no longer exists due to subtraction. 
    assert len(recs) == 1, "Should be 1 MoAs"
    moa_rec_ids = set([int(x[0]) for x in recs])

    moa_wsa_ids = set(WsAnnotation.objects.filter(ws=ws, agent__collection__name='moa.full').values_list('id', flat=True))

    #assert moa_rec_ids == moa_wsa_ids