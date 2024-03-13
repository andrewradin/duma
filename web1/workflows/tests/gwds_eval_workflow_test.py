
from mock import patch
import pytest

from dtk.tests.end_to_end_test import setup_lts_tmpdir, wait_for_background_job, shim_background_jobs_in_process, assert_file_unchanged
from dtk.tests.tmpdir_path_helper import tmpdir_path_helper

from algorithms.run_esga import MyJobInfo
from browse.models import Workspace

@patch('dtk.gwas.gwas_codes')
@patch.object(MyJobInfo, 'run', autospec=True)
@patch.object(Workspace, 'get_wsa_id_set', autospec=True)
@patch('dtk.lts.LtsRepo')
def test_gwds_eval_all_same(lts_repo, get_wsa_id_set, esga_run, gwas_codes_mock, live_server, tmpdir_path_helper):

    # Mostly testing the LOO aspects of this workflow.
    # We mock out most everything with fixed values, this doesn't actually
    # run any ESGA.

    gwas_codes_mock.return_value = ['gwds1', 'gwds2', 'gwds3', 'gwds4']

    def esga_run_func(self):
        from path_helper import make_directory
        make_directory(self.lts_abs_root)
        print("Writing out some esga results to ", self.outfile)
        with open(self.outfile, 'w') as f:
            rows = [
                    ['wsa', 'prMax'],
                    ['1', '0.1'],
                    ['2', '0.2'],
                    ['3', '0.5'],
                    ['4', '0']
                    ]
            f.write('\n'.join('\t'.join(x) for x in rows))


    esga_run.side_effect = esga_run_func

    get_wsa_id_set.return_value = [1]

    # Make sure LTS writes to the tmpdir
    setup_lts_tmpdir()
    # Make our background job run in-process (with all our mocks in place)
    shim_background_jobs_in_process()

    ws = Workspace.objects.create(name='testws')

    from workflows.gwds_eval_workflow import GwasDatasetEvalFlow
    wf = GwasDatasetEvalFlow(
            ws_id=ws.id,
            p2d_file='',
            p2d_min=0,
            p2p_file='',
            p2p_min=0,
            eval_ds='tts',
            user='test'
            )
    wf.run_to_completion()
    wf.dump()
    assert wf.succeeded(), "Failed workflow"

    from runner.models import Process
    assert len(Process.objects.all()) == 6, "1 for base, 4 for datasets, 1 final compare"



@patch('dtk.gwas.gwas_codes')
@patch.object(MyJobInfo, 'run', autospec=True)
@patch.object(Workspace, 'get_wsa_id_set', autospec=True)
@patch('dtk.lts.LtsRepo')
def test_gwds_eval_diffs(lts_repo, get_wsa_id_set, esga_run, gwas_codes_mock, live_server, tmpdir_path_helper):
    gwds = ['gwds10', 'gwds2', 'gwds3', 'gwds4']
    gwas_codes_mock.return_value = gwds

    def esga_run_func(self):
        from dtk.gwas import selected_gwas
        g = [self.parms.get(x, None) for x in gwds]
        sum_g = sum([1 for x in g if x is False])
        # For the 1st pass (sum_g==1), we want to remove gwds1
        # For the 2nd pass (sum_g==2), we want to remove gwds10
        if sum_g == 1 and g[1] is False:
            rows = [
                    ['wsa', 'prMax'],
                    ['1', '1.0'],
                    ['2', '0.2'],
                    ['3', '1.3'],
                    ['4', '0']
                    ]
        elif sum_g == 2 and g[0] is False:
            rows = [
                    ['wsa', 'prMax'],
                    ['1', '2.0'],
                    ['2', '0.2'],
                    ['3', '1.3'],
                    ['4', '0']
                    ]
        else:
            rows = [
                    ['wsa', 'prMax'],
                    ['1', '0.1'],
                    ['2', '0.2'],
                    ['3', '0.5'],
                    ['4', '0']
                    ]

        from path_helper import make_directory
        make_directory(self.lts_abs_root)
        print("Writing out some esga results to ", self.outfile)
        with open(self.outfile, 'w') as f:
            f.write('\n'.join('\t'.join(x) for x in rows))


    esga_run.side_effect = esga_run_func

    get_wsa_id_set.return_value = [1]

    # Make sure LTS writes to the tmpdir
    setup_lts_tmpdir()
    # Make our background job run in-process (with all our mocks in place)
    shim_background_jobs_in_process()

    ws = Workspace.objects.create(name='testws')

    from workflows.gwds_eval_workflow import GwasDatasetEvalFlow
    wf = GwasDatasetEvalFlow(
            ws_id=ws.id,
            p2d_file='',
            p2d_min=0,
            p2p_file='',
            p2p_min=0,
            eval_ds='tts',
            user='test'
            )
    wf.run_to_completion()
    wf.dump()
    assert wf.succeeded(), "Failed workflow"

    from runner.models import Process
    assert len(Process.objects.all()) == 11, \
            "base + 4x ds + 3x ds + 2x ds + final compare"


@patch('dtk.gwas.gwas_codes')
@patch.object(MyJobInfo, 'run', autospec=True)
@patch.object(Workspace, 'get_wsa_id_set', autospec=True)
@patch('dtk.lts.LtsRepo')
def test_gwds_eval_ttsplit(lts_repo, get_wsa_id_set, esga_run, gwas_codes_mock, live_server, tmpdir_path_helper):
    gwas_codes_mock.return_value = ['gwds1', 'gwds2', 'gwds3', 'gwds4']

    def esga_run_func(self):
        from dtk.gwas import selected_gwas

        if self.parms.get('gwds1', None) is False:
            rows = [
                    ['wsa', 'prMax'],
                    ['1', '1.0'],
                    ['2', '0.2'],
                    ['3', '0.5'],
                    ['4', '0']
                    ]
        else:
            rows = [
                    ['wsa', 'prMax'],
                    ['1', '0.1'],
                    ['2', '0.2'],
                    ['3', '0.5'],
                    ['4', '0']
                    ]

        from path_helper import make_directory
        make_directory(self.lts_abs_root)
        print("Writing out some esga results to ", self.outfile)
        with open(self.outfile, 'w') as f:
            f.write('\n'.join('\t'.join(x) for x in rows))


    esga_run.side_effect = esga_run_func

    def get_wsa_id_set_func(self, name):
        if name == 'split-train-tts':
            return [1]
        elif name == 'split-test-tts':
            return [2]
        else:
            assert False, "Didn't expect %s" % name

    get_wsa_id_set.side_effect = get_wsa_id_set_func

    # Make sure LTS writes to the tmpdir
    setup_lts_tmpdir()
    # Make our background job run in-process (with all our mocks in place)
    shim_background_jobs_in_process()

    ws = Workspace.objects.create(name='testws')

    from workflows.gwds_eval_workflow import GwasDatasetEvalFlow
    wf = GwasDatasetEvalFlow(
            ws_id=ws.id,
            p2d_file='',
            p2d_min=0,
            p2p_file='',
            p2p_min=0,
            eval_ds='split-train-tts',
            user='test'
            )
    wf.run_to_completion()
    wf.dump()
    assert wf.succeeded(), "Failed workflow"

    from runner.models import Process
    assert len(Process.objects.all()) == 9, \
            "base + 4x ds + 3x ds + final compare"




