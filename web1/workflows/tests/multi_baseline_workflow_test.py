from dtk.workflow import LeaveOneOutOptimizingWorkflow,LocalCode,EsgaStep
from mock import patch
import pytest

from dtk.tests.end_to_end_test import setup_lts_tmpdir, wait_for_background_job, shim_background_jobs_in_process, assert_file_unchanged
from dtk.tests.tmpdir_path_helper import tmpdir_path_helper

from algorithms.run_esga import MyJobInfo
from browse.models import Workspace

class MultiBaselineEvalFlow(LeaveOneOutOptimizingWorkflow):
    _fields = [
            'eval_ds',
            ]
    def _add_cm_step(self,excluded,stepname=None):
        return self._add_cm_step1(excluded, stepname)
    def _add_cm_step1(self,excluded,stepname=None):
        if not stepname:
            stepname = 'esga_minus_'+'_'.join([
                    str(x) for x in excluded
                    ])
        thresh_overrides = dict(self.thresh_overrides)
        for k in excluded:
            thresh_overrides[k] = False        
        EsgaStep(self,stepname,
                thresh_overrides=thresh_overrides,
                )
        return stepname
    def _add_cm_step2(self,excluded,stepname=None):
        if not stepname:
            stepname = 'fsga_minus_'+'_'.join([
                    str(x) for x in excluded
                    ])
        thresh_overrides = dict(self.thresh_overrides)
        for k in excluded:
            thresh_overrides[k] = False        
        EsgaStep(self,stepname,
                thresh_overrides=thresh_overrides,
                )
        return stepname
    def _add_cycle(self,excluded,baseline):
        inputs = {}
        for bs in baseline:
            self.save_steps.append(self._steps[bs])
            inputs[bs] = None
        self.cycle_count += 1
        for ds_name in self.ds_names:
            if ds_name in excluded:
                continue
            stepname = self._add_cm_step1(excluded+[ds_name])
            inputs[stepname] = ds_name
            stepname = self._add_cm_step2(excluded+[ds_name])
            inputs[stepname] = ds_name
        print('Selfdict',self.esga_code)
        LocalCode(self,'compare%d'%self.cycle_count,
                func=self._do_compare,
                inputs=inputs,
                baseline=baseline,
                excluded=excluded,
                cm_code=[self.esga_code,self.esga_code],
                rethrow=True,
                )
    def __init__(self,**kwargs):
        super(MultiBaselineEvalFlow,self).__init__(**kwargs)
        from browse.models import Workspace
        self.ws = Workspace.objects.get(pk=self.ws_id)
        self._pl.add_defaults(
                esga_code='prMax',
                thresh_overrides={},
                eval_ds=self.ws.eval_drugset,
                )
        from dtk.gwas import gwas_codes
        self.ds_names = gwas_codes(self.ws)
        baseline = [
                self._add_cm_step1(self.excluded),
                self._add_cm_step2(self.excluded),
                ]
        self._add_cycle(self.excluded,baseline)



@patch('dtk.gwas.gwas_codes')
@patch.object(MyJobInfo, 'run', autospec=True)
@patch.object(Workspace, 'get_wsa_id_set', autospec=True)
@patch('dtk.lts.LtsRepo')
def test_multibaseline_workflow_ttsplit(lts_repo, get_wsa_id_set, esga_run, gwas_codes_mock, live_server, tmpdir_path_helper):
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
        # Add enough extra WSAs that we get a measurable change in SoR1000
        # when we change the score.
        rows += [[str(x), '0.6'] for x in range(5, 500)]

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

    wf = MultiBaselineEvalFlow(
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
    assert len(Process.objects.all()) == 17, \
            "2xbase + 8x ds + 6x ds + final compare"


@patch('dtk.gwas.gwas_codes')
@patch.object(MyJobInfo, 'run', autospec=True)
@patch.object(Workspace, 'get_wsa_id_set', autospec=True)
@patch('dtk.lts.LtsRepo')
def test_multibaseline_workflow_order(lts_repo, get_wsa_id_set, esga_run, gwas_codes_mock, live_server, tmpdir_path_helper):
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
        # Add enough extra WSAs that we get a measurable change in SoR1000
        # when we change the score.
        rows += [[str(x), '0.6'] for x in range(5, 250)]
        rows += [[str(x), '1.4'] for x in range(250, 500)]

        from path_helper import make_directory
        make_directory(self.lts_abs_root)
        print("Writing out some esga results to ", self.outfile)
        with open(self.outfile, 'w') as f:
            f.write('\n'.join('\t'.join(x) for x in rows))


    esga_run.side_effect = esga_run_func

    def get_wsa_id_set_func(self, name):
        return [1]

    get_wsa_id_set.side_effect = get_wsa_id_set_func

    # Make sure LTS writes to the tmpdir
    setup_lts_tmpdir()
    # Make our background job run in-process (with all our mocks in place)
    shim_background_jobs_in_process()

    ws = Workspace.objects.create(name='testws')

    wf = MultiBaselineEvalFlow(
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
    assert len(Process.objects.all()) == 21, \
            "2xbase + 8x ds + 6x ds + 4x ds + final compare"