from __future__ import print_function
from dtk.workflow import LeaveOneOutOptimizingWorkflow,LocalCode,PathStep,GESigStep,CodesStep

class CombinedGEEvalFlow(LeaveOneOutOptimizingWorkflow):
    _fields = [
            'ts_id',
            'p2d_file',
            'p2p_file',
            'p2d_min',
            'gesig_code',
            'p2p_min',
            'eval_ds',
            ]
    def _add_cm_step(self,excluded,stepname=None):
        if not stepname:
            stepname = 'path_direct_minus_'+'_'.join([
                    str(x) for x in excluded
                    ])
        thresh_overrides = dict(self.thresh_overrides)
        for k in excluded:
            thresh_overrides[k] = 2
        thresh_overrides['detail_file'] = False
        PathStep(self,stepname,
                thresh_overrides=thresh_overrides,
                )
        return stepname

    def _add_cm_step1(self,excluded,stepname=None):
        if not stepname:
            stepname = 'path_indirect_minus_'+'_'.join([
                    str(x) for x in excluded
                    ])
        thresh_overrides = dict(self.thresh_overrides)
        for k in excluded:
            thresh_overrides[k] = 2
        thresh_overrides['detail_file'] = False
        PathStep(self,stepname,
                thresh_overrides=thresh_overrides,
                )
        return stepname

    def _add_cm_step2(self,excluded,stepname=None):
        if not stepname:
            stepname = 'Gesig_minus_'+'_'.join([
                    str(x) for x in excluded
                    ])
        thresh_overrides = dict(self.thresh_overrides)
        for k in excluded:
            thresh_overrides[k] = False
        GESigStep(self,stepname,
                thresh_overrides=thresh_overrides,
                )
        input = {stepname:self.gesig_code}
        stepname = 'codes' + stepname
        CodesStep(self,stepname,
                inputs=input,
                )
        return stepname

    def _add_cycle(self,excluded,baseline):
        for bs in baseline:
            print('bs', bs)
            self.save_steps.append(self._steps[bs])
        self.cycle_count += 1
        inputs = None
        for bs in baseline:
            if inputs is None:
                inputs = {bs:None}
            else:
                inputs[bs] = None
        for tissue in self.ts.tissue_set.all():
            _,_,_,total = tissue.sig_result_counts()
            if not total:
                continue
            print('tissue_id', tissue.id)
            print('excluded', excluded)
            if tissue.id in excluded:
                continue
            stepname = self._add_cm_step(excluded+[tissue.id])
            inputs[stepname] = tissue.id
            stepname = self._add_cm_step1(excluded+[tissue.id])
            inputs[stepname] = tissue.id
            stepname = self._add_cm_step2(excluded+[tissue.id])
            inputs[stepname] = tissue.id
        print('inputs', inputs)
        print('baseline', baseline)
        LocalCode(self,'compare%d'%self.cycle_count, True,
                func=self._do_compare,
                inputs=inputs,
                baseline=baseline,
                excluded=excluded,
                cm_code=['direct','indirect','codesMax'],
                rethrow=True,
                )
    def __init__(self,**kwargs):
        super(CombinedGEEvalFlow,self).__init__(**kwargs)
        from browse.models import Workspace, TissueSet
        self.ts = TissueSet.objects.get(pk=self.ts_id)
        self.ws = Workspace.objects.get(pk=self.ws_id)
        self._pl.add_defaults(
                thresh_overrides={},
                eval_ds=self.ws.eval_drugset,
                )
        baselines = [self._add_cm_step(self.excluded)]
        baselines.append(self._add_cm_step1(self.excluded))
        baselines.append(self._add_cm_step2(self.excluded))
        self._add_cycle(self.excluded,baselines)

