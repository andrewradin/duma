from dtk.workflow import LeaveOneOutOptimizingWorkflow,LocalCode,GESigStep,CodesStep

class gesigCodesEvalFlow(LeaveOneOutOptimizingWorkflow):
    _fields = [
            'ts_id',
            'p2d_file',
            'p2d_min',
            'gesig_code',
            'codes_code',
            'eval_ds',
            ]
    def _add_cm_step(self,excluded,stepname=None):
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
        self.save_steps.append(self._steps[baseline])
        self.cycle_count += 1
        inputs = {baseline:None}
        for tissue in self.ts.tissue_set.all():
            if tissue.id in excluded:
                continue
            stepname = self._add_cm_step(excluded+[tissue.id])
            inputs[stepname] = tissue.id
        LocalCode(self,'compare%d'%self.cycle_count,
                func=self._do_compare,
                inputs=inputs,
                baseline=baseline,
                excluded=excluded,
                cm_code=self.codes_code,
                rethrow=True,
                )
    def __init__(self,**kwargs):
        super(gesigCodesEvalFlow,self).__init__(**kwargs)
        from browse.models import Workspace, TissueSet
        self.ts = TissueSet.objects.get(pk=self.ts_id)
        self.ws = Workspace.objects.get(pk=self.ws_id)
        self._pl.add_defaults(
                thresh_overrides={},
                eval_ds=self.ws.eval_drugset,
                )
        baseline = self._add_cm_step(self.excluded)
        self._add_cycle(self.excluded,baseline)
