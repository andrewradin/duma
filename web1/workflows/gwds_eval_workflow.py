from __future__ import print_function
from dtk.workflow import LeaveOneOutOptimizingWorkflow,LocalCode,EsgaStep

class GwasDatasetEvalFlow(LeaveOneOutOptimizingWorkflow):
    _fields = [
            'p2d_file',
            'p2d_min',
            'p2p_file',
            'p2p_min',
            'eval_ds',
            ]
    def _add_cm_step(self,excluded,stepname=None):
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
    def _add_cycle(self,excluded,baseline):
        self.save_steps.append(self._steps[baseline])
        self.cycle_count += 1
        inputs = {baseline:None}
        for ds_name in self.ds_names:
            if ds_name in excluded:
                continue
            stepname = self._add_cm_step(excluded+[ds_name])
            inputs[stepname] = ds_name
        LocalCode(self,'compare%d'%self.cycle_count,
                func=self._do_compare,
                inputs=inputs,
                baseline=baseline,
                excluded=excluded,
                cm_code=self.esga_code,
                rethrow=True,
                )
    def __init__(self,**kwargs):
        super(GwasDatasetEvalFlow,self).__init__(**kwargs)
        from browse.models import Workspace
        self.ws = Workspace.objects.get(pk=self.ws_id)
        self._pl.add_defaults(
                esga_code='prMax',
                thresh_overrides={},
                eval_ds=self.ws.eval_drugset,
                )
        from dtk.gwas import gwas_codes
        self.ds_names = gwas_codes(self.ws)
        baseline = self._add_cm_step(self.excluded)
        self._add_cycle(self.excluded,baseline)

