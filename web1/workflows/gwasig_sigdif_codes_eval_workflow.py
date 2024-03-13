from __future__ import print_function
from dtk.workflow import LeaveOneOutOptimizingWorkflow,LocalCode,GWASigStep,CodesStep,SigDifStep

class gwasigSigdifCodesEvalFlow(LeaveOneOutOptimizingWorkflow):
    _fields = [
            'p2p_file',
            'p2p_min',
            'p2d_file',
            'p2d_min',
            'gwasig_code',
            'codes_code',
            'eval_ds'
            ]
    def _add_cm_step(self,excluded,stepname=None):
        if not stepname:
            stepname = 'Gwasig_minus_'+'_'.join([
                    str(x) for x in excluded
                    ])
        thresh_overrides = dict(self.thresh_overrides)
        for k in excluded:
            thresh_overrides[k] = False
        GWASigStep(self,stepname,
                thresh_overrides=thresh_overrides,
                )
        input = {stepname:self.gwasig_code}
        stepname = 'sigdif' + stepname
        print('Entering SigDif')
        SigDifStep(self,stepname,ws=self.ws,
                inputs=input,
                )
        sigdif_code = 'difEv'
        input = {stepname:sigdif_code}
        stepname = 'codes' + stepname
        CodesStep(self,stepname,
                inputs=input,
                )
        return stepname
    def _add_cycle(self,excluded,baseline):
        self.save_steps.append(self._steps[baseline])
        self.cycle_count += 1
        inputs = {baseline:None}
        for gwds in self.ds_names:
            if gwds in excluded:
                continue
            stepname = self._add_cm_step(excluded+[gwds])
            inputs[stepname] = gwds
        LocalCode(self,'compare%d'%self.cycle_count,
                func=self._do_compare,
                inputs=inputs,
                baseline=baseline,
                excluded=excluded,
                cm_code=self.codes_code,
                rethrow=True,
                )
    def __init__(self,**kwargs):
        super(gwasigSigdifCodesEvalFlow,self).__init__(**kwargs)
        from browse.models import Workspace
        self.ws = Workspace.objects.get(pk=self.ws_id)
        self._pl.add_defaults(
                thresh_overrides={},
                eval_ds=self.ws.eval_drugset,
                )
        from dtk.gwas import gwas_codes
        self.ds_names = gwas_codes(self.ws)
        baseline = self._add_cm_step(self.excluded)
        self._add_cycle(self.excluded,baseline)

