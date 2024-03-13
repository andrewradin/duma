from __future__ import print_function
from dtk.workflow import LeaveOneOutOptimizingWorkflow,LocalCode,GWASigStep,CodesStep,SigDifStep,GESigStep,gpathStep,EsgaStep

class CombinedGWASEvalFlow(LeaveOneOutOptimizingWorkflow):
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

    def _add_cm_step2(self,excluded,stepname=None):
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

    def _add_cm_step3(self,excluded,stepname=None):
        if not stepname:
            stepname = 'gpath_direct_minus_'+'_'.join([
                    str(x) for x in excluded
                    ])
        thresh_overrides = dict(self.thresh_overrides)
        for k in excluded:
            thresh_overrides[k] = False
        thresh_overrides['detail_file'] = False
        gpathStep(self,stepname,
                thresh_overrides=thresh_overrides,
                )
        return stepname

    def _add_cm_step4(self,excluded,stepname=None):
        if not stepname:
            stepname = 'gpath_indirect_minus_'+'_'.join([
                    str(x) for x in excluded
                    ])
        thresh_overrides = dict(self.thresh_overrides)
        for k in excluded:
            thresh_overrides[k] = False
        thresh_overrides['detail_file'] = False
        gpathStep(self,stepname,
                thresh_overrides=thresh_overrides,
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
        for gwds in self.ds_names:
            if gwds in excluded:
                continue
            stepname = self._add_cm_step(excluded+[gwds])
            inputs[stepname] = gwds
            stepname = self._add_cm_step2(excluded+[gwds])
            inputs[stepname] = gwds
            stepname = self._add_cm_step3(excluded+[gwds])
            inputs[stepname] = gwds
            stepname = self._add_cm_step4(excluded+[gwds])
            inputs[stepname] = gwds
        LocalCode(self,'compare%d'%self.cycle_count, True,
                func=self._do_compare,
                inputs=inputs,
                baseline=baseline,
                excluded=excluded,
                cm_code=[self.codes_code, self.esga_code,'gds', 'gis'],
                rethrow=True,
                )
    def __init__(self,**kwargs):
        super(CombinedGWASEvalFlow,self).__init__(**kwargs)
        from browse.models import Workspace
        self.ws = Workspace.objects.get(pk=self.ws_id)
        self._pl.add_defaults(
                esga_code='prMax',
                thresh_overrides={},
                eval_ds=self.ws.eval_drugset,
                )
        from dtk.gwas import gwas_codes
        self.ds_names = gwas_codes(self.ws)
        baselines = [self._add_cm_step(self.excluded)]
        baselines.append(self._add_cm_step2(self.excluded))
        baselines.append(self._add_cm_step3(self.excluded))
        baselines.append(self._add_cm_step4(self.excluded))
        self._add_cycle(self.excluded,baselines)

