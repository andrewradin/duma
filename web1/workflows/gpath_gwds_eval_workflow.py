from dtk.workflow import LeaveOneOutOptimizingWorkflow,LocalCode,gpathStep

class gPathGWDSEvalFlow(LeaveOneOutOptimizingWorkflow):
    _fields = [
            'p2d_file',
            'p2d_min',
            'p2p_file',
            'p2p_min',
            'gpath_code',
            'eval_ds',
            ]
    def _add_cm_step(self,excluded,stepname=None):
        if not stepname:
            stepname = 'gpath_minus_'+'_'.join([
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
                cm_code=self.gpath_code,
                rethrow=True,
                )
    def __init__(self,**kwargs):
        super(gPathGWDSEvalFlow,self).__init__(**kwargs)
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

