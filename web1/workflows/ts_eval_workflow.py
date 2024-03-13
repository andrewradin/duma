from dtk.workflow import LeaveOneOutOptimizingWorkflow,LocalCode,PathStep


class TissueSetEvalFlow(LeaveOneOutOptimizingWorkflow):
    _fields = [
            'ts_id',
            'p2d_file',
            'p2p_file',
            'p2d_min',
            'p2p_min',
            'path_code',
            'eval_ds',
            # XXX leave the following off for now; they're problematic because
            # XXX the valid choices are dependent on the selected tissue set;
            # XXX but we may not need to solve this because:
            # XXX - you can pre-exclude a tissue by temporarily moving it
            # XXX   outside the tissue set
            # XXX - we may eventually store ideal thresholds in the tissue
            # XXX   records, so there would be no need to set them again here
            # 'excluded', # list of tissue ids
            # 'thresh_overrides', # list of tissue/thresh pairs
            ]
    def _add_cm_step(self,excluded,stepname=None):
        excluded.sort()
        if not stepname:
            stepname = 'path_minus_'+'_'.join([
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
                cm_code=self.path_code,
                rethrow=True,
                )
    def __init__(self,**kwargs):
        super(TissueSetEvalFlow,self).__init__(**kwargs)
        from browse.models import TissueSet
        self.ts = TissueSet.objects.get(pk=self.ts_id)
        self.ws = self.ts.ws
        self._pl.add_defaults(
                thresh_overrides={},
                eval_ds=self.ws.eval_drugset,
                )
        baseline = self._add_cm_step(self.excluded)
        self._add_cycle(self.excluded,baseline)

