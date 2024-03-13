from builtins import range
from dtk.workflow import Workflow,WorkStep,LocalCode,PathStep

class PpiThreshEvalFlow(Workflow):
    _fields = [
            'ts_id',
            'p2d_file',
            'p2p_file',
            'p2d_min',
            'p2p_min',
            ]
    def _add_pathstep(self,ppi_thresh):
        stepname='path_ppi_thresh_%f' % ppi_thresh
        PathStep(self,stepname,
                p2p_t=ppi_thresh,
                )
        return stepname
    def _save_result(self,save_step):
        self.save_scoreset(self._label())
    def __init__(self,**kwargs):
        super(PpiThreshEvalFlow,self).__init__(**kwargs)
        self._pl.add_defaults(
                exclude_tissues=[],
                thresh_overrides={},
                )
        from browse.models import TissueSet
        self.ts = TissueSet.objects.get(pk=self.ts_id)
        self.ws = self.ts.ws
        inputs={}
        thresholds = [1-2**-x for x in range(1,6)]
        for ppi_thresh in thresholds:
            stepname = self._add_pathstep(ppi_thresh)
            inputs[stepname]=None
        LocalCode(self,'save',
                func=self._save_result,
                inputs=inputs,
                rethrow=True,
                )
