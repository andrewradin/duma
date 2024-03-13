from dtk.workflow import Workflow,WorkStep,LocalCode,PathStep

class ProteinInteractionCrossFlow(Workflow):
    _fields = [
            'ts_id',
            ]
    def _add_pathstep(self,p2d_file,p2p_file,stepname=None):
        if not stepname:
            stepname = 'path_%s_%s' % (p2d_file,p2p_file)
        PathStep(self,stepname,
                p2d_file=p2d_file,
                p2p_file=p2p_file,
                )
        return stepname
    def _do_checkpoint(self,checkpoint_step):
        self.checkpoint()
        self.save_scoreset(self._label())
    def __init__(self,**kwargs):
        super(ProteinInteractionCrossFlow,self).__init__(**kwargs)
        from dtk.prot_map import DpiMapping,PpiMapping
        self._pl.add_defaults(
                thresh_overrides={},
                p2d_list = DpiMapping.dpi_names(),
                p2p_list = PpiMapping.ppi_names(),
                )
        from browse.models import TissueSet
        self.ts = TissueSet.objects.get(pk=self.ts_id)
        self.ws = self.ts.ws
        inputs = {}
        for p2d_file in self.p2d_list:
            for p2p_file in self.p2p_list:
                stepname = self._add_pathstep(p2d_file,p2p_file)
                inputs[stepname] = True
        LocalCode(self,'checkpoint',
                func=self._do_checkpoint,
                inputs=inputs,
                )

