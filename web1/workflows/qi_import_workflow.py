from dtk.workflow import Workflow,WorkStep,LocalCode,WsCopyStep

class QuantifyImprovementImportFlow(Workflow):
    _fields = [
            ]
    def _do_checkpoint(self,checkpoint_step):
        self.checkpoint()
        self.save_scoreset(self._label())
    def __init__(self,**kwargs):
        super(QuantifyImprovementImportFlow,self).__init__(**kwargs)
        from browse.models import Workspace
        self.ws = Workspace.objects.get(pk=self.ws_id)
        inputs = {}
        score_suffix='_%d.tsv'%self.ws_id
        from path_helper import PathHelper
        data=PathHelper.repos_root+'experiments/quantifying_improvement/data'
        from dtk.files import scan_dir
        for time_obj in scan_dir(data,output=lambda x:x):
            for score_obj in scan_dir(
                    time_obj.full_path,
                    filters=[lambda x:x.filename.endswith(score_suffix)],
                    output=lambda x:x,
                    ):
                timepoint = time_obj.filename
                score_type = score_obj.filename[:-len(score_suffix)]
                stepname = '_'.join([timepoint,score_type])
                WsCopyStep(self,stepname,
                        from_ws='qi_data',
                        from_score='/'.join([
                                time_obj.filename,
                                score_obj.filename,
                                ])
                        )
                inputs[stepname] = True
        LocalCode(self,'checkpoint',
                func=self._do_checkpoint,
                inputs=inputs,
                )

