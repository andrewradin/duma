from __future__ import print_function
from builtins import range
import dtk.workflow as dwf
from workflows.refresh_workflow import StandardWorkflow


class AnimalModelFlow(dwf.Workflow):
    _fields = [
            'animalmodel_parts',
            'resume_scoreset_id',
            'p2d_file',
            'p2d_min',
            'p2p_file',
            'p2p_min',
            ]
    #####
    # update scoreset
    #####
    def checkpoint(self):
        super().checkpoint()
        # At each checkpoint, add any new successful jobs to the scoreset
        self.add_done_steps_to_scoreset(self.scoreset)

    #####
    # start workflow
    #####
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        from browse.models import Workspace
        self.ws = Workspace.objects.get(pk=self.ws_id)
        if self.resume_scoreset_id:
            self.scoreset = self.resume_old_scoreset(self.resume_scoreset_id)
        else:
            self.scoreset = self.make_scoreset(self._label())

        self.std_wf = StandardWorkflow(ws=self.ws)
        self.eff = self.std_wf.eff_agg

        from dtk.dynaform import AnimalModelPartsFieldType
        part_types = AnimalModelPartsFieldType.get_parts_cls(self.ws)
        for part_idx in self.animalmodel_parts:
            part = part_types[int(part_idx)]
            part.add_to_workflow(self)
