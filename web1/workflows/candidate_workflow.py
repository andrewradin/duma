from __future__ import print_function
import dtk.workflow as dwf

from dtk.dynaform import FieldType

import logging
logger = logging.getLogger(__name__)

def get_wzs_for_wsa(wsa):
    jid = wsa.get_marked_or_best_eff_jid()
    if not jid:
        raise Exception("%s has no marking prescreen" % wsa)
    return jid

def get_indication_wsas(ws, indication):
    from browse.models import WsAnnotation
    qs_ws = WsAnnotation.objects.filter(ws=ws, indication=indication)
    return set(qs_ws)

def get_election_wsas(ws, elections):
    from browse.models import WsAnnotation
    return set(WsAnnotation.objects.filter(vote__election__in=elections))

def find_replacements(wsas):
    new_wsas = set()
    for wsa in wsas:
        new_wsas.update(wsa.replacement_for.all())

    if new_wsas:
        repl_repl = find_replacements(new_wsas)
        new_wsas.update(repl_repl)
    return new_wsas


class IndirectTargetImportanceStep(dwf.BackgroundStepBase):
    def __init__(self, *args, **kwargs):
        self.data = kwargs.pop('data')
        super(IndirectTargetImportanceStep,self).__init__(*args, **kwargs)

    def _get_plugin_info(self):
        return 'trgscrimp',None
    def _build_settings(self):
        # build default settings
        wzs_jid = self.data['wzs_jid']
        wsas = self.data['wsas']
        logger.info("Building indtrgimp step with %d wsas, jid %s", len(wsas), wzs_jid)
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # apply overrides
        settings.update(
                wzs_jid=wzs_jid,
                ws_id=self.ws.id,
                count=0,
                indirect_scores=True,
                extra_wsas='\n'.join([str(x.id) for x in wsas]),
                max_input_prots=self.max_input_prots,
                )
        return settings

class DrugNotePrecomputeStep(dwf.BackgroundStepBase):
    def __init__(self, *args, **kwargs):
        self.data = kwargs.pop('data')
        super(DrugNotePrecomputeStep,self).__init__(*args, **kwargs)

    def _get_plugin_info(self):
        return 'dnprecompute',None
    def _build_settings(self):
        # build default settings
        wsas = self.data['wsas']
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # apply overrides
        settings.update(
                wsas_text='\n'.join([str(x.id) for x in wsas]),
                )
        return settings

def add_indtrgscrimp_to_workflow(wf, name, wsas):
    wsas.sort(key=get_wzs_for_wsa)
    import itertools
    for wzs_jid, wsa_group in itertools.groupby(wsas, get_wzs_for_wsa):
        data = {'wzs_jid': wzs_jid, 'wsas': list(wsa_group)}
        print(("Adding indtrgscrimp ", data))
        IndirectTargetImportanceStep(wf, name + "_" + str(wzs_jid), data=data)

def add_dnprecompute_to_workflow(wf, name, wsas):
    data = {'wsas': wsas}
    DrugNotePrecomputeStep(wf, name, data=data)

class CandidateFlow(dwf.Workflow):
    # These select the form elements for configuring the workflow.
    # They correspond to FieldType classes in dtk/dynaform.py.
    _fields = [
            'candidate_parts',
            'indication',
            'max_input_prots',
            'review_round',
            ]
    #####
    # update scoreset
    #####
    def checkpoint(self):
        super(CandidateFlow,self).checkpoint()
        # At each checkpoint, add any new successful jobs to the scoreset
        self.add_done_steps_to_scoreset(self.scoreset)
    #####
    # start workflow
    #####
    def __init__(self,**kwargs):
        super(CandidateFlow,self).__init__(**kwargs)
        from browse.models import Workspace,TissueSet
        self.ws = Workspace.objects.get(pk=self.ws_id)
        self.scoreset = self.make_scoreset(self._label())
        from dtk.dynaform import CandidatePartsFieldType
        self.parts = CandidatePartsFieldType.parts
        dwf.LocalCode(self,'Executing...',
                func=self._run,
                inputs={k:True for k in self._order},
                )
    def _run(self,my_step):
        wsas = get_indication_wsas(self.ws, self.indication)
        wsas &= get_election_wsas(self.ws, self.review_round)
        wsas |= find_replacements(wsas)
        wsas = list(wsas)

        logger.info(f"Operating on {', '.join(wsa.get_name(False) for wsa in wsas)}")

        for ind in self.candidate_parts:
            part = self.parts[int(ind)]
            part.add_to_workflow(self,wsas)
