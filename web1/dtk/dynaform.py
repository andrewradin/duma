################################################################################
# Support for dynamic configuration forms
################################################################################
from dtk.subclass_registry import SubclassRegistry
from django import forms

# Each of the FieldType derived classes below defines a particular
# configuration input, and knows how to collect that input via either a
# django form or an argparse command line.
#
# The 'code' member of each class defines the name of the corresponding
# field.  The classes define the following methods:
# - add_to_form() appends the needed field(s) to a FormFactory (defined below)
# - add_to_parser() likewise for argparse
# - XXX some TBD post-processing hook to do any reformatting of data into the
#   XXX final hash

class FormFactory:
    source_label_field_size=40
    def __init__(self):
        from collections import OrderedDict
        self._fields = OrderedDict()
        self._parms = {'base_fields':self._fields}
    def get_form_class(self):
        return type('DynaForm', (forms.BaseForm,), self._parms)
    def add_field(self,code,field):
        # XXX could have optional params for clean function, etc.
        self._fields[code] = field

class FieldType(SubclassRegistry):
    @classmethod
    def get_by_code(cls,code):
        if isinstance(code,str):
            context = {}
        else:
            # should be a tuple, then
            code,context = code
        for label,SubClass in cls.get_subclasses():
            if SubClass.code == code:
                ft=SubClass(context)
                return ft
        raise KeyError("'%s' does not exist" % code)
    def __init__(self,context={}):
        self._context = context
    def add_fallback_context(self,**kwargs):
        kwargs.update(self._context)
        self._context=kwargs
    def add_override_context(self,**kwargs):
        self._context.update(kwargs)
    def get_workspace(self):
        # service routine for a common context retrieval
        from browse.models import Workspace
        try:
            ws_id = self._context['ws_id']
        except KeyError:
            # not a real workspace, but it can still be used
            # for retrieving dpi and ppi defaults, etc.
            return Workspace()
        return Workspace.objects.get(pk=ws_id)

class TissueSetIdFieldType(FieldType):
    code='ts_id'
    def add_to_form(self,ff):
        ws_id = self._context['ws_id']
        from browse.models import TissueSet
        qs = TissueSet.objects.filter(ws_id=ws_id).order_by('id')
        field = forms.ChoiceField(
                    choices=qs.values_list('id','name'),
                    required=True,
                    label='Tissue Set',
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code,type=int)

class P2DFileFieldType(FieldType):
    code='p2d_file'
    def add_to_form(self,ff):
        ws = self.get_workspace()
        from dtk.prot_map import DpiMapping
        field = forms.ChoiceField(
                    choices=DpiMapping.choices(ws),
                    required=True,
                    label='DPI dataset',
                    initial=ws.get_dpi_default(),
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class P2DMinFieldType(FieldType):
    code='p2d_min'
    def add_to_form(self,ff):
        ws = self.get_workspace()
        field = forms.FloatField(
                    label='Min DPI evidence',
                    initial=ws.get_dpi_thresh_default(),
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code,type=float)

class P2PFileFieldType(FieldType):
    code='p2p_file'
    def add_to_form(self,ff):
        ws = self.get_workspace()
        from dtk.prot_map import PpiMapping
        field = forms.ChoiceField(
                    choices=PpiMapping.choices(),
                    required=True,
                    label='PPI dataset',
                    initial=ws.get_ppi_default(),
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class P2PMinFieldType(FieldType):
    code='p2p_min'
    def add_to_form(self,ff):
        ws = self.get_workspace()
        field = forms.FloatField(
                    label='Min PPI evidence',
                    initial=ws.get_ppi_thresh_default(),
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code,type=float)

class PathCodeFieldType(FieldType):
    code='path_code'
    def add_to_form(self,ff):
        field = forms.ChoiceField(
                    choices=[(x,x) for x in ('direct','indirect')],
                    required=True,
                    label='Score to evaluate',
                    initial=self._context.get('initial','direct')
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class gpathCodeFieldType(FieldType):
    code='gpath_code'
    def add_to_form(self,ff):
        field = forms.ChoiceField(
                    choices=[('gds', 'Direct'),('gis','Indirect')],
                    required=True,
                    label='Score to evaluate',
                    initial=self._context.get('initial','gds')
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class codesCodeFieldType(FieldType):
    code='codes_code'
    def add_to_form(self,ff):
        field = forms.ChoiceField(
                    choices=[('codesMax', 'CoDES max'),
                             ('posDir','CoDES direction'),
                             ('negDir','CoDES neg. direction')
                            ],
                    required=True,
                    label='Score to evaluate',
                    initial=self._context.get('initial','codesMax')
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class gesigCodeFieldType(FieldType):
    code='gesig_code'
    def add_to_form(self,ff):
        field = forms.ChoiceField(
                    choices=[('ev', 'Evidence'),
                             ('fold','Fold Change'),
                             ('tisscnt','Tissue Count'),
                             ('avgDir','Mean direction')
                            ],
                    required=True,
                    label='Score to evaluate',
                    initial=self._context.get('initial','ev')
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class gwasigCodeFieldType(FieldType):
    code='gwasig_code'
    def add_to_form(self,ff):
        field = forms.ChoiceField(
                    choices=[('ev', 'Evidence'),
                             ('gwascnt','GWAS Count'),
                            ],
                    required=True,
                    label='Score to evaluate',
                    initial=self._context.get('initial','ev')
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class EvalDrugSetFieldType(FieldType):
    code='eval_ds'
    def add_to_form(self,ff):
        ws_id = self._context['ws_id']
        from browse.models import Workspace
        ws = Workspace.objects.get(pk=ws_id)
        field = forms.ChoiceField(
                    choices=ws.get_wsa_id_set_choices(train_split=True,test_split=True),
                    required=True,
                    label='Check enrichment of',
                    initial=ws.eval_drugset,
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class NovDrugSetFieldType(FieldType):
    code='nov_ds'
    def add_to_form(self,ff):
        ws_id = self._context['ws_id']
        from browse.models import Workspace
        ws = Workspace.objects.get(pk=ws_id)
        field = forms.ChoiceField(
                    choices=ws.get_wsa_id_set_choices(),
                    required=True,
                    label='Check novelty against',
                    initial='tts',
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

### I wrote these two with the idea they could share a base class
### but that messed up some subregisty code and I didn't take the time
### to sort it out. The result is that the only difference b/t them is
### the _get_choices()
class RefreshPartsFieldType(FieldType):
    code='refresh_parts'
    def _get_choices(self,swf):
        return (
                swf.get_refresh_part_choices(),
                swf.get_refresh_part_initial(),
                )
    def add_to_form(self,ff):
        from browse.models import Workspace
        from workflows.refresh_workflow import StandardWorkflow
        ws = Workspace.objects.get(pk=self._context['ws_id'])
        swf = StandardWorkflow(ws=ws)
        choices,initial = self._get_choices(swf)
        field = forms.MultipleChoiceField(
                    choices=choices,
                    label='Include these subparts',
                    initial=initial,
                    widget=forms.SelectMultiple(
                            attrs={'size':len(choices)}
                            ),
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class ReviewPartsFieldType(FieldType):
    code='review_parts'
    def _get_choices(self,swf):
        return (
                swf.get_review_part_choices(),
                swf.get_review_part_initial(),
                )
    def add_to_form(self,ff):
        from browse.models import Workspace
        from workflows.refresh_workflow import StandardWorkflow
        ws = Workspace.objects.get(pk=self._context['ws_id'])
        swf = StandardWorkflow(ws=ws)
        choices,initial = self._get_choices(swf)
        field = forms.MultipleChoiceField(
                    choices=choices,
                    label='Include these subparts',
                    initial=initial,
                    widget=forms.SelectMultiple(
                            attrs={'size':len(choices)}
                            ),
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class ResumeScoresetIdFieldType(FieldType):
    code='resume_scoreset_id'
    def get_scoreset_choices(self):
        ws_id = self._context['ws_id']
        wf_name = self._context.get('wf_name')
        from browse.models import ScoreSet
        choices = [('', 'None')]
        from dtk.text import fmt_time,limit
        for id, user, ts, wf_job, desc in ScoreSet.objects.filter(
                ws=ws_id
                ).values_list(
                        'id', 'user', 'created', 'wf_job', 'desc'
                        ).order_by('-id'):
            ts = fmt_time(ts)
            if wf_name:
                if desc != wf_name:
                    continue
                label = f'{id} ({user} {wf_job} | {ts})'
            else:
                label = f'{id} ({user} {wf_job}_{desc} | {ts})'
            choices.append((str(id), limit(label)))
        return choices


    def add_to_form(self,ff):
        field = forms.ChoiceField(
                    label='Resume from scoreset',
                    required=False,
                    choices=self.get_scoreset_choices(),
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class DrugOrderingJobIDScoreFieldType(FieldType):
    code='wzs_ordering_jid'
    def add_to_form(self,ff):
        from browse.models import Workspace
        ws = Workspace.objects.get(pk=self._context['ws_id'])
        field = forms.ChoiceField(
                    label='WZS ordering',
                    choices = ws.get_prev_job_choices('wzs')
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class SelectModelJobIDFieldType(FieldType):
    code='select_model_jid'
    def add_to_form(self,ff):
        from browse.models import Workspace
        ws = Workspace.objects.get(pk=self._context['ws_id'])
        field = forms.ChoiceField(
                    label='Selectability Model',
                    choices = ws.get_prev_job_choices('selectabilitymodel'),
                    # Can't require this, might not be any usable models (e.g. tests).
                    required = False,
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class DrugOrderingJobIDFieldType(FieldType):
    code='ordering_jid'
    def add_to_form(self,ff):
        field = forms.IntegerField(
                    label='Drug Ordering Job',
                    required=True,
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)
class DrugOrderingScoreFieldType(FieldType):
    code='ordering_score'
    def add_to_form(self,ff):
        field = forms.CharField(
                    label='Drug Ordering Score',
                    required=True,
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class FlagCountFieldType(FieldType):
    code='flag_count'
    def add_to_form(self,ff):
        field = forms.IntegerField(
                    label='Drugs (or MoAs) to post-process',
                    initial=1000,
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code,type=int)

class UseCondensedFieldType(FieldType):
    code='use_condensed'
    def add_to_form(self,ff):
        field = forms.BooleanField(
                    label='Count via condensed',
                    initial=True,
                    required=False,
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code,type=bool)

class UseFDFEffAggFieldType(FieldType):
    code='use_fdf_eff_agg'
    def add_to_form(self,ff):
        field = forms.BooleanField(
                    label='Use FDF in Efficacy Aggregation?',
                    initial=True,
                    required=False,
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code,type=bool)

class UseFDFReduceFieldType(FieldType):
    code='use_fdf_reduce'
    def add_to_form(self,ff):
        field = forms.BooleanField(
                    label='Use FDF in Data Reduction?',
                    initial=True,
                    required=False,
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code,type=bool)

class EvalTestTrainFieldType(FieldType):
    code='eval_testtrain'
    def add_to_form(self,ff):
        field = forms.BooleanField(
                    label='Run WZS on Test/Train data split',
                    initial=True,
                    required=False,
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code,type=bool)


class UnwantedImportantProtSetType(FieldType):

    code='unwanted_imp_prot_set'
    def add_to_form(self, ff):
        from browse.models import Workspace
        ws = Workspace.objects.get(pk=self._context['ws_id'])
        field = forms.ChoiceField(
                    label='Unwanted Important Prot Set',
                    choices = [
                            ('','None'),
                            ]+ws.get_uniprot_set_choices(),
                    required=False,
                    )
        ff.add_field(self.code, field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)


class IndirectTargetImportance:
    label='Indirect Target Importance'
    name='indtrgscrimp'
    def diagram(self,cy,src_node):
        from dtk.grid_layout import VBox,HBox
        cy.add_link(src_node,self.name)
        cms = HBox()
        cms.mid.append(self.name)
        return VBox(mid=[self.name, cms])
    def add_to_workflow(self,wf,wsas):
        from workflows.candidate_workflow import add_indtrgscrimp_to_workflow
        add_indtrgscrimp_to_workflow(wf, self.name, wsas)

class DrugNotePrecompute:
    label='DrugNote Precompute'
    name='dnprecompute'
    def diagram(self,cy,src_node):
        from dtk.grid_layout import VBox,HBox
        cy.add_link(src_node,self.name)
        cms = HBox()
        cms.mid.append(self.name)
        return VBox(mid=[self.name, cms])
    def add_to_workflow(self,wf,wsas):
        from workflows.candidate_workflow import add_dnprecompute_to_workflow
        add_dnprecompute_to_workflow(wf, self.name, wsas)

class CandidatePartsFieldType(FieldType):
    code='candidate_parts'

    parts = [
                IndirectTargetImportance(),
                DrugNotePrecompute(),
                ]
    @classmethod
    def get_parts_initial(cls):
        return list(range(len(cls.parts)))

    def get_choices(self):
        return list(enumerate([p.label for p in self.parts]))

    def add_to_form(self,ff):
        choices = self.get_choices()
        field = forms.MultipleChoiceField(
                    choices=choices,
                    label='Include these subparts',
                    initial=self.get_parts_initial(),
                    widget=forms.SelectMultiple(
                            attrs={'size':len(choices)}
                            ),
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class AnimalModelPartsFieldType(FieldType):
    code='animalmodel_parts'
    @classmethod
    def get_parts_cls(cls, ws):
        parts = []
        # iterate through CMs in Parts order, letting each add its parts
        from runner.process_info import JobInfo
        CMs = [
            'gesig',
            'path',
            'customsig',
        ]
        for cm_name in CMs:
            uji = JobInfo.get_unbound(cm_name)
            uji.add_workflow_parts(ws,parts,nonhuman=True)
        return parts
    
    def get_parts(self):
        from browse.models import Workspace
        ws = Workspace.objects.get(pk=self._context['ws_id'])
        return self.get_parts_cls(ws)

    def get_parts_initial(self):
        return list(range(len(self.get_parts())))

    def get_choices(self):
        return list(enumerate([p.label for p in self.get_parts()]))

    def add_to_form(self,ff):
        choices = self.get_choices()
        field = forms.MultipleChoiceField(
                    choices=choices,
                    label='Include these subparts',
                    initial=self.get_parts_initial(),
                    widget=forms.SelectMultiple(
                            attrs={'size':len(choices)}
                            ),
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)


class Election(FieldType):
    code='election'
    def add_to_form(self,ff):
        ws = self.get_workspace()
        from browse.models import Election
        elections = Election.objects.filter(ws=ws)
        election_choices = [
                (election.id, '(%s) %s' % (election.id, election.elec_label(), ))
                for election in elections
                ]
        field = forms.ChoiceField(
                    choices=election_choices,
                    required=True,
                    label='Review Round',
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class Indication(FieldType):
    code='indication'
    def add_to_form(self,ff):
        ws = self.get_workspace()
        from browse.models import WsAnnotation
        wsa = WsAnnotation
        field = forms.ChoiceField(
                    choices=wsa.grouped_choices(),
                    required=True,
                    label='Indication',
                    initial=wsa.indication_vals.REVIEWED_PREDICTION,
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)


class MaxInputProts(FieldType):
    code='max_input_prots'
    def add_to_form(self,ff):
        field = forms.IntegerField(
                    label='Max Input Prots (Indirect Target Importance)',
                    required=True,
                    initial=5000
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class MinOTargProts(FieldType):
    code='min_otarg_prots'
    def add_to_form(self,ff):
        field = forms.IntegerField(
                    label='Min OpenTarget Prots',
                    required=True,
                    initial=20,
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)

class UseOTargKnownDrug(FieldType):
    code='use_otarg_knowndrug'
    def add_to_form(self,ff):
        field = forms.BooleanField(
                    label='Use OpenTarget Knowndrug?',
                    required=False,
                    initial=False,
                    )
        ff.add_field(self.code,field)
    def add_to_argparse(self,parser):
        parser.add_argument(self.code)


class ReviewRoundFieldType(FieldType):
    code='review_round'
    def _get_choices(self):
        from browse.models import Workspace, Election
        ws = Workspace.objects.get(pk=self._context['ws_id'])
        elections = Election.objects.filter(ws=ws)
        def label(e):
            return f'({e.id}) {e.elec_label()}'
        choices = [(e.id, label(e)) for e in elections]
        choices.reverse()
        return choices

    def add_to_form(self,ff):
        choices = self._get_choices()
        field = forms.MultipleChoiceField(
                    choices=choices,
                    label='Molecules from these rounds',
                    initial=[x[0] for x in choices],
                    widget=forms.SelectMultiple(
                            attrs={'size':len(choices)}
                            ),
                    )
        ff.add_field(self.code,field)
