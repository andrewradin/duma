import dtk.workflow as dwf

# For Modular workflow design see the refresh_workflow,
# from which this was modified

class TargetImportance:
    label='Target Importance'
    name='trgscrimp'
    def diagram(self,cy,src_node):
        from dtk.grid_layout import VBox,HBox
        cy.add_link(src_node,self.name)
        cms = HBox()
        cms.mid.append(self.name)
        return VBox(mid=[self.name, cms])
    def add_to_workflow(self,wf):
        dwf.TargetImportanceStep(wf,self.name,
# this was put in to keep memory usage reasonable during dev
# may not need this anymore, but selectivity runs faster than most of the other steps
                after=[Selectivity.name]
                )

class Flagger:
    label='Pre-screen Flagging'
    _flag_node='flag'
    name=_flag_node
    def diagram(self,cy,src_node):
        from dtk.grid_layout import VBox,HBox
        cy.add_link(src_node,self._flag_node)
        cy.add_link(TargetImportance.name,self._flag_node)
        return VBox(mid=[self._flag_node])
    def add_to_workflow(self,wf):
        # The important unwanted protein flagger requires target importance
        # to have run first.  (If it hasn't, that flagger won't do anything.)
        after = [TargetImportance.name]
        dwf.FlagStep(wf,self._flag_node,after=after)

class Novelty:
    label='Novelty estimation'
    name='novelty'
    parts=('lbn',)
    def __init__(self):
        self.agg=NoveltyAggregator()
    def diagram(self,cy,src_node):
        from dtk.grid_layout import VBox,HBox
        cms = HBox()
        for part in self.parts:
            cy.add_link(src_node,part)
            cy.add_link(part,self.agg.input_node)
            cms.mid.append(part)
        return VBox(mid=[
                self.agg.diagram(cy),
                cms,
                ])
    def add_to_workflow(self,wf):
        self.wf = wf # XXX stash for agg_inputs_ready (any better alternative?)
        # loop and instantiate all input CMs
        self.agg.hold('pass1')
        self.agg.set_callback(self._agg_inputs_ready)
        self.agg.enabled=True
        for part in self.parts:
            if part == 'lbn':
                dwf.LbnStep(wf,part,
                        add_drugs=self.wf.nov_ds,
                        )
                self.agg.inputs.append(part)
            else:
                raise NotImplementedError('unknown part: '+part)
                # XXX if we need more inputs here, then we should probably
                # XXX convert self.parts to a list of objects that can
                # XXX both supply the node name, and add the needed steps
                # XXX to a workflow; if they have post-processing, we might
                # XXX factor out the post-process handling for the main_CMs,
                # XXX and re-use it here
        dwf.LocalCode(self.wf,'nov_first_pass_complete',
                func=self._do_first_pass_complete,
                inputs={k:True for k in self.agg.inputs},
                )
    def _do_first_pass_complete(self,my_step):
        self.agg.release('pass1')
    def _agg_inputs_ready(self):
        fvs_name = 'nov_fvs'
        self.agg.create_fm(self.wf,
                score_type='novelty',
                fvs_name=fvs_name,
                fvs_flavor='wsa_novelty',
                training_set=self.wf.nov_ds,
                )
        ml_name = 'ml'
        dwf.MlStep(self.wf,ml_name,
                inputs={fvs_name:None},
                )

from .refresh_workflow import Aggregator,StandardWorkflow
class NoveltyAggregator(Aggregator):
    def __init__(self):
        super(NoveltyAggregator,self).__init__()
        self.input_node = 'nov_fvs'
        self.output_node = 'ml'

class Selectivity:
    label='Selectivity'
    name='selectivity'
    def diagram(self,cy,src_node):
        from dtk.grid_layout import VBox,HBox
        result = VBox(mid=[self.name])
        result.node_width = 100
        return result
    def add_to_workflow(self,wf):
        # Let everything else finish first
        dwf.SelectivityStep(wf,self.name)

class Selectability:
    label='Selectability'
    name='selectability'
    def diagram(self,cy,src_node):
        from dtk.grid_layout import VBox,HBox
        cy.add_link(src_node,self.name)
        cy.add_link(TargetImportance.name,self.name)
        cy.add_link(Novelty.parts[-1],self.name)
        cy.add_link(Selectivity.name,self.name)
        result = VBox(mid=[self.name])
        result.node_width = 100
        return result
    def add_to_workflow(self,wf):
        # Let everything else finish first
        after = [Novelty.parts[-1], TargetImportance.name, Selectivity.name]
        dwf.SelectabilityStep(wf,self.name,after=after)

class Stability:
    label='Stability'
    name='stability'
    def diagram(self,cy,src_node):
        from dtk.grid_layout import VBox,HBox
        cy.add_link(src_node,self.name)
        result = VBox(mid=[self.name])
        result.node_width = 80
        return result
    def add_to_workflow(self,wf):
        inputs = [
                wf.get_base_job(name)
                for name in ('wzs-test','wzs-train')
                ]
        if None in inputs:
            raise RuntimeError(
                    f"No test/train data for jid {wf.wzs_ordering_jid}",
                    )
        dwf.AprStep(wf,self.name,sources=[str(x)+'_wzs' for x in inputs])

class SingleProtein:
    label='Single Protein Scores'
    name='singleprotein'
    def diagram(self,cy,src_node):
        from dtk.grid_layout import VBox,HBox
        cy.add_link(src_node,self.name)
        result = VBox(mid=[self.name])
        result.node_width = 80
        return result
    def add_to_workflow(self,wf):
        if not wf.base_ss or wf.base_ss.desc != 'RefreshFlow':
            raise RuntimeError(
                    f"No refresh workflow for jid {wf.wzs_ordering_jid}",
                    )
        from dtk.prot_map import DpiMapping
        dpi_by_category = dict(DpiMapping.choices(ws=wf.ws))
        up_list = dpi_by_category.get('Uniprot',[])
        if not up_list:
            raise RuntimeError(
                    f"No uniprot dpi",
                    )
        # clone settings from base refresh workflow; if multiple runs were
        # required to build up the scoreset, we assume the last run had
        # the most correct settings
        # XXX maybe have a parameter to override this?
        sources = wf.base_ss.get_contributing_workflow_jobs()
        wf_job = sources[0]
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(wf_job.name)
        prev=uji.settings_defaults(ws=wf.ws)[wf_job.name]
        label='default'
        print('Source refresh jobs:')
        from dtk.text import compare_refresh_wf_settings
        for p in sources:
            cur = p.settings()
            print('  ',
                    p.status,
                    p.id,
                    compare_refresh_wf_settings(label,prev,cur),
                    )
            prev = cur
            label = 'prev'
        my_settings = sources[-1].settings()
        # disable aggregation
        agg_idx = wf.std_wf.get_refresh_part_agg_idx()
        my_settings['refresh_parts'] = [
                x
                for x in my_settings['refresh_parts']
                if int(x) != agg_idx
                ]
        #my_settings['refresh_parts'] = ["2"] # XXX speedup for testing
        # don't pick up a resume scoreset from the base workflow
        my_settings['resume_scoreset_id'] = ""
        # XXX As a potential speedup, we could clone the base scoreset,
        # XXX retaining any jobs that don't depend on the DPI setting,
        # XXX and pass that here as a resume_scoreset.
        my_settings['p2d_file'] = up_list[0][0]
        # Invoke a workstep to queue the refresh job
        dwf.RefreshWfStep(wf,self.name,model_settings=my_settings)

review_flow_parts = [
        SingleProtein,
        Stability,
        Flagger,
        TargetImportance,
        Selectability,
        Selectivity,
        Novelty,
        ]

class ReviewFlow(dwf.Workflow):
    # These select the form elements for configuring the reviewflow.
    # They correspond to FieldType classes in dtk/dynaform.py.
    _fields = [
            'review_parts',
            'resume_scoreset_id',
            'wzs_ordering_jid',
            'flag_count',
            'use_condensed',
            'p2d_file',
            'p2d_min',
            'nov_ds',
            'select_model_jid',
            ]
    #####
    # update scoreset
    #####
    def checkpoint(self):
        super(ReviewFlow,self).checkpoint()
        # At each checkpoint, add any new successful jobs to the scoreset
        self.add_done_steps_to_scoreset(self.scoreset)
    #####
    # start workflow
    #####
    def __init__(self,**kwargs):
        super(ReviewFlow,self).__init__(**kwargs)
        from browse.models import Workspace,TissueSet
        self.ws = Workspace.objects.get(pk=self.ws_id)
        if self.resume_scoreset_id:
            self.scoreset = self.resume_old_scoreset(self.resume_scoreset_id)
        else:
            self.scoreset = self.make_scoreset(self._label())
        self.std_wf = StandardWorkflow(ws=self.ws)
        self.parts = self.std_wf._post_eff_parts_loader()
        # find base_scoreset from wzs jid
        from browse.models import ScoreSetJob
        qs = ScoreSetJob.objects.filter(job_id=self.wzs_ordering_jid)
        if qs.count() == 1:
            self.base_ss = qs[0].scoreset
        else:
            self.base_ss = None
        dwf.LocalCode(self,'Executing...',
                func=self._do_nov,
                inputs={k:True for k in self._order},
                )
    def get_base_job(self,name):
        '''Return job_id from underlying refresh workflow based on stepname.'''
        from browse.models import ScoreSetJob
        if not self.base_ss:
            return None
        try:
            ssj = ScoreSetJob.objects.get(
                    scoreset=self.base_ss,
                    job_type=name,
                    )
        except ScoreSetJob.DoesNotExist:
            return None
        return ssj.job_id
    def _do_nov(self,my_step):
        for ind in self.review_parts:
            part = self.parts[int(ind)]
            part.add_to_workflow(self)
