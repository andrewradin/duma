from __future__ import print_function
from builtins import range
import dtk.workflow as dwf

# Modular workflow design
# - the StandardWorkflow class describes a work structure where:
#   - there are a number of data sources
#   - they are processed through some reusable processing pipelines
#   - the outputs are aggregated
#   - some post-aggregation stuff is run to prepare for review
# - the StandardWorkflow both manages how these parts are implemented,
#   and construction of a diagram showing what gets done in each part
# - in the UI, this work is split between a RefreshWorkflow, also
#   implemented in this file, and ReviewWorkflow
# - the RefreshWorkflow itself is only encapsulates the idea that "a bunch
#   of stuff happens, and then maybe you run an aggregation on the results"
# - the ReviewWorkflow is even less structured, and is just used to run
#   the list of things that happen after aggregation is complete
# - the StandardWorkflow object is based on this structure:
#   - there are a number of independent 'main' CMs that define the core
#     scores available in the system
#   - downstream of these CMs are a set of reusable post-processing blocks
#   - the exact behavior and routing within the post-processing blocks
#     can be configured for each main CM
#   - currently, there are 2 post-processing blocks implemented; the
#     PostD class configures post-processing for drug scores, and the
#     PostP class configures post-processing for protein scores
#   - there are also distinct pre-processing blocks upstream of the main
#     CMs; these primarily are used for diagram generation, because the
#     actual jobs will be run manually as part of setting up the workspace;
#     the existing blocks are PreClin, PreGE, PreGWAS, PreExt, and PrePheno
#     - in the PreClin case, the pre block will actually queue a faers job
#       for the requested clinical dataset, while also assuring that only
#       one such job gets run in the workflow, even pre is called multiple
#       times by different consumers
#   - the different pre- and post- block implementations have slightly
#     different interfaces, and so are not completely interchangeable; this
#     is reasonable (a CM that produces protein scores can't be paired
#     with a post block that processes drug scores), but is not very
#     strongly formalized yet
#   - all of this is configured in the main_CMs array; altering this array
#     changes the workflow and the diagram together
# - the RefreshWorkflow operates as follows:
#   - the 'refresh_parts' field in dtk.dynaform asks the StandardWorkflow
#     for a list of parts of the workflow that the user can select to
#     enable or disable
#   - the StandardWorkflow constructs this list by calling add_workflow_parts()
#     on the unbound JobInfo of each 'main' CM in its list; each plugin can
#     examine the list of parts so far, and can choose to add parts to the
#     list
#     - a 'part' is an instance of a duck-typed class, generally declared
#       in-line in each add_workflow_parts() implementation. It must have:
#       - a label property shown to the user in the selection list
#       - an add_to_workflow method that instantiates WorkSteps needed to
#         implement that part
#     - a part may also have an optional boolean enabled_default property;
#       if it exists, it controls whether that part is enabled by default
#       in a refresh workflow. Parts without this property are always enabled.
#     - the order of the entries in the main_CMs list controls the order of
#       the parts seen by the user
#     - a special part is appended to the list to control the WZS step
#   - the user selects the parts to run and starts the RefreshWorkflow
#   - the RefreshWorkflow passes the refresh_parts value from the settings
#     and gets back a list of 'part' objects to be run
#   - RefreshWorkflow then calls add_to_workflow on each part to instantiate
#     WorkSteps
#     - each add_to_workflow method implementation will typically add only
#       a single WorkStep directly, but will call add_pre_steps and
#       add_post_steps methods on the objects defining the pre- and post-
#       blocks to instantiate other WorkSteps
#     - add_to_workflow can instantiate LocalCode WorkSteps to set up
#       callbacks if it must examine the results from some WorkSteps before
#       instantiating others



class PreBase(object):
    def interconnect(self,cm,cy): pass
    @classmethod
    def add_pre_nodes(cls,cy): return None
    def add_pre_steps(self,wf): pass

class PreFaersSig(PreBase):
    order=5

class PreClin(PreBase):
    order=1
    def interconnect(self,cm,cy):
        cy.add_link('faers',cm)
    @classmethod
    def add_pre_nodes(cls,cy):
        from dtk.grid_layout import VBox
        return VBox(start=['faers'])
    def add_pre_steps(self,wf,cds):
        from dtk.faers import get_vocab_for_cds_name
        faers_name = get_vocab_for_cds_name(cds)+'_faers'
        if faers_name not in wf._order:
            dwf.FaersStep(wf,faers_name,
                    cds=cds,
                    search_term=wf.ws.get_disease_default(
                                    get_vocab_for_cds_name(cds)),
                    )
        return faers_name

class PreMolGSig(PreBase):
    order=1
    def interconnect(self,cm,cy):
        cy.add_link('gesig',cm)
    @classmethod
    def add_pre_nodes(cls,cy):
        from dtk.grid_layout import VBox
        return VBox(start=['gesig'])
    def add_pre_steps(self,wf,tissue_role,ts):
        input_name = f'{tissue_role}_gesig'
        if input_name not in wf._order:
            dwf.GESigStep(wf,input_name,
                    ts=ts,
                    thresh_overrides={},
                    )
        return input_name

class PreGE(PreBase):
    order=2
    def interconnect(self,cm,cy):
        cy.add_link('sig',cm)
    @classmethod
    def add_pre_nodes(cls,cy):
        cy.add_link('meta','sig')
        from dtk.grid_layout import VBox
        return VBox(start=['sig','meta'])

class PreGWAS(PreBase):
    order=3

class PreExt(PreBase):
    order=4

class PrePheno(PreBase):
    order=6

class PostD:
    def __init__(self,direct,gpbr):
        self.direct = direct
        self.gpbr = gpbr
    def order(self):
        if self.gpbr:
            return 1
        return 2
    def interconnect(self,cm,cy,agg):
        if self.direct:
            cy.add_link(cm,agg)
        if self.gpbr:
            cy.add_link(cm,'gpbr')
    def add_post_steps(self,wf,src_step):
        if ':' in src_step:
            src_step,name_base = src_step.split(':')
        else:
            name_base = src_step
        if self.direct:
            wf.eff.inputs.append(src_step)
        if self.gpbr:
            gpbr_name = name_base+'_gpbr'
            dwf.gpbrStep(wf,gpbr_name,
                    inputs={src_step:True},
                    )
            wf.eff.inputs.append(gpbr_name)

class PostP:
    def __init__(self,codes,sigdif_codes,glf,sigdif_glf):
        self.codes = codes
        self.sigdif_codes = sigdif_codes
        self.glf = glf
        self.sigdif_glf = sigdif_glf
    def order(self):
        if self.sigdif_codes or self.sigdif_glf:
            return 5
        if self.codes:
            return 4
        return 3
    def interconnect(self,cm,cy,agg):
        if self.codes:
            cy.add_link(cm,'codes')
        if self.glf:
            cy.add_link(cm,'glf')
        if self.sigdif_codes or self.sigdif_glf:
            cy.add_link(cm,'sigdif')
    def _add_codes(self,wf,src_step,src_score,name_base):
        codes_name = name_base+'_codes'
        dwf.CodesStep(wf,codes_name,
                inputs={src_step:src_score},
                )
        wf.eff.inputs.append(codes_name)
    def _add_glf_depend(self,wf,src_step,src_score,name_base):
        glf_name = name_base+'_glf'
        dwf.GlfStep(wf,glf_name,
                inputs={src_step:src_score},
                )
        depend_name = glf_name+'_depend'
        dwf.DependStep(wf,depend_name,
                inputs={glf_name:True},
                )
        wf.eff.inputs.append(depend_name)
    def add_post_steps(self,wf,src_step,src_score):
        if ':' in src_step:
            src_step,name_base = src_step.split(':')
        else:
            name_base = src_step
        if self.codes:
            self._add_codes(wf,src_step,src_score,name_base)
        if self.glf:
            self._add_glf_depend(wf,src_step,src_score,name_base)
        if self.sigdif_codes or self.sigdif_glf:
            sigdif_name = name_base+'_sigdif'
            dwf.SigDifStep(wf,sigdif_name,
                    ws=wf.ws,
                    inputs={src_step:src_score},
                    )
            if self.sigdif_codes:
                self._add_codes(wf,sigdif_name,'difEv',sigdif_name)
            if self.sigdif_glf:
                self._add_glf_depend(wf,sigdif_name,'difEv',sigdif_name)

class PostMolGSig:
    def __init__(self):
        pass
    def order(self):
        return 2
    def interconnect(self,cm,cy,agg):
        pass
    def add_post_steps(self,wf,src_step):
        if ':' in src_step:
            src_step,name_base = src_step.split(':')
        else:
            name_base = src_step

        defus_name = name_base+'_defus'
        dwf.DefusStep(wf,defus_name,
                inputs={src_step:True},
                )
        wf.eff.inputs.append(defus_name)

class FDF:
    label='FVS Drug Filter'
    _node='fdf'
    def diagram(self,cy,src_node):
        from dtk.grid_layout import VBox,HBox
        cy.add_link(src_node,self._node)
        return VBox(mid=[self._node])
    def add_to_workflow(self,wf):
        dwf.FdfStep(wf,self._node,
                )

class Aggregator(object):
    # This class:
    # - creates the aggregation portion of the diagram
    # - hosts the list that accumulates WorkSteps that provide
    #   aggregation inputs
    # - implements a hold/release protocol for determining when to start
    #   aggregation WorkSteps
    def __init__(self):
        self.inputs = []
        self.enabled=False
        self.holds=set()
    def diagram(self,cy):
        from dtk.grid_layout import VBox,HBox
        inp = self.input_node
        outp = self.output_node
        cy.add_link(inp,outp)
        return VBox(mid=[outp,inp])
    def set_callback(self,ready_callback):
        self._ready_callback = ready_callback
    def hold(self,hold_name):
        self.holds.add(hold_name)
    def release(self,hold_name):
        self.holds.remove(hold_name)
        if self.holds:
            return
        if not self.enabled:
            return # this will terminate the workflow job
        self._ready_callback()
    def create_fm(self,wf,score_type,fvs_name,fvs_flavor,training_set,
            presets={},
            ):
        # service routine for running fvs on aggregator scores
        inputs = {}
        from runner.process_info import JobInfo
        for name in sorted(self.inputs):
            if name in presets:
                codes = presets[name]
            else:
                bji = JobInfo.get_bound(wf.ws, wf.step(name).job_id)
                cat = bji.get_data_catalog()
                codes = [
                        dc_code
                        for dc_code in cat.get_codes('wsa','feature')
                        if all([
                                not cat.is_type(dc_code,'meta_out'),
                                cat.is_type(dc_code,score_type),
                                ])
                        ]
            print(name, codes)
            inputs[name] = codes
        # start fvs
        dwf.FvsStep(wf,fvs_name,
                flavor=fvs_flavor,
                training_set=training_set,
                inputs=inputs,
                )

class EfficacyAggregator(Aggregator):
    # In addition to the functions of the Aggregator base class, this
    # class functions as a 'Part' that the user can select to run
    # or not in the RefreshFlow.
    # XXX currently, it only supplies a label and not an add_to_workflow,
    # XXX because the latter is done in-line; this may change
    def __init__(self):
        super(EfficacyAggregator,self).__init__()
        self.label = 'Efficacy Aggregation' # to function as a Part
        self.input_node = 'eff_fvs'
        self.output_node = 'wzs'
        self.output_score = 'wzs' # for post-eff stuff

from dtk.lazy_loader import LazyLoader
class StandardWorkflow(LazyLoader):
    _kwargs = ['ws']
    from collections import namedtuple
    I=namedtuple('CmInfo','name pre post')
    main_CMs = [
            I('path',PreGE(),PostD(1,1)),
            I('gesig',PreGE(),PostP(1,1,1,1)),
            I('esga',PreGWAS(),PostD(1,0)),
            I('gwasig',PreGWAS(),PostP(1,1,1,1)),
            I('gpath',PreGWAS(),PostD(1,1)),
            I('tcgamut',PreExt(),PostP(1,1,1,1)),
            I('dgn',PreExt(),PostP(1,1,1,1)),
            I('otarg',PreExt(),PostP(1,1,1,1)),
            I('agr',PreExt(),PostP(1,1,1,1)),
            I('capp',PreClin(),PostD(1,1)),
            I('defus',PreClin(),PostD(1,0)),
            I('faerssig',PreFaersSig(),PostP(1,0,1,0)),
            I('customsig',PreExt(),PostP(1,1,1,1)),
            I('misig',PrePheno(),PostP(1,1,1,1)),
            I('mips',PrePheno(),PostD(1,1)),
            I('molgsig',PreMolGSig(),PostMolGSig()),
            ]
    def _pre_eff_parts_loader(self):
        parts = []
        # iterate through CMs in Parts order, letting each add its parts
        from runner.process_info import JobInfo
        for cm_info in self.main_CMs:
            uji = JobInfo.get_unbound(cm_info.name)
            uji.add_workflow_parts(self.ws,parts)
        return parts
    def _eff_agg_loader(self):
        return EfficacyAggregator()
    def _post_eff_parts_loader(self):
        from .review_workflow import review_flow_parts
        return [x() for x in review_flow_parts]
    def _parts_loader(self):
        return self.pre_eff_parts+[self.eff_agg]+self.post_eff_parts
    def _name_idx_loader(self):
        return {x.name:x for x in self.main_CMs}
    def get_main_cm_info(self,name):
        return self.name_idx[name]
    def _get_refresh_parts(self):
        return self.pre_eff_parts+[self.eff_agg]
    def get_refresh_part_choices(self):
        return list(enumerate(p.label for p in self._get_refresh_parts()))
    def get_refresh_part_agg_idx(self):
        return self._get_refresh_parts().index(self.eff_agg)
    def get_refresh_part_initial(self):
        def ok(part):
            # if the part implements an 'enabled_default' member, use that
            try:
                return part.enabled_default
            except AttributeError:
                return True # otherwise enable
        return [
                i
                for i,part in enumerate(self._get_refresh_parts())
                if ok(part)
                ]
    def get_review_part_choices(self):
        return list(enumerate([p.label for p in self.post_eff_parts]))
    def get_review_part_initial(self):
        # all review parts are enabled by default
        return [x[0] for x in self.get_review_part_choices()]
    # XXX this is a little confusing:
    # XXX - the review workflow has a shorter parts list, with correspondingly
    # XXX   different idx numbering
    # XXX - so, it doesn't work with anything based on _enabled_parts below;
    # XXX   enabled_post_eff_parts is only called in a loop for handling the
    # XXX   no-longer-used case where a post_eff part is enabled during
    # XXX   review
    # XXX - and, in fact, they only work with the review workflow because
    # XXX   the review parts are at the front of the parts list
    # XXX - this probably indicates that some of the choices/enables logic
    # XXX   below should be removed from here and placed in the corresponding
    # XXX   workflows
    def _enabled_parts(self,idxs,subset):
        '''Return list of 'Part-like' objects matching idxs.'''
        idxs = set([int(x) for x in idxs])
        subset = set(subset)
        out = [
                part
                for i,part in enumerate(self.parts)
                if i in idxs and part in subset
                ]
        return out
    def enabled_pre_eff_parts(self,idxs):
        return self._enabled_parts(idxs,self.pre_eff_parts)
    def is_eff_enabled(self,idxs):
        return bool(self._enabled_parts(idxs,[self.eff_agg]))
    def enabled_post_eff_parts(self,idxs):
        return self._enabled_parts(idxs,self.post_eff_parts)
    def diagram(self,cy):
        from dtk.grid_layout import VBox,HBox
        frame = VBox()
        frame.mid.append(self._post_eff_diagram(cy))
        frame.mid.append(self.eff_agg.diagram(cy))
        frame.mid.append(VBox(margin=30))
        frame.mid.append(self._post_cm_diagram(cy))
        frame.mid.append(VBox(margin=30))
        frame.mid.append(self._main_cm_diagram(cy))
        positions = []
        frame.layout(positions,0,0)
        cy.set_abs_positions(positions)
    def _post_eff_diagram(self,cy):
        from dtk.grid_layout import VBox,HBox
        result = HBox()
        for item in self.post_eff_parts:
            result.mid.append(item.diagram(cy,self.eff_agg.output_node))
        return result
    def _post_cm_diagram(self,cy):
        # This isn't delegated to the Post classes because the
        # layout is somewhat customized on a global level to
        # get the interconnects to show up more clearly
        post_classes = set([
                x.post.__class__
                for x in self.main_CMs
                ])
        # if this assert fires, more custom layout is needed below
        assert post_classes == set([PostD,PostP,PostMolGSig])
        from dtk.grid_layout import VBox,HBox
        cy.add_link('gpbr',self.eff_agg.input_node)
        cy.add_link('codes',self.eff_agg.input_node)
        cy.add_link('sigdif','codes')
        cy.add_link('sigdif','glf')
        cy.add_link('glf','depend')
        cy.add_link('depend',self.eff_agg.input_node)
        cy.add_link('molgsig', 'defus')
        return HBox(
                start=['gpbr'],
                end=[
                    VBox(start=[HBox(mid=['codes'],margin=80)],margin=60),
                    VBox(mid=['depend','glf','sigdif']),
                    ],
                )
    def _main_cm_diagram(self,cy):
        from dtk.grid_layout import VBox,HBox
        # the margin here pushes the Post layer out to the sides,
        # so the connection links are clearer
        result = HBox(margin=100)
        # extract and order data source groups
        groups = {}
        for cm_info in self.main_CMs:
            l = groups.setdefault(cm_info.pre.__class__,[])
            l.append(cm_info.name)
        ordered_groups = sorted(list(groups.keys()),key=lambda x:x.order)
        # order CMs within groups
        for group in ordered_groups:
            frame = VBox()
            result.mid.append(frame)
            # the margin here leaves a little space between groups
            cm_box = HBox(margin=40)
            frame.start.append(cm_box)
            ordered_cms = sorted(
                groups[group],
                key=lambda x:self.get_main_cm_info(x).post.order(),
                )
            for cm in ordered_cms:
                cm_box.mid.append(cm)
            pre_box = group.add_pre_nodes(cy)
            if pre_box:
                frame.start.append(pre_box)
        # set up nodes and interconnects
        for cm_info in self.main_CMs:
            cm_info.pre.interconnect(cm_info.name,cy)
            cm_info.post.interconnect(cm_info.name,cy,self.eff_agg.input_node)
        return result

class RefreshFlow(dwf.Workflow):
    _fields = [
            'refresh_parts',
            'resume_scoreset_id',
            'p2d_file',
            'p2d_min',
            'p2p_file',
            'p2p_min',
            'use_fdf_eff_agg',
            'use_fdf_reduce',
            'min_otarg_prots',
            'use_otarg_knowndrug',
            'eval_testtrain',
            'eval_ds',
            ]
    #####
    # update scoreset
    #####
    def checkpoint(self):
        super(RefreshFlow,self).checkpoint()
        # At each checkpoint, add any new successful jobs to the scoreset
        self.add_done_steps_to_scoreset(self.scoreset)

    def _validate(self):
        """Sanity-check any settings that might conflict."""

        dpi_choice = self.p2d_file
        opt_choice = self.eval_ds
        doing_eff = self.std_wf.is_eff_enabled(self.refresh_parts)
        if doing_eff:
            from dtk.prot_map import DpiMapping
            dpi_type = DpiMapping(dpi_choice).get_dpi_type() 
            err = f"DPI type '{dpi_type}' incompatible with opt target '{opt_choice}'"
            if dpi_type == 'moa':
                assert 'moa-' in opt_choice, err
            else:
                assert 'moa-' not in opt_choice, err


    #####
    # start workflow
    #####
    def __init__(self,**kwargs):
        super(RefreshFlow,self).__init__(**kwargs)
        from browse.models import Workspace,TissueSet
        self.ws = Workspace.objects.get(pk=self.ws_id)
        if self.resume_scoreset_id:
            self.scoreset = self.resume_old_scoreset(self.resume_scoreset_id)
        else:
            self.scoreset = self.make_scoreset(self._label())

        self.std_wf = StandardWorkflow(ws=self.ws)
        print('Refresh Workflow for ws',self.ws.id,'with parts:')
        for idx in self.refresh_parts:
            print('  ',self.std_wf.parts[int(idx)].label)
        # This represents the efficacy aggregation part.
        self.eff = self.std_wf.eff_agg
        self._validate()
        # Aggregation can't just rely on an input list to delay a pre-queued
        # job step, because some CMs, like otarg, don't know the full set
        # of inputs they'll produce until somewhere into the process. So,
        # Aggregator implements a hold/release mechanism where the job
        # will be queued when all holds are released. The 'first_pass' hold
        # delays the aggregation until all inputs queued in the first set
        # of add_to_workflow calls have completed. If a CM wants to delay
        # starting further, it needs to set a hold of its own, and arrange
        # to release it when appropriate.
        self.eff.hold('first_pass')
        self.eff.set_callback(self._do_eff_agg_ready)
        # For each enabled part in the standard workflow, call the appropriate
        # start function. When all the jobs triggered by those start functions
        # complete, call _do_first_pass_complete.
        for part in self.std_wf.enabled_pre_eff_parts(self.refresh_parts):
            part.add_to_workflow(self)
        if self.std_wf.is_eff_enabled(self.refresh_parts):
            self.eff.enabled = True
        dwf.LocalCode(self,'first_pass_complete',
                func=self._do_first_pass_complete,
                inputs={k:True for k in self._order},
                )
    def _do_first_pass_complete(self,my_step):
        self.eff.release('first_pass')
    def _do_eff_agg_ready(self):
        self.eff_fvs_presets = {'knowndrug_otarg_codes': []}
        fvs_name = 'eff_fvs'
        self.eff.create_fm(self,
                score_type='efficacy',
                fvs_name=fvs_name,
                fvs_flavor='wsa_efficacy',
                training_set=self.ws.eval_drugset,
                presets=self.eff_fvs_presets,
                )
        fm_src = fvs_name
        if self.use_fdf_eff_agg:
            fdf_name = fvs_name+'_fdf'
            dwf.FdfStep(self,fdf_name,
                        inputs={fm_src:None},
                        )
            fm_src = fdf_name
# do the test/train split (if applicable) before the final WZS which should represent the final action
        wzs_name = 'wzs'
        # Could probably just use self.eval_testtrain going forward, but it was causing trouble
        # for scripted WF invocations cloned from settings prior to this existing.
        base_ds = self.ws.eval_drugset
        if hasattr(self, 'eval_ds'):
            base_ds = self.eval_ds
        if getattr(self, 'eval_testtrain', False):

            for grp in ['train', 'test']:
                split_drugset_name = "-".join(['split',grp, base_ds])
                try:
                    from dtk.kt_split import get_split_drugset
                    get_split_drugset(split_drugset_name, self.ws)
                    t_step = dwf.WzsStep(self,wzs_name + '-' +grp,
                            inputs={fm_src:None},
                            )
                    t_step.auto_drug_set = split_drugset_name
                except Exception as e:
                    print("Couldn't find a KT split", e)

        wzs_step = dwf.WzsStep(self,wzs_name,
                # WZS defaults are now set for efficacy aggregation
                inputs={fm_src:None},
                )
        if hasattr(self, 'eval_ds'):
            wzs_step.auto_drug_set = self.eval_ds

        # NOTE: currently the post_eff part of the workflow is done
        # separately, in the Review workflow. So this loop doesn't
        # do anything.
        for part in self.std_wf.enabled_post_eff_parts(self.refresh_parts):
            part.add_to_workflow(self)
