from __future__ import print_function
import six
# The recommended way to test new workflows is via scripts/run_workflow:
# - running with --rethrow immediately exposes any errors in the setup of
#   a new workstep (alternatively, you can temporarily code
#   self.rethrow=True into the workflow ctor)
# - this creates a wf_checkpoint.json file, which can be used with the -r
#   option to skip over any successful steps in the workflow; ideally, you
#   would maintain a safe copy of this file for use with -r, and update
#   that copy from wf_checkpoint.json each time another step completed
#   successfully
################################################################################
# Parameter lookup helper
################################################################################
class ParmLookup(object):
    def __init__(self,parms,defaults={},fallback=None):
        self._parms = parms
        self._defaults = defaults
        self._fallback = fallback
    def add_defaults(self,**kwargs):
        self._defaults.update(kwargs)
    def get(self,n):
        try:
            return self._parms[n]
        except KeyError:
            pass
        if self._fallback:
            try:
                return getattr(self._fallback,n)
            except AttributeError:
                pass
        try:
            return self._defaults[n]
        except KeyError:
            pass
        raise AttributeError("no attribute '%s'"%n)
    def __str__(self):
        return 'ParmLookup'+repr(self._parms)

################################################################################
# Workstep and Workflow base classes
#
# Theory of Operation
# - a workflow consists of a collection of uniquely-named steps
# - each step has a state from the following hierarchy:
#     pending - the step has not yet started
#       ready - any prerequisite steps are complete
#       blocked - some prerequiste step is not yet complete
#     dispatched - the step has started
#     complete - the step has completed
#       error - the step completed with an error condition
#       ok - the step completed without error
# - a mix of synchronous and asynchronous steps are supported:
#   - each step implements a _start() method
#   - when a step is complete, it must arrange for calling a done() method
#     - for synchronous steps, this may be called inside _start()
#     - for asynchronous steps, _start() may register a poll function, which
#       will call done() at some future point
#   - the workflow method run_to_completion() will continuously cycle, starting
#     any ready steps, and calling any registered poll functions, until the
#     entire workflow is complete
################################################################################
class WorkStep(object):
    # states
    PEND='pending'
    ACTV='dispatched'
    DONE='complete'
    # substates for PEND
    BLKD='blocked'
    REDY='ready'
    def __init__(self,workflow,name,after=None,**kwargs):
        # if there's an 'inputs' arg, it's a dict whose keys are the names
        # of the job steps providing inputs, and whose values are additional
        # information about those inputs (usually as a dict); use this to
        # construct an 'after' param
        if 'inputs' in kwargs:
            after = list(kwargs['inputs'].keys())
        self._workflow = workflow
        self._name = name
        self._state = self.PEND
        self._after = after or []
        self._pl = ParmLookup(
                kwargs,
                dict(
                        debug = False,
                        rethrow = False,
                        ),
                self._workflow,
                )
        workflow._register(self,name)
    def restart_data(self): return None
    def single_input(self):
        # a convenience function for case where self.inputs is expected
        # to be a single-key hash: verify that fact and return the key,value
        inputs = self.inputs
        assert(len(inputs) == 1)
        return next(six.iteritems(inputs))
    def step(self,n):
        return self._workflow.step(n)
    def __getattr__(self,n):
        return self._pl.get(n)
    def get(self,n,default=None):
        try:
            return getattr(self,n)
        except AttributeError:
            return default
    def state(self):
        substate = None
        if self._state == self.PEND:
            substate = self.REDY if all([
                    self._workflow._steps[name].state()[0] == self.DONE
                    for name in self._after
                    ]) else self.BLKD
        elif self._state == self.DONE:
            substate = self._error
        return (self._state,substate)
    def _dispatch(self):
        if self.debug:
            print('dispatching',self,self._name,self._pl)
        # now flip the state
        self._state = self.ACTV
        # and initiate activity
        try:
            self._start() # must be implemented by derived class
        except Exception as ex:
            if self.debug:
                print(self._name,'got exception',ex)
            if self.rethrow:
                raise
            import traceback
            print(self._name,'got exception',traceback.format_exc())
            self.done(str(ex))
    def done(self,err=None):
        # can be called directly from _start() by an in-line derived class,
        # or must somehow be called on completion of a background derived class
        # XXX Depending on the callback mechanism for async tasks, this
        # XXX will probably need some kind of locking protection
        self._error = err
        self._state = self.DONE
    def abort(self):
        # in the base class this does nothing; for asynchronous steps,
        # BackgroundStepBase overrides it to pass the abort to the process
        # system; long-running synchronous steps should periodically check
        # self._workflow._aborting and exit if set
        pass

class Workflow(object):
    _flows=None
    @classmethod
    def wf_list(cls,ws=None):
        # XXX To be moved:
        # - workflow/combo_flow.py - runs a combo treatment flow; lots of
        #   testing junk in the file that needs clearing out
        # Left in place:
        # - pi_eval/drop_prot_workflow.py - this isn't complete; it's an
        #   attempt at trying to thin very dense protein interaction
        #   networks by randomly dropping proteins
        if cls._flows is None:
            flows = []
            wf_dir = 'workflows'
            from dtk.files import scan_dir,name_match
            from path_helper import PathHelper
            for fn in scan_dir(
                            PathHelper.website_root+wf_dir,
                            filters=[name_match(r'.*_workflow.py$')],
                            output=lambda x:x.filename,
                            ):
                modname = fn.rstrip('.py')
                mod = __import__(wf_dir+'.'+modname)
                mod = getattr(mod,modname)
                for k,v in six.iteritems(mod.__dict__):
                    try:
                        if not issubclass(v,cls):
                            continue
                        # by convention, leaf classes (which are runnable)
                        # have names ending in 'Flow', and base and
                        # intermediate classes have names ending in
                        # 'Workflow'; don't display the latter
                        if v.__name__.endswith('Workflow'):
                            continue
                        flows.append(v)
                    except TypeError:
                        continue
            flows.sort(key=lambda x:x.__name__.lower())
            cls._flows = flows
        class Wrapper:
            # This exists primarily because, since all workflow parameters
            # are optional, you can't pass the class to a template (it will
            # try to instantiate it as a callable).  This protects the class
            # so that some class methods can be accessed in the template.
            def __init__(self,cls):
                self.cls = cls
            def code(self):
                return self.cls._code()
            def jobname(self):
                return 'wf_%d_%s'%(ws.id,self.code(),)
            def label(self):
                return self.cls._label()
        return [Wrapper(x) for x in cls._flows]
    @classmethod
    def get_by_code(cls,code):
        for wrapper in cls.wf_list():
            if wrapper.code() == code:
                return wrapper
        raise ValueError("no workflow '%s'" % code)
    @classmethod
    def _label(self):
        # can be overridden to provide prettier workflow naming for UI
        return self._code()
    @classmethod
    def _code(self):
        return self.__name__
    def __init__(self,**kwargs):
        self._order = []
        self._steps = {}
        self._pl = ParmLookup(
                kwargs,
                dict(
                    wrapper_job_id=None,
                    ),
                )
        self._poll_funcs = set()
        self._checkpoint_filename = None
        self._aborting = False
        self._abort_propagated = False
    def __getattr__(self,n):
        return self._pl.get(n)
    def add_poll_function(self,f):
        self._poll_funcs.add(f)
    def remove_poll_function(self,f):
        self._poll_funcs.discard(f)
    def step(self,n):
        return self._steps[n]
    def _register(self,step,name):
        assert name not in self._steps
        self._steps[name] = step
        self._order.append(name)
    def restart_data(self):
        result = {}
        for name in self._order:
            step = self._steps[name]
            data = step.restart_data()
            if data:
                result[name] = data
        return result
    def checkpoint(self):
        if not self._checkpoint_filename:
            return
        with open(self._checkpoint_filename,'w') as f:
            import json
            json.dump(
                    self.restart_data(),
                    f,
                    sort_keys=True,
                    indent=4,
                    separators=(',',':'),
                    )
            f.write('\n')
    def dump(self):
        # debug output of state of each step
        # XXX model for eventual cytoscape rendering
        for name in self._order:
            step = self._steps[name]
            print(name,step.state())
    def abort(self):
        # try to avoid race conditions with cycle() by just setting a
        # flag here, and having cycle() cache the flag value on entry
        self._aborting = True
    def cycle(self):
        # push processing forward -- start any ready jobs
        # may be called at any time (won't do anything if nothing to be done)
        abort = self._aborting
        if abort and not self._abort_propagated:
            for step in self._steps.values():
                if step.state()[0] == WorkStep.ACTV:
                    step.abort()
            # abort_propagated was an attempt to prevent multiple aborts
            # from being sent to the same process, but it seems more
            # responsive if we allow cycle() to re-issue aborts on every
            # call until nothing is in the ACTV state. If this causes problems,
            # uncomment the next line.
            #self._abort_propagated = True
        for step in list(self._steps.values()):
            if step.state()[1] == WorkStep.REDY:
                # check for upstream errors or an abort
                skip = any([
                    self.step(name).state()[1]
                    for name in step._after
                    ])
                if skip:
                    step.done('Upstream error')
                elif abort:
                    step.done('Abort')
                else:
                    step._dispatch()
    def complete(self):
        # this means all steps completed; no more activity pending
        return all([
                step.state()[0] == WorkStep.DONE
                for step in self._steps.values()
                ])
    def succeeded(self):
        # completed, plus all steps were successful
        return all([
                step.state() == (WorkStep.DONE,None)
                for step in self._steps.values()
                ])
    def running(self):
        # this means we're awaiting at least one done() call
        return any([
                step.state()[0] == WorkStep.ACTV
                for step in self._steps.values()
                ])
    def run_to_completion(self,
            do_sleep=True,
            dump_on_failure=True,
            periodic_dump=False,
            ):
        if do_sleep:
            import time
            self.add_poll_function(lambda:time.sleep(1))
        if periodic_dump:
            self.add_poll_function(self.dump)
        while not self.complete():
            if self.running():
                for f in self._poll_funcs:
                    f()
            self.cycle()
        if dump_on_failure and not self.succeeded():
            self.dump()
    def make_scoreset(self,desc):
        # save the master scoreset record
        from browse.models import ScoreSet
        ss = ScoreSet(
                ws=self.ws,
                user=self.user,
                desc=desc,
                wf_job=self.wrapper_job_id,
                )
        ss.save()
        return ss
    def resume_old_scoreset(self,ss_id):
        # This method leverages bind_old_job() to let you restart a failed
        # workflow that produces a scoreset. To use:
        # - add 'resume_scoreset_id' to the workflow _fields list
        # - in the workflow __init__, if resume_scoreset_id is set, call
        #   this rather than make_scoreset()
        # - in the workflow checkpoint override, call
        #   add_done_steps_to_scoreset() (which is probably already
        #   happening in any workflow producing a scoreset)
        # See RefreshFlow for an example.
        from browse.models import ScoreSet
        ss = ScoreSet.objects.get(ws=self.ws,pk=ss_id)
        self.restart_from = {
                ssj.job_type:{'job_id':ssj.job_id}
                for ssj in ss.scoresetjob_set.all()
                }
        return ss
    def add_to_scoreset(self,ss,step):
        from browse.models import ScoreSetJob
        ssj = ScoreSetJob(
                scoreset=ss,
                job_id=step.job_id,
                job_type=step._name,
                )
        ssj.save()
    def save_scoreset(self,
            desc,
            steps=None,
            ):
        if not steps:
            steps = []
            for name in self._order:
                step = self._steps[name]
                if hasattr(step,'job_id'):
                    steps.append(step)
        ss = self.make_scoreset(desc)
        for step in steps:
            self.add_to_scoreset(ss,step)
        return ss.id
    def add_done_steps_to_scoreset(self,ss):
        recorded_jobs = set(
                ss.scoresetjob_set.values_list('job_id',flat=True)
                )
        for step in self._steps.values():
            if hasattr(step,'job_id'):
                if step.state() == (WorkStep.DONE,None):
                    if step.job_id not in recorded_jobs:
                        self.add_to_scoreset(ss,step)

################################################################################
# Wrapper to simplify running workflow-specific code as a workstep
################################################################################
class LocalCode(WorkStep):
    def _start(self):
        # slightly tricky -- self.func is not binding a class method;
        # it's just looking up a parameter stored under the name 'func'.
        # So the explicit passing of self as a parameter is required
        # in order for the function to have access to its WorkStep.
        # Note that in one use-case, func could actually be a bound
        # method of the Workflow, so in that case, the Workflow is the
        # first parameter, and the WorkStep is the second.
        self.func(self)
        # this workstep type allows code to be executed in-line, so once
        # that code returns, we're done immediately; contrast this to
        # background tasks, which would only be launched in the _start
        # method, and would need a separate completion polling mechanism
        # to call done()
        self.done()

################################################################################
# Common base class for running a Duma background job as a WorkStep
# - The derived class must implement:
#   - _get_plugin_info(), which returns the plugin name and the job name
#   - _build_settings(), which returns a dict holding the job settings.
#     The recommended procedure for constructing this dict is:
#     - call settings_defaults() on the unbound job info object, and extract
#       the appropriate set of defaults; this may depend on things like the
#       workspace or tissue set, but is not customized beyond that
#     - modify the settings dict based on workflow or workstep parameters;
#       for each parameter, the code may require it to be present, or may
#       leave the default in place if it's not specified; some parameters
#       like thresh_overrides may require complex internal parsing and
#       affect multiple settings values -- if the handling for these is
#       needed in several different worksteps, a service routine can
#       be added here
# XXX Note that most plugins don't yet implement settings_defaults(), and
# XXX so end up with a hard-coded set of defaults in the derived workstep
# XXX below. These should be converted over time, as it's easier to maintain
# XXX as settings change.
# Conversion plan:
# - gpath - finish pathsum-related cases
# - gesig - finish tissueset-related cases
# - glee, struct - other complex cases
# - remaining simple cases (maybe as-needed)
################################################################################
class BackgroundStepBase(WorkStep):
    active_jobs = {}
    new_launch = False
    def _start(self):
        settings = self._build_settings()
        if self.bind_old_job():
            if self.debug:
                print(self._name,'bound to previous job_id',self.job_id)
            self.done()
            return
        # queue the job for execution
        from runner.process_info import JobCrossChecker
        import json
        jcc=JobCrossChecker()
        plugin,jobname = self._get_plugin_info()
        if not jobname:
            jobname = plugin + '_%d'%self.ws.id
        job_id=jcc.queue_job(
                plugin,
                jobname,
                user=self.user,
                settings_json=json.dumps(
                        settings,
                        separators=(',',':'),
                        sort_keys=True,
                        ),
                )
        if job_id:
            # refer to class explicitly so that it works
            # even with intermediate derived classes
            BackgroundStepBase.active_jobs[job_id] = self
            BackgroundStepBase.new_launch = True
            self._workflow.add_poll_function(BackgroundStepBase.poll)
            self.job_id = job_id
            print('starting',self._name,'job',self.job_id)

            # If possible, let's track which jobs are associated with which workflows in the db.
            # Not all workflows seem to set wrapper_job_id, but the important ones do.
            if getattr(self._workflow, 'wrapper_job_id'):
                from browse.models import WorkflowJob
                WorkflowJob.objects.create(
                    wf_job_id=self._workflow.wrapper_job_id,
                    child_job_id=job_id,
                )

    def restart_data(self):
        try:
            return dict(job_id=self.job_id)
        except AttributeError:
            return None
    def bind_old_job(self):
        try:
            data = self.restart_from[self._name]
            self.job_id = data['job_id']
            return True
        except (AttributeError,KeyError):
            return False
    def abort(self):
        if self._state == self.ACTV:
            for job_id,step in self.active_jobs.items():
                if step == self:
                    print(self._name,'aborting job',job_id)
                    from runner.models import Process
                    Process.abort(job_id)
                    return
            print(self._name,'abort but no matching job_id')

    @classmethod
    def poll(cls):
        from runner.models import Process
        if cls.new_launch:
            Process.drive_background()
            cls.new_launch = False
        if cls.active_jobs:
            for process in Process.objects.filter(pk__in=cls.active_jobs):
                if process.status in Process.active_statuses:
                    continue
                step = cls.active_jobs[process.id]
                status_label=Process.status_vals.get('label',process.status)
                print('completed',step._name,'job',process.id,status_label)
                if process.status == process.status_vals.SUCCEEDED:
                    step.done()
                    step._workflow.checkpoint()
                else:
                    step.done('job failed')
                del cls.active_jobs[process.id]

################################################################################
# WorkStep wrappers for various plugins
################################################################################
class FvsStep(BackgroundStepBase):
    def _get_plugin_info(self):
        return 'fvs','fvs_%s_%d'%(self.flavor,self.ws.id)
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)[self.flavor]
        # now update with passed_in overrides
        settings.update(
                ws_id=self.ws.id,
                )
        # optional overrides
        for parm in ('training_set',):
            val = self.get(parm)
            if val is not None:
                settings[parm] = val
        for stepname,l in six.iteritems(self.inputs):
            for dc_code in l:
                settings['feat_'+stepname+'_'+dc_code] = True
            settings['srm_'+stepname+'_srcjob'] = self.step(stepname).job_id
            settings['srm_'+stepname+'_label'] = stepname
        return settings

class FdfStep(BackgroundStepBase):
    def _get_plugin_info(self):
        return 'fdf',None
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # now update with passed_in overrides
        settings.update(
                ws_id=self.ws.id,
                )
        stepname,dummy = self.single_input()
        src_bji = JobInfo.get_bound(self.ws,self.step(stepname).job_id)
        fm_choices = src_bji.get_feature_matrix_choices()
        assert len(fm_choices) == 1
        settings['fm_code'] = fm_choices[0][0]
        return settings

class MlStep(BackgroundStepBase):
    def _get_plugin_info(self):
        return 'ml',None
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # now update with passed_in overrides
        settings.update(
                ws_id=self.ws.id,
                )
        stepname,dummy = self.single_input()
        src_bji = JobInfo.get_bound(self.ws,self.step(stepname).job_id)
        fm_choices = src_bji.get_feature_matrix_choices()
        assert len(fm_choices) == 1
        settings['fm_code'] = fm_choices[0][0]
        return settings

class WzsStep(BackgroundStepBase):
    def _get_plugin_info(self):
        return 'wzs',None
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # supply workspace
        settings.update(ws_id=self.ws.id)
        # now update with optional overrides
        for parm in (
                'algo',
                'wtr_cutoff',
                'auto_tune',
                'auto_drug_set',
                'auto_cap',
                'auto_step',
                'auto_metric',
                ):
            try:
                v = getattr(self,parm)
                settings[parm] = v
            except AttributeError:
                pass
        stepname,dummy = self.single_input()
        src_bji = JobInfo.get_bound(self.ws,self.step(stepname).job_id)
        fm_choices = src_bji.get_feature_matrix_choices()
        assert len(fm_choices) == 1
        settings['fm_code'] = fm_choices[0][0]
        return settings

class gpbrStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='gpbr'
        return plugin,JobInfo._format_jobname_for_workspace(plugin,self.ws)
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # now update with passed_in overrides
        stepname,dummy = self.single_input()
        settings.update(
                ws_id=self.ws.id,
                pathjob=self.step(stepname).job_id,
                )
        return settings

class MISigStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='misig'
        return plugin,JobInfo._format_jobname_for_workspace(plugin,self.ws)
    def _build_settings(self):
        # build default settings
        (val,dd)=self.ws.get_disease_default('Monarch',return_detail=True)
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # now update with passed_in overrides
        settings.update(
                ws_id=self.ws.id,
                disease=val,
                )
        return settings

class mipsStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='mips'
        return plugin,JobInfo._format_jobname_for_workspace(plugin,self.ws)
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # now update with passed_in overrides
        settings.update(
                ws_id=self.ws.id,
                combo_with=self.get('combo_with'),
                combo_type=self.get('combo_type','add'),
                p2d_file=self.p2d_file,
                p2p_file=self.p2p_file,
                p2d_t=self.p2d_min,
                p2p_t=self.p2p_min,
                )
        # extend this list as needed
        for parm in ('p2p_t',):
            val = self.get(parm)
            if val is not None:
                settings[parm] = val
        return settings

class FaersStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='faers'
        return plugin,JobInfo._format_jobname_for_workspace(plugin,self.ws)
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # now update with passed_in overrides
        settings.update(
                ws_id=self.ws.id,
                drug_set=self.ws.eval_drugset,
                search_term=self.search_term,
                cds=self.cds,
                )
        return settings

class TcgaMutStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='tcgamut'
        return plugin,JobInfo._format_jobname_for_workspace(plugin,self.ws)
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # now update with passed_in overrides
        settings.update(
                ws_id=self.ws.id,
                input_file=self.input_file,
                )
        return settings

class OpenTargStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='otarg'
        return plugin,JobInfo._format_jobname_for_workspace(plugin,self.ws)
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # now update with passed_in overrides
        settings.update(
                ws_id=self.ws.id,
                disease=self.disease,
                min_prots=self.min_otarg_prots,
                incl_knowndrug=self.use_otarg_knowndrug,
                )
        return settings

class DGNStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='dgn'
        return plugin,JobInfo._format_jobname_for_workspace(plugin,self.ws)
    def _build_settings(self):
        # build default settings
        (val,dd)=self.ws.get_disease_default('DisGeNet',return_detail=True)
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # now update with passed_in overrides
        settings.update(
                ws_id=self.ws.id,
                disease=val,
                )
        return settings

class AGRStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='agr'
        return plugin,JobInfo._format_jobname_for_workspace(plugin,self.ws)
    def _build_settings(self):
        # build default settings
        (val,dd)=self.ws.get_disease_default('AGR',return_detail=True)
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # now update with passed_in overrides
        settings.update(
                ws_id=self.ws.id,
                disease=val,
                )
        return settings

class CappStep(BackgroundStepBase):
    def __init__(self, *args, **kwargs):
        self.source = kwargs.pop('source', 'dgn')
        super().__init__(*args, **kwargs)
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='capp'
        return plugin,JobInfo._format_jobname_for_workspace(plugin,self.ws)
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # now update with passed_in overrides
        stepname,dummy = self.single_input()
        settings.update(
                ws_id=self.ws.id,
                faers_run=self.step(stepname).job_id,
                p2d_file=self.p2d_file,
                p2p_file=self.p2p_file,
                p2d_t=self.p2d_min,
                p2p_t=self.p2p_min,
                combo_with=self.get('combo_with'),
                combo_type=self.get('combo_type','add'),
                )
        
        if self.source == 'dgn':
            settings.update(
                use_opentargets=False,
                use_disgenet=True,
            )
        elif self.source == 'otarg':
            settings.update(
                use_opentargets=True,
                use_disgenet=False,
            )
        else:
            assert False, f'Unknown source type {self.source}'
        return settings

class FaersSigStep(BackgroundStepBase):
    def _get_plugin_info(self):
        return 'faerssig', None
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # now update with passed_in overrides
        stepname,dummy = self.single_input()
        settings.update(
            ws_id=self.ws.id,
            capp_job=self.step(stepname).job_id,
            )
        return settings

class PathStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='path'
        return plugin,JobInfo._format_jobname_for_tissue_set(plugin,self.ts)
    def _build_settings(self):
        from algorithms.run_path import get_tissue_settings_keys
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ts.ws)[self.ts.name]
        # now update with passed_in overrides
        settings.update(
                ws_id=self.ws.id,
                combo_with=self.get('combo_with'),
                combo_type=self.get('combo_type','add'),
                p2d_file=self.p2d_file,
                p2p_file=self.p2p_file,
                p2d_t=self.p2d_min,
                p2p_t=self.p2p_min,
                )
        # extend this list as needed
        for parm in ('p2p_t',):
            val = self.get(parm)
            if val is not None:
                settings[parm] = val
        for parts in self.thresh_overrides.items():
            if not isinstance(parts[0], int):
                # We're also passing non-tissue IDs through here now.  Anything non-integer
                # just apply as a setting directly, as we do in gpath.
                settings[parts[0]] = parts[1]
                continue

            # each item is: t_id, ev_thresh (, optional_fc_thresh);
            # 'None' can be passed for ev_thresh or fc_thresh to retain
            # the default for that threshold while overriding the other;
            # (t_id,None,None) is allowed, but has no effect (both defaults
            # are retained)
            if len(parts) == 2:
                parts = list(parts) + [None]
            t_id, ev_t, fc_t = parts
            ev_key, fc_key = get_tissue_settings_keys(t_id)
            if ev_t is not None:
                settings[ev_key] = ev_t
            if fc_t is not None:
                settings[fc_key] = fc_t
        return settings

class gpathStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='gpath'
        return plugin,JobInfo._format_jobname_for_workspace(plugin,self.ws)
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # now update with passed_in overrides
        settings.update(
                ws_id=self.ws.id,
                combo_with=self.get('combo_with'),
                combo_type=self.get('combo_type','add'),
                p2d_file=self.p2d_file,
                p2p_file=self.p2p_file,
                p2d_t=self.p2d_min,
                p2p_t=self.p2p_min,
                )
        # extend this list as needed
        for parm in ('p2p_t',):
            val = self.get(parm)
            if val is not None:
                settings[parm] = val
        for gwds_id, include in self.thresh_overrides.items():
            settings[gwds_id] = include
        return settings

class EsgaStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='esga'
        return plugin,JobInfo._format_jobname_for_workspace(plugin,self.ws)
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # update with passed-in overrides
        settings.update(
                ws_id=self.ws.id,
                dpi_file=self.p2d_file,
                min_dpi=self.p2d_min,
                ppi_file=self.p2p_file,
                min_ppi=self.p2p_min,
                )
        for id,val in six.iteritems(self.thresh_overrides):
            settings[id] = val
        return settings

class CodesStep(BackgroundStepBase):
    def _get_plugin_info(self):
        return 'codes',None
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # now update with passed_in overrides
        stepname,code = self.single_input()
        settings.update(
                ws_id=self.ws.id,
                input_score='%d_%s' % (self.step(stepname).job_id,code),
                combo_with=self.get('combo_with'),
                p2d_file=self.p2d_file,
                p2d_t=self.p2d_min,
                )
        return settings

class GleeStep(BackgroundStepBase):
    def _get_plugin_info(self):
        return 'glee',None
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # now update with passed_in overrides
        stepname,code = self.single_input()
        settings.update(
                ws_id=self.ws.id,
                input_score='%d_%s' % (self.step(stepname).job_id,code),
                )
        return settings
class GlfStep(BackgroundStepBase):
    def _get_plugin_info(self):
        return 'glf',None
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # now update with passed_in overrides
        stepname,code = self.single_input()
        settings.update(
                ws_id=self.ws.id,
                input_score='%d_%s' % (self.step(stepname).job_id,code),
                )
        return settings

class DependStep(BackgroundStepBase):
    def _get_plugin_info(self):
        return 'depend',None
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # now update with passed_in overrides
        stepname,dummy = self.single_input()
        from dtk.d2ps import D2ps
        settings.update(
                ws_id=self.ws.id,
                glee_run=str(self.step(stepname).job_id),
                dpi_file=self.p2d_file,
                ppi_file=self.p2p_file,
                combo_with=self.get('combo_with'),
                score_type=D2ps.default_method,
                )
        if stepname.endswith('_glf'):
            settings.update(
                glf_run=str(self.step(stepname).job_id),
                glee_run='0'
            )
        elif stepname.endswith('_glee'):
            settings.update(
                glee_run=str(self.step(stepname).job_id),
                glf_run='0'
            )
        else:
            assert False, 'Unacceptable score type'
        return settings

class GESigStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='gesig'
        return plugin,JobInfo._format_jobname_for_tissue_set(plugin,self.ts)
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ts.ws)[self.ts.name]
        # now update with passed_in overrides
        settings.update(ws_id=self.ws.id)
        for t_id,val in six.iteritems(self.thresh_overrides):
            settings['t_%d'%t_id] = val
        return settings

class GWASigStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='gwasig'
        return plugin,JobInfo._format_jobname_for_workspace(plugin,self.ws)
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # update with passed-in overrides
        settings.update(
                ws_id=self.ws.id,
                )
        from dtk.gwas import gwas_codes
        for gwds in gwas_codes(self.ws):
            val = self.get(gwds,True)
            settings[gwds] = val
        for id,val in six.iteritems(self.thresh_overrides):
            settings[id] = val
        return settings

class MolGSigStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='molgsig'
        return plugin,JobInfo._format_jobname_for_workspace(plugin,self.ws)
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        stepname,dc_code = self.single_input()
        input_jid = self.step(stepname).job_id
        input_score = f'{input_jid}_ev'
        settings.update(
                ws_id=self.ws.id,
                input_score=input_score,
                )
        return settings

class DefusStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='defus'
        return plugin,JobInfo._format_jobname_for_workspace(plugin,self.ws)
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # update with passed-in overrides
        stepname,dc_code = self.single_input()
        input_jid = self.step(stepname).job_id
        if stepname.endswith('_molgsig'):
            settings.update(
                faers_run=None,
                molgsig_run=input_jid,
                # Set min sim to >1.0 for these, which disables them.
                # They are very slow for this large dataset.
                indigo=1e99,
                prMax=1e99,
                # This could be made to work with a higher min_sim threshold and better indexing.
                pathway=1e99,
                )
        else:
            settings.update(
                faers_run=input_jid,
                molgsig_run=None,
                )

        settings.update(
                ws_id=self.ws.id,
                p2d_file=self.p2d_file,
                p2d_t=self.p2d_min,
                p2p_file=self.p2p_file,
                p2p_t=self.p2p_min,
                )
        return settings

class SigDifStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='sigdif'
        return plugin,JobInfo._format_jobname_for_workspace(plugin,self.ws)
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # update with passed-in overrides
        stepname,dc_code = self.single_input()
        settings.update(
                ws_id=self.ws.id,
                ppi_file=self.p2p_file,
                min_ppi=self.p2p_min,
                input_score='%d_%s'%(self.step(stepname).job_id,dc_code),
                )
        return settings

class StructStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='struct'
        return plugin,JobInfo._format_jobname_for_workspace(plugin,self.ws)
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        return settings

class WsCopyStep(BackgroundStepBase):
    def _get_plugin_info(self):
        from runner.process_info import JobInfo
        plugin='wscopy'
        return plugin,JobInfo._format_jobname_for_workspace(plugin,self.ws)
    def _build_settings(self):
        # build default settings
        settings = dict(
                ws_id=self.ws.id,
                from_ws=self.from_ws,
                from_score=self.from_score,
                )
        return settings

class TargetImportanceStep(BackgroundStepBase):
    def _get_plugin_info(self):
        return 'trgscrimp',None
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # apply overrides
        if hasattr(self, 'wzs_ordering_jid'):
            self.ordering_score='wzs'
            self.ordering_jid = self.wzs_ordering_jid
        settings.update(
                ws_id=self.ws.id,
                wzs_jid=self.ordering_jid,
                score=self.ordering_score,
                count=self.flag_count,
                condensed=self.use_condensed,
                )
        return settings

class FlagStep(BackgroundStepBase):
    def _get_plugin_info(self):
        return 'flag',None
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # apply overrides
        if hasattr(self, 'wzs_ordering_jid'):
            self.ordering_score='wzs'
            self.ordering_jid = self.wzs_ordering_jid
        settings.update(
#TODO add PPI and DPI
                ws_id=self.ws.id,
                job_id=self.ordering_jid,
                score=self.ordering_score,
                count=self.flag_count,
                condensed=self.use_condensed,
                )
        return settings

class LbnStep(BackgroundStepBase):
    def _get_plugin_info(self):
        return 'lbn',None
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # apply overrides
        if hasattr(self, 'wzs_ordering_jid'):
            self.ordering_score='wzs'
            self.ordering_jid = self.wzs_ordering_jid
        settings.update(
                ws_id=self.ws.id,
                job_id=self.ordering_jid,
                score=self.ordering_score,
                count=self.flag_count,
                condensed=self.use_condensed,
                dpi_file=self.p2d_file,
                dpi_t=self.p2d_min,
                )
        # optional overrides
        for parm in ('add_drugs',):
            val = self.get(parm)
            if val is not None:
                settings[parm] = val
        return settings

class SelectabilityStep(BackgroundStepBase):
    def _get_plugin_info(self):
        return 'selectability',None
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # apply overrides
        settings.update(
                ws_id=self.ws.id,
                model_jid=self.select_model_jid,
                wzs_jid=self.wzs_ordering_jid,
                count=self.flag_count,
                condensed=self.use_condensed,
                )
        return settings

class SelectivityStep(BackgroundStepBase):
    def _get_plugin_info(self):
        return 'selectivity',None
    def _build_settings(self):
        # build default settings
        plugin,_ = self._get_plugin_info()
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound(plugin)
        settings = uji.settings_defaults(self.ws)['default']
        # apply overrides
        settings.update(
                ws_id=self.ws.id,
                dpi_file=self.p2d_file,
#TODO add PPI
                )
        return settings

class AprStep(BackgroundStepBase):
    def _get_plugin_info(self):
        return 'apr',None
    def _build_settings(self):
        return dict(
                ws_id=self.ws.id,
                jids='|'.join(x for x in self.sources),
                )

class RefreshWfStep(BackgroundStepBase):
    def _get_plugin_info(self):
        return 'wf',f'wf_{self.ws.id}_RefreshFlow'
    def _build_settings(self):
        settings = dict(self.model_settings)
        settings.update(
            user = self.user,
            ws_id=self.ws.id,
            )
        return settings

################################################################################
# tools for leave-one-out optimization of multiple input CMs
#
# To use these tools, you create a workflow derived from
# LeaveOneOutOptimizingWorkflow. You define 3 methods:
# __init__ - see details below in base class
# _add_cm_step - given a list of inputs to exclude, creates a pipeline of
#     one or more worksteps that evaluate that set of inputs, and return
#     the stepname of the final workstep in that list (from which a set of
#     drug scores can be retrieved)
# _add_cycle - given a set of previously excluded inputs, calls _add_cm_step
#     once for each remaining input (that is, it kicks off a bunch of parallel
#     processing pipelines for each potential next exclusion) and then
#     arranges to call the base class _do_compare method when the final
#     step is complete in all of those pipelines
# The base class then takes care of calculating metrics on all the returned
# scores, evaluating each of the candidate runs looking for a consistently
# worst dataset and, if one is found, invokes the next round of evaluation
# with another call to _add_cycle.
################################################################################
def get_evaluation_matrix(runs,code,metrics,ds,ws,baseline=None):
    # Return {stepname:{metric:rating,...},...}
    # Each stepname corresponds to one candidate for the next excluded input.
    # Each metric in a text label of a way of evaluating the goodness of a
    # result. Each rating is a score for that run under that evaluation method.
    # Ratings for the same metric can be compared across runs, but ratings
    # can't be compared between different metrics.
    #
    # 'runs' is {stepname:data_catalog_instance,...}
    # 'code' identifies a score within the data catalog to be evaluated
    # 'metrics' is an iterable over names of metrics to be used
    # 'ds' is the set of drug ids used to check the score (usually KTs)
    from dtk.enrichment import EnrichmentMetric,EMInput,fill_wsa_ordering
    print('baseline', baseline)
    print('evmruns', runs)
    print('evmcode', code)
    print('evmmetrics', metrics)
    print('ds', ds)
    metric_info = [
            (metric, EnrichmentMetric.lookup(metric))
            for metric in metrics
            ]
    print('metric_info', metric_info)
    result = {}
    from dtk.files import Quiet

    if baseline is not None:
        for i,bs in enumerate(baseline):
            for stepname,cat in six.iteritems(runs):
                if bs.split('_minus_')[0] in stepname:
                    ratings = {}
                    result[stepname] = ratings
                    ordering = cat.get_ordering(code[i],True)
                    ordering = fill_wsa_ordering(ordering, ws)
                    emi = EMInput(ordering,ds)
                    for metric,Type in metric_info:
                        em = Type()
                        with Quiet() as tmp:
                            em.evaluate(emi)
                        ratings[metric] = em.rating
    else:
        for stepname,cat in six.iteritems(runs):
            ratings = {}
            result[stepname] = ratings
            ordering = cat.get_ordering(code,True)
            ordering = fill_wsa_ordering(ordering, ws)
            emi = EMInput(ordering,ds)
            for metric,Type in metric_info:
                em = Type()
                with Quiet() as tmp:
                    em.evaluate(emi)
                ratings[metric] = em.rating
    print('evmresult', result)
    return result

class LeaveOneOutOptimizingWorkflow(Workflow):
    def _do_compare(self,compare_step, multiple=False):
        # input removal qualifications:
        # - preferred over baseline by at least 4 of 5 metrics
        # - preference of over 10% on at least 1 metric
        # if multiple inputs qualify, select the one with the
        # highest average rank
        #
        # 'compare_step' is assumed to point to a LocalCode instance with
        # the following properties:
        #  inputs - a dict keyed by stepname, including the stepnames
        #      to be evaluated for each candidate exclusion in this cycle,
        #      each with a value of the dataset to be excluded, and the
        #      'base' stepname with all the active datasets for this cycle,
        #      with a value of None
        #  cm_code - the Data Catalog code for the score to be retrieved
        #      from the job for each stepname in 'inputs'
        #  metrics - a list of metric names to use for evaluation (a default
        #      is set up by the base class)
        #  eval_ds - the drugset to be used for evaluation
        #
        # Get data catalog for each score source to be evaluated
        runs = {}
        from dtk.text import print_table
        from runner.process_info import JobInfo
        import numpy as np
        import re
        self.compare_step = compare_step
        for stepname in self.compare_step.inputs:
            step = self.compare_step.step(stepname)
            bji = JobInfo.get_bound(self.compare_step.ws,step.job_id)
            cat = bji.get_data_catalog()
            runs[stepname] = cat
        # construct a matrix rating each score source under several metrics
        print('compare step', self.compare_step.cm_code)
        matrix = get_evaluation_matrix(
                runs,
                self.compare_step.cm_code,
                self.compare_step.metrics,
                self.compare_step.ws.get_wsa_id_set(self.compare_step.eval_ds),
                baseline=None if type(self.compare_step.baseline) is not list else self.compare_step.baseline,
                ws=self.compare_step.ws,
                )

        from dtk.kt_split import parse_split_drugset_name
        split_data = parse_split_drugset_name(self.compare_step.eval_ds)
        if split_data.is_split_drugset:
            test_ds = split_data.complement_drugset
            validation_matrix = get_evaluation_matrix(
                    runs,
                    self.compare_step.cm_code,
                    self.compare_step.metrics,
                    self.compare_step.ws.get_wsa_id_set(test_ds),
                    baseline=None if type(self.compare_step.baseline) is not list else self.compare_step.baseline,
                    ws=self.compare_step.ws,
                    )
        else:
            validation_matrix = None

        # extract the ratings for the case where no inputs are excluded
        if type(self.compare_step.baseline) is not list:
            base_ratings = matrix[self.compare_step.baseline]
            # get a list of the metrics used in sorted order
            self.cols = sorted(base_ratings.keys())
        else:
            def get_ratings_and_matrices(matrix):
                base_ratings = []
                for bs in self.compare_step.baseline:
                    base_ratings.append(matrix[bs])
                self.cols = sorted(base_ratings[0].keys())
                matrices = []
                for i,bs in enumerate(self.compare_step.baseline):
                    matrices.append({})
                    for key in matrix.keys():
                        if bs.split('minus')[0] in key:
                            matrices[i][key] = matrix[key]
                return base_ratings, matrices
            base_ratings, matrices = get_ratings_and_matrices(matrix)
            if validation_matrix:
                validation_base_ratings, validation_matrices = \
                        get_ratings_and_matrices(validation_matrix)
        self._get_required_col_inds()
        if type(self.compare_step.baseline) is not list:
            def evaluate(matrix):
                # for each score source, contruct a list of the percentage change
                # in each rating, relative to the baseline
                cur_base_ratings = matrix[self.compare_step.baseline]
                pct_delta = get_pct_delta(matrix, self.cols, cur_base_ratings)
                # get rank data for every score source
                ranksum = self.get_average_rankings(matrix)
                # merge rank data with pct delta data
                ordered_deltas = [
                        (
                            stepname,
                            pct_delta[stepname]
                        )
                        for stepname,rank_detail in ranksum
                        ]
                # print it all out in the log file
                print()
                if True:
                    print('% deltas from baseline:')
                    table=[]
                    table.append(['','rank']+self.cols)
                    for i,tup in enumerate(ordered_deltas):
                        stepname,deltas = tup
                        table.append([
                                str(self.compare_step.inputs[stepname] or ''),
                                str(i+1),
                                ]+[str(x) for x in deltas])
                    print_table(table)
                return pct_delta, ranksum, ordered_deltas
            pct_delta, ranksum, ordered_deltas = evaluate(matrix)
            if validation_matrix:
                print("Validation/test matrix")
                evaluate(validation_matrix)
            # look for a 'clearly worst' run
            # - we're going through the list in avg_rank order, worst to best,
            #   and the list includes the baseline (no exclusion), so if we hit
            #   the baseline and haven't yet found a clearly worst candidate,
            #   give up
            # - apply separate thresholds for:
            #   A) did the majority of prefered metrics not get worse
            #   B) the number of metrics that must not be worse that baseline,
            #   and C) the number that must be at least self.min_thresh% worse than baseline.
            #   All criteria must be met to accept as 'clearly worst'
            print('Compare Step', self.compare_step.inputs)
            top_step = self._get_top_step(ordered_deltas, self.compare_step.inputs)
            # If we found a clearly worst run, add its dataset to the excluded
            # list and run another cycle. Otherwise we're done.
            if top_step:
                next_exclude = self.compare_step.inputs[top_step]
                print(next_exclude,'selected for exclusion')
                self._add_cycle(
                    excluded=self.compare_step.excluded+[next_exclude],
                    baseline=top_step,
                    )
            else:
                # re-run the selected setup, so it's the default for evaluation
                print('no next exclusion found')
                print('re-running optimum step...')
                self._add_cm_step(self.compare_step.excluded,'final')
                self.save_scoreset(self._label(),self.save_steps)
            self.checkpoint()
        else:
            def print_stats_and_get_top_step(matrices, base_ratings):
                ds_pct_delta = []
                step_checker = self._generalize_steps(set([k
                                                          for mat in matrices
                                                          for k in mat
                                                         ])
                                                     )
                input_steps = list(step_checker.keys())

                combined_pct_delta = {stepname:[0.0]*len(self.cols) for stepname in input_steps}
                for i,mat in enumerate(matrices):
                    pct_delta = get_pct_delta(mat, self.cols, base_ratings[i])
                    print('% deltas from baseline:')
                    table=[]
                    for full_k,l in six.iteritems(pct_delta):
                        k = self._extract_minus_name(full_k)
                        for j,v in enumerate(l):
                            combined_pct_delta[k][j] += v
                        table.append([str(k)]+[str(x) for x in l])
                    print('WF',i,self.compare_step.cm_code[i])
                    print_table([['']+self.cols]+sorted(table, key = lambda x: x[0]))
                    ds_pct_delta.append(pct_delta)
                cumulative = sorted(list(combined_pct_delta.items()),
                                     key = lambda x: (x[1][self.reqrd_col_inds[0]],
                                                      x[1][self.reqrd_col_inds[1]],
                                                     ),
                                     reverse = True
                                     )
                print('Cumulative % deltas from baseline:')
                table = [['']+self.cols]
                for tup in cumulative:
                    table.append([str(tup[0])]+[str(x) for x in tup[1]])
                print_table(table)
                top_step = self.meta_top_step(cumulative, step_checker, ds_pct_delta)
                return top_step
            top_step = print_stats_and_get_top_step(matrices, base_ratings)
            if validation_matrix:
                print("-----------------------")
                print("Validation/test metrics")
                print("-----------------------")
                print_stats_and_get_top_step(validation_matrices, validation_base_ratings)
                print("---------------------------")
                print("End Validation/test metrics")
                print("---------------------------")
            if top_step is not None:
                ds_to_remove = str(top_step)
                if '_' in ds_to_remove:
                    dss_to_remove = ds_to_remove.split('_')
                    print('dss to remove', dss_to_remove)
                    for entry in dss_to_remove:
                        if 'gwds' in entry and entry not in self.compare_step.excluded:
                            ds_to_remove = entry
                            print("Actual DS to remove", entry)
                            break
                        elif 'gwds' not in entry and int(entry) not in self.compare_step.excluded:
                            ds_to_remove = entry
                            print("Actual DS to remove", entry)
                            break
                next_exclude = ds_to_remove
                print(next_exclude,'selected for exclusion')
                baseline_no_under = []
                for mem in self.compare_step.baseline:
                    datasetsremoved = mem.split('_')[2:] + [next_exclude]
                    print(datasetsremoved)
                    datasetsremoved_noempty = []
                    for i in datasetsremoved:
                        if i != '' and i != 'minus':
                            datasetsremoved_noempty.append(i)
                    base_mem_list = mem.split('minus')[0]
                    if base_mem_list[-1] == '_':
                        base_mem_list= base_mem_list[:-1]
                    print('datasetsremoved_noempty',datasetsremoved_noempty)
                    print('_'.join([base_mem_list, 'minus'] + datasetsremoved_noempty))
                    print('base join splt', mem.split('minus'))
                    print('base join', '_'.join([base_mem_list, 'minus']))
                    print('base join list', [base_mem_list, 'minus'] + datasetsremoved_noempty)
                    baseline_no_under.append('_'.join([base_mem_list, 'minus'] + datasetsremoved_noempty))
                print('baseline_no_under', baseline_no_under)
                if 'gwds' in next_exclude:
                    self._add_cycle(
                        excluded=self.compare_step.excluded+[next_exclude],
                        baseline=baseline_no_under,
                        )
                else:
                    self._add_cycle(
                        excluded=self.compare_step.excluded+[int(next_exclude)],
                        baseline=baseline_no_under,
                        )
            else:
                # re-run the selected setup, so it's the default for evaluation
                print('no next exclusion found')
                print('re-running optimum step...')
                self._add_cm_step(self.compare_step.excluded,'final')
                self.save_scoreset(self._label(),self.save_steps)
            self.checkpoint()
    def _get_required_col_inds(self):
        self.reqrd_col_inds = [
                self.cols.index(x)
                for x in self.preferred_metrics
                ]
    def get_average_rankings(self, matrix):
    # Return [(stepname,avg_rank,[metric_1_rank,metric_2_rank,...]),...]
    # The list is ordered by rank, of prefered metrics, descending (so, the first item
    # in the list is the overall least attractive according to our most preferred metric)
    # Input is a matrix like the one returned by get_evaluation_matrix()
        from dtk.scores import Ranker
        from dtk.num import avg
        rankers = []
        example = next(iter(matrix.values()))
        for metric in example:
            step_ratings = [
                    (stepname,ratings[metric])
                    for stepname,ratings in six.iteritems(matrix)
                    ]
            step_ratings.sort(
                    key=lambda x:x[1],
                    reverse=True,
                    )
            rankers.append(Ranker(step_ratings))
        ranksum = []
        for stepname in matrix:
            rankings = [
                    ranker.get(stepname)
                    for ranker in rankers
                    ]
            ranksum.append( (stepname,rankings) )
        ranksum = sorted(ranksum, key = lambda x: (x[1][self.reqrd_col_inds[0]],
                                         x[1][self.reqrd_col_inds[1]],
                                        ))
        return ranksum
    def _extract_minus_name(self, k, None_fill_in = 'baseline'):
        x = self.compare_step.inputs[k]
        if x is not None:
            return x
        return None_fill_in
    def _generalize_steps(self,ks):
        d = {self._extract_minus_name(k):True
              for k in ks
            }
        nk = [k for k,v in six.iteritems(self.compare_step.inputs) if v is None][0]
        d[self._extract_minus_name(nk)] = False
        return d
    def _evaluate_row(self,deltas):
        if not self._check_mandatory_metrics(deltas):
            print('failed _check_mandatory_metrics')
            return False
        not_behind = sum([v >= 0 for v in deltas])
        over = sum([v > self.min_thresh for v in deltas])
        if not_behind < self.min_not_behind:
            print('failed min_not_behind test')
            return False
        if over < self.min_over_thresh:
            print('failed over test')
            return False
        return True
    def _check_mandatory_metrics(self, deltas):
        negs = 0
        for i in self.reqrd_col_inds:
            if deltas[i] < 0:
                negs += 1
# require that all of the preferred metrics did not get worse
        if negs:
            print(' '.join([
                        'Of',
                        str(len(self.reqrd_col_inds)),
                        'preferred metrics',
                        str(negs),
                        'got worse.'
                       ]))
            return False
        return True
    def _get_top_step(self, ordered_deltas, compare_step_inputs, steps_to_ignore = None):
        if steps_to_ignore is None:
            steps_to_ignore = []
        top_step = None
        for stepname,deltas in ordered_deltas:
            if not compare_step_inputs[stepname]:
                break # give up if we hit baseline
            if stepname in steps_to_ignore:
                continue
            print('evaluating', stepname)
            if not self._evaluate_row(deltas):
                continue # disqualify anything not over thresholds
            # first qualifying run wins
            top_step = stepname
            break
        return top_step
    def meta_top_step(self, cumulative, inputs, ds_pct_delta, steps_tried=None):
        if steps_tried is None:
            steps_tried = []
        top_step = self._get_top_step(cumulative, inputs, steps_tried)
        # now check that that step satisifies all the criteria for each table individually
        if top_step:
            print('verifying:', top_step)
            for pct_delta in ds_pct_delta:
                row = None
                for k in pct_delta:
                    if self._extract_minus_name(k) == top_step:
                        row = pct_delta[k]
                        break
                assert row is not None
                if not self._check_mandatory_metrics(row):
                    print(k, 'failed _check_mandatory_metrics')
                    steps_tried.append(top_step)
                    return self.meta_top_step(cumulative,
                                              inputs,
                                              ds_pct_delta,
                                              steps_tried
                                             )
        return top_step
    def __init__(self,**kwargs):
        super(LeaveOneOutOptimizingWorkflow,self).__init__(**kwargs)
        preferred_metrics = [
                 'AUR',
                 'SigmaOfRank1000',
                 ]
        metrics=preferred_metrics +[
                 'SigmaOfRank',
                 'wFEBE',
                 'FEBE',
                 'APS',
                 'DEA_ES',
                 'DEA_AREA',
                 ]
        self._pl.add_defaults(
                metrics=metrics,
                preferred_metrics=preferred_metrics,
                min_not_behind=5,
                min_over_thresh=1,
                min_thresh=10,
                excluded=[],
                )
        self.save_steps=[]
        self.cycle_count=0
        # TODO: derived class __init__ must do the following, after
        # - invoking base class init above
        # - adding any more needed defaults
        #   - note that no default is provided for eval_ds; this is because
        #     the default is in the Workspace object, which isn't known here
        # - setting self.ws
        #
        #baseline = self._add_cm_step(self.excluded)
        #self._add_cycle(self.excluded,baseline)
    # TODO: derived class must define the following methods:
    #def _add_cm_step(self,excluded,stepname=None):
    #   -calculate a unique stepname if none is passed in (based on excluded)
    #   -construct a WorkStep object to run the underlying CM
    #   -return the stepname
    #def _add_cycle(self,excluded,baseline):
    #   -call _add_cm_step for each remaining input to be excluded
    #   -construct a LocalCode WorkStep to call _do_compare when all cm steps
    #    complete

def get_pct_delta(matrix, cols, base_ratings):
    return{
           stepname:[
                   int(100 *
                           (ratings[metric]-base_ratings[metric])
                           /(base_ratings[metric] or 0.1)
                           )
                   for metric in cols
                   ]
           for stepname,ratings in six.iteritems(matrix)
           }
