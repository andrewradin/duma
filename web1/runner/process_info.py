import re
import os
import django_setup
from runner.models import Process
from collections import OrderedDict
from browse.models import Tissue,TissueSet,Workspace
from path_helper import PathHelper,make_directory
from algorithms.exit_codes import status_file_line
from dtk.files import path_exists_cmp
from django import forms

import logging
logger = logging.getLogger(__name__)

class JobInfo(object):
    '''
    Base class for plugin background algorithms.

    These are never instantiated directly.  Instead, each algorithm type
    defines a derived class that holds all algorithm information needed
    by other parts of the system.  This derived class may also implement
    the top level of the algorithm (so the shared configuration is
    guaranteed to match), but this isn't required.

    These objects are instantiated through one of 3 class methods:
    - get_unbound() returns an instance that isn't associated with a
      specific job, but may be queried for general information about
      the plugin.
    - get_bound() returns an instance that may be queried for information
      about a specific job
    - execute() returns an instance that is used within a job's process
      while it is running

    This class also holds shared helper functions for common operations,
    both within the algorithm itself, and within views and templates
    presenting algorithm information.

    The JobCrossChecker class still handles inter-job dependencies, as
    that seems much clearer than trying to build them from individual
    plugin configurations. There is also a 'level_names' list at the
    top of JobCrossChecker that needs to be updated if a new type of
    plugin is added -- it defines how jobs are ordered on the Workflow
    page, among other things.
    '''
    directory = {}
    @classmethod
    def get_unbound(cls,name):
        # 'name' can be the plugin name, or a job name, or some other
        # job subtype designator; force to a plugin name here
        name = name.split('_')[0]
        # now return the unbound plugin
        if name not in cls.directory:
            logger.debug('loading plugin for %s',name)
            modname = 'run_%s' % name
            mod = __import__('algorithms.'+modname)
            mod = getattr(mod,modname)
            info = mod.MyJobInfo()
            cls.directory[name] = (mod,info)
            # we don't need to stash the name in the info object
            # because the ctor re-creates it (as job_type)
        return cls.directory[name][1]
    @classmethod
    def get_all_bound(cls,ws_or_id,jobs_or_ids):
        if type(ws_or_id) == Workspace:
            ws = ws_or_id
        else:
            ws = Workspace.objects.get(pk=ws_or_id)
        # Pull out just the IDs
        ids = [int(x) for x in jobs_or_ids if type(x) != Process]
        # Fetch them all into a dict {id:Process}
        jobs = Process.objects.in_bulk(ids)
        # Replace any IDs in our list with the correspond process.
        procs = [x if type(x) == Process else jobs[int(x)] for x in jobs_or_ids]
        out = []
        for job in procs:
            name = job.job_type()
            cls.get_unbound(name) # make sure plugin is loaded
            mod = cls.directory[name][0]
            out.append(mod.MyJobInfo(ws,job))
        return out
    @classmethod
    def get_bound(cls,ws_or_id,job_or_id):
        if type(ws_or_id) == Workspace:
            ws = ws_or_id
        elif ws_or_id is not None:
            ws = Workspace.objects.get(pk=ws_or_id)
        else:
            ws = None
        # Some jobs support loading with a ws of None.
        # Could also allow others to dynamically support loading from config.

        if type(job_or_id) == Process:
            job = job_or_id
        else:
            job = Process.objects.get(pk=job_or_id)
        name = job.job_type()
        cls.get_unbound(name) # make sure plugin is loaded
        mod = cls.directory[name][0]
        return mod.MyJobInfo(ws,job)
    @classmethod
    def get_plugin_names(cls):
        name_pattern=r'run_(.*)\.py$'
        import re
        from dtk.files import scan_dir,name_match
        return [
                re.match(name_pattern,x).group(1)
                for x in scan_dir(
                        PathHelper.website_root+'algorithms',
                        filters=[name_match(name_pattern)],
                        output=lambda x:x.filename,
                        )
                if x != 'run_example.py'
                ]
    @classmethod
    def get_my_job(cls):
        # This method only functions inside background jobs launched
        # by the runner system.  It will return the job row assigned
        # to that job (allowing it to retrieve parameters, etc.)
        from runner.run_process import Operation
        job_id = int(os.environ[Operation.job_id_name])
        return Process.objects.get(pk=job_id)
    def __repr__(self):
        if not self.job:
            return '<JobInfo %s unbound>' % self.job_type
        return '<JobInfo %s job %d ws %d>' % (
                    self.job_type,
                    self.job.id,
                    self.ws.id,
                    )
    def __init__(self,ws,job,src,short_label,page_label):
        if not hasattr(self,'use_LTS'):
            self.use_LTS=False
        if ws is None and job:
            # If you don't provide a workspace, but do provide a job, let's see if
            # we can imply a workspace from the job.
            try:
                from browse.models import Workspace
                ws = Workspace.objects.get(pk=job.settings()['ws_id'])
            except KeyError:
                # No ws setting; some jobs don't care, some will crash later.
                pass
        self.ws = ws
        self.job = job
        import re
        m = re.match(r'.*run_(.*).pyc?$',src)
        self.job_type = m.group(1)
        self.short_label = short_label
        self.page_label = page_label
        self.in_run_menu = True
        self.needs_sources = False
        self.log_prefix = ''
        self.logger = None
        self.publinks = []
        self.otherlinks = []
        self.qc_plot_files = []
        self._data_catalog = None
        if ws:
            self.parms = self.job.settings()
            self.root = PathHelper.storage+'%d/%s/%d/' % (
                                                    self.ws.id,
                                                    self.job.job_type(),
                                                    self.job.id
                                                    )
            self.indir = self.root+'input/'
            self.outdir = self.root+'output/'
            self.tmp_pubdir = self.outdir+'publish/'
            import os
            from runner.common import LogRepoInfo
            lri = LogRepoInfo(self.job.id)
            self.progress = lri.progress_path()
            if self.use_LTS:
                # set LTS-related names and paths; rather than keeping a local
                # lts_repo instance, we rely on the LtsRepo class-level cache
                # so that any role-based keys that expire over the course of
                # a long-running job get automatically renewed.
                self._lts_repo_name = str(self.ws.id)
                self._lts_branch = PathHelper.cfg('lts_branch')
                self.lts_rel_root = os.path.join(
                            self.job.job_type(),
                            str(self.job.id),
                            )
                lts_repo = self.get_lts_repo()
                self.lts_abs_root = os.path.join(
                            lts_repo.path(),
                            self.lts_rel_root,
                            )+'/'
                self.final_pubdir = self.lts_abs_root + 'publish/'
            else:
                self.pubdir_container = PathHelper.ws_publish(self.ws.id) \
                                        + self.job.job_type() + '/'
                self.final_pubdir = self.pubdir_container+str(self.job.id)+'/'
            from aws_op import Machine
            self.mch = Machine.name_index[PathHelper.cfg('worker_machine_name')]
    # derived classes can call the following if they need an LtsRepo object
    def get_lts_repo(self):
        from dtk.lts import LtsRepo
        return LtsRepo.get(self._lts_repo_name,self._lts_branch)
    # The following 2 methods should be overridden by the plugin to allow
    # the process to be launched from the generic job_start page.  The first
    # returns the HTML to render the config form.  The second returns a
    # 2-element tuple:
    # - if the form didn't validate, the HTML to re-render the form for
    #   correction is returned in the first element, and the second element
    #   is None
    # - if the form was OK, the first element is None, and the second
    #   element is a URL to track progress of the (already queued) job
    def get_config_html(self,ws,job_name,copy_job,sources=None):
        from django.utils.html import format_html
        return format_html('<h4>{}</h4>{}'
                    ,"Start Disabled"
                    ,"get_config_html not overridden"
                    )
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        return (self.get_config_html(ws,jobname,None),None)
    # a common shared jobname format
    @staticmethod
    def _format_jobname_for_workspace(plugin,ws):
        return "%s_%d" % (plugin,ws.id)
    # The following method should be overridden if the plugin has
    # any jobnames other than <plugin_name>_<ws_id>. If this function
    # returns multiple job names, source_label should also be overridden
    # to supply a unique label for each name, and if the name affects
    # the settings, settings_defaults should return multiple options
    # as well.
    def get_jobnames(self,ws):
        '''Return a list of all jobnames for this plugin in a workspace.'''
        return [self._format_jobname_for_workspace(self.job_type,ws)]
    # another common shared jobname format; private to derived plugin
    # classes (which can expose it via a get_jobname_for_tissue_set
    # method if they choose)
    @staticmethod
    def _format_jobname_for_tissue_set(plugin,ts):
        return "%s_%d_%d" % (plugin,ts.id,ts.ws_id)
    # a hook to allow a plugin to supply descriptive text for the
    # workflow diagram
    def description_text(self):
        return ''
    def description(self):
        return '<b>%s</b><br><br>%s'%(self.page_label,self.description_text())
    # retrieve warnings from log; this reports situations that don't cause
    # the job to terminate abnormally, but may require human review before
    # accepting results; can be overridden to look for additional job-specific
    # patterns, or to supply more context; the override may just call the base
    # class with custom patterns or ignore conditions. For example:
    # def get_warnings(self):
    #     return super(MyJobInfo,self).get_warnings(
    #             ignore_conditions=self.base_warning_ignore_conditions+[
    #                     lambda x:'use warnings() to see' in x,
    #                     ],
    #             )

    base_warning_ignore_conditions = [
            lambda x:'plotly/tools.py:154' in x,
            # We used to print this out before switching it to info, but this helps ignore in older jobs.
            lambda x:'Using fake_mp in pmap' in x,
            # Something has been uploaded to S3 that this branch doesn't know about yet.
            lambda x:' unknown role ' in x,
            ]
    base_warning_patterns = [
            'warning',
            'exception',
            ]
    def get_warnings(self,patterns=None,ignore_conditions=None):
        if not self.job:
            return []
        if patterns is None:
            patterns = self.base_warning_patterns
        if ignore_conditions is None:
            ignore_conditions = self.base_warning_ignore_conditions
        from dtk.files import open_pipeline
        fh = open_pipeline([[
                'egrep',
                '-n',
                '-i',
                '|'.join(patterns),
                self.job.logfile(fetch=True),
                ]])
        with fh as src:
            return ['See log line '+x.rstrip()
                    for x in src
                    if not any(test(x) for test in ignore_conditions)
                    ]


    report_stat_prefix = "___REPORT___:"
    @classmethod
    def report_info(cls, info):
        """Adds <info> to the job report, which is displayed on the progress page.

        Use this to display informational data about the job that the user can see without
        wading through the log file.
        """
        print(cls.report_stat_prefix, info)
    
    def get_reported_info(self):
        """Retrieves a list of reported data for a job."""
        if not self.job:
            return []
        from dtk.files import open_pipeline
        fh = open_pipeline([[
                'grep',
                self.report_stat_prefix,
                self.job.logfile(),
                ]])
        def clean(x):
            return x[x.index(self.report_stat_prefix) + len(self.report_stat_prefix):]
        with fh as src:
            return [clean(x) for x in src]

    # Can be overridden to allow custom buttons to be added to a job's
    # start page.  Sample output:
    # [{'name': 'reset_btn', 'action': lambda: self.reset()}]
    #
    # The name should correspond to the button name in the form, and the
    # action is invoked as the 'post_valid' handler when submitted.
    def get_buttons(self):
        return []
    # The following can be overridden so that a bound JobInfo can return
    # the ids of any previous jobs it relied on for input.
    # XXX All the aggregation and 'shared tools' CMs have been retrofit to
    # XXX return their input scores. This could be further expanded to
    # XXX retrieve non-efficacy jobs (e.g. FAERS for capp and defus, sig
    # XXX for path and gesig)
    def get_input_job_ids(self):
        return set()
    # This should not need to be overridden. It relies on get_input_job_ids
    # to recursively search back through all inputs.
    def get_all_input_job_ids(self):
        result = set()
        batch = {int(x) for x in self.get_input_job_ids()}
        while batch:
            additions = batch - result
            result |= batch
            batch = set()
            bjis = JobInfo.get_all_bound(self.ws,additions)
            for bji in bjis:
                batch |= {int(x) for x in bji.get_input_job_ids()}
        return result

    def incorporates_combo_base(self):
        """Returns whether this takes into account combo base drugs.

        If so, subsequent jobs that depend on this might not want to try to
        additionally incorporate combo base drugs on top.
        """
        input_jids = self.get_input_job_ids()
        if input_jids:
            bjis = JobInfo.get_all_bound(self.ws,input_jids)
            for bji in bjis:
                if bji.incorporates_combo_base():
                    return True
        return False
    # The following method is called on an unbound jobinfo to return a
    # tuple containing two sets of descriptions of jobs (either job names
    # or roles) that cause the passed-in job to be out-of-date. The
    # first set holds completed jobs; the second set holds jobs in progress.
    # The uji must be of the same type as the job itself. The default
    # implementation never reports a job to be out-of-date with respect
    # to other jobs (although it may be reported as NotLatest if there's
    # a newer job with the same jobname and role).
    def out_of_date_info(self,job,jcc):
        return (set(),set())
    # a canned implementation of out_of_date_info that may be called by
    # a derived class if job out-of-dateness can be determined solely
    # by name (i.e. there are no roles within a name)
    def _out_of_date_from_names(self,job,nameset):
        result = (set(),set())
        qs = Process.objects.filter(name__in=nameset,pk__gt=job.id)
        def filtered_names(**kwargs):
            return set(qs.filter(**kwargs).values_list('name',flat=True))
        active = filtered_names(status__in=Process.active_statuses)
        newer = filtered_names(status=Process.status_vals.SUCCEEDED)
        return newer,active
    # a variation on the above where the names are all the sig jobs
    # in a tissue set
    def _out_of_date_from_tissue_set(self,job,ts_id):
        nameset = set([
                t.get_sig_jobname()
                for t in Tissue.objects.filter(tissue_set_id=ts_id)
                ])
        return self._out_of_date_from_names(job,nameset)
    # a canned implementation of out_of_date_info that may be called by
    # a derived class to determine out-of-dateness based on a set of job
    # ids that consititute the actual input to this job (i.e. if there is
    # a newer version of any of those input jobs, this job is out of date)
    def _out_of_date_from_ids(self,job,id_set,jcc):
        # get name and role for each job id
        l = Process.objects.filter(
                id__in=id_set,
                ).values_list('id','role','name')
        # retrieve latest job in each role
        newer = set()
        pending = set()
        for job_id,role,name in l:
            p = jcc.role_info_for_name(name)[role]
            if p.id > job_id:
                if p.status in Process.active_statuses:
                    pending.add(role)
                else:
                    newer.add(role)
        return (newer,pending)
    # For the most part, CM jobs that depend on the output of other
    # jobs can't be scheduled until those jobs are complete, because
    # only completed jobs appear in their input forms (or for jobs in
    # workflows, the workflow handles ordering). But the following
    # two methods allow some name-oriented ordering, for jobs where
    # outputs are stored by jobname rather than job id.
    #
    # The following method is called on an unbound jobinfo to return a
    # list of jobnames that can't be running at the same time as the
    # input jobname.  These will normally be 'upstream' jobnames, but
    # the list not only prevents this job from starting while any of
    # the specified jobs are running; it also causes any of those jobs
    # to get queued behind this one as long as it's active.  The rationale
    # is that the returned jobs have reusable output buffers, and those
    # buffers must not be touched while the specified job is running.
    # Note that jobs which have reusable output buffers should return
    # their own names here (or in blocking_jobnames below) along with
    # the names of any jobs whose output would be used as input.
    def ordering_jobnames(self,jobname,jcc=None):
        return set()
    # The following is a more extreme version of the above, which prevents
    # a job from being queued at all if any of the returned jobnames are
    # active
    def blocking_jobnames(self,jobname,jcc=None):
        return set()
    # If there are one or more standard settings that might serve as
    # a comparison point in the job settings list, return a dict where
    # the key is a string defining the standard (e.g. 'default'), and
    # the value is a dict of the settings values
    def settings_defaults(self,ws):
        return {}
    # The following can be overridden to select the best DEA value to use
    # for score comparison, if multiple dea runs are produced per score.
    # The default algorithm is to return the first matching code_.*wsBack.*;
    # failing that, the first one matching code_.*; failing that, ''.
    def get_best_dea_name(self,code,er):
        keep = []
        for name in er.get_names():
            if name.startswith(code+'_'):
                keep.append(name)
                if 'wsBack' in name:
                    return name
        if keep:
            return keep[0]
        return ''
    # The following can be overridden to select the drugset(s) to use when
    # evaluating this job, if there is something better than ws default.
    def eval_drugsets(self):
        return [('', self.ws.eval_drugset)]
    # For some plugins, there are multiple possible jobnames that
    # correspond to different roles, and return independent data.
    # Those plugins should override this method to return a source
    # label corresponding to a jobname.  For jobs with a single role,
    # this fallback implementation returns the job short label.
    def source_label(self,jobname):
        return self.short_label
    # Build a role code based on job settings; by default this is just the
    # plugin name.
    def build_role_code(self,jobname,settings_json):
        return self.job_type
    # service routine to return a role code based on a tissue set
    def _tissue_set_role_code(self,ts_id):
        from dtk.files import safe_name
        ts = TissueSet.objects.get(pk=ts_id)
        name = safe_name(ts.name).lower()
        name_map = {
                'default':'cc',
                'case_control':'cc',
                'treatment_response':'tr',
                }
        return '_'.join([name_map.get(name,name),self.job_type])
    # service routine to return a role code based on an upstream job role
    multi_uniprot_scores = False # override in derived classes
    def _upstream_role_code(self,job_id,code=None):
        from runner.models import Process
        p = Process.objects.get(pk=job_id)
        l = p.role.split('_')
        uji = self.get_unbound(p.name)
        if uji.multi_uniprot_scores:
            assert code
            # For sources with multiple potential outputs, downstream roles
            # need to indicate which output they're processing, which is
            # done by adding a prefix to the role (eg. literature_otarg_codes
            # rather than just otarg_codes). This was originally only for
            # otarg, but the otarg code for building workflow steps got
            # incorrectly copied to agr, dgn, and misig, and it was easier
            # to fix the roles to match than to back out all the incorrect
            # workflow data products. Also, this might legitimately be
            # used for another source someday.
            l = [code]+l
        l.append(self.job_type)
        return '_'.join(l)
    def role_label(self):
        return self.source_label(self.job.name)
    def _upstream_role_label(self):
        return(self.role2label(self.job.role))
    @classmethod
    def role2label(cls,role):
        parts = list(reversed(role.split('_')))
        labels = []
        code_map = {
                # common tissue sets
                'cc':'Case/Control',
                'tr':'Treatment/Response',
                # meaningless prefixes to ignore
                'agrs':None,
                'dgns':None,
                # for misig, the prefix and role name are the same,
                # so handle it differently; the rightmost misig
                # breaks out of the loop, and then either one or
                # two misigs map to the same label
                'misig':'MISig',
                'misig_misig':'MISig',
                }
        # put all otarg codes in the map
        uji = cls.get_unbound('otarg')
        cg = uji.get_data_code_groups()[0]
        code_map.update({
                cp._code:cp.label()
                for cp in cg.get_code_proxies()
                })
        while parts:
            if parts[0] in list(code_map.keys()):
                break
            try:
                uji = cls.get_unbound(parts[0])
            except ImportError:
                break
            labels.append(uji.short_label)
            parts = parts[1:]
        remainder = '_'.join(parts)
        if remainder:
            remainder_label = code_map.get(remainder,remainder)
            if remainder_label:
                labels.append(remainder_label)
        return ' '.join(reversed(labels))
    # The following should be overridden to allow a job to report custom
    # progress information (for example, percentage complete on a long
    # job step)
    def get_progress(self):
        progress = self._get_progress_parts()
        return progress[0]+progress[1]
    # This utility function will do most of the work for an override
    # implementation of get_progress()
    def _get_progress_parts(self):
        if self.job.status == self.job.status_vals.QUEUED:
            return (tuple(),(('In Queue','Waiting for prerequisites...'),))
        from tools import ProgressReader
        p_rd = ProgressReader(self.progress)
        progress = p_rd.get()
        if progress[1]:
            if self.job.active():
                progress[1][0][1] = "in progress..."
            else:
                from django.utils.safestring import mark_safe
                progress[1][0][1] = mark_safe(
                    '<span class="text-danger">TERMINATED</span>'
                    )
        return progress
    def _refresh_db_connection(self):
        # During long CM runs, the sql database connection can time out.
        # An attempt to access the database after this happens throws an
        # error that will cause the CM to fail. A long-running CM that
        # needs to access the database at the end of its run can call
        # this beforehand to force a new connection.
        # See https://code.djangoproject.com/ticket/21597#comment:29
        from django.db import connection
        connection.close()
    # The following method should be overridden in every plugin that
    # publishes data.  It will eventually replace lots of other stuff
    # in here.
    def get_data_code_groups(self):
        return []
    # A convenience function for Progress page rendering
    def get_subtitle(self):
        if not self.job:
            return None
        label = self.source_label(self.job.name)
        if label != self.page_label and label != self.short_label:
            return label
        return None
    # should this plugin go into the source list when loading defaults?
    def is_a_source(self):
        # XXX This seems a little broken. Many of the CMs that output
        # XXX uniprot scores seem to force this to True. Since there's
        # XXX a single source list underlying all score keys, maybe
        # XXX this should just be True?
        cat = self.get_data_catalog()
        return any(cat.get_codes('wsa','score'))
    # Another convenience function returning score codes and labels
    # in 'choices' format
    def get_score_choices(self,dtc='wsa'):
        cat = self.get_data_catalog()
        return [
                ('%d_%s'%(self.job.id,code), cat.get_label(code))
                for code in cat.get_codes(dtc,'score')
                ]
    def qc_plots(self):
        # unlike output_links below, this doesn't supply a div number,
        # because a single QC page will hold plots from more than one
        # JobInfo, so div numbers need to be assigned centrally
        # XXX probably there should be a single unique div generator
        # XXX object that gets used in both the QC and progress views,
        # XXX and none of that should happen here
        result=[]
        from dtk.plot import PlotlyPlot
        for fn in self.qc_plot_files:
            path = self.final_pubdir+fn
            if path_exists_cmp(path):
                result.append(PlotlyPlot.build_from_file(path,thumbnail=True))
        return result
    # The following method shouldn't need to be overridden; the derived
    # class bound instance init code just needs to set the publinks member
    def output_links(self):
        result = []
        from dtk.plot import PlotlyPlot
        div_count = 0
        # Originally, publinks contained a linkname and filename, which
        # this function converted to a linkname and url.  That functionality
        # still exists, but has been extended for plotly support in 2 ways:
        # - if the linkname is empty, the filename is used to create a
        #   PlotlyPlot object, which the template will render as an inline
        #   png thumbnail, with a clickthrough to an interactive plot
        # - if a plotly file exists which differs only in extension from the
        #   supplied filename, the above mode is also triggered.  This
        #   simplfies backwards compatibility, where new jobs have plotly
        #   images, but old jobs require file links.
        for linkname,filename in self.publinks:
            path = self.final_pubdir+filename
            if False:
                # LTS directories can contain plotly files without png
                # files. When these get pulled to a dev environment, the
                # missing png causes the code below not to run, and no
                # plots to appear. Enabling this block works around that.
                # XXX There's probably a better answer to this.
                if not os.path.exists(path):
                    plotly_path = os.path.splitext(path)[0]+'.plotly'
                    if os.path.exists(plotly_path):
                        path=plotly_path
                        linkname=None
            if path_exists_cmp(path):
                if linkname:
                    plotly_path = os.path.splitext(path)[0]+'.plotly'
                    if path_exists_cmp(plotly_path):
                        path=plotly_path
                        do_plotly = True
                    else:
                        result.append( (PathHelper.url_of_file(path),linkname) )
                        do_plotly = False
                else:
                    do_plotly = True # no link means plotly mode
                if do_plotly:
                    pc = PlotlyPlot.build_from_file(path,thumbnail=True)
                    div_count += 1
                    result.append( (pc, "plotly_div_%d"%div_count) )
        for linkname,url in self.otherlinks:
            result.append( (url,linkname) )
        return result
    # provides access to all plugin data products
    def get_data_catalog(self):
        if self._data_catalog is None:
            import runner.data_catalog as dc
            cat = dc.Catalog()
            self.fetch_lts_data()
            try:
                for cg in self.get_data_code_groups():
                    cat.add_group('',cg)
            except Exception:
                logger.exception('Got exception extracting score')
                raise
            self._data_catalog = cat
        return self._data_catalog
    # Remove any ws-dependent scaling from a score. This base class method
    # does nothing, but if a CM returns scores with a value bias that varies
    # from workspace to workspace, they should override this method with
    # one that attempts to correct the bias. 'ordering' and the return value
    # are lists in the format returned by the data catalog get_ordering method,
    # and 'code' identifies which CM score this is (in case the scaling to
    # be removed is different for different score products).
    def remove_workspace_scaling(self,code,ordering):
        return ordering
    # helper function for defining a 'standard fetcher', where the fetcher
    # parameter is a file path; returns the value of the specified attribute
    # on a bound instance, or None on an unbound instance (fetchers should
    # never be invoked on unbound instances)
    # The real meat of this implementation is in data_catalog.CodeGroup,
    # in the constructor.
    def _std_fetcher(self,attr_name):
        if self.job:
            return getattr(self,attr_name)
        return None

    def dpi_codegroup_type(self, dpi_parmname):
        if not self.job or dpi_parmname not in self.parms:
            codetype = 'wsa'
        else:
            from dtk.prot_map import DpiMapping
            dpi_name = self.parms[dpi_parmname]
            dpi = DpiMapping(dpi_name)
            codetype = dpi.mapping_type()
        return codetype

    # make sure any LTS data is available on the local machine
    def fetch_lts_data(self):
        if self.use_LTS and self.job:
            self.get_lts_repo().lts_fetch(self.lts_rel_root)
        if self.job:
            from runner.common import LogRepoInfo
            LogRepoInfo(self.job.id).fetch_log()
    # return a list of (code,label) feature matrix choices available
    # from this job; the base class returns an empty list; it can be
    # overridden by derived classes that output feature matrices.
    def get_feature_matrix_choices(self):
        return []
    # extract a job id from a feature matrix code; this defines a
    # common convention for feature matrix codes that allows going
    # from a code to a bji and then to a feature matrix
    @classmethod
    def extract_feature_matrix_job_id(cls,code):
        import re
        m = re.match(r'([a-z]+)([0-9]+)$',code)
        if m:
            prefix = m.group(1)
            job_id = int(m.group(2))
            return job_id
        raise NotImplementedError('unsupported code: %s'%code)
    # get a feature matrix object; base class returns None;
    # derived classes returning feature matrices must override
    def get_feature_matrix(self,code):
        return None
    # service routine to return a default feature matrix code and label;
    # this can be used by derived classes as part of the implementation
    # of their custom get_feature_matrix_choices() method
    def _get_default_feature_matrix_choice(self):
        return (
                '%s%d' % (self.job_type,self.job.id),
                '%s job %d' % (
                            self.source_label(self.job.name),
                            self.job.id,
                            ),
                )

    def get_dpi_choice(self):
        """DPI used for this CM.

        Defaults to checking the two most common names, can also override.
        """
        if 'p2d_file' in self.parms:
            return self.parms['p2d_file']
        if 'dpi_file' in self.parms:
            return self.parms['dpi_file']
        return None

    # a helper function for retrieving pathsum detail information; this
    # will return None if a CM doesn't have pathsum detail information
    # (which is signalled by the derived class setting a pathsum_detail
    # data member)
    # Any CM setting pathsum_detail should also set pathsum_detail_label
    # to label what disease characteristic the DPI is matching against
    # (e.g. GE tissue, GWAS phenotype, Co-morbidity).
    def get_target_set(self):
        try:
            fn = self.pathsum_detail
        except AttributeError:
            return None
        self.fetch_lts_data()
        from algorithms.pathsum2 import TargetSet
        ts = TargetSet()
        ts.load_from_file(fn)
        return ts
    def has_target_detail(self):
        return hasattr(self, 'pathsum_detail')
    def get_target_detail(self,wsa):
        # This throws IOError to match the old way of doing things;
        # some code uses this to distinguish failure conditions
        ts = self.get_target_set()
        if ts is None:
            return (None,None)
        key = self.get_target_key(wsa)
        if not key:
            return (None,None)
        target = ts.get_target(key)
        return (ts,target)
    class AmbiguousKeyError(Exception):pass
    def get_target_keys(self,wsa):
        from dtk.prot_map import DpiMapping
        dm = DpiMapping(self.job.settings()['p2d_file'])
        return set(dm.get_dpi_keys(wsa.agent))
    def get_target_key(self,wsa):
        keys = self.get_target_keys(wsa)
        if len(keys) != 1:
            raise self.AmbiguousKeyError(
                    'ambiguous path key: '+' '.join(keys)
                    )
        return keys.pop()
    def get_dea_path(self):
        return self.final_pubdir
    def get_legacy_dea_path(self):
        if self.job_type == 'ml':
            return PathHelper.ml_history(self.ws.id)+str(self.job.id)
        if self.job_type == 'path':
            return PathHelper.pathsum_history(self.ws.id)+str(self.job.id)
        return self.get_dea_path()
    def get_available_deas(self):
        from dea import FileHandleFactory,LegacyFactory,EnrichmentResult
        fhf = FileHandleFactory()
        fhf.dirpath = self.get_dea_path()
        codes = fhf.scandir()
        if not codes:
            fhf = LegacyFactory(self.get_legacy_dea_path(),fhf)
            codes = fhf.scandir()
        codes.sort()
        result = []
        for code in codes:
            result.append(EnrichmentResult(self,'_'.join(code)))
        return result
    def get_label_of_code(self,code):
        cat = self.get_data_catalog()
        return cat.get_label(code)
    ##
    # Unbound instance methods
    ##
    # default command; can be overridden, but probably shouldn't
    # (just pass needed info in settings_json in job record)
    def cmd(self):
        return 'algorithms/run_%s.py'%self.job_type
    def rundir(self):
        return PathHelper.website_root
    ##
    # enabled_default helper methods (called on uji)
    ##
    def disease_name_set(self,ws,vocab):
        '''Return True is a disease name is explicitly set.'''
        name,detail = ws.get_disease_default(vocab,return_detail=True)
        return (name and detail)
    def data_status_ok(self,ws,source_type,score_label):
        '''Return True if there's adequate data.'''
        from dtk.ws_data_status import DataStatus
        ds = DataStatus.lookup(source_type)(ws)
        try:
            # note this returns False if the score is 'nan'
            return ds.scores()[score_label] >= 1
        except KeyError:
            return False
    ##
    # background job utility methods
    ##
    def set_logger(self,logger):
        self.logger = logger
    def _do_logging(self,level,msg,*args):
        fmt = self.log_prefix+msg
        import sys
        if sys.stdout == sys.stderr:
            # assume we're in a background process
            print((fmt % args).encode('utf8'))
        if self.logger:
            log_func = getattr(self.logger,level)
            log_func(fmt,*args)
    def debug(self,msg,*args):
        self._do_logging('debug',msg,*args)
    def info(self,msg,*args):
        self._do_logging('info',msg,*args)
    def warning(self,msg,*args):
        self._do_logging('warning',msg,*args)
    def error(self,msg,*args):
        self._do_logging('error',msg,*args)
    def fatal(self,msg,*args):
        self._do_logging('error','FATAL:'+msg,*args)
        raise Exception()
    def copy_input_to_remote(self):
        self.divider('starting copy to')
        import datetime
        start = datetime.datetime.now()
        rem_indir = self.mch.get_remote_path(self.indir).rstrip('/')
        self.mch.copy_to(self.indir,rem_indir)
        end = datetime.datetime.now()
        self.divider('completed copy to in '+str(end-start))

        self.mch.check_rsync_web1()

    def copy_output_from_remote(self):
        self.divider('starting copy from')
        import datetime
        start = datetime.datetime.now()
        rem_outdir = self.mch.get_remote_path(self.outdir)
        self.mch.copy_from(rem_outdir,self.outdir.rstrip('/'))
        end = datetime.datetime.now()
        self.divider('completed copy from in '+str(end-start))
    def divider(self,text):
        border='*' * 80
        print(border)
        print(border[0],text)
        print(border)
    def make_remote_directories(self,dirlist):
        if self.use_LTS:
            if any([d.startswith(self.lts_abs_root) for d in dirlist]):
                raise ValueError('direct LTS output on worker not supported')
        dirlist = [
                self.mch.get_remote_path(d)
                for d in dirlist
                ]
        self.mch.check_remote_cmd('mkdir -p '+' '.join(dirlist))
    def finalize(self):
        if os.path.exists(self.tmp_pubdir):
            if not self.use_LTS:
                make_directory(self.pubdir_container)
            import shutil
            shutil.move(self.tmp_pubdir,self.final_pubdir)
        if self.use_LTS:
            self.get_lts_repo().lts_push(self.lts_rel_root)
    def check_enrichment(self,dea_options=None):
        logger.info("check_enrichment no longer does anything")
    def enrichment_metrics(self):
        """Override this to change the metrics for a CM."""
        return ('AURCondensed', )
    def enrichment_metric_summaries(self):
        from dtk.enrichment import MetricProcessor
        mp = MetricProcessor()
        summaries = []
        codes = list(self.get_data_catalog().get_codes('wsa','score'))
        for metric in self.enrichment_metrics():
            for code in codes:
                val = mp.compute(metric, self.ws, self.job, code)
                line = f'{code} {metric}: <b>{val:.3f}</b>'
                summaries.append(line)
        return summaries


    # If the derived class provides an abort_handler() method, then if
    # a SIGHUP arrives while the job is executing, it will be caught
    # and the abort_handler() method called. Otherwise, the SIGHUP will
    # not be caught and the job will die immediately. (SIGHUP gets sent
    # when the Abort button is pressed on the web interface.)
    def _abort_wrapper(self,signum,frame):
        self.abort_handler()
    @classmethod
    def execute(cls, logger, ws_id=None):
        """Runs this job instance, usually invoked as a new process."""
        assert cls != JobInfo, "Run this on a subclass or bound job"
        import sys
        import json

        # Python3 changed the rules, stderr gets buffered if we are
        # being redirected to a file.  But they also added more control
        # so we can explicitly control the buffering.
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

        from browse.models import Workspace
        my_job = cls.get_my_job()
        ws_id = ws_id or json.loads(my_job.settings_json)['ws_id']
        info = cls(Workspace.objects.get(pk=ws_id), my_job)
        info.set_logger(logger)
        if hasattr(info,'abort_handler'):
            import signal
            signal.signal(signal.SIGHUP,info._abort_wrapper)
        sys.exit(info.run())

# provides boilerplate for implementing ConfigForm.as_dict() methods;
# it can be called directly in cases where there's no special processing
# in the final copy loop, or copied and used as a model
def form2dict(form,**kwargs):
    result = kwargs
    if form.is_bound:
        src = form.cleaned_data
    else:
        src = {fld.name:fld.field.initial for fld in form}
    for fld in form:
        key = fld.name
        result[key] = src[key]
    return result

# utility function to rebuild all role fields in the process table;
# use for initial population, or to re-populate after changes to
# build_role_code implementations
def refresh_roles():
    from runner.models import Process
    # By processing rows in ascending job_id order, we can count on the
    # role for any input jobs being available as each job is processed.
    qs = Process.objects.filter(status=2).order_by('id')
    print('scanning',qs.count(),'job records')
    from collections import Counter
    fail_ctr = Counter()
    seen = 0
    for p in qs:
        seen += 1
        if seen%1000 == 0:
            print(seen,'records processed')
        try:
            info = JobInfo.get_unbound(p.job_type())
        except ImportError:
            fail_ctr[p.job_type()] += 1
            continue
        try:
            role = info.build_role_code(p.name,p.settings_json)
        except Exception as ex:
            fail_ctr[p.job_type()+'_'+ex.__class__.__name__] += 1
            role = ''
        if p.role != role:
            p.role = role
            p.save()
    print(seen,'records processed')
    if fail_ctr:
        print('Failures:',fail_ctr)

class JobCrossChecker:
    active_labels = [
        Process.status_vals.get('label',x)
            for x in Process.active_statuses
        ]
    level_names = "meta sig aesearch path gesig esga gpath gwasig mips misig dgn sigdif gpbr codes glee glf depend tcgamut otarg jacsim prsim struct faers capp defus synergy rankdel fvs fdf ml wzs flag lbn tsp wscopy wf trgscrimp selectivity dnprecompute selectabilitymodel selectability faerssig agr customsig gesearchmodel compositesig apr molgsig ctretro".split()
    def __init__(self):
        self._latest_jobs = None
        self._ood = {}
        self._job_ws = None
        self._ws_job_map = {}
        self._role_info = {}
    def latest_jobs(self):
        '''Return a dict of the most recent run for each jobname.'''
        # XXX this might eventually go away; it primarily supports the
        # XXX tissue page and some job ordering stuff that should probably
        # XXX all be done differently anyway
        if self._latest_jobs is None:
            self._latest_jobs = { x.name: x for x in Process.latest_jobs_qs() }
        return self._latest_jobs
    def role_info_for_name(self,jobname):
        # return {role_code:job,...}
        # XXX if performance is an issue, we could add a method to pre-load
        # XXX multiple jobnames in a single query
        if jobname not in self._role_info:
            # find latest running or successful job id for each role
            statuses = Process.active_statuses+[Process.status_vals.SUCCEEDED]
            proc_qs = Process.objects.filter(name=jobname,status__in=statuses)
            from django.db.models import Max
            d = dict(proc_qs.values_list('role').annotate(Max('id')))
            # replace job ids with job instances
            d2 = {x.id:x for x in Process.objects.filter(id__in=list(d.values()))}
            for x in d.keys():
                d[x] = d2[d[x]]
            self._role_info[jobname] = d
        return self._role_info[jobname]
    def extended_status(self,job):
        class ExtendedStatus:
            label = 'NeverRun'
            newer = set()
            pending = set()
        result = ExtendedStatus()
        if not job:
            return result
        if job.invalidated:
            # The 'invalidated' flag lets us mark a job as needing to be
            # redone for reasons outside the normal prereq system (for
            # example, if some critical configuration or software version
            # changed, or an output file was lost).  The initial use case
            # is toggling the meta method.
            result.label = "ReDo"
            return result
        enum = Process.status_vals
        result.label = enum.get('label',job.status)
        # add state-specific enhancements
        uji = JobInfo.get_unbound(job.name)
        if job.status == enum.SUCCEEDED:
            role_info = self.role_info_for_name(job.name)
            if role_info[job.role].id != job.id:
                result.label = "NotLatest"
            else:
                result.newer,result.pending = uji.out_of_date_info(job,self)
                if result.pending:
                    result.label = "Pending"
                elif result.newer:
                    result.label = "OutOfDate"
                else:
                    result.label = "UpToDate"
        elif job.status in (enum.FAILED,enum.FAILED_PREREQ):
            # If any inputs have been successfully re-run, this job may be
            # worth re-trying
            newer,pending = uji.out_of_date_info(job,self)
            if newer:
                result.label = "Retry?"
        return result
    def check_job_status(self,curjob,job_id):
        # return options:
        # (job,0) - rendering current job; check if in progress
        # (job,job_id) - rendering historical job; no launch controls
        # (0,0) - no previous job; launch controls only
        if curjob and curjob.id == int(job_id):
            job_id = 0 # treat history load of latest job as current
        if job_id:
            job = Process.objects.get(pk=job_id)
        else:
            job = curjob
        return (job,job_id)

    def job_ws_obj(self,name,ws=None):
        if self._job_ws is None:
            self._job_ws={}
            if ws is None:
                ws_list = Workspace.objects.all()
            else:
                ws_list = [ws]
            for ws in ws_list:
                for jobname in self.ws_jobnames(ws):
                    self._job_ws.setdefault(jobname,set()).add(ws)
        return self._job_ws.get(name,set())
    def job_ws(self,name):
        return {ws.name for ws in self.job_ws_obj(name)}
    def _ws_job_structure(self,ws):
        # returns {job_type:[ jobname,...],...}
        if ws.id not in self._ws_job_map:
            out = OrderedDict()
            for level in self.level_names:
                info = JobInfo.get_unbound(level)
                out[level] = info.get_jobnames(ws)
            self._ws_job_map[ws.id] = out
        return self._ws_job_map[ws.id]
    def ws_jobnames(self,ws):
        '''Return every jobname in a workspace.
        '''
        out = set()
        for l in self._ws_job_structure(ws).values():
            out |= set(l)
        return out
    def ordered_ws_jobnames(self,ws):
        '''Return every jobname in a workspace. ordered by level
        '''
        seen = set()
        out = []
        for l in self._ws_job_structure(ws).values():
            for name in l:
                if name not in seen:
                    out.append(name)
                    seen.add(name)
        return out
    def ws_summary(self,ws):
        '''Return status of every jobname in a workspace.
        '''
        # get latest job ids for each unique name/role combo in ws
        qs = Process.objects.filter(name__in=self.ws_jobnames(ws))
        qs = qs.exclude(role='').values_list('name','role')
        from django.db.models import Max
        qs = qs.annotate(Max('id'))
        latest_ids = [x[2] for x in qs]
        # get the jobs corresponding to those ids
        latest_jobs = {}
        for p in Process.objects.filter(pk__in=latest_ids):
            l=latest_jobs.setdefault(p.name,[])
            l.append(p)
        # build a display structure
        out = OrderedDict()
        for level,l in self._ws_job_structure(ws).items():
            level_jobs = OrderedDict()
            for name in l:
                for job in latest_jobs.get(name,[]):
                    name = job.name+' '+job.role
                    level_jobs[name] = self.extended_status(job).label
            out[level] = level_jobs
        return out
    def queue_job(self,job_type,jobname,**kwargs):
        info = JobInfo.get_unbound(job_type)
        kwargs['role'] = info.build_role_code(
                    jobname,
                    kwargs.get('settings_json',''),
                    )
        # guard against legacy 'job_spec' calls
        assert isinstance(jobname,str)
        latest_jobs = self.latest_jobs()
        blocking = info.blocking_jobnames(jobname,jcc=self)
        for n in blocking:
            if n in latest_jobs:
                prev = latest_jobs[n]
                if prev.status in Process.active_statuses:
                    logger.debug("job %s already scheduled",n)
                    return
        logger.debug("queuing job %s",jobname)
        ordering = info.ordering_jobnames(jobname,jcc=self)
        run_after = []
        for n in ordering:
            if n in latest_jobs:
                j = latest_jobs[n]
                if j.status in Process.active_statuses:
                    run_after.append(j.id)
        kwargs['run_after'] = run_after
        kwargs['ordering_names'] = ordering
        self._latest_jobs = None # force reload
        return Process.queue_process(jobname
                    ,info.rundir()
                    ,info.cmd()
                    ,**kwargs
                    )

class StdJobInfo(JobInfo):
    """Standard job info baseclass.
    
    JobInfo is super flexible, this class makes some standard/common decisions,
    currently tailored towards non-tissue-based CMs with 0 or 1 input jobs.

    See run_faerssig.py as a example of how to use this class.
    """
    force_local=False
    def __init__(self, ws, job, src, use_LTS=True):
        self.use_LTS=use_LTS
        super().__init__(
                    ws,
                    job,
                    src,
                    self.short_label,
                    self.page_label,
                    )

        self.needs_sources = False
        if self.job:
            from algorithms.exit_codes import ExitCoder
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.ec = ExitCoder()

    def description_text(self):
        return self.descr
    def settings_defaults(self,ws):
        if self.needs_sources:
            from dtk.scores import SourceList
            self.sources = SourceList(ws)
        cfg=self.make_job_form(ws, None)
        return {
                'default':cfg.as_dict(),
                }
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        self.sources = sources
        form = self.make_job_form(ws, None)
        if copy_job:
            # XXX The original from_json approach to loading past settings
            # XXX was replaced in some later CMs by constructing the form
            # XXX using the 'initial' parameter of the django form, i.e.:
            # XXX   form = ConfigForm(initial=copy_job.settings())
            # XXX This is more standard django, but doesn't provide a hook
            # XXX for those (few) cases where settings need to be re-written
            # XXX before loading them into the form. Review this before
            # XXX converting too many more CMs.
            form.from_json(copy_job.settings_json)
        return form.as_html()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        self.sources = sources
        import json
        form = self.make_job_form(ws, post_data)
        if not form.is_valid():
            if form.errors:
                logger.warning(f"Form errors: '{form.errors}'")
            return (form.as_html(),None)
        settings = form.as_dict()
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(settings)
                            )
        next_url = ws.reverse('nav_progress',job_id)
        return (None, next_url)
    def role_label(self):
        return self._upstream_role_label()
    
    def single_upstream_jid(self, settings_json=None):
        upstream_jid_func = getattr(self, 'upstream_jid', None)
        if upstream_jid_func:
            if settings_json is None:
                settings_json = self.job.settings_json
            import json
            d = json.loads(settings_json)
            return upstream_jid_func(d)
        else:
            return None, None
    
    def build_role_code(self, jobname, settings_json):
        """Generate's a _-separated code if an upstream, otherwise just job type."""
        jid, code = self.single_upstream_jid(settings_json)
        if jid:
            return self._upstream_role_code(jid, code)
        else:
            return self.job_type
    
    def get_input_job_ids(self):
        jid, code = self.single_upstream_jid()
        if jid:
            # not all CMs correctly interpret legacy settings, with the
            # result that jid may not be an integer; just ignore these
            # cases so we can easily roll up a mix of valid and invalid
            # jids
            try:
                return set([int(jid)])
            except ValueError:
                pass
        return set()

    def out_of_date_info(self,job,jcc):
        jid, _ = self.single_upstream_jid(job.settings_json)
        if jid:
            return self._out_of_date_from_ids(job,[jid],jcc)
        else:
            return super().out_of_date_info(job, jcc)

    def make_std_dirs(self):
        from path_helper import make_directory
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        if self.use_LTS:
            make_directory(self.lts_abs_root)
    
    def reserve_step(self, wanted_or_func, slow=False):
        def func():
            if callable(wanted_or_func):
                wanted = wanted_or_func()
            else:
                wanted = wanted_or_func
            
            self.reserved = self.rm.wait_for_resources(self.job.id,wanted,slow)
            return f"Requested {wanted}, got {self.reserved}"
        return func

    def run_steps(self, steps):
        headings = [step[0] for step in steps]
        from tools import ProgressWriter
        from reserve import ResourceManager
        p_wr = ProgressWriter(self.progress, headings)
        self.rm = ResourceManager()
        for name, step_func in steps:
            from datetime import datetime
            start = datetime.now()
            details = step_func()
            duration = datetime.now() - start
            from dtk.text import fmt_timedelta
            duration = fmt_timedelta(duration)
            status = f'Complete ({duration})'
            if details:
                status += f' - {details}'
            p_wr.put(name, status)
    
    def _run_local(self, cmd):
        """Can be overridden in tests"""
        import subprocess
        subprocess.check_call(cmd)

    def run_remote_cmd(self, cmd, options, local=False):
        local = local or self.force_local
        from path_helper import PathHelper
        if local:
            cvt = lambda x: x
        else:
            cvt = self.mch.get_remote_path

        from pathlib import Path

        str_opts = []
        # Convert all paths to strings (with conversion if running remote).
        for opt in options:
            if isinstance(opt, Path):
                str_opts.append(cvt(str(opt)))
            else:
                str_opts.append(opt)

        cmd_path = cvt(os.path.join(PathHelper.website_root, cmd))

        full_cmd = [cmd_path] + str_opts

        if local:
            self._run_local(full_cmd)
            return

        self.copy_input_to_remote()
        self.make_remote_directories([
                                self.tmp_pubdir,
                                ])
        cmd_str = ' '.join(full_cmd)
        self.mch.check_remote_cmd(cmd_str)
        self.copy_output_from_remote()

class StdJobForm(forms.Form):
    def __init__(self, ws, data):
        self.ws = ws
        super().__init__(data)

    def as_html(self):
        from django.utils.html import format_html
        return format_html('''
                <table>{}</table>
                '''
                ,self.as_table()
                )
    def as_dict(self):
        # this returns the settings_json for this form; it may differ
        # from what appears in the user interface; there are 2 use
        # cases:
        # - extracting default initial values from an unbound form
        #   for rendering the settings comparison column
        # - extracting user-supplied values from a bound form
        if self.is_bound:
            src = self.cleaned_data
        else:
            src = {fld.name:fld.field.initial for fld in self}
        p ={'ws_id': self.ws.id}
        for f in self:
            key = f.name
            value = src[key]
            p[key] = value
        return p
    def from_json(self,init):
        import json
        p = json.loads(init)
        for f in self:
            if f.name in p:
                f.field.initial = p[f.name]
