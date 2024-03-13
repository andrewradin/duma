#!/usr/bin/env python3

import sys
from path_helper import PathHelper,make_directory

import os
import django
import django_setup

import logging
logger = logging.getLogger("algorithms.run_wf")

import json
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo

class MyJobInfo(JobInfo):
    def _get_form_class(self,job_name):
        ws_id, wf_name = self._parse_jobname(job_name)
        from dtk.workflow import Workflow
        wrapper = Workflow.get_by_code(wf_name)
        from dtk.dynaform import FormFactory,FieldType
        ff = FormFactory()
        for code in wrapper.cls._fields:
            ft = FieldType.get_by_code(code)
            ft.add_fallback_context(ws_id=ws_id,wf_name=wf_name)
            ft.add_to_form(ff)
        return ff.get_form_class()
    def _form_to_html(self,form,job_name):
        from django.utils.html import format_html
        return format_html(u'''
                <table>{}</table>
                ''',
                form.as_table(),
                )
    def get_config_html(self,ws,job_name,copy_job,sources=None):
        FormClass = self._get_form_class(job_name)
        form = FormClass()
        if copy_job:
            s = copy_job.settings()
            for fld in form:
                if fld.name in s:
                    fld.field.initial = s[fld.name]
        return self._form_to_html(form,job_name)
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        FormClass = self._get_form_class(jobname)
        form = FormClass(post_data)
        if not form.is_valid():
            return (self._form_to_html(form,jobname),None)
        p = dict(form.cleaned_data)
        p['user'] = user.username
        p['ws_id'] = ws.id
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(p)
                            )
        next_url = ws.reverse('nav_progress',job_id)
        return (None,next_url)

    def get_scoreset(self):
        from browse.models import ScoreSet
        from runner.models import Process
        try:
            return ScoreSet.objects.get(ws=self.ws, wf_job=self.job.id)
        except ScoreSet.DoesNotExist:
            settings = self.job.settings()
            # If we did a resume, we won't have a scoreset pointing directly
            # at us, but we'll have the ID in our settings, go find it.
            ss_id = settings.get('resume_scoreset_id', None)
            if ss_id:
                return ScoreSet.objects.get(pk=ss_id)
            else:
                return None

    def get_scoreset_weights(self):
        # Find a wzs job and delegate to that.
        ss = self.get_scoreset()
        name_to_jid = ss.job_type_to_id_map()
        if 'wzs' not in name_to_jid:
            return {}
        wzs_jid = name_to_jid['wzs']
        wzs_bji = JobInfo.get_bound(self.ws, wzs_jid)
        weights = wzs_bji.get_score_weights()

        # Typically weight_name will be something like:
        #   gwasig_sigdif_glf_depend_PSSCORE_MAX
        # and job_name will be e.g.:
        #   gwasig_sigdif_glf
        #
        # It is entirely possible for input jobs to have multiple
        # weights if they have multiple final outputs.
        output = {}
        for job_name, jid in name_to_jid.items():
            job_weights = []
            for weight_name, weight in weights:
                if job_name in weight_name:
                    job_weights.append(weight)

            output[jid] = job_weights
        return output

    def get_warnings(self):
        base_list = super(MyJobInfo,self).get_warnings()
        ss = self.get_scoreset()
        if not ss:
            return base_list + ['Unable to locate scoreset for workflow']
        from dtk.warnings import get_scoreset_warning_summary
        return base_list + get_scoreset_warning_summary(self.ws,ss)

    def __init__(self,ws=None,job=None):
        # base class init
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "Workflow",
                "Launch Workflow",
                )
        # any base class overrides for unbound instances go here
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.fn_checkpoint = self.outdir+'checkpoint.json'
            timeline_url = self.ws.reverse('nav_workflow_timeline', self.job.id)
            self.otherlinks = [
                ('Workflow Runtimes',  timeline_url),
            ]
    def custom_report_html(self):
        ss = self.get_scoreset()
        if not ss:
            return ''
        
        out = ['<div style="max-height:300px;overflow: auto; display: inline-block;border: 1px solid #aaa; padding: 0.5rem; margin-top: 1rem;">']
        from dtk.html import link, glyph_icon
        from runner.models import Process
        from browse.models import WorkflowJob

        jobtype_to_id = ss.job_type_to_id_map()
        all_ids = set(jobtype_to_id.values())

        wf_job_ids = WorkflowJob.objects.filter(wf_job=self.job.id).values_list('child_job', flat=True)
        # Filter out anything that'll show up in the scoreset.
        wf_job_ids = [x for x in wf_job_ids if x not in all_ids]
        all_ids.update(wf_job_ids)

        p_info = dict(Process.objects.filter(pk__in=all_ids).values_list('id', 'status'))
        p_name = dict(Process.objects.filter(pk__in=all_ids).values_list('id', 'role'))


        if wf_job_ids:
            out += ['<h4>Running/Failed Jobs</h4>']
        for id in sorted(wf_job_ids, reverse=True):
            job_link = link(
                            glyph_icon('link'),
                            self.ws.reverse('nav_progress',id),
                            )
            status = Process.status_vals.get('label', p_info[id])
            jtype = p_name[id]
            out.append(f'<li>{jtype} ({id}) {status} {job_link}')
                
        
        out += ['<h4>Completed Jobs</h4>']
        # Sort so that newest at the top.
        for jtype, id in sorted(jobtype_to_id.items(), key=lambda x: -x[1]):
            job_link = link(
                            glyph_icon('link'),
                            self.ws.reverse('nav_progress',id),
                            )
            status = Process.status_vals.get('label', p_info[id])
            out.append(f'<li>{jtype} ({id}) {status} {job_link}')
        
        out += ['</div>']
        
        return ''.join(out)


        
    def get_jobnames(self,ws):
        '''Return a list of all jobnames for this plugin in a workspace.'''
        from dtk.workflow import Workflow
        return [
                "%s_%d_%s" % (self.job_type,ws.id,wf.code())
                for wf in Workflow.wf_list()
                ]
    def settings_defaults(self,ws):
        from runner.process_info import form2dict
        jobnames = self.get_jobnames(ws)
        out = {}
        for job_name in jobnames:
            ConfigForm = self._get_form_class(job_name)
            cfg=ConfigForm()
            out[job_name] = form2dict(cfg)
        return out
        
    def _parse_jobname(self,jobname):
        fields = jobname.split('_')
        return (int(fields[1]),fields[2])
    def source_label(self,jobname):
        ws_id, wf_name = self._parse_jobname(jobname)
        return wf_name
    def abort_handler(self):
        if hasattr(self,'wf'):
            # propagate abort through workflow; this will cause the workflow
            # to wind down quickly, and the run() method below to return
            self.wf.abort()
        else:
            # else, if we can't find the workflow, exit immediately (as if
            # we hadn't caught the signal)
            self.warning('in abort handler w/o workflow')
            import sys
            sys.exit(1)
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        # Even though this doesn't request any resources, if we don't register
        # the job with the RM, then it gets counted as one of the pre-start
        # jobs. This means 10 workflows can lock up the job queue. This will
        # get to be more of an issue when we start having workflows run other
        # workflows.
        # XXX since this is required for all plugins, should it be pushed down
        # XXX into the JobInfo base class?
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[0])
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "run workflow",
                "cleanup",
                ])
        ws_id, wf_name = self._parse_jobname(self.job.name)
        from dtk.workflow import Workflow
        wrapper = Workflow.get_by_code(wf_name)
        self.wf = wrapper.cls(wrapper_job_id=self.job.id,**self.job.settings())
        self.wf._checkpoint_filename = self.fn_checkpoint
        p_wr.put("wait for resources","complete")
        self.wf.run_to_completion()
        if not self.wf.succeeded():
            print("Workflow failed, dumping state")
            self.wf.dump()
            raise RuntimeError('workflow failed')

        p_wr.put("run workflow","complete")
        p_wr.put("cleanup","complete")

if __name__ == "__main__":
    MyJobInfo.execute(logger)
