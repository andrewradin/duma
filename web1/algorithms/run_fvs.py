#!/usr/bin/env python3

from __future__ import print_function
import sys
import six
# retrieve the path where this program is located, and
# use it to construct the path of the website_root directory
try:
    from path_helper import PathHelper,make_directory
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")
    from path_helper import PathHelper,make_directory

import os
import django
if 'django.core' not in sys.modules:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    django.setup()

import subprocess
import json
from tools import ProgressWriter
from runner.process_info import JobInfo
import runner.data_catalog as dc
from reserve import ResourceManager

from django import forms

import logging
logger = logging.getLogger("algorithms.run_ml")

def make_feature_code(src_code,dc_code):
    return 'feat_'+src_code+'_'+dc_code
def is_feature_code(code):
    return code.startswith('feat_')
def extract_src_from_feature_code(code):
    return '_'.join(code.split('_')[1:-1])

class ConfigForm(forms.Form):
    # Note that all the fields below must have 'initial' specified
    # even if it's overridden, because the initial param defines
    # the type conversion for JSON generation for unbound instances
    training_set = forms.ChoiceField(
                    label='Mark rows for',
                    initial='',
                    choices=(('','None'),),
                    )
    plug_unknowns = forms.BooleanField(
                    label='Supply defaults for missing values',
                    required=False,
                    initial=False,
                    )
    hf_labels = forms.BooleanField(
                    label='Use human-friendly column labels',
                    required=False,
                    initial=True,
                    )
    def __init__(self, ws, srm, flavor, copy_job, *args, **kwargs):
        super(ConfigForm,self).__init__(*args, **kwargs)
        from runner.models import Process
        self.ws = ws
        self.srm = srm
        import runner.data_catalog as dc
        for src_code,src in srm.sources():
            enabled = set(src.enabled_codes())
            # XXX we could either back out the SourceListWithEnables change,
            # XXX or hard-code a 'manual' filter type that selects the
            # XXX enabled scores instead of relying on a data catalog flag
            # for each job, gather all potential feature sources,
            # and then make an enable field for each one; the
            # field names are grouped under the job to simplify
            # later html generation
            cat = src.bji().get_data_catalog()
            for dc_code in cat.get_codes(flavor.key,'feature'):
                full_code = make_feature_code(src_code,dc_code)
                label = src.label()+' '+cat.get_label(dc_code)
                checked = not cat.is_type(dc_code,'meta_out')
                if checked and flavor.filt:
                    checked = cat.is_type(dc_code,flavor.filt)
                self.fields[full_code] = forms.BooleanField(
                                            label=cat.get_label(dc_code,False),
                                            required=False,
                                            initial=checked,
                                            )
        flavor.customize_form(self)
        if copy_job:
            for f in self:
                if f.name in copy_job:
                    f.field.initial = copy_job[f.name]
                elif is_feature_code(f.name):
                    # unused features are excluded from settings
                    f.field.initial = False
    def as_html(self):
        # split names into groups
        features_by_group = {}
        others = []
        for name in self.fields:
            if is_feature_code(name):
                src_code = extract_src_from_feature_code(name)
                features_by_group.setdefault(src_code,[]).append(name)
            else:
                others.append(name)
        # build config HTML
        from django.utils.html import format_html,format_html_join
        from django.utils.safestring import mark_safe
        group_html = []
        # format 1st group (ML-specific rather than source-specific)
        rep = format_html_join(
                '',
                '<tr><td>{}:</td><td>&nbsp;&nbsp;</td><td>{}</td></tr>',
                [
                    (self.fields[name].label,mark_safe(self[name]))
                    for name in others
                ],
                )
        group_html.append(format_html(
                        '<h4>{}:</h4><table>{}</table>',
                        "General parameters",
                        rep,
                        ))
        # format parameters specific to each job
        from dtk.html import pad
        for src_code,src in self.srm.sources():
            if src_code not in features_by_group:
                continue
            rep = format_html_join(
                    pad(),
                    '<span style="white-space:nowrap">{}: {}</span>',
                    [
                        (self.fields[name].label, mark_safe(self[name]))
                        for name in features_by_group[src_code]
                    ],
                    )
            group_html.append(format_html(
                            '<h4>Features from {}:</h4>{}',
                            src.label(),
                            rep,
                            ))
        # format errors, if any
        error_list = []
        for k,v in six.iteritems(self.errors):
            label = '' if k == '__all__' else k+': '
            for msg in v:
                error_list.append( (label+msg,) )
        error_html = format_html_join(
                    '',
                    '<span class="text-danger">{}</span><br>',
                    error_list,
                    )
        # put it all together and return it
        return format_html_join(
                '',
                '{}',
                [(x,) for x in [error_html]+group_html],
                )
    def as_dict(self):
        import runner.data_catalog as dc
        if self.is_bound:
            src = self.cleaned_data
        else:
            src = {fld.name:fld.field.initial for fld in self}
        p = {
                'ws_id':self.ws.id,
                }
        used_srcs = set()
        for k,v in six.iteritems(src):
            if is_feature_code(k):
                # job field:
                # - settings set to 'True' if present
                # - also track contributing job, so we can record
                #   meta-data at the end
                if v:
                    used_srcs.add(extract_src_from_feature_code(k))
                    p[k] = True
            else:
                # non-job field -- just copy across
                p[k] = v
        # now add job metadata
        for src_code in used_srcs:
            p.update(self.srm.metadata(src_code))
        return p

class Flavor(object):
    flavors=[]
    def __init__(self,key,filt,label):
        self.key = key
        self.filt = filt
        self.label = label
        self.flavors.append(self)
    def code(self):
        if self.filt:
            return '_'.join([self.key,self.filt])
        return self.key
    @classmethod
    def get_by_code(cls,code):
        for flavor in cls.flavors:
            if code == flavor.code():
                return flavor
        raise ValueError("No flavor '%s'"%code)
    def customize_form(self,form):
        f = form.fields['training_set']
        if self.key == 'wsa':
            f.choices = form.ws.get_wsa_id_set_choices()
        elif self.key == 'uniprot':
            f.choices = form.ws.get_uniprot_set_choices()
            if not f.choices:
                # This effectively locks out the run button, because
                # the field is required, so '' is not an acceptable option
                f.choices = [('','(No Protein Sets in Workspace)')]
        else:
            raise NotImplementedError('unknown key type')
        f.initial = f.choices[0][0]
        if self.key == 'wsa':
            from browse.default_settings import EvalDrugset
            f.initial = EvalDrugset.value(form.ws)
            form.fields['exclude_hidden'] = forms.BooleanField(
                    label='Exclude hidden drugs',
                    required=False,
                    initial=True,
                    )
    def get_key_whitelist(self,bji):
        if self.key == 'wsa':
            # saved scores may include wsa_ids that are no longer in the
            # workspace, so set up a whitelist filter; if we're excluding
            # hidden drugs, don't include them in the filter
            from browse.models import WsAnnotation
            qs = WsAnnotation.objects.filter(ws=bji.ws)
            if bji.parms['exclude_hidden']:
                qs = qs.exclude(agent__hide=True)
            return set(qs.values_list('id',flat=True))
        elif self.key == 'uniprot':
            return None # anything is ok
        else:
            raise NotImplementedError('unknown key type')
    def set_fm_target(self,bji,fm):
        set_name = bji.parms['training_set']
        if self.key == 'wsa':
            target_keys = bji.ws.get_wsa_id_set(set_name)
        elif self.key == 'uniprot':
            target_keys = bji.ws.get_uniprot_set(set_name)
        else:
            raise NotImplementedError('unknown key type')
        fm.target = [1 if x in target_keys else 0 for x in fm.sample_keys]
        fm.target_names = ['False','True']

Flavor('wsa','efficacy','Drug Efficacy')
Flavor('wsa','novelty','Drug Novelty')
Flavor('uniprot','','Protein')

class MyJobInfo(JobInfo):
    def get_jobnames(self,ws):
        return [
                "%s_%s_%d" % (self.job_type,flavor.code(),ws.id)
                for flavor in Flavor.flavors
                ]
    def _get_flavor_from_jobname(self,jobname):
        parts = jobname.split('_')
        assert parts[0] == self.job_type
        return Flavor.get_by_code('_'.join(parts[1:-1]))
    def source_label(self,jobname):
        flavor = self._get_flavor_from_jobname(jobname)
        return flavor.label+' '+self.short_label
    def settings_defaults(self,ws):
        # construct default with an empty source list, so it includes
        # only non-source-specific settings
        from dtk.scores import SourceList
        sl=SourceList(ws)
        from dtk.job_prefix import SourceRoleMapper
        srm = SourceRoleMapper(sl)
        return {
                flavor.code():ConfigForm(ws,srm,flavor,None).as_dict()
                for flavor in Flavor.flavors
                }
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        flavor = self._get_flavor_from_jobname(job_type)
        from dtk.job_prefix import SourceRoleMapper
        srm = SourceRoleMapper(sources)
        warning = srm.non_unique_warning()
        if warning:
            return warning
        if copy_job:
            form = ConfigForm(ws,srm,flavor,copy_job.settings())
        else:
            form = ConfigForm(ws,srm,flavor,None)
        return form.as_html()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        flavor = self._get_flavor_from_jobname(jobname)
        from dtk.job_prefix import SourceRoleMapper
        srm = SourceRoleMapper(sources)
        warning = srm.non_unique_warning()
        if warning:
            return (warning,None)
        form = ConfigForm(ws,srm,flavor,None,post_data)
        if not form.is_valid():
            return (form.as_html(),None)
        p = form.cleaned_data
        settings = form.as_dict()
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(settings)
                            )
        next_url = ws.reverse('nav_progress',job_id)
        return (None, next_url)
    def _get_input_job_ids(self,job):
        parms = job.settings()
        from dtk.job_prefix import SourceRoleMapper
        return SourceRoleMapper.get_source_job_ids_from_settings(parms)
    def get_input_job_ids(self):
        return self._get_input_job_ids(self.job)
    def out_of_date_info(self,job,jcc):
        job_ids = self._get_input_job_ids(job)
        return self._out_of_date_from_ids(job,job_ids,jcc)
    def __init__(self,ws=None,job=None):
        # base class init
        self.use_LTS=True
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "Features",
                "Feature Matrix",
                )
        # any base class overrides for unbound instances go here
        self.needs_sources = True
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.fm_stem = self.lts_abs_root+"feature_matrix"
    def run(self):
        #make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.lts_abs_root)
        p_wr = ProgressWriter(self.progress
                , [ "wait for resources"
                  , "setup"
                  , "create feature matrix"
                  , "cleanup"
                  ]
                )
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        p_wr.put("setup","complete")
        self.write_feature_matrix()
        p_wr.put("create feature matrix","complete")
        self.finalize()
        p_wr.put("cleanup","complete")
    def write_feature_matrix(self):
        flavor = self._get_flavor_from_jobname(self.job.name)
        # build source list from settings
        from dtk.job_prefix import SourceRoleMapper
        srm=SourceRoleMapper.build_from_settings(self.ws,self.parms)
        print('Recovered sources:',srm.sources())
        # build list of feature columns to include
        import runner.data_catalog as dc
        attrs = [];
        for src_code,src in srm.sources():
            for parm in self.parms:
                if not is_feature_code(parm):
                    continue
                if src_code != extract_src_from_feature_code(parm):
                    continue
                dc_code = parm.split('_')[-1]
                # since only True values are stored, just look at the keys
                attrs.append('%d_%s'%(src.bji().job.id,dc_code))
        print('Recovered attributes:',attrs)
        # set up options
        import dtk.features as feat
        options={}
        if self.parms['plug_unknowns']:
            options['plug_unknowns']=0
        options['key_whitelist'] = flavor.get_key_whitelist(self)
        if self.parms['hf_labels']:
            options['job_labels'] = {
                    src.job_id():src_code
                    for src_code,src in srm.sources()
                    }
        # instantiate DCSpec and make FM
        spec = feat.DCSpec(self.ws.id,attrs,**options)
        fm = feat.FMBase.load_from_recipe(spec)
        # construct target attribute here, based on the selected training set
        flavor.set_fm_target(self,fm)
        # save the feature matrix
        fm.save(self.fm_stem)
    def get_feature_matrix_choices(self):
        return [
                self._get_default_feature_matrix_choice()
                ]
    def get_feature_matrix(self,code):
        assert code == self._get_default_feature_matrix_choice()[0]
        import dtk.features as feat
        return feat.FMBase.load_from_file(self.fm_stem)

if __name__ == "__main__":
    MyJobInfo.execute(logger)
