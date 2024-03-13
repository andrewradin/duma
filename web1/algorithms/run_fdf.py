#!/usr/bin/env python3

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

import logging
logger = logging.getLogger("algorithms.run_fdf")
verbose = True

import json
import numpy as np
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo
import runner.data_catalog as dc
from django import forms
import random
from dtk.subclass_registry import SubclassRegistry

################################################################################
# Configuration
################################################################################
class ConfigForm(forms.Form):
    fm_code = forms.ChoiceField(
                label = 'Feature Matrix',
                )
    ubiq = forms.BooleanField(
                    label='Remove previously annotated Ubiquitous drugs',
                    required=False,
                    initial=True,
                    )
    nfs = forms.BooleanField(
                    label='Remove drugs labeled not-for-sale by ZINC',
                    required=False,
                    initial=False,
                    )
    no_zinc_lab = forms.BooleanField(
                    label='...also remove drugs without ZINC labels',
                    required=False,
                    initial=False,
                    )

    def __init__(self, ws, copy_job, *args, **kwargs):
        super(ConfigForm,self).__init__(*args, **kwargs)
        self.ws = ws
        # set feature matrix choices
        f = self.fields['fm_code']
        f.choices = ws.get_feature_matrix_choices(exclude=set(['ml']))
    def as_html(self):
        from django.utils.html import format_html,format_html_join
        return format_html('''
                    <table>{}</table>
                    '''
                    ,self.as_table()
                    )
    def as_dict(self):
        if self.is_bound:
            src = self.cleaned_data
        else:
            src = {fld.name:fld.field.initial for fld in self}
        p = {
            'ws_id':self.ws.id,
            }
        for k,v in six.iteritems(src):
            p[k] = v
        return p

################################################################################
# JobInfo
################################################################################
class MyJobInfo(JobInfo):
    def description_text(self):
        return '''
        This CM inputs and outputs drug-keyed feature matrices.
        There are a (growing) number of common filters that can be
        applied using this CM. As a result this CM will always result
        in a FM with the same or fewer total drugs as the starting matrix.
        '''
    def settings_defaults(self,ws):
        cfg=ConfigForm(ws,None)
        return {
                'default':cfg.as_dict(),
                }
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        if copy_job:
            form = ConfigForm(ws,copy_job.settings())
        else:
            form = ConfigForm(ws,None)
        return form.as_html()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources):
        form = ConfigForm(ws,None,post_data)
        if not form.is_valid():
            return (form.as_html(),None)
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(form.as_dict())
                            )
        next_url = ws.reverse('nav_progress',job_id)
        return (None, next_url)
    def _get_input_job_ids(self,job):
        parms = job.settings()
        code = parms['fm_code']
        return set([self.extract_feature_matrix_job_id(code)])
    def get_input_job_ids(self):
        return self._get_input_job_ids(self.job)
    def out_of_date_info(self,job,jcc):
        job_ids = self._get_input_job_ids(job)
        return self._out_of_date_from_ids(job,job_ids,jcc)
    def __init__(self,ws=None,job=None):
        self.use_LTS=True
        # base class init
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "Filter drug FVs",
                "Filter drug FVs",
                )
        # any base class overrides for unbound instances go here
        self.publinks = [
#            (None, 'wt_sds.plotly'),
        ]
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.fm_stem = self.lts_abs_root+"feature_matrix"
#            self.final_wt_plot = self.tmp_pubdir+"final_wts.plotly"
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        make_directory(self.lts_abs_root)
        steps = [
                "wait for resources",
                "Filter candidates"
                ]
        if self.parms['ubiq']:
            steps.append('...Ubiquitious')
        if self.parms['nfs']:
            steps.append('...Not for sale')
        steps.append('...Restored interesting')
        steps.append("Create new feature matrix")
        p_wr = ProgressWriter(self.progress, steps)
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        res = self.filter()
        p_wr.put("Filter candidates",str(len(self.to_remove))+ ' filtered')
        for tup in res:
            p_wr.put(tup[0], tup[1])
        self.save()
        p_wr.put("Create new feature matrix","complete")
        self.finalize()
    def save(self):
        # remove rows keyed by IDs in self.to_remove
        self.fm.exclude_by_key(self.to_remove)
        # save the feature matrix
        self.fm.save(self.fm_stem)
    def filter(self):
        from browse.models import WsAnnotation
        to_ret = []
        self.to_remove = set()
        # get FM
        self.fm = self.ws.get_feature_matrix(self.parms['fm_code'])
        # extract keys and labels
        self.all_drugs = self.fm.sample_keys
        self.agent2wsaid = {
                wsa.agent_id:wsa.id
                for wsa in WsAnnotation.objects.filter(id__in=self.all_drugs)
                }
        if self.parms['ubiq']:
            cnt = self._find_ubiq_drugs()
            to_ret.append(('...Ubiquitious', str(cnt)))
        if self.parms['nfs']:
            cnt = self._find_nfs_drugs()
            to_ret.append(('...Not for sale', str(cnt)))

        cnt = self._restore_interesting()
        to_ret.append(('...Restored interesting', str(cnt)))
        return to_ret
    def _find_ubiq_drugs(self):
        from browse.models import WsAnnotation
        from scripts.flag_drugs_for_demerits import find_demeritted_drugs
        flags = find_demeritted_drugs(WsAnnotation.objects.filter(
                                        agent_id__in=list(self.agent2wsaid.keys()),
                                      ).exclude(
                                        demerit_list='',
                                      ),
                                      ['Ubiquitous']
                                     )
        ids_to_remove = [self.agent2wsaid[agent] for agent in flags]
        self.to_remove.update(set(ids_to_remove))
        return len(ids_to_remove)
    def _find_nfs_drugs(self):
        from browse.models import WsAnnotation
        from scripts.flag_drugs_for_zinc import get_zinc_labels
        labels_oi = set(['not-for-sale+in-cells',
                         'not-for-sale+in-vitro',
                        ]
                       )
        if self.parms['no_zinc_lab']:
            from dtk.zinc import zinc
            z = zinc()
            labels_oi.add(z.no_label_description())
        flags = get_zinc_labels(self.all_drugs)
        ids_to_remove = set()
        for wsa_id,l in six.iteritems(flags):
            remove = True
            for x in l:
                if len(set(x[1]) & labels_oi) == 0:
                    remove = False
                    break
            if remove:
                ids_to_remove.add(wsa_id)
        self.to_remove.update(set(ids_to_remove))
        return len(ids_to_remove)
    def _restore_interesting(self):
        from browse.models import WsAnnotation
        iv = WsAnnotation.indication_vals
        wsas = WsAnnotation.all_objects.filter(
                pk__in=self.to_remove,
                indication__in=[iv.UNCLASSIFIED, iv.INACTIVE_PREDICTION],
                )
        new_to_remove = wsas.values_list('id', flat=True)
        restored = set(self.to_remove) - set(new_to_remove)
        self.to_remove = set(new_to_remove)
        restored_wsas = WsAnnotation.all_objects.filter(pk__in=restored)
        for wsa in restored_wsas:
            print("Restored: ", wsa.agent.canonical)
        return len(restored)
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
