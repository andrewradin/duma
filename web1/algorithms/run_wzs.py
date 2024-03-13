#!/usr/bin/env python3

from __future__ import print_function
from builtins import range
import sys
import six


from path_helper import PathHelper,make_directory

import os
import django_setup

import logging
logger = logging.getLogger(__name__)
verbose = True

import json
import numpy as np
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo
import runner.data_catalog as dc
from django import forms
import random
from dtk.cache import cached

from scripts.wzs import GeneticAutoTuner, AutoTuner


################################################################################
# Configuration
################################################################################
class ConfigForm(forms.Form):
    fm_code = forms.ChoiceField(
                label = 'Feature Matrix',
                )
    algo = forms.ChoiceField(label='Combining algorithm'
                ,initial='wffl'
                ,choices=(
                        ('wtr','Weighted Top Rank'),
                        ('wts','Weighted Top Score'),
                        ('wffl','Weighted Fitted Floor'),
                        ('wfsig','Weighted Fitted Sigmoid'),
                        ('wzs','Weighted Z sum'),
                        ('wzm','Weighted Z max'),
                        ('wze','Weighted Z median'),
                        )
                )
    wtr_cutoff = forms.IntegerField(label='WT Rank Cutoff'
                ,required=False,# initial set dynamically below
                )
    auto_tune = forms.ChoiceField(label='Search for optimum weights'
                ,required=False,initial=GeneticAutoTuner.choice_code
                ,choices=AutoTuner.get_choices()
                )

    norm_choice = forms.ChoiceField(label='Normalization',
                 initial='mm'
                ,choices=(
                    ('z', 'Z-score normalization'),
                    ('mm', 'Min/Max normalization'),
                    ('lmm', 'Log Min/Max normalization'),
                    ('sc', 'Score Calibration'),
                    ('scl', 'Score Calibration (Log)'),
                    )
                )

    auto_iter = forms.IntegerField(label='Weight search iterations'
                ,required=False,initial=600
                )
    auto_cap = forms.IntegerField(label='Grid Walk max adjustment'
                ,required=False,initial=10
                )
    auto_step = forms.ChoiceField(label='Grid Walk initial stepsize'
                ,choices=(
                        (2**x,'%d steps'%2**x if x else '1 step')
                        for x in range(5)
                        )
                ,initial=8
                )
    auto_min_delta = forms.FloatField(
                label='Grid Walk minimum fractional improvement'
                ,required=False,initial=0.0,
                )
    auto_spacing = forms.FloatField(label='Grid Walk spacing'
                ,required=False,initial=0.2
                )
    auto_top_count = forms.IntegerField(label='Genetic top survivors'
                ,required=False,initial=12
                ,min_value=1
                )
    auto_extra_count = forms.IntegerField(label='Genetic other survivors'
                ,required=False,initial=18
                ,min_value=0
                )
    auto_new_count = forms.IntegerField(label='Genetic children'
                ,required=False,initial=120
                ,min_value=1
                )
    auto_init_sigma = forms.FloatField(
                label='Genetic initial mutation sigma'
                ,required=False,initial=0.5
                ,min_value=0
                )
    auto_init_freq = forms.FloatField(
                label='Genetic initial mutation fraction'
                ,required=False,initial=1.0
                ,min_value=0,max_value=1
                )
    auto_mut_sigma = forms.FloatField(
                label='Genetic mutation sigma'
                ,required=False,initial=0.6
                ,min_value=0
                )
    auto_mut_freq = forms.FloatField(
                label='Genetic mutation fraction'
                ,required=False,initial=0.9
                ,min_value=0,max_value=1
                )
    auto_w_min = forms.FloatField(
                label='Genetic min weight'
                ,required=False,initial=0.0
                )
    auto_anneal_cycles = forms.IntegerField(label='Anneal cycles'
            ,initial=6
            )
    auto_anneal_sigma = forms.BooleanField(label='Anneal Sigma'
            ,initial=True
            ,required=False
            )
    auto_frac_children = forms.BooleanField(label='Frac Children'
            ,initial=False
            ,required=False
            )
    auto_weighted_resample = forms.BooleanField(label='Weighted Resample'
            ,initial=True
            ,required=False
            )
    auto_directed_mutate = forms.FloatField(label='Directed Mutate'
            ,initial=0.5
            ,required=False
            )
    auto_dropout = forms.FloatField(label='Dropout Rate'
            ,initial=0.0
            ,required=False
            )
    auto_l2_reg = forms.FloatField(label='Regularization Strength'
            ,initial=0.5
            ,required=False
            ,help_text="How much regularization to apply (maximum fraction of score to penalize; 0.0 is no regularization, 1.0 is a lot)"
            )
    auto_reg_importance = forms.BooleanField(label='Regularize importance rather than weights'
            ,initial=True
            ,required=False
            )
    # This is currently a bit of a strange parameter in genetic search.
    # When w_min = 0, w_max conceptually constrains nothing.
    # This is because any unconstrained solution has an equivalent solution
    # that falls within our constraints (e.g. normalize all the weights).
    # However, because of how we search, there is almost 0 probability of
    # us finding those solutions.  The mutation sigma is sufficiently high
    # that it is nearly impossible for us to find solutions with lots of
    # weights that are very small.
    # Thus, this parameter is actually quite useful, but is heavily coupled
    # with mutation_sigma.
    auto_w_max = forms.FloatField(
                label='Genetic max weight'
                ,required=False,initial=1
                )
    auto_n_parents = forms.IntegerField(label='Genetic parents per child'
                ,required=False,initial=2
                ,min_value=1
                )
    auto_drug_set = forms.ChoiceField(label='Weight search target'
                ,initial=''
                ,choices=(('','None'),)
                )
    auto_metric = forms.ChoiceField(label='Weight search metric'
                ,initial=''
                ,choices=(('','None'),)
                )
    auto_gradient_descent = forms.BooleanField(label="Gradient Descent (if available)"
            ,initial=True
            ,required=False
            )
    auto_seed = forms.IntegerField(label='Random seed (-1 for random)'
            ,required=False
            ,initial=0
            )
    auto_constraints = forms.CharField(
                    widget=forms.Textarea(),
                    label='Score Constraint Groups',
                    required=False,
                    initial=json.dumps([
                        ["defus_indJacScore", "defus_dirJacScore", "defus_prMaxScore", "defus_pathwayScore"],
                        ["defus_indigoScore", "defus_rdkitScore"],
                        # All these capp scores come in via either opentargets or disgenet (either way same suffix).
                        # Constrain each subscore so they don't overwhlem, though we may want to do additional merging later.
                        ["capp_capis"],
                        ["capp_capds"],
                        ["capp_gpbr_directbgnormed"],
                        ["capp_gpbr_indirectbgnormed"],
                    ], indent=2)
                    )
    def __init__(self, ws, copy_job, *args, **kwargs):
        super(ConfigForm,self).__init__(*args, **kwargs)
        self.ws = ws
        # set wtr_cutoff initial value
        f = self.fields['wtr_cutoff']
        # for collections with lots of drugs (i.e. drugs are more
        # experimental) we want to look deeper when aggregating;
        # the exact numbers below are mostly guesses; we've
        # traditionally used 400 with drugbank, and started
        # using 1000 with chembl
        #
        # TODO think about setting this dynamically from the size of the FVS
        # (i.e. the number of drugs actually passed in)
        # to do this we'd also want to change wtr_cutoff to a portion
        drugs = ws.wsannotation_set.count()
        f.initial = 400 if drugs < 15000 else 1000
        # set feature matrix choices
        f = self.fields['fm_code']
        f.choices = ws.get_feature_matrix_choices(exclude=set(['ml']))
        f.choices = [(id, label) for id, label in f.choices if not 'Novelty' in label]
        f.initial = f.choices[0][0]
        # set auto-tune choices
        f = self.fields['auto_drug_set']
        f.choices = self.ws.get_wsa_id_set_choices(train_split=True,test_split=True)
        f.initial = self.ws.eval_drugset
        from dtk.enrichment import EnrichmentMetric
        f = self.fields['auto_metric']
        rec_metric = ['SigmaOfRank1000', 'SigmaOfRankCondensed']
        f.choices = [('Recommended', [(name, name) for name in rec_metric])]
        f.choices += [('Other', [
                (name,name)
                for name,cls in EnrichmentMetric.get_subclasses()
                if name not in rec_metric
                ])]
        f.initial = 'SigmaOfRank1000Condensed'
        if copy_job:
            for f in self:
                if f.name in copy_job:
                    v = copy_job[f.name]
                    # legacy copy mods
                    if f.name == 'auto_tune' and v == True:
                        v = 'gridwalk'
                    f.field.initial = v
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
        This is a meta method that takes prediction scores from a number of
        component methods and combines them. The basic method works by
        Z-norming each score, then taking a weighted average of all the scores.
        Several more complex norming and weighting variations are also
        available. It also includes an auto-tune feature which adjusts
        the per-score weights to optimize selection of a specified drugset
        (usually known treatments).
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
        try:
            code = parms['fm_code']
            return set([self.extract_feature_matrix_job_id(code)])
        except KeyError:
            pass
        from dtk.job_prefix import SourceRoleMapper
        return SourceRoleMapper.get_source_job_ids_from_settings(parms)
    def get_input_job_ids(self):
        return self._get_input_job_ids(self.job)
    def out_of_date_info(self,job,jcc):
        job_ids = self._get_input_job_ids(job)
        return self._out_of_date_from_ids(job,job_ids,jcc)

    def enrichment_metrics(self):
        return ('SigmaOfRank1000Condensed', 'SigmaOfRank1000', 'wFEBE')

    def __init__(self,ws=None,job=None):
        self.use_LTS=True
        # base class init
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "Weighted Z-Score",
                "Weighted Z-Score",
                )
        # any base class overrides for unbound instances go here
        # Prepopulate publinks in the order we want.
        self.publinks = [
            (None, 'combined.plotly'),
            (None, 'bar.plotly'),
            (None, 'box.plotly'),
            (None, 'final_wts.plotly'),
            (None, 'atl.plotly'),
            (None, 'hm_time.plotly'),
            (None, 'hm_init.plotly'),
            (None, 'hm_1qr.plotly'),
            (None, 'hm_mid.plotly'),
            (None, 'hm_3qr.plotly'),
            (None, 'hm_final.plotly'),
            (None, 'wt_sds.plotly'),
        ]

        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.fn_score=self.lts_abs_root+'wz_score.tsv'
            self.wts_file=self.lts_abs_root+'weights.tsv'
            self.details_file=self.lts_abs_root+'details.json'
            self.train_file=self.lts_abs_root+'train.tsv'

            try:
                # Add extra publinks at the end for anything not listed explicitly above.
                for fn in sorted(os.listdir(self.final_pubdir)):
                    if fn.endswith('.plotly') or fn.endswith('.plotly.gz'):
                        fn = fn.replace('.gz', '')
                        if (None, fn) not in self.publinks:
                            self.publinks.append((None, fn))
            except FileNotFoundError:
                pass

            if 'fm_code' in self.parms:
                ds = self.parms.get('auto_drug_set', self.ws.eval_drugset)
                grid_url = f"{self.ws.reverse('nav_score_grid')}?ds={ds}&from_wzs={self.job.id}"
                self.otherlinks = [
                    ('Score Grid',  grid_url),
                ]
    def get_data_code_groups(self):
        return [
                dc.CodeGroup('wsa',self._std_fetcher('fn_score'),
                        dc.Code('wzs',label='WZS',
                                meta_out=True,
                                ),
                        )
                ]

    def make_agg_model(self):
        fm_code = self.parms['fm_code']
        fm = self.ws.get_feature_matrix(fm_code)
        from scripts.wzs import make_agg_model
        return make_agg_model(
                self.parms,
                fm=fm,
                )

    def get_train_ids(self):
        '''Return set of WSA ids used to train the weights.'''
        if self.explicit_train_ids():
            from dtk.files import get_file_records
            return set(int(x[0]) for x in get_file_records(self.train_file))
        return self.ws.get_wsa_id_set(self.parms['auto_drug_set'])

    def explicit_train_ids(self):
        '''Were training WSA ids recorded explicitly?

        (versus being inferred from the current content of the auto_drug_set)
        Note that this is only valid once the job has finished running.
        '''
        return os.path.exists(self.train_file)

    def get_score_weights(self):
        self.fetch_lts_data()
        from dtk.files import get_file_records
        weights = [
                (frs[0],float(frs[1]))
                for frs in get_file_records(self.wts_file)
                ]
        return weights

    def get_score_floors(self):
        if self.job.settings()['algo'] != 'wffl':
            return []
        features = [x[0] for x in self.get_score_weights()]
        data = self.get_full_weights()
        assert len(data) == 2*len(features)
        return zip(features,data[len(features):])

    @cached(argsfunc=lambda self: self.job.id)
    def get_score_weights_and_sources(self):
        # Return info from a previously-completed run to allow rebuilding
        # final scores.
        from scripts.wzs import get_norm_for_ordering, get_per_feature_norms
        # recover feature keys and weights from file
        weights = self.get_score_weights()
        # rebuild srcs list
        try:
            fm_code = self.parms['fm_code']
        except KeyError:
            # this WZS run predates FM integration; reconstruct the
            # znorm list from settings instead
            from dtk.job_prefix import SourceRoleMapper
            srm=SourceRoleMapper.build_from_settings(self.ws,self.parms)
            role2cat={
                    role:ss.bji().get_data_catalog()
                    for role,ss in srm.sources()
                    }
            srcs = []
            for f_key,_ in weights:
                parts = f_key.split('_')
                dc_code = parts[-1]
                role = '_'.join(parts[:-1])
                ordering = role2cat[role].get_ordering(dc_code,True)
                srcs.append(get_norm_for_ordering(ordering,self.parms))
        else:
            fm = self.ws.get_feature_matrix(fm_code)
            srcs = get_per_feature_norms(fm,self.parms,self.details_file)
        return (weights, srcs)

# for newer WZSs this includes flooring information appended to the end of the weight data
    def get_full_weights(self, indices=None):
        if os.path.exists(self.details_file):
            with open(self.details_file, 'r') as f:
                import json
                weights = json.loads(f.read())['all_weights']
        else:
            # Older WZS job, get it from the older file.
            weights = [v for k, v in self.get_score_weights() if k]

        import numpy as np
        if indices is None:
            return np.array(weights)

        # Convert to numpy for fancy indexing.
        weights = np.array(weights)
        out_weights = list(weights[indices])

        score_weights = [(k,v) for k,v in self.get_score_weights() if k]
        N = len(score_weights)
        assert len(weights) == N or len(weights) == 2*N or len(weights) == 3*N, "New weights type, make sure it meets expectations"
        if len(weights) == N:
            return out_weights

        if len(weights) >= 2*N:
            shift_indices = np.array(indices) + N
            out_weights += list(weights[shift_indices])

        if len(weights) >= 3*N:
            shift_indices = np.array(indices) + 2*N
            out_weights += list(weights[shift_indices])

        return np.array(out_weights)

    def generation_size(self):
        if self.parms['auto_tune'] == GeneticAutoTuner.choice_code:
            return sum([
                    self.parms['auto_top_count'],
                    self.parms['auto_extra_count'],
                    self.parms['auto_new_count'],
                    ])
        else:
            return 1

    def run_remote(self):
        fm_stem = os.path.join(self.indir, 'fm')
        fm_code = self.parms['fm_code']
        fm = self.ws.get_feature_matrix(fm_code)
        parms = self.parms

        # Fill in the params we need from DB
        all_wsa_ids = fm.sample_keys
        train_wsa_ids = self.ws.get_wsa_id_set(self.parms['auto_drug_set'])
        from dtk.kt_split import parse_split_drugset_name
        splitds = parse_split_drugset_name(self.parms['auto_drug_set'])
        if splitds.is_split_drugset:
            test_wsa_ids = self.ws.get_wsa_id_set(splitds.complement_drugset)
        else:
            test_wsa_ids = train_wsa_ids

        from dtk.enrichment import make_dpi_group_mapping
        wsa_to_group, _ = make_dpi_group_mapping(all_wsa_ids)
        parms.update({
            'all_wsa_ids': list(all_wsa_ids),
            'train_wsa_ids': list(train_wsa_ids),
            'test_wsa_ids': list(test_wsa_ids),
            'wsa_to_group': dict(wsa_to_group),
            })


        parms_fn = os.path.join(self.indir, 'parms.json')
        if self.parms['norm_choice'] in ['sc','scl']:
            d = dict(self.parms)
            d['norm_choice'] = 'none'
            from dtk.score_calibration import FMCalibrator
            fmc=FMCalibrator(fm=fm)
            fmc.calibrate(logscale=bool(self.parms['norm_choice']=='scl'))
        else:
            d = self.parms
        with open(parms_fn, 'w') as f:
            f.write(json.dumps(d, indent=2))

        # record training WSAs as an audit trail
        from dtk.files import FileDestination
        with FileDestination(self.train_file) as fd:
            for wsa_id in train_wsa_ids:
                fd.append((wsa_id,))

        fm.save_flat(fm_stem)

        local=False
        if local:
            cvt = lambda x:x
        else:
            cvt = self.mch.get_remote_path

        script = cvt(os.path.join(
                        PathHelper.website_root, "scripts", "wzs.py"
                        ))
        cmd = [
            script,
            '--feature-matrix', cvt(fm_stem),
            '--parms', cvt(parms_fn),
            '-o', cvt(self.outdir),
            '-p', cvt(self.tmp_pubdir),
            '-c', str(self.remote_cores_got)
            ]

        if local:
            import subprocess
            subprocess.check_call(cmd)
        else:
            self.copy_input_to_remote()
            self.make_remote_directories([
                                    self.tmp_pubdir,
                                    ])
            self.mch.check_remote_cmd(' '.join(cmd))
            self.copy_output_from_remote()


        # Copy output to lts
        import shutil
        shutil.copy(self.outdir+'wz_score.tsv',self.lts_abs_root)
        shutil.copy(self.outdir+'weights.tsv',self.lts_abs_root)
        shutil.copy(self.outdir+'details.json',self.lts_abs_root)


    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        make_directory(self.lts_abs_root)
        steps = [
                "wait for resources",
                "run remote",
                "finalize",
                ]

        p_wr = ProgressWriter(self.progress, steps)
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        self.remote_cores_wanted = (8, 1 + self.generation_size() // 4)
        got = self.rm.wait_for_resources(self.job.id,[0, self.remote_cores_wanted])
        self.remote_cores_got = got[1]
        p_wr.put("wait for resources",f"complete ({self.remote_cores_got} cores)")
        self.run_remote()
        p_wr.put("run remote","complete")
        self.check_enrichment()
        self.finalize()
        p_wr.put("finalize","complete")
    def eval_drugsets(self):
        # If we trained on a split drugset, eval against the complement.
        from dtk.kt_split import parse_split_drugset_name
        splitds = parse_split_drugset_name(self.parms['auto_drug_set'])
        if splitds.is_split_drugset:
            return [
                ('test', splitds.complement_drugset),
                ('train', splitds.drugset_name),
                ('all', splitds.input_drugset),
                ]
        else:
            return super(MyJobInfo, self).eval_drugsets()

    def get_warnings(self):
        return super().get_warnings(
                 ignore_conditions=self.base_warning_ignore_conditions+[
                        # We're not using this parameter, so either a dependency is, or the depr warning is just very verbose.  Ignore either way.
                        lambda x: 'calling gather (from tensorflow.python.ops.array_ops) with validate_indices' in x
                         ],
                )



# Straight Z-norming has some serious limitations.  For some scores
# (e.g. structure in DN) the top 1000 scores have a norm above 2,
# and in some cases up to 10.  It would be nice to cap these
# so they don't dominate other scores.  A sigma scaler
# will do this, but setting the low end to pick up the
# correct number of scores would have to be done adaptively.
#
# The wtr option above addresses this problem more easily;
# it still requires a manual cutoff setting, but that number is
# more intuitive to set than something based on standard deviations.

# XXX In the long run, we should factor out the auto-tune framework,
# XXX and allow multiple variables per score, different spacings per
# XXX variable, and a choice between grid search and random search.
# XXX Then these cutoff parameters could be searched for as well.


if __name__ == "__main__":
    MyJobInfo.execute(logger)
