#!/usr/bin/env python3

from __future__ import print_function
from builtins import range
import sys
from path_helper import PathHelper,make_directory

import os
import django
import django_setup

import logging
logger = logging.getLogger("algorithms.run_faers")

from django import forms

import json
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo
import runner.data_catalog as dc
from dtk.faers import ClinicalEventCounts,get_vocab_for_cds_name


def demos_from_settings(settings):
    demo_covariates = []
    if settings['include_sex']:
        demo_covariates.append('sex')
    if settings['include_weight']:
        demo_covariates.append('wt_kg')
    if settings['include_age']:
        demo_covariates.append('age_yr')
    if settings['include_date']:
        demo_covariates.append('date')
    return demo_covariates

class ConfigForm(forms.Form):
    from dtk.faers import DemoFilter
    filt = DemoFilter()
    search_term = forms.CharField(
                label='Disease Name Patterns',
                required=False,
                )
    cds_base = forms.ChoiceField(
                label='Clinical Dataset',
                choices=(('','None'),),
                )
    # XXX if we converted this whole thing to a dynaform, we could just
    # XXX iterate through filter_parts here, rather than coding each
    # XXX part explicitly
    sex_filter = filt.get_form_field('sex_filter')
    min_age = filt.get_form_field('min_age')
    max_age = filt.get_form_field('max_age')
    drug_set = forms.ChoiceField(
                label='Drug Set',
                choices=(('','None'),),
                )
    C = forms.ChoiceField(
                label='Inverse Regularization Strength (C)',
                initial='1',
                choices=((x,float(x)) for x in [
                        '.0001',
                        '.003',
                        '.001',
                        '.03',
                        '.01',
                        '.3',
                        '1',
                        '3',
                        '10',
                        '30',
                        '100',
                        ])
                )
    autoscale_C = forms.BooleanField(
                label='Scale C relative to # of samples',
                required=False,
                initial=False,
                help_text="""The logreg error is relative to # of samples, whereas regularization only to # of features.
                    This setting counteracts that by rescaling C by # of samples. This means that L1 or elasticnet will
                    pick a similar number of features regardless of how many cases we have for this indication.
                """
                )
    class_weight = forms.ChoiceField(
                label='Class Weight Mode',
                initial='stratified',
                choices=(('None', 'None'), ('balanced', 'Balanced'), ('stratified', 'Stratified')),
                )
    penalty = forms.ChoiceField(
                label='Penalty',
                initial='l2',
                choices=(('l1', 'L1-Norm'), ('l2', 'L2-Norm'), ('elasticnet', 'ElasticNet')),
                help_text="If ElasticNet, consider setting C=0.1 for a sparse solution, otherwise it may take a long time to converge."
                )

    include_sex = forms.BooleanField(
                label='Include sex as feature',
                required=False,
                initial=True,
                )
    include_weight = forms.BooleanField(
                label='Include weight as feature',
                required=False,
                initial=True
                )
    include_age = forms.BooleanField(
                label='Include age as feature',
                required=False,
                initial=True
                )
    include_date = forms.BooleanField(
                label='Include event date as feature',
                required=False,
                initial=True
                )
    method = forms.ChoiceField(
                label='Logistic regression method',
                initial='normal',
                choices=(('normal', 'Normal'),('VB', 'Variational Bayes'), ('EB', 'Empirical Bayes'))
                )
    prefiltering = forms.ChoiceField(
                label='Matrix pre-filtering',
                initial='pvalue',
                choices=(('variance', 'Variance'), ('None', 'None'), ('pvalue', 'P Value')),
                help_text="Variance: Removes low-prevalence features.<br> P-Value: Removes features whose correlation could be spurious.",
                )

    split_drug_disease = forms.BooleanField(
                label='Split Drugs & Indis',
                required=False,
                initial=True,
                help_text="Fits drug and indication scores separately instead of jointly.",
                )
    
    ignore_known_treatments = forms.BooleanField(
                label='Ignore Known Treatments',
                required=False,
                initial=False,
                help_text="Fits drug scores without known treatments.",
                )

    ignore_single_indi = forms.BooleanField(
                label='Ignore Single Indi',
                required=False,
                initial=True,
                help_text="Filter out records that contain only a single indication (only for disease & only if split-drug-disease)",
                )

    def __init__(self, ws, *args, **kwargs):
        if 'initial' in kwargs:
            d = kwargs['initial']
            if 'cds' in d:
                from dtk.faers import DemoFilter
                stem,filt = DemoFilter.split_full_cds(d.pop('cds'))
                d['cds_base'] = stem
                df = DemoFilter(filt)
                d.update(df.as_dict())
        super(ConfigForm,self).__init__(*args, **kwargs)
        self.ws = ws
        # customize clinical dataset
        f = self.fields['cds_base']
        f.choices = self.ws.get_cds_choices()
        f.initial = f.choices[0][0]
        # ...then drugset
        f = self.fields['drug_set']
        f.choices = self.ws.get_wsa_id_set_choices()
        f.initial = self.ws.eval_drugset

        f = self.fields['include_sex']
        demos_url = self.ws.reverse('faers_demo')
        demos_link = f'<a href="{demos_url}">Demographics</a>'
        f.help_text=f"Take a look at {demos_link} to pick which demographics to include as features.",

        import browse.default_settings as ds
        self.fields['include_sex'].initial = ds.FaersIncludeSex.value(ws=ws)
        self.fields['include_age'].initial = ds.FaersIncludeAge.value(ws=ws)
        self.fields['include_weight'].initial = ds.FaersIncludeWeight.value(ws=ws)
        self.fields['include_date'].initial = ds.FaersIncludeDate.value(ws=ws)

    def as_dict(self):
        if self.is_bound:
            src = self.cleaned_data
        else:
            src = {fld.name:fld.field.initial for fld in self}
        settings = dict(src)
        # replace individual cds+filt items with a combined cds string
        cds = settings['cds_base']
        del settings['cds_base']
        from dtk.faers import DemoFilter
        df = DemoFilter()
        df.load_from_dict(settings,del_from_src=True)
        if df.as_string():
            cds += '?' + df.as_string()
        settings['cds'] = cds
        # supply workspace
        settings['ws_id'] = self.ws.id
        # supply search term fallback
        if not settings['search_term']:
            settings['search_term'] = self.ws.get_disease_default(
                                    get_vocab_for_cds_name(src['cds_base'])
                                    )
        return settings
    def as_html(self):
        from django.utils.html import format_html
        return format_html(
                '''<div class="well"><p style='font-weight:bold'>Disease Name Patterns</p>{}<p><p>
                If left blank, the default for this workspace will be used.
                <p>
                For ideas on what you might want to match, you can
                <b>see all
                <a href="{}">FAERS</a>
                disease names.</b>
                <br>

                <p style='font-weight:bold'>Inverse Regularization Strength</p> This number
                is the hyperparameter C for the logistic
                regression model. It is inversely proportional
                to the degree of sparsity it causes on the
                coefficients of the model (i.e. the lower the value, the more zero
                coefficients).
                <b>For Bayesian LR, this should be set to the minimum value.</b>
                </p>
                <p style='font-weight:bold'>Class Weight</p> Setting this to 'balanced' tells the
                model to fit itself using a weight for each class that is inversely proportional to its
                frequency. In short, this will adjust the model to reduce the error associated with classifying
                a class that occurs infrequently.
                </p>
                <p style='font-weight:bold'>Penalty</p> The method used to calculate the regularization penalty.
                <span style='font-style:italic'>L1-Norm</span>, or 'least absolute deviations', is the minimization
                of a sum of the absolute differences
                between the target and the estimated values. <span style='font-style:italic'> L2-Norm</span>,
                or 'least squares', minimizes the sum of the square of the differences between the target and
                estimates values. <br><br>
                <a href='http://www.chioka.in/differences-between-the-l1-norm-and-the-l2-norm-least-absolute-deviations-and-least-squares/'>Click here</a> for the pros and cons of either method.
                </p>
                <p style='font-weight:bold'>Including Demographics as Features</p><p> Currently the logistic regression model does not accomodate null or missing values. Adding certain demographic features like weight and age will significantly limit the number of total events included as samples in the model. Most events contain sex, so including this demographic as a feature will have the smallest effect on sample size. However, it is important to note that including demographics as covariates can be very important when seeking to limit spurious results.</p>
                <p style='font-weight:bold'>Logistic Regression Methods</p> <p>Three logistic regression methods are available. "Normal" or regularized logistic regression, which uses the basic logistic regression class provided by scikit-learn, <a href='https://en.wikipedia.org/wiki/Variational_Bayesian_methods'>Variational Bayes</a>, and <a href='https://en.wikipedia.org/wiki/Empirical_Bayes_method'>Empirical Bayes</a>. For more information on how bayesian logistic regression approaches have been applied to clinical data, read <a href='https://projecteuclid.org/download/pdfview_1/euclid.ss/1346849939'>Multivariate Bayesian Logistic Regression for Analysis of Clinical Study Safety Issues</a> by William DuMouchel.</p>
                <p>{}
                </p>
                </div>
                '''
                ,   '''
                    The Disease Name Patterns field is one or more
                    SQL-style patterns separated by '|' symbols.
                    An SQL-style pattern is case-insensitive, and
                    uses '%' symbols as wildcards.  Each SQL pattern
                    must match the entire FAERS disease name, so to
                    match a word or phrase in the middle of a name,
                    begin and end the pattern with '%'.
                    '''
                , self.ws.reverse('nav_ontolist','FAERS')
                , self.as_p()
                )

class MyJobInfo(JobInfo):
    def description_text(self):
        return '''
        <b>FAERS</b> repurposes adverse event databases to measure
        statistical enrichment (or depletion) of disease occurence
        in patients taking a drug. It currently supports the FAERS
        clinical dataseti but could be adapted to other similarly
        structured data.

        The FAERS algorithm also produces disease co-occurance data.
        '''
    def settings_defaults(self,ws):
        form=ConfigForm(ws)
        return {
                'default':form.as_dict(),
                }
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        if copy_job:
            form = ConfigForm(ws,initial=copy_job.settings())
        else:
            form = ConfigForm(ws)
        return form.as_html()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        form = ConfigForm(ws,post_data)
        if not form.is_valid():
            return (form.as_html(),None)
        settings = form.as_dict()
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(settings)
                            )
        next_url = ws.reverse('nav_progress',job_id)
        return (None, next_url)
    def build_role_code(self,jobname,settings_json):
        import json
        d = json.loads(settings_json)
        prefix = get_vocab_for_cds_name(d['cds'])
        return '_'.join([prefix,self.job_type])
    def role_label(self):
        # This one's a little unusual.
        # For a while we supported both FAERS and CVAROD datasets
        # through the single run_faers CM, but to avoid labels like
        # FAERS FAERS, we just label with the dataset name, and drop
        # the CM name part.  But, if it's a legacy format, use the job type.
        if self.job.role:
            return self.job.role.split('_')[0]
        return self.short_label
    def __init__(self,ws=None,job=None):
        # base class init
        self.use_LTS = True
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "FAERS",
                "FAERS Drug/Disease Co-occurrence",
                )
        # any base class overrides for unbound instances go here
        self.publinks = (
                (None,'barpos.plotly'),
                (None, 'scatter_lr.plotly'),
                (None,'scatter.plotly'),
                (None, 'scatter_atc_1.plotly'),
                (None, 'scatter_atc_2.plotly'),
                (None, 'scatter_atc_3.plotly'),
                (None, 'scatter_atc_4.plotly')
                )
        self.qc_plot_files = (
                'barpos.plotly',
                'scatter_lr.plotly',
                )
        # job-specific properties

        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            # inputs
            # outputs
            self.fn_enrichment = self.lts_abs_root+'faers_output.tsv'
            self.lr_enrichment = self.lts_abs_root+'lr_faers_output.tsv'
            self.fn_coindications = self.lts_abs_root+'coindications.tsv'
            self.important_stats = self.lts_abs_root+'important_stats.pkl'
            self.tmp_important_stats = self.outdir+'important_stats.pkl'
            self.fn_model_output = self.lts_abs_root+'all_model_results.tsv.gz'
            # published output files
            self.scatter = self.tmp_pubdir+"scatter.plotly"
            self.barpos = self.tmp_pubdir+"barpos.plotly"

            if os.path.exists(self.fn_model_output):
                url = f"{self.ws.reverse('faers_run_table')}?jid={self.job.id}"
                self.otherlinks = [
                    ('FAERS Data Table', url),
                ]
    def get_data_code_groups(self):
        from math import log
        return [
                dc.CodeGroup('wsa',self._std_fetcher('fn_enrichment'),
                        dc.Code('dcop',label='Disease CO Portion',
                                efficacy=False,
                                ),
                        dc.Code('dcoe',label='Disease CO+',
                                efficacy=False,
                                ),
                        dc.Code('dcod',label='Disease CO-',
                                # lambda avoids operator.neg(0.0) => -0.0
                                calc=(lambda x:0-x,'dcoe'),
                                efficacy=False,
                                ),
                        dc.Code('dcoq',label='Disease CO FDR',
                                efficacy=False,
                                fmt='%.3e',
                                ),
                        dc.Code('drugPor',label='Drug portion',
                                efficacy=False,
                               )
                        ),
                dc.CodeGroup('wsa', self._std_fetcher('lr_enrichment'),
                        dc.Code('lrpvalue', label='LR P-Value',
                                efficacy=False,
                                fmt='%.3e',
                                ),
                        dc.Code('lrenrichment', label='LR Score',
                                efficacy=False,
                                ),
                        dc.Code('lrdir', label='LR Direction',
                                efficacy=False,
                                ),
                        ),
                dc.CodeGroup('mindi',self._std_fetcher('fn_coindications'),
                        dc.Code('coior', label='Co-ind Odds Ratio'),
                        dc.Code('coiqv', label='Co-ind Q-value'),
                        ),
                ]
    def get_warnings(self):
        return super().get_warnings(
                 ignore_conditions=self.base_warning_ignore_conditions+[
                        # per the comment in the script noted below,
                        # the divide by 0 doesn't impact results and is a reasonable thing to occur in the data
                        # so no need to be concerned
                        lambda x:'2xar/twoxar-demo/web1/scripts/faers_lr_model.py:76: RuntimeWarning: divide by zero encountered in double_scalars' in x,
                         ],
                )
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        make_directory(self.lts_abs_root)
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "extract data",
                "run statistics",
                "plotting results",
                "check enrichment",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.extract()
        p_wr.put("extract data",'complete'
                )
        self.run_statistics()
        self.run_logReg()
        p_wr.put("run statistics","complete")
        self.save_results()
        self.plot()
        self.plot_lr()
        p_wr.put("plotting results","complete")
        self.report()
        self.check_enrichment()
        self.finalize()
        p_wr.put("check enrichment","complete")

    def save_results(self):
        from dtk.files import gzip_file
        gzip_file(self.outdir+'all_model_results.tsv', self.fn_model_output)

    def report(self):
        with open(self.fn_enrichment, 'w') as f:
            f.write("\t".join(["wsa",
                               "dcoq",
                               "dcoe",
                               "dcop",
                               "drugPor",
                              ]) + "\n")
            for k,l in self.results.items():
                f.write("\t".join([str(x)
                                   for x in [k] + l
                                  ]) + "\n")
        with open(self.lr_enrichment, 'w') as f:
            f.write("\t".join(["wsa",
                               "lrpvalue",
                               "lrenrichment",
                               "lrdir"
                              ]) + "\n")
            for k,l in self.lr_results.items():
                f.write("\t".join([str(x)
                                   for x in [k] + l
                                  ]) + "\n")

    def plot(self):
        import pickle
        with open(self.tmp_important_stats, 'rb') as handle:
            statsdict = pickle.load(handle)
            acc = [statsdict['accuracy'], statsdict['wtd_accuracy']]
            pos = statsdict['target_sum']
            statsdict['search_term'] = self.indi_set
            statsdict['total'] = self.disease_total
            with open(self.important_stats, 'wb') as f:
                pickle.dump(statsdict, f)

        from dtk.plot import scatter2d, PlotlyPlot, annotations, Color, fig_legend
        go_bars = []
        go_bars.append(dict(x = ['FAERS Breakdown'],
                            y =  [pos],
                            name = 'Full Demographics',
                            type='bar',
                          ))
        go_bars.append(dict(x = ['FAERS Breakdown'],
                            y =  [self.disease_total-pos],
                            name = 'Disease Events',
                            type='bar',
                           ))
        pp_bar = PlotlyPlot(go_bars,
               {
                'width':400,
                'height':500,
                'barmode':'stack'
               }
        )
        pp_bar._layout['annotations'] = [fig_legend([
                                         'The number of disease events'
                                         ,'matched with and without all'
                                         ,'selected demographics. We need'
                                         ,'at minimum 100 events with the'
                                         ,'selected demographics.'
                                         ],-0.1)]
        pp_bar._layout['margin'] = dict(b=120)
        pp_bar.save(self.barpos)
        print('Bar saved')
        pp = scatter2d(
                    'Disease prevalence',
                    'Log2(odds ratio)',
                    self.xy,
                    refline=False,
                    ids=('drugpage',self.ids),
                    text=self.text,
                    classes=[
                            ('Unknown',{'color':Color.default, 'opacity':0.5}),
                            ('KT',{'color':Color.highlight, 'opacity':0.5}),
                            ],
                    class_idx=self.class_idx,
                    logscale = 'xaxis',
                    bins=True
                    )
        pp.save(self.scatter)
    def extract(self):
        self.indi_set = self.parms['search_term'].split('|')
        cec=ClinicalEventCounts(self.parms['cds'])
        # get portion data
        (
                bg_ctr,
                self.bg_total,
                self.indi_ctr,
                self.disease_total,
                )=cec.get_drug_portions(self.indi_set)
        self.bg_per_cas = dict(bg_ctr)

    def run_statistics(self):
        import scipy.stats as stats
        from math import log
        from dtk.enrichment import mult_hyp_correct
        from dtk.faers import CASLookup
        import numpy as np
### TODO move this somewhere better/to the UI
        self.minp=0.05
###
        dropped = {'no_overlap':set(), 'no_cnt':set()}
        self.cas_for_cols=[]
        kts = self.ws.get_wsa_id_set(self.parms['drug_set'])
        self.cas_lookup=CASLookup(self.ws.id)
        results = []
        ps = []
        self.ids = []
        self.xy = []
        self.text = []
        self.class_idx = []
        for cas,disease_cnt in self.indi_ctr:
            tmp = self.cas_lookup.get_name_and_wsa(cas)
            if not tmp or tmp[1] is None:
                try:
                    dropped['no_overlap'].add(self.cas_lookup.get_name_and_wsa(cas)[0])
                except TypeError:
                    pass
                continue
            name,wsa = tmp
            bg_cnt = self.bg_per_cas[cas]
            if not disease_cnt or not bg_cnt:
                try:
                    dropped['no_cnt'].add(self.cas_lookup.get_name_and_wsa(cas)[0])
                except TypeError:
                    pass
                continue
            oddsr, p = stats.fisher_exact(self._generate_confusion_matrix(
                                            disease_cnt,
                                            bg_cnt
                                           )
                                         )
            if p < self.minp:
                self.cas_for_cols.append(cas)
            ps.append(p)
            prev = float(disease_cnt) / self.disease_total
            oddsr = log(oddsr,2)
            if np.isinf(oddsr):
                oddsr = 10
            results.append([
                            oddsr,
                            prev,
                            float(disease_cnt) / bg_cnt
                           ])
            self.ids.append(wsa.id)
            self.class_idx.append( 1 if wsa.id in kts else 0 )
            self.xy.append((prev, oddsr))
            self.text.append('%s<br>%d / %d' % (name,disease_cnt,bg_cnt) )
        for k,s in dropped.items():
            print(k+': '+", ".join(s))
        qs = mult_hyp_correct(ps)
        self.results = {}
        for i in range(len(self.ids)):
            self.text[i] += '<br>%.3e' % (qs[i])
            self.results[self.ids[i]] = [qs[i]] + results[i]

    def _generate_confusion_matrix(self, disease_cnt, bg_cnt):
        a = disease_cnt
        b = self.disease_total - disease_cnt
        c = bg_cnt - disease_cnt
        d = self.bg_total - a - b - c
        return [[a,b],
                [c,d]
               ]
    def run_logReg(self):
        demo_covariates = demos_from_settings(self.parms)

        C = self.parms['C']
            

        local = False
        if local:
            # special hook for testing LR script changes
            # without deploying to worker machine
            from scripts.faers_lr_model import FAERS_LR
            model = FAERS_LR(
                    self.parms['cds'],
                    self.indi_set,
                    self.indir,
                    self.outdir,
                    C,
                    self.parms['penalty'],
                    self.parms['class_weight'],
                    demo_covariates,
                    self.parms['method'],
                    )
            model.build_matrix()
            model.fit_and_summarize()
            model.plot_atc()
            return
        self.make_remote_directories([
                                self.tmp_pubdir,
                                ])
        rem_cmd = self.mch.get_remote_path(
                                    PathHelper.website_root+"scripts/faers_lr_model.py"
                                    )

        options = [ '"'+self.parms['cds']+'"',
                    '"'+self.parms['search_term']+'"',
                    self.mch.get_remote_path(self.indir),
                    self.mch.get_remote_path(self.outdir),
                    self.parms['method'],
                    '--C '+str(C),
                    '--class_weight '+str(self.parms['class_weight']),
                    '--penalty '+str(self.parms['penalty']),
                    '--prefiltering '+str(self.parms['prefiltering']),
                    '--demos '+" ".join(demo_covariates)
                   ]
        
        if self.parms['split_drug_disease']:
            options += ['--split-drug-disease']
        if self.parms['autoscale_C']:
            options += ['--autoscale-C']
        if self.parms['ignore_single_indi']:
            options += ['--ignore-single-indi']
        if self.parms['ignore_known_treatments']:
            from browse.models import WsAnnotation
            from drugs.models import Drug, Tag
            known_treatment_ids = self.ws.get_wsa_id_set('tts')
            kt_wsas = WsAnnotation.objects.filter(pk__in=known_treatment_ids)
            tags_mm = Drug.bulk_prop(kt_wsas.values_list("agent_id", flat=True), self.ws.get_dpi_version(), 'cas', Tag)
            all_cas = list(tags_mm.rev_map().keys())
            ignore_path = os.path.join(self.indir, 'ignore_features.txt')
            with open(ignore_path, 'w') as f:
                f.write('\n'.join(all_cas))

            options += ['--ignore-features', self.mch.get_remote_path(ignore_path)]

        self.copy_input_to_remote()
        self.mch.check_remote_cmd(' '.join([rem_cmd]+options))
        self.copy_output_from_remote()

    def plot_lr(self):
        import pandas as pd
        import numpy as np

        ### Build base dataframe that consists of only events positive for indication
        ### (disease)
        #dis_dat = self._build_a_df(sc.filtered_events(self.indi_set, 'drug4indi'))

        ### Fetch all events negative for indication (disease) that also are positive
        ### for at least one CAS number in positive indication event set

        results_df = pd.read_csv(self.outdir+'all_model_results.tsv', sep='\t')

        class_idx = []
        results_df["HR Feature Names"] = results_df["Feature"]

        click_through = []

        coindications = []

        print('classifying features')

        # Adjust P-Values in Dataframe to a value greater than 0
        from dtk.enrichment import mult_hyp_correct,no_zero_pvalues
        results_df['P-Value'].fillna(1.0, inplace=True)
        # This is because the bayesian approaches control for false positives
        results_df['Q-Value'] = mult_hyp_correct(results_df['P-Value']) if self.parms['method']=='normal' else no_zero_pvalues(results_df['P-Value'])

        kts = self.ws.get_wsa_id_set(self.parms['drug_set'])
        for ix in results_df.index:
            feature  = results_df.loc[ix, 'Feature']
            (_class, drug_info)= self.feature_classifier(feature, kts)

            try:
                (name, wsa) = drug_info
            except TypeError:
                name = wsa = None

            if name:
                results_df.at[ix, 'HR Feature Names'] = name

            if wsa:
                click_through.append(wsa.id)
            else:
                click_through.append(0)

            class_idx.append(_class)

            if _class == 0:
                # Code for coindications (we want to record these separately)
                coindications.append(results_df.loc[ix, :].values)

        coind_df = pd.DataFrame(coindications, columns=results_df.columns)
        coind_df = coind_df[['Odds Ratio', 'Q-Value', 'Feature']]
        coind_df.set_index('Feature', inplace=True)
        coind_df.to_csv(self.fn_coindications, sep='\t',
                index_label='mindi',
                header=['coior','coiqv'],
                )

        text=['{0}<br>Q-Value:{1}<br>Log2(Odds Ratio):{2}'.format(str(title),
                                                                  '%.2E' % p,
                                                                  str(np.round(odds, 3))
                                                                 )
              for title, p, odds
              in zip(results_df['HR Feature Names'].values,
                                results_df['Q-Value'],
                                results_df['Odds Ratio'].apply(lambda x: np.log2(x)).values)
             ]

        formatted_q_values = list(results_df['Q-Value'].apply(lambda x: -np.log10(x)).values)
        formatted_OR = list(results_df['Odds Ratio'].apply(lambda x: np.log2(x)).values)
        import pickle
        with open(self.important_stats, 'rb+') as handle:
            statsdict = pickle.load(handle, encoding='utf8')
            acc = [statsdict['accuracy'], statsdict['wtd_accuracy']]
            pos = statsdict['target_sum']
        from dtk.plot import scatter2d, PlotlyPlot, annotations, Color,fig_legend
        pp_2 = scatter2d(
                    '-log10(Q-Value)',
                    'log2(Odds ratio)',
                    zip(formatted_q_values, formatted_OR),
                    refline=False,
                    text = text,
                    ids = ('drugpage', click_through),
                    classes=[
                            ('Indication',{'color':Color.highlight2, 'opacity':0.5}),
                            ('Drug',{'color':Color.default, 'opacity':0.5}),
                            ('KT',{'color':Color.highlight, 'opacity':0.5}),
                            ('Demographic',{'color':Color.highlight3, 'opacity':0.5}),
                            ('Unmapped CAS number', {'color':Color.negative,'opacity':0.5}),
                            ('Date', {'color':Color.highlight4, 'opacity':0.5})
                            ],
                    class_idx=class_idx,
                    bins = True,
                    annotations = annotations(
                                      f'Total accuracy: {acc[0]:.4f}',
                                      f'Weighted accuracy: {acc[1]:.4f}',
                                  )
                    )
        pp_2._layout['annotations'].append(fig_legend([
                                         'Each dot in this plot is a factor used to build a '
                                         +'logistic regression model to predict if an event had'
                                         ,'the disease. The y-axis is the influence that '
                                         +'factor had on the prediction and the X is the significance.'
                                         ,'Negative Y values are consistent with the factor '
                                         +'having a "protective effect" meaning the disease'
                                         ,'is less likely when that factor is present.'
                                         ],-0.25))
        pp_2._layout['margin']=dict(b=120)
        pp_2.save(self.tmp_pubdir+'scatter_lr.plotly')
        print('LR plot saved')

        self.lr_results = {}
        for row in results_df.iterrows():
            tup =  self.cas_lookup.get_name_and_wsa(row[1]['Feature'])
            if tup and tup[1] is not None:
                wsa_id = tup[1].id
                q_value = row[1]['Q-Value']
                OR = row[1]['Odds Ratio']
                enrichment = abs(np.log2(OR))
                direction = np.sign(np.log2(OR))
                self.lr_results[wsa_id] = [q_value, enrichment, direction]
            else:
                # not a drug
                continue

        with open(self.outdir+'neg_atc_dict.pkl', 'rb') as f:
            neg_dict = pickle.load(f)

        with open(self.outdir+'pos_atc_dict.pkl', 'rb') as f:
            pos_dict = pickle.load(f)

        pos_dict, neg_dict  = self._get_atcs(pos_dict, neg_dict)
        atc_levels = [1,2,3,4]
        bar_names = ['Pos', 'Neg']
        max_limit = 20
        for level in atc_levels:
            go_bars = []
            print('creating ATC plot #{0}'.format(str(level)))
            limit = 20 if max_limit > len(list(pos_dict[level].keys())) else max_limit
            # Sorting logic to get the bar plots sorted in decending order of ATC frequency in drug set 1.
            ordered_atcs = sorted(pos_dict[level],
                                  reverse=True,
                                  key=lambda x: (float(pos_dict[level][x]-neg_dict[level][x])/
                                                 (neg_dict[level][x]+1e-9)) if x in list(pos_dict[level].keys()) else neg_dict[level][x]
                                 )[:limit]
            atc_counts_1 = [pos_dict[level][k] if k in pos_dict[level]
                                               else 0
                            for k in ordered_atcs
                           ]
            atc_counts_2 = [neg_dict[level][k] if k in neg_dict[level]
                                               else 0
                            for k in ordered_atcs
                           ]
            ordered_atcs.extend(list(set(ordered_atcs)-set(neg_dict.keys())))
            atc_map = self._get_atc_code_map()
            max_tick_lengths = []
            for i, ds_counts in enumerate([atc_counts_1, atc_counts_2]):
                x_tick_labels = [atc_map[code] if code in atc_map else code
                                 for code in ordered_atcs
                                ]
                max_tick_lengths.append(max([len(label)
                                             for label in x_tick_labels
                                            ], default=0))
                go_bars.append(dict(x = x_tick_labels,
                                    y =  ds_counts,
                                    name = bar_names[i],
                                    type='bar',
                                   ))

            pp3 = PlotlyPlot(go_bars,
                           {'title':'Comparison of Level {0} ATC Code Frequencies'.format(level),
                            'yaxis':{'title':'Proportion of Drugs Level {0} ATC Code'.format(level)},
                            'xaxis':{'title':'Level {0} ATC Code'.format(level)},
                            'width':max([600,
                                len(ordered_atcs)*25
                                ]),
                            'height':600+max(max_tick_lengths)*10,
                             'margin':dict(
                             b=max(max_tick_lengths)*10
                             ),
                             'barmode':'stack'
                           }
                    )


    def _get_atcs(self, *args):
        # Partially based on /home/ubuntu/2xar/twoxar-demo/experiments/IBD_MTs/aggregate_paths_atcs_per_scoreboard.py
        from collections import defaultdict
        atc_cut_pts = [1,3,4,5]
        atc_levels = [1,2,3,4]
        ds1_atcs = {n:defaultdict(int) for n in atc_levels}
        ds2_atcs = {n:defaultdict(int) for n in atc_levels}
        from algorithms.run_orfex import get_atc_cache
        self.atc_cache = get_atc_cache()

        from browse.models import WsAnnotation

        from collections import Counter
        ds_atcs = [ds1_atcs, ds2_atcs]
        drug_sets = args

        # For each drug set, get each drug's ATC codes at each level, store in a temporary dictionary (ds*_atcs)
        ds1_atc_count = 0
        ds2_atc_count = 0
        atc_counters = [ds1_atc_count, ds2_atc_count]
        for ix, ds in enumerate(drug_sets):
            for drug_id, cnt in ds.items():
                tup = self.cas_lookup.get_name_and_wsa(drug_id)
                if tup and tup[1]:
                    wsa = tup[1]
                    l = self.atc_cache.get(wsa.agent_id, None)
                    if l is None:
                        l = []
                    for atc in l:
                        atc_counters[ix] += cnt
                        for i,lev in enumerate(atc_levels):
                             ds_atcs[ix][lev][atc[0:atc_cut_pts[i]]] += cnt

        # For each drug set, aggregate occurances of ATC codes at each level, then transform into proportion,
        # storing each in self.ds*_atc_counts, a dictionary of dictionaries. This will be used to create a grouped
        # bar plot in _plot_atc_freq(self)
        ds1_atc_counts = {}
        ds2_atc_counts = {}
        ds_atc_counts = [ds1_atc_counts, ds2_atc_counts]

        for ix,ds_atc_dict in enumerate(ds_atcs):
            for level, inner_dict in ds_atc_dict.items():
                # Use a tiny epsilon to avoid divide-by-zero.
                ds_atc_counts[ix][level] = {k:float(v)/(1e-16 + atc_counters[ix]) for k,v in inner_dict.items()}

        return ds_atc_counts
    @staticmethod
    def _get_atc_code_map():
         '''
         Static method that pulls down the atc_code.txt file,
         which maps codes to readable strings from S3. To refresh this file,
         run use Makefile in databases/atc.
         '''
         from dtk.s3_cache import S3MiscBucket,S3File
         from dtk.files import get_file_records
         s3_file = S3File(S3MiscBucket(),'atc_code.tsv')
         s3_file.fetch()
         result = {}
         for rec in get_file_records(s3_file.path()):
             result[rec[0]] = rec[1].title()
         return result
    def feature_classifier(self, feature, kts):
        import re
        class_dict = {'Indication':0,
                      'KT':2,
                      'Other Drug':1,
                      'Demographic':3,
                      'Unmapped Drug':4,
                      'Date':5}
        tmp = self.cas_lookup.get_name_and_wsa(feature)
        if tmp:
            # CAS lookup successful
            if tmp[1]:
                # WSA ID exists
                if tmp[1].id in kts:
                    _class = class_dict['KT']
                else:
                    _class = class_dict['Other Drug']
            elif tmp[0]:
                # No WSA, but we have a name, mark as drug
                _class = class_dict['Other Drug']
        elif any([
                feature in ['sex', 'weight_kg', 'age_yr'],
                feature in ['male', 'female'],
                (str(feature).startswith('age ') and str(feature).endswith(' yrs')),
                (str(feature).startswith('weight ') and str(feature).endswith(' kg')),
                ]):
            # Demographic feature
            _class = class_dict['Demographic']
        elif not tmp and re.match('[1-9]{1}[0-9]{1,5}-\d{2}-\d{1,5}', str(feature)):
	    # CAS lookup failed, but this is definitely a drug
            _class = class_dict['Unmapped Drug']
        elif any([
                str(feature) in ['Q1', 'Q2', 'Q3', 'Q4'],
                re.match('the[0-9]0s',str(feature)),
                ]):
            # This is a date covariate
            _class = class_dict['Date']
        else:
            _class = class_dict['Indication']
        return (_class, tmp)

    def _report_lr(self):
        import numpy as np
        print(self.lr_result.summary())
        conf = self.lr_result.conf_int()
        conf['OR'] = self.lr_result.params
        conf.columns = ['2.5%', '97.5%', 'OR']
        for i,x in conf.iterrows():
            if np.sign(x[0]) == np.sign(x[1]):
                print(x)
        #print np.exp(conf)

if __name__ == "__main__":
    MyJobInfo.execute(logger)
