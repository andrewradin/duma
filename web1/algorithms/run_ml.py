#!/usr/bin/env python3

from __future__ import print_function
import sys
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

class ConfigForm(forms.Form):
    fm_code = forms.ChoiceField(
                label = 'Feature Matrix',
                )
    method = forms.ChoiceField(label='Classifier method',
                initial='RF',
                choices=(
                        ('RF', "Random Forest"),
                        ('logistic', "Logistic Regression"),
                        ('attrSel_RF', "RF with attribute selection"),
                        ('attrSel_lr', "LR with attribute selection"),
                        ('RF_weight', "RF with class weighting"),
                        ('RF_tune', "RF with parameter tuning"),
                        ('decorate', "DECORATE using RF"),
                        ('naiveBayes', "Naive Bayes"),
                        ('svm', "SVM"),
                        ),
                )
    outer = forms.IntegerField(label='Training sets'
                ,required=False,initial=20
                )
    inner = forms.IntegerField(label='Trainings per set'
                ,required=False,initial=35
                )
    def __init__(self, ws, copy_job, *args, **kwargs):
        super(ConfigForm,self).__init__(*args, **kwargs)
        from runner.models import Process
        self.ws = ws
        f = self.fields['fm_code']
        f.choices = ws.get_feature_matrix_choices(exclude=set(['ml']))
        if copy_job:
            for f in self:
                if f.name in copy_job:
                    f.field.initial = copy_job[f.name]
    def as_html(self):
        return self.as_p()
    def as_dict(self):
        import runner.data_catalog as dc
        if self.is_bound:
            src = self.cleaned_data
        else:
            src = {fld.name:fld.field.initial for fld in self}
        p = {
                'ws_id':self.ws.id,
                }
        for f in self:
            key = f.name
            p[key] = src[key]
        return p

class MyJobInfo(JobInfo):
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
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        form = ConfigForm(ws,None,post_data)
        if not form.is_valid():
            return (form.as_html(),None)
        p = form.cleaned_data
        fm = ws.get_feature_matrix(p['fm_code'])
        if fm.target_names != ['False', 'True']:
            raise NotImplementedError(
                    "unsupported target structure: '%s'"%repr(fm.target_names)
                    )
        kts = sum(fm.target)
        kts_needed = 6
        if kts < kts_needed:
            form.add_error(None,
                "Your training set is too small;"
                " you need at least %d examples;"
                " you have %d"
                % (kts_needed,kts)
                )
            return (form.as_html(),None)
        settings = form.as_dict()
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(settings)
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
    def __init__(self,ws=None,job=None):
        # base class init
        self.use_LTS = True
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "ML",
                "Machine Learning Classifier",
                )
        # any base class overrides for unbound instances go here
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.tmpdir = PathHelper.ws_ml_publish(self.ws.id)
            self.fn_arff = self.indir+"full_vector.arff"
            self.fn_druglist = self.indir+"druglist.csv"
            self.fn_case_count = self.indir+"progress_cases"
            self.R_score_name = 'allProbabilityOfAssoications.csv'
            self.fn_scores = self.lts_abs_root+'allPofA.tsv'
            plotdir=self.final_pubdir
            self.publinks.append((None,"cv_stats_boxplot.plotly"))
            self.publinks.append((None,"infoGainBoxplot.plotly"))
            for link,path in (
                    ("ordered probability (top 100)",plotdir+"probabilityOfAssociation_top100.png"),
                    ("ordered probability (all)",plotdir+"probabilityOfAssociation.png"),
                    ("select attributes",plotdir+"attributes_used.txt"),
                    ("precision vs recall",plotdir+"precRecall.png"),
                    ("ROC",plotdir+"roc.png"),
#                    ("Info Gain table",plotdir+"infoGainTable.txt"),
                    ):
                if os.path.isfile(path):
                    self.otherlinks.append( (link,PathHelper.url_of_file(path)) )
    score_by_key=[
            ('wsa','ml','Drug ML'),
            ('uniprot','protml','Protein ML'),
            ]
    def get_data_code_groups(self):
        if self.job:
            # In the bound case, return only the code group for the relevant
            # key; it will access the file and verify that the header is as
            # expected.
            #
            # To do this, we need to know the key. It's simple, but not
            # necessarily fast, to retrieve the FM and obtain it from there.
            # XXX - this can be done quickly with a sequence like:
            # XXX     fm=ws.get_feature_matrix('fvs44588')
            # XXX     specargs=fm.archive.get('specargs').tolist()
            # XXX     spec = DCSpec(**specargs)
            # XXX     codes=spec.get_codes()
            # XXX     cat=spec.get_data_catalog()
            # XXX     sample_key = cat.get_keyname(codes[0])
            # XXX   but this assumes it's a spec-type FM
            # So, get it from the score file.
            try:
                from dtk.readtext import parse_delim
                with open(self.fn_scores) as f:
                    header = next(parse_delim(f))
                    sample_key = header[0]
            except (IOError,StopIteration):
                return []
            to_return = [x for x in self.score_by_key if x[0] == sample_key]
        else:
            # In the unbound case, return all possible scores, all configured
            # for the same filename (the file won't exist, so they'll all be
            # listed).
            to_return = self.score_by_key
        return [
                dc.CodeGroup(key,self._std_fetcher('fn_scores'),
                        dc.Code(score,label=label,
                                meta_out=True,
                                ),
                        )
                for key,score,label in to_return
                ]
    def get_progress(self):
        progress = self._get_progress_parts()
        if len(progress[1]) > 0:
            if progress[1][0][0] == "run weka":
                try:
                    with open(self.fn_case_count) as f:
                        cases = int(f.next().strip())
                    remote_root =  self.mch.get_remote_path(self.root)
                    self.mch.check_remote_cmd("find "
                        + remote_root+"predictions"
                        + " -name '*.tsv'"
                        + " | wc -l"
                        ,hold_output=True)
                    count = int(self.mch.stdout.strip())
                    progress[1][0][1] = str(100*count/cases)+"% complete"
                except:
                    progress[1][0][1] = "???% complete"
        return progress[0]+progress[1]
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.lts_abs_root)
        p_wr = ProgressWriter(self.progress
                , [ "wait for resources"
                  , "setup"
                  , "wait for remote resources"
                  , "initialize remote"
                  , "run weka"
                  , "cleanup"
                  , "check enrichment"
                  ]
                )
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.setup()
        p_wr.put("setup","complete")

        got = self.rm.wait_for_resources(
                            self.job.id,
                            [0,(1,self.parms['inner'])],
                            )
        self.remote_cores = got[1]
        p_wr.put("wait for remote resources","got %d cores" % got[1])
        self.run_weka_remote(p_wr)
        p_wr.put("run weka","complete")

        self.rm.wait_for_resources(self.job.id,[1])
        self.fix_score_file()
        p_wr.put("cleanup","complete")

        self.check_enrichment()
        self.finalize()
        p_wr.put("check enrichment","complete")
    def setup(self):
        label = [
                x[1]
                for x in self.ws.get_feature_matrix_choices()
                if x[0] == self.parms['fm_code']
                ][0]
        fm = self.ws.get_feature_matrix(self.parms['fm_code'])
        # the following is adapted from fm.save_to_arff to add target column
        from dtk.features import ArffDestination,FeatureVector
        attrs = list(zip(fm.feature_names,fm.typenames))
        attrs += [('Target','bool')]
        dest = ArffDestination(
                self.fn_arff,
                label,
                attrs,
                #sparse=True,
                )
        for i,row in enumerate(fm.data):
            dest.append(FeatureVector(row,fm.target[i]))
        # now write key file; this is somewhat mis-named, as the samples are
        # no longer necessarily drugs, and the key no longer necessarily a
        # wsa_id, but leave as-is for now
        from dtk.files import FileDestination
        with FileDestination(self.fn_druglist) as f:
            for wsa_id in fm.sample_keys:
                f.append([wsa_id])
        # stash the appropriate score file header to use in fix_score_file
        matched = [x for x in self.score_by_key if x[0] == fm.sample_key]
        assert len(matched)==1,"score_by_key doesn't handle "+fm.sample_key
        item=matched[0]
        self.out_header = [item[0], item[1]]
    def run_weka_remote(self,p_wr):
        local = False
        if local:
            cvt = lambda x:x
        else:
            cvt = self.mch.get_full_remote_path
            self.copy_input_to_remote()
        # calculate and record progress denominator
        cases = self.parms['inner'] * self.parms['outer']
        self.debug("cases: %d",cases)
        with open(self.fn_case_count,'w') as f:
            f.write(str(cases)+'\n')
        p_wr.put("initialize remote","complete")
        # execute command remotely
        parms = [self.fn_druglist
                ,self.fn_arff
                ,self.root
                ,self.tmp_pubdir
                ]
        parms = [cvt(x) for x in parms]
        parms.append(str(self.ws.id))
        parms.append(str(self.remote_cores))
        parms += ['--inner_iters', str(self.parms['inner'])]
        parms += ['--outer_iters', str(self.parms['outer'])]
        parms += ['--ml_method', str(self.parms['method'])]
        if local:
            print('executing locally; parms',parms)
            import subprocess
            subprocess.check_call('cd '
                    +cvt(PathHelper.MLscripts)
                    +' && ./metaML_wrapper.py '
                    +" ".join(parms)
                    ,shell=True)
            subprocess.check_call('mkdir -p '+cvt(self.tmp_pubdir),shell=True)
            subprocess.check_call('cp'
                    +' '+cvt(self.root+'/PlotsAndFinalPredictions/*.*')
                    +' '+cvt(self.tmp_pubdir)
                    ,shell=True)
            return
        self.mch.check_remote_cmd(
                'cd '
                +cvt(PathHelper.MLscripts)
                +' && ./metaML_wrapper.py '
                +" ".join(parms)
                )
        self.mch.check_remote_cmd('mkdir -p '+cvt(self.tmp_pubdir))
        self.mch.check_remote_cmd('cp'
                +' '+cvt(self.root+'/PlotsAndFinalPredictions/*.*')
                +' '+cvt(self.tmp_pubdir)
                )
        # copy back results
        self.copy_output_from_remote()
    def fix_score_file(self):
        # copy to LTS directory and add header
        from dtk.files import get_file_records,FileDestination
        with FileDestination(self.fn_scores,self.out_header) as dest:
            for rec in get_file_records(self.tmp_pubdir+self.R_score_name):
                dest.append(rec)
    def get_feature_matrix_choices(self):
        return [
                self._get_default_feature_matrix_choice()
                ]
    def get_feature_matrix(self,code):
        assert code == self._get_default_feature_matrix_choice()[0]
        import dtk.features as feat
        return feat.FMBase.load_from_arff(
                self.fn_arff,
                druglist_path=self.fn_druglist,
                )

if __name__ == "__main__":
    MyJobInfo.execute(logger)
