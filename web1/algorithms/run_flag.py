#!/usr/bin/env python3

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

import logging
logger = logging.getLogger("algorithms.run_flag")

from django import forms

import json
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo

class MyJobInfo(JobInfo):
    def description_text(self):
        return '''
        These serve to surface likely important data about a drug (commercial 
        availability, abundance of scores from non-novel proteins, etc.) that may 
        lead a reviewer to remove it. This still allows a human to review the 
        information and eliminate the drug. (As opposed to a filter which would 
        remove the drug without ever showing it to a human.)
        '''
    def get_config_form_class(self,ws):
        # To add more scripts:
        # - pick a stepname for the script; a good choice is the
        #   unique part of the script name: flag_drugs_for_<stepname>.py
        # - add a flag boolean called flag_<stepname> to the form below
        # - add any other parameters special to the script immediately
        #   beneath the flag parameter. Labels should begin with '...'
        #   and the internal field names should begin with the stepname.
        # - add the name to the 'scripts' array below
        # - add a do_<stepname> method that either executes the script
        #   in subprocess, or imports the underlying class and instantiates
        #   and runs it; this method should return a string to be displayed
        #   on the progress page
        class ConfigForm(forms.Form):
            job_id = forms.IntegerField(
                        label='Drug Ordering Job',
                        )
            score = forms.CharField(
                        label='Drug Ordering Score',
                        )
            start = forms.IntegerField(
                        label='Initial Drugs to skip',
                        initial=0,
                        )
            count = forms.IntegerField(
                        label='Drugs to examine',
                        initial=200,
                        )
            condensed = forms.BooleanField(
                    label='Count via condensed',
                    initial=True,
                    required=False,
                    )
            flag_demerits = forms.BooleanField(
                        initial=True,
                        required=False,
                        )
            flag_ncats = forms.BooleanField(
                        initial=True,
                        required=False,
                        )
            flag_zinc = forms.BooleanField(
                        initial=True,
                        required=False,
                        )
            flag_shadows = forms.BooleanField(
                        initial=True,
                        required=False,
                        )
            flag_previous_targets = forms.BooleanField(
                        initial=True,
                        required=False,
                        )
            from dtk.prot_map import DpiMapping
            previous_targets_dpi = forms.ChoiceField(
                        label='...DPI mapping',
                        choices=DpiMapping.choices(ws),
                        initial=ws.get_dpi_default(),
                        )
            previous_targets_dpi_threshold = forms.FloatField(
                        label='...DPI evidence threshold',
                        initial=ws.get_dpi_thresh_default(),
                        )
            flag_review_similarity = forms.BooleanField(
                        initial=True,
                        required=False,
                        )
            review_similarity_dpi = forms.ChoiceField(
                        label='...DPI mapping',
                        choices=DpiMapping.choices(ws),
                        initial=ws.get_dpi_default(),
                        )
            review_similarity_dpi_threshold = forms.FloatField(
                        label='...DPI evidence threshold',
                        initial=ws.get_dpi_thresh_default(),
                        )
            from dtk.prot_map import PpiMapping
            review_similarity_ppi = forms.ChoiceField(
                        label='...PPI mapping',
                        choices=PpiMapping.choices(),
                        initial=ws.get_ppi_default(),
                        )
            review_similarity_ppi_threshold = forms.FloatField(
                        label='...PPI evidence threshold',
                        initial=ws.get_ppi_thresh_default(),
                        )
            review_similarity_method = forms.ChoiceField(
                        label='...Clustering method',
                        choices=[
                                ('AP','Affinity Propagation'),
                                ('ST','Similarity Threshold'),
                                ],
                        initial='ST',
                        )
            review_similarity_st_dir_thresh = forms.FloatField(
                        label='...ST direct cutoff',
                        initial=0.7,
                        )
            review_similarity_st_ind_thresh = forms.FloatField(
                        label='...ST indirect cutoff',
                        initial=0.2,
                        )
            review_similarity_repulsion = forms.FloatField(
                        label='...AP clustering threshold',
                        initial=0.5,
                        )
            review_similarity_damping = forms.FloatField(
                        label='...AP damping',
                        initial=0.8,
                        )
            review_similarity_max_iter = forms.IntegerField(
                        label='...AP max iterations',
                        initial=1000,
                        )
            flag_unwanted_targets = forms.BooleanField(
                        initial=True,
                        required=False,
                        )
            unwanted_targets_uniprot_set = forms.ChoiceField(
                        label='Intolerable uniprots set',
                        choices = ws.get_uniprot_set_choices(),
                        initial = ws.get_intolerable_ps_default(),
                        required=False,
                        )
            unwanted_targets_additional_uniprots = forms.CharField(
                        label='...Additional uniprots (blank separated)',
                        required=False
                        )
            unwanted_targets_dpi = forms.ChoiceField(
                        label='...DPI mapping',
                        choices=DpiMapping.choices(ws),
                        initial=ws.get_dpi_default(),
                        )
            unwanted_targets_threshold = forms.FloatField(
                        label='...DPI evidence threshold',
                        initial=ws.get_dpi_thresh_default(),
                        )
            flag_unwanted_important_targets = forms.BooleanField(
                        initial=True,
                        required=False,
                        )
            unwanted_important_targets_uniprot_set = forms.ChoiceField(
                        label='...Additional uniprot set',
                        choices = ws.get_uniprot_set_choices(),
                        initial=ws.get_nonnovel_ps_default(),
                        required=False,
                        )
            unwanted_important_targets_additional_uniprots = forms.CharField(
                        label='...Additional uniprots (blank separated)',
                        required=False
                        )
            unwanted_important_targets_threshold = forms.FloatField(
                        label='...Importance threshold',
                        initial=0.8,
                        )
            unwanted_important_targets_set_indication = forms.BooleanField(
                        label='...Mark unclassified as inactive',
                        initial=True,
                        required=False
                        )
            flag_availability = forms.BooleanField(
                        initial=False,
                        required=False,
                        )
            flag_orange_book_patents = forms.BooleanField(
                        initial=False,
                        required=False,
                        )
            flag_novelty = forms.BooleanField(
                        initial=False,
                        required=False,
                        )
            novelty_alpha = forms.FloatField(
                        label='...significance level',
                        initial=0.05,
                        )
            novelty_thresh = forms.FloatField(
                        label='...min log odds ratio',
                        initial=-1.0,
                        )
            flag_no_targets = forms.BooleanField(
                        initial=True,
                        required=False,
                        )
            no_targets_dpi = forms.ChoiceField(
                        label='...DPI mapping',
                        choices=DpiMapping.choices(ws),
                        initial=ws.get_dpi_default(),
                        )
            no_targets_threshold = forms.FloatField(
                        label='...DPI evidence threshold',
                        initial=ws.get_dpi_thresh_default(),
                        )
            flag_commercial_db = forms.BooleanField(
                        initial=True,
                        required=False,
                        )
            flag_score_quality = forms.BooleanField(
                        initial=True,
                        required=False,
                        )
            sq_pathway_threshold = forms.FloatField(
                        label='...Pathways-only threshold',
                        initial=0.95,
                        )
            sq_single_source_threshold = forms.FloatField(
                        label='...Single source threshold',
                        initial=0.95,
                        )
            sq_known_drug_threshold = forms.FloatField(
                        label='...Known drug threshold',
                        initial=0.75,
                        )
            sq_weak_targets_threshold = forms.FloatField(
                        label='...Weak targets threshold',
                        initial=0.2,
                        )
        return ConfigForm
    def settings_defaults(self,ws):
        from runner.process_info import form2dict
        cfg=self.get_config_form_class(ws)()
        return {
                'default':form2dict(cfg,ws_id=ws.id),
                }
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        ConfigForm = self.get_config_form_class(ws)
        if copy_job:
            form = ConfigForm(initial=copy_job.settings())
        else:
            form = ConfigForm()
        from dtk.html import join,bulk_update_links
        return join(
                bulk_update_links('flag_'),
                form.as_p(),
                )
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        ConfigForm = self.get_config_form_class(ws)
        form = ConfigForm(post_data)
        if not form.is_valid():
            return (form.as_p(),None)
        settings = dict(form.cleaned_data)
        settings['ws_id'] = ws.id
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(settings)
                            )
        next_url = ws.reverse('nav_progress',job_id)
        return (None, next_url)
    def __init__(self,ws=None,job=None):
        # base class init
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "Flagging",
                "Run Flagging Scripts",
                )
        # any base class overrides for unbound instances go here
        # TODO: self.publinks
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            # TODO: calculate paths to individual files
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        scripts=[
                'demerits',
                'ncats',
                'zinc',
                'shadows',
                'previous_targets',
                'review_similarity',
                'unwanted_targets',
                'unwanted_important_targets',
                'availability',
                'orange_book_patents',
                'novelty',
                'no_targets',
                'commercial_db',
                'score_quality',
                ]
        stepnames=[
                x.replace('_',' ')
                for x in scripts
                ]
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                ]+stepnames)
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        for step,stepname in zip(scripts,stepnames):
            if self.parms.get('flag_'+step):
                self.divider('starting '+step)
                func = getattr(self,'do_'+step)
                status = func()
            else:
                status = 'N/A'
            p_wr.put(stepname,status)
    def do_novelty(self):
        from scripts.flag_drugs_for_novelty import NoveltyFlagger
        flagger = NoveltyFlagger(
                ws_id=self.parms['ws_id'],
                job_id=self.parms['job_id'],
                score=self.parms['score'],
                start=self.parms['start'],
                count=self.parms['count'],
                alpha=self.parms['novelty_alpha'],
                thresh=self.parms['novelty_thresh'],
                condensed=self.parms['condensed'],
                )
        flagger.flag_drugs()
        return "complete"
    def do_previous_targets(self):
        from scripts.flag_drugs_for_previous_targets import PreviousTargetsFlagger
        flagger = PreviousTargetsFlagger(
                ws_id=self.parms['ws_id'],
                job_id=self.parms['job_id'],
                score=self.parms['score'],
                start=self.parms['start'],
                count=self.parms['count'],
                dpi=self.parms['previous_targets_dpi'],
                dpi_threshold=self.parms['previous_targets_dpi_threshold'],
                condensed=self.parms['condensed'],
                )
        flagger.flag_drugs()
        return "complete"
    def do_availability(self):
        from scripts.flag_drugs_for_availability import Flagger
        flagger = Flagger(
                ws_id=self.parms['ws_id'],
                job_id=self.parms['job_id'],
                score=self.parms['score'],
                start=self.parms['start'],
                count=self.parms['count'],
                condensed=self.parms['condensed'],
                )
        flagger.flag_drugs()
        return "complete"
    def do_demerits(self):
        from scripts.flag_drugs_for_demerits import DemeritFlagger
        flagger = DemeritFlagger(
                ws_id=self.parms['ws_id'],
                job_id=self.parms['job_id'],
                score=self.parms['score'],
                start=self.parms['start'],
                count=self.parms['count'],
                condensed=self.parms['condensed'],
                )
        flagger.flag_drugs()
        return "complete"
    def do_ncats(self):
        from scripts.flag_drugs_for_ncats import NcatsFlagger
        flagger = NcatsFlagger(
                ws_id=self.parms['ws_id'],
                job_id=self.parms['job_id'],
                score=self.parms['score'],
                start=self.parms['start'],
                count=self.parms['count'],
                condensed=self.parms['condensed'],
                )
        flagger.flag_drugs()
        return "complete"
    def do_zinc(self):
        from scripts.flag_drugs_for_zinc import ZincFlagger
        flagger = ZincFlagger(
                ws_id=self.parms['ws_id'],
                job_id=self.parms['job_id'],
                score=self.parms['score'],
                start=self.parms['start'],
                count=self.parms['count'],
                condensed=self.parms['condensed'],
                )
        flagger.flag_drugs()
        return "complete"
    def do_shadows(self):
        from scripts.flag_drugs_for_shadows import ShadowFlagger
        flagger = ShadowFlagger(
                ws_id=self.parms['ws_id'],
                job_id=self.parms['job_id'],
                score=self.parms['score'],
                start=self.parms['start'],
                count=self.parms['count'],
                condensed=self.parms['condensed'],
                )
        flagger.flag_drugs()
        return "complete"
    def do_orange_book_patents(self):
        from scripts.flag_drugs_for_orange_book_patents import PatentFlagger
        flagger = PatentFlagger(
                ws_id=self.parms['ws_id'],
                job_id=self.parms['job_id'],
                score=self.parms['score'],
                start=self.parms['start'],
                count=self.parms['count'],
                condensed=self.parms['condensed'],
                )
        flagger.flag_drugs()
        return "complete"
    def do_review_similarity(self):
        from scripts.flag_drugs_for_review_similarity import Flagger
        flagger = Flagger(
                ws_id=self.parms['ws_id'],
                job_id=self.parms['job_id'],
                score=self.parms['score'],
                start=self.parms['start'],
                count=self.parms['count'],
                dpi=self.parms['review_similarity_dpi'],
                dpi_threshold=self.parms['review_similarity_dpi_threshold'],
                ppi=self.parms['review_similarity_ppi'],
                ppi_threshold=self.parms['review_similarity_ppi_threshold'],
                repulsion=self.parms['review_similarity_repulsion'],
                damping=self.parms['review_similarity_damping'],
                max_iter=self.parms['review_similarity_max_iter'],
                method=self.parms['review_similarity_method'],
                st_dir_thresh=self.parms['review_similarity_st_dir_thresh'],
                st_ind_thresh=self.parms['review_similarity_st_ind_thresh'],
                condensed=self.parms['condensed'],
                )
        flagger.flag_drugs()
        return "complete"
    def do_unwanted_targets(self):
        from scripts.flag_drugs_for_dpi import DpiFlagger
        uniprots = []
        explicits = self.parms['unwanted_targets_additional_uniprots']
        if explicits:
            uniprots += explicits.split()
        protset = self.parms['unwanted_targets_uniprot_set']
        if protset:
            uniprots += list(self.ws.get_uniprot_set(protset))
        flagger = DpiFlagger(
                ws_id=self.parms['ws_id'],
                job_id=self.parms['job_id'],
                score=self.parms['score'],
                start=self.parms['start'],
                count=self.parms['count'],
                uniprots=uniprots,
                dpi=self.parms['unwanted_targets_dpi'],
                threshold=self.parms['unwanted_targets_threshold'],
                condensed=self.parms['condensed'],
                )
        flagger.flag_drugs()
        return "complete"
    def do_unwanted_important_targets(self):
        from scripts.flag_drugs_for_prot_importance import ProtImportanceFlagger
        uniprots = []
        explicits = self.parms['unwanted_important_targets_additional_uniprots']
        if explicits:
            uniprots += explicits.split()
        protset = self.parms['unwanted_important_targets_uniprot_set']
        if protset:
            uniprots += list(self.ws.get_uniprot_set(protset))
        flagger = ProtImportanceFlagger(
                ws_id=self.parms['ws_id'],
                job_id=self.parms['job_id'],
                score=self.parms['score'],
                start=self.parms['start'],
                count=self.parms['count'],
                uniprots=uniprots,
                threshold=self.parms['unwanted_important_targets_threshold'],
                set_indication=self.parms['unwanted_important_targets_set_indication'],
                condensed=self.parms['condensed'],
                )
        flagger.flag_drugs()
        return "complete"
    def do_no_targets(self):
        from scripts.flag_drugs_for_no_targets import NoTargetsFlagger
        flagger = NoTargetsFlagger(
                ws_id=self.parms['ws_id'],
                job_id=self.parms['job_id'],
                score=self.parms['score'],
                start=self.parms['start'],
                count=self.parms['count'],
                dpi=self.parms['no_targets_dpi'],
                threshold=self.parms['no_targets_threshold'],
                condensed=self.parms['condensed'],
                )
        flagger.flag_drugs()
        return "complete"
    def do_commercial_db(self):
        from scripts.flag_drugs_for_commercial_db import CommercialDbFlagger
        flagger = CommercialDbFlagger(
                ws_id=self.parms['ws_id'],
                job_id=self.parms['job_id'],
                score=self.parms['score'],
                start=self.parms['start'],
                count=self.parms['count'],
                condensed=self.parms['condensed'],
                )
        flagger.flag_drugs()
        return "complete"
    def do_score_quality(self):
        from scripts.flag_drugs_for_score_quality import ScoreQualityFlagger 
        flagger = ScoreQualityFlagger(
                ws_id=self.parms['ws_id'],
                job_id=self.parms['job_id'],
                score=self.parms['score'],
                start=self.parms['start'],
                count=self.parms['count'],
                pathway_threshold=self.parms['sq_pathway_threshold'],
                single_source_threshold=self.parms['sq_single_source_threshold'],
                known_drug_threshold=self.parms['sq_known_drug_threshold'],
                weak_targets_threshold=self.parms['sq_weak_targets_threshold'],
                condensed=self.parms['condensed'],
                )
        flagger.flag_drugs()
        return "complete"
        

if __name__ == "__main__":
    MyJobInfo.execute(logger)
