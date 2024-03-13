#!/usr/bin/env python3

import sys
from path_helper import PathHelper,make_directory

import os
import django
import django_setup

from django import forms

from tools import ProgressWriter
from runner.process_info import JobInfo, StdJobInfo, StdJobForm
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager
import runner.data_catalog as dc
from dtk.prot_map import DpiMapping, PpiMapping

import json
import logging
logger = logging.getLogger("algorithms.run_selectability")

from dtk.selectability import INPUT_FEATURESETS as featuresets
from dtk.selectability import IndicationFeatureSet


def workspace_choices():
    label_featuresets = featuresets + [IndicationFeatureSet]
    def ws_is_good(ws):
        for featureset in featuresets:
            if not featureset.available(ws):
                return False
        return True
    from browse.models import Workspace
    all_ws = Workspace.objects.all()
    good_ws = [ws for ws in all_ws if ws_is_good(ws)]
    ws_choices = [(ws.id, ws.name) for ws in good_ws]
    return ws_choices

class MyJobInfo(StdJobInfo):
    short_label = 'Selectability Model'
    page_label = 'Selectability Model'
    descr = 'Trains the selectability model and saves it, for use in run_selectability.'
    def make_job_form(self, job_ws, data):
        ws_choices = workspace_choices()
        from browse.models import Workspace
        all_ws = Workspace.objects.all()

        ws_avail_table = []
        for ws in all_ws:
            avails = [fs.available(ws) for fs in featuresets]
            count = sum(avails)
            row = [ws.name, count]
            row += ['&#10004;' if x else '' for x in avails]
            ws_avail_table.append(row)
        ws_avail_cols = ['Workspace', '#'] + [fs.name() for fs in featuresets]
        import json
        ws_avail_table_data = json.dumps(ws_avail_table)
        ws_avail_table_cols = json.dumps([{'title': x} for x in ws_avail_cols])

        from dtk.html import WrappingCheckboxSelectMultiple
        class MyForm(StdJobForm):
            workspaces = forms.MultipleChoiceField(
                    label='Workspaces',
                    choices=ws_choices,
                    required=True,
                    initial=[x[0] for x in ws_choices],
                    widget=forms.SelectMultiple(
                            attrs={'size':20}
                            ),
                    )
            kfold = forms.IntegerField(
                    label='Validation Folds',
                    initial=3,
                    )

            def as_html(self):
                from django.utils.html import format_html, mark_safe
                return format_html('''
                        <table>{}</table>
                        <hr/>
                        <div class='panel panel-info' style='overflow-x:auto'>
                            <div class='panel-heading panel-title' data-toggle='collapse' href='#panel-body' style='cursor:pointer'><small>&#9660;</small>Workspace Feature Availability</div>
                            <div id='panel-body' class='panel-body panel-collapse collapse'>
                            <table id='ws-table' style='text-align:center' class='table table-condensed'>
                            </table>

                            </div>
                        </div>
                        <script>
                            $('#ws-table').DataTable({{
                                data: {},
                                columns: {},
                                order: [[ 1, 'desc' ]],
                            }});
                        </script>
                        '''
                        ,self.as_table()
                        ,mark_safe(ws_avail_table_data)
                        ,mark_safe(ws_avail_table_cols)
                        )
        return MyForm(job_ws, data)


    def __init__(self,ws=None,job=None):
        # base class init
        if job:
            from browse.models import Workspace
            ws = Workspace.objects.get(pk=job.settings()['ws_id'])
            assert ws
        super().__init__(ws, job, __file__)
        # job-specific properties
        if self.job:
            from django.urls import reverse
            url = reverse('selectabilitymodel', args=[self.job.id])
            self.otherlinks.append(('Model Information', url))
            self.plot_prefix = os.path.join(self.tmp_pubdir, 'plot')
            self.log_prefix = str(self.ws)+":"
            self.outfile = os.path.join(self.lts_abs_root, 'model.json')
            if os.path.exists(self.final_pubdir):
                pubs = os.listdir(self.final_pubdir)
                plots = [x for x in pubs if x.endswith('.plotly') or x.endswith('.plotly.gz')]
                self.publinks.extend([(None, plot) for plot in plots])
            self.publinks = tuple(self.publinks)
            self.xval_fn = os.path.join(self.lts_abs_root, 'xval.pickle')

    def run(self):
        self.make_std_dirs()
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "full train",
                "cross-validate",
                "finalize",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")

        from dtk.selectability import instantiate_featuresets
        self.featuresets = instantiate_featuresets(featuresets)

        # We run full training first, because it will refresh all the caches.
        # This ensures the eval part of cross_validate uses all new data.
        self.train()
        p_wr.put("full train","complete")
        self.cross_validate()
        p_wr.put("cross-validate","complete")
        self.finalize()
        p_wr.put("finalize","complete")

    def cross_validate(self):
        k = self.parms['kfold']
        if k < 2:
            print("No validation, too few folds")
            return

        from dtk.selectability import Selectability, WsaTrainingSource, cross_validate, plot_cross_validation
        from sklearn.metrics import roc_curve
        from dtk.plot import scatter2d, annotations
        from browse.models import Workspace, WsAnnotation
        print("Starting cross-validation")
        ws_ids = self.parms['workspaces']
        workspaces = Workspace.objects.filter(pk__in=ws_ids)
        ivals = WsAnnotation.indication_vals
        labelset = IndicationFeatureSet(min_indication=ivals.INITIAL_PREDICTION)
        evalsets = [
            IndicationFeatureSet(min_indication=ivals.INITIAL_PREDICTION),
            IndicationFeatureSet(min_indication=ivals.REVIEWED_PREDICTION),
            IndicationFeatureSet(min_indication=ivals.HIT),
            ]
        datas = cross_validate(WsaTrainingSource(), workspaces, self.featuresets, labelset, evalsets, k=k)
        print("Plotting cross-validation")
        plots = plot_cross_validation(datas)
        for i, plot in enumerate(plots):
            fn = "%s-%d.plotly" % (self.plot_prefix, i)
            plot.save(fn)

        with open(self.xval_fn, 'wb') as f:
            import pickle
            f.write(pickle.dumps(datas))

    def load_xval_data(self):
        with open(self.xval_fn, 'rb') as f:
            import pickle
            return pickle.loads(f.read())


    def train(self):
        from dtk.selectability import Selectability, WsaTrainingSource
        from browse.models import Workspace, WsAnnotation
        s = Selectability()
        ws_ids = self.parms['workspaces']
        print("Using workspaces ", ws_ids)
        workspaces = Workspace.objects.filter(pk__in=ws_ids)
        ivals = WsAnnotation.indication_vals
        labelset = IndicationFeatureSet(min_indication=ivals.INITIAL_PREDICTION)
        model = s.train(WsaTrainingSource(), workspaces, self.featuresets, labelset)

        model.save(self.outfile)

    def load_trained_model(self):
        self.fetch_lts_data()
        from dtk.selectability import MLModel
        return MLModel.from_file(self.outfile)

    def get_jobnames(self,ws):
        """Overridden, name doesn't change per-workspace"""
        return ['selectabilitymodel']


if __name__ == "__main__":
    MyJobInfo.execute(logger)
