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
logger = logging.getLogger("algorithms.run_gsearchmodel")


def workspace_choices():
    def ws_exclude(ws):
        name = ws.name.lower()
        bad_words = ['qa', 'combo base', 'outdated']
        for bad_word in bad_words:
            if bad_word in name:
                return True
        return False
    
    def ws_label(ws, imported):
        active_str = 'active, ' if ws.active else ''
        return f'{ws.name} ({active_str}{imported} imported)'
    from browse.models import Workspace
    good_ws = Workspace.objects.exclude(tissueset__isnull=True)
    imported = [len(x.imported_geo_ids()) for x in good_ws]
    ws_choices = [(ws.id, ws_label(ws, imported))
                  for ws, imported in sorted(zip(good_ws, imported), key=lambda x: (not x[0].active, -x[1]))
                  if imported > 0 and not ws_exclude(ws)
                  ]
    return ws_choices

class MyJobInfo(StdJobInfo):
    short_label = 'GE Search Model'
    page_label = 'GE Search Model'
    descr = 'Trains the gesearch model and saves it, for use in GE/Omics search.'
    def make_job_form(self, ws, data):
        ws_choices = workspace_choices()
        model_choices = [
            ('bagofwords', 'Bag of Words'),
            ('previous', 'Pre-existing AESearch (eval only)'),
        ]
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

            model = forms.ChoiceField(
                    label='Model',
                    choices=model_choices,
                    required=True
                    )
            kfold = forms.IntegerField(
                    label='Validation Folds',
                    initial=4,
                    )
            hyperopt_iters = forms.IntegerField(
                    label='Hyperparameter Opt Iters',
                    initial=500,
                    )

        return MyForm(ws, data)


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
            url = self.ws.reverse('ge:searchmodel', self.job.id)
            self.otherlinks.append(('Model Information', url))
            self.plot_prefix = os.path.join(self.tmp_pubdir, 'plot')
            self.log_prefix = str(self.ws)+":"
            self.outfile = os.path.join(self.lts_abs_root, 'model.json')
            if os.path.exists(self.final_pubdir):
                pubs = os.listdir(self.final_pubdir)
                plots = [x for x in pubs if x.endswith('.plotly') or x.endswith('.plotly.gz')]
                self.publinks.extend([(None, plot) for plot in plots])
            self.publinks = tuple(self.publinks)
            self.xval_fn = os.path.join(self.lts_abs_root, 'xval.pickle.gz')
            self.in_dataset_fn = os.path.join(self.indir, 'dataset.pickle.gz')
            self.tmp_results = os.path.join(self.outdir, 'results.pickle.gz')

    def run(self):
        self.make_std_dirs()
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "hyperopt",
                "analyze",
                "finalize",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.hyperopt()
        self.load_results()
        p_wr.put("hyperopt","complete")
        self.analyze()
        p_wr.put("analyze","complete")
        self.finalize()
        p_wr.put("finalize","complete")

        
    def analyze(self):
        all_rocs = []
        xval_results = self.results['xval']
        for fold, result in enumerate(xval_results):
            all_rocs.append(result['metrics']['roc'])
            for name, plot in result.get('plots', {}).items():
                fn = f'{name}.{fold}.plotly'
                fn = os.path.join(self.tmp_pubdir, fn)
                logger.info(f"Saving plot at {fn}")
                plot.save(fn)

        import numpy as np
        self.report_info(f"XVal Mean ROC: {np.mean(all_rocs):.3f}")

        import gzip
        with gzip.open(self.xval_fn, 'wb') as f:
            import pickle
            pickle.dump(xval_results, f)

        # Save model and try loading it.
        model_data = self.results['model']
        with open(self.outfile, 'w') as f:
            import json
            f.write(json.dumps(model_data))

        model = self.load_trained_model()
        rating = model.run_eval(self.dataset)
        self.report_info(f"Train (self-fit) ROC score: {rating:.3f}")

    def load_xval_data(self):
        import gzip
        with gzip.open(self.xval_fn, 'rb') as f:
            import pickle
            return pickle.loads(f.read())

    def hyperopt(self):
        from scripts.gesearch_model import build_dataset
        import gzip
        dataset = build_dataset(self.parms['workspaces'])
        self.dataset = dataset
        with gzip.open(self.in_dataset_fn, 'wb') as f:
            import pickle
            pickle.dump({
                'settings': self.parms,
                'dataset': dataset,
            }, f)

        cvt = self.mch.get_remote_path
        options = ['-i', cvt(self.in_dataset_fn),
                   '-o', cvt(self.tmp_results),
                  ]
        rem_cmd = cvt(
                os.path.join(PathHelper.website_root, "scripts", "gesearch_model.py")
                )
        self.copy_input_to_remote()
        self.make_remote_directories([
                                self.tmp_pubdir,
                                ])
        self.mch.check_remote_cmd(' '.join([rem_cmd]+options))
        self.copy_output_from_remote()


    def load_results(self):
        import isal.igzip as gzip
        with gzip.open(self.tmp_results, 'rb') as f:
            import pickle
            self.results = pickle.loads(f.read())

    def train(self):
        from scripts.gesearch_model import train
        logger.info("Training")
        ws_ids = self.parms['workspaces']

        model, dataset = train(ws_ids, self.parms)
        model.save(self.outfile)

        # Verify we can reload the model and get reasonable results.
        model = self.load_trained_model()
        rating = model.run_eval(dataset)
        self.report_info(f"Train (self-fit) ROC score: {rating:.3f}")


    def load_trained_model(self):
        self.fetch_lts_data()
        from scripts.gesearch_model import load_model
        return load_model(self.parms, self.outfile)

    def get_jobnames(self,ws):
        """Overridden, name doesn't change per-workspace"""
        return ['gesearchmodel']


if __name__ == "__main__":
    MyJobInfo.execute(logger)
