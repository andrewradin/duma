#!/usr/bin/env python

import sys
try:
    import path_helper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")

import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")

import django
django.setup()

#-------------------------------------------------------------------------------
# weka wrapper
#-------------------------------------------------------------------------------
class modelBuilder:
    '''Using the ARFF produced by build.py, build and save a ML model
    '''
    def __init__(self, **kwargs):
        self.arff_file=kwargs.get('ifile', None)
        self.ofp=kwargs.get('ofile_prefix', None)
        self.method=kwargs.get('method', 'attrSel_lr')
    def run(self):
        try:
            import run_eval_weka as rew
        except ImportError:
            sys.path.insert(1, (sys.path[0] or '.')+"/../../web1/ML")
            import run_eval_weka as rew
        self._setup()
        # then use that arff to build a model
        weka_model_out = rew.runWeka(self.method, self.arff_file, None, model_name=self.model_file,
                                     testing = True, build = True, cost_list = self.cost_list
                                     )
        rew.analyze_report_weka(weka_model_out, True, False, None, False, self.ofp, write_stats=True)
        self._get_attributes()
        self._save_model_data()
    def _get_attributes(self):
        self.used_attrs = []
        with open(self.arff_file, 'r') as f:
            for l in f:
                if not l.startswith('@ATTRIBUTE'):
                    continue
                term = l.split('"')[1]
                if term != 'DumaReady':
                   self.used_attrs.append(term)
    def _save_model_data(self):
        import pickle
        with open(self.ofp+'.pickle', 'wb') as handle:
            pickle.dump([self.cost_list,
                         self.method,
                         self.model_file,
                         self.used_attrs], handle)
    def _setup(self):
        self.model_file = self.ofp + '.model'

        if self.method == 'RF_weight':
            self.cost_list = [0,
                              len([v for v in self.drug_labels.values()
                                   if v == 'False']),
                              len([v for v in self.drug_labels.values()
                                   if v == 'True']),
                              0]
        else:
            self.cost_list = None

#-------------------------------------------------------------------------------
# Driver
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                description='build ML model',
                )
    parser.add_argument("i", help="ARFF file from build.py")
    parser.add_argument("o", help="Outfile prefix")
    args = parser.parse_args()

    b = modelBuilder(ifile=args.i, ofile_prefix=args.o)
    b.run()
