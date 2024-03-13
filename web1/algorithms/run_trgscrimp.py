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
from dtk.prot_map import DpiMapping

import json
import logging
logger = logging.getLogger("algorithms.run_trgscrimp")


class RemoteDataSetup(object):
    def __init__(self, data, indir):
        self.files = []
        self.data = data
        self._walk_data(data)
        self.indir = indir

    def _walk_data(self, data):
        from scripts.extract_importance import FileWrapper
        datatype = type(data)
        if datatype == dict:
            for k, v in data.items():
                self._walk_data(v)
        elif datatype == tuple or datatype == list:
            [self._walk_data(x) for x in data]
        elif datatype == FileWrapper:
            self.files.append(data)

    def replace_filenames(self, cvt):
        local_to_remote_filenames = {}
        for f in self.files:
            if f.filename not in local_to_remote_filenames:
                # Grab the filename and last few directories to identify it.
                basename = '_'.join(f.filename.split(os.path.sep)[-3:])
                unique_name = "%d_%s" % (len(local_to_remote_filenames), basename)
                indir_path = os.path.join(self.indir, unique_name)
                local_to_remote_filenames[f.filename] = indir_path
            else:
                indir_path = local_to_remote_filenames[f.filename]
            f.filename = cvt(indir_path)

        from shutil import copyfile
        for local, remote in local_to_remote_filenames.items():
            print("Copying %s to %s" % (local,remote))
            copyfile(local, remote)

def parse_wsa_list(valstr):
    if not valstr:
        return []
    import re
    wsas = re.split(r'[,\s]+', valstr)
    return [int(x.strip()) for x in wsas if x.strip()]

class MyJobInfo(StdJobInfo):
    descr = "Computes target score importance across multiple drugs."
    short_label = "Target Score Importance"
    page_label = "Target Score Importance"

    def make_job_form(self, ws, data):
        from dtk.target_importance import METHOD_CHOICES
        from dtk.html import MultiWsaField
        class MyForm(StdJobForm):
            wzs_jid = forms.ChoiceField(
                    label='WZS JobID',
                    choices = ws.get_prev_job_choices('wzs'),
                    required=True,
                 )
            method = forms.ChoiceField(
                    label='Target importance scoring method',
                    choices = METHOD_CHOICES,
                    initial='peel_cumulative',
                    required=True,
                 )
            start = forms.IntegerField(
                                label='Initial Drugs to skip',
                                initial=0,
                                )
            count = forms.IntegerField(
                                label='# to examine',
                                initial=200,
                                )
            condensed = forms.BooleanField(
                    label='Count via condensed',
                    initial=True,
                    required=False,
                    )
            indirect_scores = forms.BooleanField(
                                label='Compute only indirect scores',
                                initial=False,
                                required=False,
                                )
            extra_wsas = MultiWsaField(
                    label='Extra WSAs to run',
                    required=False,
                    ws_id=ws.id,
                    initial='',
                    )
            max_input_prots = forms.IntegerField(
                    label='(For Indirect) Maximum # of input prots in jobs to include',
                    required=False,
                    initial=5000,
                    )

        return MyForm(ws, data)

    def __init__(self,ws=None,job=None):
        super().__init__(ws, job, __file__)
    def get_warnings(self):
        return super().get_warnings(
                 ignore_conditions=self.base_warning_ignore_conditions+[
                        # It's worth noting these in the log,
                        # but we at this point it doesn't warrant a UI warning
                         lambda x:'Overriding peel method to LOI for depend job' in x,
                         lambda x:'aborting because we will never converge' in x,
                         lambda x:'aborting to avoid iter size' in x,
                         ],
                )
    def run(self):
        self.make_std_dirs()
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "setup",
                "run score importance",
                "wait for remote resources",
                "remote",
                "load_results",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        local=False
        if local:
            cvt = lambda x:x
        else:
            cvt = self.mch.get_remote_path
        self.setup(cvt)
        p_wr.put("setup","complete")
        self.run_score_importance()
        p_wr.put("run score importance","complete")
        got = self.rm.wait_for_resources(self.job.id,[0, self.remote_cores_wanted], slow=True)
        self.remote_cores_got = got[1]
        p_wr.put("wait for remote resources",
                "complete (%d cores for %d items)" % (self.remote_cores_got, self.num_items))
        self.run_remote(cvt, local)
        p_wr.put("remote","complete")
        self.load_results()
        p_wr.put("load_results","complete")

        # We're not actually pushing any data, so nothing to finalize
        # (trying to push no data throws an exception).
        #self.finalize()



    def setup(self, cvt):
        from dtk.target_importance import TargetScoreImportance
        self.tsi = TargetScoreImportance(self.ws.id,
            self.parms['wzs_jid'],
            wsa_start=self.parms['start'],
            wsa_count=self.parms['count'],
            imp_method=self.parms['method'],
            indirect_scores=self.parms['indirect_scores'],
            extra_wsas=parse_wsa_list(self.parms['extra_wsas']),
            max_input_prots=self.parms['max_input_prots'],
            condensed=self.parms['condensed'],
            )

        self.pieces = self.tsi.make_pieces()
        self.num_items = len(self.pieces)
        print("We have %d pieces" % self.num_items)
        input_data = self.tsi.gen_piece_inputs(self.pieces)


        rds = RemoteDataSetup(input_data, self.indir)
        rds.replace_filenames(cvt)

        self.infile = os.path.join(self.indir, 'in.pickle')
        with open(self.infile, 'wb') as f:
            import pickle
            f.write(pickle.dumps(rds.data))

        self.remote_cores_wanted=(10, self.num_items)
        print(("Requesting remote cores", self.remote_cores_wanted))

    def run_score_importance(self):
        """Computes and caches score importance values.

        We don't really care about the results right now, we just want
        the cache to get populated to make viewing this data faster.
        """
        wsa_ids = self.tsi.get_wsa_ids()
        from dtk.score_importance import ScoreImportance
        si = ScoreImportance(self.ws.id, self.parms['wzs_jid'])
        print("Fetching score weights")
        si.get_score_importances(wsa_ids)

    def run_remote(self, cvt, local):
        options = [
                    cvt(self.infile),
                    cvt(self.outdir),
                    str(self.remote_cores_got)
                   ]

        print(('command options',options))
        rem_cmd = cvt(
                os.path.join(PathHelper.website_root, "scripts", "trgscrimp.py")
                )
        if local:
            import subprocess
            subprocess.check_call([rem_cmd]+options)
            return
        self.copy_input_to_remote()
        self.make_remote_directories([
                                self.tmp_pubdir,
                                ])
        self.mch.check_remote_cmd(' '.join([rem_cmd]+options))
        self.copy_output_from_remote()

    def load_results(self):
        outfile = os.path.join(self.outdir, 'output.pickle.gz')
        import gzip
        with gzip.open(outfile, 'rb') as f:
            import pickle
            data = pickle.load(f)
        self.tsi.load_scores(self.pieces, data)


if __name__ == "__main__":
    MyJobInfo.execute(logger)
