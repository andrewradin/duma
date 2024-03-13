#!/usr/bin/env python3

import os
import django_setup
from django import forms

from tools import ProgressWriter
from runner.process_info import StdJobInfo, StdJobForm, JobInfo
import runner.data_catalog as dc

import logging
logger = logging.getLogger(__name__)


def extract_jids_and_codes(jcs):
    jids_and_codes = []
    jobs_and_codes = jcs.split('|')
    for jid_code in jobs_and_codes:
        jid, code = jid_code.split('_')
        jids_and_codes.append((jid, code))
    return jids_and_codes

class MyJobInfo(StdJobInfo):
    descr= "Given a set of job IDs reports WSAs' average rank percentile"
    short_label = 'AvgPercRank'
    page_label = 'Average Percentile Rank'

    def make_job_form(self, ws, data):

        class MyForm(StdJobForm):
            jids=forms.CharField(
                    label='JobID_code',
                    required=True,
                 )
        return MyForm(ws, data)

    def get_data_code_groups(self):
        codetype = 'wsa'
        return [
                dc.CodeGroup(codetype,self._std_fetcher('outfile'),
                        dc.Code('avg',
                                    label='APR_avg',
                                    fmt='%0.2f',
                                    ),
                        dc.Code('med',
                                    label='APR_med',
                                    fmt='%0.2f',
                                    ),
                        ),
                ]

    def __init__(self,ws=None,job=None):
        super().__init__(ws=ws, job=job, src=__file__)
        if self.job:
            self.outfile = os.path.join(self.lts_abs_root, 'output.tsv')

    def run(self):
        self.make_std_dirs()
        p_wr = ProgressWriter(self.progress, [
                "compute scores",
                "finalize",
                ])
        self.compute_scores()
        p_wr.put("compute scores","complete")
        self.finalize()
        p_wr.put("finalize","complete")

    def compute_scores(self):
        from runner.process_info import JobInfo
        from dtk.scores import Ranker
        from statistics import median
        wsas_to_perc_list = {}
        jids_and_codes = extract_jids_and_codes(self.parms['jids'])
        # for each get the ordering, then ranker, then convert to a percentile
        for j,c in jids_and_codes:
            bji = JobInfo.get_bound(self.ws,j)
            cat = bji.get_data_catalog()
            ord=cat.get_ordering(c, True)
            rankr = Ranker(ord)
            wsas=[t[0] for t in ord]
            ranks = rankr.get_all(wsas)
            high_val = max(ranks)
            percs = [1. - ((r - 1.)/high_val) for r in ranks]
            wsa_percs = zip(wsas, percs)
            for k, v in wsa_percs:
                if k not in wsas_to_perc_list:
                    wsas_to_perc_list[k]=[]
                wsas_to_perc_list[k].append(v)
        final_data = {}
        for wsa,l in wsas_to_perc_list.items():
            # add implicit zeros if the WSAs from the jobs didn't totally agree
            len_to_add = len(jids_and_codes) - len(l)
            if len_to_add:
                l += [0.]*len_to_add
            final_data[wsa] = [sum(l)/float(len(l)), median(l)]
# then aggregate with median or average
        from atomicwrites import atomic_write
        with atomic_write(self.outfile) as f:
            f.write('\t'.join(['wsa', 'avg', 'med']) + '\n')
# should be sorting by median
            for prot, tup in sorted(final_data.items(), key=lambda x: (-x[1][1], x[0])):
                row = [str(x) for x in [prot, tup[0], tup[1]]]
                f.write('\t'.join(row) + '\n')

if __name__ == "__main__":
    MyJobInfo.execute(logger)
