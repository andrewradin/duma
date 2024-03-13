#!/usr/bin/env python3

from __future__ import print_function

import sys
import six
try:
    from path_helper import PathHelper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")
    from path_helper import PathHelper

import os
import django

if not "DJANGO_SETTINGS_MODULE" in os.environ:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    django.setup()

def warning(*args):
    print(*args,file=sys.stderr)

def show_list(label,iterable):
    l = sorted(iterable)
    if l:
        print(label+':',' '.join(str(x) for x in l))

# Notes:
# - multi_refresh.py outputs a json file pairing workspaces with a
#   refresh workflow job
# - this script takes a pair of these files as parameters
# - it then iterates through each workspace:
#   - if the workspace doesn't appear in both files, issue an error
#   - else, invoke a comparison between the two runsets, using eval
#     metric changes
#   - at the end, report comparison results

# XXX This first cut shows all score changes, ordered by % change.
# XXX Eventually, we want a more rolled-up summary:
# XXX - some single-number score combining the score change across all
# XXX   workspaces
# XXX - highlighting any scores with big changes that might indicate
# XXX   problems or require investigation

class RefreshGroupCompare:
    def __init__(self,baseline,modified, delta_threshold, consistent_flag, negative_only):
        self.banner='\n===================='
        self.per_metric_cutoffs={}
        self.default_cutoff=0.01
        self.delta_threshold=delta_threshold
        self.consistent_flag=consistent_flag
        self.negative_only=negative_only
        # the above can be modified directly by the client after
        # the __init__ call
        old = set(baseline.ws_map.keys())
        new = set(modified.ws_map.keys())
        self.dropped = old-new
        self.added = new-old
        from browse.models import Workspace
        self.by_workspace = {
                ws_id:ScoreSetCompare(
                        baseline.ws_map[ws_id]['ss'],
                        modified.ws_map[ws_id]['ss'],
                        Workspace.objects.get(pk=ws_id),
                        )
                for ws_id in old & new
                }
    def cutoff_of_metric(self,metric):
        return self.per_metric_cutoffs.get(metric,self.default_cutoff)
    def show_add_drop(self):
        if self.added or self.dropped:
            print(self.banner)
        show_list('added workspaces',self.added)
        show_list('dropped workspaces',self.dropped)
    def show_diff(self):
        self.show_add_drop()
        for ws_id in sorted(self.by_workspace):
            rs_compare = self.by_workspace[ws_id]
            print(self.banner)
            rs_compare.show_add_drop('Workspace %d'%ws_id)
            metric_names = ScoreSetCompare.metric_names
            per_metric_deltas = []
            # per_metric_deltas has one element for each element in
            # metric_names; the element is a list of namedtuples, each
            # describing the comparison of a score between runs
            for metric in metric_names:
                deltas = rs_compare.build_deltas(
                                metric,
                                self.cutoff_of_metric(metric),
                                self.delta_threshold
                                )
                if self.negative_only:
                    deltas = [x for x in deltas if x.delta < 0]
                per_metric_deltas.append(deltas)
            if self.consistent_flag:
                # only report scores that change in the same direction
                # in all metrics
                import numpy
                # first, accumulate all deltas for each score;
                # d={(role,code):[delta_direction,...],...}
                d = {}
                for deltas in per_metric_deltas:
                    for row in deltas:
                        l = d.setdefault((row.role,row.code),[])
                        l.append(numpy.sign(row.delta))
                # then build a set of (role,code) keys that are consistent
                codes_to_keep = set()
                for code,l in six.iteritems(d):
                    if len(l) == len(per_metric_deltas) and len(set(l)) == 1:
                        codes_to_keep.add(code)
                # now replace per_metric_deltas by a new list in which each
                # inner list of rows is filtered by (role,code)
                keep = []
                for deltas in per_metric_deltas:
                    l = []
                    keep.append(l)
                    for row in deltas:
                        if (row.role,row.code) in codes_to_keep:
                            l.append(row)
                per_metric_deltas = keep
            from dtk.num import median
            for metric,deltas in zip(metric_names,per_metric_deltas):
                print('\nWorkspace',ws_id,metric,'changes:')
                if deltas:
                    vals = [100 * x.delta for x in deltas]
                    fmt = "%0.2f%%"
                    print('        delta',
                            'min',fmt%vals[-1],
                            'median',fmt%median(vals),
                            'max',fmt%vals[0],
                            )
                self.print_delta_table(metric,deltas)
        print(self.banner)
    def print_delta_table(self,metric_name,deltas):
        header = (
                'CM',
                'score',
                'delta',
                'old '+metric_name,
                'new '+metric_name,
                'old job',
                'new job',
                'compare URL',
                )
        from dtk.text import print_table
        print_table([header,tuple('-'*len(x) for x in header)]+[
                (
                    row.role,
                    row.code,
                    '%0.2f%%'%(100*row.delta),
                    '%e'%row.old,
                    '%e'%row.new,
                    str(row.old_job),
                    str(row.new_job),
                    row.compare_URL,
                )
                for row in deltas
                ])

class ScoreSetCompare:
    metric_names= ['SigmaOfRank1000Condensed','AUR']
    def __init__(self,baseline,modified,ws):
        bd = baseline.job_type_to_id_map()
        md = modified.job_type_to_id_map()
        old = set(bd.keys())
        new = set(md.keys())
        self.dropped = old-new
        self.added = new-old
        from dtk.enrichment import EnrichmentMetric
        metric_classes = [
                EnrichmentMetric.lookup(x)
                for x in self.metric_names
                ]
        from runner.process_info import JobInfo
        self.by_cm = {
                job_type:JobCompare(
                        JobInfo.get_bound(ws,bd[job_type]),
                        JobInfo.get_bound(ws,md[job_type]),
                        ws.get_wsa_id_set(ws.eval_drugset),
                        metric_classes,
                        ws.id
                        )
                for job_type in old & new
                }
    def show_add_drop(self,prefix):
        show_list(prefix+' added jobs',self.added)
        show_list(prefix+' dropped jobs',self.dropped)
    def show_ordered_deltas(self,table_data):
        from dtk.text import print_table
        print_table(table_data)
    def build_deltas(self,metric_name,cutoff,delta_threshold):
        result = []
        from collections import namedtuple
        ResultType = namedtuple("ResultType"," ".join([
                "role",
                ]+JobCompare.metric_delta_columns))
        for job_type,job_compare in six.iteritems(self.by_cm):
            result += [
                    ResultType(job_type,*row)
                    for row in job_compare.get_metric_deltas(
                            metric_name,
                            cutoff,
                            delta_threshold
                            )
                    ]
        result.sort(key=lambda x:-x.delta)
        return result

class JobCompare:
    code_result_columns=[
                "old",
                "new",
                "old_job",
                "new_job",
                ]
    def __init__(self,baseline,modified,ds,metric_classes, ws_id):
        self.by_code = {}
        self.ws_id = ws_id
        bdc = baseline.get_data_catalog()
        code_list=bdc.codes_by_key()
        try:
            code_list = code_list['wsa'][0]
        except (KeyError,IndexError):
            return
        mdc = modified.get_data_catalog()
        from dtk.enrichment import EMInput
        from dtk.files import Quiet
        from collections import namedtuple
        CodeResult = namedtuple(
                "CodeResult",
                " ".join(self.code_result_columns),
                )
        for code in code_list:
            code_result = {}
            self.by_code[code] = code_result
            b_emi = EMInput(bdc.get_ordering(code,True),ds)
            m_emi = EMInput(mdc.get_ordering(code,True),ds)
            for MetricClass in metric_classes:
                b_em = MetricClass()
                m_em = MetricClass()
                with Quiet() as tmp:
                    b_em.evaluate(b_emi)
                    m_em.evaluate(m_emi)
                code_result[b_em.name()] = CodeResult(
                        b_em.rating,
                        m_em.rating,
                        baseline.job.id,
                        modified.job.id,
                        )
    metric_delta_columns=[
                "code",
                "delta",
                "compare_URL",
                ]+code_result_columns
    def get_metric_deltas(self,metric_name,cutoff,delta_theshold):
        result = []
        def get_url(code,pair,server_pref='http://localhost:8000'):
            return "%s/cv/%d/score_cmp/wsa/?x=%d_%s&y=%d_%s" % (
                    server_pref,
                    self.ws_id,
                    pair.old_job,
                    code,
                    pair.new_job,
                    code,
                    )
            #return server_pref + '/cv/' + str(self.ws_id) + '/score_cmp/wsa/?y=' + str(pair[-2]) + '_' + code + '&x=' + str(pair[-1]) + '_' + code
        def get_delta(pair):
            # Generally, we want the old value in the denomimator. But limit
            # it to cutoff at a minimum, so we don't get giant percentages
            # due to miniscule baseline amounts. If the denom is zero,
            # report as 100% change.
            denom = max(cutoff,pair.old)
            if denom:
                delta = (pair.new - pair.old)/denom
            else:
                delta = 1.0
            return delta
        def ok_to_report(pair):
            if abs(get_delta(pair)) <= delta_theshold:
                return False # the difference isn't worth reporting
            if max(pair.old,pair.new)<cutoff:
                return False #neither metric is worth considering
            return True
        from collections import namedtuple
        MetricResult = namedtuple(
                "MetricResult",
                " ".join(self.metric_delta_columns),
                )
        for code,code_result in six.iteritems(self.by_code):
            pair = code_result[metric_name]
            if ok_to_report(pair):
                delta = get_delta(pair)
                url = get_url(code, pair)
                result.append(MetricResult(
                        code,
                        delta,
                        url,
                        *pair
                        ))
        return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='compare runsets across multiple workspaces'
            )
    parser.add_argument('--SOR-cutoff',type=float, default = 1e-4)
    parser.add_argument('--FEBE-cutoff',type=float, default = 1.)
    parser.add_argument('--delta-cutoff',type=float, default = 0.)
    parser.add_argument('--consistent_only',action='store_true')
    parser.add_argument('--negative_only',action='store_true')
    parser.add_argument('baseline')
    parser.add_argument('modified')
    args=parser.parse_args()

    from dtk.refresh_group import RefreshGroup
    consistent_flag = False
    if args.consistent_only:
        consistent_flag = True
    negative_only = False
    if args.negative_only:
        negative_only = True
    baseline = RefreshGroup(args.baseline)
    modified = RefreshGroup(args.modified)
    rg_compare = RefreshGroupCompare(baseline,
                                     modified,
                                     args.delta_cutoff,
                                     consistent_flag,
                                     negative_only,
                                    )
    rg_compare.per_metric_cutoffs['SigmaOfRank'] = args.SOR_cutoff
    rg_compare.per_metric_cutoffs['FEBE'] = args.FEBE_cutoff
    rg_compare.show_diff()
