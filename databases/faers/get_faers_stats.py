#!/usr/bin/env python3

import sys
try:
    import path_helper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")

import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")

import django
django.setup()

def get_faers_jobs(min_job=None,max_job=None):
    from runner.models import Process
    jobs_by_name = {}
    qs = Process.objects.filter(
                status=Process.status_vals.SUCCEEDED,
                name__startswith='faers_',
                )
    if min_job:
        qs = qs.filter(id__gte=min_job)
    if max_job:
        qs = qs.filter(id__lte=max_job)
    for job in qs.order_by('-id'):
        jobs_by_name.setdefault(job.name,[]).append(job)
    return jobs_by_name

# XXX count number of drugs over 0.95 significance for N sample workspaces
# XXX (this would be trivial if it was published as a score)
class Evaluator:
    col_list=[
            'name',
            'job_id',
            'indi_cnt',
            'kt_cnt',
            'cas_kt_cnt',
            'dcoe_eval',
            'dcop_eval',
            'raw_eval',
            ]
    def __init__(self,fn,metric='FEBE'):
        self.out = open(fn,'w')
        self.out.write('\t'.join(self.col_list)+'\n')
        from dtk.enrichment import EnrichmentMetric
        self.MetricType = EnrichmentMetric.lookup(metric)
        # XXX StatsCollector no longer exists; this can be re-written
        # XXX using ClinicalEventCounts if anyone cares enough
        self.sc = StatsCollector()
        self.bg_per_cas = {
                k:v
                for k,v in self.sc.histogram('drug','drugsFromFAER')
                }
        self.bg_total = self.sc.count_items('distinct event','diseasesFromFAER')
    def evaluate(self,job):
        ev._setup_for_job(job)
        ev._eval_raw_enrichment()
        output = [str(getattr(self,x)()) for x in self.col_list]
        self.out.write('\t'.join(output)+'\n')
    def name(self): return self.bji.job.name
    def job_id(self): return self.bji.job.id
    def indi_cnt(self): return self.disease_total
    def kt_cnt(self): return len(self.kts)
    def cas_kt_cnt(self): return len(self.cas_kts)
    def dcoe_eval(self): return self._eval_faers_job('dcoe')
    def dcop_eval(self): return self._eval_faers_job('dcop')
    def raw_eval(self): return self.raw_rating
    def _setup_for_job(self,job):
        ws_id=int(job.name[6:])
        from browse.models import Workspace
        self.ws=Workspace.objects.get(pk=ws_id)
        from runner.process_info import JobInfo
        self.bji=JobInfo.get_bound(self.ws,job.id)
        self.kts = self.ws.get_wsa_id_set('kts')
    def _eval_faers_job(self,score_code):
        cat=self.bji.get_data_catalog()
        from dtk.enrichment import EMInput
        emi = EMInput(cat.get_ordering(score_code,True),self.kts)
        em= self.MetricType()
        em.evaluate(emi)
        return em.rating
    def _eval_raw_enrichment(self):
        indi = self.bji.job.settings()['search_term']
        indi_set = set(indi.split('|'))
        self.disease_total = self.sc.filtered_total(indi_set,'drug4indi')
        ordering = []
        import math
        for cas,disease_cnt in self.sc.filtered_counts(indi_set,'drug4indi'):
            disease = float(disease_cnt)/self.disease_total
            bg = float(self.bg_per_cas[cas])/self.bg_total
            ordering.append( (cas,math.log(disease/bg,2)) )
        from browse.models import WsAnnotation
        self.cas_kts = set()
        for x in WsAnnotation.objects.filter(pk__in=self.kts):
            self.cas_kts.update(x.agent.cas_set)
        ordering.sort(key=lambda x:x[1],reverse=True)
        from dtk.enrichment import EMInput
        emi = EMInput(ordering,self.cas_kts)
        em= self.MetricType()
        em.evaluate(emi)
        self.raw_rating = em.rating

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                description='FAERS workspace stats',
                )
    parser.add_argument('--min-job',type=int)
    parser.add_argument('--max-job',type=int)
    args = parser.parse_args()


    # db_stats functionality has been moved to get_clin_ev_stats.py
    # the remaining ws_stats functionality above can be fixed if
    # needed by using ClinicalEventCounts instead of StatsCollector
    if True:
        ev = Evaluator('faers_ws_eval.tsv')
        jobs_by_name = get_faers_jobs(args.min_job,args.max_job)
        for name in sorted(jobs_by_name,key=lambda x:int(x[6:])):
            joblist = jobs_by_name[name]
            job = joblist[0]
            print('processing',job.name,job.id)
            ev.evaluate(job)

