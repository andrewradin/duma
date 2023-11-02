#!/usr/bin/env python3
# ex: set tabstop=4 expandtab:

from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import object
import os
import sys
import django
if 'django.core' not in sys.modules:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    django.setup()

from runner.models import Load
from path_helper import PathHelper
import time
import logging
logger = logging.getLogger(__name__)

def local_cores():
    try:
        max_processes = os.sysconf('SC_NPROCESSORS_ONLN')
    except:
        max_processes = 4
    if max_processes >= 4:
        max_processes -= 1 # reserve one core on big machines
    return max_processes

# XXX This function is the definitive list of what each resource slot means.
# XXX This corresponds to a 'names' list in ResourceManager.desc().
# XXX If it gets mucn more complex, it's probably better to have name a
# XXX utility function that assembles a 'want' vector based on resource
# XXX names.
def default_totals():
    from aws_op import Machine
    worker = Machine.name_index[PathHelper.cfg('worker_machine_name')]
    return [
        local_cores(),
        worker.num_cores(),
        30, # GEO download slots
        9, # RNAseq download slots
        worker.num_cores(), # slow worker cores
        # XXX Not sure what 'slow worker cores' are. It might have been
        # XXX some early thing. run_meta uses this slot instead of the
        # XXX normal remote_cores, and also sets the slow flag. It's not
        # XXX allocated anywhere else.
        ]

class ResourceManager(object):
    max_waiting_jobs = 10
    def __init__(self):
        from dtk.lock import FLock
        self.lock = FLock(PathHelper.timestamps+'resource_reserve_lock')
    #######
    # admin functions
    #######
    def terminate_all(self):
        try:
            self.lock.acquire()
            Load.objects.exclude(job=0).delete()
        finally:
            self.lock.release()
    def set_totals(self,got=None,margin=None):
        try:
            if not got:
                got = default_totals()
            if not margin:
                # default margin is:
                # - 10%
                # - but at least 1
                # - unless only one is available, in which case no margin
                margin = [0 if x < 2 else max(1,int(0.1*x)) for x in got]
            self.lock.acquire()
            rec,new = Load.objects.get_or_create(job=0)
            rec.set_allocation(got)
            rec.want = " ".join([str(x) for x in margin])
            rec.save()
        finally:
            self.lock.release()
    def dump(self):
        for rec in Load.objects.all().order_by('job'):
            print(rec.job,rec.get_request(),rec.get_allocation())
    #######
    # normal (non-blocking) functions
    #######
    def avail(self,slow=False):
        # get total
        total = Load.objects.get(job=0)
        l = total.get_allocation()
        m = [int(x) for x in total.want.split(' ')]
        # subtract what's in use
        for rec in Load.objects.all():
            if rec.job and rec.got:
                for i,v in enumerate(rec.get_allocation()):
                    l[i] -= v
                    # any resource allocated to a non-slow job counts
                    # as part of the margin, so subtract it here to
                    # calculate the portion of the margin remaining
                    if not rec.slow:
                        m[i] = max(0,m[i]-v)
        # adjust what's available if it's a slow job
        if slow:
            l = [max(0,x-y) for x,y in zip(l,m)]
        # return difference
        return l
    def status(self,job):
        rec = Load.objects.get(job=job)
        return rec.get_allocation()
    def desc(self,job):
        rec = Load.objects.get(job=job)
        names = ['local core','remote core','GEO slot','RNAseq slot','s-core']
        items=[]
        if rec.slow:
            items.append('SLOW')
        got = rec.get_allocation()
        if got:
            for label,cnt in zip(names,got):
                if cnt:
                    if cnt > 1:
                        label += 's'
                    items.append(f'{cnt} {label}')
            return ' '.join(items)
        want = rec.get_request()
        if want:
            items.append('awaiting')
            for label,(min_cnt,max_cnt) in zip(names,want):
                if min_cnt:
                    if min_cnt > 1:
                        label += 's'
                    extra = '+' if max_cnt != min_cnt else ''
                    items.append(f'{min_cnt}{extra} {label}')
            return ' '.join(items)
    def request(self,job,want,slow=False):
        try:
            self.lock.acquire()
            try:
                rec,new = Load.objects.get_or_create(job=job)
            except Exception as ex:
                self._retry_db_connection(ex)
                rec,new = Load.objects.get_or_create(job=job)
            rec.set_request(want)
            rec.got = ''
            rec.slow = slow
            rec.save()
            self._start_all()
        finally:
            self.lock.release()
    def terminate(self,job):
        try:
            self.lock.acquire()
            try:
                Load.objects.filter(job=job).delete()
            except Exception as ex:
                self._retry_db_connection(ex)
                Load.objects.filter(job=job).delete()
            logger.info(f"Job {job} terminated")
            self._start_all()
        finally:
            self.lock.release()
    def max_runs_allowed(self):
        # We want to pre-start some jobs so they're pending on the
        # RM, but not too many.  So, we set a limit of the number of
        # jobs that currently have allocations, plus the number that
        # don't want allocations, plus a margin.  This fixes a hole
        # where we'd count the number blocked, because it didn't count
        # jobs that had launched but had not yet requested an allocation.
        non_requestors = Load.objects.filter(want='')
        grantees = Load.objects.exclude(got='')
        return grantees.count() + non_requestors.count() + self.max_waiting_jobs
    #######
    # high-level (blocking) functions
    #######
    def wait_for_resources(self,job,want,slow=False):
        logger.info(f"Job {job} waiting for {want}")
        import time
        from datetime import timedelta
        from django.utils import timezone
        start = time.monotonic()
        start_date = timezone.now()

        self.request(job,want,slow)
        while True:
            try:
                got = self.status(job)
            except Exception as ex:
                self._retry_db_connection(ex)
                continue
            if got:
                break
            time.sleep(1)
        
        end = time.monotonic()

        waited = timedelta(seconds=end - start)


        from runner.models import ProcessWait
        import json
        ProcessWait.objects.create(
            process_id=job,
            start=start_date,
            duration=waited,
            waited_for=json.dumps(want),
        )

        minutes = waited / timedelta(minutes=1)
        logger.info(f"Job {job} got {got} from {want} after {minutes:.2f}m")

        return got
    #######
    # internal functions
    #######
    def _retry_db_connection(self,ex):
        # We only retry if we believe retrying will help, such as disconnected
        # for inactivity.
        retry_on = [
            'MySQL server has gone away',
            'The client was disconnected by the server because of inactivity',
        ]
        for retry_text in retry_on:
            if retry_text in str(ex):
                django.db.connection.close()
                time.sleep(10)
                return
        
        # Error doesn't match our known retryable ones, don't retry.
        raise ex
    def _start_all(self):
        for rec in Load.objects.filter(got='').order_by('job'):
            if self._start_one(rec):
                continue
            if not rec.slow:
                return
            # if a slow job didn't start, keep looking
    def _start_one(self,rec):
        if not rec:
            return False
        avail = self.avail(rec.slow)
        got = []
        for i,r in enumerate(rec.get_request()):
            if r[0] > avail[i]:
                return False
            if r[0]:
                if not r[1]:
                    # max of 0 means as many as possible
                    cap = avail[i]
                    got.append(min(avail[i],cap))
                else:
                    # Non-zero max means that many jobs of approximately
                    # equal length will be run, one per available core.
                    # So, effectively, if the full number of cores aren't
                    # available, the jobs will be run in batches:
                    cap = r[1]
                    batches = (cap + avail[i] - 1) // avail[i]
                    # we can adjust the number of cores so they stay
                    # busy in every batch
                    want = (cap + batches - 1) // batches
                    # Note that if r[1] == r[0], the effect of the above
                    # calculation is that want == r[0].
                    got.append(min(avail[i],want))
            else:
                got.append(0)
        rec.set_allocation(got)
        rec.save()
        return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='resource reservation utility')
    parser.add_argument('job',help='0 for get/set totals')
    parser.add_argument('level'
        ,nargs='*'
        )
    args = parser.parse_args()

    job = int(args.job)
    rm = ResourceManager()
    if job:
        want = []
        for item in args.level:
            if ':' in item:
                want.append(tuple(map(int,item.split(':'))))
            else:
                want.append(int(item))
        got = rm.wait_for_resources(job,want)
        print(" ".join([str(x) for x in got]))
    else:
        # job 0 signals get/set_totals mode
        totals = [int(x) for x in args.level]
        if not totals:
            # get totals
            print('default totals',default_totals())
            print('actual totals',rm.status(0))
            print('available',rm.avail())
        else:
            rm.set_totals(totals)
