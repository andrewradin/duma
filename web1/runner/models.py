from django.db import models,transaction
from django.db.models import Max
from django.utils import timezone
from tools import Enum
import os
from path_helper import PathHelper
import threading
import subprocess
from notes.models import Note
from algorithms.exit_codes import ExitCoder

import logging
logger = logging.getLogger(__name__)

class Load(models.Model):
    job = models.IntegerField(unique=True)
    want = models.CharField(max_length=250,blank=True,default="")
    got = models.CharField(max_length=250,blank=True,default="")
    slow = models.BooleanField(default=False)
    def set_request(self,want):
        # first convert everything to tuple format
        l = [
                x if type(x) is tuple else (x,x)
                for x in want
                ]
        # validate that request is possible
        totals = Load.objects.get(job=0).get_allocation()
        for i,x in enumerate(l):
            if totals[i] < x[0]:
                raise ValueError(
                        'min request exceeds total: idx %d, want %d, have %d'
                        % (i,x[0],totals[i])
                        )
        # ok, seems good; convert to string for saving
        l = ["%d:%d" % x for x in l]
        self.want = " ".join(l)
    def get_request(self):
        if self.want:
            return [tuple(map(int,x.split(':'))) for x in self.want.split()]
        return None
    def set_allocation(self,got):
        self.got = " ".join([str(x) for x in got])
    def get_allocation(self):
        if self.got:
            return [int(x) for x in self.got.split()]
        return None

# The MaintenanceLock mechanism allows maintenance tasks to be interleaved
# with the Process mechanism without the two ever running at the same time.
#
# MaintenanceLocks are identified by a unique task name.
#
# The basic life cycle is:
# - Process.maint_request() requests a maintenance lock; if any jobs are active,
#   the lock enters a PENDING state. When the last job in the queue completes,
#   it enters the ACTIVE state.
# - While a lock is ACTIVE, no jobs can start, but they can be queued.
# - Process.maint_release() moves a maintenance lock from ACTIVE to IDLE, and
#   starts any queued jobs (if there are no other maintenance locks active)
# Maintenance locks can be manual or automatic. An automatic lock requires a
# script called web1/scripts/maint_task_<taskname>.py. This script will be
# invoked when the maintenance lock becomes active, and should call
# Process.maint_release() when it completes. A manual task is run from the
# command line; scripts/maint_lock.py should be used to request and release
# the associated lock so that the jobs system knows maintenance is in progress.
#
# To support breaking long maintenance tasks into multiple pieces (so the
# job queue isn't paused for too long), the automatic maintenance task script
# can call Process.maint_yield(), which does a release and re-request as an
# atomic operation. Also, maint_request and maint_yield both take an optional
# 'detail' parameter, which should be a dict holding state information that
# the script can use to know where to resume processing. A particular
# invocation of the script can call Process.maint_detail() to get a copy of
# the previously saved information. See maint_task_lts_scan.py as an example.
#
# Note that the maintenance task script should ALWAYS call maint_detail(),
# even if it doesn't use cached state information, because that function
# also verifies that the lock is in place and that the script is running
# as the correct user.
#
# Since automatic locks are explicitly released by the script itself, an
# unexpected condition may terminate the script before the lock is released.
# To recover from this case, you can manually release the lock using
# maint_lock.py, or you can run the maintenance task script itself from
# the command line to allow it to retry processing.
#
# If the dict passed as the detail parameter contains a 'progress' key,
# the value of that key will be displayed on the jobs page while the
# lock is ACTIVE or PENDING.
class MaintenanceLock(models.Model):
    status_vals = Enum([],
            [ ('IDLE',)
            , ('PENDING',)
            , ('ACTIVE',)
            ])
    task = models.CharField(max_length=70)
    status = models.IntegerField(choices=status_vals.choices())
    detail = models.TextField()
    pid = models.IntegerField(null=True)
    # XXX This could also have a priority field, such that only tasks of the
    # XXX same priority would run together, and when multiple idle tasks are
    # XXX waiting to start, only the highest priority one is started. This
    # XXX could separate fixes from scanning, for example.

class Process(models.Model):
    def __str__(self):
        return '<Process: %d %s %s>' % (
                    self.id,
                    self.name,
                    self.status_vals.get('label',self.status),
                    )
    def note_info(self,attr):
        if attr == 'note':
            return {
                'label':"job note on %s" %(
                        self.name,
                        ),
                }
        raise Exception("bad note attr '%s'" % attr)
    class Meta:
        # Used for adding indexes via django migrations.
        index_together = [
                ['name'],
                ['status'],
                ]
    suppress_execution = False
    status_vals = Enum([],
            [ ('QUEUED',)
            , ('RUNNING',)
            , ('SUCCEEDED',)
            , ('FAILED',)
            , ('FAILED_PREREQ',)
            ])
    launch_exception = ExitCoder().encode('launch_exception')
    active_statuses = [status_vals.QUEUED,status_vals.RUNNING]
    name = models.CharField(max_length=70)
    role = models.CharField(max_length=256,default='')
    rundir = models.CharField(max_length=1024)
    cmd = models.CharField(max_length=1024)
    status = models.IntegerField(choices=status_vals.choices())
    wait_for = models.ManyToManyField("self",symmetrical=False)
    sw_version = models.CharField(max_length=70,blank=True,default="")
    note = models.ForeignKey(Note,null=True,blank=True, on_delete=models.CASCADE)
    settings_json = models.TextField(blank=True,default="")
    created = models.DateTimeField(auto_now_add=True,null=True)
    started = models.DateTimeField(null=True)
    completed = models.DateTimeField(null=True)
    exit_code = models.IntegerField(null=True)
    user = models.CharField(max_length=70,blank=True,default="")
    invalidated = models.BooleanField(default=False)

    def get_note_text(self):
        return Note.get(self,'note','')
    def pidfile(self):
        return PathHelper.pidfiles+str(self.id)
    def active(self):
        return self.status in self.active_statuses
    def job_type(self):
        return self.name.split('_')[0]
    def status_text(self):
        return Process.status_vals.get('label', self.status)
    def settings(self):
        try:
            return self._cached_settings
        except AttributeError:
            if self.settings_json:
                import json
                d = json.loads(self.settings_json)
            else:
                d = {}
            setattr(self,'_cached_settings',d)
            return d
    def log_closed(self):
        '''return True if log has been committed to LTS for a while.

        This means that the process should no longer be active. This is
        only used for a specialized maintenance function to allow us
        to recover if a process's monitor thread in drive_background
        dies before the process itself.
        '''
        path = self.logfile()
        import os
        if not os.path.islink(path):
            return False
        commit_time = os.lstat(path).st_mtime
        import time
        return time.time() - commit_time > 2*60
    def logfile(self,fetch=False):
        from runner.common import LogRepoInfo
        lri = LogRepoInfo(self.id)
        if fetch:
            lri.fetch_log()
        return lri.log_path()
    def log_url(self):
        return PathHelper.url_of_file(self.logfile())
    def _dequeue_children(self):
        enum = Process.status_vals
        for p in self.process_set.all():
            p.status = enum.FAILED_PREREQ
            logger.info("%s(%d) %s",p.name,p.id,enum.get('label',p.status))
            p.save()
            ReservedJobname.objects.filter(holder=p).delete()
            p._dequeue_children()
    @classmethod
    def dummy_process(cls,name
            ,settings_json=''
            ,user=''
            ):
        '''Return a dummy job id for storing foreground output.'''
        from django.utils.timezone import now
        p = cls()
        p.name = name
        #p.rundir = rundir
        #p.cmd = cmd
        p.status = cls.status_vals.SUCCEEDED
        #p.sw_version = sw_version
        p.settings_json = settings_json
        p.user = user
        p.created = p.started = p.completed = now()
        p.save()
        return p.id
    @classmethod
    def queue_process(cls,name,rundir,cmd
            ,run_after=[]
            ,ordering_names=[]
            ,sw_version=''
            ,settings_json=''
            ,user=''
            ,role=''
            ):
        # extend run_after by any jobs reserving this job's name
        run_after = set(run_after)
        for rj in ReservedJobname.objects.filter(name=name):
            run_after.add(rj.holder)
        p = Process()
        p.name = name
        p.rundir = rundir
        p.cmd = cmd
        p.status = cls.status_vals.QUEUED
        p.sw_version = sw_version
        p.settings_json = settings_json
        p.user = user
        p.role = role
        p.save()
        logger.info("queueing %s(%d)",p.name,p.id)
        # verify all run_after numbers are active jobs
        active_job_ids = [x.id for x in cls.active_jobs_qs()]
        run_after = set([x for x in run_after if x in active_job_ids])
        p.wait_for.set(run_after)
        # create ReservedJobname records for each ordering name
        ordering_names = set(ordering_names)
        for depname in ordering_names:
            rj = ReservedJobname()
            rj.holder = p
            rj.name = depname
            rj.save()
        return p.id
    @classmethod
    def queued(cls):
        qs = Process.objects.filter(status=cls.status_vals.QUEUED)
        return qs.count()
    @classmethod
    def running(cls):
        qs = Process.objects.filter(status=cls.status_vals.RUNNING)
        return qs.count()
    @classmethod
    def already_active(cls,job_name):
        qs = Process.objects.filter(
                        status__in=cls.active_statuses,
                        name=job_name,
                        )
        return qs.exists()
    @classmethod
    def job_set_history_qs(cls,name_set):
        qs = cls.objects.filter(name__in=name_set).order_by('-id')
        return qs
    @classmethod
    def latest_jobs_qs(cls):
        # show only most recent job for each name
        latest = cls.objects.values('name').annotate(latest=Max('pk'))
        qs = cls.objects.filter(pk__in=[x['latest'] for x in latest])
        return qs
    @classmethod
    def active_jobs_qs(cls):
        # show only active jobs
        enum=cls.status_vals
        return cls.objects.filter(
                    status__in=[enum.QUEUED,enum.RUNNING]
                    )
    @classmethod
    def failed_jobs_qs(cls):
        # show only active jobs
        enum=cls.status_vals
        return cls.latest_jobs_qs().filter(status__in=[enum.FAILED])
    @classmethod
    def kill_all(cls):
        first_pass_kill = cls.status_vals.QUEUED
        for pass_num in (1,2):
            for p in cls.active_jobs_qs():
                if p.status == first_pass_kill or pass_num == 2:
                    cls.abort(p.id)
    @classmethod
    def start_all(cls):
        '''Start as many processes as possible

        Return a tuple:
            (# of processes started
            , # of processes waiting
            )
        '''
        started = 0
        import time
        while cls._try_start():
            time.sleep(0.5)
            started += 1
        queued = cls.queued()
        #logger.info("just started %d; %d still queued",started,queued)
        return (started,queued)
    @classmethod
    def _try_start(cls):
        # This can be called from any of multiple drive_background.py or
        # web processes (not just different threads).  So, use a wrapped
        # flock to assure the calls run in an orderly fashion.
        #
        # We need to assert the DB transaction inside the FLock because
        # otherwise it's possible for two threads to each start a transaction
        # with the same view of the database, and therefore to think that
        # they each need to start a copy of the same pending process.
        from dtk.lock import FLock
        with FLock(PathHelper.timestamps+"try_start_lock"):
            return cls._start_one_if_possible()
    @classmethod
    @transaction.atomic
    def _start_one_if_possible(cls):
        if cls.maint_active():
            return False
        jobs = Process.objects
        running = jobs.filter(status=cls.status_vals.RUNNING)
        from reserve import ResourceManager
        rm = ResourceManager()
        if running.count() >= rm.max_runs_allowed():
            return False
        runnable = jobs.filter(status=cls.status_vals.QUEUED,wait_for=None)
        if runnable.count() == 0:
            return False
        p = runnable[0]
        p.status = cls.status_vals.RUNNING
        p.started = timezone.now()
        p.save()
        if cls.suppress_execution:
            logger.warning("suppressing execution of %s(%d)",p.name,p.id)
        else:
            logger.info("launching %s(%d)",p.name,p.id)
            # Background execution is broken into several levels:
            # - a thread in this process waits for the background process to
            #   complete, and calls stop() with the return code
            # - the wrapper program run_process.py launched by that thread
            #   handles pid file management, process group creation,
            #   and selecting the correct working directory
            # - the background program, launched from run_process.py,
            #   performs the actual work, possibly launching sub-processes
            #   of its own; it calls reserve.py to make sure resources
            #   are available
            t = threading.Thread(target=background_wrapper
                    , args=[p]
                    )
            t.start()
        return True
    @classmethod
    @transaction.atomic
    def _stop_only(cls,p_id,return_code,start_all=True):
        # release all job resources
        from reserve import ResourceManager
        rm = ResourceManager()
        rm.terminate(p_id)
        #logger.debug("entering stop, p_id=%d",p_id)
        p = Process.objects.get(pk=p_id)
        ok = (return_code == 0)
        if ok:
            p.status = cls.status_vals.SUCCEEDED
        else:
            p.status = cls.status_vals.FAILED
        logger.info("%s(%d) %s",p.name,p.id,cls.status_vals.get('label',p.status))
        if not ok:
            p._dequeue_children()
        p.process_set.clear()
        p.completed = timezone.now()
        p.exit_code = return_code
        p.save()
        ReservedJobname.objects.filter(holder=p).delete()
        # If nothing is running, and maintenance is pending, start it
        cls._maint_check_start()
    @classmethod
    def stop(cls,p_id,return_code,start_all=True):
        # this is broken out so that the transaction lock doesn't cover the
        # call to start_all, which invokes a transaction lock of its own
        cls._stop_only(p_id,return_code)
        # note that if a maintenance lock got asserted in _stop_only,
        # the following will not do anything
        if start_all:
            cls.start_all()

    @classmethod
    @transaction.atomic
    def maint_request(cls,task,detail={}):
        logger.info("maint_request for %s",task)
        import json
        enum = MaintenanceLock.status_vals
        ml,created = MaintenanceLock.objects.get_or_create(
                task=task,
                defaults={
                        'status':enum.IDLE,
                        'detail':json.dumps({}),
                        }
                )
        assert ml.status == enum.IDLE
        ml.status = enum.PENDING
        ml.detail = json.dumps(detail)
        ml.save()
        cls._maint_check_start()
    @classmethod
    @transaction.atomic
    def _maint_release_trx(cls,task):
        import json
        enum = MaintenanceLock.status_vals
        ml = MaintenanceLock.objects.get(task=task)
        assert ml.status == enum.ACTIVE
        ml.status = enum.IDLE
        ml.detail = json.dumps({})
        ml.pid = None
        ml.save()
    @classmethod
    def maint_release(cls,task):
        logger.info("maint_release for %s",task)
        # this is broken out because, in the test environment, drive_background
        # gets mocked and run in-process, so it can't assert its own db trx
        cls._maint_release_trx(task)
        if cls.suppress_execution:
            logger.warning("suppressing post-maintenance autostart")
            return # for testing only
        if not cls.maint_active():
            cls.drive_background()
    @classmethod
    @transaction.atomic
    def _maint_yield_trx(cls,task,detail):
        import json
        enum = MaintenanceLock.status_vals
        ml = MaintenanceLock.objects.get(task=task)
        assert ml.status == enum.ACTIVE
        ml.status = enum.PENDING
        ml.detail = json.dumps(detail)
        ml.pid = None
        ml.save()
        cls._maint_check_start()
    @classmethod
    def maint_yield(cls,task,detail={}):
        logger.info("maint_yield for %s",task)
        # this is broken out because, in the test environment, drive_background
        # gets mocked and run in-process, so it can't assert its own db trx
        cls._maint_yield_trx(task,detail)
        if cls.suppress_execution:
            logger.warning("suppressing post-maintenance autostart")
            return # for testing only
        if not cls.maint_active():
            cls.drive_background()
    @classmethod
    # should be called from inside a transaction
    def _maint_check_start(cls):
        if cls.active_jobs_qs().exists():
            return False
        any_started = False
        enum = MaintenanceLock.status_vals
        for ml in MaintenanceLock.objects.filter(status=enum.PENDING):
            ml.status = enum.ACTIVE
            ml.save()
            if cls.suppress_execution:
                logger.warning("suppressing maintenance launch")
            else:
                # look for a maintenance script with a matching name
                script_path = PathHelper.website_root \
                        + f'scripts/maint_task_{ml.task}.py'
                import os
                if os.path.exists(script_path):
                    logger.info(f"maint locking {ml.task}; starting script")
                    # One common use case here is being run from cron. Even
                    # if the conda interpreter is passed explicitly in crontab,
                    # that doesn't affect the interpreter that Popen will
                    # choose. So, explicitly replicate the current python
                    # interpreter here.
                    import sys
                    info = subprocess.Popen([sys.executable,script_path])
                    ml.pid = info.pid
                    ml.save()
                else:
                    logger.warning(f"maint locking {ml.task}; no script")
                    pass # treat as a manual lock
            any_started = True
        return any_started
    @classmethod
    def maint_active(cls):
        return MaintenanceLock.objects.filter(
                status = MaintenanceLock.status_vals.ACTIVE,
                ).exists()
    @classmethod
    def maint_detail(cls,task):
        err = Process.prod_user_error()
        if err:
            print(err)
            import sys
            sys.exit(1)
        import json
        enum = MaintenanceLock.status_vals
        ml = MaintenanceLock.objects.get(task=task)
        assert ml.status == enum.ACTIVE
        return json.loads(ml.detail)
    @classmethod
    def maint_status(cls):
        from collections import namedtuple
        ReturnType = namedtuple('ReturnType','task status pid progress')
        import json
        return [ReturnType(
                        x.task,
                        x.status_vals.get('label',x.status),
                        x.pid,
                        json.loads(x.detail).get('progress',''),
                        )
                for x in MaintenanceLock.objects.all()
                ]

    @classmethod
    def pid_wait(cls, pid, max_wait_secs=30):
        from psutil import pid_exists
        import time
        start = time.monotonic()
        while pid_exists(pid) and time.monotonic() - start < max_wait_secs:
            time.sleep(0.5)
        
        return not pid_exists(pid)

    @classmethod
    def abort(cls,p_id):
        # read pidfile and issue kill
        p = Process.objects.get(pk=p_id)
        if p.wait_for.count():
            # this won't happen if the job is running, but
            # may happen for queued waiting jobs
            p.wait_for.set([])
            p.save()
        logger.info("aborting job %d",p_id)
        import signal
        try:
            with open(p.pidfile(),"r") as f:
                pid = int(f.read().strip())
                infostr = f'{p.name} ({p.id}, pid={pid})'
                logger.info(f"aborting {infostr}")
                os.killpg(pid, signal.SIGHUP)
                # XXX We've seen failures where the above log appears,
                # XXX but the slack report of the arrival of the signal
                # XXX never happens. So, wait a while to see if the abort
                # XXX succeeds, and report if it doesn't.
                if not cls.pid_wait(pid):
                    from dtk.alert import slack_send
                    # Warn via slack. There was a thought to fall back to
                    # SIGKILL here, but this mostly happens in workflows,
                    # and in that case the SIGKILL does more harm than good
                    # (it kills the workflow itself, depriving us of clean
                    # logging of the workflow-specific cleanup logic).
                    slack_send(f"{infostr} still running after abort, requires manual intervention", add_host=True)
        except Exception as ex:
            logger.info("abort of %s(%d) got exception doing kill: %s"
                    ,p.name,p.id,str(ex)
                    )


        # In the normal case, since run_process now logs and ignores
        # the SIGHUP sent above, it should run its normal completion
        # code when its child process exits, removing the pid file,
        # so it no longer needs to be done here. If the process died
        # because of a machine shutdown, leaving the Process table
        # record in a running state, we could try to clean up the pidfile
        # here (we can detect this by getting a "No such process"
        # exception in the try block above). But that cleanup would
        # be incomplete, leaving dangling LTS data and possibly
        # other things, so we should handle it manually until it's
        # well enough understood to automate (and even then it should
        # probably be delegated to a utility script, rather than done
        # in-line here).
        #try:
        #    os.remove(p.pidfile())
        #except:
        #    logger.info("abort of %s(%d) got exception doing remove"
        #            ,p.name,p.id
        #            )
        # call stop to update db
        ec = ExitCoder()
        cls.stop(p_id,ec.encode('aborted'))
    @classmethod
    def prod_user_error(cls):
        prod_install_root = '/home/www-data/2xar/'
        if os.path.exists(prod_install_root):
            if PathHelper.install_root != prod_install_root:
                return '''
                This looks like a production system.
                To start jobs, you need to be running as www-data.
                '''
        return ''
    @classmethod
    def drive_background(cls):
        if cls.prod_user_error():
            logger.warning("Not starting background process, wrong user")
            return False
        logger.info("Invoking drive_background")
        cmd = PathHelper.website_root+'runner/drive_background.py' \
                + ' >>/var/log/drive_background.log 2>&1' \
                + ' &'
        subprocess.call([cmd],shell=True,close_fds=True)
        return True
    
    def waited(self):
        from datetime import timedelta
        return sum(ProcessWait.objects.filter(process=self).values_list('duration', flat=True), timedelta(seconds=0))

class ReservedJobname(models.Model):
    holder = models.ForeignKey(Process, on_delete=models.CASCADE)
    name = models.CharField(max_length=70)

class ProcessWait(models.Model):
    """Tracks how much time processes spent idle waiting for resources."""
    process = models.ForeignKey(Process, on_delete=models.CASCADE)
    start = models.DateTimeField()
    duration = models.DurationField()
    waited_for = models.TextField()



def background_wrapper(p):
    p_id = p.id
    try:
        logger.info("Launching subprocess for %s", p_id)
        # Currently, all output that doesn't get captured in the bg_log
        # file set up by run_process.py ends up in drive_background.log.
        # This isn't serialized in any way, and just relies on the
        # relatively low volume of output for readability.
        # XXX If this is insufficient, one option would be to use
        # XXX capture_output mode in the call below, and then flush
        # XXX all the job output in a single write, to reduce the
        # XXX chances of intermixing with other jobs.
        return_code = subprocess.call(
                [PathHelper.website_root+"runner/run_process.py"
                ,p.rundir
                ,str(p_id)
                ,p.pidfile()
                ,p.logfile()
                ,p.cmd
                ])
        logger.info("Done subprocess for %s, %s", p_id, return_code)
    except Exception as ex:
        logger.error("got exception '%s' spawning job %d"
                ,str(ex)
                ,p_id
                )
        return_code = Process.launch_exception
    try:
        logger.info("Stopping process %s %s", p_id, return_code)
        Process.stop(p_id,return_code)
    except Exception as ex:
        # Just log it on its way up to make sure we know about it.
        logger.error("got exception '%s' stopping job %d"
                ,str(ex)
                ,p_id
                )
        raise

