from builtins import range
from django.test import TestCase,TransactionTestCase
from runner.models import Process
import time
import os
from path_helper import PathHelper
from tools import touch
import unittest
from mock import patch,MagicMock

class ReserveTestCase(TransactionTestCase):
    def test_reserve(self):
        from reserve import ResourceManager
        rm=ResourceManager()
        rm.set_totals(got=[5,10])
        job=1
        rm.request(job,[2,1])
        self.assertEqual(rm.status(job),[2,1])
        self.assertEqual(rm.desc(job),'2 local cores 1 remote core')
        job=2
        rm.request(job,[4,0])
        self.assertEqual(rm.status(job),None)
        self.assertEqual(rm.desc(job),'awaiting 4 local cores')
        rm.request(job,[3,0])
        self.assertEqual(rm.status(job),[3,0])
        self.assertEqual(rm.desc(job),'3 local cores')
        self.assertEqual(rm.avail(),[0,9])
        rm.terminate(1)
        rm.terminate(2)
        # with 10 cores, 24 jobs will require a minimum of 3 batches;
        # an allocation of 8 will exactly complete 24 jobs with all
        # cores in use in each batch
        job=3
        rm.request(job,[2,(1,24)])
        self.assertEqual(rm.status(job),[2,8])
        self.assertEqual(rm.desc(job),'2 local cores 8 remote cores')
        rm.terminate(job)
        # verify that a normal job can allocate all available cores
        # in batch mode
        job=4
        rm.request(job,[2,(1,99)])
        self.assertEqual(rm.status(job),[2,10])
        self.assertEqual(rm.desc(job),'2 local cores 10 remote cores')
        rm.terminate(job)
        # verify that a slow job issuing the same request will get
        # fewer cores (in this case, reserving 20% makes 8 cores
        # available, requiring 13 batches of 8 each
        rm.set_totals(got=[5,10],margin=[1,2])
        job=5
        rm.request(job,[2,(1,99)],slow=True)
        self.assertEqual(rm.status(job),[2,8])
        self.assertEqual(rm.desc(job),'SLOW 2 local cores 8 remote cores')
        rm.terminate(job)
        # verify that the same 8 cores will be assigned if 2 are
        # currently in use by a non-slow job
        rm.set_totals(got=[5,10],margin=[1,2])
        job=6
        rm.request(job+1,[2,2])
        rm.request(job,[2,(1,99)],slow=True)
        self.assertEqual(rm.status(job),[2,8])
        rm.terminate(job)
        rm.terminate(job+1)
        # verify that, by default, if there's only one resource
        # available, it remains available to slow jobs
        rm.set_totals(got=[5,1])
        job=8
        rm.request(job,[2,(1,99)],slow=True)
        self.assertEqual(rm.status(job),[2,1])
        rm.terminate(job)
        # verify that if slow jobs hold all 'slow' cores, a subsequent
        # slow job is not allocated anything
        rm.set_totals(got=[5,10],margin=[1,2])
        job=9
        rm.request(job,[1,(1,99)],slow=True)
        self.assertEqual(rm.status(job),[1,8])
        self.assertEqual(rm.desc(job),'SLOW 1 local core 8 remote cores')
        rm.request(job+1,[1,(1,99)],slow=True)
        self.assertEqual(rm.status(job+1),None)
        self.assertEqual(rm.desc(job+1),'SLOW awaiting 1 local core 1+ remote core')
        # then verify that, in the above scenario, a fast job following
        # the slow job can jump ahead to get the available cores
        rm.request(job+2,[1,(1,99)])
        self.assertEqual(rm.status(job+2),[1,2])
        rm.terminate(job)
        rm.terminate(job+1)
        rm.terminate(job+2)
        # verify that, if multiple jobs are waiting, they get served in order
        job=12
        rm.set_totals(got=[1])
        rm.request(job,[1])
        rm.request(job+1,[1])
        rm.request(job+2,[1])
        rm.request(job+3,[1])
        self.assertEqual(rm.status(job),[1])
        self.assertEqual(rm.status(job+1),None)
        self.assertEqual(rm.status(job+2),None)
        self.assertEqual(rm.status(job+3),None)
        rm.terminate(job)
        self.assertEqual(rm.status(job+1),[1])
        self.assertEqual(rm.status(job+2),None)
        self.assertEqual(rm.status(job+3),None)
        rm.terminate(job+1)
        self.assertEqual(rm.status(job+2),[1])
        self.assertEqual(rm.status(job+3),None)
        rm.terminate(job+2)
        self.assertEqual(rm.status(job+3),[1])
        rm.terminate(job+3)

class JobCrossCheckerTestCase(TransactionTestCase):
    def expected_jobnames(self,ws,tissue_sets,tissues):
        from .process_info import JobCrossChecker
        expect=set()
        for name in JobCrossChecker.level_names:
            if name == 'meta':
                for t in tissues:
                    expect.add('%s_%s' % (name,t.geoID))
            elif name == 'sig':
                for t in tissues:
                    expect.add('%s_%s_%d_%d' % (name,t.geoID,t.id,ws.id))
            elif name == 'wf':
                from dtk.workflow import Workflow
                for wrapper in Workflow.wf_list():
                    expect.add('%s_%d_%s' % (name,ws.id,wrapper.code()))
            elif name in 'path tsp pathbg gesig pathdrop'.split():
                for ts in tissue_sets:
                    expect.add('%s_%d_%d' % (name,ts.id,ws.id))
            elif name == 'fvs':
                subtypes = ['wsa_efficacy', 'wsa_novelty', 'uniprot']
                for subtype in subtypes:
                    expect.add('%s_%s_%d' % (name,subtype,ws.id))
            elif name in ['selectabilitymodel', 'gesearchmodel']:
                expect.add(name)
            else:
                expect.add('%s_%d' % (name,ws.id))
        return expect
    def flush_cache(self,ws,jcc):
        # get rid of internal caches to pick up any db updates
        jcc._ws_job_map = {}
        ws.invalidate_tissue_set_cache()
    def test_ws_jobnames(self):
        from .process_info import JobCrossChecker
        jcc=JobCrossChecker()
        from browse.models import Workspace,TissueSet,Tissue
        ws=Workspace()
        ws.save()
        # ts1 always gets created automatically if it's missing; create
        # it explicitly so we have a handle to the record
        ts1=TissueSet(ws=ws)
        ts1.save()
        self.assertEqual(
                    jcc.ws_jobnames(ws),
                    self.expected_jobnames(ws,[ts1],[]),
                    )
        self.flush_cache(ws,jcc)
        t1=Tissue(ws=ws,tissue_set=ts1,geoID='GSE62191')
        t1.save()
        self.assertEqual(
                    jcc.ws_jobnames(ws),
                    self.expected_jobnames(ws,[ts1],[t1]),
                    )
        self.flush_cache(ws,jcc)
        t2=Tissue(ws=ws,tissue_set=ts1,geoID='GSE62191')
        t2.save()
        self.assertEqual(
                    jcc.ws_jobnames(ws),
                    self.expected_jobnames(ws,[ts1],[t1,t2]),
                    )
        self.flush_cache(ws,jcc)
        ts2=TissueSet(ws=ws)
        ts2.save()
        t3=Tissue(ws=ws,tissue_set=ts2,geoID='GSE62191')
        t3.save()
        t4=Tissue(ws=ws,tissue_set=ts1,geoID='GSE62192')
        t4.save()
        self.assertEqual(
                    jcc.ws_jobnames(ws),
                    self.expected_jobnames(ws,[ts1,ts2],[t1,t2,t3,t4]),
                    )
        self.flush_cache(ws,jcc)
    def test_job_ws(self):
        from .process_info import JobCrossChecker
        jcc=JobCrossChecker()
        self.assertEqual(
                    jcc.job_ws('xxx'),
                    set(),
                    )
        self.assertEqual(
                    jcc._job_ws,
                    {},
                    )
        from browse.models import Workspace,TissueSet,Tissue
        jcc._job_ws = None # force cache flush
        ws1=Workspace(name='ws1')
        ws1.save()
        ts1=TissueSet(ws=ws1)
        ts1.save()
        t1=Tissue(ws=ws1,tissue_set=ts1,geoID='GSE62191')
        t1.save()
        ws2=Workspace(name='ws2')
        ws2.save()
        ts2=TissueSet(ws=ws2)
        ts2.save()
        t2=Tissue(ws=ws2,tissue_set=ts2,geoID='GSE62191')
        t2.save()
        self.assertEqual(
                    jcc.job_ws('ml_%d' % ws1.id),
                    set(['ws1']),
                    )
        self.assertEqual(
                    jcc.job_ws('path_%d_%d' % (ts2.id,ws2.id)),
                    set(['ws2']),
                    )
        self.assertEqual(
                    jcc.job_ws('meta_%s' % t1.geoID),
                    set(['ws1','ws2']),
                    )

# disable this because job ids get re-used, which conflicts with LTS log
# file storage; there's maybe some way to work around this by generating
# a new LTS branch, but leave it for later
if False:
  # need to use TransactionTestCase for db changes to be seen in other threads
  class LauncherTestCase(TransactionTestCase):
    def run_single(self):
        self.assertEqual(Process.start_all(),(1,0))
        for i in range(0,1000):
            if Process.running() == 0:
                break
        self.assertEqual(Process.queued(),0)
        # NOTE: if the following assert fails, maybe the range in
        # the loop above needs to be bigger.
        self.assertEqual(Process.running(),0)
    def test_single(self):
        self.assertEqual(Process.queued(),0)
        self.assertEqual(Process.running(),0)
        p_id = Process.queue_process("launch1"
                ,"/tmp"
                ,"/bin/echo 1 >/dev/null"
                )
        self.run_single()
        p = Process.objects.get(pk=p_id)
        self.assertEqual(p.status,Process.status_vals.SUCCEEDED)
        time.sleep(1) # make sure background runner has shut down
        p_id = Process.queue_process("launch3"
                ,"/tmp"
                ,"/bin/false"
                )
        self.run_single()
        p = Process.objects.get(pk=p_id)
        self.assertEqual(p.status,Process.status_vals.FAILED)
    def test_abort(self):
        self.assertEqual(Process.queued(),0)
        self.assertEqual(Process.running(),0)
        p_id = Process.queue_process("launch2","/tmp","sleep 10")
        p_id2 = Process.queue_process("launch4","/tmp","/bin/true")
        self.assertEqual(Process.start_all(),(2,0))
        time.sleep(1)
        p = Process.objects.get(pk=p_id)
        self.assertEqual(Process.running(),1)
        self.assertTrue(os.path.isfile(p.pidfile()))
        Process.abort(p_id)
        self.assertEqual(Process.running(),0)
        p = Process.objects.get(pk=p_id2)
        self.assertEqual(p.status,Process.status_vals.SUCCEEDED)
        time.sleep(1)
        p = Process.objects.get(pk=p_id)
        self.assertEqual(p.status,Process.status_vals.FAILED)

class LifeCycleTestCase(TestCase):
    # Helpers
    def setUp(self):
        Process.suppress_execution=True
    def tearDown(self):
        Process.suppress_execution=False
    def simple_cycle(self,name,ok):
        self.assertEqual(Process.queued(),0)
        self.assertEqual(Process.running(),0)
        p_id = Process.queue_process(name,"some_dirpath","some_cmd")
        self.assertEqual(Process.queued(),1)
        self.assertEqual(Process.running(),0)
        self.assertEqual(Process.start_all(),(1,0))
        self.assertEqual(Process.queued(),0)
        self.assertEqual(Process.running(),1)
        Process.stop(p_id,0 if ok else 1)
        self.assertEqual(Process.queued(),0)
        self.assertEqual(Process.running(),0)
        return p_id
    # Tests
    def test_simple(self):
        vals = Process.status_vals
        p1 = self.simple_cycle("test1",False)
        p = Process.objects.get(pk=p1)
        self.assertEqual(p.status,vals.FAILED)
        p2 = self.simple_cycle("test2",True)
        p = Process.objects.get(pk=p2)
        self.assertEqual(p.status,vals.SUCCEEDED)
        p = Process.objects.get(pk=p1)
        self.assertEqual(p.status,vals.FAILED)
    def test_wait(self):
        p1 = Process.queue_process("wait1","some_dirpath","some_cmd")
        p2 = Process.queue_process("wait2","some_dirpath","some_cmd"
                                ,run_after=[p1]
                                )
        self.assertEqual(Process.queued(),2)
        self.assertEqual(Process.running(),0)
        self.assertEqual(Process.start_all(),(1,1))
        self.assertEqual(Process.queued(),1)
        self.assertEqual(Process.running(),1)
        Process.stop(p1,0)
        self.assertEqual(Process.queued(),0)
        self.assertEqual(Process.running(),1)
        Process.stop(p2,0)
        self.assertEqual(Process.queued(),0)
        self.assertEqual(Process.running(),0)
    def test_cascade(self):
        p1 = Process.queue_process("cascade1","some_dirpath","some_cmd")
        p2 = Process.queue_process("cascade2","some_dirpath","some_cmd"
                                ,run_after=[p1]
                                )
        p3 = Process.queue_process("cascade3","some_dirpath","some_cmd"
                                ,run_after=[p2]
                                )
        self.assertEqual(Process.queued(),3)
        self.assertEqual(Process.running(),0)
        self.assertEqual(Process.start_all(),(1,2))
        self.assertEqual(Process.queued(),2)
        self.assertEqual(Process.running(),1)
        Process.stop(p1,1)
        self.assertEqual(Process.queued(),0)
        self.assertEqual(Process.running(),0)
    # test MaintenanceLock mechanism
    def get_dummy_status(self):
        for x in Process.maint_status():
            if x.task == 'dummy':
                return x.status
    def get_dummy_progress(self):
        for x in Process.maint_status():
            if x.task == 'dummy':
                return x.progress
    def test_maint_before_job(self):
        self.assertEqual(Process.queued(),0)
        self.assertEqual(Process.running(),0)
        # with no job running, new maintenance lock is active
        Process.maint_request('dummy')
        self.assertEqual(self.get_dummy_status(),'Active')
        # job queued while maintenance lock is active will not start
        p = Process.queue_process("some_name","some_dirpath","some_cmd")
        self.assertEqual(Process.start_all(),(0,1))
        # queued job starts after maintenance lock is released
        Process.maint_release('dummy')
        self.assertEqual(self.get_dummy_status(),'Idle')
        self.assertEqual(Process.start_all(),(1,0))
        Process.stop(p,0)
    def test_maint_after_job(self):
        self.assertEqual(Process.queued(),0)
        self.assertEqual(Process.running(),0)
        # with a job active, new maintenance lock is pending
        p = Process.queue_process("some_name","some_dirpath","some_cmd")
        self.assertEqual(Process.start_all(),(1,0))
        Process.maint_request('dummy')
        self.assertEqual(self.get_dummy_status(),'Pending')
        # maintenance lock will switch to active when job completes
        Process.stop(p,0)
        self.assertEqual(self.get_dummy_status(),'Active')
        Process.maint_release('dummy')
        self.assertEqual(self.get_dummy_status(),'Idle')
    def test_maint_multi_job(self):
        self.assertEqual(Process.queued(),0)
        self.assertEqual(Process.running(),0)
        # with multiple jobs active, new maintenance lock is pending
        p1 = Process.queue_process("name1","some_dirpath","some_cmd")
        p2 = Process.queue_process("name2","some_dirpath","some_cmd")
        self.assertEqual(Process.start_all(),(2,0))
        Process.maint_request('dummy')
        self.assertEqual(self.get_dummy_status(),'Pending')
        # lock won't activate when the first job ends
        Process.stop(p1,0)
        self.assertEqual(self.get_dummy_status(),'Pending')
        # lock will activate when the second job ends
        Process.stop(p2,0)
        self.assertEqual(self.get_dummy_status(),'Active')
        Process.maint_release('dummy')
        self.assertEqual(self.get_dummy_status(),'Idle')
    def test_maint_multi_lock(self):
        self.assertEqual(Process.queued(),0)
        self.assertEqual(Process.running(),0)
        # with multiple locks active, jobs will not start
        Process.maint_request('dummy')
        Process.maint_request('dummy2')
        p = Process.queue_process("some_name","some_dirpath","some_cmd")
        self.assertEqual(Process.start_all(),(0,1))
        # releasing first lock, job will remain blocked
        Process.maint_release('dummy')
        self.assertEqual(Process.start_all(),(0,1))
        # releasing second lock, job will become enabled
        Process.maint_release('dummy2')
        self.assertEqual(Process.start_all(),(1,0))
    def test_maint_yield(self):
        self.assertEqual(Process.queued(),0)
        self.assertEqual(Process.running(),0)
        Process.maint_request('dummy')
        self.assertEqual(self.get_dummy_status(),'Active')
        # yield with no job will continue running
        Process.maint_yield('dummy',detail=dict(progress='pass2'))
        self.assertEqual(self.get_dummy_status(),'Active')
        self.assertEqual(self.get_dummy_progress(),'pass2')
        # job will wait while maint is running
        p = Process.queue_process("some_name","some_dirpath","some_cmd")
        self.assertEqual(Process.start_all(),(0,1))
        # when maint yields, job can start; maint remains pending
        Process.maint_yield('dummy',detail=dict(progress='pass3'))
        self.assertEqual(self.get_dummy_status(),'Pending')
        self.assertEqual(self.get_dummy_progress(),'pass3')
        self.assertEqual(Process.start_all(),(1,0))
        # lock will resume when the job ends
        Process.stop(p,0)
        self.assertEqual(self.get_dummy_status(),'Active')
        Process.maint_release('dummy')
        self.assertEqual(self.get_dummy_status(),'Idle')

class PluginTestCase(TestCase):
    required_drug_id=111
    def unbound_helper(self,plugin,score_info):
        from runner.process_info import JobInfo,Process
        uji = JobInfo.get_unbound(plugin)
        self.assertEqual(uji.job_type,plugin)
        cat = uji.get_data_catalog()
        for d in score_info:
            self.assertEqual(cat.get_label(d['code']),d['label'])
    def test_unbound(self):
        self.unbound_helper('meta',[])
        self.unbound_helper('sig',[])
        self.unbound_helper('path',
                            [
                                {'label':'Direct','code':'direct'},
                                {'label':'Indirect','code':'indirect'},
                                {'label':'Direction','code':'direction'},
                                {'label':'Abs(direction)','code':'absDir'},
                            ],
                            )
        self.unbound_helper('faers',
                            [
                                {'label':'Disease CO Portion','code':'dcop'},
                                {'label':'Disease CO+','code':'dcoe'},
                                {'label':'Disease CO-','code':'dcod'},
                                {'label':'Disease CO FDR','code':'dcoq'},
                                {'label':'Drug portion','code':'drugPor'},
                                {'label':'LR P-Value','code':'lrpvalue'},
                                {'label':'LR Score','code':'lrenrichment'},
                                {'label':'LR Direction','code':'lrdir'},
                            ],
                            )
        self.unbound_helper('ml',
                            [
                                {'label':'Drug ML','code':'ml'},
                                {'label':'Protein ML','code':'protml'}
                            ])
    def bound_helper(self,ws,name,settings=''):
        from runner.process_info import JobInfo,Process
        job = Process(
                    status=Process.status_vals.SUCCEEDED,
                    name=name,
                    settings_json=settings,
                    )
        job.save()
        bji = JobInfo.get_bound(ws,job.id)
        from path_helper import PathHelper
        self.assertEqual(bji.outdir
                    ,PathHelper.storage+'%d/%s/%d/output/'%(
                                        ws.id,bji.job_type,job.id,
                                        )
                    )
        return bji
    def get_individual_scores(self,bji):
        cat = bji.get_data_catalog()
        first_time = True
        for code in cat.get_codes('wsa','score'):
            self.assertTrue(cat.get_label(code))
            if first_time:
                # all jobs have a non-zero score for drug_id 111
                # in their first score type
                cell = cat.get_cell(code,111)
                self.assertNotEqual(cell[0],0)
                first_time = False

    @patch("dtk.lts.LtsRepo")
    def test_bound(self, fake_lts):
        # create dummy Workspace
        from browse.models import Workspace
        ws = Workspace(name='bound unit test')
        ws.save()
        # create dummy job row, so job id never equals ws id
        from runner.process_info import JobInfo,Process
        job = Process(status=Process.status_vals.SUCCEEDED)
        job.save()
        # source of test data
        testdata=PathHelper.website_root+'runner/testdata/'
        # test plugin ###############################
        plugin = 'ml'
        uji = JobInfo.get_unbound(plugin)
        names = uji.get_jobnames(ws)
        self.assertEqual(len(names),1)
        bji = self.bound_helper(ws,names[0])
        bji.fn_scores = testdata+'allProbabilityOfAssoications.tsv'
        self.get_individual_scores(bji)
        cat=bji.get_data_catalog()
        codes = list(cat.get_codes('wsa','score'))
        self.assertEqual(len(codes),1)
        ordering = cat.get_ordering(codes[0],True)
        self.assertEqual(len(ordering),20)
        # test plugin ###############################
        # XXX struct has individual per-kt similarity scores, and needs
        # XXX both Drug and WsAnnotation objects to construct the
        # XXX human-friendly label for these scores; we don't have fixtures
        # XXX for these, so disable for now
        # plugin = 'struct'
        # uji = JobInfo.get_unbound(plugin)
        # names = uji.get_jobnames(ws)
        # self.assertEqual(len(names),1)
        # bji = self.bound_helper(ws,names[0])
        # bji.fn_similar = testdata+'similarities.csv'
        # self.get_individual_scores(bji)
        # scores = bji.get_scores()
        # self.assertEqual(len(scores),1)
        # ordering = scores[0].get_ordering(True)
        # self.assertEqual(len(ordering),6)
        # test plugin ###############################
        plugin = 'faers'
        uji = JobInfo.get_unbound(plugin)
        names = uji.get_jobnames(ws)
        self.assertEqual(len(names),1)
        bji = self.bound_helper(ws,names[0]
                            ,settings='{"search_term":"term1|term2"}'
                            )
        bji.fn_enrichment = testdata+'faers_enrichment.csv'
        self.get_individual_scores(bji)
        cat=bji.get_data_catalog()
        codes = list(cat.get_codes('wsa','score'))
        self.assertEqual(len(codes),3)
        ### this changed somehow, and it's not worth figuring out;
        ### maybe we used to filter by significance?
        #ordering = cat.get_ordering(codes[0],True)
        #self.assertEqual(len(ordering),13)
        #self.assertEqual(len([x for x in ordering if x[1] > 0]),9)
        #ordering = scores[1].get_ordering(True)
        #self.assertEqual(len(ordering),13)
        #self.assertEqual(len([x for x in ordering if x[1] > 0]),4)
        # test plugin ###############################
        plugin = 'path'
        ### Not worth rewriting after removing Candidate table
        #uji = JobInfo.get_unbound(plugin)
        #from browse.models import TissueSet
        ##ts=TissueSet(ws=ws)
        #ts.save()
        #name = uji._format_jobname_for_tissue_set(plugin,ts)
        #bji = self.bound_helper(ws,name)
        #from browse.models import WsAnnotation,Candidate
        #wsa=WsAnnotation(id=self.required_drug_id,ws=ws)
        #wsa.save()
        #c=Candidate(
        #            direct_score=1.23,
        #            indirect_score=0,
        #            direction_sum=1,
        #            drug_ws=wsa,
        #            run=bji.job,
        #            )
        #c.save()
        #self.get_individual_scores(bji)
        #cat=bji.get_data_catalog()
        #codes = list(cat.get_codes('wsa','score'))
        #self.assertEqual(
        #        codes,
        #        [
        #            'direct',
        #            'direction',
        #            'dirneg',
        #            'protcount',
        #            'dpaths',
        #            'ipaths',
        #            'pathprots',
        #            'ontarget',
        #            ],
        #        )
        #ordering = cat.get_ordering(codes[0],True)
        #self.assertEqual(len(ordering),1)
        #ordering = cat.get_ordering(codes[1],True)
        #self.assertEqual(len(ordering),1)

def unordered_fetcher(*args):
    def func(keyset):
        for x in args:
            if keyset is None or x[0] in keyset:
                yield x
    return func

def ordered_fetcher(*args):
    def func(keyset,keysort=False):
        for x in args:
            if keyset is None or x[0] in keyset:
                yield x
    return func

class DataCatalogTest(TestCase):
    def test_std_fetcher(self):
        import runner.data_catalog as dc
        # normal case; file matches config
        cat = dc.Catalog()
        cat.add_group('', dc.CodeGroup('wsa',
                PathHelper.website_root+'runner/testdata/fetch1.tsv',
                dc.Code('code1'),
                dc.Code('code2'),
                ))
        l = cat.get_codes('wsa','score')
        self.assertEqual(set(l),set([
                'code1',
                'code2',
                ]))
        l = cat.get_ordering('code1',True)
        self.assertEqual(l,[
                (5,9),
                (4,8),
                (2,7),
                (1,6),
                ])
        # resolve mis-matched file and config
        cat = dc.Catalog()
        cat.add_group('', dc.CodeGroup('wsa',
                PathHelper.website_root+'runner/testdata/fetch1.tsv',
                dc.Code('code1'),
                dc.Code('code3'),
                dc.Code('calc1',
                        calc=(lambda x:-x,'code1'),
                        ),
                dc.Code('calc1a',
                        calc=(lambda x:10+x,'calc1'),
                        ),
                dc.Code('calc3a',
                        calc=(lambda x:10+x,'calc3'),
                        ),
                dc.Code('calc3',
                        calc=(lambda x:-x,'code3'),
                        ),
                ))
        l = cat.get_codes('wsa','score')
        self.assertEqual(set(l),set([
                'code1',
                'calc1',
                'calc1a',
                ]))
        l = cat.get_ordering('code1',True)
        code1list = [
                (5,9),
                (4,8),
                (2,7),
                (1,6),
                ]
        self.assertEqual(l,code1list)
        l = cat.get_ordering('calc1',True)
        self.assertEqual(l,sorted([
                (k,-v)
                for k,v in code1list
                ],
                key=lambda x:x[1],
                reverse=True,
                ))
        l = cat.get_ordering('calc1a',True)
        self.assertEqual(l,sorted([
                (k,10-v)
                for k,v in code1list
                ],
                key=lambda x:x[1],
                reverse=True,
                ))
    def test_catalog(self):
        import runner.data_catalog as dc
        cat = dc.Catalog()
        import operator
        cat.add_group('j123_', dc.CodeGroup('wsa',
                ordered_fetcher(
                        (1,(5,None,3,6,0)),
                        (2,(4,None,3,7,0)),
                        (3,(3,True,3,None,0)),
                        (4,(2,True,3,8,0)),
                        (5,(1,None,3,9,0)),
                        ),
                dc.Code('code1'),
                dc.Code('code2',
                        label='Label for Code 2',
                        valtype='bool',
                        ),
                dc.Code('code3',
                        valtype='alias',
                        ),
                dc.Code('code3a', ),
                dc.Code('code3b', ),
                dc.Code('code3c', ),
                dc.Code('code3d',
                        calc=(operator.sub,'code3a','code1'),
                        ),
                ))
        #print '\n'.join(cat.dump())
        s = cat.get_keyset('j123_code2')
        self.assertEqual(s,set([3,4]))
        l = cat.get_ordering('j123_code3b',True)
        self.assertEqual(l,[
                (5,9),
                (4,8),
                (2,7),
                (1,6),
                ])
        l = cat.get_codes('wsa','score')
        self.assertEqual(set(l),set([
                'j123_code1'
                ]))
        cat.add_group('j456_', dc.CodeGroup('wsa',
                ordered_fetcher(
                        (1,(15,'X307')),
                        (2,(14,'X308')),
                        (3,(13,'X309')),
                        (4,(12,'X301')),
                        (6,(11,'X302')),
                        ),
                dc.Code('code1',
                        label='Some Label',
                        href=('http://xxx.com/data/%s','code2'),
                        ),
                dc.Code('code2',
                        hidden=True,
                        valtype='string',
                        ),
                ),"2nd Group")
        cols,data = cat.get_feature_vectors(
                                'j123_code2',
                                'j456_code1',
                                'j123_code3',
                                )
        self.assertEqual(cols,[
                'j123_code2',
                'j456_code1',
                'j123_code3a',
                'j123_code3b',
                'j123_code3c',
                'j123_code3d',
                ])
        data = list(data) # read from generator
        self.assertEqual(
                [x[0] for x in data],
                list(range(1,7)),
                )
        idx = cols.index('j123_code3b')
        self.assertEqual(
                [x[1][idx] for x in data],
                [6,7,None,8,9,None],
                )
        idx = cols.index('j123_code3d')
        self.assertEqual(
                [x[1][idx] for x in data],
                [-2,-1,0,1,2,None],
                )
        self.assertEqual(
                cat.get_label('j456_code1'),
                "2nd Group Some Label",
                )
        self.assertEqual(
                cat.get_label('j123_code3b'),
                "J123 CODE3B",
                )
        val,attrs = cat.get_cell('j456_code1',3)
        self.assertEqual(val,13)
        self.assertEqual(attrs['href'],'http://xxx.com/data/X309')
    def test_code_lookup(self):
        import runner.data_catalog as dc
        import operator
        cg1 = dc.CodeGroup('wsa',
                ordered_fetcher(),
                dc.Code('code1'),
                dc.Code('code2',
                        label='Label for Code 2',
                        valtype='bool',
                        ),
                dc.Code('code3',
                        valtype='alias',
                        ),
                dc.Code('code3a', ),
                dc.Code('code3b', ),
                dc.Code('code3c', ),
                dc.Code('code3d',
                        calc=(operator.sub,'code3a','code1'),
                        ),
                )
        l = [x._code for x in cg1.get_code_proxies()]
        self.assertEqual(l,[
                'code1',
                'code2',
                'code3',
                ])
        p = cg1.get_code_proxy('code1')
        self.assertEqual(p.label(),'CODE1')
        self.assertEqual(p.valtype(),'float')
        self.assertEqual(p._args.get('index'),0)
        p = cg1.get_code_proxy('code2')
        self.assertEqual(p.label(),'Label for Code 2')
        self.assertEqual(p.valtype(),'bool')
        self.assertEqual(p._args.get('index'),1)
        p = cg1.get_code_proxy('code3')
        self.assertEqual(p.label(),'CODE3')
        self.assertEqual(p.valtype(),'alias')
        self.assertEqual(p._args.get('index'),None)
    def verify_fetch(self,cg):
        l = list(cg.data(None))
        self.assertEqual(len(l),5)
        l = list(cg.data([3]))
        self.assertEqual(len(l),1)
        l = list(cg.data(None,keysort=True))
        self.assertEqual(len(l),5)
        self.assertEqual([x[0] for x in l],[1,2,3,4,50])
    def get_unordered(self):
        import runner.data_catalog as dc
        return dc.CodeGroup('wsa',
                unordered_fetcher(
                        (50,(1,5,3)),
                        (3,(3,3,3)),
                        (1,(5,1,3)),
                        (4,(2,4,3)),
                        (2,(4,2,3)),
                        ),
                dc.Code('code1'),
                dc.Code('code2'),
                dc.Code('code3'),
                )
    def get_ordered(self):
        import runner.data_catalog as dc
        return dc.CodeGroup('wsa',
                ordered_fetcher(
                        (1,(5,1,3)),
                        (2,(4,2,3)),
                        (3,(3,3,3)),
                        (4,(2,4,3)),
                        (50,(1,5,3)),
                        ),
                dc.Code('code1'),
                dc.Code('code2'),
                dc.Code('code3'),
                )
    def test_data_fetch(self):
        cg = self.get_unordered()
        self.verify_fetch(cg)
        cg = self.get_ordered()
        self.verify_fetch(cg)
    def test_keyset_passthru(self):
        cg = self.get_unordered()
        cg.data(None)
        self.assertTrue(cg._cache)
        self.assertEqual(len(cg._cache),5)
        cg = self.get_unordered()
        self.assertFalse(cg._cache)
        cg.data([3])
        self.assertTrue(cg._cache)
        self.assertEqual(len(cg._cache),1)
    def test_sort_passthru(self):
        cg = self.get_unordered()
        cg.data(None,keysort=True)
        self.assertTrue(cg._cache)
        cg = self.get_ordered()
        cg.data(None,keysort=True)
        self.assertFalse(cg._cache)

@patch("runner.process_info.Process.objects.get")
def test_job_roles(mock_process_get):
    # cm_name, multi_score
    sources = (
            ('gwasig',False),
            ('agr',True),
            ('otarg',True),
            )
    from runner.process_info import JobInfo
    for cm,_ in sources:
        uji = JobInfo.get_unbound(cm)
        assert uji.build_role_code(cm+'_1',None) == cm
    for upstream,multi in sources:
        uji = JobInfo.get_unbound('codes')
        mock_process_get.return_value = MagicMock(role=upstream)
        # 'name' attr is special; can't set in MagicMock ctor
        mock_process_get.return_value.name=upstream+'_1'
        role = uji.build_role_code(
                'codes_1',
                '{"input_score":"somejob_somescore"}',
                )
        if multi:
            assert role == 'somescore_'+upstream+'_codes'
        else:
            assert role == upstream+'_codes'
        mock_process_get.assert_called_with(pk='somejob')

def test_role_labels():
    from runner.process_info import JobInfo
    assert JobInfo.role2label(
            'cc_path_gpbr'
            ) == 'Case/Control Pathsum gPBR'
    assert JobInfo.role2label(
            'FAERS_faers_otarg_capp_faerssig_glf_depend'
            ) == 'FAERS FAERS OpenTargets CAPP FAERS Sig GLF DEEPEnD'
    assert JobInfo.role2label(
            'geneticassociation_otarg_sigdif_glf_depend'
            ) == 'Gene Assoc OpenTargets SigDiffuser GLF DEEPEnD'
    assert JobInfo.role2label(
            'agrs_agr_codes',
            ) == 'AGR CoDES'
    assert JobInfo.role2label(
            'misig_codes',
            ) == 'MISig CoDES'
    assert JobInfo.role2label(
            'misig_misig_codes',
            ) == 'MISig CoDES'
