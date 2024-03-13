import pytest
from mock import patch
from dtk.tests import make_ws

import dtk.copy_ws as cp

@pytest.fixture
def src_ws(make_ws):
    ws_attrs = [
        ('DB001', 'canonical', 'Lepirudin'),
        ('DB002', 'canonical', 'Cetuximab'),
        ('DB003', 'canonical', 'Dornase alfa'),
        ]
    ws = make_ws(ws_attrs,name='src_ws')
    return ws

@pytest.fixture
def dst_ws(db):
    from browse.models import Workspace
    ws = Workspace.objects.create(name='dst_ws')
    return ws

class MockLtsRepo:
    # client should set up active_branch and tmpdir class members
    instances = []
    def __init__(self,repo_name,branch,*,cacheable=False):
        assert branch == self.active_branch
        import os
        self._my_path = os.path.join(self.tmpdir,repo_name)
        from path_helper import make_directory
        make_directory(self._my_path)
        self.fetched = set()
        self.pushed = set()
        self.instances.append((repo_name,cacheable,self))
    def path(self):
        return self._my_path
    def lts_fetch(self,relpath):
        self.fetched.add(relpath)
    def lts_push(self,relpath):
        self.pushed.add(relpath)
    @classmethod
    def get(cls,repo_name,branch):
        for name,cacheable,instance in cls.instances:
            if cacheable and name == repo_name:
                return instance
        return cls(repo_name,branch,cacheable=True)

@patch("dtk.lts.LtsRepo",MockLtsRepo)
def test_copy_job(tmpdir,src_ws,dst_ws):
    ####
    # setup
    ####
    # set up mocked-out LTS to use temporary paths
    from path_helper import PathHelper
    MockLtsRepo.active_branch = PathHelper.cfg('lts_branch')
    MockLtsRepo.tmpdir = tmpdir
    # set up context
    ctx = cp.CopyContext(
            user='unit test exec',
            src_ws=src_ws,
            dst_ws=dst_ws,
            )
    # source workspace needs a tissue to create a sig job
    job_type = 'sig'
    from browse.models import Tissue
    ts = src_ws.get_tissue_sets()[0]
    tissue = Tissue.objects.create(
            ws = src_ws,
            tissue_set = ts,
            name = 'copy job test tissue',
            geoID = 'GSE37981',
            )
    # build dummy source job record
    from runner.process_info import JobInfo
    uji = JobInfo.get_unbound(job_type)
    jobname = uji.get_jobnames(src_ws)[0]
    print(jobname)
    from runner.models import Process
    src_job = Process.objects.create(
            name=jobname,
            status=Process.status_vals.SUCCEEDED,
            settings_json = '{"tissue_id":%d}'%tissue.id
            )
    # build dummy source directory content
    from runner.process_info import JobInfo
    src_bji = JobInfo.get_bound(ctx.src_ws,src_job.id)
    from path_helper import make_directory
    make_directory(src_bji.lts_abs_root)
    import os
    from dtk.files import FileDestination
    testfiles = [
            ('sigprot.tsv',[
                    ('uniprot','ev','dir','fold'),
                    ('A2RUC4','9.989e-01','1','5.378e+00'),
                    ('P17066','9.773e-01','1','4.352e+00'),
                    ]),
            ]
    # XXX test publish subdir?
    for fn,tuples in testfiles:
        path = os.path.join(src_bji.lts_abs_root,fn)
        with FileDestination(path) as fd:
            for x in tuples:
                fd.append(x)
    # the tissue is also needed in the destination workspace
    tissue.id = None
    tissue.ws = dst_ws
    tissue.tissue_set = dst_ws.get_tissue_sets()[0]
    tissue.save()
    ####
    # Now execute the client code
    ####
    dst_job_id = ctx.copy_sig_job(src_job.id)
    ####
    # check results
    ####
    # src job data fetched from lts?
    assert src_bji.lts_rel_root in src_bji.get_lts_repo().fetched
    dst_bji = JobInfo.get_bound(ctx.dst_ws,dst_job_id)
    # dst lts dir exists?
    assert os.path.isdir(dst_bji.lts_abs_root)
    # destination files present?
    for fn,_ in testfiles:
        path = os.path.join(dst_bji.lts_abs_root,fn)
        assert os.path.isfile(path)
    # destination files pushed to lts?
    assert dst_bji.lts_rel_root in dst_bji.get_lts_repo().pushed
    # check that the log file is present and pushed to lts
    from runner.common import LogRepoInfo
    lri = LogRepoInfo(dst_bji.job.id)
    assert os.path.isfile(lri.progress_path())
    assert os.path.isfile(lri.log_path())
    assert lri._path_parts[2] in lri.get_repo().pushed

@patch("dtk.lts.LtsRepo",MockLtsRepo)
def test_copy(tmpdir,src_ws,dst_ws):
    # set up mocked-out LTS to use temporary paths
    from path_helper import PathHelper
    MockLtsRepo.active_branch = PathHelper.cfg('lts_branch')
    MockLtsRepo.tmpdir = tmpdir
    # execute copies in order
    ctx = cp.CopyContext(
            user='unit test exec',
            src_ws=src_ws,
            dst_ws=dst_ws,
            )
    for subclass in cp.get_ordered_subclasses():
        # for each copy subclass, find the matching tester class
        tester = globals()[subclass.__name__]
        # run test
        tester.prep_source(ctx)
        try:
            kwargs = tester.kwargs
        except AttributeError:
            kwargs = {}
        subclass.copy(ctx,**kwargs)
        tester.check_dest(ctx)

class DiseaseNames:
    test_vocab='DisGeNet'
    test_val='C0036341'
    @classmethod
    def prep_source(cls,ctx):
        ctx.src_ws.set_disease_default(
                cls.test_vocab,
                cls.test_val,
                'unit test setup',
                )
    @classmethod
    def check_dest(cls,ctx):
        assert ctx.dst_ws.get_disease_default(cls.test_vocab) == cls.test_val

class Versions:
    test_attr = 'EvalDrugset'
    test_val = 'p3ts'
    @classmethod
    def prep_source(cls,ctx):
        from browse.models import VersionDefault
        VersionDefault.set_defaults(ctx.src_ws.id,[
                (cls.test_attr,cls.test_val),
                ],'unit test setup')
    @classmethod
    def check_dest(cls,ctx):
        from browse.models import VersionDefault
        ws_d = VersionDefault.get_defaults(ctx.dst_ws.id)
        glob_d = VersionDefault.get_defaults(None)
        assert glob_d[cls.test_attr] != cls.test_val
        assert ws_d[cls.test_attr] == cls.test_val

class DrugImports:
    kwargs=dict(
            ind_choices=["1"],
            )
    @classmethod
    def prep_source(cls,ctx):
        # the source WSAs are created in the src_ws fixture, but
        # here we tweak their attributes
        from browse.models import WsAnnotation
        key2src = {
                x.agent.test_col_id : x
                for x in WsAnnotation.objects.filter(ws=ctx.src_ws)
                }
        # DB001 matches ind_choices, and should be copied along with demerits
        src_wsa = key2src['DB001']
        src_wsa.indication = 1
        src_wsa.demerit_list = "1,2,3"
        src_wsa.save()
        # DB002 matches ind_choices, but already exists in the destination
        # workspace as invalid (to exercise the update path), so the old
        # indication should be preserved
        src_wsa = key2src['DB002']
        src_wsa.indication = 1
        src_wsa.save()
        dst_wsa,new = WsAnnotation.objects.get_or_create(
                ws=ctx.dst_ws,
                agent_id=src_wsa.agent_id,
                )
        dst_wsa.invalid = True
        dst_wsa.indication = 2
        dst_wsa.save()
        # DB003 doesn't match ind_choices, and so should show up as unknown
        src_wsa = key2src['DB003']
        src_wsa.indication = 3
        src_wsa.save()
    @classmethod
    def check_dest(cls,ctx):
        from browse.models import WsAnnotation
        key2src = {
                x.agent.test_col_id : x
                for x in WsAnnotation.objects.filter(ws=ctx.src_ws)
                }
        key2dst = {
                x.agent.test_col_id : x
                for x in WsAnnotation.objects.filter(ws=ctx.dst_ws)
                }
        assert len(key2src) == len(key2dst)
        assert key2src['DB001'].indication == key2dst['DB001'].indication
        assert key2src['DB001'].demerits() == key2dst['DB001'].demerits()
        assert key2dst['DB002'].indication == 2
        assert key2dst['DB002'].demerit_list == ''
        assert key2dst['DB003'].indication == 0
        assert key2dst['DB003'].demerit_list == ''

class AESearches:
    test_config = {
            ('term1','2020-01-01'):set([
                    ('GSE12345',0.75,'reject reason'),
                    ('GSE12222',0.70,None),
                    ]),
            ('term2','2020-01-01'):set([
                    ('GSE12666',0.75,'another reject reason'),
                    ]),
            }
    @staticmethod
    def dt_conv(s):
        import datetime
        dt = datetime.datetime.fromisoformat(s)
        from django.utils import timezone
        return timezone.make_aware(dt)
    @classmethod
    def prep_source(cls,ctx):
        from browse.models import AeSearch,AeAccession,AeDisposition,AeScore
        for (term,ts),results in cls.test_config.items():
            srch = AeSearch.objects.create(
                    ws=ctx.src_ws,
                    term=term,
                    when=cls.dt_conv(ts),
                    )
            for geoID,score,rejected in results:
                acc = AeAccession.objects.create(
                        ws=ctx.src_ws,
                        geoID=geoID,
                        )
                AeScore.objects.create(
                        search=srch,
                        accession=acc,
                        score=score,
                        )
                if rejected:
                    AeDisposition.objects.create(
                            accession=acc,
                            rejected=rejected,
                            )
        # put a duplicate search in dest; it shouldn't get replicated.
        # note that a time change is still considered a duplicate
        AeSearch.objects.create(
                ws=ctx.dst_ws,
                term='term1',
                when=cls.dt_conv('2020-01-02'),
                )
    @classmethod
    def check_dest(cls,ctx):
        from browse.models import AeSearch,AeDisposition
        mode_default = AeSearch.mode_vals.CC
        species_default = AeSearch.species_vals.human
        exp = {
                (term,mode_default,species_default):results
                for (term,ts),results in cls.test_config.items()
                }
        # every source search should exist in dest, but not necessarily
        # vice-versa
        srch_qs = AeSearch.objects.filter(ws=ctx.dst_ws)
        got = set(
                (srch.term,srch.mode,srch.species)
                for srch in srch_qs
                )
        assert set(exp.keys()).issubset(got)
        # every implied disposition should exist in dest
        for key,results in exp.items():
            for geoID,score,rejected in results:
                if rejected:
                    assert AeDisposition.objects.filter(
                            accession__ws = ctx.dst_ws,
                            accession__geoID = geoID,
                            mode = mode_default,
                            rejected = rejected,
                            ).exists()

class GE:
    t1_name='test 1'
    @classmethod
    def prep_source(cls,ctx):
        ts_list = ctx.src_ws.get_tissue_sets() # make sure defaults present
        from browse.models import Tissue
        t1 = Tissue.objects.create(
                ws=ctx.src_ws,
                tissue_set=ts_list[0],
                name=cls.t1_name,
                geoID='GSE12679',
                )
        from notes.models import Note
        Note.set(t1,'note','test_setup','some text',private=False)
        # XXX Additional test cases:
        # XXX - neither tissue or tissue set present in dst
        # XXX - tissue_set present, but not tissue
        # XXX - both present
        # XXX - with sig_result_job (job creation code could be factored
        # XXX   out of test_copy_job)
    @classmethod
    def check_dest(cls,ctx):
        from browse.models import Tissue
        t1 = Tissue.objects.get(ws=ctx.dst_ws,name=cls.t1_name)
        assert t1.tissue_set.name == 'default'
        dst_note = t1.get_note_text()
        assert 'some text' in dst_note
        assert 'Cloned' in dst_note
        # XXX verify parameters match
        # XXX verify Samples copied

def make_test_drugset(ws,name,agent_names):
    from browse.models import DrugSet,WsAnnotation
    ds = DrugSet(name=name,ws=ws)
    ds.save()
    for wsa in WsAnnotation.objects.filter(ws=ws):
        if wsa.agent.canonical in agent_names:
            ds.drugs.add(wsa)

class DrugSets:
    @classmethod
    def prep_source(cls,ctx):
        make_test_drugset(ctx.src_ws,'test ds 1',set([
                'Lepirudin',
                ]))
        make_test_drugset(ctx.src_ws,'test ds 2',set([
                'Lepirudin',
                'Dornase alfa',
                ]))
        make_test_drugset(ctx.dst_ws,'test ds 2',set([
                'Dornase alfa',
                ]))
    @classmethod
    def check_dest(cls,ctx):
        from browse.models import DrugSet
        for name,drugs in [
                ('test ds 1',['Lepirudin']),
                ('test ds 2',['Lepirudin','Dornase alfa']),
                ]:
            ds = DrugSet.objects.get(ws=ctx.dst_ws,name=name)
            got = set([wsa.agent.canonical for wsa in ds.drugs.all()])
            assert set(drugs) == got

def make_test_protset(ws,name,uniprots):
    from browse.models import ProtSet,Protein
    ps = ProtSet.objects.create(name=name,ws=ws)
    for uniprot in uniprots:
        prot,new = Protein.objects.get_or_create(uniprot=uniprot)
        ps.proteins.add(prot)

class ProtSets:
    @classmethod
    def prep_source(cls,ctx):
        make_test_protset(ctx.src_ws,'test ps 1',set([
                'P07305',
                ]))
        make_test_protset(ctx.src_ws,'test ps 2',set([
                'P02763',
                'P07305',
                ]))
        make_test_protset(ctx.dst_ws,'test ps 2',set([
                'P07305',
                ]))
    @classmethod
    def check_dest(cls,ctx):
        from browse.models import ProtSet
        for name,uniprots in [
                ('test ps 1',['P07305']),
                ('test ps 2',['P02763','P07305']),
                ]:
            ps = ProtSet.objects.get(ws=ctx.dst_ws,name=name)
            got = set([x.uniprot for x in ps.proteins.all()])
            assert set(uniprots) == got

class GWAS:
    @classmethod
    def prep_source(cls,ctx):
        from browse.models import GwasDataset
        gds = GwasDataset.objects.create(
                ws = ctx.src_ws,
                phenotype = 'schizophrenia',
                pubmed_id = '21926974'
                )
        from notes.models import Note
        Note.set(gds,'note','test_setup','some text',private=False)
    @classmethod
    def check_dest(cls,ctx):
        from browse.models import GwasDataset
        gds = GwasDataset.objects.get(
                ws = ctx.dst_ws,
                phenotype = 'schizophrenia',
                pubmed_id = '21926974'
                )
        dst_note = gds.get_note_text()
        assert 'some text' in dst_note
        assert 'Cloned' in dst_note
