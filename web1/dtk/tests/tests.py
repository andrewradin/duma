"""
This file demonstrates writing tests using the unittest module. These will pass
when you run "manage.py test".

"""
from django.test import TestCase,TransactionTestCase

import os
from path_helper import PathHelper,make_directory
import subprocess
import unittest
from mock import patch

import six
import pytest

from dtk.files import rm_readonly_tree

def rm_bare_lts_repo(name):
    assert name.startswith('test')
    from aws_op import Machine
    lts_machine = Machine.name_index['lts']
    lts_machine.run_remote_cmd('rm -rf 2xar/lts/%s.git' % name, venv=None)

def rm_lts_s3_repo(name):
    assert name.startswith('test')
    subprocess.call([
            's3cmd',
            'del',
            '--recursive',
            's3://2xar-duma-lts/%s/' % name,
            ])

class LtsTestCase(TestCase):
    repo_name='test'
    local_path = os.path.join(PathHelper.lts,repo_name)
    repo_name2='test2'
    second_root='/tmp/lts-testing'
    def _remove_previous(self):
        # remove leftovers from previous runs
        rm_bare_lts_repo(self.repo_name)
        rm_bare_lts_repo(self.repo_name2)
        rm_readonly_tree(self.local_path)
        rm_readonly_tree(self.second_root)
        rm_lts_s3_repo(self.repo_name)
        rm_lts_s3_repo(self.repo_name2)
        self.assertFalse(os.path.isdir(self.local_path))
    def _mkdir(self,repo,rel_dir):
        abs_path = os.path.join(repo.path(), rel_dir)
        make_directory(abs_path)
    def _mkfile(self,repo,rel_dir,fname):
        abs_path = os.path.join(repo.path(), rel_dir, fname)
        f = open(abs_path,'w')
        f.write('some content for file %s\n' % abs_path)
        f.close()
    def _verify_create(self):
        # instatiate LtsRepo object; should create repository
        from dtk.lts import LtsRepo
        self.main_lts=LtsRepo(self.repo_name,'master')
        self.assertEqual(self.main_lts.path(),self.local_path)
        self.assertTrue(os.path.isdir(self.main_lts.path()))
        output=subprocess.check_output([
                'git','symbolic-ref','HEAD',
                ],
                cwd=self.main_lts.path(),
                )
        self.assertEqual(output.strip().decode('utf8'),'refs/heads/master')
        # verify master branch is replicated
        self.assertTrue(self.main_lts.is_replicated())
        #  create subdir and file; push
        jobnum = 12345
        self.rel_root='plugin_name/%d/input' % jobnum
        self._mkdir(self.main_lts,self.rel_root)
        for fnum in range(3):
            fname = 'file%d' % fnum
            self._mkfile(self.main_lts,self.rel_root,fname)
        self.main_lts.lts_push(self.rel_root)
        # verify data is present here and on s3
        output=subprocess.check_output([
                'git','annex','whereis',
                os.path.join(self.rel_root,'file0')
                ],
                cwd=self.main_lts.path(),
                ).decode('utf8')
        lines=output.split('\n')
        self.assertTrue('2 copies' in lines[0])
        self.assertTrue('s3' in lines[1]+lines[2])
        self.assertTrue('here' in lines[1]+lines[2])
    def _verify_2nd_main(self):
        # create a second tree to simulate worker machine
        from dtk.lts import LtsRepo
        self.worker_lts=LtsRepo(self.repo_name,'master',
                local_lts_root=self.second_root,
                )
        self.assertEqual(
                self.worker_lts.path(),
                os.path.join(self.second_root,self.repo_name),
                )
        self.assertTrue(os.path.isdir(self.worker_lts.path()))
        # verify that pull makes data accessible
        self.worker_lts.lts_sync()
        self.worker_lts.lts_fetch(self.rel_root)
        src=open(
                os.path.join(self.main_lts.path(),self.rel_root,'file0')
                ).readlines()
        dest=open(
                os.path.join(self.worker_lts.path(),self.rel_root,'file0')
                ).readlines()
        self.assertEqual(src,dest)
    def _make_dev_branchname(self):
        # obtain date of checkin
        output=subprocess.check_output([
                'git','log','-n','1','--pretty=format:%ct',
                ],
                cwd=self.main_lts.path(),
                )
        ts=int(output.strip())
        import datetime
        dt=datetime.datetime.utcfromtimestamp(ts)+datetime.timedelta(minutes=2)
        self.dev_branch = 'dev_branch_before_%s' % dt.strftime('%Y.%m.%d.%H.%M')
    def _verify_dev_branching(self):
        # simulate dev machine branching (re-use worker tree)
        from dtk.lts import LtsRepo
        if LtsRepo._in_use_check:
            # test that branch switch fails if old worker_lts still exists
            self.assertRaises(RuntimeError,LtsRepo,
                    self.repo_name,
                    self.dev_branch,
                    local_lts_root=self.second_root,
                    )
        # test that it succeeds after old instance is deleted
        del(self.worker_lts)
        self.worker_lts = self._verify_branch_checkout(
                        self.repo_name,
                        self.dev_branch,
                        self.second_root,
                        )
        # test dev branch case where local repo doesn't exist
        del(self.worker_lts)
        rm_readonly_tree(self.second_root)
        self.worker_lts = self._verify_branch_checkout(
                        self.repo_name,
                        self.dev_branch,
                        self.second_root,
                        )
        # test dev branch case where bare repo doesn't exist (new ws on dev)
        del(self.worker_lts)
        rm_readonly_tree(self.second_root)
        self.worker_lts = self._verify_branch_checkout(
                        self.repo_name2,
                        self.dev_branch,
                        self.second_root,
                        )
        del(self.worker_lts)
    def _verify_branch_checkout(self,repo,branch,root):
        from dtk.lts import LtsRepo
        lts=LtsRepo(
                repo,
                branch,
                local_lts_root=root,
                )
        import subprocess
        output=subprocess.check_output([
                'git','symbolic-ref','HEAD',
                ],
                cwd=lts.path(),
                ).decode('utf8')
        self.assertEqual(output.strip(),'refs/heads/'+branch)
        # verify dev branches are not replicated
        self.assertFalse(lts.is_replicated())
        return lts
    def _verify_replication(self):
        # go back to original repo name
        self.worker_lts = self._verify_branch_checkout(
                        self.repo_name,
                        self.dev_branch,
                        self.second_root,
                        )
        self.assertFalse(self.worker_lts.is_replicated())
        # push file
        rel_dir = 'dev_test/666'
        self._mkdir(self.worker_lts,rel_dir)
        self._mkfile(self.worker_lts,rel_dir,'myfile')
        self.worker_lts.lts_push(rel_dir)
        # verify branch not visible on main
        self.main_lts._git_cmd(['fetch'])
        self.assertFalse(os.path.exists(os.path.join(
                        self.main_lts.path(),
                        '.git/refs/remotes/origin',
                        self.worker_lts._branch,
                        )))
        # force replication mode
        self.worker_lts.force_replication()
        self.assertTrue(self.worker_lts.is_replicated())
        # verify branch visible on main
        self.main_lts._git_cmd(['fetch'])
        self.assertTrue(os.path.exists(os.path.join(
                        self.main_lts.path(),
                        '.git/refs/remotes/origin',
                        self.worker_lts._branch,
                        )))

    def test_life_cycle(self):
        self._remove_previous()
        self._verify_create()
        self._verify_2nd_main()
        self._make_dev_branchname()
        self._verify_dev_branching()
        self._verify_replication()

        # XXX test ability to decrypt the raw S3 files? (Is there really
        # XXX a reasonable recovery scenario in which we'd need to do that?)

class FaersTestCase(TransactionTestCase):
    def test_cas_lookup(self):
        # Set up test case:
        # - a collection of 3 drugs with different CAS numbers
        # - the first 2 drugs imported into a workspace, the third not
        # - a second collection, duplicating one CAS from the first
        from drugs.models import Prop,Tag,Drug,Collection
        name_prop=Prop(name=Prop.NAME,prop_type=Prop.prop_types.TAG)
        name_prop.save()
        cas_prop=Prop(name='cas',prop_type=Prop.prop_types.TAG)
        cas_prop.save()
        # force reload of prop cache to make sure the above are included;
        # otherwise you can get weird interactions with other tests
        Prop.reset()
        c = Collection(name='my_collection')
        c.save()
        drugs = []
        for i in range(3):
            d = Drug(collection=c)
            d.save()
            t = Tag(drug=d,prop=name_prop,value='drug%d'%(i+1))
            t.save()
            t = Tag(drug=d,prop=cas_prop,value='1111-22-%d'%(i+1))
            t.save()
            drugs.append(d)
        # create a drug in another collection with a matching CAS
        # to verify that it doesn't screw things up
        c2 = Collection(name='alt_collection')
        c2.save()
        d = Drug(collection=c2)
        d.save()
        t = Tag(drug=d,prop=name_prop,value='drug1alt')
        t.save()
        t = Tag(drug=d,prop=cas_prop,value='1111-22-1')
        t.save()
        from browse.models import Workspace,WsAnnotation
        ws=Workspace(name='test')
        ws.save()
        wsa1=WsAnnotation(ws=ws,agent=drugs[0])
        wsa1.save()
        wsa2=WsAnnotation(ws=ws,agent=drugs[1])
        wsa2.save()
        # Test CASLookup behavior
        from dtk.faers import CASLookup
        cl = CASLookup(ws.id)
        # unknown CAS returns None
        self.assertEqual(
                cl.get_name_and_wsa('random'),
                None,
                )
        # CAS in workspace returns name + wsa
        single_cas = lambda x:next(iter(x.cas_set))
        self.assertEqual(
                cl.get_name_and_wsa(single_cas(drugs[0])),
                (drugs[0].canonical,wsa1),
                )
        self.assertEqual(
                cl.get_name_and_wsa(single_cas(drugs[1])),
                (drugs[1].canonical,wsa2),
                )
        # CAS outside workspace returns name + None
        self.assertEqual(
                cl.get_name_and_wsa(single_cas(drugs[2])),
                (drugs[2].canonical,None),
                )

class VocabTestCase(TestCase):
    def test_vocab_lookup(self):
        # verify vocabulary lookup
        from dtk.vocab_match import DiseaseVocab
        choices = DiseaseVocab.get_choices()
        l = [x[0] for x in choices if x[1] == 'OrangeBook']
        self.assertEqual(len(l),1)
        vocab = DiseaseVocab.get_instance(
                l[0],
                version_defaults=dict(orange_book='v1'),
                )
        self.assertEqual(vocab.name(),"OrangeBook")
        l = list(vocab.items())
        d = dict(l)
        self.assertEqual(len(l),len(d))
        self.assertTrue(l[0][0].startswith('U-'))
        self.assertTrue(isinstance(l[0][1],str))
    def test_matcher(self):
        # verify matcher construction
        l = [
            'a test',
            'another test',
            'Type I Diabetes',
            'Type II Diabetes',
            'a capitalized Test',
            ]
        from dtk.vocab_match import VocabMatcher
        vm = VocabMatcher(enumerate(l))
        wmap = vm.map_words('test phrase')
        self.assertEqual(wmap,[
                ('test',['test']),
                ('phrase',[]),
                ])
        plist = vm.score_phrases([['diabetes']])
        self.assertEqual(set(x[2] for x in plist),set([2,3]))

class ReadTextTestCase(TestCase):
    def test_member_access(self):
        l = [
            ['a','b','c'],
            ['xxx',3,'17'],
            ]
        from dtk.readtext import convert_records_using_header
        l2 = list(convert_records_using_header(iter(l)))
        self.assertEqual(len(l2),1)
        self.assertEqual(l2[0].a,'xxx')
        self.assertEqual(l2[0].b,3)
        self.assertEqual(l2[0].c,'17')
        from dtk.readtext import convert_records_using_colmap
        colmap = [
            ('x','a',len),
            ('y','c',lambda x:x),
            ('z','c',int),
            ]
        l2 = list(convert_records_using_colmap(iter(l),colmap))
        self.assertEqual(len(l2),1)
        self.assertEqual(l2[0].x,3)
        self.assertEqual(l2[0].y,'17')
        self.assertEqual(l2[0].z,17)

class PathHelperTestCase(TestCase):
    def _ph(self,arg):
        '''Return output of path_helper.py command.'''
        import subprocess
        kwargs = {}
        if six.PY3:
            kwargs = {'encoding': 'utf8'}
        return subprocess.check_output(['./path_helper.py',arg], **kwargs).strip()
    def test_cli(self):
        self.assertTrue(self._ph('storage').endswith('ws/'))
        self.assertTrue(self._ph('worker_machine_name').startswith('worker'))
    def test_config(self):
        from path_helper import PathHelper
        cfg = PathHelper.get_config()
        self.assertTrue(cfg.get('worker_machine_name'))
        self.assertTrue(PathHelper.cfg('worker_machine_name'))
        cfg = PathHelper._get_config('/dev/null')
        self.assertEqual(cfg.get('worker_machine_name'),'worker-test')
        cfg = PathHelper._get_config('dtk/testdata/local_settings.py')
        self.assertEqual(cfg.get('worker_machine_name'),'kilroy')

class FMArchiveTestCase(TestCase):
    def test_basics(self):
        stem='/tmp/archive'
        from dtk.features import FMArchive
        outfile=stem+FMArchive.suffix
        from dtk.files import remove_if_present
        remove_if_present(outfile)
        self.assertFalse(os.path.exists(outfile))
        fma = FMArchive()
        fma.put('part1',[1,2,3])
        fma.put('part2',[4,5,6])
        d = {1: 'two', 3: [4, 5]}
        fma.put('part3',d)
        fma.write(stem)
        self.assertTrue(os.path.exists(outfile))
        fma2 = FMArchive(path=stem)
        self.assertEqual(fma2.get('part1').tolist(),[1,2,3])
        self.assertEqual(fma2.get('part2').tolist(),[4,5,6])
        d2=fma2.get('part3')
        self.assertEqual(d2,d)
        # ...which is fantastic, but,
        self.assertEqual(type(d2).__name__,'ndarray')
        # ...so we need to unwrap it...
        self.assertEqual(type(d2.tolist()),dict)
        # The above seems a bit kludgy to rely on, but it's useful; the
        # assumption that it works is encapsulated in DCRecipe code for
        # saving and restoring specs.
        self.assertEqual(fma2.content(),set(['part1','part2','part3']))
        self.assertEqual(fma2.partitions(),set())
    def test_partitions(self):
        stem='/tmp/archive'
        from dtk.features import FMArchive
        outfile=stem+FMArchive.suffix
        from dtk.files import remove_if_present
        remove_if_present(outfile)
        def test_set(n):
            s=3
            return ('part%d'%n,list(range(s*n,s*n+s)))
        fma = FMArchive()
        fma.put(*test_set(0))
        fma.put(*test_set(1))
        fmap0 = fma.partition(0)
        fmap0.put(*test_set(2))
        fmap0.put(*test_set(3))
        fmap1 = fma.partition(1)
        fmap1.put(*test_set(4))
        fmap1.put(*test_set(5))
        fmap2 = fmap0.partition(6)
        fmap2.put(*test_set(6))
        fmap2.put(*test_set(7))
        fma.write(stem)
        self.assertTrue(os.path.exists(outfile))
        fma2 = FMArchive(path=stem)
        self.assertEqual(fma2.content(),set(['part0','part1']))
        self.assertEqual(fma2.partitions(),set([0,1]))
        fma2p0 = fma2.partition(0)
        self.assertEqual(fma2p0.content(),set(['part2','part3']))
        self.assertEqual(fma2p0.get('part2').tolist(),[6,7,8])
        self.assertEqual(fma2p0.partitions(),set([6]))
        fma2p0p6 = fma2p0.partition(6)
        self.assertEqual(fma2p0p6.content(),set(['part6','part7']))
        self.assertEqual(fma2p0p6.get('part7').tolist(),[21,22,23])
        self.assertEqual(fma2p0p6.partitions(),set())

class FeatureMatrixTestCase(TransactionTestCase):
    def test_fmbase_access(self):
        import dtk.features as feat
        b = feat.FMBase(key1=2, key3=6)
        self.assertEqual(b.key1,2)
        self.assertEqual(b['key1'],2)
        self.assertEqual(b.key3,6)
        self.assertEqual(b.get('key3'),6)
        self.assertEqual(b.get('another_key'),None)
        with self.assertRaises(KeyError):
            b['another_key']
        with self.assertRaises(AttributeError):
            b.another_key
        # test dynamic loading
        magic_string = 'I am dynamically supplied data'
        class FM(feat.FMBase):
            def _attr_loader(self,attr):
                if attr == 'data':
                    self.data = magic_string
                else:
                    super(FM,self)._attr_loader(attr)
        fm = FM()
        self.assertEqual(fm.data,magic_string)
        self.assertEqual(fm['data'],magic_string)
        # test representation layer
        # pylint: disable=function-redefined
        class FM(feat.NDArrayRepr,feat.FMBase): pass
        self.do_repr_layer_test(FM())
        class FM(feat.SKSparseRepr,feat.FMBase): pass
        self.do_repr_layer_test(FM())
        # verify feature name order is preserved in sparse arrays
        class FM(feat.SKSparseRepr,feat.FMBase): pass
        fm = FM()
        colnames = ['one','two','three','four']
        fm.load_from_row_stream(
                [
                    (10,[0,0,0,4]),
                    (11,[1,0,0,0]),
                    (12,[0,0,3,0]),
                    (13,[0,2,0,0]),
                    ],
                colnames,
                )
        self.assertEqual(fm.data[0,3],4)
        self.assertEqual(fm.data[1,0],1)
        self.assertEqual(fm.data[2,2],3)
        self.assertEqual(fm.data[3,1],2)
        self.assertEqual(fm.feature_names,colnames)
        # test recipe layer
        spec = feat.DCSpec(None,['pfxeven','pfxodd'],self.dummy_catalog())
        fm = feat.FMBase.load_from_recipe(spec)
        # DCRecipe adds a 'j' to the prefix, and doesn't load 'target'
        self.do_content_check(fm,prefix='jpfx',check_target=False)
        self.assertEqual(fm.sample_key, 'wsa')
        # test saving
        stem = '/tmp/fm_test'
        fm.save(stem)
        if False:
            fm2 = feat.FMBase.load_from_file(stem)
            # XXX the dummy catalog isn't recoverable on a reload (since it's
            # XXX not based on real job ids)
        else:
            # at least verify what's in the archive
            store = feat.FMArchive(path=stem)
            content = store.content()
            self.assertFalse('data' in content)
            self.assertFalse('spec' in content)
            self.assertTrue('specargs' in content)
        # test flattening
        fm.save_flat(stem)
        fm2 = feat.FMBase.load_from_file(stem)
        self.assertEqual(fm2.sample_key, 'wsa')
        # check equality of csr_sparse arrays: all values should be
        # equal except the one unknown (nan != nan)
        self.assertEqual( (fm.data != fm2.data).sum(), 1)
        self.assertEqual(fm.sample_keys, fm2.sample_keys)
        self.assertEqual(fm.feature_names, fm2.feature_names)
        # test arff output
        if six.PY2:
            from StringIO import StringIO
        else:
            # This exists in py2, but expects unicodes which is inconvenient.
            from io import StringIO
        tmp = StringIO()
        fm2.save_to_arff(tmp,'a description')
        lines = tmp.getvalue().split('\n')
        #print lines
        idx = ['RELATION' in x for x in lines].index(True)
        self.assertEqual(idx,0)
        self.assertTrue('a description' in lines[idx])
        idx = ['DATA' in x for x in lines].index(True)
        self.assertEqual(lines[idx+1],'{0 2.0,1 3.0}')
        self.assertEqual(lines[idx+5],'{0 10.0,1 ?}')
        # test plugging unknowns
        my_plug=6
        spec = feat.DCSpec(None,['pfxeven','pfxodd'],
                catalog=self.dummy_catalog(),
                plug_unknowns=my_plug,
                )
        fm = feat.FMBase.load_from_recipe(spec)
        self.assertEqual(fm.data[4,1],my_plug)
        # test filtering: this is a special hook for the fdf plugin;
        # it should only be available for DCRecipe FMs, and should
        # support removing rows by key
        fm = feat.FMBase()
        def base_filter_attempt():
            fm.exclude_by_key(set())
        self.assertRaises(AttributeError,base_filter_attempt)
        spec = feat.DCSpec(None,['pfxeven','pfxodd'],
                catalog=self.dummy_catalog(),
                )
        fm = feat.FMBase.load_from_recipe(spec)
        all_keys = set(fm.sample_keys)
        to_exclude = set([3,4])
        fm.exclude_by_key(to_exclude)
        self.assertEqual(all_keys-set(fm.sample_keys),to_exclude)
    def dummy_matrix(self):
        return [
                (x,(2*x,None if x == 5 else 2*x+1))
                for x in range(1,10)
                ]
    def dummy_catalog(self):
        import runner.data_catalog as dc
        def dummy_fetcher(keyset):
            assert keyset is None,'Not supported'
            # 'pure' unknown row at front should be stripped
            return [(0,(None,None))]+self.dummy_matrix()
        cat=dc.Catalog()
        cat.add_group(
                'pfx',
                dc.CodeGroup('wsa',dummy_fetcher,
                        dc.Code('even'),
                        dc.Code('odd'),
                        ),
                )
        return cat
    def do_content_check(self,fm,prefix='',check_target=True):
        self.assertEqual(fm.feature_names,[prefix+x for x in ('even','odd')])
        self.assertEqual(fm.sample_keys,list(range(1,10)))
        if check_target:
            self.assertEqual(fm.target,[1,0]*4+[1])
        m = fm.data
        self.assertEqual(m[0,0],2)
        self.assertEqual(m[2,1],7)
        self.assertEqual(fm.data_as_array().shape,(9,2))
        self.assertEqual(fm.data_as_array()[0].shape,(2,))
        import math
        self.assertTrue(math.isnan(m[4,1]))
    def do_repr_layer_test(self,fm):
        fm.load_from_row_stream(
                self.dummy_matrix(),
                ['even','odd'],
                )
        fm.target = [x%2 for x in fm.sample_keys]
        self.do_content_check(fm)
        stem = '/tmp/fm_test'
        fm.save(stem)
        import dtk.features as feat
        fm2 = feat.FMBase.load_from_file(stem)
        self.do_content_check(fm2)
    def test_basics(self):
        import dtk.features as feat
        fm = feat.FMBase.load_from_arff(
                'dtk/testdata/full_vector.arff.gz',
                druglist_path='dtk/testdata/druglist.csv',
                )
        n_samples = 7648
        n_attrs = 20
        n_kts = 21
        self.assertTrue('data' in list(fm.keys()))
        self.assertTrue(fm['data'] is fm.data)
        self.assertEqual(len(fm.data),n_samples)
        self.assertTrue(all([len(x)==n_attrs for x in fm.data]))
        self.assertTrue('target' in list(fm.keys()))
        self.assertEqual(len(fm.target),n_samples)
        from collections import Counter
        ctr = Counter(fm.target)
        self.assertEqual(ctr[1],n_kts)
        self.assertEqual(ctr[0],n_samples-n_kts)
        self.assertEqual(fm.target_names,['False','True'])
        self.assertTrue('feature_names' in list(fm.keys()))
        self.assertEqual(len(fm.feature_names),n_attrs)
        self.assertTrue('sample_keys' in list(fm.keys()))
        # test counts of missing values by column
        d = dict(zip(fm.feature_names,fm.feature_missing_values()))
        self.assertEqual(d['DEEPEnD_DIRECTION'],2944)
        self.assertEqual(d['FAERS_DCOE'],6849)
        self.assertEqual(max(d.values()),7528)
        self.assertEqual(min(d.values()),2851)
        # test counts of missing values by row
        ctr = fm.sample_missing_values()
        self.assertEqual(ctr[n_attrs],2721)
        self.assertEqual(ctr[13],max(ctr.values()))
        self.assertEqual(ctr[9],min(ctr.values()))
        # extract some columns, impute values, and replace
        col_list = [1,3,5]
        subset = fm.extract_features(col_list)
        self.assertEqual(subset.shape,(n_samples,len(col_list)))
        import numpy as np
        for col in subset.T:
            missing = [np.isnan(x) for x in col]
            col[missing] = 0
        fm.replace_features(col_list,subset)
        m = fm.feature_missing_values()
        still_missing = [m[x] for x in col_list]
        self.assertTrue(not any(still_missing))
        # verify reading with imputation
        fm = feat.FMBase.load_from_arff(
                'dtk/testdata/full_vector.arff.gz',
                druglist_path='dtk/testdata/druglist.csv',
                default=0,
                )
        self.assertTrue(not any(fm.feature_missing_values()))
        # XXX the initial use case for imputation is the FAERS demographic
        # XXX data; that data doesn't fit the simplistic imputation model
        # XXX in several ways:
        # XXX - we filter those records long before we have a feature vector
        # XXX - one of the columns is categorical rather than numeric
        # XXX - we may want to impute disease and non-disease populations
        # XXX   separately, and/or do something like KNNimpute
        # XXX FAERS DB stats are:
        # XXX - 7.7M events have demographics records (of about 8M total events)
        # XXX - 80K are missing sex
        # XXX - 2.5M are missing age_yr
        # XXX - 5.5M are missing weight_kg
        # XXX - 1.7M have all 3
        # XXX Note that if we don't do anything, we can use inclusion of
        # XXX demographics data to reduce the processing time.  Also, for
        # XXX weight, we'd need to impute almost 3x as much data as we
        # XXX actually have.
        # XXX
        # XXX If we did want to impute this data, we'd need to do something
        # XXX like:
        # XXX - modify the EventDemoScanner to not skip nulls
        # XXX - accumulate into a sparse np array instead of to a file
        # XXX - either impute at that point, or wait until all three matricies
        # XXX   are combined
        # XXX   - in either case, imputation would happen on an np array,
        # XXX     which is compatible with slicing and replacing the FM data
        # XXX     by column

class ScoresTestCase(TransactionTestCase):
    import six
    @patch("dtk.lts.LtsRepo")
    def test_source_list(self, fake_lts):
        from browse.models import Workspace
        ws=Workspace(name='test')
        ws.save()
        # basic SourceList test
        from dtk.scores import SourceList
        sl = SourceList(ws)
        self.assertEqual(sl.to_string(),'')
        self.assertEqual(len(sl.sources()),0)
        sl.load_defaults()
        self.assertEqual(len(sl.sources()),0)
        from runner.models import Process
        p=Process(
                name='faers_%d'%ws.id,
                status=Process.status_vals.SUCCEEDED,
                role='FAERS_faers'
                )
        p.save()
        sl.load_defaults()
        self.assertEqual(len(sl.sources()),1)
        src=sl.sources()[0]
        self.assertEqual(src.label(),'FAERS')
        sl.set_label(src,'Custom')
        self.assertEqual(src.label(),'Custom')
        # see that Enables version does the base-class stuff
        from dtk.scores import SourceListWithEnables
        sl = SourceListWithEnables(ws)
        self.assertEqual(len(sl.sources()),0)
        sl.load_defaults()
        self.assertEqual(len(sl.sources()),1)
        src=sl.sources()[0]
        #self.assertIn(('dcoe','Disease CO+'),src.get_enable_choices())
        #
        # test session restore/save
        #
        expect=sl.to_string()
        dummy_session={}
        sl.load_from_session(dummy_session)
        # should fall back to load_defaults
        self.assertEqual(len(sl.sources()),1)
        self.assertEqual(dummy_session[sl.src_key()],expect)
        def parse_session_key(session,key):
            return [
                    x.split(':')
                    for x in dummy_session[key].split('|')
                    ]
        enabled = parse_session_key(dummy_session,sl.en_key())
        self.assertEqual(len(enabled),1)
        self.assertEqual([str(src.job_id())],enabled[0])

        # This dummy job has no underlying data files.
        # Data catalog thus has no codes for it, which prevents enables.
        # We would have to make the place the data file lives customizable
        # to be able to test this.
        if False:
            self.assertIn('dcoe',enabled[0])
            # test label and enable modifications
            src=sl.sources()[0]
            sl.set_label(src,'Custom')
            self.assertEqual(src.label(),'Custom')
            sl.set_enables(src,['dcop'])
            self.assertNotIn('dcoe',src.enabled_codes())
            # should be preserved in session
            sl = SourceListWithEnables(ws)
            sl.load_from_session(dummy_session)
            src=sl.sources()[0]
            self.assertEqual(src.label(),'Custom')
            self.assertNotIn('dcoe',src.enabled_codes())
            # load_defaults should clear
            sl.load_defaults()
            src=sl.sources()[0]
            self.assertEqual(src.label(),'FAERS')
            self.assertIn('dcoe',src.enabled_codes())
            # and that should also be reflected back in the session
            enabled = parse_session_key(dummy_session,sl.en_key())
            self.assertEqual(len(enabled),1)
            self.assertIn('dcoe',enabled[0])
        #
        # test SourceTable
        #
        import dtk.scores as st
        st = st.SourceTable(sl,[
                st.SelectColType('score','Score'),
                st.LabelColType(),
                st.EnablesColType(),
                ])
        FormClass = st.make_form_class()
        form = FormClass()
        table = st.get_table(form)
        self.assertEqual(len(table.headers()),3)
        r = table.rows()
        self.assertEqual(len(r),1)
        self.assertEqual(len(r[0]),3)

class FileScanTestCase(TestCase):
    def try_options(self,fn):
        from dtk.files import get_file_lines
        l = list(get_file_lines(fn))
        self.assertEqual(len(l),6)
        l = list(get_file_lines(fn,keep_header=False))
        self.assertEqual(len(l),5)
        l = list(get_file_lines(fn,grep=['target string']))
        self.assertEqual(len(l),2)
        self.assertTrue(l[0].startswith('line 1'))
        self.assertTrue(l[1].startswith('line 3'))
        l = list(get_file_lines(fn,grep=['target string'],keep_header=False))
        self.assertEqual(len(l),1)
        self.assertTrue(l[0].startswith('line 3'))
    def test_basic_parsing(self):
        from path_helper import PathHelper
        fn=PathHelper.website_root+'dtk/testdata/file_scan_test.txt'
        self.try_options(fn)
    def test_header_escaping(self):
        # this issue here is, since the header line is extracted and
        # re-inserted by a shell command, make sure it's not confused
        # by quoting in the header text
        from path_helper import PathHelper
        fn=PathHelper.website_root+'dtk/testdata/file_scan_test2.txt'
        self.try_options(fn)

def test_file_select():
    import tempfile
    import six
    kwargs = {}
    if six.PY3:
        kwargs = {'encoding': 'utf8', 'mode': 'wt+'}
    with tempfile.NamedTemporaryFile(**kwargs) as fh:
        expect = []
        template = [
                (('no','matching','data'),False),
                (('either','matching','but in wrong column'),False),
                (('match','either','in correct column'),True),
                (('match','or','in correct column'),True),
                ]
        for rec,match in template:
            fh.write(','.join([str(x) for x in rec])+'\n')
            if match:
                expect.append(list(rec))
        fh.flush()
        from dtk.files import get_file_records
        l = list(get_file_records(
                fh.name,
                keep_header=None,
                parse_type='csv',
                select=(['either','or'],1),
                ))
        assert l == expect
        # verify non-matching header comes through
        l = list(get_file_records(
                fh.name,
                keep_header=True,
                parse_type='csv',
                select=(['either','or'],1),
                ))
        assert l == [list(rec) for rec,_ in template[:1]]+expect
        # verify matching header gets stripped when asked
        l = list(get_file_records(
                fh.name,
                keep_header=False,
                parse_type='csv',
                select=(['no','either'],0),
                ))
        assert l == [list(x[0]) for x in template[1:2]]

        # verify that None will match any column
        l = list(get_file_records(
                fh.name,
                keep_header=None,
                parse_type='csv',
                select=(['either'],None),
                ))
        assert l == [list(x[0]) for x in template[1:3]]
        # Multiple matches in a row.
        l = list(get_file_records(
                fh.name,
                keep_header=None,
                parse_type='csv',
                select=(['matching','either'],None),
                ))
        assert l == [list(x[0]) for x in template[0:3]]

class AwsOpTestCase(TestCase):
    def test_machine_kw_parsing(self):
        from aws_op import Machine
        with self.assertRaises(TypeError) as context:
            m=Machine('',bad_kw_arg=1)
        self.assertIn('bad_kw_arg',str(context.exception))

class PlotTestCase(TestCase):
    def test_scatter(self):
        from dtk.plot import scatter2d
        lx='my x label'
        ly='my y label'
        pp = scatter2d(lx,ly,[(0,1),(3,4)])
        # check for refline
        self.assertEqual(len(pp._data),2)
        self.assertEqual(pp._data[0]['mode'],'lines')
        # check scatterplot itself
        self.assertEqual(pp._data[1]['x'],(0,3))
        self.assertEqual(pp._data[1]['y'],(1,4))
        self.assertEqual(pp._layout['xaxis']['title'],lx)
        self.assertEqual(pp._layout['yaxis']['title'],ly)
        # check histogram
        pp = scatter2d(lx,ly,[(0,1),(3,4)],bins=True)
        self.assertEqual(len(pp._data),4)
        self.assertEqual(pp._data[2]['type'],'histogram')
        self.assertEqual(pp._data[3]['type'],'histogram')
        # check histogram with classes
        import random
        pp = scatter2d(lx,ly,[
                (random.gauss(1,1),random.gauss(1,2))
                for x in range(50)
                ],
                bins=True,
                refline=False,
                class_idx=[x%2 for x in range(50)],
                classes=[
                        ('even',{'color':'blue'}),
                        ('odd',{'color':'red'}),
                        ]
                )
        self.assertEqual(len(pp._data),6)
        # The scatter elements don't seem to have a 'type' parameter.
        #self.assertEqual(pp._data[0]['type'],'scatter')
        self.assertEqual(pp._data[1]['type'],'histogram')
        self.assertEqual(pp._data[2]['type'],'histogram')
        #self.assertEqual(pp._data[3]['type'],'scatter')
        self.assertEqual(pp._data[4]['type'],'histogram')
        self.assertEqual(pp._data[5]['type'],'histogram')
        if False:
            # test visualization
            pp.save('/tmp/xxx.plotly',True)

class QueryStringTestCase(TestCase):
    def test_encoding(self):
        from dtk.duma_view import qstr
        self.assertEqual(qstr({}),'?')
        self.assertEqual(qstr({'a':'b'}),'?a=b')
        d={'a':'b','c':'d'}
        safe=d.copy()
        self.assertEqual(qstr(d),'?a=b&c=d')
        self.assertEqual(qstr(d,c=1),'?a=b&c=1')
        self.assertEqual(d,safe)
        self.assertEqual(qstr(d,c=None),'?a=b')
        self.assertEqual(qstr({'a':'b','aa':1,'x':2,'u':5}),'?a=b&aa=1&u=5&x=2')

class DynaFormTestCase(TransactionTestCase):
    def test_context_precedence(self):
        from dtk.dynaform import FieldType
        ft = FieldType.get_by_code('ts_id')
        self.assertEqual(ft._context,{})
        ft.add_fallback_context(ws_id=2)
        self.assertEqual(ft._context,{'ws_id':2})
        ft.add_override_context(ws_id=3)
        self.assertEqual(ft._context,{'ws_id':3})
        ft.add_fallback_context(ws_id=4)
        self.assertEqual(ft._context,{'ws_id':3})
        ft = FieldType.get_by_code( ('ts_id',{'ws_id':1}) )
        self.assertEqual(ft._context,{'ws_id':1})
        ft.add_fallback_context(ws_id=2)
        self.assertEqual(ft._context,{'ws_id':1})
        ft.add_override_context(ws_id=3)
        self.assertEqual(ft._context,{'ws_id':3})
    def test_context_usage(self):
        from browse.models import Workspace,TissueSet
        ws = Workspace()
        ws.save()
        ts_names=[]
        for i in range(2):
            name='ts%d'%(i+1)
            ts_names.append(name)
            ts = TissueSet(name=name,ws=ws)
            ts.save()
        from dtk.dynaform import FormFactory,FieldType
        ff = FormFactory()
        ft = FieldType.get_by_code('ts_id')
        ft.add_override_context(ws_id=ws.id)
        ft.add_to_form(ff)
        self.assertEqual(
                [x[1] for x in ff._fields['ts_id'].choices],
                ts_names,
                )
        ft = FieldType.get_by_code('p2d_file')
        ft.add_to_form(ff)
        from dtk.prot_map import DpiMapping
        self.assertEqual(ff._fields['p2d_file'].initial,DpiMapping.preferred)
        ft = FieldType.get_by_code('p2p_file')
        ft.add_to_form(ff)
        from dtk.prot_map import PpiMapping
        self.assertEqual(ff._fields['p2p_file'].initial,PpiMapping.preferred)
        ft = FieldType.get_by_code('path_code')
        ft.add_to_form(ff)
        self.assertEqual(ff._fields['path_code'].initial,'direct')
        ft = FieldType.get_by_code( ('path_code',{'initial':'indirect'}) )
        ft.add_to_form(ff)
        self.assertEqual(ff._fields['path_code'].initial,'indirect')
        ft = FieldType.get_by_code( ('eval_ds',{'ws_id':ws.id}) )
        ft.add_to_form(ff)
        self.assertTrue('kts' in [x[0] for x in ff._fields['eval_ds'].choices])

class EnrichmentMetricTestCase(TestCase):
    def test_no_kt_scores(self):
        from dtk.enrichment import EnrichmentMetric,EMInput,DPI_bg_corr
        emi=EMInput(
                #score=[],
                score=[(i+1,float(max(0,5-i))/5) for i in range(10,0,-1)],
                kt_set=(15,16,17),
                )
        from dtk.files import Quiet
        for name,MetricType in EnrichmentMetric.get_subclasses():
            if MetricType == DPI_bg_corr:
                # This one has a different constructor that we can't handle here.
                continue
            if 'Condensed' in MetricType.__name__:
                # We haven't set up WsAnnotations for auto-condensing.
                continue
            em = MetricType()
            with Quiet() as tmp:
                em.evaluate(emi)
            self.assertEqual(em.rating,0,'for '+name)

# helpful for score weight testing
def ramp(points):
    for i in range(0,points):
        yield (i,1.0-float(i)/points)

def show_baseline_by_kts():
    import dea
    # How does the baseline change as the number of KTs increases?
    r = dea.Runner(fileHandle='',weight=0,nPermuts=1000, set_of_interest = ['filler'])
    r.verbose=False
    for kts in range(1,50):
        r.score_list = [(x,0) for x in range(0,1000)]
        r.set_oi = [x for x in range(0,kts)]
        r.run()
        print(kts, r.boostrappedESSummary)

def show_weighted_score_by_position():
    import dea
    r = dea.Runner(fileHandle='',weight=1,nPermuts=100, score_list = list(ramp(101)), set_of_interest = ['filler'])
    r.verbose=False
    for pos in (1,2,10,20,50,98,99,100):
        r.set_oi = [pos]
        r.run()
        print(pos, r.es, r.boostrappedESSummary)

class DeaTestCase(TestCase):
    def broken_test_weight_0(self):
        # XXX all the test data below passes zero scores, but this causes
        # XXX the tie detection mechanism to kick in; fix, someday, maybe
        import dea
        dea.verbose = False
        #show_weighted_score_by_position()
        #show_baseline_by_kts()
        r = dea.Runner(fhf=None,weight=0,nPermuts=100, alpha = 0.05,  set_of_interest = ['filler'])
        r.verbose=False
        # regardless of the size of the background...
        for points in (3,33,3333):
            # a single item in the first position yields a peak score of 1
            r.score_list = [(x,0) for x in range(0,points)]
            r.set_oi = [0]
            r.run()
            self.assertAlmostEqual(r.es,1)
            # a single item in the last position yields a peak score of 0
            r.set_oi = [points-1]
            r.run()
            self.assertAlmostEqual(r.es,0)
            # a single item in the middle position yields a peak score of 0.5
            r.set_oi = [points/2]
            r.run()
            self.assertAlmostEqual(r.es,0.5)
        # two kts together at the beginning score 1
        points=500
        r.score_list = [(x,0) for x in range(0,points)]
        r.set_oi = [0,1]
        r.run()
        self.assertAlmostEqual(r.es,1,places=2)
        # one at the start and one more than 50% later scores 0.5
        r.set_oi = [0,points-points/3]
        r.run()
        self.assertAlmostEqual(r.es,0.5,places=2)
        # one at the start and one 25% of the way through scores 0.75
        r.set_oi = [0,points/4]
        r.run()
        self.assertAlmostEqual(r.es,0.75,places=2)
        # pvalue should not be zero
        self.assertGreater(r.pval,0)
        # the baseline decreases as kts increase, but isn't much affected
        # by changes in background size (if background >> kts)
        r = dea.Runner(fileHandle='',weight=0,nPermuts=500, alpha = 0.05,  set_of_interest = ['filler'])
        r.verbose=False
        self.longMessage=True
        for kts,expected_baseline in (
                                    # these are derived experimentally
                                    (1, 0.5),
                                    (2, 0.36),
                                    (10, 0.18),
                                    (20, 0.125),
                                    (40, 0.09),
                                    ):
            for points in (1000,5000,50000):
                r.score_list = [(x,0) for x in range(0,points)]
                r.set_oi = list(range(0,kts))
                r.run()
                baseline = r.boostrappedESSummary[0]
                pct_error = 100*(baseline/expected_baseline - 1)
                #print kts,points,pct_error
                self.assertLess(abs(pct_error),20,
                        msg="baseline %f; points %d, kts %d"
                            % (baseline, points, kts),
                        )

    def broken_test_weight_1(self):
        # Sept 2016 conversion to new dea broke what remained of the
        # regression tests
        import dea
        dea.verbose = False
        r = dea.Runner(fhf=None,weight=1,nPermuts=100, score_list=list(ramp(100)), set_oi = [0], alpha=0.05)
        r.verbose=False
        # a single kt at the beginning still scores 1.0
        r.run()
        self.assertAlmostEqual(r.es,1,places=2)
        # as do 2 kts
        r.set_oi = [0,1]
        r.run()
        self.assertAlmostEqual(r.es,1,places=2)
        # single kt score is proportional to position on ramp
        self.longMessage=True
        r.score_list = list(ramp(101))
        for pos in (1,2,10,20,50,98,99,100):
            r.set_oi = [pos]
            r.run()
            self.assertAlmostEqual(r.es,1.0-float(pos)/100,
                        places=2,
                        msg="pos %d" % pos,
                        )
        # more iterations for greater baseline accuracy
        r = dea.Runner(fhf=None,weight=1,nPermuts=1000, score_list=list(ramp(102)), set_oi = [10,21],alpha=0.05)
        r.verbose=False
        # if all kts are on the leading edge, the score is the same
        # as a single kt score in the lowest position (surprising, but
        # the total up-bumps and the divisor are the same, so always 1,
        # minus the sum of the down-bumps to the left of the rightmost kt);
        # the ramp is 102 long so each of the 100 down-bumps is 0.01
        # for simplicity
        r.run()
        self.assertAlmostEqual(r.es,0.8,places=2)
        # XXX the following no longer works, probably due to algorithm changes
        # # for the weighted case, the baseline should be the
        # # weight 0 baseline * average score * kts / passed-in divisor
        # expected_baseline = (0.36 * 0.5 * 2)/(0.9+0.8)
        # pct_error = 100*(r.boostrappedESSummary[0]/expected_baseline - 1)
        # self.assertLess(abs(pct_error),20)

