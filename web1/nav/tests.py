from django.test import TestCase,TransactionTestCase
import unittest
import six
from mock import MagicMock

class Mock(object):
    def __init__(self,**kwargs):
        for k,v in six.iteritems(kwargs):
            setattr(self,k,v)

class MockRequest(Mock):
    def __init__(self,**kwargs):
        self.path='/dummy/path/'
        self.method='GET'
        self.GET={}
        self.META={}
        self.session={}
        self.user=Mock()
        self.user.is_authenticated=True
        self.user.is_staff=True
        self.user.groups = MagicMock()
        super(MockRequest,self).__init__(**kwargs)
    def get_full_path(self):
        return self.path

class ScoreboardViewTestCase(TransactionTestCase):
    def test_basics(self):
        from browse.models import Workspace
        ws=Workspace(id=1,name='test workspace')
        ws.save()
        from nav.views import ScoreboardView
        # verify unauthenticated access goes to login page
        sv=ScoreboardView.as_view()
        req=MockRequest()
        req.user.is_authenticated=False
        rsp=sv(req,ws_id=ws.id)
        self.assertEqual(rsp.status_code,302)
        self.assertTrue(
                rsp.headers['location'].startswith('/account/login/')
                )
        # verify default access succeeds
        sv=ScoreboardView.as_view()
        req=MockRequest()
        rsp=sv(req,ws_id=ws.id)
        self.assertEqual(rsp.status_code,200)
        # XXX rsp.content can be parsed and examined
        # verify reload works correctly

class KeyFilterTestCase(TestCase):
    def test_basics(self):
        from nav.views import KeyFilter
        kf = KeyFilter()
        self.assertTrue(kf.ok(1)) # an empty filter passes everything
        self.assertEqual(kf.count(),0)
        self.assertTrue(kf.excluding())
        kf.merge('k1','l1','v1',(1,2,3,))
        self.assertTrue(kf.ok(1))
        self.assertTrue(kf.ok(2))
        self.assertTrue(kf.ok(3))
        self.assertFalse(kf.ok(4))
        self.assertEqual(kf.count(),3)
        self.assertFalse(kf.excluding())
        kf.merge('k2','l2','v2',(2,3,4,))
        self.assertFalse(kf.ok(1))
        self.assertTrue(kf.ok(2))
        self.assertTrue(kf.ok(3))
        self.assertFalse(kf.ok(4))
        self.assertEqual(kf.count(),2)
        self.assertFalse(kf.excluding())
        kf2 = KeyFilter(copy_from=kf)
        kf.merge('k3','l3','v3',(2,),exclude=True)
        self.assertFalse(kf.ok(1))
        self.assertFalse(kf.ok(2))
        self.assertTrue(kf.ok(3))
        self.assertFalse(kf.ok(4))
        self.assertEqual(kf.count(),1)
        self.assertFalse(kf.excluding())
        self.assertEqual(kf._parts,{
                'k1': KeyFilter.FilterDetail('k1', 'l1', 'v1'),
                'k2': KeyFilter.FilterDetail('k2', 'l2', 'v2'),
                'k3': KeyFilter.FilterDetail('k3', 'l3', 'v3'),
                })
        self.assertEqual(kf2.count(),2)
        self.assertEqual(kf2._parts,{
                'k1': KeyFilter.FilterDetail('k1', 'l1', 'v1'),
                'k2': KeyFilter.FilterDetail('k2', 'l2', 'v2'),
                })
        kf = KeyFilter()
        kf.merge('k3','l3','v3',(2,),exclude=True)
        self.assertEqual(kf.count(),-1)
        self.assertTrue(kf.excluding())
        self.assertFalse(kf.ok(2))
        self.assertTrue(kf.ok(3))
