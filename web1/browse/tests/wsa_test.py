from __future__ import print_function
from django.test import TestCase

from drugs.models import Drug,Prop
from drugs.tests import create_dummy_collection

# Create your tests here.

class ModelsTestCase(TestCase):
    def setUp(self):
        Prop.reset() # clear static state (since db rolls back on each test)

    def test_cache(self):
        # create a test collection
        pt=Prop.prop_types
        create_dummy_collection('drugbank',
                prop_info=dict(
                    synonym=dict(multival=True),
                    approved=dict(type=pt.FLAG),
                    ),
                data=[
                    ("DB00007","canonical","Leuprolide"),
                    ("DB00007","approved",True),
                    ("DB00007","synonym","Leuprorelin"),
                    ("DB00007","inchi_key","GFIJNRVAKGFPGQ-LIJARHBVSA-N"),
                    ("DB00008","canonical","Peginterferon alfa-2a"),
                    ("DB00008","approved",True),
                    ("DB00009","canonical","Alteplase"),
                    ("DB00009","approved",True),
                    ("DB00009","synonym","t-PA"),
                    ("DB00009","synonym","t-plasminogen activator"),
                    ("DB00666","canonical","Not imported"),
                ],
                )
        # create a test workspace and import most of the above collection
        from browse.models import Workspace, WsAnnotation
        ws,new = Workspace.objects.get_or_create(name='Test')
        for drug in Drug.objects.filter(collection__name='drugbank'):
            if drug.drugbank_id == "DB00666":
                continue
            WsAnnotation.objects.get_or_create(ws=ws,agent=drug)
        self.assertEqual(ws.wsannotation_set.count(),3)
        # do a query and attempt to access attributes
        from django.test.utils import CaptureQueriesContext
        from django.db import connection
        with CaptureQueriesContext(connection) as context:
            qs=WsAnnotation.objects.filter(ws=ws)
            qs=WsAnnotation.prefetch_agent_attributes(qs)
            l=list(qs)
        for x in context:
            print(x['sql'])
        # query should do 2nd fetch for tag attributes
        # That fetch now takes 2 queries, bringing us up to 3.
        expected_queries = 2
        from django.db import connection
        if connection.mysql_version[:2] == (5,5):
            expected_queries += 1
        self.assertEqual(len(context),expected_queries)
        self.assertEqual(len(l),3)
        # now access tag attributes
        with CaptureQueriesContext(connection) as context:
            d = {
                    wsa.agent.drugbank_id:wsa.agent.canonical
                    for wsa in l
                    }
        print(d)
        for x in context:
            print(x['sql'])
        # tag attribute values should be available with no query
        self.assertEqual(len(context),0)
        self.assertEqual(d['DB00009'],"Alteplase")
        self.assertEqual(len(d),3)
        # flag attributes should cause one query per drug
        with CaptureQueriesContext(connection) as context:
            d = {
                    wsa.agent.drugbank_id:wsa.agent.approved
                    for wsa in l
                    }
        for x in context:
            print(x['sql'])
        self.assertEqual(len(context),len(l))
        # now try again, prefetching both flags and tags
        with CaptureQueriesContext(connection) as context:
            qs=WsAnnotation.objects.filter(ws=ws)
            qs=WsAnnotation.prefetch_agent_attributes(qs,[pt.TAG,pt.FLAG])
            l=list(qs)
        for x in context:
            print(x['sql'])
        # query should do 2/3rd fetch for tags and 4/5th for flags
        expected_queries = 3
        if connection.mysql_version[:2] == (5,5):
            expected_queries += 2
        self.assertEqual(len(context),expected_queries)
        self.assertEqual(len(l),3)
        # now, neither tag nor flag accesses should trigger queries
        with CaptureQueriesContext(connection) as context:
            d1 = {
                    wsa.agent.drugbank_id:wsa.agent.approved
                    for wsa in l
                    }
            d2 = {
                    wsa.agent.drugbank_id:wsa.agent.canonical
                    for wsa in l
                    }
        for x in context:
            print(x['sql'])
        self.assertEqual(len(context),0)
        # test case where base query is a tags-based search
        from browse.utils import drug_search_wsa_filter
        with CaptureQueriesContext(connection) as context:
            qs=WsAnnotation.objects.filter(ws=ws)
            qs=drug_search_wsa_filter(qs,'ro')
            qs=WsAnnotation.prefetch_agent_attributes(qs,[pt.TAG])
            l=list(qs)
        expected_queries = 2
        if connection.mysql_version[:2] == (5,5):
            expected_queries += 1
        self.assertEqual(len(context),expected_queries)
        # extract CAS values; shouldn't trigger queries
        with CaptureQueriesContext(connection) as context:
            d = {
                    wsa.agent.drugbank_id:wsa.agent.inchi_key
                    for wsa in l
                    }
        self.assertEqual(len(context),0)
        self.assertEqual(len(d),2)
        self.assertEqual(d['DB00007'],'GFIJNRVAKGFPGQ-LIJARHBVSA-N')
        self.assertEqual(d['DB00008'],None)

