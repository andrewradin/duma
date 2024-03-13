from django.test import TestCase

class TargetCacheTestCase(TestCase):
    def setUp(self):
        from drugs.models import Prop
        Prop.reset() # clear static state (since db rolls back on each test)

    def test_target_cache(self):
        # build collection
        from drugs.tests import create_dummy_collection
        create_dummy_collection('drugbank',
                data=[
                    ("DB00007","canonical","Leuprolide"),
                    ("DB00008","canonical","Peginterferon alfa-2a"),
                    ("DB00009","canonical","Alteplase"),
                    ("DB00666","canonical","Dummy"),
                    ("DB00667","canonical","alias"),
                    ("DB00667","m_drugbank_id","DB00007"),
                ],
                )
        # set up test-relevant Proteins
        from browse.models import Protein
        Protein.objects.get_or_create(uniprot='P17181',gene='IFNAR1')
        Protein.objects.get_or_create(uniprot='P02671',gene='FGA')
        Protein.objects.get_or_create(uniprot='P00747',gene='PLG')
        Protein.objects.get_or_create(uniprot='P30968',gene='GNRHR')
        #Protein.objects.get_or_create(uniprot='P48551',gene='IFNAR2')
        # above is left out to test gene name defaulting
        Protein.objects.get_or_create(uniprot='Q96P88',gene='GNRHR2')
        # get agents
        from drugs.models import Drug
        agents = Drug.objects.filter(collection__name='drugbank')
        # get test drugs
        db8 = [x for x in agents if x.drugbank_id == "DB00008"][0]
        # build cache
        from dtk.prot_map import AgentTargetCache,DpiMapping
        atc = AgentTargetCache(
                mapping=DpiMapping('drugbank.default'),
                agent_ids=[x.id for x in agents],
                dpi_thresh=0.9,
                )
        # check raw uniprot results
        db8_prots = set(x.uniprot_id for x in atc.raw_info_for_agent(db8.id))
        self.assertEquals(db8_prots,set(['P17181','P48551']))
        # check converted results
        self.assertEquals(
                atc.info_for_agent(db8.id),
                [('P48551','(P48551)',1),('P17181','IFNAR1',1)],
                )
