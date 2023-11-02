from django.test import TestCase

from drugs.models import Collection,Drug,Prop,Tag,AttributeCacheLoader

def import_drugs(iterable,collection_name,key_name):
    """
    This skips a bunch of the implicit stuff that normally happens
    during import, but that is fine for these tests.
    """
    c = Collection.get_or_create(collection_name,key_name)
    last_id = "";
    last_drug = None;
    name_prop = Prop.get(Prop.NAME)
    for row in iterable:
        if row[0] != last_id:
            last_id = row[0]
            if row[1] == Prop.NAME:
                last_drug = c.get_or_create_drug(last_id,row[2])
                continue # row is fully processed
            # just get, don't create
            last_drug = c.get_or_create_drug(last_id)
        prop = Prop.get(row[1])
        last_drug.set_prop(prop,row[2])

def create_dummy_collection(collection_name,
        prop_info={},
        data=[],
        extra_props=[],
        ):
    pt=Prop.prop_types
    keyname = collection_name.split('.')[0]
    for prop in set(x[1] for x in data)|set(extra_props):
        overrides = prop_info.get(prop,{})
        Prop.get_or_create(
                prop,
                overrides.get('type',pt.TAG),
                multival=overrides.get('multival',False),
                )
    import_drugs(data,collection_name,keyname+'_id')

# Create your tests here.

class ModelsTestCase(TestCase):
    def setUp(self):
        Prop.reset() # clear static state (since db rolls back on each test)

    def test_create(self):
        c = Collection.get_or_create("collection 1","col1_id")
        d = c.get_or_create_drug('external_id','drug_1_name')
        p_alias = Prop.get_or_create("alias",multival=True)
        p_inchi_key = Prop.get_or_create("inchi_key")
        d.set_prop(p_alias, "drug 1 alias")
        d.set_prop(p_inchi_key, "drug 1 inchi_key")

        qs=Drug.objects.filter(tag__prop=p_inchi_key
                            ,tag__value="drug 1 alias"
                            )
        self.assertEqual(qs.count(),0)
        qs=Drug.objects.filter(tag__prop=p_alias
                            ,tag__value="drug 1 alias"
                            )
        self.assertEqual(qs.count(),1)
        qs=Drug.objects.filter(tag__prop__in=(p_alias,p_inchi_key)
                            ,tag__value="drug 1 inchi_key"
                            )
        self.assertEqual(qs.count(),1)
        qs=Drug.objects.filter(tag__prop__in=(p_alias,p_inchi_key)
                            ,tag__value="drug 1 alias"
                            )
        self.assertEqual(qs.count(),1)
        # test access via attributes
        self.assertEqual(d.inchi_key,'drug 1 inchi_key')
        self.assertEqual(d.alias_set,set(['drug 1 alias']))
        self.assertEqual(d.inchi_key_set,set(['drug 1 inchi_key']))
        with self.assertRaises(AttributeError):
            a = d.alias

    def test_cardinality(self):
        pt=Prop.prop_types
        p1 = Prop.get_or_create("one_to_one",pt.TAG)
        pn = Prop.get_or_create("one_to_many",pt.TAG,multival=True)
        c = Collection.get_or_create("cardinality test","ct_id")
        d = c.get_or_create_drug('external_id','some drug name')
        d.set_prop(p1,"val 1");
        d.set_prop(p1,"val 2");
        d.set_prop(pn,"val 3","my href");
        d.set_prop(pn,"val 4");
        self.assertEqual(d.one_to_one,"val 2")
        self.assertEqual(d.one_to_one_set,set(["val 2"]))
        with self.assertRaises(AttributeError):
            d.one_to_many
        self.assertEqual(d.one_to_many_set,set(["val 3","val 4"]))
        qs = d.tag_set.filter(value='val 3')
        self.assertEqual(qs.count(),1)
        self.assertEqual(qs[0].href,"my href")
        d.set_prop(pn,"val 4");
        qs = Tag.objects.filter(drug=d,prop=pn)
        self.assertEqual(qs.count(),2)
        d.del_prop(pn,"val 4")
        qs = Tag.objects.filter(drug=d,prop=pn)
        self.assertEqual(qs.count(),1)
        d.del_prop(pn)
        qs = Tag.objects.filter(drug=d,prop=pn)
        self.assertEqual(qs.count(),0)

    def test_import(self):
        pt=Prop.prop_types
        names="synonym brand"
        for name in names.split():
            Prop.get_or_create(name,pt.TAG,multival=True)
        names="canonical cas kegg drugbank_id"
        for name in names.split():
            Prop.get_or_create(name,pt.TAG,multival=False)
        for name in ("approved","investigational"):
            Prop.get_or_create(name,pt.FLAG)

        data = [
        ("DB00007","canonical","Leuprolide"),
        ("DB00007","approved",True),
        ("DB00007","investigational",True),
        ("DB00007","synonym","Leuprorelin"),
        ("DB00007","brand","Eligard"),
        ("DB00007","brand","Enantone"),
        ("DB00007","brand","Leuplin"),
        ("DB00007","brand","LeuProMaxx"),
        ("DB00007","brand","Leupromer"),
        ("DB00007","brand","Lupron"),
        ("DB00007","brand","Lutrate"),
        ("DB00007","brand","Memryte"),
        ("DB00007","brand","Prostap 3"),
        ("DB00007","brand","Prostap SR"),
        ("DB00007","brand","Viadur"),
        ("DB00007","cas","53714-56-0"),
        ("DB00007","kegg","C07612"),
        ("DB00007","kegg","D08113"),
        ("DB00008","canonical","Peginterferon alfa-2a"),
        ("DB00008","approved",True),
        ("DB00008","investigational","True"), # string for boolean
        ("DB00008","brand","Pegasys"),
        ("DB00008","cas","198153-51-4"),
        ("DB00009","canonical","Alteplase"),
        ("DB00009","approved",True),
        ("DB00009","synonym","t-PA"),
        ("DB00009","synonym","t-plasminogen activator"),
        ("DB00009","synonym","tPA"),
        ("DB00009","brand","Activase"),
        ("DB00009","cas","105857-23-6"),
        ("DB00009","kegg","D02837"),
        ]
        import_drugs(data,'drugbank','drugbank_id')

    def test_prop_cache(self):
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
                    ("DB00007","cas","53714-56-0"),
                    ("DB00008","canonical","Peginterferon alfa-2a"),
                    ("DB00008","approved",True),
                    ("DB00009","canonical","Alteplase"),
                    ("DB00009","approved",True),
                    ("DB00009","synonym","t-PA"),
                    ("DB00009","synonym","t-plasminogen activator"),
                ],
                )
        from django.test.utils import CaptureQueriesContext
        from django.db import connection
        with CaptureQueriesContext(connection) as context:
            qs=Drug.objects.filter(collection__name='drugbank')
            l=list(qs) # force query to execute
        self.assertEqual(len(context),1)
        self.assertEqual(len(l),3)
        with CaptureQueriesContext(connection) as context:
            l2=[x.canonical for x in l]
        self.assertEqual(len(context),3)
        self.assertEqual(len(l2),3)
        with CaptureQueriesContext(connection) as context:
            l2=[x.drugbank_id for x in l]
        self.assertEqual(len(context),0)
        self.assertEqual(len(l2),3)
        # now start over with a bulk load
        with CaptureQueriesContext(connection) as context:
            qs=Drug.objects.filter(collection__name='drugbank')
            acl=AttributeCacheLoader(qs,'',Prop.prop_types.TAG)
            for drug in qs:
                acl.load_cache(drug)
            # query executes in loop above, so the list isn't
            # really needed, but it makes the following asserts
            # more consistent
            l=list(qs)
        expected_queries = 2
        from django.db import connection
        if connection.mysql_version[:2] == (5,5):
            expected_queries += 1
        self.assertEqual(len(context),expected_queries)
        self.assertEqual(len(l),3)
        with CaptureQueriesContext(connection) as context:
            l2=[x.canonical for x in l]
        self.assertEqual(len(context),0)
        self.assertEqual(len(l2),3)

    def test_matched_props(self):
        create_dummy_collection('drugbank.full',
                data=[
                    ("DB00007","canonical","Leuprolide"),
                    ("DB00007","m_chembl_id","CHEMBL000001"),
                    ("DB00008","canonical","not linked"),
                ],
                )
        create_dummy_collection('chembl.full',
                data=[
                    ("CHEMBL000001","canonical","Leuprolide also"),
                    ("CHEMBL000002","canonical","not Leuprolide"),
                ],
                )
        drugbank_agent_ids = Drug.objects.filter(
                collection__name='drugbank.full',
                ).values_list('id',flat=True)
        # verify that the linked_id_map finds the link between
        # DB00007 and CHEMBL000001
        matched_mm = Drug.matched_id_mm(drugbank_agent_ids, version=None)
        summary = []
        for key in matched_mm.fwd_map():
            src = Drug.objects.get(pk=key)
            for key2 in matched_mm.fwd_map()[key]:
                dest = Drug.objects.get(pk=key2)
                summary.append((
                        src.canonical,
                        src.drugbank_id,
                        dest.canonical,
                        dest.chembl_id,
                        ))
        # We should find both drugbank drugs matched against themself.
        # Additionally, Leuprolide should find its matched CHEMBL drug.
        assert sorted(summary) == [
                ("Leuprolide","DB00007","Leuprolide",None),
                ("Leuprolide","DB00007","Leuprolide also","CHEMBL000001"),
                ("not linked","DB00008","not linked",None),
                ]
        # Now verify that we can easily retrieve and cache all matched values
        drug_qs=Drug.objects.filter(id__in=list(matched_mm.rev_map().keys()))
        acl = AttributeCacheLoader(drug_qs,None,Prop.prop_types.TAG)
        for drug in drug_qs:
            acl.load_cache(drug)
        from django.test.utils import CaptureQueriesContext
        from django.db import connection
        linked_names = []
        with CaptureQueriesContext(connection) as context:
            for drug in drug_qs:
                linked_names.append(drug.canonical)
        assert set(linked_names) == set(['not linked', 'Leuprolide', 'Leuprolide also'])
        self.assertEqual(len(context),0)
