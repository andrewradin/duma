import pytest

@pytest.fixture
def coll_name(db):
    '''Create a collection and return its name.
    '''
    from drugs.models import Prop
    for ext_key in (
            'med_chem_express',
            'selleckchem',
            'cayman',
            'ncats',
            'ttd',
            'chembl',
            'bindingdb',
            'stitch',
            ):
        Prop.get_or_create(ext_key+'_id',Prop.prop_types.TAG,False)
        Prop.get_or_create('m_'+ext_key+'_id',Prop.prop_types.TAG,True)
    Prop.get_or_create('pubchem_cid',Prop.prop_types.TAG,False)
    from drugs.tests import create_dummy_collection
    coll_name='drugbank.default'
    create_dummy_collection(coll_name,data=[
            ("DB01002","canonical","Levobupivacaine"),
            ("DB01002","m_drugbank_id","DB00297"),
            ("DB01002","m_chembl_id","CHEMBL1200396"),
            ("DB01002","m_chembl_id","CHEMBL1200749"),
            ("DB01002","m_chembl_id","CHEMBL1201193"),
            ("DB01002","m_bindingdb_id","BDBM50350789"),
            ("DB01002","m_bindingdb_id","BDBM50350790"),
            ("DB01002","m_bindingdb_id","BDBM50350791"),
            ("DB01002","m_bindingdb_id","BDBM50417951"),
            # the following aren't real, but are needed to pull in Ki data
            ("DB01002","m_bindingdb_id","BDBM50059492"),
            ("DB01002","m_chembl_id","CHEMBL3828555"),
            ])
    return coll_name

def test_assay_data_fetch(coll_name):
    # get drug to test
    from browse.models import Drug
    agent = Drug.objects.get(
            collection__name=coll_name,
            tag__prop__name='canonical',
            tag__value='Levobupivacaine',
            )
    # This represents the contents of the actual assay files at a
    # particular point in time, and may change if they're re-extracted
    expected = [
            ['c50', 'BDBM50417951', 'Q12809', -1, 20892.96, 1, 0.0],
            ['c50', 'BDBM50350789', 'Q12809', -1, 84000.0, 1, 0.0],
            ['c50', 'BDBM50350790', 'Q12809', -1, 26533.0, 2, 7071.07],
            ['c50', 'BDBM50350791', 'Q12809', -1, 13000.0, 1, 0.0],
            ['c50', 'CHEMBL1200396', 'Q12809', -1, 26533.0, 2, 7071.07],
            ['c50', 'CHEMBL1200749', 'Q12809', -1, 13000.0, 2, 0.0],
            ['c50', 'CHEMBL1201193', 'P08684', 0, 10000.0, 1, 0.0],
            ['c50', 'CHEMBL1201193', 'Q12809', -1, 20892.96, 1, 0.0],
            ['c50', 'CHEMBL1201193', 'A0A087X1C5', -1, 196.4, 1, 0.0],
            ['c50', 'CHEMBL1200396', 'A0A087X1C5', -1, 400.0, 1, 0.0],
            ['c50', 'CHEMBL1201193', 'P05177', 0, 5011.87, 1, 0.0],
            ['ki', 'BDBM50059492', 'P21917', 0, 7.5, 1, 0.0],
            ['ki', 'BDBM50059492', 'P35462', 0, 2210.0, 1, 0.0],
            ['ki', 'CHEMBL3828555', 'P0DMS8', 0, 890.0, 1, 0.0],
            ['ki', 'CHEMBL3828555', 'P29275', 0, 337.0, 1, 0.0],
            ]
    aa = agent.get_assay_info(version=None)
    for row in aa.assays:
        print(row)
    assert sorted(aa.assays) == sorted(expected)

@pytest.fixture
def ws(coll_name):
    '''Return a workspace with a collection imported.
    '''
    from browse.models import Workspace,WsAnnotation
    ws,new = Workspace.objects.get_or_create(name='Test Workspace')
    from drugs.models import Drug
    for drug in Drug.objects.filter(collection__name=coll_name):
        WsAnnotation.objects.get_or_create(ws=ws,agent=drug)
    return ws

import six
def test_assay_view(ws,client,django_user_model):
    username = "user1"
    password = "bar"
    user_obj = django_user_model.objects.create_user(
            username=username,
            password=password,
            )
    from dtk.tests.end_to_end_test import mark_user_access_known
    mark_user_access_known(user_obj)
    client.login(username=username, password=password)
    from browse.models import WsAnnotation
    wsa = WsAnnotation.objects.get(ws=ws,agent__tag__value='Levobupivacaine')
    rsp = client.get('/mol/{}/assays/{}/'.format((ws.id), (wsa.id)))
    assert rsp.status_code == 200
    from lxml import html, etree
    dom = html.fromstring(rsp.content)
    tables = list(dom.iter('table'))
    assert len(tables) == 2
    # length should match 'expected' array above, plus a header row
    rows = list(tables[0].iter('tr'))
    assert len(rows) == 16
    # first data row should have lowest uM
    # gene name should be missing because protein table isn't loaded
    cells = [''.join(x.itertext()) for x in rows[1].iter('td')]
    assert cells == [
            'ki', u'BDBM50059492\xa0', u'None (P21917)\xa0', '0', '7.5', '1', '0.0',
            ]

