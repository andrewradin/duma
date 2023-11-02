from __future__ import print_function

from drugs.models import Drug,Prop,Collection,Tag,DpiMergeKey
from drugs.tests import create_dummy_collection

from browse.models import WsCollectionInfo,Workspace,WsAnnotation
import pytest
from dtk.tests import make_ws

def check_collection_info(
        ws_id,
        collection,
        all_keys=[],
        loaded_keys=[],
        loaded_agent_keys=[],
        blocked_keys=[],
        removed_keys=[],
        counts=None,
        ):
    wci = WsCollectionInfo(ws_id)
    total,in_ws,can_add,duped = wci.import_counts(collection)
    assert total == in_ws + can_add + duped
    if counts:
        assert total == counts[0]
        assert in_ws == counts[1]
        assert can_add == counts[2]
        assert duped == counts[3]


def import_drug(ws_id,col,native_key):
    p = Prop.objects.get(name=col.key_name)
    t = Tag.objects.get(prop=p,value=native_key)
    wsa = WsAnnotation(ws_id=ws_id,agent_id=t.drug_id)
    wsa.save()

def add_to_cluster(cluster_key,version,other_key_name,other_key_value):
    # create DpiMergeKey records for all agents with 'other key'
    # across all collections
    for tag in Tag.objects.filter(
            prop__name=other_key_name,
            value=other_key_value,
            ):
        dmk = DpiMergeKey(
                drug_id=tag.drug_id,
                version=version,
                dpimerge_key=cluster_key,
                )
        dmk.save()

#def test_unversioned_counts(db):
#    run_collection_test(None)

def test_versioned_counts(db):
    run_collection_test(1)

def run_collection_test(version):
    # When this test was originally written, all the clustering and blocking
    # was key-based, so the keys are tracked below and passed to the check
    # function. This is no longer true, but the expected keys document what
    # the test is doing, so they're left in place.
    #
    # empty collection and workspace
    c1_name='chembl.test1'
    create_dummy_collection(c1_name,data=[])
    c1 = Collection.objects.get(name=c1_name)
    ws,new = Workspace.objects.get_or_create(name='Test Workspace')
    # mock to return expected test version
    if version:
        from browse.default_settings import DpiDataset
        dpi_default = f'dummy_dpi.v{version}'
        DpiDataset.set(ws, dpi_default, 'test')
    # verify workspace is set up correctly
    assert new
    assert ws.get_dpi_version() == version
    expect_c1={}
    expect_c1['counts'] = [0,0,0,0]
    check_collection_info(ws.id,c1,**expect_c1)
    # drug in collection, not imported
    create_dummy_collection(c1_name,data=[
            ('CHEMBL1','canonical','name1'),
            ('CHEMBL3','canonical','name3'),
            ])
    expect_c1['all_keys'] = ['CHEMBL1','CHEMBL3']
    expect_c1['counts'] = [2,0,2,0]
    check_collection_info(ws.id,c1,**expect_c1)
    # drug in collection, imported
    import_drug(ws.id,c1,'CHEMBL1')
    expect_c1['loaded_keys'] = ['CHEMBL1']
    expect_c1['loaded_agent_keys'] = ['CHEMBL1']
    expect_c1['counts'] = [2,1,1,0]
    check_collection_info(ws.id,c1,**expect_c1)
    import_drug(ws.id,c1,'CHEMBL3')
    expect_c1['loaded_keys'] += ['CHEMBL3']
    expect_c1['loaded_agent_keys'] += ['CHEMBL3']
    expect_c1['counts'] = [2,2,0,0]
    check_collection_info(ws.id,c1,**expect_c1)
    # another collection subset
    c2_name='chembl.test2'
    create_dummy_collection(c2_name,data=[
            ('CHEMBL1','canonical','name1'),
            ('CHEMBL2','canonical','name2'),
            ])
    c2 = Collection.objects.get(name=c2_name)
    check_collection_info(ws.id,c1,**expect_c1)
    expect_c2=dict(
            all_keys=['CHEMBL1','CHEMBL2'],
            loaded_keys=['CHEMBL1','CHEMBL3'],
            counts=[2,0,1,1],
            )
    check_collection_info(ws.id,c2,**expect_c2)
    # test that removed drugs aren't counted in collection
    create_dummy_collection(c2_name,data=[
            ('CHEMBL4','canonical','name4'),
            ])
    agent = Drug.objects.get(tag__value='name4')
    agent.removed=True
    agent.save()
        # at this point:
        # - c1 has CHEMBL1, CHEMBL3
        # - c2 has CHEMBL1, CHEMBL2, CHEMBL4 (with CHEMBL4 marked removed)
        # no clusters
        # ws has CHEMBL1, CHEMBL3 from c1
    check_collection_info(ws.id,c1,**expect_c1)
    check_collection_info(ws.id,c2,**expect_c2)
    # ...or in workspace
    import_drug(ws.id,c2,'CHEMBL4')
    check_collection_info(ws.id,c1,**expect_c1)
    check_collection_info(ws.id,c2,**expect_c2)
    # check that removed drugs don't block
    if version:
        add_to_cluster('cluster1',version,'chembl_id','CHEMBL2')
        add_to_cluster('cluster1',version,'chembl_id','CHEMBL4')
    else:
        agent = Drug.objects.get(tag__value='name4')
        agent.set_prop('m_chembl_id','CHEMBL2')
        # at this point:
        # - c1 has CHEMBL1, CHEMBL3
        # - c2 has CHEMBL1, CHEMBL2, CHEMBL4 (with CHEMBL4 marked removed)
        # CHEMBL2 and CHEMBL4 are clustered
        # ws has CHEMBL1, CHEMBL3 from c1, CHEMBL4 from c2
    check_collection_info(ws.id,c1,**expect_c1)
    check_collection_info(ws.id,c2,**expect_c2)
    # ...but normal drugs do
    if version:
        add_to_cluster('cluster1',version,'chembl_id','CHEMBL3')
        expect_c1['blocked_keys'] = ['CHEMBL3']
        # even though it's removed, CHEMBL4 is now blocked by CHEMBL3
        expect_c2['blocked_keys'] = ['CHEMBL2','CHEMBL4']
    else:
        agent = Drug.objects.get(tag__value='name3')
        agent.set_prop('m_chembl_id','CHEMBL2')
        # note that CHEMBL2 shows up in the blocked list for both subsets,
        # even though it's only present in c2, but it doesn't show up in counts
        expect_c1['blocked_keys'] = ['CHEMBL2']
        expect_c2['blocked_keys'] = ['CHEMBL2']
        # at this point:
        # - c1 has CHEMBL1, CHEMBL3
        # - c2 has CHEMBL1, CHEMBL2, CHEMBL4 (with CHEMBL4 marked removed)
        # CHEMBL2, CHEMBL3, CHEMBL4 are clustered
        # ws has CHEMBL1, CHEMBL3 from c1, CHEMBL4 from c2
    expect_c2['counts'] = [2,0,0,2]
    check_collection_info(ws.id,c1,**expect_c1)
    check_collection_info(ws.id,c2,**expect_c2)

def test_import_single(make_ws):
    attrs = [
                ("DB07","canonical","Drug1"),
                ("DB08","canonical","Drug2"),
            ]
    ws = make_ws(attrs)

    agents = Drug.objects.all()
    ws2 = Workspace.objects.create(name='Test WS')
    ws2.get_dpi_version = lambda *x: 11



    assert len(WsAnnotation.objects.filter(ws=ws2)) == 0
    ws2.import_single_molecule(agents[0].id, user='')
    assert len(WsAnnotation.objects.filter(ws=ws2)) == 1
    ws2.import_single_molecule(agents[1].id, user='')
    assert len(WsAnnotation.objects.filter(ws=ws2)) == 2

    with pytest.raises(Exception):
        # Can't reimport.
        ws2.import_single_molecule(agents[0].id, user='')


    # Remove/invalidate a WSA.
    wsa = WsAnnotation.objects.get(ws=ws2, agent=agents[0])
    wsa.invalid = True
    wsa.save()

    assert len(WsAnnotation.objects.filter(ws=ws2)) == 1

    # Reimporting it should bring it back.
    ws2.import_single_molecule(wsa.agent_id, user='')
    assert len(WsAnnotation.objects.filter(ws=ws2)) == 2



