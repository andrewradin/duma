from __future__ import print_function

import pytest

# This file is mostly an experiment at this point. I'm playing with
# pytest fixtures to get some idea what a generic hierarchy of fixtures
# for testing views might look like. The approach has been to provide
# the minimum necessary environment for a view to function, but in
# some cases that might be the wrong approach (e.g. it might be easier
# to set up all the standard properties rather than stub them out
# one by one.

@pytest.fixture
def user():
    '''Return a stub User class.
    '''
    class Dummy: pass
    result = Dummy()
    result.user_name='carl'
    result.is_authenticated=lambda:True
    result.is_staff=True
    return result

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
            ("DB00007","canonical","Leuprolide"),
            ])
    return coll_name

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

@pytest.fixture
def cts(ws):
    '''Return a CT Search with a single drugname.
    '''
    from ctsearch.models import CtSearch,CtDrugName
    cts,new = CtSearch.objects.get_or_create(
            ws=ws,
            user='fixture',
            config=''' {"disease":"schizophrenia"
                       ,"phases":["Phase 3","Phase 4"]
                       ,"completed":true
                       ,"after":""
                    } ''',
            )
    CtDrugName.objects.get_or_create(
            search=cts,
            drug_name='leupro',
            )
    return cts

def dump_tags(node,indent=''):
    '''Show a parsed html tag hierarchy.
    '''
    print(indent,node.tag,list(node.items()))
    for child in node:
        dump_tags(child,indent+'  ')

def nodes_with_text(root,text):
    '''Return a list of all html nodes holding some specified text.
    '''
    return [node
            for node in root.iter("*")
            if node.text and text in node.text
            ]

import pytest
@pytest.mark.skip(reason="Page no longer used, delete once code is gone")
def test_resolve(rf,user,cts):
    '''Test one special case in the Resolve view.

    Given one drug name, that's not an exact match for anything,
    but is a partial match for a single workspace drug, does the
    association happen? Get the resolve page and verify it has
    a link to the Duma drug. (This tests a path that depends on
    find_wsa_match_info returning a set in the first tuple position.)
    '''
    request=rf.get('/cts/%d/ct_resolve/%d/'%(cts.ws.id,cts.id))
    request.user=user
    from ctsearch.views import CtResolveView
    print(ws, cts)
    response = CtResolveView.as_view()(request,ws_id=cts.ws.id,search_id=cts.id)
    assert response.status_code == 200
    from lxml import html, etree
    dom = html.fromstring(response.content)
    l = nodes_with_text(dom,'Duma drug page for')
    assert len(l) == 1
    assert 'Leuprolide' in l[0].text
