


from dtk.tests import auth_client

from browse.models import Workspace

def test_workspace_defaults_view(auth_client):
    ws = Workspace.objects.create(name='ws')
    url = '/%s/ws_vdefaults/' % ws.id
    resp = auth_client.get(url)
    assert resp.status_code == 200
    from lxml import html
    from lxml.html import submit_form
    dom = html.fromstring(resp.content)

    def http_fn(method, form_url, values):
        print(("Calling http", method, url, values))
        assert method.lower() == 'post'
        assert form_url == None # This means same as origin page
        # lxml's form data doesn't include the button being pressed.
        post_resp = auth_client.post(url, dict(values, save_btn=True))
        assert post_resp.status_code == 302, "Should redirect on success"

    # Find the form we want.
    form = [x for x in dom.forms if x.get_element_by_id('id_DpiDataset', None) != None]
    assert len(form) == 1
    form = form[0]


    assert ws.get_uniprot_set(ws.get_nonnovel_ps_default()) == set()
    # Submit it.
    submit_form(form, open_http=http_fn)

    assert ws.get_uniprot_set(ws.get_nonnovel_ps_default()) == set()
    expected_intolerable = ws.get_uniprot_set(ws.get_intolerable_ps_default())
    # unwanted in intolerable + nonnovel, but non-novel is empty.
    assert ws.get_uniprot_set('autops_wsunwanted') == expected_intolerable


    form.get_element_by_id('id_DiseaseNonNovelUniprotsSet').value = 'globps_unwanted_tier2'
    ws.refresh_from_db()
    assert ws.get_nonnovel_ps_default() == 'autops_none'
    submit_form(form, open_http=http_fn)
    ws.refresh_from_db()
    assert ws.get_nonnovel_ps_default() == 'globps_unwanted_tier2'

def test_versioned_defaults(db):
    from browse.models import VersionDefault,Workspace,VersionDefaultAudit
    audit_expect = []
    auto_user = 'auto_populated'
    test_user = 'test_user'
    # global get operation should populate any missing global defaults
    g_qs=VersionDefault.objects.filter(ws=None)
    assert g_qs.count() == 0
    d = VersionDefault.get_defaults(None)
    audit_expect.append((None,'flavor2.v1',auto_user))
    assert g_qs.filter(file_class='test').count() == 1
    assert d['test'] == 'flavor2.v1'
    # ws get operation should populate any missing workspace defaults
    ws=Workspace(name='dummy')
    ws.save()
    ws_qs=VersionDefault.objects.filter(ws_id=ws.id)
    assert ws_qs.count() == 0
    d = VersionDefault.get_defaults(ws.id)
    audit_expect.append((ws.id,'flavor2.v1',auto_user))
    assert ws_qs.filter(file_class='test').count() == 1
    assert d['test'] == 'flavor2.v1'
    # ws set should alter ws but not global
    VersionDefault.set_defaults(ws.id,[
            ('test','flavor1.v2'),
            ],'test_user')
    audit_expect.append((ws.id,'flavor1.v2',test_user))
    d = VersionDefault.get_defaults(ws.id)
    assert d['test'] == 'flavor1.v2'
    d = VersionDefault.get_defaults(None)
    assert d['test'] == 'flavor2.v1'
    # global set should alter global but not ws
    VersionDefault.set_defaults(None,[
            ('test','another_flavor.v3'),
            ],'test_user')
    audit_expect.append((None,'another_flavor.v3',test_user))
    d = VersionDefault.get_defaults(ws.id)
    assert d['test'] == 'flavor1.v2'
    d = VersionDefault.get_defaults(None)
    assert d['test'] == 'another_flavor.v3'
    # a new workspace should get then-current globals, not hard-wired defaults
    ws=Workspace(name='dummy2')
    ws.save()
    ws_qs=VersionDefault.objects.filter(ws_id=ws.id)
    assert ws_qs.count() == 0
    d = VersionDefault.get_defaults(ws.id)
    audit_expect.append((ws.id,'another_flavor.v3',auto_user))
    assert ws_qs.filter(file_class='test').count() == 1
    assert d['test'] == 'another_flavor.v3'
    # validate audit trail
    l = list(VersionDefaultAudit.objects.filter(
            file_class='test'
            ).values_list('ws_id','choice','user'))
    assert audit_expect == l
