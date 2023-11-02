
import pytest
from dtk.tests import make_ws, mock_dpi

username = "unit-test-consultant-username"
password = "unit-test-consultant-password"

@pytest.fixture
def cslt_client(django_user_model, client):
    if len(django_user_model.objects.filter(username=username)) == 0:
        user_obj = django_user_model.objects.create_user(
                username=username,
                password=password,
                email="not-an-email")
    else:
        user_obj = django_user_model.objects.filter(username=username)[0]

    from django.contrib.auth.models import Group
    consultant_group, _ = Group.objects.get_or_create(name='consultant')
    consultant_group.user_set.add(user_obj)

    assert not user_obj.is_staff

    from dtk.tests.end_to_end_test import mark_user_access_known
    mark_user_access_known(user_obj)

    client.login(username=username, password=password)

    return client

@pytest.fixture
def basic_ws(make_ws, mock_dpi):
    ws_attrs = []
    for i in range(1, 8):
        ws_attrs += [('DB0%d' % i, 'canonical', 'Drug%d' % i),
                    ('DB0%d' % i, 'm_dpimerge_id', 'DB0%d' % i)]

    ws = make_ws(ws_attrs, name='Main Workspace')
    dpi = [
        ('dpimerge_id', 'uniprot_id', 'evidence', 'direction'),
        ('DB01', 'P01', '0.5', '0'),
        ('DB02', 'P01', '0.9', '0'),
        ('DB03', 'P01', '0.5', '0'),
        ('DB04', 'P01', '0.5', '0'),
        ('DB04', 'P02', '0.5', '1'),
        ('DB05', 'P01', '0.5', '0'),
        ('DB05', 'P02', '0.5', '1'),
        ]
    mock_dpi('fake_dpi', dpi)
    return ws

@pytest.fixture
def basic_ws2(make_ws, mock_dpi):
    ws_attrs = []
    for i in range(1, 8):
        ws_attrs += [('DB0%d' % i, 'canonical', 'Drug%d' % i),
                    ('DB0%d' % i, 'm_dpimerge_id', 'DB0%d' % i)]

    ws = make_ws(ws_attrs, name='Workspace 2')
    return ws

def rsp_text(rsp):
    from lxml import html, etree
    dom = html.fromstring(rsp.content)
    text = dom.text_content()
    return text

def test_login(cslt_client):
    rsp = cslt_client.get('/')
    assert rsp.status_code == 302, "Any inaccessible page should redirect"
    assert rsp['Location'] == '/consultant/'

    rsp = cslt_client.get('/', follow=True)
    assert rsp.status_code == 200, 'Following 302 should get you to a 200'

def test_front_page(cslt_client, basic_ws, basic_ws2, django_user_model):
    rsp = cslt_client.get('/consultant/')
    assert rsp.status_code == 200
    assert 'Recommendation' not in rsp_text(rsp), "No molecules to review yet"

    # Create an election.
    from browse.models import Election, Vote, ElectionFlavor, WsAnnotation
    from rvw.views import ElectionView
    from notes.models import Note
    election = Election.objects.create(
            ws=basic_ws,
            due="2111-11-11",
            flavor='pass1',
            )
    
    reviewer_username = 'Reviewer1'
    reviewer = django_user_model.objects.create_user(username=reviewer_username)
    consultant = django_user_model.objects.get(username=username)

    # Setup an election with one reviewer, who isn't the consultant.
    users = [reviewer]
    wsas = list(WsAnnotation.objects.filter(ws=basic_ws))[:2]
    ElectionView.setup_election(users, wsas, election)

    rsp = cslt_client.get('/consultant/')
    assert rsp.status_code == 200
    assert 'Recommendation' not in rsp_text(rsp), "Still no molecules to review yet"

    # Add the consultant to the election, test access.
    users = [reviewer, consultant]
    wsas = list(WsAnnotation.objects.filter(ws=basic_ws))[:2]
    non_wsas = list(WsAnnotation.objects.filter(ws=basic_ws))[2:]
    ElectionView.setup_election(users, wsas, election)
    rsp = cslt_client.get('/consultant/')
    assert rsp.status_code == 200
    page_text = rsp_text(rsp)
    assert 'Recommendation' in page_text, "Molecules to review!"
    assert basic_ws.name in page_text, 'Should be in first workspace'
    assert basic_ws2.name not in page_text, 'Nothing to review in second one'

    assert wsas[0].agent.canonical in page_text, "Reviewed drug should be there"
    assert non_wsas[0].agent.canonical not in page_text, "Non-reviewed drug should not be there"

    rsp = cslt_client.get(f'/consultant/{basic_ws.id}/molecule/{wsas[0].id}/')
    assert rsp.status_code == 200, "Able to access page for reviewed drug"

    rsp = cslt_client.get(f'/consultant/{basic_ws.id}/molecule/{non_wsas[0].id}/')
    assert rsp.status_code == 302, "Unable to access page for unreviewed drug"


    # Check that note text is appropriately visible.
    consultant_v = Vote.objects.get(drug=wsas[0], reviewer=username)
    reviewer_v = Vote.objects.get(drug=wsas[0], reviewer=reviewer_username)
    Note.set(consultant_v,'note', username, "Consultant drug review comment")
    Note.set(reviewer_v,'note', reviewer_username, "Reviewer drug review comment")
    consultant_v.save()
    reviewer_v.save()
    rsp = cslt_client.get('/consultant/')
    assert rsp.status_code == 200
    page_text = rsp_text(rsp)
    assert 'Consultant drug review comment' in page_text
    assert 'Reviewer drug review comment' not in page_text


    # Force-close the election, make sure it disappears
    election.vote_set.filter(recommended__isnull=True).update(
                                                disabled=True
                                                )
    rsp = cslt_client.get('/consultant/')
    assert rsp.status_code == 200
    page_text = rsp_text(rsp)
    assert 'Recommendation' not in page_text, "No more molecules to review!"


    rsp = cslt_client.get(f'/consultant/{basic_ws.id}/molecule/{wsas[0].id}/')
    assert rsp.status_code == 302, "No longer able to access page for closed drug"
