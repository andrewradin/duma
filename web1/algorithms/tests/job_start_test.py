
from dtk.tests.end_to_end_test import setup_users
import pytest
from mock import patch
from browse.models import Workspace
from dtk.tests.tmpdir_path_helper import tmpdir_path_helper

# There are a variety of CMs that we aren't (yet?) explicitly testing.
# Let's just make sure their start pages can successfully load and generate
# forms and such.
PAGE_NAMES = [
    'ltsconv',
    'phar',
    'struct',
    'tcgamut',
]

@pytest.mark.django_db
@patch("dtk.lts.LtsRepo")
@pytest.mark.parametrize('page_name', PAGE_NAMES)
def test_job_start_page(lts_repo, tmpdir_path_helper, client, django_user_model, page_name):
    ws,new = Workspace.objects.get_or_create(name='Test Workspace')
    setup_users(django_user_model, client)
    url = '/cv/%d/job_start/%s_%d/' % (ws.id, page_name, ws.id)
    assert client.get(url).status_code == 200
    
