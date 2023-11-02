import pytest


from dtk.tests.moa_test import setup_fixture, MOADPI
from dtk.tests.tmpdir_path_helper import tmpdir_path_helper
from dtk.tests.end_to_end_test import assert_good_response
from dtk.tests import make_ws, auth_client
from mock import patch, MagicMock
from wsadmin.custom_dpi import CustomDpiModes
import logging
logger = logging.getLogger(__name__)


class FakeDpiRemote:
    store = {}

    @classmethod
    def push(cls, data, uid):
        logger.info(f"Storing Fake Custom DPI for {uid}: {data}")
        cls.store[uid] = data
    
    @classmethod
    def fetch(cls, uid):
        logger.info(f"Fetching Fake Custom DPI for {uid}")
        if uid in cls.store:
            return cls.store[uid]
        else:
            raise FileNotFoundError(f"Fake dpi remote file not found - {uid}")

# There's code to write pytests to a different namespace in s3, but let's just mock it out anyway,
# it's best not to write to s3 during tests.
@patch('wsadmin.custom_dpi.S3Remote', MagicMock())
@patch('wsadmin.custom_dpi.CustomDpiRemote', FakeDpiRemote)
def test_custom_dpi(setup_fixture, auth_client):
    ws = setup_fixture

    assert_good_response(auth_client.get(f'/wsadmin/{ws.id}/custom_dpi/'))

    from browse.models import ProtSet, Protein
    ps = ProtSet.objects.create(
            ws=ws,
            name='Test PS',
        )
    Protein.objects.create(uniprot='P001', gene='G001')
    Protein.objects.create(uniprot='P002', gene='G002')
    a_prot = Protein.objects.all()[0]

    ps.proteins.add(a_prot)

    from dtk.prot_map import DpiMapping
    from browse.default_settings import DpiDataset
    choices = DpiMapping.choices(ws=ws)
    print("Choices are " ,choices)
    default_dpi = DpiDataset.value(ws=ws)

    from wsadmin.models import CustomDpi

    opts = {
            'dpi': default_dpi,
            'protset': f'ps{ps.id}',
            'mode': CustomDpiModes.EXCLUSIVE,
            'name': 'A Custom Dpi',
            'descr': 'Description',
            'create_btn': True,
            }

    assert len(CustomDpi.objects.all()) == 0

    resp = auth_client.post(f'/wsadmin/{ws.id}/custom_dpi/', opts, follow=True)
    assert_good_response(resp)


    assert len(CustomDpi.objects.all()) == 1

    assert 'A Custom Dpi' in str(DpiMapping.choices(ws=ws))
    assert 'A Custom Dpi' not in str(DpiMapping.choices(ws=None))