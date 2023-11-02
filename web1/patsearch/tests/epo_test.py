
from patsearch import epo

from dtk.tests.std_vcr import std_vcr
import json
import pytest
from mock import patch, MagicMock


import six
@std_vcr.use_cassette('patsearch/tests/data/epo_parsing.yaml', record_mode='once')
def test_parsing():
    client = epo.EpoClient()
    content = client.fetch_patent('CA-2142016-A1')
    EXPECTED = 'patsearch/tests/data/epo_parsing_expected.json'
    content_json = json.dumps(content, indent=2)

    # Use this to update the expected values if needed.
    if False:
        with open(EXPECTED, 'w') as f:
            f.write(content_json)

    with open(EXPECTED, 'r') as f:
        expected_json = f.read()


    assert json.loads(content_json) == json.loads(expected_json)


@pytest.mark.django_db
@patch('patsearch.epo.EpoClient')
def test_fill_missing(epo_client):
    storage = MagicMock()



    from patsearch.models import Patent, PatentFamily
    PatentFamily.objects.create(family_id='123')
    Patent.objects.create(
            pub_id='CA-0001'
            )
    Patent.objects.create(
            pub_id='US-0001'
            )
    Patent.objects.create(
            pub_id='WO-0001',
            family_id='123'
            )

    def find_content(patent):
        if patent.pub_id == 'WO-0001':
            return None
    storage.find_best_content.side_effect = find_content

    stored = []
    def store_content(family_id, contents):
        stored.append((family_id, contents))
    storage.store_patent_content.side_effect = store_content

    def fetch_patent(pub_id):
        if pub_id == 'CA-0001':
            return {
                'publication_number': 'CA-0001',
                'family_id': '002'
            }
        elif pub_id == 'WO-0001':
            return {
                'publication_number': 'WO-0001',
                'family_id': '123'
            }
        else:
            assert False, "Why are we requesting extra patents? %s" % pub_id
    epo_client().fetch_patent.side_effect = fetch_patent


    epo.fill_missing_content(['CA-0001', 'US-0001', 'WO-0001'], storage)

    assert len(stored) == 2

    assert Patent.objects.get(pk='CA-0001').family_id == '002', "Updated family"



