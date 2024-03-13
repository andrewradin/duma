import pytest

import scripts.authkeys as ak
from path_helper import PathHelper

def test_key_db():
    key_db = ak.KeyDb()
    assert key_db.path == PathHelper.authkeys_keydb
    key_db.path = PathHelper.website_root+'scripts/tests/mock_keydb/'
    assert key_db.by_user['user1'] == [
            ('machine1','pubkey1'),
            ('machine2','pubkey2 with spaces'),
            ]
    assert key_db.by_user['user2'] == [
            ('machine1','pubkey3'),
            ]
    assert 'user3' not in key_db.by_user
