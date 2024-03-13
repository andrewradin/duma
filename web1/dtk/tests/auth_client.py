

import pytest

@pytest.fixture
def auth_client(django_user_model, client):
    from .end_to_end_test import setup_users
    setup_users(django_user_model, client)
    return client
