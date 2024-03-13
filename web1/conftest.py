# Anything in this file is scoped to all web1 tests.
# Used for global session-level fixtures.

import pytest

@pytest.fixture(scope='session', autouse=True)
def auto_reuse_s3():
    # We hit S3 a lot during tests for pointless things, let's try to re-use connections for a big speedup.
    # We could also do this in production, but have to be careful about timeouts / expiry there.
    from unittest.mock import patch
    from dtk.aws_api import AwsBoto3Base
    shared_aws = AwsBoto3Base()

    print("Setting up shared s3 objects")

    # Don't worry about properly patching, since we are intentionally keeping this across all our tests.
    AwsBoto3Base.s3 = shared_aws.s3
    AwsBoto3Base.s3_client = shared_aws.s3_client