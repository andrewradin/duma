

from .auth_client import auth_client
from .ws_with_attrs import ws_with_attrs, make_ws, make_score_job
from .mock_protmap import mock_dpi, mock_ppi
from .mock_pathways import mock_pathways
from .tmpdir_path_helper import tmpdir_path_helper, s3cachedir
from .mock_remote import mock_remote_machine, local_stdjobinfo


def is_pytest():
    import os
    return 'PYTEST_CURRENT_TEST' in os.environ