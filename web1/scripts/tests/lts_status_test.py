

import runpy
from mock import patch


# NOTE: Nothing here is mocked out, so we are only testing things that don't
# write anything.  Also, some operations take a really long time, probably
# want to build out a smaller mock LTS setup if we want to test those.

def test_jobcount():
    with patch('sys.argv', ['', 'jobcount']):
        runpy.run_module('scripts.lts_status', run_name='__main__')

def test_branch():
    with patch('sys.argv', ['', 'branch']):
        runpy.run_module('scripts.lts_status', run_name='__main__')
