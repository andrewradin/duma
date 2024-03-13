

import runpy
from mock import patch

# NOTE: Nothing here is mocked out, so we are only testing things that don't
# write anything.

def test_list_vpcs():
    with patch('sys.argv', ['', 'list_vpcs']):
        runpy.run_module('scripts.make_vpc', run_name='__main__')

def test_list_instances():
    with patch('sys.argv', ['', 'list_instances', '--vpc-name', 'twoxar_vpc']):
        runpy.run_module('scripts.make_vpc', run_name='__main__')

def test_show_vpc():
    with patch('sys.argv', ['', 'show_vpc', '--vpc-name', 'twoxar_vpc']):
        runpy.run_module('scripts.make_vpc', run_name='__main__')
