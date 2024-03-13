

from dtk.tests import mock_dpi, mock_ppi
from mock import patch
import runpy

import pytest
import six
def test_connect(mock_dpi, mock_ppi):
    if six.PY2:
        pytest.skip("This is a py3 script now")
    mock_dpi('dpi', [
        ('dpimerge_id', 'uniprot_id', 'evidence', 'direction'),
        ('DB01', 'P0CG48', '1.0', '0'),
        ('DB02', 'Q8TBF5', '1.0', '0'),
        ('DB03', 'Q8TBF5', '0.1', '-1'),
        ])
    mock_ppi('ppi', [
        ('uni1', 'uni2', 'evidence', 'direction'),
        ('P0CG48', 'Q8TBF5', '1.0', '0'),
        ('PABCDE', 'Q8TBF5', '1.0', '0'),
        ('Q5MIZ7', 'P60510', '1.0', '0'),
        ('Q5MIZ7', 'P60511', '0.1', '0'),
        ])
    with patch('sys.argv', ['', '--dpi=fake-dpi-file', '--ppi=fake-ppi-file', '-o=/tmp/d2ps.tmp']):
        runpy.run_module('scripts.connect_drugs_to_proteinSets', run_name='__main__')





