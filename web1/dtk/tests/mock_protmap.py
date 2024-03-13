
from mock import patch
import pytest 

@pytest.fixture
def mock_dpi(tmp_path):
    """
    Sets up a fake DPI file with the specified attributes, and forces
    DpiMapping to always return it.
    
    The fixture provides a function that should be invoked with the
    dpi name and the tsv tuple data to insert.
    """
    with patch('dtk.prot_map.DpiMapping.get_path') as dpi_get_path:
        def fn(name, tsv_tuples):
            path = str(tmp_path / (name + ".tsv"))
            dpi_get_path.return_value = path
            with open(path, 'w') as f:
                f.write('\n'.join(['\t'.join(row) for row in tsv_tuples]))
            
        yield fn

@pytest.fixture
def mock_ppi(tmp_path):
    """
    Same as above, but with PPI.
    """
    with patch('dtk.prot_map.PpiMapping.get_path') as ppi_get_path:
        def fn(name, tsv_tuples):
            path = str(tmp_path / (name + ".tsv"))
            with open(path, 'w') as f:
                f.write('\n'.join(['\t'.join(row) for row in tsv_tuples]))

            sqlpath = str(tmp_path / (name + ".sqlsv"))
            ppi_get_path.return_value = sqlpath
            from dtk.tsv_alt import SqliteSv
            SqliteSv.write_from_tsv(sqlpath, path, [str, str, float, int])
            
        yield fn

