

from mock import patch
import pytest 

@pytest.fixture
def mock_pathways(tmp_path):
    """
    Sets up a fake pathways file.
    
    The fixture provides a function that should be invoked with the
    pathways data to insert.

    [(pathway_id, [P1, P2, P3, ...]), ...]

    NOTE: This is probably an incomplete mocking, there are more places that this
    could be made to override.
    """
    with patch('dtk.gene_sets.get_gene_set_file') as get_gs_file, \
         patch('dtk.gene_sets.get_pathway_sets') as get_pw_sets:
        def fn(pathway_data):
            path = str(tmp_path / ("fake_pathways.tsv"))
            class FakeS3File:
                def fetch(self): pass
                def path(self):
                    return path
            get_gs_file.return_value = FakeS3File()
            with open(path, 'w') as f:
                for pw, prots in pathway_data:
                    f.write('\t'.join([pw, ','.join(prots)]) + '\n')
        
        get_pw_sets.return_value = []
            
        yield fn
