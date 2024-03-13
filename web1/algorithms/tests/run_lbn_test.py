
import os
import pytest
from mock import patch, Mock, MagicMock

from dtk.tests.tmpdir_path_helper import tmpdir_path_helper
from dtk.tests.std_vcr import std_vcr

from dtk.tests.entrez_utils_test import ez_vcr


import six
@patch('dtk.lts.LtsRepo')
@patch('browse.models.WsAnnotation')
@patch('flagging.utils.get_target_wsa_ids')
@ez_vcr.use_cassette('dtk/testdata/run_lbn.yaml', record_mode='once')
def test_run_lbn(lts_repo, WsAnnotation, get_target_wsa_ids, tmpdir_path_helper, tmpdir):
    get_target_wsa_ids.return_value = [1]
    wsa = MagicMock()
    wsa.id = 1
    wsa.get_name.return_value = 'oxygen'
    WsAnnotation.objects.filter.return_value = [wsa]
    ws = Mock()
    job = Mock()
    ws.name = 'Idiopathic Pulmonary Fibrosis'
    ws.id = 12345678
    from dtk.prot_map import DpiMapping
    job.settings.return_value = {
            'job_id': 1000,
            'score': 'wzs',
            'start': 0,
            'count': 10,
            'condensed': False,
            'add_drugs': 'none',
            'dpi_file': DpiMapping.preferred
            }
    job.job_type.return_value = 'lbn'
    job.id = 7
    from algorithms.run_lbn import MyJobInfo
    ji = MyJobInfo(ws=ws, job=job)
    ji.outfile = str(tmpdir/'lbn.tsv')
    ji.target_outfile = str(tmpdir/'target_lbn.tsv')
    
    uniprot2gene = {
            'P1000': 'DRD2',
            'P00750': 'PLAT'
            }
    ji.get_uniprot_to_gene = lambda *args: (uniprot2gene, None)

    ji.run_lbn()


    assert os.path.exists(ji.target_outfile)


    with open(ji.outfile, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 2
        vals = [float(x.strip()) for x in lines[1].strip().split('\t')]
        # These values aren't meaningful, just check that they haven't
        # changed unintentionally.  We are using a fairly strict comparsion.
        # If it becomes troublesome/brittle we can be more lenient, but for
        # now this lets us know if we are making changes or doing things that
        # aren't reproducible.
        #
        # NOTE: If you change the http recording, these numbers will almost
        # certainly change marginally (e.g. some of are computed based on
        # total # of documents in PubMed, which always changes).  If so,
        # you can update the values below to the new ones if they seem
        # reasonable.
        expecteds = (1.0, 1.1616589972583247, 2.099735119863692e-112, 0.062613430127, 0.000906533445665, 0.0, 0.0)
        for actual, expected in zip(vals, expecteds):
            assert actual == pytest.approx(expected), "Actual results: %s" % vals 
        assert len(vals) == len(expecteds), "Actual results: %s" % vals
