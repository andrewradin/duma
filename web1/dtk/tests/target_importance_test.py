from __future__ import print_function


from dtk.target_importance import TargetScoreImportance
import pytest
from pytest import approx
from algorithms.run_trgscrimp import RemoteDataSetup
from path_helper import PathHelper,make_directory
import re
import os
import six
from mock import patch

import six

def test_run_intermediate(tmpdir):
    """Functional test of target importance execution.

    We load in pregenerated input data (generated via dtk/target_importance.py)
    and verify that the generated output matches what we've seen before.

    This will (intentionally) fail if any output score changes.

    If that happens, and it is an expected change, just delete the expected file and rerun this test
    to regenerate it.

    If the input/intermediate data needs to be regenerated, the command is:
    python -m dtk.target_importance --mode output-intermediate -w 119  -z 91148 -c 1 -s 18 --intermediate-file trgimp.pkl
    """

    import gzip
    with gzip.open("dtk/testdata/target_importance.pickle.gz", 'rb') as f:
        from six.moves import cPickle as pickle
        data = pickle.loads(f.read())

    # We fix up the inputs here to make sure we're looking for the input files
    # in the right place. For each of the files that we would copy over to the
    # worker machine, make sure we have it fetched from LTS and update the
    # path in case the machine has a different LTS root.
    rds = RemoteDataSetup(data=data, indir=str(tmpdir))
    for remotefile in rds.files:
        from dtk.lts import LtsRepo
        print("Making sure we have %s" % remotefile.filename)
        m = re.search(r'lts/(\d+)/(\w+)/(\d+)/(.*)', remotefile.filename)
        wsid, jobtype, jobid, relpath = m.group(1, 2, 3, 4)
        lts_rel_root = os.path.join(jobtype, jobid)
        lts_repo = LtsRepo.get(wsid, PathHelper.cfg('lts_branch'))
        lts_repo.lts_fetch(lts_rel_root)

        trg_path = os.path.join(lts_repo.path(), lts_rel_root, relpath)
        print("Look for %s at %s" % (remotefile.filename, trg_path))
        remotefile.filename = trg_path

    # All the paths are now correct, and all the data exists locally, we
    # can now copy it over to the temporary directory to mimic what do when
    # running on worker.
    rds.replace_filenames(cvt=lambda x:x)


    output = TargetScoreImportance.run_pieces(rds.data)

    EXPECTED_FN = 'dtk/testdata/target_importance_expected.json'
    # If the file doesn't exist, regenerate it.
    if not os.path.exists(EXPECTED_FN):
        with open(EXPECTED_FN, 'w') as f:
            import json
            # Sort keys for better diffs when this changes.
            f.write(json.dumps(output, indent=2, sort_keys=True))

    with open(EXPECTED_FN, 'r') as f:
        import json
        expected = json.loads(f.read())

    assert len(output) == len(expected)
    for o, e in zip(output, expected):
        assert type(o) == type(e)
        assert sorted(o.keys()) == sorted(e.keys())
        assert o['version'] == e['version']
        assert sorted(o['score_names']) == sorted(e['score_names'])

        for (out_prot, out_scores), (exp_prot, exp_scores) in zip(sorted(six.iteritems(o['prots'])), sorted(six.iteritems(e['prots']))):
            assert out_prot == exp_prot
            assert out_scores == approx(exp_scores)

        for (out_prot, out_scores), (exp_prot, exp_scores) in zip(sorted(six.iteritems(o.get('pathways', {}))), sorted(six.iteritems(e.get('pathways', {})))):
            assert out_prot == exp_prot
            assert out_scores == approx(exp_scores)


    # If we reached the end successfully, clean up the tmpdir, it can be quite
    # big.
    import shutil
    shutil.rmtree(str(tmpdir))


from runner.process_info import JobInfo
class MyJobInfo(JobInfo):
    def __init__(self,ws=None,job=None):
        super(MyJobInfo,self).__init__(ws,job,'run_fake.py',"","")

@pytest.mark.django_db
def test_src_jobids():
    from runner.models import Process
    from browse.models import Workspace
    import json

    import sys
    JobInfo.directory['fake'] = (sys.modules[__name__], MyJobInfo())

    job_with_src = Process.objects.create(
            name='fake',
            settings_json=json.dumps({
                'srm_blah1_srcjob': '1234',
                'srm_blah_2_srcjob': 567
                }),
            status=0
            )

    ws = Workspace.objects.create()

    from dtk.target_importance import get_wzs_jids
    jids = get_wzs_jids(ws.id, job_with_src.settings())
    assert jids == {'1234': 'blah1', 567: 'blah_2'}

    job_with_fm_code = Process.objects.create(
            name='fake',
            settings_json=json.dumps({
                'fm_code': 'fdf%d' % job_with_src.id,
                }),
            status=0
            )
    fm_jids = get_wzs_jids(ws.id, job_with_fm_code.settings())
    assert fm_jids == jids

    job_with_fm_code2 = Process.objects.create(
            name='fake',
            settings_json=json.dumps({
                'fm_code': 'fvs%d' % job_with_fm_code.id,
                }),
            status=0
            )
    fm2_jids = get_wzs_jids(ws.id, job_with_fm_code2.settings())
    assert fm2_jids == jids







