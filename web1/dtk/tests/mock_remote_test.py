#!/usr/bin/env python3


"""
A minimal test of the mock_remote infrastructure.

Creates a barebones JobInfo that runs this file remotely, sending along
a single input text file and generating a single output text file.

This should all actually run locally, but using separate tmpdirs for
input and output.
"""
from __future__ import print_function
import os
import sys
if not 'django' in sys.modules:
    # This is only here because this script is also invoked as the
    # remote command, and in that case the django environment isn't
    # set up. In django 1.11, the attempt to import JobInfo without
    # the proper setup will fail.
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    import django
    django.setup()
import mock
from dtk.tests.mock_remote import mock_remote_machine
from path_helper import PathHelper,make_directory
from runner.process_info import JobInfo
from dtk.tests.tmpdir_path_helper import tmpdir_path_helper

class MyJobInfo(JobInfo):
    def __init__(self,ws=None,job=None):
        # base class init
        self.use_LTS=True
        super(MyJobInfo,self).__init__(
                    ws,
                    job,
                    "run_mock_remote_test.py",
                    "Test Job",
                    "Test Job",
                    )


    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)

        self.run_remote()

    
    def run_remote(self):
        infile = os.path.join(self.indir, 'testfile.txt')
        with open(infile, 'w') as f:
            f.write("Some input data")


        cvt = self.mch.get_remote_path
        remote_cores_got = 1
        options = [
                    cvt(infile),
                    cvt(self.outdir),
                    str(remote_cores_got)
                ]
        rem_cmd = cvt(
                os.path.join(PathHelper.website_root, "dtk/tests/", "mock_remote_test.py")
                )
        self.copy_input_to_remote()
        self.make_remote_directories([
                                self.tmp_pubdir,
                                ])
        self.mch.check_remote_cmd(' '.join([rem_cmd]+options))
        self.copy_output_from_remote()

MSG_START = 'Started'
MSG_IN_FILE = 'Checked input file'
MSG_OUT_FILE = 'Wrote output file'
MSG_DONE = 'Done'


@mock.patch("dtk.lts.LtsRepo")
def test_mock_remote(lts_repo, mock_remote_machine, tmpdir_path_helper):
    lts_repo.return_value.path.return_value = PathHelper.lts
    lts_repo.get.return_value = lts_repo.return_value

    from aws_op import Machine
    from mock import MagicMock
    ws = MagicMock()
    ws.id.return_value = "123"
    job = MagicMock()
    job.job_type.return_value = 'test_job'

    ji = MyJobInfo(ws=ws, job=job)
    ji.run()

    expected_msgs = [
            MSG_START,
            'Input: Some input data',
            MSG_IN_FILE,
            MSG_OUT_FILE,
            MSG_DONE
            ]
    for expected, actual in zip(expected_msgs, mock_remote_machine.stdout_lines):
        assert expected == actual

    with open(os.path.join(ji.outdir, 'output.txt')) as f:
        output = f.read()

    assert output == 'Test output data'


if __name__ == "__main__":
    # This gets executed fake-remotely as part of the test.
    import argparse
    parser = argparse.ArgumentParser(description='Run fake remote script')
    parser.add_argument("input", help="Input file")
    parser.add_argument("outdir", help="Where to write output")
    parser.add_argument("cores", type=int, help="Number of cores to use")
    args = parser.parse_args()
    print(MSG_START)

    with open(args.input) as f:
        in_data = f.read()
        print("Input: %s" % in_data)
    print(MSG_IN_FILE)

    with open(os.path.join(args.outdir, 'output.txt'), 'w') as f:
        f.write("Test output data")

    print(MSG_OUT_FILE)
    print(MSG_DONE)
