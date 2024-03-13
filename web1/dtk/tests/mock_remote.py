from mock import patch
import pytest
import os
import logging
logger = logging.getLogger(__name__)

from threading import Lock

local_stdjob_lock = Lock()

@pytest.fixture
def local_stdjobinfo():
    """Forces stdjobinfo to run remote commands locally in-process
    
    Unlike mock_remote_machine this keeps things in-process so mocks are still in place,
    but only intercepts stdjobinfo's run_remote_cmd.
    """
    def run_local(self, cmd):
        logger.info(f"Running stdjobinfo run_remote_cmd locally: {cmd}")

        import runpy
        with local_stdjob_lock:
            # Important to set fake_mp if we're running jobs in-process, because pmap will otherwise fork
            # and forking will randomly break things in threaded python code.
            # The most common way for this to manifest is MySQL giving spurious lost connection errors in
            # workflow-style tests.
            with patch('dtk.parallel.force_fake_mp', True):
                with patch('sys.argv', [''] + cmd[1:]):
                    runpy.run_path(cmd[0], run_name='__main__')
        


    with patch('runner.process_info.StdJobInfo._run_local', side_effect=run_local, autospec=True):
        from runner.process_info import StdJobInfo
        with patch.object(StdJobInfo, 'force_local', True):
            yield None


@pytest.fixture(autouse=True)
def mock_remote_machine(tmpdir):
    from path_helper import PathHelper,make_directory
    fake_remote_dir=str(tmpdir/"remote_mch")  + "/"

    make_directory(os.path.join(fake_remote_dir, '2xar', 'publish'))
    #make_directory(os.path.join(fake_remote_dir, '2xar', 'ws', 'dpi'))
    #make_directory(os.path.join(fake_remote_dir, '2xar', 'ws', 'glee'))
    #make_directory(os.path.join(fake_remote_dir, '2xar', 'ws', 'd2ps'))
    make_directory(os.path.join(fake_remote_dir, 'tmp'))

    from aws_op import Machine
    class MockRemote(Machine):
        SSH_HOST_SEP = ''
        def run_remote_cmd(self, cmd, hold_output=False, venv=None):
            import subprocess
            cmd_parts = cmd.split(' ')
            shell = False
            if cmd_parts[0] == 'source' and cmd_parts[1].endswith('activate'):
                # We're trying to do activate, which is not necessary here.
                cmd_parts = cmd_parts[4:]
            if '&&' in cmd_parts or 'faers' in cmd or '|' in cmd or '*' in cmd:
                # && isn't going to work, I'd like to get rid of these
                # but for now we can use shell.
                # faers quotes around spaces, also needs this for now
                cmd_parts = ' '.join(cmd_parts)
                shell = True


            print(("Mock running remote cmd", cmd_parts))
            proc = subprocess.Popen(cmd_parts, stdout=subprocess.PIPE, shell=shell)
            out, err = proc.communicate()
            out = out.decode('utf8')
            print("STDOUT: ")
            print(out)
            self.stdout_lines = out.split('\n')

            return proc.returncode


        def do_start(self):
            make_directory(fake_remote_dir)

        def _ssh_dest(self):
            return ''

        def _remote_mkdir(self, remote_dir):
            dir_to_create = os.path.join(fake_remote_dir, remote_dir)
            # print("Test creating dir %s" % dir_to_create)
            from path_helper import make_directory
            make_directory(dir_to_create)

        @staticmethod
        def get_remote_path(local_path):
            out = os.path.relpath(local_path, PathHelper.home)
            #print("Getting remote of ", local_path)
            if not out.startswith('..'):
                if local_path.endswith('/') and not out.endswith('/'):
                    out += '/'
                # print("Getting path %s relative to %s = %s" % (local_path, PathHelper.install_root, out))
                return os.path.join(fake_remote_dir, out)
            elif local_path.startswith('/tmp') and not local_path.startswith('/tmp/teste2ecache'):
                return os.path.join(fake_remote_dir, local_path[1:])
            else:
                return local_path

        def get_full_remote_path(self,local_path):
            # For testing purposes don't need any difference right now.
            return self.get_remote_path(local_path)



    with patch('aws_op.Machine', new=MockRemote):
        # When we import Machine here, it initializes a bunch of predefined
        # machines.  We want to wipe those out and replace them with our
        # test ones.
        old_name_index = Machine.name_index
        Machine.name_index = {}
        worker_test_mch = MockRemote('worker-test'
                ,instance_type='m4.4xlarge'
                ,role='worker'
                ,has_buttons=True)
        # This one is hardcoded for e.g. accessing S3.
        worker_mch = MockRemote('worker'
                ,instance_type='m4.4xlarge'
                ,role='worker'
                ,has_buttons=True)
        qa01_mch = MockRemote('worker-qa'
                ,instance_type='m4.4xlarge'
                ,role='worker'
                ,has_buttons=True)
        lts_mch = MockRemote('lts')
        from path_helper import PathHelper
        worker = Machine.name_index[PathHelper.cfg('worker_machine_name')]
        yield worker

        Machine.name_index = old_name_index

