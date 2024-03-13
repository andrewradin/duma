#!/usr/bin/env python3
# ex: set tabstop=4 expandtab:

import time
import subprocess
import os
import logging
logger = logging.getLogger(__name__)

class Machine:
    name_index = {}
    op_prefix = "do_"
    # This only exists so it can be overridden in a test.
    SSH_HOST_SEP = ':'
    @classmethod
    def get_op_list(cls):
        l = []
        for n in cls.__dict__:
            if n.startswith(cls.op_prefix):
                l.append(n[len(cls.op_prefix):])
        return l
    def __init__(self,name,**kwargs):
        self.name = name
        if self.name:
            self.name_index[name] = self
        self.on_vpc_instance = None
        self.region = kwargs.pop('region','us-west-2')
        self.poll_interval_secs = 4
        self.max_ssh_wait_secs = 20
        self.max_ssh_wait_secs_after_start = 120
        self.max_wait_secs = 180
        self.build_spec = kwargs.pop('template',None)
        self.has_buttons = kwargs.pop('has_buttons',False)
        self.default_type = kwargs.pop('instance_type','t2.micro')
        self.verbose = kwargs.pop('verbose',True)
        self.ssh_user = kwargs.pop('ssh_user','ubuntu')
        # special case for apache
        # XXX there may be a better way to do this; also, if we're going to
        # XXX expect apache to have keyfiles here, the install script should
        # XXX verify their presence (although it's a one-time thing)
        home = os.environ.get('HOME','/home/www-data/')
        # ssh_keypair should normally be empty, and ssh access should use
        # the client machine's default key, as managed by authkey.py;
        # the capability was left in place in case it's needed someday
        self.ssh_keypair = kwargs.pop('ssh_keypair',None)
        if self.ssh_keypair:
            self.ssh_keypath = '%s/.ssh/%s.pem' % (home,self.ssh_keypair)
        else:
            self.ssh_keypath = None
        self.local_file = kwargs.pop('local_file','/tmp/test')
        self.remote_file = kwargs.pop('remote_file','')
        self.remote_cmd = kwargs.pop('remote_cmd','uname -a')
        self.role = kwargs.pop('role',None)
        self.local_key_buffer = kwargs.pop('local_key_buffer','key-in-transit')
        self.dest_ip = None
        self._ec2_instance = None
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)
    def _in_aws(self):
        if self.on_vpc_instance == None:
            from dtk.aws_api import AwsBoto3Base
            ctrl = AwsBoto3Base()
            self.on_vpc_instance = ctrl.in_aws
        return self.on_vpc_instance
    def _get_vpc_control(self):
        from dtk.aws_api import VpcControl
        ctrl = VpcControl()
        ctrl.region_name = self.region
        return ctrl
    def _ssh_opts(self):
        opts = ['-o', 'StrictHostKeyChecking=no']
        if self.ssh_keypath:
            opts += ['-i',self.ssh_keypath]
        return opts
    def _ssh_dest(self):
        if self.dest_ip is None:
            i = self.get_ec2_instance()
            self.dest_ip = i.dest_ip
        return self.ssh_user + '@' + self.dest_ip
    def do_show_ips(self):
        i = self.get_ec2_instance()
        print('private_ip:',i.private_ip_address)
        print('public_ip:',i.ip_address)
        print('best ssh ip:',i.dest_ip)
    def check_rsync_web1(self):
        from path_helper import PathHelper
        if PathHelper.cfg('rsync_worker'):
            # This is a dev setting to rsync the web1 directory over to the worker,
            # to save the hassle of iterating over git branches.
            cmd = ['rsync', '-v', '-r',
                    # Arcane syntax to only do .py files, but also exclude local_settings.py.
                    '--exclude=local_settings.py',
                    '--exclude=js/', # Also don't need the big js dir
                    '--include=*/',
                    '--include=*.py',
                    '--include=*.sh',
                    '--include=*.R',
                    '--exclude=*',
                    '-e', 'ssh',
                    PathHelper.website_root,
                    self._ssh_dest() + self.SSH_HOST_SEP + PathHelper.website_root]
            self.run_popen(cmd)
    def run_remote_cmd(self,cmd,hold_output=False,venv='py3web1'):
        if venv:
            import os
            from path_helper import PathHelper
            activate_path = os.path.join(PathHelper.venv, 'bin/activate')
            src = "source " + self.get_remote_path(activate_path) + " " + venv
            cmd = src + " && " + cmd

        i = self.do_start()
        # XXX The -tt option below was supposed to cause a local sighup to
        # XXX show up as a sighup on the other side, but that may be mistaken.
        # XXX The most convincing info I could find indicates that it just
        # XXX affects what happens when a ^C is typed (the actual character
        # XXX gets sent to the remote pty, which interprets it and generates
        # XXX the remote signal). See:
        # XXX https://unix.stackexchange.com/questions/102061/ctrl-c-handling-in-ssh-session/161878#161878
        # XXX So, it's maybe not necessary, and in fact simple experimenting
        # XXX with "ssh worker-test sleep 100 &" seems to indicate that the
        # XXX remote process dies when the local ssh is killed. I think the
        # XXX mechanism is:
        # XXX - when the platform-side process dies, the platform OS closes
        # XXX   the socket
        # XXX - when sshd sees the socket closed, it sends SIGHUP to its child
        # XXX - if the child is bash (it usually is), it distributes the SIGHUP
        # XXX   to its children
        # XXX But, it doesn't seem to hurt anything, and something prompted us
        # XXX to put it here in the first place, so I'm leaving it in case it
        # XXX catches some obscure case with multiple worker-side processes.
        ssh_cmd = ['ssh','-tt'] + self._ssh_opts()
        # Running multi_refresh.py on dev07 (on azure) results in some
        # long-running remote jobs terminating on worker-test, but the
        # master ssh job not terminating on dev07. These eventually time
        # out in 130-140 minutes, and a "broken pipe" error appears in
        # the log. The sshd process and the netstat connection on the
        # worker side disappear long before this.
        #
        # Since this happens only on long-running jobs, presumably there's
        # some kind of timeout that causes data not to be successfully
        # passed back to Azure when the worker job finally terminates.
        # Maybe this happens in the Azure-side firewall/NAT (there's
        # some evidence on the web that this has a short ~2 minute timeout).
        #
        # In any event, turning on keep-alive seems to fix it. It could
        # be made configurable, but using a 30-second interval everywhere
        # shouldn't cause problems.
        ssh_cmd += ['-o','ServerAliveInterval=30']
        ssh_cmd += [ self._ssh_dest(), cmd ]
        return self.run_popen(ssh_cmd,hold_output=hold_output)
    def check_remote_cmd(self,cmd,hold_output=False,venv='py3web1'):
        exit_code = self.run_remote_cmd(cmd,hold_output=hold_output,venv=venv)
        if exit_code:
            raise RuntimeError("got bad exit code %d" % exit_code)
        return exit_code
    def run_popen(self,local_cmd
                ,stdin=subprocess.PIPE
                ,hold_output=False
                ):
        outmode = subprocess.PIPE if hold_output else None
        if self.verbose:
            logger.info("executing %s",local_cmd)
        p = subprocess.Popen(local_cmd
                , stdin=stdin
                , stdout=outmode
                , stderr=outmode
                )
        (self.stdout,self.stderr) = p.communicate()
        if self.verbose:
            if self.stdout:
                print('STDOUT:')
                print(self.stdout)
            if self.stderr:
                print('STDERR:')
                print(self.stderr)
        if self.verbose:
            logger.info("Done executing with returncode %s", p.returncode)
        return p.returncode
    @staticmethod
    def get_remote_path(local_path):
        remote_path = local_path
        if remote_path.startswith('/home/'):
            # use relative path name on remote side
            l = remote_path.split('/')
            remote_path = '/'.join(l[3:])
        return remote_path
    def get_full_remote_path(self,local_path):
        # R commands don't run in the worker home directory,
        # so they need a full path to the file
        # XXX this could probably replace the above, which
        # XXX was trying to avoid a $HOME dependency on the
        # XXX worker side
        remote_path = local_path
        if remote_path.startswith('/home/'):
            l = remote_path.split('/')
            l[2] = self.ssh_user
            remote_path = '/'.join(l)
        return remote_path
    def _get_remote_file(self):
        if self.remote_file:
            return self.remote_file
        return self.get_remote_path(self.local_file)
    def do_copy_to(self):
        self.copy_to(self.local_file,self._get_remote_file())
    def do_copy_from(self):
        self.copy_from(self._get_remote_file(),self.local_file)

    def _remote_mkdir(self, remote_dir):
        self.check_remote_cmd("mkdir -p "+remote_dir)
    # both the following have the form (src,dest)
    # - dst should never end with /
    # - if src ends with / it's a recursive copy
    def copy_to(self,local_file,remote_file):
        assert not remote_file.endswith('/')
        self.do_start()
        l = remote_file.split('/')
        remote_dir = '/'.join(l[:-1])
        self._remote_mkdir(remote_dir)
        scp_cmd = ['scp'] + self._ssh_opts()
        if local_file.endswith('/'):
            scp_cmd.append('-r')
            # destination dir name must match
            assert l[-1] == local_file.split('/')[-2]
            remote_file = remote_dir
        scp_cmd += [ local_file ]
        scp_cmd += [ "%s%s%s" % (self._ssh_dest(),self.SSH_HOST_SEP,remote_file) ]
        if self.verbose:
            print("executing",scp_cmd)
        subprocess.check_call(scp_cmd)
    def copy_from(self,remote_file,local_file):
        assert not local_file.endswith('/')
        i = self.do_start()
        l = local_file.split('/')
        local_dir = '/'.join(l[:-1])
        subprocess.check_call(['mkdir','-p',local_dir])
        scp_cmd = ['scp'] + self._ssh_opts()
        if remote_file.endswith('/'):
            scp_cmd.append('-r')
            # destination dir name must match
            assert l[-1] == remote_file.split('/')[-2]
            local_file = local_dir
        scp_cmd += [ "%s%s%s" % (self._ssh_dest(),self.SSH_HOST_SEP,remote_file) ]
        scp_cmd += [ local_file ]
        if self.verbose:
            print("executing",scp_cmd)
        subprocess.check_call(scp_cmd)
    def do_remote_cmd(self):
        self.check_remote_cmd(self.remote_cmd)
    def get_ec2_instance(self,force_refresh=False):
        if self._ec2_instance is None:
            from dtk.aws_api import Ec2Instance
            self._ec2_instance = Ec2Instance(name=self.name)
            self._ec2_instance.region_name = self.region
        if force_refresh:
            self._ec2_instance.refresh_instance_data()
        return self._ec2_instance
    def _get_all_regions(self):
        return set(x.region for x in self.name_index.values())
    def get_all_vpcs(self):
        '''Yields a VpcControl for each VPC across all regions.

        This simplifies writing commands that should affect the entire
        deployment.
        '''
        from dtk.aws_api import VpcControl
        for region in self._get_all_regions():
            rctrl = VpcControl()
            rctrl.region_name = region
            for ctrl in rctrl.all_region_vpcs():
                yield ctrl
    def do_backup_check(self):
        age_thresh = 1 # in days
        size_thresh = 1 # in percent
        from dtk.aws_api import Bucket
        # get names and sizes of everything in the database backup bucket
        b=Bucket(name='2xar-backups')
        prefix='production/database/'
        l=[
            (x.key[len(prefix):],x.size)
            for x in b.aws_bucket.objects.filter(Prefix=prefix)
            ]
        # get info on most recent backup
        l.sort(key=lambda x:x[0],reverse=True)
        last_file,last_size = l[0]
        # figure out how old it is
        import datetime
        last_date = datetime.date(*[int(x) for x in last_file.split('.')[:3]])
        age_in_days = (datetime.date.today() - last_date).days
        # warn if too old
        if age_in_days > age_thresh:
            print('no backup since',last_file)
        # Compare latest backup vs 2nd latest.
        # Comparing against e.g. the average instead is problematic because if this does
        # legitimately change by >thresh, it will warn for days.
        second_last_size = l[1][1]
        # get percent deviation from second last backup.
        pct_dev = 100 * (last_size-second_last_size)/second_last_size
        # report anything suspicious; typical variation is on the order of
        # a few hundreths of a percent
        if abs(pct_dev) > size_thresh:
            # use flag_file to warn only once per backup.
            flag_file = f'/tmp/.aws_warn.{last_file}'
            if not os.path.exists(flag_file):
                print(f'backup size changed by {pct_dev:.3f}%')
                # Write out this file so we don't re-warn for this backup.
                open(flag_file, 'w')

    def do_s3_check(self):
        from dtk.aws_api import S3Check
        ctrl = S3Check()
        ctrl.check_access()
    def do_sg_check(self):
        for ctrl in self.get_all_vpcs():
            ctrl.one_vpc_sg_check()
    def do_inst_check(self):
        stopped = set()
        running = set()
        for ctrl in self.get_all_vpcs():
            for i in ctrl.instances:
                name = ctrl.get_tag('Name',i.tags)
                if i.state['Name'] == 'stopped':
                    stopped.add(name)
                else:
                    # anything not stopped counts as running
                    running.add(name)
        expect_running = set([
                'selenium',
                'lts',
                'platform',
                #'dev08',
                'qa01',
                'ariaweb1',
                ])
        expect_stopped = set(Machine.name_index) - expect_running
        for label,machines in [
                ('running:',running & expect_running),
                ('    extra:',running - expect_running),
                ('    missing:',expect_running - running),
                ('stopped:',stopped & expect_stopped),
                ('    extra:',stopped - expect_stopped),
                ('    missing:',expect_stopped - running - stopped),
                ]:
            if not machines:
                if label.startswith(' '):
                    continue # don't show empty indented labels
                machines = 'None'
            else:
                machines = ' '.join(sorted(machines))
            # wrap labels in '*' for bold on slack
            print(f'*{label}*',machines)

    def do_rebuild_sg(self):
        from dtk.alert import slack_send
        slack_send(f'running rebuild_sg on {os.uname()[1]}')
        for ctrl in self.get_all_vpcs():
            ctrl.one_vpc_sg_rebuild()
    def alter_sftp_fw(self,acct,enable):
        raise NotImplementedError()
        # XXX If this is ever needed again, it can be implemented under
        # XXX the FirewallConfig setup by doing something like this:
        # XXX - use acct name to retrieve region, SG, and new IP
        # XXX - if enable:
        # XXX   - build a ruleset to enable the new IP
        # XXX   - call ruleset.add_to_sg
        # XXX - else:
        # XXX   - extract FC name from SG name
        # XXX   - extract unmodified ruleset from FC
        # XXX   - call configure_sg to restore old ruleset
    def do_fw_open(self):
        self.alter_sftp_fw(self.sftp_account,enable=True)
    def do_fw_close(self):
        self.alter_sftp_fw(self.sftp_account,enable=False)
    def list_images(self,ubuntu_release):
        from dtk.aws_api import Ec2Image
        for img in sorted(
                Ec2Image.get_image_list(ubuntu_release,self.region),
                key=lambda x: x.name,
                ):
            print(img.id, img.name)
    def do_list_releases(self):
        from dtk.aws_api import Ec2Image
        imgs = Ec2Image.get_image_list('*',self.region)
        releases = set(
                tuple(x.name.split('/')[3].split('-')[:3])
                for x in imgs
                )
        for t in sorted(releases,key=lambda x:x[2]):
            print('-'.join(t))
    def wait_for_ssh(self,wait_secs=None):
        # Since an ssh timeout can itself take a considerable amount of
        # time, the approach in _change_state to pre-calculate the max
        # number of cycles isn't appropriate here.  Instead, wait for the
        # time specified by max_ssh_wait_secs, and guarantee there will be
        # at least one retry after a failure.  The sleep is still needed to
        # prevent banging on the instance in the 'connection refused' case,
        # which returns quickly.
        if wait_secs is None:
            wait_secs = self.max_ssh_wait_secs
        end_time = time.time() + wait_secs
        extra_cycles = 1
        poll_interval = 1
        while extra_cycles >= 0:
            time.sleep(poll_interval)
            poll_interval = self.poll_interval_secs
            ssh_cmd = ['ssh'] + self._ssh_opts()
            ssh_cmd += [ self._ssh_dest(), 'echo',
                    'ssh to', self.name, 'successful'
                    ]
            if subprocess.call(ssh_cmd) == 0:
                return
            if time.time() > end_time:
                extra_cycles = extra_cycles - 1
        raise RuntimeError("instance not responding")
    def _change_state(self,want,when,issue):
        i = self.get_ec2_instance()
        if i.state == want:
            if want != 'running':
                return i
            # 'i' might be holding a cached instance, so maybe the underlying
            # machine has shut down, and we just don't know about it yet.
            # Since the whole point of cached instances is to avoid hitting
            # the AWS API at too high a rate, validate that we're running
            # using SSH instead
            try:
                self.wait_for_ssh()
                return i
            except RuntimeError:
                i = self.get_ec2_instance(force_refresh=True)
        polls_to_wait = self.max_wait_secs / self.poll_interval_secs
        while polls_to_wait > 0:
            # print(i.state,want,when,issue,polls_to_wait)
            if i.state == when:
                if self.verbose:
                    print('sending',issue,'to',self.name)
                getattr(i,issue)()
            time.sleep(self.poll_interval_secs)
            i = self.get_ec2_instance(force_refresh=True)
            if i.state == want:
                if want == 'running':
                    self.wait_for_ssh(self.max_ssh_wait_secs_after_start)
                return i
            polls_to_wait -= 1
        raise RuntimeError("%s not %s" % (self.name,want))
    def do_stop(self):
        self._change_state(want='stopped',when='running',issue='stop')
    def do_create(self):
        try:
            self.get_ec2_instance().state # throws if instance doesn't exist
            print(self.name,"instance already exists")
            return
        except RuntimeError:
            pass
        if self.verbose:
            print("creating",self.name)
        from dtk.alert import slack_send
        slack_send(f'creating {self.name} from {os.uname()[1]}')
        kwargs = self.build_spec.get_create_parameters(self)
        ctrl = self._get_vpc_control()
        ctrl.set_vpc_from_subnet(self.build_spec.subnet_id)
        # the following is done outside get_create_parameters to avoid
        # side effects during unit tests (prep_sg_id_list alters AWS config)
        kwargs['SecurityGroupIds'] = ctrl.prep_sg_id_list(self.name)
        rsp=ctrl.ec2_client.run_instances(MinCount=1,MaxCount=1,**kwargs)
        instance_list = rsp['Instances']
        assert len(instance_list) == 1
        b3inst = ctrl.ec2.Instance(instance_list[0]['InstanceId'])
        b3inst.create_tags(Tags=[ctrl.make_tag('Name',self.name)])
        # tag all volumes on instance
        while not b3inst.block_device_mappings:
            print(self.name,'waiting for device mappings')
            time.sleep(1)
            b3inst.reload()
        prefix = '/dev/'
        # XXX Note that this code builds machines with volume name tags
        # XXX like myname, myname-sdf,... but the volume upgrade code
        # XXX expects names like myname-root, myname-extra. The system
        # XXX here is probably better, or at least more extendable (but
        # XXX maybe keep the sda1 suffix for the root volume).
        # XXX Reconcile this someday.
        for d in b3inst.block_device_mappings:
            dev = d['DeviceName']
            assert dev.startswith(prefix)
            stem = dev[len(prefix):]
            if stem == 'sda1':
                vol_name = self.name
            else:
                vol_name = self.name+'-'+stem
            vol_id = d['Ebs']['VolumeId']
            print('tagging',vol_id,'with name',vol_name)
            b3vol = ctrl.ec2.Volume(vol_id)
            b3vol.create_tags(Tags=[ctrl.make_tag('Name',vol_name)])
    def do_start(self):
        old_state = self.get_ec2_instance().state
        self._change_state(want='running',when='stopped',issue='start')
        if old_state != 'running':
            # when you start your own worker machine, check that the number
            # of worker cores in the local database is correct
            from path_helper import PathHelper
            if self.name == PathHelper.cfg('worker_machine_name'):
                from reserve import ResourceManager,default_totals
                try:
                    rm = ResourceManager()
                    worker_core_idx = 1
                    current = rm.status(0)
                    got = current[worker_core_idx]
                    want = default_totals()[worker_core_idx]
                    if got != want:
                        print('changing worker cores from',got,'to',want)
                        current[worker_core_idx] = want
                        rm.set_totals(current)
                except Exception as ex:
                    print('exception setting RM:',ex)
    def do_get_pub_key(self):
        path='~/.ssh/id_rsa'
        path_pub=path + '.pub'
        self.run_remote_cmd('[ -s %s ] || ssh-keygen -t rsa -N "" -f %s'
                        % (path_pub,path)
                        , venv=None)
        self.copy_from(path_pub,self.local_key_buffer)
    def do_authorize_pub_key(self):
        with open(self.local_key_buffer,"r") as f:
            i = self.do_start()
            ssh_cmd = ['ssh'] + self._ssh_opts()
            ssh_cmd += [ self._ssh_dest(), "cat - >> ~/.ssh/authorized_keys" ]
            self.run_popen(ssh_cmd,stdin=f)
    def change_instance_type(self,new_type):
        i = self.get_ec2_instance()
        if i.instance_type == new_type:
            return
        self.do_stop()
        i = self.get_ec2_instance(force_refresh=True)
        i.modify_attribute(InstanceType={'Value':new_type})
        i = self.get_ec2_instance(force_refresh=True)
    def _get_near_match(self,prefix,suffix):
        for idx,v in enumerate(self.upgrade_path):
            if v.startswith(prefix) and v.endswith(suffix):
                return idx
    def get_adjacent_types(self):
        i = self.get_ec2_instance()
        try:
            idx = self.upgrade_path.index(i.instance_type)
        except ValueError:
            group,size = i.instance_type.split('.')
            idx = self._get_near_match(group[0],'.'+size)
            if idx is None:
                return (None,None)
        try:
            upgrade_type = self.upgrade_path[idx+1]
        except IndexError:
            upgrade_type = None
        try:
            downgrade_type = self.upgrade_path[idx-1] if idx else None
        except IndexError:
            downgrade_type = None
        return (downgrade_type,upgrade_type)
    def do_upgrade(self):
        downgrade_type,upgrade_type = self.get_adjacent_types()
        self.change_instance_type(upgrade_type)
    def do_downgrade(self):
        downgrade_type,upgrade_type = self.get_adjacent_types()
        self.change_instance_type(downgrade_type)
    def get_vols(self):
        return self.get_ec2_instance().get_vols()
    def do_listdev(self):
        vols=self.get_vols()
        for v in vols:
            print(v.mount_key)
    def make_snapshots(self,**name_map):
        '''Create a snapshot for each volume on this machine.

        The snapshot name for each volume is specified by the passed-in
        name_map, e.g.:
        make_snapshot(sdf='root_snap_name',sda1='extra_snap_name')

        If a volume is missing from the name_map, no snapshot is created
        for that volume.
        '''
        # XXX To provide a warning here about whether the platform instance
        # XXX has jobs running:
        # XXX - do a wait_for_ssh call in a try block, similar to the one
        # XXX   in _change_state. If this doesn't succeed, the instance
        # XXX   isn't running anyway.
        # XXX - do a check_remote_cmd:
        # XXX   'ls `2xar/twoxar-demo/web1/path_helper.py pidfiles`|wc -l'
        # XXX   with hold_output=True
        # XXX - then extract the pidfile count with int(self.stdout.strip())
        # XXX   and abort with a warning if non-zero
        self.do_stop()
        vols=self.get_vols()
        initiated=0
        for v in vols:
            dev = v.mount_key
            try:
                snap_name = name_map[dev]
                if self.verbose:
                    print('creating snapshot',snap_name,'for dev',dev)
                v.create_snapshot(snap_name)
                initiated += 1
            except KeyError:
                continue
        if self.verbose:
            print(initiated,'snapshots initiated')
    def num_cores(self):
        i = self.get_ec2_instance()
        return self.instance_properties[i.instance_type][0]
    instance_properties = {
        # {instance type:(cores,mem,$/hr)}
        # this comes from the "On-Demand Instance Prices" table
        # at http://aws.amazon.com/ec2/pricing/
        't2.micro':(1, 1, 0.013),
        't2.small':(1, 2, 0.026),
        't2.medium':(2, 4, 0.052),
        't2.large':(2, 8, 0.052),
        'm3.large':(2, 7.5, 0.133),
        'm4.large':(2, 8, 0.108),
        'm4.2xlarge':(8, 32, 0.504),
        'm4.4xlarge':(16, 64, 1.008),
        'm4.10xlarge':(40, 160, 2.394),
        'm4.16xlarge':(64, 256, 3.83),
        'm5.large':(2, 8, 0.096),
        'm5.2xlarge':(8, 32, 0.384),
        'm5.4xlarge':(16, 64, 0.768),
        'm5.8xlarge':(32, 128, 1.536),
        'm5.12xlarge':(48, 192, 2.304),
        'm5.16xlarge':(64, 256, 3.072),
        'm5.24xlarge':(96, 384, 4.608),
        'c4.2xlarge':(8, 15, 0.441),
        'c4.4xlarge':(16, 30, 0.882),
        'c3.4xlarge':(16, 30, 0.84),
        'r3.xlarge':(4, 30.5, 0.35),
        'r3.2xlarge':(8, 61, 0.70),
        'r5.xlarge':(4, 32, 0.252),
        'r5.2xlarge':(8, 64, 0.504),
        'i2.2xlarge':(8, 61, 1.705),
        # GPU options:
        'p2.xlarge':(4, 61, 0.9),
        # from get_instance_info.py --props "c5a m6a r6i"
        'm6a.large':(2, 8, 0.0864),
        'c5a.xlarge':(4, 8, 0.154),
        'r6i.large':(2, 16, 0.126),
        'm6a.xlarge':(4, 16, 0.1728),
        'c5a.2xlarge':(8, 16, 0.308),
        'r6i.xlarge':(4, 32, 0.252),
        'm6a.2xlarge':(8, 32, 0.3456),
        'c5a.4xlarge':(16, 32, 0.616),
        'r6i.2xlarge':(8, 64, 0.504),
        'm6a.4xlarge':(16, 64, 0.6912),
        'c5a.8xlarge':(32, 64, 1.232),
        'c5a.12xlarge':(48, 96, 1.848),
        'r6i.4xlarge':(16, 128, 1.008),
        'm6a.8xlarge':(32, 128, 1.3824),
        'c5a.16xlarge':(64, 128, 2.464),
        'm6a.12xlarge':(48, 192, 2.0736),
        'r6i.8xlarge':(32, 256, 2.016),
        'm6a.16xlarge':(64, 256, 2.7648),
        'r6i.12xlarge':(48, 384, 3.024),
        'r6i.16xlarge':(64, 512, 4.032),
        }
    upgrade_path=[
            'm6a.large',
            'r6i.xlarge',
            'm6a.2xlarge',
            'r6i.2xlarge',
            'c5a.4xlarge',
            'm6a.4xlarge',
            'm6a.8xlarge',
            'm6a.12xlarge',
            'm6a.16xlarge',
            ]

# As of PLAT-2950, machine creation and the attributes that are only
# meaningful for creation have been factored out of the Machine class
# and moved here. The idea is that, rather than having a set of invisible
# defaults, and specifying overrides for each machine, we have a set of
# templates which are each complete specifications, and we associate
# a machine with each.
#
# Two creation-related parameters are left in the Machine class:
# - instance_type, because it varies more often than other creation parameters,
#   and would thus lead to a proliferation of templates
# - role, as that is used on an ongoing basis when setting up authentication
#   to perform AWS control operations
#
# Only 'role', 'has_buttons', and (if not defaulted) 'region' need to be
# specified for Machine instances on an ongoing basis.
#
# For creation, 'instance_type' and 'template' must also be specified, but
# they can be removed after instance creation, or left for archival
# purposes. Since we often adjust instance parameters dynamically, no
# assumptions should be made about those parameters reflecting the current
# state of the instance (and the code contains no such assumptions).

class MachineSpec:
    extra_disks={}
    @classmethod
    def get_create_parameters(cls,mch):
        ctrl = mch._get_vpc_control()
        # if the following doesn't fail, it indicates that the subnet_id
        # and region match
        ctrl.set_vpc_from_subnet(cls.subnet_id)
        # ssh_createkey is the initial key used to create instances;
        # this is only used for an initial login to the instance, prior
        # to replacing the authorized_keys file with the output of
        # scripts/authkeys.py; it can be re-created periodically, but
        # isn't too important as it doesn't allow access to any active
        # instance
        ssh_createkey = 'twoxar-create-'+mch.region.replace('-','')
        kwargs = {'KeyName':ssh_createkey
                ,'SubnetId':cls.subnet_id
                ,'InstanceType':mch.default_type
                ,'Placement':{'AvailabilityZone':ctrl.default_az}
                }
        if mch.role:
            kwargs['IamInstanceProfile']={
                    'Name':mch.role,
                    }
        ebs_root_size = cls.ebs_root_size
        if cls.extra_disks and not ebs_root_size:
            ebs_root_size = 8
        if ebs_root_size:
            l=[]
            kwargs['BlockDeviceMappings']=l
            def add_device(dev,size):
                l.append({
                        'DeviceName':dev,
                        'Ebs':{
                                'VolumeSize':size,
                                'DeleteOnTermination':False,
                                },
                        })
            add_device('/dev/sda1',ebs_root_size)
            for dev in cls.extra_disks:
                add_device('/dev/'+dev,cls.extra_disks[dev])
        from dtk.aws_api import Ec2Image
        imgs = Ec2Image.get_image_list(cls.ubuntu_release,mch.region)
        # the name sort also puts the images in chronological order,
        # oldest to newest
        imgs.sort(key=lambda x: x.name)
        kwargs['ImageId'] = imgs[-1].id
        return kwargs

class SeleniumMachineSpec(MachineSpec):
    ubuntu_release='20.04'
    ebs_root_size=30
    subnet_id='subnet-649c5301'

class AriaWebMachineSpec(MachineSpec):
    ubuntu_release='20.04'
    ebs_root_size=30
    subnet_id='subnet-06f050fa39d63a0ac'

class Ubuntu16MachineSpec(MachineSpec):
    # This is meant to be a generic spec for dev machines, etc.
    # Variations with larger drives can be created for special-purpose
    # and production machines.
    ubuntu_release='16.04'
    ebs_root_size=50
    subnet_id = 'subnet-a314c0c6'
    extra_disks={
            'sdf':500,
            }

class Ubuntu20MachineSpec(MachineSpec):
    # This is meant to be a generic spec for dev machines, etc.
    # Variations with larger drives can be created for special-purpose
    # and production machines.
    ubuntu_release='20.04'
    ebs_root_size=50
    subnet_id = 'subnet-a314c0c6'
    extra_disks={
            'sdf':500,
            }

class Ubuntu20TemplateMachineSpec(MachineSpec):
    ubuntu_release='20.04'
    ebs_root_size=50
    subnet_id = 'subnet-a314c0c6'

class Lts18MachineSpec(MachineSpec):
    ubuntu_release='18.04'
    ebs_root_size=30
    subnet_id = 'subnet-a314c0c6'
    # extra disk grafted manually during upgrade

class Lts20MachineSpec(MachineSpec):
    ubuntu_release='20.04'
    ebs_root_size=30
    subnet_id = 'subnet-a314c0c6'
    # extra disk grafted manually during upgrade

Machine('worker'
            ,role='worker'
            ,has_buttons=True
            )
Machine('worker-test'
            ,role='worker'
            ,has_buttons=True
            )
Machine('worker-qa'
            ,role='worker'
            ,has_buttons=True
            )
Machine('platform'
            ,role='platform'
            # no buttons to prevent accidental shutdown
            )
Machine('lts'
            # no role -- no aws operations
            # no buttons to prevent accidental shutdown
            ,instance_type='t2.micro'
            ,template=Lts20MachineSpec
            )
Machine('dev01'
            ,role='platform'
            ,has_buttons=True
            )
Machine('dev08'
            ,role='platform'
            ,has_buttons=True
            ,template=Ubuntu16MachineSpec
            )
Machine('qa01'
            ,role='platform'
            ,has_buttons=True
            )
Machine('etl'
            ,role='platform'
            ,has_buttons=True
            )

Machine('selenium'
            ,role=None
            ,has_buttons=False
            ,instance_type='t3a.medium'
            ,region='us-west-1'
            ,template=SeleniumMachineSpec
            )

Machine('ariaweb1'
        ,role=None
        ,has_buttons=False
        ,instance_type='t3a.medium'
        ,region='us-west-1'
        ,template=AriaWebMachineSpec
)

# XXX Eventually, we'll want to add the ability to swap in
# XXX a volume created from a snapshot, to restore a default volume, etc.  A
# XXX better structure might be one which could combine multiple operations
# XXX into a single stop/start cycle. This might look like:
# XXX - multiple commands on command line
# XXX - if one is start or stop, that indicates the desired final state
# XXX - if any operations require a stop, that is done first, then all the
# XXX   stop-only operations, then (if requested) the start
# XXX   - a start at the end might be the default if no start or stop was
# XXX     specified, but the machine was stopped because of other commands

# XXX Also, to better support machine cloning, it would be nice to have
# XXX commands for:
# XXX - given an instance, create snapshots of all its volumes
# XXX - given a set of snapshots, create matching volumes
# XXX - given a set of volumes, replace the volumes on an instance
# XXX - swap an EIP from one instance to another (and update instance names,
# XXX   volume names, etc.)
# XXX The key to getting this to work correctly is coming up with a naming
# XXX convention for snapshots and volumes that successfully encodes:
# XXX - what instance they belong to
# XXX - what device name they should be attached with
# XXX - what set of volumes/snapshots they belong to
# XXX One possibility is:
# XXX - snapshot names are <instance>-<group>-<devname>
# XXX   (e.g. dev01-premigration-sda1)
# XXX   group name is supplied by user making the snapshot; it could default
# XXX   to a date or serial number or something
# XXX - volume names are <newinstance>-<group>-<devname>
# XXX - a volswap operation on an instance substitutes one group for another;
# XXX   (maybe there's a command to make existing volumes on an instance
# XXX   conform to this naming convention; default groupname could be 'orig')
# XXX
# XXX Another issue I had after spinning the disks up on a new machine (and,
# XXX the internet implies, after changing hostname) is needing to:
# XXX  rm ~/.config/nw.js-plotly.js/Singleton*

if __name__ == '__main__':
    deploy_snapshot_prefix='deploy'
    import argparse
    parser = argparse.ArgumentParser(description='aws utility')
    parser.add_argument('-m','--machine'
        ,default='worker'
        ,help="("+", ".join(list(Machine.name_index.keys()))
                +") default %(default)s"
        )
    parser.add_argument('-f','--file'
        ,help="full local-side pathname for file transfer"
        )
    parser.add_argument('-r','--rfile'
        ,help="full remote-side pathname for file transfer (default same)"
        )
    parser.add_argument('-c','--cmd'
        ,help="command to run on remote machine"
        )
    parser.add_argument('-s','--sftp-account'
        ,help="account for fw_open/fw_close"
        )
    parser.add_argument('--ubuntu-release'
        ,help="ubuntu version for list_images only"
        )
    parser.add_argument('op'
        ,help="(%s)" % ", ".join([
                'status',
                'full_backup',
                deploy_snapshot_prefix+'###',
                ]+Machine.get_op_list())
        )
    args = parser.parse_args()

    mch = Machine.name_index[args.machine]
    if args.file:
        mch.local_file = args.file
    if args.rfile:
        mch.remote_file = args.rfile
    if args.cmd:
        mch.remote_cmd = args.cmd
    if args.sftp_account:
        mch.sftp_account = args.sftp_account
    if args.ubuntu_release:
        mch.ubuntu_release = args.ubuntu_release
    if args.op == 'status':
        for mch in list(Machine.name_index.values()):
            s = mch.name+":"
            try:
                i = mch.get_ec2_instance()
                s += " "+str(i.state)
                s += " "+str(i.instance_type)
                s += " "+str(i.ip_address)
                s += " "+str(i.image_id)
            except RuntimeError as e:
                s += " "+str(e)
            print(s)
    elif args.op == 'list_images':
        if args.ubuntu_release:
            mch.list_images(args.ubuntu_release)
        else:
            print('please specify --ubuntu-release')
    elif args.op == 'full_backup':
        name_map={}
        for v in mch.get_vols():
            key = v.mount_key
            name = '%s-backup-%s' %(mch.name,key)
            name_map[key] = name
        mch.make_snapshots(**name_map)
    elif args.op.startswith(deploy_snapshot_prefix):
        sprint_id = int(args.op[len(deploy_snapshot_prefix):])
        if mch.name == 'platform':
            mch.make_snapshots(
                    sda1='pre-sprint%d-root'%sprint_id,
                    sdf='pre-sprint%d-extra'%sprint_id,
                    )
        elif mch.name == 'worker':
            mch.make_snapshots(
                    sda1='pre-sprint%d-worker-root'%sprint_id,
                    sdf='pre-sprint%d-worker-extra'%sprint_id,
                    )
        else:
            raise NotImplementedError
    elif "do_"+args.op in Machine.__dict__:
        getattr(mch,"do_"+args.op)()
    else:
        raise RuntimeError("unknown op: '"+args.op+"'")
