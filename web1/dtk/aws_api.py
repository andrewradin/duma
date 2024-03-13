from dtk.lazy_loader import LazyLoader

class InstanceMetadata(LazyLoader):
    url_stem = 'http://169.254.169.254/latest/dynamic/'
    def _identity_document_loader(self):
        # When running on an EC2 instance, this will return a dict holding
        # some basic identity information for the instance. When not on an
        # EC2 instance, it returns None. We set a 100ms timeout so that it
        # returns quickly in either case, making it a good way to test for
        # whether we're running in EC2 (typical response time is ~1ms).
        try:
            import requests
            resp = requests.get(
                    self.url_stem+'instance-identity/document',
                    timeout=0.1,
                    )
            if resp.status_code == 200:
                import json
                return json.loads(resp.text)
        except IOError:
            pass
        return None

class RoleAccessToken(LazyLoader):
    _kwargs=['role']
    key_map = {
            'Token':'AWS_SESSION_TOKEN',
            'AccessKeyId':'AWS_ACCESS_KEY_ID',
            'SecretAccessKey':'AWS_SECRET_ACCESS_KEY',
            }
    base_url='http://169.254.169.254/latest/'
    def _keys_loader(self):
        return {
                self.key_map[k]:v
                for k,v in self.response_data.items()
                if k in self.key_map
                }
    def _expiration_loader(self):
        return self.response_data['Expiration']
    def _response_data_loader(self):
        return self.rsp2.json()
    def _rsp2_loader(self):
        import requests
        return requests.get(
                self.base_url+'meta-data/iam/security-credentials/'+self.role,
                headers={'X-aws-ec2-metadata-token':self.rsp1.text},
                )
    def _rsp1_loader(self):
        import requests
        return requests.put(
                self.base_url+'api/token',
                headers={'X-aws-ec2-metadata-token-ttl-seconds':'300'},
                )

class AwsBoto3Base(LazyLoader):
    profile_name='twoxar'
    region_name='us-west-2'
    default_az='us-west-2a'
    def _instance_metadata_loader(self):
        return InstanceMetadata()
    def _in_aws_loader(self):
        '''return True if code is running in an AWS instance.'''
        return bool(self.instance_metadata.identity_document)
    def _session_loader(self):
        import boto3
        kwargs=dict(region_name=self.region_name)
        if not self.in_aws:
            kwargs['profile_name'] = self.profile_name
        return boto3.session.Session(**kwargs)
    # In boto3, each AWS service has a corresponding high-level ServiceResource
    # object, and a low-level client object. The client object should only be
    # used for functionality not implemented in the service resource. By
    # convention, we name the high-level object with the service name, and the
    # low-level object with the service name followed by _client
    def _ec2_loader(self):
        return self.session.resource('ec2')
    def _ec2_client_loader(self):
        return self.session.client('ec2')
    def _s3_loader(self):
        return self.session.resource('s3')
    def _s3_client_loader(self):
        return self.session.client('s3')
    @classmethod
    def get_tag(cls,key,l):
        for d in l:
            if key == d['Key']:
                return d['Value']
    @staticmethod
    def make_tag(key,value):
        return dict(Key=key,Value=value)

class Ec2Image:
    '''Wrapper for a boto3 ami/image.'''
    def __init__(self,boto3_image):
        self._boto3_image = boto3_image
    @property
    def id(self):
        return self._boto3_image['ImageId']
    @property
    def name(self):
        prefix_len = 1 + len(self._boto3_image['OwnerId'])
        return self._boto3_image['ImageLocation'][prefix_len:]
    @classmethod
    def get_image_list(cls,ubuntu_release,region):
        ami_publisher = '099720109477'
        fmt = 'ubuntu/images/hvm-ssd/ubuntu-*-%s-amd64-server-*'
        ami_name_pattern = fmt % ubuntu_release
        b3b=AwsBoto3Base()
        b3b.region_name = region
        result=b3b.ec2_client.describe_images(
                Filters=[{'Name':'name','Values':[ami_name_pattern]}],
                Owners=[ami_publisher],
                )
        return [cls(x) for x in result['Images']]

class Ec2Volume:
    '''Wrapper for a boto3 volume.'''
    def __init__(self,boto3_volume):
        self._boto3_volume = boto3_volume
    def create_snapshot(self,name):
        self._boto3_volume.create_snapshot(
                Description=name,
                TagSpecifications=[{
                        'ResourceType':'snapshot',
                        'Tags':[
                                {'Key':'Name','Value':name},
                                ],
                        }],
                )
    @property
    def mount_key(self):
        l = self._boto3_volume.attachments
        assert len(l) == 1
        return l[0]['Device'].replace('/dev/','')

class Ec2Instance(AwsBoto3Base):
    '''An abstract proxy for an EC2 instance.

    This replaces direct use of the instance object from the (now deprecated)
    boto module. As a result, it factors direct dependence on a particular
    AWS interface library out of aws_op.
    '''
    _kwargs=['name']
    _cached_instance = None # initial value without __init__ override
    instance_refresh = 120
    def refresh_instance_data(self):
        self._cached_instance = None
        # this is to pick up state changes, not reconfiguration, so don't
        # clear self.dest_ip
    def _get_boto3_instance(self):
        import time
        if self._cached_instance:
            if time.time() - self._cached_instance_ts < self.instance_refresh:
                return self._cached_instance
        l = list(self.ec2.instances.filter(
                Filters=[{'Name':'tag:Name','Values':[self.name]}]
                ))
        if l:
            if len(l) > 1:
                raise RuntimeError("multiple instance for '"+self.name+"'")
                # the original boto code just ignored multiple instances
                # and returned the first one, but that seems wrong.
            self._cached_instance = l[0]
            self._cached_instance_ts = time.time()
            return self._cached_instance
        raise RuntimeError("no '"+self.name+"' instance found")
    def _dest_ip_loader(self):
        '''Return best ip address to reach this instance from the local machine.

        This will be the public IP address, unless both machines are in the
        same VPC, in which case it will be the private IP address
        '''
        dest = self._get_boto3_instance()
        if self.in_aws:
            id_doc = self.instance_metadata.identity_document
            src = self.ec2.Instance(id_doc['instanceId'])
            if src.vpc_id == dest.vpc_id:
                return dest.private_ip_address
        return dest.public_ip_address
    def start(self):
        self._get_boto3_instance().start()
    def stop(self):
        self._get_boto3_instance().stop()
    def modify_attribute(self,**kwargs):
        self._get_boto3_instance().modify_attribute(**kwargs)
    def get_vols(self):
        return [Ec2Volume(x) for x in self._get_boto3_instance().volumes.all()]
    @property
    def state(self):
        return self._get_boto3_instance().state['Name']
    @property
    def instance_type(self):
        return self._get_boto3_instance().instance_type
    @property
    def image_id(self):
        return self._get_boto3_instance().image_id
    @property
    def ip_address(self):
        return self._get_boto3_instance().public_ip_address
    @property
    def private_ip_address(self):
        return self._get_boto3_instance().private_ip_address

class S3Check(AwsBoto3Base):
    '''S3 Security checker.

    Confirms that S3 bucket ACLs are set up as expected.
    '''

    # Unfortunately there are several layers of overlapping access
    # configuration for S3. My first thought was to check bucket-level
    # ACLs, but these don't protect individual files, so we'd need to
    # check each object's ACL as well if we continued relying on these.
    #
    # Instead, I've switched to setting Public Access Block on each
    # bucket, which allows much quicker verification. The bucket ACL
    # check is left in place mostly because it was already written.
    #
    # FYI, you can access a bucket as a website (if configured) with
    # curl -v http://<bucket>.s3-website-<region>.amazonaws.com
    my_id='9240fcdb9631ac99f3b5ec3b111fc45feec8daa805dd210d7af4d56e1212540f'
    def _bucket_names_loader(self):
        return [x['Name']
                for x in self.s3_client.list_buckets()['Buckets']
                ]
    public_buckets=[
            'twoxar.com',
            '2xar.com',
            'aria.com',
            ]
    def check_access(self):
        for name in self.bucket_names:
            errors = []
            errors += self.check_acl(name)
            if name not in self.public_buckets:
                errors += self.check_blkpub(name)
            if errors:
                print(name)
                for label,detail in errors:
                    print('   ',label+':',detail)
    def blkpub_config(self):
        return {x:True for x in (
                'BlockPublicAcls',
                'IgnorePublicAcls',
                'BlockPublicPolicy',
                'RestrictPublicBuckets',
                )}
    def set_blkpub(self,name):
        self.s3_client.put_public_access_block(
                Bucket=name,
                PublicAccessBlockConfiguration=self.blkpub_config(),
                )
    def check_blkpub(self,name):
        want = self.blkpub_config()
        import botocore
        try:
            rsp = self.s3_client.get_public_access_block(Bucket=name)
        except botocore.exceptions.ClientError:
            return [('BlockPublic','not configured')]
        if rsp['PublicAccessBlockConfiguration'] != want:
            return [('BlockPublic','mis-configured')]
        return []
    @staticmethod
    def grant_desc(d):
        perm = d['Permission']
        grantee = d['Grantee']
        gtype = grantee['Type']
        if gtype == 'CanonicalUser':
            detail = grantee['ID']
        elif gtype == 'Group':
            detail = grantee['URI'].split('/')[-1]
        else:
            detail = 'Unexpected type'
        return (perm,gtype,detail)
    def check_acl(self,name):
        acl=self.s3.BucketAcl(name)
        # set up default ok grants
        ok_grants = set([
                ('READ','CanonicalUser',self.my_id),
                ('WRITE','CanonicalUser',self.my_id),
                ('READ_ACP','CanonicalUser',self.my_id),
                ('WRITE_ACP','CanonicalUser',self.my_id),
                ('FULL_CONTROL','CanonicalUser',self.my_id),
                ])
        # modify ok grants per bucket name
        if name == '2xar-logs':
            ok_grants |= set([
                    ('WRITE','Group','LogDelivery'),
                    ('READ_ACP','Group','LogDelivery'),
                    ])
        # report unexpected grants
        bad_grants = []
        for grant in acl.grants:
            desc = self.grant_desc(grant)
            if desc not in ok_grants:
                bad_grants.append(("unexpected ACL",desc))
        return bad_grants

class Bucket(AwsBoto3Base):
    '''Low-level S3 Bucket interface.

    - takes name and cache path as parameter
    - namelist returns bucket content (no local overrides)
    - get and put transfer between cache_path and S3; filename == s3 key
    '''
    _kwargs=['name','cache_path']
    def namelist(self):
        return [x.key for x in self.aws_bucket.objects.all()]
    def _fn2key(self,filename):
        return filename
    def put_file(self,filename):
        import os
        path = os.path.join(self.cache_path,filename)
        with open(path,'rb') as data:
            self.aws_bucket.upload_fileobj(data,self._fn2key(filename))
    def get_file(self,filename):
        import os
        path = os.path.join(self.cache_path,filename)
        # need to use lower-level interface to force binary mode
        from atomicwrites import AtomicWriter
        with AtomicWriter(path,mode='wb',overwrite=True).open() as data:
            self.aws_bucket.download_fileobj(self._fn2key(filename),data)
    def _aws_bucket_loader(self):
        return self.s3.Bucket(self.name)

class VBucket(Bucket):
    '''Virtual Bucket interface (multiple cache paths share one S3 bucket).

    - takes file_class as parameter; this is used to derive the cache_path
      directory, and a prefix to the bucket key
    - namelist returns bucket content sharing the prefix (no local overrides)
    - get and put transfer between cache_path and S3; s3 key is filename with
      file_class prefix
    '''
    _kwargs=['file_class']
    aws_bucket_name = '2xar-versioned-datasets'
    def namelist(self):
        return [
                x.key[len(self.file_class)+1:]
                for x in self.aws_bucket.objects.filter(
                        Prefix=self.file_class+'/',
                        )
                ]
    def _fn2key(self,filename):
        return self.file_class+'/'+filename
    def _aws_bucket_loader(self):
        return self.s3.Bucket(self.aws_bucket_name)
    def _cache_path_loader(self):
        assert self.file_class
        from path_helper import PathHelper
        import os
        return os.path.join(PathHelper.s3_cache_root,self.file_class)+'/'

class VolumeUpgrade(AwsBoto3Base):
    '''Manage upgrade of AWS instance root volume.

    For OS upgrades, and in possibly other scenarios, it's convenient to
    build a single example instance and then use clones of its root volume
    to replace the original root volumes on other instances.

    The manual preparation steps are:
    - create a new instance with the desired root volume configuration
      (size, OS, etc.) and install Duma
      - this instance should have an authorized_keys file that at least
        allows access from the machine that will run upgrade_root_volume.py
      - this instance doesn't need a second data drive
      - the instance also doesn't need to be in DNS, or have a permanent
        IP address, since the scripts only access a snapshot of the
        instance's root volume, not the instance itself
    - shut the instance down, and snapshot the root volume, giving it the
      name <target_name>-root-template, where target_name identifies the
      upgrade being performed (e.g. "ubuntu1604")

    Then, for each instance to be upgraded:
    - ssh in and run install/disk_layout.py to verify that the instance's
      volume configuration is as expected; fix any issues found
    - from another machine, run:
      scripts/upgrade_root_volume.py <instance_name> <target_name>

    Note that, generally, the instance being upgraded will already have
    an EIP and be registered in DNS, so the upgrade script shouldn't
    have access issues. For special situations, you can usually work
    around issues by creating an entry in .ssh/config on the machine
    running upgrade_root_volume.py.

    For special cases, you can instantiate a VolumeUpgrade object in
    a django shell, and execute step manually. In particular, it's
    useful to do a backup, clone, swap and configure step even when
    the precheck and/or splice steps need to be run manually.

    Worker machines don't exactly fit this model:
    - they don't have an lts directory, but you can create an empty one
      to work around this
    - they typically don't log in to other machines, and so don't need
      an ssh key pair. You can fix this by setting the no_ssh flag.
    '''
    _kwargs=['instance_name','target']
    steps = [
            'initial',
            'precheck',
            'keysave',
            'backup',
            'clone',
            'swap',
            'splice',
            'keyrestore',
            'configure',
            'cleanup',
            ]
    vol_alias='root'
    no_detach_ok=False
    no_ssh=False
    host_key_stash='/mnt2/hostkeys'
    def tag_progress(self,tag_step):
        # write completed step to instance as a checkpoint
        self.instance.create_tags(
                Tags=[self.make_tag(self.progress_tag,tag_step)]
                )
    def run(self):
        tag_step = self.get_tag(self.progress_tag,self.instance.tags)
        if tag_step is None:
            tag_step = self.steps[0]
        for i in range(len(self.steps)-1):
            prev,cur = self.steps[i:i+2]
            if tag_step == prev:
                # execute step
                self.banner(cur)
                func = getattr(self,cur+'_step')
                func()
                tag_step = cur
                self.tag_progress(tag_step)
            else:
                print('skipping '+cur+'; already complete')
        print(
                self.instance_name
                +' successfully upgraded to '
                +self.target
                )
    def reset(self):
        '''Roll back instance state to backup.'''
        self.mch.do_stop()
        for vol_alias in self.name2dev.keys():
            backup_name = self.backup_name(vol_alias)
            vol_name = self.vol_name(vol_alias)
            print('rolling back %s to %s:'%(vol_name,backup_name))
            self.create_vol_from_snapshot(backup_name,vol_name)
            self.swap_volumes(
                    self.name2dev[vol_alias],
                    vol_name+'-before-reset',
                    vol_name,
                    no_detach_ok=True,
                    )
        self.tag_progress('backup')
    ######
    # processing steps
    ######
    def precheck_step(self):
        self.mch.do_start()
        # make sure instance is correctly configured
        from path_helper import PathHelper
        self.remote_cmd(
                ['disk_layout.py','--no-fix'],
                cmd_dir=PathHelper.repos_root+'install',
                )
    def keysave_step(self):
        self.mch.do_start()
        # stash host keys
        self.remote_cmd(
                ['sudo','mkdir','-p',self.host_key_stash],
                activate=False,
                )
        self.remote_cmd(
                ['sudo','sh','-c',
                    # double quotes cause -c to see entire thing as one command
                    '"cp /etc/ssh/ssh_host_* '+self.host_key_stash+'"',
                    ],
                check=False, # ok if no host keys to copy
                activate=False,
                )
    def backup_step(self):
        # shut down and make volume snapshots
        self.mch.do_stop()
        description='{classname} pre-upgrade backup'.format(
                classname=self.__class__.__name__,
                )
        for vol in self.instance.volumes.all():
            assert len(vol.attachments) == 1
            mount=vol.attachments[0]
            assert mount['InstanceId'] == self.instance.id # sanity check
            snapshot_name=self.backup_name(self.dev2name[mount['Device']])
            self.create_snapshot(vol,snapshot_name,description)
    def clone_step(self):
        # find template snapshot
        snapshot_name='{target}-{vol}-template'.format(
                target=self.target,
                vol=self.vol_alias,
                )
        self.create_vol_from_snapshot(snapshot_name,self.new_vol_name)
    def swap_step(self):
        self.mch.do_stop()
        self.swap_volumes(
                self.name2dev[self.vol_alias],
                self.old_vol_name,
                self.new_vol_name,
                no_detach_ok=self.no_detach_ok,
                )
        # swapping in a new root volume swaps in a new ssh fingerprint;
        # clear out the stale one
        self.clear_local_known_host()
    def splice_step(self):
        self.mch.do_start()
        from path_helper import PathHelper
        self.remote_cmd(
                ['disk_layout.py','--force-fix','--splice'],
                cmd_dir=PathHelper.repos_root+'install',
                )
    def keyrestore_step(self):
        self.mch.do_start()
        # restore host keys
        self.remote_cmd(
                ['sudo','sh','-c',
                    # double quotes cause -c to see entire thing as one command
                    '"cp '+self.host_key_stash+'/ssh_host_* /etc/ssh"'
                    ],
                check=False, # ok if no host keys to copy
                activate=False,
                )
        self.clear_local_known_host()
    def configure_step(self):
        self.mch.do_start()
        # set remote hostname
        self.mch.check_remote_cmd(
                'sudo sh -c "echo %s >/etc/hostname"'%self.instance_name,
                )
        self.mch.check_remote_cmd(
                'sudo sh -c "grep %s /etc/hosts || echo 127.0.1.1 %s >>/etc/hosts"'%(self.instance_name,self.instance_name),
                )
        self.mch.check_remote_cmd(
                'sudo hostname %s'%self.instance_name,
                )
        # update ssh key configuration
        import scripts.authkeys as ak
        ak_mch = ak.Machine(self.instance_name)
        # push authorized_keys file to upgraded machine
        print('upgrading authorized_keys on '+self.instance_name)
        ak_mch.do_put_ssh()
        # generate and push new key pair to upgraded machine
        if self.no_ssh:
            return
        print('generating key pair for '+self.instance_name)
        ak_mch.do_new_key_pair()
        # ...and to all machines accessed by upgraded machine
        from aws_op import Machine
        for machine in ak_mch.accessed_by(ak_mch.access_key()):
            if machine in ak_mch.inactive_list:
                # Machines that are normally shut down are problematic
                # to upgrade. We could shut them back down after upgrading,
                # but that causes a problem if they happen to be in use.
                # So, for these machines, just warn that they should be
                # upgraded, and then bypass.
                print('WARNING: authorized_keys should be upgraded on '+machine)
                continue
            print('upgrading authorized_keys on '+machine)
            aws_op_mch = Machine.name_index[machine]
            aws_op_mch.do_start()
            ak_other_mch = ak.Machine(machine)
            ak_other_mch.do_put_ssh()
    def cleanup_step(self):
        old_vol = self.select_by_name(
                self.ec2.volumes,
                self.old_vol_name,
                )
        assert old_vol.state == 'available'
        old_vol.delete()
        new_vol = self.select_by_name(
                self.ec2.volumes,
                self.new_vol_name,
                )
        base_name = self.vol_name(self.vol_alias)
        new_vol.create_tags(Tags=[self.make_tag('Name',base_name)])
    ######
    # utilities
    ######
    def clear_local_known_host(self):
        import subprocess
        for key in (self.mch.name, self.mch.get_ec2_instance().dest_ip):
            subprocess.run(['ssh-keygen','-R',key])
    def backup_name(self,vol_alias):
        return '{instance}-{vol_alias}-pre-{target}'.format(
                    instance=self.instance_name,
                    vol_alias=vol_alias,
                    target=self.target,
                    )
    def vol_name(self,vol_alias):
        return '{instance}-{vol}'.format(
                instance=self.instance_name,
                vol=vol_alias,
                )
    def create_vol_from_snapshot(self,snapshot_name,new_vol_name):
        snap = self.select_by_name(self.ec2.snapshots, snapshot_name)
        vol = self.ec2.create_volume(
                AvailabilityZone=self.default_az,
                SnapshotId=snap.id,
                VolumeType='gp2',
                )
        vol.create_tags(Tags=[self.make_tag('Name',new_vol_name)])
        self.wait_for_volume(vol.id,'available','create')
    def swap_volumes(self,dev,substitute_name,new_vol_name,no_detach_ok=False):
        # get new volume
        new_vol = self.select_by_name(self.ec2.volumes,new_vol_name)
        # get old volume
        old_vol = None
        for vol in self.instance.volumes.all():
            assert len(vol.attachments) == 1
            mount=vol.attachments[0]
            if mount['Device'] == dev:
                old_vol = vol
                break
        # swap
        if old_vol:
            old_vol.create_tags(Tags=[self.make_tag('Name',substitute_name)])
            self.instance.detach_volume(
                    Device=dev,
                    VolumeId=old_vol.id,
                    )
            self.wait_for_volume(old_vol.id,'available','detach')
        else:
            assert no_detach_ok
        self.instance.attach_volume(
                Device=dev,
                VolumeId=new_vol.id,
                )
        self.wait_for_volume(new_vol.id,'in-use','attach')
    def wait_for_volume(self,vol_id,want_state,label):
        max_polls = 10
        poll_delay = 5
        initial_delay = 2
        import time
        for i in range(max_polls):
            time.sleep(poll_delay if i else initial_delay)
            rsp = self.ec2_client.describe_volumes(VolumeIds=[vol_id])
            vol_info = rsp['Volumes'][0]
            # the following is useful for debugging:
            # print(vol_info['State'],vol_info['Attachments'])
            if vol_info['State'] == 'available':
                break
            if i == 0:
                print('waiting for '+label+'...')
    def banner(self,stepname):
        border=40*'#'
        print(border)
        print(border[0]+' '+stepname)
        print(border)
    def create_snapshot(self,volume,name,description):
        s = volume.create_snapshot(Description=description)
        s.create_tags(Tags=[self.make_tag('Name',name)])
        return s
    def remote_cmd(self,parts,cmd_dir=None,activate=True,check=True):
        parts=list(parts) # local copy can be modified
        # XXX it's nice that this translates the command path; OTOH
        # XXX some of the parameters might need translation also, and the
        # XXX inconsistency might lead to errors
        import os
        if cmd_dir:
            parts[0] = self.mch.get_remote_path(
                    os.path.join(cmd_dir,parts[0])
                    )
        if activate:
            parts.insert(0,self.activate_cmd+' &&')
        exit_code = self.mch.run_remote_cmd(' '.join(parts), venv=None)
        if exit_code and check:
            raise RuntimeError("got bad exit code %d" % exit_code)
        return exit_code
    def filter1(self,collection,filters):
        result=list(collection.filter(Filters=filters))
        assert len(result) == 1
        return result[0]
    def select_by_name(self,collection,name):
        return self.filter1(
                collection,
                [{'Name':'tag:Name','Values':[name]}],
                )
    def _instance_loader(self):
        return self.select_by_name(
                self.ec2.instances,
                self.instance_name,
                )
    def _mch_loader(self):
        from aws_op import Machine
        return Machine.name_index[self.instance_name]
    def _progress_tag_loader(self):
        return self.target+'_progress'
    def _activate_cmd_loader(self):
        from path_helper import PathHelper
        return 'source '+self.mch.get_remote_path(
                PathHelper.venv+'bin/activate py3web1'
                )
    def _vol_aliases_loader(self):
        from dtk.data import MultiMap
        return MultiMap([
                ('root','/dev/sda1'),
                ('extra','/dev/sdf'),
                ])
    def _dev2name_loader(self):
        from dtk.data import MultiMap
        return dict(MultiMap.flatten(self.vol_aliases.rev_map()))
    def _name2dev_loader(self):
        from dtk.data import MultiMap
        return dict(MultiMap.flatten(self.vol_aliases.fwd_map()))
    def _new_vol_name_loader(self):
        return self.vol_name(self.vol_alias)+'-new'
    def _old_vol_name_loader(self):
        return self.vol_name(self.vol_alias)+'-old'

class FirewallConfig:
    '''Represents an abstract firewall configuration for an EC2 instance.

    The purpose of this class is to provide a straightforward way to
    configure the desired acces

    EC2 instances supporting the same access pattern can share the same
    filewall config, even if they're in different VPCs. The firewall
    config will translate into a series of VPC-specific security groups.
    '''
    _name_index={}
    @classmethod
    def by_name(cls,instance_name):
        try:
            return cls._name_index[instance_name]
        except KeyError:
            return cls._name_index['default']
    def _special_ip_func(self,name):
        try:
            return getattr(self,f'_{name}_ip_func')
        except AttributeError:
            return None
    def __init__(self,instance_name,copy=[],ssh=[],allow=[]):
        self.name = instance_name
        self._name_index[instance_name] = self
        self.rules = []
        self._ssh_names = set(ssh)
        for name in copy:
            src = self._name_index[name]
            for rule in src.rules:
                self.rules.append(tuple(rule))
            self._ssh_names |= src._ssh_names
        from dtk.known_ips import KnownIps
        for name in ssh:
            if KnownIps.ssh_ips(name):
                self.rules.append(('tcp',(22,22),name))
        for spec in allow:
            if isinstance(spec[0],int):
                port_range = (spec[0],spec[0])
            else:
                port_range = spec[0]
            assert (port_range[0] <= port_range[1])
            self.rules.append(('tcp',port_range,spec[1]))
    # the _ip_func methods implement special CIDR blocks not extracted
    # from KnownIps
    def _any_internal_ip_func(self,vpc):
        return [vpc.vpc.cidr_block]
    def _any_ip_func(self,vpc):
        return ['0.0.0.0/0']
    def get_sg_name(self,vpc):
        return f'{vpc.vpc_designator}-{self.name}-sg'
    def get_expected_ruleset(self,vpc):
        result = RuleSet()
        from dtk.known_ips import KnownIps
        for prot,port_range,ip_spec in self.rules:
            func = self._special_ip_func(ip_spec)
            if func:
                cidr_list = func(vpc)
            else:
                cidr_list = [x+'/32' for x in KnownIps.ssh_ips(ip_spec)]
            result.add(prot,port_range,cidr_list)
        return result
    def get_accessors(self):
        return set([x.split('.')[0] for x in self._ssh_names])
    def get_keydb_entries(self,keydb):
        for accessor in sorted(self._ssh_names):
            if '.' in accessor:
                # this is a fully-qualified name; return only the matching
                # key (if any)
                accessor,qualifier = accessor.split('.')
            else:
                # return all key entries for this accessor
                qualifier = None
            for which,key in sorted(keydb.by_user[accessor]):
                if qualifier is None or qualifier == which:
                    yield (accessor,which,key)

class RuleSet:
    '''Represents a set of access rules.

    These are based on AWS SG configuration, but only support the subset
    of SGs defined by CIDR blocks. RuleSets can be build either from
    existing AWS SGs, or from a desired FirewallConfig, and then compared
    to verify that instance access is as expected.

    Although our current approach uses only one SG per instance, this class
    supports more than one. This was useful during testing and migration
    from the previous approach, and may be useful again someday.
    '''
    def __init__(self):
        self.rules = {}
        self.errors = set()
    def add(self,prot,port_range,cidr_list):
        s = self.rules.setdefault((prot,port_range),set())
        s |= set(cidr_list)
    def add_from_sg(self,sg):
        # AWS creates a default SG for each VPC. This is named 'default'
        # and although we don't use it, it can't be deleted.
        # It allows connections from other instances in the same SG, by
        # specifying a UserIdGroupPair rather than a list of CIDRs.
        # None of this code handles this configuration. It currently
        # slips past show_vpc (which iterates over CIDRs on output)
        # and sg_check (which notes but doesn't inspect unused SGs).
        #
        # As extra protection here, in case one of these SGs or another
        # similar SG gets assigned to an instance, we look for unexpected
        # keys in the AWS rule configuration, and report their presence
        # as an error string that will show up in repr comparisons.
        supported_keys = (
                'IpProtocol',
                'FromPort',
                'ToPort',
                'IpRanges',
                )
        for ipp in sg.ip_permissions:
            unknowns = [
                    key
                    for key,val in ipp.items()
                    if val and key not in supported_keys
                    ]
            if unknowns:
                self.errors.add('unexpected format in '+sg.id)
            else:
                self.add(
                        ipp['IpProtocol'],
                        (ipp['FromPort'],ipp['ToPort']),
                        [x['CidrIp'] for x in ipp['IpRanges']],
                        )
    def add_to_sg(self,sg):
        for (prot,port_range),s in self.rules.items():
            for port in range(port_range[0],port_range[1]+1):
                for cidr in s:
                    sg.authorize_ingress(
                            CidrIp=cidr,
                            IpProtocol=prot,
                            FromPort=port,
                            ToPort=port,
                            )
    def __repr__(self):
        result = repr(sorted(
                (prot,port_range,sorted(s))
                for (prot,port_range),s in self.rules.items()
                ))
        if self.errors:
            result += ' '+' '.join(sorted(self.errors))
        return result

# the basis for most machines, and used directly by a typical dev machine
FirewallConfig("default",
        ssh=['carl','acd']
                +['aar.home','aar.laptop'] # aar but not aar.cloud
                #+['dev08'] # let developers do direct ssh to other machines
                ,
        allow=[
                (22,'any_internal'),
                #(22,'office'),
                ],
        )
# ssh list allows all these machines to pull production data
FirewallConfig("lts",
        copy=['default'],
        ssh=['dev01','dev08','etl', 'platform','qa01'],
        )
# allow production web access from anywhere
FirewallConfig("platform",
        copy=['default'],
        ssh=['dev01','dev08'], # allow grab_ge.sh and other direct copies
        allow=[
                (80,'any'),
                (443,'any'),
                ],
        )
# workers allow their UI machines ssh access
FirewallConfig("worker",
        copy=['default'],
        ssh=['platform'],
        )
FirewallConfig("worker-test",
        copy=['default'],
        ssh=['dev01','dev08','etl'],
        )
FirewallConfig("worker-qa",
        copy=['default'],
        ssh=['qa01'],
        )

FirewallConfig("ariaweb1",
    copy=['default'],
    allow=[
            (80,'any'),
            (443,'any'),
            ],
    )

class VpcControl(AwsBoto3Base):
    ##########
    # initialization
    # - vpc_name can be set to allow operating on a particular VPC
    # - if cidr_block is set also, it allows a new VPC to be created
    ##########
    def set_vpc_from_subnet(self,subnet_id):
        sn=self.ec2.Subnet(subnet_id)
        self.vpc=self.ec2.Vpc(sn.vpc_id)
        self.default_az=sn.availability_zone
    ##########
    # basic data fetch
    ##########
    def _vpc_loader(self):
        for x in self.ec2.vpcs.all():
            if self.get_tag('Name',x.tags) == self.vpc_name:
                return x
        raise RuntimeError("No VPC named '%s'"%self.vpc_name)
    def _vpc_designator_loader(self):
        name = self.get_tag('Name',self.vpc.tags)
        return self.get_designator(name)
    def _instances_loader(self):
        return list(self.vpc.instances.all())
    def _security_groups_loader(self):
        l = list(self.vpc.security_groups.all())
        l.sort(key=lambda x:(x.group_name,x.id))
        return l
    def _name2sg_loader(self):
        from dtk.data import MultiMap
        mm = MultiMap(
                (x.group_name,x)
                for x in self.security_groups
                )
        return dict(MultiMap.flatten(
                mm.fwd_map(),
                selector=lambda x:'duplicate SG group names',
                ))
    ##########
    # utilities
    ##########
    @classmethod
    def get_tag(cls,key,l):
        base_result = super(VpcControl,cls).get_tag(key,l)
        if base_result is None:
            return ' No '+key
        return base_result
    @classmethod
    def get_designator(cls,vpc_name):
        '''Return a string designating the purpose of the VPC.

        This will be a single word with no punctuation, so it can be used
        to build multi-part names with a variety of separators.
        '''
        # This is generally the vpc name tag with the suffix _vpc
        # stripped off. If any non-standard designators need to be
        # supported, that will be handled here.
        parts = vpc_name.split('_')
        assert len(parts) == 2
        assert parts[-1] == 'vpc'
        return parts[0]
    @staticmethod
    def ip_permission_to_port(perm):
        keys=('IpProtocol','FromPort','ToPort')
        prot,fr,to = (perm.get(k) for k in keys)
        if prot in ('tcp','udp','icmp'):
            # only these protocols support port-level filtering
            if fr == to:
                return '%d (%s)'%(fr,prot)
            else:
                return '%d-%d (%s)'%(fr,to,prot)
        elif prot == '-1':
            return 'all'
        else:
            return 'all (%s)'%prot
    @staticmethod
    def ip_permission_to_cidrs(perm):
        return [x['CidrIp'] for x in perm['IpRanges']]
    def instance_sg_pairs(self):
        for instance in self.instances:
            inst_name = self.get_tag('Name',instance.tags)
            for sg in instance.security_groups:
                yield (inst_name,sg['GroupId'])
    def instance_subnet_pairs(self):
        for instance in self.instances:
            inst_name = self.get_tag('Name',instance.tags)
            for ni in instance.network_interfaces_attribute:
                yield (inst_name,ni['SubnetId'])
    def get_sg_name(self,base_name):
        return '-'.join([self.vpc_designator,base_name])
    def show_route_tables(self):
        from dtk.text import print_table
        for rt in self.vpc.route_tables.all():
            print('Route Table:',rt.id)
            cols=(
                    'GatewayId',
                    'DestinationCidrBlock',
                    'State',
                    )
            rows = [
                    ['   ']+[x[y] for y in cols]
                    for x in rt.routes_attribute
                    ]
            print_table(rows)
    def show_subnets(self):
        from dtk.data import MultiMap
        i2sn = MultiMap(self.instance_subnet_pairs())
        from dtk.text import print_table
        rows = [
                ['   ',x.id,x.cidr_block,x.availability_zone,x.state]
                for x in self.vpc.subnets.all()
                ]
        if rows:
            print('Subnets:')
            rows2=[]
            d = i2sn.rev_map()
            for row in rows:
                try:
                    instances = sorted(d[row[1]])
                except KeyError:
                    instances = ['NO INSTANCES']
                for instance in instances:
                    rows2.append(row+[instance])
                    row = row[:1]+['']*(len(row)-1)
            print_table(rows2)
        else:
            print('NO SUBNETS')
    def show_security_groups(self):
        from dtk.data import MultiMap
        i2sg = MultiMap(self.instance_sg_pairs())
        from dtk.text import print_table
        for sg in self.security_groups:
            print('Security Group:',sg.id,sg.group_name)
            used_by=i2sg.rev_map().get(sg.id)
            if used_by:
                print('   used by:',', '.join(sorted(used_by)))
            else:
                print('   UNUSED')
            rows = []
            from dtk.known_ips import KnownIps
            def labeled_cidr(cidr):
                parts = cidr.split('/')
                if parts[1] == '32':
                    label = KnownIps.hostname(parts[0])
                    if label != parts[0]:
                        return f'{cidr} ({label})'
                return cidr
            for ipp in sg.ip_permissions:
                port = self.ip_permission_to_port(ipp)
                for cidr in self.ip_permission_to_cidrs(ipp):
                    rows.append(('   ','ingress',port,labeled_cidr(cidr)))
            for ipp in sg.ip_permissions_egress:
                port = self.ip_permission_to_port(ipp)
                for cidr in self.ip_permission_to_cidrs(ipp):
                    rows.append(('   ','egress',port,labeled_cidr(cidr)))
            rows.sort()
            print_table(rows)
    def all_region_vpcs(self):
        '''Yield a VpcControl for each VPC in this VpcControl's region.

        self can be a VpcControl with only region_name specified.
        '''
        for vpc in self.ec2.vpcs.all():
            vpc_ctrl = VpcControl()
            vpc_ctrl.region_name = self.region_name
            vpc_ctrl.vpc_name = self.get_tag('Name',vpc.tags)
            yield vpc_ctrl
    def one_vpc_sg_check(self):
        from termcolor import colored
        used_sg_names = set()
        # note there's no guarantee against duplicate instance names
        i_list = sorted(
                (self.get_tag('Name',i.tags),i)
                for i in self.instances
                )
        for i_name,instance in i_list:
            fc = FirewallConfig.by_name(i_name)
            want_cfg = fc.get_expected_ruleset(self)
            got_cfg = RuleSet()
            i_sg_names = [d['GroupName'] for d in instance.security_groups]
            used_sg_names |= set(i_sg_names)
            for sg in [self.name2sg[x] for x in i_sg_names]:
                got_cfg.add_from_sg(sg)
            show = []
            # compare configurations
            cfg_diff = self.compare(repr(want_cfg),repr(got_cfg))
            if cfg_diff:
                show.append(('config',cfg_diff))
            # compare SG associations
            want_sga = fc.get_sg_name(self)
            got_sga = ' '.join(sorted(i_sg_names))
            sga_diff = self.compare(repr(want_sga),repr(got_sga))
            if sga_diff:
                show.append(('SG associations',sga_diff))
            # output instance-level differences
            if show:
                print(colored(
                        f'{i_name} ({instance.id} {self.vpc_name})',
                        color='red',
                        ))
                for label,detail in show:
                    print('   ',label+':',detail)
        # show unused SGs
        unused = set(x.group_name for x in self.security_groups)-used_sg_names
        # ... but don't complain about unused default SG
        unused = unused - set(['default'])
        if unused:
            print(
                    colored(f'{self.vpc_name} unused SGs:',color='red'),
                    ', '.join(sorted(unused)),
                    )
    def compare(self,v1,v2):
        if v1 == v2:
            return
        from termcolor import colored
        from dtk.text import diff_fmt
        return diff_fmt(
                    v1,
                    v2,
                    del_fmt=lambda x:colored(x,on_color='on_red'),
                    add_fmt=lambda x:colored(x,on_color='on_cyan'),
                    minmatch=5,
                    )
    def configure_sg(self,sg_name,desc,ruleset):
        try:
            sg = self.name2sg[sg_name]
        except KeyError:
            sg = self.ec2.create_security_group(
                    GroupName=sg_name,
                    Description=desc,
                    VpcId=self.vpc.id,
                    )
        else:
            # if we find an existing group, clear its permissions
            if sg.ip_permissions:
                sg.revoke_ingress(IpPermissions=sg.ip_permissions)
        # supply rules
        ruleset.add_to_sg(sg)
    # for testing
    #def do_check1(self):
    #    self.one_vpc_sg_check()
    #def do_rebuild1(self):
    #    self.one_vpc_sg_rebuild()
    def one_vpc_sg_rebuild(self):
        '''Rebuild all SGs in a VPC from FirewallConfig objects.'''
        dry_run = False # for testing
        # collect all used FCs
        used_fcs = set()
        for instance in self.instances:
            i_name = self.get_tag('Name',instance.tags)
            used_fcs.add(FirewallConfig.by_name(i_name))
        # make sure the corresponding SG is up to date
        for fc in used_fcs:
            sg_name = fc.get_sg_name(self)
            print('rebuilding',sg_name)
            if not dry_run:
                self.configure_sg(
                        sg_name,
                        'auto-maintained SG '+fc.name,
                        fc.get_expected_ruleset(self)
                        )
        # make sure any newly-created SGs are in the attributes
        self.clear_sg_caching()
        # make sure instances point to the correct SGs
        for instance in self.instances:
            i_name = self.get_tag('Name',instance.tags)
            fc = FirewallConfig.by_name(i_name)
            sg_name = fc.get_sg_name(self)
            print('setting SG assignment for',i_name)
            if not dry_run:
                instance.modify_attribute(Groups=[self.name2sg[sg_name].id])
    def clear_sg_caching(self):
        for attr in ('security_groups','name2sg'):
            try:
                delattr(self,attr)
            except AttributeError:
                pass
    def prep_sg_id_list(self,i_name):
        # helper routine for instance creation
        fc = FirewallConfig.by_name(i_name)
        sg_name = fc.get_sg_name(self)
        self.configure_sg(
                sg_name,
                'auto-maintained SG '+fc.name,
                fc.get_expected_ruleset(self)
                )
        self.clear_sg_caching()
        return [self.name2sg[sg_name].id]
    ##########
    # command infrastructure
    ##########
    op_prefix='do_'
    @classmethod
    def get_op_list(cls):
        return sorted([
                name[len(cls.op_prefix):]
                for name in cls.__dict__
                if name.startswith(cls.op_prefix)
                ])
    def get_op_function(self,op):
        return getattr(self,self.op_prefix+op)
    ##########
    # commands
    ##########
    def do_list_vpcs(self):
        from dtk.text import print_table
        rows = [
                (self.get_tag('Name',x.tags),x.vpc_id,x.cidr_block)
                for x in self.ec2.vpcs.all()
                ]
        print_table(rows)
    def do_list_instances(self):
        from dtk.text import print_table
        rows = [
                (self.get_tag('Name',x.tags),x.instance_id,x.state['Name'])
                for x in self.instances
                ]
        print_table(rows)
    def do_show_vpc(self):
        # display info
        print('Id:',self.vpc.id)
        print('CIDR Block:',self.vpc.cidr_block)
        print('Internet Gateways:',', '.join(
                x.id for x in self.vpc.internet_gateways.all()
                ))
        self.show_route_tables()
        self.show_subnets()
        self.show_security_groups()
    def do_create_vpc(self):
        # create VPC
        # somewhat based on:
        # https://blog.ipswitch.com/how-to-create-and-configure-an-aws-vpc-with-python
        # validate supplied name before doing anything else
        assert self.get_designator(self.vpc_name)
        try:
            vpc = self.vpc
            print('using existing VPC with CIDR block '+vpc.cidr_block)
        except RuntimeError:
            vpc = self.ec2.create_vpc(CidrBlock=self.cidr_block)
            vpc.create_tags(Tags=[self.make_tag('Name',self.vpc_name)])
            vpc.wait_until_available()
            self.vpc = vpc
        # create internet gateway
        gws = list(vpc.internet_gateways.all())
        if gws:
            print('using existing internet GWs:'+' '.join(x.id for x in gws))
            igw = gws[0]
        else:
            igw = self.ec2.create_internet_gateway()
            vpc.attach_internet_gateway(InternetGatewayId=igw.id)
        # create routing table
        rts = list(vpc.route_tables.all())
        if rts:
            print('using existing Routing Tables:'+' '.join(x.id for x in rts))
            rt = rts[0]
        else:
            rt = vpc.create_route_table()
        rt.create_route(DestinationCidrBlock='0.0.0.0/0', GatewayId=igw.id)
        # create subnets
        sns = list(vpc.subnets.all())
        if sns:
            print('using existing Subnets:'+' '.join(x.id for x in sns))
        else:
            # for now, create one /24 subnet at the bottom of the range,
            # only if the VPC is larger than that
            parts = vpc.cidr_block.split('/')
            assert len(parts) == 2
            assert int(parts[1]) < 24
            sn_cidr = '/'.join([parts[0],'24'])
            subnet = self.ec2.create_subnet(
                    CidrBlock=sn_cidr,
                    VpcId=vpc.id,
                    AvailabilityZone=self.default_az,
                    )
            rt.associate_with_subnet(SubnetId=subnet.id)
            # set subnet so instances get a public IP by default
            # see https://github.com/boto/boto3/issues/276
            # although this is clearly never going to happen
            subnet.meta.client.modify_subnet_attribute(
                    SubnetId=subnet.id,
                    MapPublicIpOnLaunch={"Value": True},
                    )

