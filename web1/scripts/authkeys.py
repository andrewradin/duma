#!/usr/bin/env python3

# Overview:
# - on personal laptops, etc.:
#   - generate a unique key pair; add the public key to KeyDb
#   - place user's unique IAM key pair in .aws/credentials and .s3cfg
# - for twoxar AWS cloud instances:
#   - use new_key_pair command to generate a unique ssh key pair and add
#     the public key to KeyDb
#   - use put_ssh command to push out authorized_keys file
#   - no .aws/credentials file needed -- handled by IAM roles
# - for twoxar cloud instances outside AWS:
#   - generate a unique key pair; add the public key to KeyDb
#   - use put_ssh command to push out authorized_keys file
#   - place external_platform_role's IAM key pair in .aws/credentials and .s3cfg

# AWS key management
# - AWS keys aren't necessary for any functionality on AWS cloud instances;
#   that is now handled through AWS roles.
# - For personal computers only accessed by a single developer, you can
#   manually generate a user-specific key in IAM and place it in
#   .aws/credentials and .s3cfg files
# - In the past, we used to have shared cloud instances outside AWS. These
#   needed keys in .aws/credentials and .s3cfg, but using user-specific
#   IAM keys wasn't appropriate, because multiple developers could see the
#   keys on the instance. Instead, we created an 'external_platform_role'
#   user and used a shared key that was rotated any time a developer left
#   the company. Pushing these keys out was only partially automated, and
#   this mechanism is no longer used, but some of the infrastructure was
#   left as an example in case we need to do this in the future. Specifically:
#   - keys can be stored in the local_settings file on machines needing to
#     run this script, and retrieved with the function get_aws_keys()
#   - code for patching these keys into the .s3cfg file was left in place
#   There never was any code for distributing these keys into the
#   .aws/credentials file, but that would be easy to write. All this should
#   happen in the 'put_aws' command, which is currently unused.
#
# SSH key management
# - all developers have access to ssh private keys on cloud machines, but
#   since an IP whitelist must be satisfied before these keys can be used,
#   there's less reason to regenerate them on every personnel change
# - sshd now logs key usage by default; logs look like:
#   "Accepted publickey for" user "from" ip "port" port "ssh2: RSA" fingerprint
#   We could maybe monitor this...
# - for machines running apache, the id_rsa files need to be in /var/www/.ssh,
#   not /home/www-data
# - this script makes the authorized_keys files read-only; this was an
#   accident, but it's probably a good idea to discourage manual editing

# This functionality isn't currently used, but if we need to distribute
# aws keys again, they can be retrieved from local_settings.py.
# They only need to exist on machines that run this script (probably one
# or two at most). If the keys are lost, new ones can be generated via
# the AWS console, set in local_settings.py, and distributed to the
# instances that need them.
def get_aws_key(stem,half):
    assert half in ('access','secret')
    from path_helper import PathHelper
    return PathHelper.cfg('authkeys_'+stem+'_key')[1 if half=='secret' else 0]

# The key is installed by placing it in the ~/.aws/credentials file.
# After a new key is installed, it can be tested by running a command
# like 'aws_op -m worker-test remote_cmd'.

# The following is a template for an s3cfg file. Values of '???' mark where
# keys can be filled in (but this is currently disabled). Note that s3cmd
# supplies defaults for any values not specified, but requires that the
# file exists.
s3cfg_template='''
[default]
access_key = ???
secret_key = ???
'''

##########
# Key database
# The following class allows access to a file-based database of public
# keys. Keys are checked into git in twoxar-demo/keydb, one key per file.
# This simplifies scripted updates to the database. Each file has a
# two-part name:
#   <user>.<machine> for physical machines
#   <machine>.default for cloud machines (which all share the ubuntu user)
#
# The ssh list in the FirewallConfig class grants access to a machine via
# either the first part of the name (which loads all keys matching that
# first part) or the fully-specified name (which matches only a single key).
# Names where the second part is specified but doesn't match any key are
# ignored (they're presumed to match an IP address in known_ips instead,
# which allows more fine-grained control of machine access where desired).
#
# Because our cloud machines share a single user, their ssh keys are visible
# to all developers, so they should be refreshed periodically with the
# 'new_key_pair' command.
#
# To test connectivity between cloud machines, use aws_op.py remote_cmd;
# trying directly with ssh from the command line tries to use the public
# IP of the target machine, and firewall rules expect in-VPC connections
# to use the private IP
#
# WARNING: on an apache machine, access testing must be done as www-data.
##########
from dtk.lazy_loader import LazyLoader
from path_helper import PathHelper
class KeyDb(LazyLoader):
    path=PathHelper.authkeys_keydb
    def save(self,user,machine,data):
        import os
        fpath = os.path.join(self.path,user+'.'+machine)
        # accept data with or without trailing newline
        open(fpath,'w').write(data.strip()+'\n')
        # force cache reload
        try:
            delattr(self,'by_user')
        except AttributeError:
            pass
    def _by_user_loader(self):
        import os
        result = {}
        for name in os.listdir(self.path):
            user,machine = name.split('.')
            fpath = os.path.join(self.path,name)
            pubkey = open(fpath).read().strip()
            result.setdefault(user,[]).append((machine,pubkey))
        for l in result.values():
            l.sort()
        return result

import subprocess

class Machine(LazyLoader):
    def __init__(self,shortname,**kwargs):
        super(Machine,self).__init__(**kwargs)
        self._shortname = shortname
        self._pemfile = None
        self._keydb = KeyDb()
        self.apache = False # can override after instantiation
    def _aws_mch_loader(self):
        import aws_op
        return aws_op.Machine.name_index.get(self._shortname)
    def _ssh_dest_loader(self):
        if self.aws_mch:
            # delegate to aws_op to determine whether to access this
            # via internal or external IP address
            return self.aws_mch._ssh_dest()
        # not a managed cloud machine; use DNS
        return self.shortname+'.twoxar.com'
    def access_key(self):
        return self._shortname
    def do_chk_ssh(self):
        subprocess.check_call(self.base_ssh_cmd() + [
                'echo','hello',
                ])
    ########
    # ssh key pair generation
    ########
    def make_key_pair(self):
        import os
        tmp_path = '/tmp/authkeys_work'
        stem=os.path.join(tmp_path,'id_rsa')
        from dtk.files import remove_tree_if_present
        remove_tree_if_present(tmp_path)
        os.mkdir(tmp_path)
        subprocess.check_call([
                'ssh-keygen',
                '-N','',
                '-f',stem,
                '-C','ubuntu@'+self.access_key(),
                ])
        private_key=open(stem).read()
        public_key=open(stem+'.pub').read()
        remove_tree_if_present(tmp_path)
        return (private_key,public_key)
    def do_new_key_pair(self):
        # XXX validate that this is a cloud machine?
        # XXX have a way to fetch an existing key pair?
        (private_key,public_key) = self.make_key_pair()
        self._keydb.save(
                self.access_key(),
                'apache' if self.apache else 'default',
                public_key,
                )
        tmpfile='tmpfile'
        if self.apache:
            store_path='/var/www/.ssh'
            chown='www-data'
        else:
            store_path='.ssh'
            chown=None
        import os
        for name,data,mode in [
                ('id_rsa',private_key,'600'),
                ('id_rsa.pub',public_key,'644'),
                ]:
            self._put_to_file(tmpfile,data)
            dest_file = os.path.join(store_path,name)
            subprocess.check_call(self.base_ssh_cmd() + [
                    'sudo','mv',tmpfile,dest_file,
                    ])
            if chown:
                subprocess.check_call(self.base_ssh_cmd() + [
                        'sudo','chown',chown,dest_file,
                        ])
            subprocess.check_call(self.base_ssh_cmd() + [
                    'sudo','chmod',mode,dest_file,
                    ])
    ########
    # authorized_keys management
    ########
    def my_key_entries(self):
        from dtk.aws_api import FirewallConfig
        fc = FirewallConfig.by_name(self._shortname)
        return list(fc.get_keydb_entries(self._keydb))
    def authorized_keys(self):
        return '\n'.join([
                key
                for user,which,key in self.my_key_entries()
                ])+'\n'
    def do_show_access(self):
        if self._shortname == '-':
            # show all ssh access
            for user in sorted(self._keydb.by_user.keys()):
                for which,key in sorted(self._keydb.by_user[user]):
                    print(user,which,key.split(' ')[2:])
                print('  can access:',self.accessed_by(user))
                print()
        else:
            print(self.accessed_by(self.access_key()))
    @classmethod
    def all_machine_names(cls):
        import aws_op
        return sorted(aws_op.Machine.name_index.keys())
    @classmethod
    def accessed_by(cls,access_key):
        from dtk.aws_api import FirewallConfig
        return sorted([
                name
                for name in cls.all_machine_names()
                if access_key in FirewallConfig.by_name(name).get_accessors()
                ])
    def do_dump_ssh(self):
        for user in sorted(self._keydb.by_user.keys()):
            for which,key in self._keydb.by_user[user]:
                print(user,which,key)
    def do_list_ssh(self):
        for user,which,key in self.my_key_entries():
            print(user,which,key.split(' ')[2:])
    def base_ssh_cmd(self):
        l = ['ssh','-o','StrictHostKeyChecking=no','-l','ubuntu']
        if self._pemfile:
            l += ['-i',self._pemfile]
        return l+[self.ssh_dest]
    def _put_to_file(self,name,text):
        ssh_cmd = self.base_ssh_cmd() + ['(umask 0377; cat - >%s)' % name]
        p = subprocess.Popen(ssh_cmd,stdin=subprocess.PIPE)
        p.communicate(input=text.encode())
        if p.returncode:
            raise SystemError('non-zero exit code %d' % p.returncode)
    def do_put_ssh(self):
        # Do write and move as separate operations, so we don't
        # trash the authorized_keys file with a partial copy.
        # (If we do lose access to an instance, we can re-mount the
        # disk as a 2nd drive on another instance, and fix authorized_keys.)
        self._put_to_file('authorized_keys',self.authorized_keys())
        subprocess.check_call(self.base_ssh_cmd() + [
                'mv','authorized_keys','.ssh'
                ])
    def do_show_ssh(self):
        print(self.authorized_keys().strip()) # avoid double newline
    ########
    # LTS key management
    # This is a temporary measure until git annex works with roles, which
    # we'll hopefully see in ubuntu 20.04. Until then, we distribute a
    # dedicated AWS key that allows access to the LTS bucket. This key
    # will need to be rotated whenever someone with developer access
    # leaves the company, so distribution is automated here.
    #
    # Configure the keys by creating a .lts_keys file in the twoxar-demo/web1
    # directory on the machine running authkeys. Then do:
    # authkeys.py put_lts all
    ########
    # XXX This is disabled because LTS no longer uses shared IAM keys. Most
    # XXX machines use role keys. Machines outside AWS that still need IAM
    # XXX keys should each belong to a different IAM user, who sets up the
    # XXX .lts_keys file manually. The lts_status rekey subcommand should no
    # XXX longer be needed because keys are always passed in the environment,
    # XXX rather than relying on caching in .git/annex/creds.
    def OLD_do_put_lts(self):
        user = 'www-data' if self.apache else 'ubuntu'
        dest_dir = f'/home/{user}/2xar/twoxar-demo/web1'
        from path_helper import PathHelper
        fn = '.lts_keys'
        text = open(PathHelper.website_root+fn).read()
        self._put_to_file(fn,text)
        subprocess.check_call(self.base_ssh_cmd() + [
                'sudo','mv',fn,dest_dir,
                ])
        if user != 'ubuntu':
            import os
            dest_path = os.path.join(dest_dir,fn)
            subprocess.check_call(self.base_ssh_cmd() + [
                    'sudo','chown',user,dest_path,
                    ])
        self.lts_status('--aws-key',f'"{text}"','rekey')
    def lts_status(self,*args):
        cmd = [
                '2xar/opt/conda/envs/py3web1/bin/python3',
                '2xar/twoxar-demo/web1/scripts/lts_status.py',
                ]+list(args)
        if self.apache:
            subcmd = ' '.join(cmd)
            cmd = [f"sudo su -s /bin/bash www-data -c '{subcmd}'"]
        subprocess.check_call(self.base_ssh_cmd() + cmd)
    ########
    # AWS key management
    # This section is no longer used, but is left as an example.
    ########
    def s3cfg(self):
        if True:
            # keys are no longer required; just strip markers from template
            return s3cfg_template.replace('???','')
        # the code below is left as an example in case we want to manage this
        # file on a machine outside AWS in the future (in which case it would
        # need key values)
        lines = []
        key_to_use = 'external_platform_role'
        for line in s3cfg_template.split('\n'):
            if line.endswith('???'):
                if line.startswith('access'):
                    line = line[:-3]+get_aws_key(key_to_use,'access')
                elif line.startswith('secret'):
                    line = line[:-3]+get_aws_key(key_to_use,'secret')
                else:
                    raise Exception('unknown fillin line: '+line)
            lines.append(line)
        return '\n'.join(lines)
    def do_show_aws(self):
        print(self.s3cfg().strip()) # strip leading/trailing newline
    def do_put_aws(self):
        self._put_to_file('s3cfg',self.s3cfg())
        subprocess.check_call(self.base_ssh_cmd() + [
                'mv','s3cfg','.s3cfg'
                ])
    inactive_list = [
            'ono-ui',
            'ono-worker',
            ]

if __name__ == '__main__':
    cmd_prefix='do_'
    cmds = [x[len(cmd_prefix):]
            for x in Machine.__dict__
            if x.startswith(cmd_prefix)
            ]
    mch_list = Machine.all_machine_names()
    import argparse
    parser = argparse.ArgumentParser(
            description='manage keys'
            )
    parser.add_argument('--pem')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--force-start',action='store_true')
    parser.add_argument('--apache',action='store_true',
            help='(for new_key_pair only)')
    parser.add_argument('cmd',help=', '.join(sorted(cmds)))
    parser.add_argument('machine',help=', '.join(mch_list))
    args=parser.parse_args()

    def do_func(m):
        func = getattr(m,cmd_prefix+args.cmd)
        if args.dry_run:
            print('execute',func.__name__,'on',m._shortname)
        else:
            func()
    if args.machine == 'all':
        if args.apache:
            raise NotImplementedError(
                    '--apache with "all" not supported; configure individually'
                    )
        need_start = args.cmd not in ('list_ssh','show_access','show_ssh')
        # assume twoxar.com domain and ubuntu login
        for shortname in mch_list:
            # Based on the command, exclude machines from 'all'.
            if args.cmd == 'put_lts' and 'worker' in shortname:
                continue
            # Don't interact with machines that we shouldn't start.
            if shortname in Machine.inactive_list:
                if args.force_start:
                    print('WARNING: shut down '+shortname+' after update')
                else:
                    print('WARNING: skipping '+shortname+'; update manually')
                    continue
            print(shortname+':')
            m = Machine(shortname)
            if need_start and m.aws_mch:
                if args.dry_run:
                    print('start',shortname)
                else:
                    m.aws_mch.do_start()
            do_func(m)
    else:
        m = Machine(args.machine)
        if args.apache:
            m.apache = True
        if args.pem:
            m._pemfile = args.pem
        do_func(m)
