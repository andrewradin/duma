#!/usr/bin/env python3

from dtk.lazy_loader import LazyLoader
from path_helper import PathHelper
import os
import subprocess

# XXX Somewhere we should support:
# XXX - resize volume (requires filesystem resize); see
# XXX   https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/recognize-expanded-volume-linux.html
# XXX - reconfigure mysql to use data volume

class VolumeManager(LazyLoader):
    data_vol_root = '/mnt2/ubuntu'
    verbose = False
    splice_mode = False
    offer_fix = True
    force_fix = False
    error_count = 0
    def error(self,*args):
        print('ERROR:',*args)
        self.error_count += 1
        assert not self.force_fix # in force_fix mode, die on an unfixable error
    def ok(self,*args):
        if self.verbose:
            print('OK:',*args)
    def fix(self,prompt,func,*args):
        if not self.offer_fix:
            self.error(*args)
            return False
        print('ERROR:',*args)
        while not self.force_fix:
            rsp = input(prompt+' [y/N]: ').lower()
            if rsp == 'y':
                break
            if rsp in ('n',''):
                self.error_count += 1
                return False
        func()
        return True
    def snippet(self,cmd):
        '''Run a pre-defined code snippet.'''
        subprocess.check_call([
                'make',
                '-f',PathHelper.repos_root+'install/snippets.mk',
                ]+cmd
                )
    def get_cmd_output(self,cmd):
        '''Return output of a command as an array of lines.

        Unicode conversion is done and newlines are stripped.
        '''
        lines = subprocess.check_output(cmd).decode('ascii')
        # don't return empty line resulting from final newline
        return lines.split('\n')[:-1]
    def check_layout(self):
        self.check_blkdevs()
        self.check_volumes()
        self.check_roots()
        self.check_symlinks()
        self.check_deprecated_packages()
        self.check_deprecated_directories()
        self.check_selenium_drivers()
    def check_selenium_drivers(self):
        for driver in ['geckodriver']:
            if os.path.exists('/usr/local/bin/'+driver):
                self.ok(driver,'installed')
            else:
                self.fix('install?',
                        lambda:self.install_selenium_driver(driver),
                        driver,
                        'not installed',
                        )
    def install_selenium_driver(self,driver):
        subprocess.check_call([
                'make',
                '-f',PathHelper.repos_root+'selenium/Makefile',
                driver,
                ]
                )
    def check_deprecated_packages(self):
        installed = set(self.get_cmd_output(['apt-mark','showmanual']))
        deprecated = set([
                'r-base',
                'r-base-dev',
                ])
        for pkg in sorted(installed & deprecated):
            self.error('apt package',pkg,'should be removed')
    def check_deprecated_directories(self):
        # first check the directories that can be deleted without sudo
        deprecated = [
                    os.path.join(PathHelper.install_root,'opt',x)
                    for x in (
                            'python-louvain',
                            )
                ]
        from dtk.files import rm_readonly_tree
        for path in deprecated:
            if os.path.exists(path):
                self.fix('remove?',
                        lambda:rm_readonly_tree(path),
                        path,
                        'should be removed',
                        )
        # now check the directories needing elevated privs; don't offer removal
        deprecated = [
                '/usr/local/lib/R',
                ]
        for path in deprecated:
            if os.path.exists(path):
                self.error(path,'should be removed (using sudo)')
    def check_blkdevs(self):
        '''Verify disk devices.'''
        seen = set(x.strip()
                for x in self.get_cmd_output(['ls','/dev/disk/by-label'])
                )
        wanted_vols = [
                    'cloudimg-rootfs',
                    'DumaData',
                    ]
        for vol in wanted_vols:
            if vol not in seen:
                self.error('no volume with label',vol)
                # XXX provide further instruction to user? This should only
                # XXX happen for the DumaData volume, in which case do
                # XXX whatever is needed from the list below:
                # - create volume in AWS; assign name (like dev08-extra)
                # - attach volume to the instance as /dev/sdf
                # - sudo mkfs.ext4 /dev/nvme1n1
                # - sudo e2label /dev/nvme1n1 DumaData
                self.offer_fix = False
    def check_volumes(self):
        '''Verify disk mounts and sizes.'''
        G=1024*1024
        # for each mount point, the minimum disk size and max fraction used
        # XXX This has caused problems on a couple of instances.
        # XXX - provide better defaults?
        # XXX - provide an easier override than hacking this file?
        expected = {
                '/':(30*G,.8),
                '/mnt2':(60*G,.8),
                }
        lines = self.get_cmd_output(['df'])
        # process lines, skipping header
        seen = set()
        for line in lines[1:]:
            fs,size,used,_,_,mount = line.split()
            size = int(size)
            used = int(used)
            try:
                min_size, max_used = expected[mount]
            except KeyError:
                continue
            seen.add(mount)
            if size < min_size:
                self.error(fs,'on',mount,'too small')
            elif used > size * max_used:
                self.error(fs,'on',mount,'too full')
            else:
                self.ok(fs,'on',mount)
        for missing in sorted(set(expected.keys())-seen):
            if not self.fix('mount?',
                    lambda:self.snippet([
                            '-C',PathHelper.install_root,
                            'mount',
                            ]),
                    'no volume on',missing,
                    ):
                self.offer_fix = False
    def check_roots(self):
        '''Verify that the major root directories are present.'''
        # XXX how to fix:
        # XXX venv and install_root should be created by install.sh
        # XXX data_vol_root should be created by:
        # - sudo mkdir /mnt2/ubuntu
        # - sudo chown ubuntu /mnt2/ubuntu
        for path in (
                PathHelper.venv,
                PathHelper.install_root,
                self.data_vol_root,
                ):
            path = os.path.abspath(path)
            if not os.path.exists(path):
                self.error(path,'missing')
                self.offer_fix = False
            else:
                rp = os.path.realpath(path)
                if path != rp:
                    self.error(path,'points to',rp)
                    self.offer_fix = False
                else:
                    self.ok(path)
    def check_symlinks(self):
        '''Verify that the proper subdirs of ~/2xar are symlinked.'''
        symlinks = ('lts','publish','ws','twoxar-demo')
        # Just a quick warning here that, in splice mode, at some point in
        # the loop the twoxar-demo directory from the template will be
        # replaced with the one from the existing data drive, which will
        # cause the version of snippets.mk to potentially be different.
        # This would only be an issue if this script was actively being
        # developed during an upgrade. If it causes a problem, the
        # workaround is:
        # - the splice step should fail
        # - ssh into the instance being upgraded, and pull the correct
        #   version of the source from github, then log out
        # - re-run the upgrade script, which should re-try the splice step,
        #   which should now succeed
        with os.scandir(PathHelper.install_root) as it:
            seen_symlinks = set()
            for entry in it:
                if entry.name in symlinks:
                    seen_symlinks.add(entry.name)
                    want = os.path.join(self.data_vol_root,entry.name)
                    if entry.is_symlink():
                        got = os.path.realpath(entry.path)
                        if want != got:
                            self.error(entry.path,'points to',got,'not',want)
                        else:
                            self.ok(entry.path) # symlink ok
                    else:
                        if self.splice_mode:
                            self.splice(entry.name,'not a symlink')
                        else:
                            self.fix('move?',
                                    lambda:self.wrap_dvol_snippet(
                                            'to_dvol',
                                            entry.name,
                                            ),
                                    entry.path,
                                    'is not a symlink',
                                    )
                elif entry.is_symlink():
                    self.error(entry.path,'is a symlink not a directory')
                else:
                    self.ok(entry.path) # not a symlink, as expected
            missed = set(symlinks) - seen_symlinks
            for name in missed:
                if self.splice_mode:
                    self.splice(name,'missing')
                else:
                    self.error('symlink not present:',missed)
    def wrap_dvol_snippet(self,operation,to_move):
        rundir = PathHelper.install_root
        self.snippet([
                '-C',rundir,
                'TO_MOVE='+to_move,
                'DVOL='+self.data_vol_root,
                operation,
                ]),
        leftover = os.path.join(rundir,to_move+'.old')
        if os.path.exists(leftover):
            from dtk.files import rm_readonly_tree
            rm_readonly_tree(leftover)
    def splice(self,name,detail):
        self.fix('splice?',
                lambda:self.wrap_dvol_snippet('splice_dvol',name),
                name,
                'is',
                detail,
                )

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='''Show/alter disk layout.

This script helps to construct a 'standard' Duma disk layout on a machine.

The basic idea is that the root volume of a machine holds all the installed
non-Duma software, and an external volume holds the Duma git repo and the
data. For AWS instances, this allows an easier OS upgrade by building a
template root volume and cloning it to replace the root volumes of instances,
while leaving the data volumes intact.
''',
            )
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--no-fix',action='store_true')
    parser.add_argument('--splice',action='store_true')
    parser.add_argument('--force-fix',action='store_true')
    args = parser.parse_args()

    vol_mgr = VolumeManager()
    vol_mgr.verbose = args.verbose
    vol_mgr.offer_fix = not args.no_fix
    vol_mgr.splice_mode = args.splice
    vol_mgr.force_fix = args.force_fix
    vol_mgr.check_layout()
    import sys
    sys.exit(1 if vol_mgr.error_count else 0)
