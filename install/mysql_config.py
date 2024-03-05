#!/usr/bin/env python3

from dtk.lazy_loader import LazyLoader
from path_helper import PathHelper
import os
import subprocess
import shutil

# XXX should share a common base class with disk_layout.py

class MysqlConfig(LazyLoader):
    data_vol_root = '/mnt2/mysql_data'
    std_location = '/var/lib/mysql'
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
            return self.error(*args)
        print('ERROR:',*args)
        while not self.force_fix:
            rsp = input(prompt+' [y/N]: ').lower()
            if rsp == 'y':
                break
            if rsp in ('n',''):
                self.error_count += 1
                return
        func()
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
    def _datadir_loader(self):
        return os.path.join(self.data_vol_root,'mysql')
    def check_data_location(self):
        cur_loc = os.path.normpath(self.get_variable('datadir'))
        if cur_loc == self.datadir:
            print('Data already in',self.data_vol_root)
        elif cur_loc == self.std_location:
            self.fix('move?',self.relocate_data,
                    'Data in standard location',
                    )
        else:
            self.error('Data in unexpected location:',cur_loc)
    def relocate_data(self):
        self.service('mysql','stop')
        subprocess.check_call(['sudo','mkdir','-p',self.data_vol_root])
        subprocess.check_call([
                'sudo',
                'rsync',
                '-av',
                '--delete',
                self.std_location,
                self.data_vol_root,
                ])
        self.append_to_file('/etc/mysql/my.cnf',[
                '[mysqld]',
                'datadir='+self.datadir,
                ])
        self.append_to_file('/etc/apparmor.d/local/usr.sbin.mysqld',[
                '/mnt2/mysql_data/ r,'
                '/mnt2/mysql_data/** rwk,'
                ])
        self.service('apparmor','restart')
        self.service('mysql','start')
    def append_to_file(self,path,lines):
        for line in lines:
            subprocess.check_call([
                    'sudo','sh','-c',
                    'echo '+line+' >> '+path,
                    ])
    def service(self,service,cmd):
        subprocess.check_call(['sudo','service',service,cmd])
    def get_variable(self,var_name):
        vname = '@@'+var_name
        result = self.get_cmd_output([
                'mysql',
                '-u','root',
                '-e','select '+vname,
                ])
        assert len(result) == 2
        assert result[0] == vname
        return result[1]

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='''Show/alter mysql configuration.

This script helps to set up a 'standard' mysql on a machine.

Data should reside on the data volume.
''',
            )
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--no-fix',action='store_true')
    parser.add_argument('--force-fix',action='store_true')
    args = parser.parse_args()

    mysql_mgr = MysqlConfig()
    mysql_mgr.verbose = args.verbose
    mysql_mgr.offer_fix = not args.no_fix
    mysql_mgr.force_fix = args.force_fix
    mysql_mgr.check_data_location()
    import sys
    sys.exit(1 if mysql_mgr.error_count else 0)
