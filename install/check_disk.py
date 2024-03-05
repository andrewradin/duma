#!/usr/bin/env python3
# ex: set tabstop=4 expandtab:

import subprocess
import os
import pwd
import shutil
from collections import namedtuple
import re

# XXX Note that this script originally pre-dated support for running cron
# XXX jobs in the conda environment, and so works fairly hard to not import
# XXX any outside Duma code. This could probably be relaxed, but as of
# XXX 10/2021 I opted to keep things as they were, and add custom code to
# XXX pull limit info from path_helper.
# XXX I did move in the right direction by forcing python 3 above, but that's
# XXX only needed on pre-20.04 machines (i.e. my laptop at the moment).

# big-ticket items by machine type:
# - dev machines:
#   - not much wasted space in the normal case; there are some sig/meta error
#     cases where extra stuff gets transfered back; try to fix cpio first
# - platform:
#   - should be same as dev, but under www-data rather than ubuntu
# - worker:
#   - lots of sig/meta stuff
#   - lots of intermediate ML stuff; this keeps getting overwritten, so it's
#     really only a savings for unused workspaces

# LATER:
# - _series_matrix.txt.gz?
# - old .rds files?
# - everything under ML/x/results (ticket suggests keeping
#   ML/x/results/*_vector/PlotsAndFinalPredictions, but that may not be needed
#   on worker machines, which is where ML takes up space)
# - big-ticket items from platform:
#   - publish/*/GSE*.gz
#   - publish/E-*/*.zip
#   - publish/*/GPL*.soft
# - keep ws/dpi files gzip'ed?

######
# copied from path_helper
#
def current_user_name():
    return pwd.getpwuid(os.getuid())[0]

def user_install_root(user):
    return '/home/' + user + '/2xar/'
#
######

class DiskCleaner:
    default_age = 14
    default_cutoff = '5M'
    default_groups = 'geo_dl ae_dl pub_rnaseq ws bigtmp'.split()
    def __init__(self,user=None,debug=False):
        self.debug = debug
        if user is None:
            user = current_user_name()
        self.user = user
        self.root = user_install_root(user)
        if self.debug:
            print('root:',self.root)
        self.age_days = self.default_age
        self.set_cutoff(self.default_cutoff)
    def set_cutoff(self,val):
        suffix = val[-1]
        val = float(val[:-1])
        factor = {'K':1,'M':1024,'G':1024*1024}
        self.cutoff = val * factor[suffix.upper()]
    def remove(self,relpath):
        fullpath = relpath if relpath.startswith('/') else self.root+relpath
        print('removing',fullpath)
        if os.path.isfile(fullpath):
            os.remove(fullpath)
        else:
            shutil.rmtree(fullpath)
    def get_local_settings(self,setting_name):
        return subprocess.check_output([
                        self.root+'twoxar-demo/web1/path_helper.py',
                        setting_name,
                ]).decode().strip()
    def check_available_space(self):
        '''Check for low disk space, as configured in local_settings.'''
        identity = subprocess.check_output(['hostname']).decode().strip()
        limit_info = eval(compile(
                self.get_local_settings('disk_space_limits'),
                'pathhelper',
                'eval'))
        result = []
        for mountpoint,limit in limit_info:
            data = os.statvfs(mountpoint)
            blks_per_mb = (1024*1024)/data.f_bsize
            mb_size = data.f_bavail/blks_per_mb
            print("%s: %s free space; limit %s" % (
                            mountpoint,
                            self.space(1024*mb_size),
                            self.space(1024*limit),
                            ))
            if mb_size < limit:
                result.append('%s: %s free space below limit %s' % (
                            mountpoint,
                            self.space(1024*mb_size),
                            self.space(1024*limit),
                            ))
        if result:
            return ['Low disk space on '+identity] + result
    def post_to_monitor(self,msg):
        subprocess.check_call([
                    self.root+'twoxar-demo/web1/scripts/slack_send.sh',
                    msg,
                    ])
    def get_output_lines(self,cmd):
        '''Run cmd in user's 2xar directory; return lines of output.
        '''
        output = subprocess.check_output(cmd
                    ,cwd=self.root
                    ).decode('utf8').rstrip('\n')
        if self.debug:
            print(cmd,'returned',len(output),'bytes')
        if not output:
            return []
        return output.split('\n')
    def get_used_space(self,paths):
        '''Return (blocks,path) pairs for all supplied paths.
        '''
        result = []
        paths = list(paths) # may be a generator
        if not paths:
            return []
        for line in self.get_output_lines(['du'
                        ,'-s'
                        ]+list(paths)
                    ):
            parts = line.split('\t')
            result.append((int(parts[0]),parts[1]))
        return result
    def get_all_used_space(self,section):
        '''Return (blocks,path) pairs for all directories in a section.
        '''
        result = []
        for line in self.get_output_lines(['du'
                        ,'-a'
                        ,'%s/' % section
                        ]
                    ):
            parts = line.split('\t')
            result.append((int(parts[0]),parts[1]))
        return result
    ALL_WS = '[all_ws]'
    BIGTMP = '[bigtmp]'
    def get_paths(self,section,ptype,valid):
        '''Return paths matching parameters.

        Generally, search under 2xar/<section>, although special handling
        can be implemented by section name.
        'ptype' specifies the -type parameter to find (generally 'f' or 'd').
        'valid' is a one-parameter callable that takes a path and returns
        True if the path should be returned; generally 'ptype' is selected
        to make name-matching logic in 'valid' easy to implement, since
        either file or directory paths are acceptable to the deletion logic.
        '''
        # allow age_days to be customized by section
        age_days = self.age_days
        if section == self.BIGTMP:
            tmpdir = self.get_local_settings('bigtmp')
            if tmpdir == '/tmp':
                return # only scan if bigtmp is defined
            cmd = ['find', tmpdir,
                    '-maxdepth','1', # since no patterns, just age
                    '-user', self.user, # since TMPDIR is potentially shared
                    ]
            # Note that this is currently set up not to use ptype, and to
            # just delete all top-level bigtmp files more than 2 days old.
            # If we wanted to restrict it more to fasterq, we could create
            # 2 groups:
            # - one matching files like tmp*.py
            # - one matching directories like fasterq.tmp.worker.*
            # ptype is unused
            age_days = 2
        elif section == self.ALL_WS:
            sections = []
            ws_root = os.path.join(self.root, 'ws')
            for subdir in os.listdir(ws_root):
                try:
                    wsid = int(subdir)
                    sections.append('ws/%s/' % wsid)
                except ValueError:
                    pass
            cmd = ['find'] + sections + [
                    '-type',
                    ptype,
                    ]
        else:
            cmd = ['find'
                    ,'%s/' % section
                    ,'-type'
                    ,ptype
                    ]
        if age_days:
            cmd += ['-ctime'
                    ,'+%d' % age_days
                    ]
        for line in self.get_output_lines(cmd):
            if valid(line):
                yield line
    @staticmethod
    def space(blocks):
        if blocks < 512:
            return "%dK" % blocks
        meg = blocks/1024.0
        if meg < 10:
            return "%0.1fM" % meg
        if meg < 512:
            return "%0.0fM" % meg
        gig = meg/1024.0
        return "%0.1fG" % gig
    @staticmethod
    def usage_stat(space_items):
        cnt = len(space_items)
        if cnt == 0:
            return 'none'
        used = DiskCleaner.space(sum([x[0] for x in space_items]))
        return '%d using %s' % (cnt,used)
    def search_disk(self):
        items = []
        items += self.get_all_used_space('publish')
        items += self.get_all_used_space('ws')
        items.sort(reverse=True)
        for item in items:
            if item[0] < self.cutoff:
                break
            print(' %7s %s' % (self.space(item[0]),item[1]))
    Search=namedtuple('Search','code label section ptype matcher')
    groups=(
        Search('geo_dl','GEO downloads','publish','d'
            ,lambda x: re.match(r'publish/[^/_]*/GSE[^/]*$',x)),
        Search('ae_dl','AE downloads','publish','f'
            ,lambda x: re.match(r'publish/E-[^/]*/[^/]*\.zip$',x)),
        Search('pub_rnaseq','RNA seq aborts','publish','d'
            ,lambda x: re.match(r'publish/[^/_]*/rnaSeqAnalysis$',x)),
        Search('pub_rds','publish rds','publish','f'
            ,lambda x: re.match(r'publish/.*\.rds$',x)),
        Search('pub_gz','publish gz','publish','f'
            ,lambda x: re.match(r'publish/.*\.gz$',x)),
        Search('soft','publish soft','publish','f'
            ,lambda x: re.match(r'publish/.*soft',x)),
        Search('ws','Temp WS Inputs', ALL_WS, 'd'
            ,lambda x: re.match(r'ws/[^/]*/(trgscrimp|esga|gpath|path|capp|gpbr|codes)/.*$',x)),
        Search('bigtmp','tempfiles', BIGTMP, ''
            ,lambda x: True),
        )
    @classmethod
    def group_descriptions(cls):
        return ['%s - %s' % (x.code,x.label) for x in cls.groups]
    @classmethod
    def get_groups_by_name(cls,names):
        return [x for x in cls.groups if x.code in names]
    def scan_disk(self,verbose=False,remove=False,groups=None):
        if groups is None:
            groups = self.get_groups_by_name(self.default_groups)
        for group in groups:
            matches = self.get_paths(group.section,group.ptype,group.matcher)
            used = self.get_used_space(matches)
            print("%s: %s" % (group.label,self.usage_stat(used)))
            if verbose:
                for item in used:
                    print(' %7s %s' % (self.space(item[0]),item[1]))
            if remove:
                for item in used:
                    self.remove(item[1])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''clean up disk

Search beneath a 2xar install directory for various groups
of file or directory names known to take significant
unneccessary space, and remove them.  The currently
available groups are:
  '''+'\n  '.join(DiskCleaner.group_descriptions()),
        )
    parser.add_argument('-d','--debug',action='store_true'
        ,help='show internal debugging information'
        )
    parser.add_argument('-v','--verbose',action='store_true'
        ,help='show each discovered path'
        )
    parser.add_argument('-r','--remove',action='store_true'
        ,help='remove located files (default is list only)'
        )
    parser.add_argument('-s','--search',action='store_true'
        ,help='show all heavy usage, not just pattern matches'
        )
    parser.add_argument('-u','--user'
        ,help='look under specified user for 2xar dir (for www-data)'
        )
    parser.add_argument('-c','--cutoff'
        ,help='minimum file/dir size to display with -s (default %s)' % (
                    DiskCleaner.default_cutoff,
                    )
        )
    parser.add_argument('-a','--age'
        ,help='minimum age of files to qualify (default %d days)' % (
                    DiskCleaner.default_age,
                    )
        )
    parser.add_argument('group',nargs='*'
        ,help='specific patterns to search for (default %s)' % (
                    " ".join(DiskCleaner.default_groups),
                    )
        )
    args=parser.parse_args()

    parms = {'debug':args.debug}
    if args.user:
        parms['user'] = args.user
    dc = DiskCleaner(**parms)

    if args.cutoff:
        dc.set_cutoff(args.cutoff)
    if args.age:
        dc.age_days = int(args.age)

    if args.search:
        dc.search_disk()
    else:

        warnings = dc.check_available_space()
        if warnings:
            dc.post_to_monitor('\n'.join(warnings))
        if args.group:
            groups=DiskCleaner.get_groups_by_name(args.group)
        else:
            groups=None
        dc.scan_disk(verbose=args.verbose
                ,remove=args.remove
                ,groups=groups
                )
