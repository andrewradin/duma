#!/usr/bin/env python3

from __future__ import print_function
import sys
import os
import re
import shutil
import six

# Make sure we can find PathHelper
try:
    import path_helper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+'/..')

import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()

from path_helper import PathHelper

def get_downloads(make_dir):
    '''Return filenames from show_downloads make target.'''
    from dtk.files import open_pipeline
    pipe = [
            ['make','--no-print-directory','-C',make_dir,'show_downloads']
            ]
    files = set()
    with open_pipeline(pipe) as fh:
        for line in fh:
            for path in line.strip().split():
                dir,fn = os.path.split(path)
                if not dir.endswith('ws/downloads'):
                    continue
                files.add(fn)
    return files

class FileIndex:
    def __init__(self):
        # build directory from download filename to ETL source directory,
        # and vice-versa
        self.srcs = {}
        self.files = {}
        import glob
        makefiles=glob.glob(PathHelper.repos_root+'databases/*/Makefile')
        for makefile in makefiles:
            make_dir = os.path.split(makefile)[0]
            downloads = get_downloads(make_dir)
            if downloads:
                src = os.path.split(make_dir)[1]
                for fn in downloads:
                    self.srcs.setdefault(fn,[]).append(src)
                    self.files.setdefault(src,[]).append(fn)

class S3Index:
    suffix = '.tgz'
    def __init__(self):
        from dtk.aws_api import Bucket
        self.bucket = Bucket(name='duma-etl-downloads',cache_path='.')
    def get_existing_srcs(self):
        return set([
                key[:-len(self.suffix)]
                for key in self.bucket.namelist()
                if key.endswith(self.suffix)
                ])
    def archive(self,src,files):
        tarfile = src + self.suffix
        dnld = PathHelper.downloads
        import subprocess
        subprocess.check_call(['tar','cvzf',tarfile,'-C',dnld]+list(files))
        self.bucket.put_file(tarfile)
        os.unlink(tarfile)
    def restore(self,src):
        tarfile = src + self.suffix
        self.bucket.get_file(tarfile)
        dnld = PathHelper.downloads
        import subprocess
        subprocess.check_call(['tar','xvzf',tarfile,'-C',dnld])
        os.unlink(tarfile)

def restore(*srcs):
    s3 = S3Index()
    already_archived = s3.get_existing_srcs()
    for src in srcs:
        assert src in already_archived
        s3.restore(src)

def clean(*srcs):
    s3 = S3Index()
    already_archived = s3.get_existing_srcs()
    for src in srcs:
        assert src in already_archived
        make_dir = PathHelper.repos_root+'databases/'+src
        files = get_downloads(make_dir)
        for fn in files:
            path = PathHelper.downloads+fn
            if os.path.exists(path):
                print('removing',path)
                os.unlink(path)
            else:
                print('nothing at',path)

def archive(*srcs):
    s3 = S3Index()
    already_archived = s3.get_existing_srcs()
    for src in srcs:
        assert force_mode or src not in already_archived
        make_dir = PathHelper.repos_root+'databases/'+src
        files = get_downloads(make_dir)
        assert files
        s3.archive(src,files)

def analyze():
    fi = FileIndex()
    s3 = S3Index()
    # verify that names are unambiguous
    for k,v in six.iteritems(fi.srcs):
        if len(v) != 1:
            print(k,'is ambiguous:',v)
            del(fi.srcs[k]) # treat as unknown
    # go through downloads directory, and get space used per ETL source,
    # also reporting any unmapped files
    sizes = {}
    from dtk.files import get_dir_file_names
    for fn in get_dir_file_names(PathHelper.downloads):
        if fn not in fi.srcs:
            print('unknown file:',fn)
        else:
            path = os.path.join(PathHelper.downloads,fn)
            size = os.stat(path).st_size
            sizes.setdefault(fi.srcs[fn][0],[]).append(size)
    # now report per ETL source
    already_archived = s3.get_existing_srcs()
    for src,files in six.iteritems(fi.files):
        units='BKMGTP'
        idx = 0
        if src in sizes:
            size = sum(sizes[src])
            cnt = len(sizes[src])
        else:
            size = 0
            cnt = 0
        while size > 1024:
            size /= 1024
            idx += 1
        archived = '+' if src in already_archived else '-'
        msg = '%s%s: %d%s, %d file(s)' % (archived,src,size,units[idx],cnt)
        missing = len(files) - cnt
        if missing:
            msg += '; %d file(s) missing' % missing
        print(msg)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description="manage download directory",
            )
    parser.add_argument('--force',action='store_true')
    parser.add_argument('cmd',choices=['analyze','archive','clean','restore'])
    parser.add_argument('src',nargs='*')
    args = parser.parse_args()
    global force_mode
    force_mode=args.force
    func = locals()[args.cmd]
    func(*args.src)
