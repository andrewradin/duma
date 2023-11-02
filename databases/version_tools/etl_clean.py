#!/usr/bin/env python3

func_name = 'custom_cleanup_data'

program_description=f'''\
Manage ETL disk space usage.

Versioned ETL directories accumulate older version outputs over time.
These should be deleted eventually to save space, although keeping
more recent versions can be useful for debugging and analysis.

When run in a versioned ETL directory, this program deletes older
versioned outputs. A function {func_name}() may be defined in the
local versions.py file to specify additional cleanup files. See the
source of this script for details.
'''

from path_helper import PathHelper
import os

class GroupedFileList(list):
    def add(self,group,fn):
        self.append((group,fn))
    def groups(self):
        return sorted(set([x[0] for x in self]))
    def ordered_names(self):
        return [x[1] for x in sorted(self)]
    def show_names(self,indent='  '):
        for fn in self.ordered_names():
            print(indent+fn)
    def total_size(self):
        return sum([os.path.getsize(x[1]) for x in self])
    def summary(self,label):
        from dtk.text import fmt_size
        print(label,*self.groups())
        print('  Total size:',fmt_size(self.total_size()))
        self.show_names()

def cwd_to_etl_name():
    return os.path.abspath('.').split('/')[-1]

def clean_etl_dir(etl_name,keep,remove):
    from dtk.etl import get_last_published_version, get_versions_namespace
    last_ver = get_last_published_version(etl_name)
    last_to_delete = last_ver - keep
    import re
    filetypes = [
            (re.compile(r'.*\.v([0-9]+)\..*$'),int,last_to_delete,False),
            ]
    ns = get_versions_namespace(etl_name)
    if func_name in ns:
        # If we find a custom function, add its output to the list of
        # things to be cleaned up. The output is expected to be a list
        # of (regex,cvt,cutoff,is_dir) triples, where:
        # - regex is a string or compiled regular expression that uniquely
        #   matches filenames to be considered for deletion, where group(1)
        #   is the version string portion of the filename
        # - cvt is a converter applied to the filename version string to
        #   produce something that sorts in chronological order; this would
        #   typically be 'str' for fixed-width strings, or 'int' for
        #   variable-width integer version numbers
        # - cutoff is the most recent version to be deleted, which should
        #   be comparable to the return value of cvt (e.g. int or str)
        # - is_dir is True if the matched name is a directory, False if it's
        #   a file
        # Note that because of the odd way we import the versions.py file,
        # the function invocation below can't access the versions dict as
        # an enclosing scope, so we pass it as a parameter. FAERS and AACT
        # are examples of ETLs that use this mechanism.
        filetypes += ns[func_name](ns['versions'],last_to_delete)
    from dtk.etl import get_etl_dir
    for regex,cvt,cutoff,is_dir in filetypes:
        if isinstance(regex,str):
            regex = re.compile(regex)
        clean_dir(get_etl_dir(etl_name),regex,cvt,cutoff,is_dir,remove)

def clean_dir(etl_dir,regex,cvt,cutoff,is_dir,remove):
    # XXX if 'is_dir' is true, the sizes reported by summary() will
    # XXX only include the directory itself, not what's in it. Fix someday.
    delete = GroupedFileList()
    keep = GroupedFileList()
    for fn in os.listdir(etl_dir):
        m = regex.match(fn)
        if not m:
            continue
        ver = cvt(m.group(1))
        if ver > cutoff:
            keep.add(ver,fn)
            continue
        delete.add(ver,fn)
    if remove:
        delete.summary('Removing versions:')
        for fn in delete.ordered_names():
            if is_dir:
                import shutil
                shutil.rmtree(fn)
            else:
                os.remove(fn)
        keep.summary('Keeping versions:')
    else:
        delete.summary('Versions to delete:')
        keep.summary('Versions to keep:')

def extract_arguments(ns,argnames):
    return {name:getattr(ns,name) for name in argnames}

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=program_description,
            )
    parser.add_argument('--keep',type=int,default=3,
            help='number of versions to retain (default: %(default)s)')
    parser.add_argument('--remove',action='store_true',
            help='actually remove files, rather than just listing',
            )
    args = parser.parse_args()
    # for now, just do local directory
    clean_etl_dir(
            cwd_to_etl_name(),
            **extract_arguments(args,['keep','remove']),
            )
