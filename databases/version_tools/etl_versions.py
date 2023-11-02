#!/usr/bin/env python3

# We want to import versions.py in the directory that called this script
# (NOT this script's source directory/module). This is not something py3
# considers a use case, and the inclusion of '.' in sys.path that made it
# work in py2 (and in our transitional nested py3 environment) no longer
# happens, so simulate it here, and then put things back after the import.
import os
import sys
sys.path = [os.getcwd()] + sys.path
import versions
sys.path = sys.path[1:]

pub_ver_fn='last_published_version'

def write_latest():
    latest_version_id = max(versions.versions.keys())
    latest_published_id = int(open(pub_ver_fn).read())
    d = versions.versions[latest_version_id]
    from atomicwrites import atomic_write
    with atomic_write('versions.mk',overwrite=True) as fh:
        def out(var,val):
            s = '%s=%s\n' % (var,str(val))
            fh.write(s)
        out('OUT_VER_NBR',latest_version_id)
        for var in sorted(d.keys()):
            out(var,d[var])
        if latest_version_id <= latest_published_id:
            out('PUBLISHED','yes')

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='''\
''',
            )
    args = parser.parse_args()
    write_latest()
