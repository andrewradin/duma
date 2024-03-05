# This is a tool for piecing an interrupted copy back together. The scenario
# is, you do:
#   mv source dest
# and then interrupt it and repeat it. You end up with no source, and your data
# split between dest and dest/dest.
# 
# This script can be called for each pair of subdirectories:
#   merge.py dest/a dest/dest/a
# It will try to move everything into the second directory, handling most
# cases.  It will die if both directories have non-identical files with
# the same name (in this case you should most likely manually remove the
# first, with should be shorter).
#
# After successfully running the script on each subdirectory, do:
#   mv dest/dest/* dest
#   rmdir dest/dest
#
def merge(from_dir,to_dir):
    import os
    import shutil
    import subprocess
    assert os.path.isdir(from_dir)
    assert os.path.isdir(to_dir)
    for name in os.listdir(from_dir):
        fpath = os.path.join(from_dir,name)
        tpath = os.path.join(to_dir,name)
        if not os.path.exists(tpath):
            print 'move',fpath,'to',tpath
            shutil.move(fpath,tpath)
        elif all(map(os.path.isfile,[fpath,tpath])):
            subprocess.check_call(['cmp',fpath,tpath])
            # if we get here, the files are identical (if not, the
            # previous command will throw an exception)
            print 'remove',fpath
            subprocess.check_call(['sudo','rm',fpath])
        else:
            print 'merge',fpath,'into',tpath
            merge(fpath,tpath)
    os.rmdir(from_dir)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('from_dir')
    parser.add_argument('to_dir')
    args = parser.parse_args()

    merge(args.from_dir,args.to_dir)

