#!/usr/bin/env python3

import os
import sys
try:
    from path_helper import PathHelper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
    from path_helper import PathHelper

import subprocess

def move_s3_files(
        bucket,
        filename,
        action='get',
        gzip=False,
        verbose=False,
        downloads_archive=False,
        ):
    from dtk.s3_cache import S3Bucket,S3MiscBucket,S3File
    if bucket == 'storage':
        s3b = S3MiscBucket()
    else:
        s3b = S3Bucket(bucket)
    # extract inner low-level bucket object
    b=s3b.bucket
    if downloads_archive:
        # This is a bit of a hack. The 'bucket' parameter must be a name that
        # resolves to a VBucket. If that's the case, we patch the name of the
        # S3 bucket to pull from, and the directory to use. The
        # duma-etl-downloads S3 bucket is structured like
        # 2xar-versioned-datasets, with a subdirectory for each file_class,
        # but we want all the files to be in a single directory locally.
        # Also, by clearing s3b.meta, we disable the compress-on-write path.
        assert b.__class__.__name__ == 'VBucket'
        b.aws_bucket_name = 'duma-etl-downloads'
        b.cache_path = PathHelper.downloads
        s3b.meta = None
    if action == 'exists':
        exists = (filename in b.namelist())
        if verbose:
            print(filename,'exists' if exists else 'does not exist','on S3')
        return 0 if exists else 1
    elif action == 'put':
        if s3b.compress_on_write(filename):
            # Special auto-compress handling
            # - passed-in filename should reflect what's present in cache dir
            # - if file isn't compressed, compress it
            # - after write, leave either compressed or uncompressed version
            #   in cache, to match what's configured to happen on read
            # By handling pre-compressed files, we allow the ETL the option
            # of using less disk space, and prevent accidental double
            # compression.
            # Note that the behavior of changing the cache state in a --put
            # operation, although it makes things consistent for consumers
            # of the data, makes it harder to re-publish previously published
            # files (although by default the versioning ETL guards against
            # this anyway).
            if filename.endswith('.gz'):
                write_fn = filename
                zip_mode = None
                if s3b.decompress_on_read(filename):
                    cleanup_mode = 'unzip'
                else:
                    cleanup_mode = None
            else:
                write_fn = filename+'.gz'
                if s3b.decompress_on_read(filename):
                    zip_mode = 'tmp'
                    cleanup_mode = 'remove'
                else:
                    zip_mode = 'normal'
                    cleanup_mode = None
        elif gzip:
            # legacy gzip command option
            write_fn = filename+'.gz'
            zip_mode = 'tmp'
            cleanup_mode = 'remove'
        else:
            # just write
            write_fn = filename
            zip_mode = None
            cleanup_mode = None
        if zip_mode:
            cmd = ["gzip"]
            if zip_mode == 'tmp':
                # leave both zipped and unzipped files in place
                cmd.append('-k')
            cmd.append(b.cache_path+'/'+filename)
            subprocess.check_call(cmd)
        if verbose:
            print('putting',write_fn)
        b.put_file(write_fn)
        if cleanup_mode == 'remove':
            os.remove(b.cache_path+'/'+write_fn)
        elif cleanup_mode == 'unzip':
            subprocess.check_call(['gunzip',b.cache_path+'/'+write_fn])
    else:
        assert not gzip # deprecated
        # the gzip option on read is unused as of sprint 250, and complicates
        # things for AutoCompress (because a call with the wrong mode could
        # load the wrong thing into the cache). Disallow it, and instead
        # delegate this to a normal fetch, which needs to handle all the
        # cases anyway.
        if verbose:
            print('getting',filename)
        s3f = S3File(s3b,filename)
        s3f.fetch()
    return 0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='upload/download S3 files')
    parser.add_argument('bucket')
    parser.add_argument('file')
    parser.add_argument('--gzip',action='store_true')
    parser.add_argument('--put',action='store_true')
    parser.add_argument('--exists',action='store_true')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--downloads-archive',action='store_true')
    args = parser.parse_args()
    if args.exists:
        action = 'exists'
    elif args.put:
        action = 'put'
    else:
        action = 'get'

    if not args.bucket:
        # Passing an empty string for the bucket means that the file arg
        # should be a full path to the cache directory, which can be parsed
        # here to provide the bucket and file parts. This simplifies some
        # Makefiles
        parts = args.file.split('/')
        ws_dir = '/'.join(parts[:-2])
        assert ws_dir+'/' == PathHelper.storage
        args.file = parts[-1]
        args.bucket = parts[-2]
    result = move_s3_files(
            args.bucket,
            args.file,
            action,
            args.gzip,
            args.verbose,
            args.downloads_archive,
            )
    sys.exit(result)
