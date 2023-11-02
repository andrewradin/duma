#!/usr/bin/env python3

def preferred_path(fn,do_exception):
    file_class = fn.split('.')[0]
    import os
    local_dir = os.path.join('..',file_class)
    local_path = os.path.join(local_dir,fn)
    if os.path.exists(local_path):
        return local_path
    # not available in source directory; check S3
    from dtk.s3_cache import S3File
    s3f = S3File(file_class,fn)
    try:
        s3f.fetch()
    except OSError as ex:
        if do_exception:
            raise OSError(
                    f"'{fn}' does not exist in {local_dir} or on S3"
                    ) from ex
        else:
            # The use case for this function is to allow ETL Makefiles to pull
            # inputs from other ETL directories in the normal case, falling
            # back to S3 to allow regeneration of files from old data. If
            # neither version is available, default to returning the local
            # path, so the error from 'make' is more sensible.
            return local_path
    return s3f.path()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='''
    Find files in etl directories or on S3.
    ''')
    parser.add_argument('file',nargs='+')
    parser.add_argument('--exception',action='store_true')
    args = parser.parse_args()

    print(*[
            preferred_path(x,args.exception)
            for x in args.file
            ])
