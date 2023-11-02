#!/usr/bin/env python3

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='''
    Run any makes needed to complete partial input builds.
    ''')
    parser.add_argument('--no-publish',action='store_true')
    parser.add_argument('source',nargs='*')
    args = parser.parse_args()

    # change inputs from filenames to (file_class,version) pairs
    source_list = []
    for item in args.source:
        parts = item.split('.')
        source_list.append((parts[0],int(parts[2][1:])))
    # figure out which inputs are out-of-date
    from dtk.etl import get_last_published_version
    unpublished = [
            name
            for name,ver in source_list
            if ver > get_last_published_version(name)
            ]
    # run needed makes
    import subprocess
    target = 'build' if args.no_publish else 'publish_s3'
    for src in unpublished:
        subprocess.check_call(['make','-C','../'+src,target])
