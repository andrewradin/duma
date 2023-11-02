#!/usr/bin/env python3

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description='''
Extract info about latest efo release.
''',
            )
    parser.add_argument('url')
    args = parser.parse_args()

    import urllib.request
    with urllib.request.urlopen(args.url) as rsp:
        for line in rsp:
            line = line.decode()
            if line.startswith('data-version:'):
                parts = line.split('/')
                version = parts[-2]
                break
    # at this point, version looks like 'v3.18.0'
    import re
    m = re.match('v(\d+)\.(\d+)',version)

    print('\n======================================')
    print('versions.py entry should contain:')
    print("EFO_VER='%s_%s'"%(m.group(1),m.group(2)))
    print('\nThen:')
    print('make input')
    print('make build')
    print('make publish_s3')
