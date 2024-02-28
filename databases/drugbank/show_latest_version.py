#!/usr/bin/env python3

def fetch_directory(url,license):
    from dtk.files import open_pipeline
    import json
    return json.load(open_pipeline([
            ['curl','-u',license,url+'.json'],
            ]))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description='''
Extract info about latest drugbank release.
''',
            )
    parser.add_argument('license')
    parser.add_argument('url')
    args = parser.parse_args()

    l = fetch_directory(args.url,args.license)
    # find first (latest) XML-style entry
    for d in l:
        if d['format'] == 'XML':
            break
    print('\n======================================')
    print('versions.py entry should contain:')
    print("DRUGBANK_VER='%s'"%d['id'])
    print('\nThen:')
    print('make input')
    print('make build')
    print('make publish_s3')
