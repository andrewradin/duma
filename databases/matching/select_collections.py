#!/usr/bin/env python3

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='''
    extract specified collections from an ingredients file
    ''')
    parser.add_argument('file')
    parser.add_argument('collection',nargs='*')
    args = parser.parse_args()

    colls = set(args.collection)
    for name in open(args.file).read().split():
        parts = name.split('.')
        if parts[0] in colls or not colls:
            print(name)
