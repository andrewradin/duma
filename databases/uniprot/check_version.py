#!/usr/bin/env python

def check_version(want_version,fn):
    from lxml import etree
    tree=etree.parse(open(fn))
    got_version=tree.getroot().find('{*}version').text
    if want_version != got_version:
        src=tree.getroot().find('{*}publisher/{*}url').text
        raise RuntimeError(
                "Current version %s at %s doesn't match %s"%(
                        got_version,
                        src,
                        want_version,
                        )
                )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='check uniprot version')
    parser.add_argument('version')
    parser.add_argument('metalink_file')
    args = parser.parse_args()

    check_version(args.version,args.metalink_file)
