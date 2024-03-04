#!/usr/bin/env python3

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description='''
Extract info about latest Selleck Chem release.
''',
            )
    args = parser.parse_args()

    from get_download_urls import SelleckLibraryURLs
    slu = SelleckLibraryURLs()
    prefix = 'https://file.selleckchem.com/downloads/library/'
    prefix_len = len(prefix)
    to_dwnld = []
    for url in sorted(slu.xls_urls):
        if url.startswith(prefix):
            # output entry for versions.py file list
            to_dwnld.append(f"'{url[prefix_len:]}',")
        else:
            # output warning
            print(f"# SKIPPING {url}")

    print('\n======================================')
    print('======================================')
    print('versions.py entry should contain:')
    print("<NEXT_VERSION_NUMBER_HERE>:dict(")
    print("    description=<COMMENT HERE>,",)
    print("    SELLECKCHEM_FILES=' '.join(["+'\n'.join(to_dwnld))
    print('        ])),')
    print('\nThen:')
    print('make input')
    print('make build')
    print('make publish_s3')

