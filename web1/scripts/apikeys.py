#!/usr/bin/env python3
program_description='''Manage keys for 3rd-party APIs.
'''

from dtk.apikeys import ApiKey

def process_keys(to_process,renew,force):
    if force:
        after = None
    else:
        # don't renew unless expiration is less than 7 days away
        from datetime import date,timedelta
        after = date.today() + timedelta(days=7)
    for key in to_process:
        if renew:
            if after and key.expiry_date > after:
                margin = (key.expiry_date-after).days
                print(f'Skipping {key.name}; {margin} days remaining')
            else:
                print('Renewing',key.name)
                key.renew()
        else:
            print(key.name,'renew before',key.expiry_date)
            print('  ',key.api_key)

if __name__ == '__main__':
    good_key_names = set(x.name for x in ApiKey.all_keys())
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=program_description,
            )
    parser.add_argument('--renew',action='store_true')
    parser.add_argument('--force',action='store_true')
    parser.add_argument('key_name',nargs='*',
            metavar='key_name',
            help='valid names are: '+', '.join(sorted(good_key_names)),
            )
    args=parser.parse_args()

    # argparse doesn't properly handle nargs=* and choices together.
    # So, validate input here and report any problems
    bad_key_names = [x for x in args.key_name if x not in good_key_names]
    if bad_key_names:
        import sys
        print('ERROR: invalid key names: '+(', '.join(bad_key_names)),
                file=sys.stderr,
                )
        parser.print_help()
        sys.exit(1)
    to_process = [
            x
            for x in ApiKey.all_keys()
            if not args.key_name or x.name in args.key_name
            ]
    process_keys(to_process,renew=args.renew,force=args.force)
