#!/usr/bin/env python3

program_description='''\
Set and clear maintenance locks.
'''

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=program_description,
            )
    parser.add_argument('command',choices=('request','release'))
    parser.add_argument('task')
    args = parser.parse_args()

    import django_setup
    from runner.models import Process
    err = Process.prod_user_error()
    if err:
        print(err)
        import sys
        sys.exit(1)
    if args.command == 'request':
        Process.maint_request(args.task)
    elif args.command == 'release':
        Process.maint_release(args.task)
    else:
        raise NotImplementedError()
