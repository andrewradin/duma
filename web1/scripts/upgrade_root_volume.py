#!/usr/bin/env python3

import sys
try:
    from dtk.aws_api import VpcControl
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")
    from dtk.aws_api import VpcControl

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='''
Swap the root volume of an instance
'''
            )
    parser.add_argument('--reset',
            action='store_true',
            help='roll back to initial snapshots',
            )
    parser.add_argument('--no-detach-ok',
            action='store_true',
            help="don't require old volume to be attached before swap",
            )
    parser.add_argument('--no-ssh',
            action='store_true',
            help="don't set up a new ssh key pair",
            )
    parser.add_argument('instance',
            help="instance name (AWS name tag and aws_op Machine name)",
            )
    parser.add_argument('target',
            help="AWS snapshot name minus '-root-template' suffix",
            )
    args=parser.parse_args()

    from dtk.aws_api import VolumeUpgrade

    vu = VolumeUpgrade(instance_name=args.instance,target=args.target)
    vu.no_detach_ok = args.no_detach_ok
    vu.no_ssh = args.no_ssh
    if args.reset:
        vu.reset()
    else:
        vu.run()
