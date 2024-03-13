#!/usr/bin/env python3

from dtk.aws_api import VpcControl

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='''
Create all VPC components needed to host twoXAR instances.
'''
            )
    parser.add_argument('--region-name',
            )
    parser.add_argument('--vpc-name',
            )
    parser.add_argument('--default-az',
            help='Should only need to set this when creating a new VPC.',
            )
    parser.add_argument('--cidr-block',
            help='IPV4 address range, like 172.31.0.0/16',
            )
    parser.add_argument('command',
            metavar='command',
            help='operation to perform; one of: %(choices)s',
            choices=VpcControl.get_op_list(),
            )
    args=parser.parse_args()

    ctrl = VpcControl()
    for parm in ('region_name','vpc_name','cidr_block','default_az'):
        val = getattr(args,parm)
        if val is not None:
            setattr(ctrl,parm,val)
    func = ctrl.get_op_function(args.command)
    try:
        func()
    except AttributeError as ex:
        msg = str(ex)
        if msg.startswith('VpcControl has no attribute'):
            parts = msg.split("'")
            print('Please specify --'+parts[1].replace('_','-'))
        else:
            raise
