#!/usr/bin/env python3

import sys
import path_helper

import os

import django
import django_setup

import argparse
from dtk.workflow import Workflow

def build_base_parser():
    parser = argparse.ArgumentParser(
                description='run a workflow',
                )
    parser.add_argument('-r','--restart-from',
            help='json file of previous run data to re-use, if possible',
            )
    parser.add_argument('--rethrow',
            help='exit on first exception',
            action='store_true',
            )
    parser.add_argument('--debug',
            help='output detailed activity tracing',
            action='store_true',
            )
    parser.add_argument('--ws-id',
            help='workspace id, if not implied by other parameters',
            type=int,
            )
    return parser

def list_workflows():
    print('Available Workflows:')
    for wrapper in Workflow.wf_list():
        print('   ',wrapper.code())

def try_workflow(wf_name):
    parser = build_base_parser()
    subs = parser.add_subparsers()
    myparser = subs.add_parser(wf_name)
    myparser.add_argument('user')
    wrapper = Workflow.get_by_code(wf_name)
    from dtk.dynaform import FieldType
    for code in wrapper.cls._fields:
        field_builder = FieldType.get_by_code(code)
        field_builder.add_to_argparse(myparser)
    # with just a workflow name, show help for that workflow (parse_args
    # will exit)
    args=parser.parse_args()
    kwargs=dict(vars(args))
    kwargs.pop('restart_from')
    if args.restart_from:
        # XXX It might be nice for this feature to work via the web UI
        # XXX as well. Some options are:
        # XXX - create the runset up front, add each job to the runset
        # XXX   as it completes, and be able to restart from an incomplete
        # XXX   runset (moving it toward completion)
        # XXX - build a restart_from dict from the default sources list,
        # XXX   or from anything on the default sources list that's marked
        # XXX   as up-to-date
        import json
        with open(args.restart_from) as f:
            kwargs['restart_from'] = json.load(f)
    print('Arguments:')
    for k in kwargs:
        print('   ',k+':',kwargs[k])
    wf = wrapper.cls(**kwargs)
    wf._checkpoint_filename = 'wf_checkpoint.json'
    wf.run_to_completion()

if __name__ == '__main__':
    parser = build_base_parser()
    parser.add_argument('wf_name',nargs='?')
    (args,others) = parser.parse_known_args()

    from runner.models import Process
    err = Process.prod_user_error()
    if err:
        raise RuntimeError(err)

    if not args.wf_name:
        # with no arguments, list available workflows
        list_workflows()
        sys.exit(0)
    
    try_workflow(args.wf_name)

