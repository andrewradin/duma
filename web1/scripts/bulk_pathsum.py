#!/usr/bin/env python3

import django_setup
import path_helper

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='execute a bulk pathsum operation'
            )
    parser.add_argument('--cores',type=int)
    parser.add_argument('--fake',action='store_true')
    parser.add_argument('indir')
    parser.add_argument('outdir')
    args=parser.parse_args()

    from algorithms.bulk_pathsum import WorkItem
    indir = args.indir
    worklist = WorkItem.unpickle(indir,'worklist')
    context = WorkItem.unpickle(indir,'context')
    context['indir'] = indir
    context['outdir'] = args.outdir
    WorkItem.execute(args.cores,worklist,context,args.fake)

