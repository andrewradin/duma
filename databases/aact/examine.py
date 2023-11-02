#!/usr/bin/env python

def examine(fn,show,rec_limit,width_limit=30):
    header = None
    shown = 0
    from dtk.parse_aact import aact_file_records
    for fields in aact_file_records(fn):
        if not header:
            header = fields
            continue
        for label,value in zip(header,fields):
            if show and label not in show:
                continue
            if width_limit and len(value) > width_limit:
                value = value[:width_limit]+'...'
            print(label,value.replace('\n','\\n'))
        shown += 1
        if rec_limit and shown >= rec_limit:
            break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='parse aact data')
    parser.add_argument('--file',default='clinical_study.txt')
    parser.add_argument('--limit',type=int,default=1)
    parser.add_argument('--width',type=int,default=30)
    parser.add_argument('field',nargs='*')
    args = parser.parse_args()

    examine(args.file,args.field,rec_limit=args.limit,width_limit=args.width)
