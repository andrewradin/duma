#!/usr/bin/env python
import sys
sys.path.insert(1,"../../web1")

def process(inp,outp):
    from dtk.files import FileDestination,get_file_records
    with FileDestination(outp) as dest:
        for rec in get_file_records(inp,parse_type='tsv'):
            rec[1] = rec[1].replace('&alpha;','alpha')
            assert '&' not in rec[1]
            # if the above assert fires, fix with additional replace
            # operations
            dest.append(rec)

if __name__ == '__main__':
    import argparse

    arguments = argparse.ArgumentParser(description="prepare ATC data")
    arguments.add_argument("input"),
    arguments.add_argument('output'),
    args = arguments.parse_args()
    
    process(args.input,args.output)
