#!/usr/bin/env python3

import sys
from dtk.files import open_pipeline

class Deduplicator:
    def __init__(self,out):
        self.out = out
        self.seen=set()
        self.last_key = None
        self.seen_keys = set()
        self.add_count = 0
        from collections import Counter
        self.stats = Counter()
    def flush(self):
        if self.last_key:
            diff = self.add_count - len(self.values)
            self.stats.update([diff])
            for val in self.values:
                self.out.write('%s\t%s\n'%(self.last_key,val))
            self.seen.add(self.last_key)
        self.values = []
        self.add_count = 0
        self.last_key = None
    def add_rec(self,key,uniq_id,value):
        if key != self.last_key:
            self.flush()
            assert key not in self.seen,"key %s not contiguous"%key
            self.last_key = key
        # XXX since indications don't have special pre-processing yet,
        # XXX do some cleanup here
        value=value.strip()
        if 'unknown indication' in value:
            value = 'unknown indication'
        if value not in self.values:
            self.values.append(value)
        self.add_count += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    description='remove duplicate records',
                    )
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args=parser.parse_args()

    from atomicwrites import atomic_write
    with atomic_write(args.outfile,overwrite=True) as fh:
        dd=Deduplicator(fh)
        with open_pipeline([['sort','-n',args.infile]]) as p:
            from dtk.readtext import parse_delim
            for rec in parse_delim(p):
                dd.add_rec(*rec)
        dd.flush()
    print('processed',len(dd.seen),'keys')
    for dups in sorted(dd.stats.keys()):
        print(dd.stats[dups],'keys had',dups,'duplicates')
