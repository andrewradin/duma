#!/usr/bin/env python3

from dtk.files import scan_dir
import sys


"""
This generates a mapping between report ID (i.e. ISR or Primary ID) and case ID.
Many of these reports go through multiple revisions, each of which has a unique report ID.
Generally we only want to keep one copy of each case, though.

More recent FAERS datasets contain case IDs in every datafile, making it easier to do this mapping.
Earlier datasets only contained the report <-> case mapping in the demographics file, so we need to 
build the mapping explicitly first.
"""
class CaseMapper:
    def __init__(self,outfile,last_subdir,indir):
        self.infile = 'demo'
        self.outfile = outfile
        self.last_out = None
        sub_paths = sorted(list(scan_dir(args.indir)))
        import os
        subdirs = [
                os.path.basename(path)
                for path in sub_paths
                ]
        last = subdirs.index(last_subdir)
        self.paths = sub_paths[:last+1]
        self.mapping = {}
    def scan(self):
        for path in self.paths:
            self.scan_file(path)

    def scan_file(self,path):
        date_part = path[-4:].lower()
        path += '/ascii/'+self.infile+date_part+'.txt'
        print('scanning',path)
        with open(path, encoding='cp1252') as inp:
            lines=[x.rstrip('\r\n').split('$') for x in inp]
        self.demo_parser(lines)
    def demo_parser(self,lines):
        header = [x.lower() for x in lines[0]]
        for parts in lines[1:]:
            # Some embedded newlines.  We still try to handle it if we can pull
            # out the first 2 parts, which will also include some spurious entries in the map
            # but shouldn't be problematic.
            if len(parts) < 2:
                continue 
            
            self.add_map(parts[0], parts[1])
    def add_map(self, prim_id, case_id):
        self.mapping[prim_id] = case_id

    def save(self):
        from atomicwrites import atomic_write
        import pickle
        with atomic_write(self.outfile, mode='wb', overwrite=True) as f:
            pickle.dump(self.mapping, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    description='accumulate FAERS input',
                    )
    parser.add_argument('indir')
    parser.add_argument('last_subdir')
    parser.add_argument('outfile')
    args=parser.parse_args()

    parts = args.outfile.split('.')
    m = CaseMapper(args.outfile, args.last_subdir, args.indir)
    m.scan()
    m.save()
