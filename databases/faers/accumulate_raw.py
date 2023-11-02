#!/usr/bin/env python3

from dtk.files import scan_dir
import sys


class Extractor:
    cols = dict(
        demo=['age','age_cod',('gndr_cod','sex'),'wt','wt_cod'],
        )
    def __init__(self,mode,outfile,last_subdir,indir,case_map):
        self.parser = getattr(self,mode+'_parser')
        self.infile = 'demo' if mode == 'date' else mode
        self.outfile = outfile
        self.last_out = None
        
        if self.infile != 'demo':
            # Don't need this if we're reading the demo file.
            import pickle
            with open(case_map, 'rb') as f:
                self.case_map = pickle.load(f)
        sub_paths = sorted(list(scan_dir(args.indir)))
        import os
        subdirs = [
                os.path.basename(path)
                for path in sub_paths
                ]
        last = subdirs.index(last_subdir)
        self.paths = sub_paths[:last+1]
    def scan(self):
        from atomicwrites import atomic_write
        with atomic_write(self.outfile,overwrite=True) as fh:
            for path in self.paths:
                self.scan_file(path,fh)
    def scan_file(self,path,fh):
        date_part = path[-4:].lower()
        path += '/ascii/'+self.infile+date_part+'.txt'
        # make these accessible to date_parser
        self.file_year = '20'+date_part[:2]
        self.file_quarter = date_part[3]
        print('scanning',path)
        self.discarded = 0
        with open(path, encoding='cp1252') as inp:
            # We replace any embedded tabs with spaces, because they will break our
            # tsv output.
            lines=[x.rstrip('\r\n').replace('\t', ' ').split('$') for x in inp]
        self.parser(lines,fh)
        if self.discarded:
            print(f"discarded {self.discarded}/{len(lines)}")
    def write(self,fh,*parts):
        if parts == self.last_out:
            return
        self.last_out = parts
        fh.write('\t'.join(parts)+'\n')
    def drug_parser(self,lines,fh):
        header = [x.lower() for x in lines[0]]
        idx = header.index('drugname')
        cm = self.case_map
        for parts in lines[1:]:
            if len(parts) < len(header):
                # some drug records cross multiple lines (i.e. they
                # contain embedded newlines); there aren't enough
                # to be worth bothering about; just skip them
                self.discarded += 1
                continue
            
            caseid = cm[parts[0]]
            if not caseid:
                self.discarded += 1
                continue
            self.write(fh,
                    caseid,
                    parts[0],
                    parts[idx].lower(),
                    )
    def indi_parser(self,lines,fh):
        header = [x.lower() for x in lines[0]]
        idx = header.index('indi_pt')
        cm = self.case_map
        for parts in lines[1:]:
            caseid = cm[parts[0]]
            if not caseid:
                self.discarded += 1
                continue
            self.write(fh,
                    cm[parts[0]],
                    parts[0],
                    parts[idx].lower(),
                    )
    def demo_parser(self,lines,fh):
        header = [x.lower() for x in lines[0]]
        cols=['age','age_cod','sex','wt','wt_cod','occp_cod']
        if cols[2] not in header:
            cols[2] = 'gndr_cod'
        idxs = [1, 0]+[header.index(x) for x in cols]
        for parts in lines[1:]:
            if len(parts) < len(header) or not parts[0] or not parts[1]:
                self.discarded += 1
                continue # same issue as drug_parser
            self.write(fh,*[parts[x].lower() for x in idxs])
    def date_parser(self,lines,fh):
        header = [x.lower() for x in lines[0]]
        # Prefer the event time, then the initial fda time, then file.
        cols=['event_dt', 'init_fda_dt']
        if cols[1] not in header:
            # This isn't as good as init_fda_dt, but better than nothing.
            cols[1] = 'fda_dt'
        idxs = [header.index(x) for x in cols]
        for parts in lines[1:]:
            if len(parts) < len(lines[0]) or not parts[0] or not parts[1]:
                self.discarded += 1
                continue # same issue as drug_parser
            self.write(fh,
                    parts[1], # using the demo file, so we have case number builtin.
                    parts[0],
                    self.file_year,self.file_quarter,
                    parts[idxs[0]],
                    parts[idxs[1]],
                    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    description='accumulate FAERS input',
                    )
    parser.add_argument('--case-map', help='file mapping isr/primids to case numbers')
    parser.add_argument('indir')
    parser.add_argument('last_subdir')
    parser.add_argument('outfile')
    args=parser.parse_args()

    parts = args.outfile.split('.')
    mode = parts[2]
    ext = Extractor(mode, args.outfile, args.last_subdir, args.indir, args.case_map)
    ext.scan()
