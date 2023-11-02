#!/usr/bin/env python3

from dtk.files import scan_dir
import sys
from tqdm import tqdm
from collections import defaultdict

"""
Similar to accumulate_raw, but for grabbing indi-drug-dosage usage data.
This requires scanning both the indi and drug files together.
"""


class Extractor:
    def __init__(self,mode,outfile,last_subdir,indir,case_map):
        self.outfile = outfile
        self.last_out = None
        
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
            for path in tqdm(self.paths):
                self.scan_file(path,fh)
    def scan_file(self,path,fh):
        date_part = path[-4:].lower()

        # Scan for drugs, store the order/mapping.
        # Then scan for indis, and generate output using the mapping.

        path_drug = path + '/ascii/drug'+date_part+'.txt'
        path_indi = path + '/ascii/indi'+date_part+'.txt'

        tqdm.write(f'scanning {path}')
        self.discarded = 0
        with open(path_drug, encoding='cp1252') as inp:
            # We replace any embedded tabs with spaces, because they will break our
            # tsv output.
            lines=[x.rstrip('\r\n').replace('\t', ' ').split('$') for x in inp]
        drug_map = self.drug_parser(lines,fh)

        with open(path_indi, encoding='cp1252') as inp:
            lines=[x.rstrip('\r\n').replace('\t', ' ').split('$') for x in inp]
        self.indi_parser(lines,fh,drug_map)

        if self.discarded:
            tqdm.write(f"discarded {self.discarded}/{len(lines)}")
    def write(self,fh,*parts):
        if parts == self.last_out:
            return
        self.last_out = parts
        # case_id, prim_id, indi, drug, {6 dose fields}
        assert len(parts) == 10, f"Unexpected parts for {parts}"
        fh.write('\t'.join(parts)+'\n')
    def drug_parser(self,lines,fh):
        header = [x.lower() for x in lines[0]]
        name_idx = header.index('drugname')
        seq_idx = header.index('drug_seq')

        # Route & dose_vbm exist in all versions.
        # The remaining parsed dose fields are only in newer dumps.
        dose_fields = ['route', 'dose_vbm', 'dose_amt', 'dose_unit', 'dose_form', 'dose_freq']
        dose_idxs = [header.index(x) if x in header else None for x in dose_fields]
        cm = self.case_map

        drug_map = defaultdict(dict)
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
            
            drug_name = parts[name_idx].lower()
            seqnum = parts[seq_idx]
            dose_parts = [parts[idx] if idx is not None else '' for idx in dose_idxs]

            drug_map[parts[0]][seqnum] = [drug_name, *dose_parts]

        return drug_map

    def indi_parser(self,lines,fh,drug_map):
        header = [x.lower() for x in lines[0]]
        idx = header.index('indi_pt')
        # Newer datasets change the name
        seq_name = 'drug_seq' if 'drug_seq' in header else 'indi_drug_seq'
        seq_idx = header.index(seq_name)
        cm = self.case_map
        for parts in lines[1:]:
            caseid = cm[parts[0]]
            if not caseid:
                self.discarded += 1
                continue
            
            # Generally anything after here will fail only because we
            # skipped a record due to lack of multiline support.
            drug_inst = drug_map.get(parts[0], None)
            if not drug_inst:
                self.discarded += 1
                continue

            seqnum = parts[seq_idx]
            drug_parts = drug_inst.get(seqnum, None)
            if not drug_parts:
                self.discarded += 1
                continue

            self.write(fh,
                    caseid,
                    parts[0],
                    parts[idx].lower(),
                    *drug_parts,
                    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    description='accumulate FAERS input',
                    )
    parser.add_argument('--case-map', help='file mapping isr/primids to case numbers', required=True)
    parser.add_argument('indir')
    parser.add_argument('last_subdir')
    parser.add_argument('outfile')
    args=parser.parse_args()

    parts = args.outfile.split('.')
    mode = parts[2]
    ext = Extractor(mode, args.outfile, args.last_subdir, args.indir, args.case_map)
    ext.scan()
