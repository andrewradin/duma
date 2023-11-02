#!/usr/bin/env python3

import sys
from dtk.files import open_pipeline
from collections import defaultdict

def standardize_dosage(route, verbatim, amount, unit, form, freq):
    return [route or 'n/a',amount,unit,form,freq]

class Deduplicator:
    def __init__(self):
        self.seen=set()
        self.last_key = None
        self.seen_keys = set()

        self.values = {}
        from make_matrix_files import SparseMatrixBuilder

        self.indi_and_drug_mat = SparseMatrixBuilder({}, add_missing_rows=True)
        self.dose_mat = SparseMatrixBuilder({}, add_missing_rows=True)

        self.type_map = defaultdict(lambda: {'missing': 0})
    def save_type_map(self, fn):
        import json
        out = {}
        with open(fn, 'w') as f:
            for typename, valdict in self.type_map.items():
                out[typename] = list(valdict.keys())

            # Also save out the column names here.
            out['indi_drug_cols'] = list(self.indi_and_drug_mat.col_map.keys())
            out['dose_cols'] = list(self.dose_mat.col_map.keys())

            f.write(json.dumps(out, indent=2))

    def flush(self):
        if self.last_key:
            for i, ((indi, drug), dosage) in enumerate(self.values.items()): 
                indi = indi.lower()
                row_key = f'{self.last_key}-{i}'
                self.indi_and_drug_mat.add(row_key, indi, 1)
                self.indi_and_drug_mat.add(row_key, drug, 1)

                std_dosage = standardize_dosage(*dosage)
                types = ['route', 'amount', 'unit', 'form', 'freq']
                set_any = False
                for key_type, val in zip(types, std_dosage):
                    if val:
                        val = val.lower()
                        if key_type == 'amount':
                            try:
                                entry = float(val)
                            except ValueError:
                                # This gets spammy.
                                #print("Couldn't convert ", val, " to float")
                                continue
                        else:
                            key_type_map = self.type_map[key_type]
                            if val not in key_type_map:
                                idx = len(key_type_map)
                                key_type_map[val] = idx
                            entry = key_type_map[val]

                        self.dose_mat.add(row_key, key_type, entry)
                        set_any = True
                assert set_any, "Need to set something or rows won't line up"

            self.seen.add(self.last_key)
        self.values = {}
        self.add_count = 0
        self.last_key = None
    def add_rec(self,key,uniq_id,indi,drug,dosage):
        if key != self.last_key:
            self.flush()
            assert key not in self.seen,"key %s not contiguous"%key
            self.last_key = key
        assert len(dosage) == 6,"bad record at key %s, indi %s, drug %s, dosage %s"%(key,indi,drug,str(dosage))
        self.values[(indi, drug)] = dosage

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    description='remove duplicate records',
                    )
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args=parser.parse_args()

    dd=Deduplicator()
    with open_pipeline([['sort','-n',args.infile]]) as p:
        from dtk.readtext import parse_delim
        for i, rec in enumerate(parse_delim(p)):
            dd.add_rec(rec[0], rec[1], rec[2], rec[3], rec[4:])
            if i % 1000000 == 0:
                print("At ", i)
    dd.flush()
    print('processed',len(dd.seen),'keys')
    
    def save(builder, stem, dtype):
        fm = builder.get_matrix(dtype)
        from scipy.sparse import save_npz
        save_npz(stem + '_mat.npz', fm)

    save(dd.indi_and_drug_mat, args.outfile + 'indi_drug', dtype=bool)
    save(dd.dose_mat, args.outfile + 'dose', dtype=float)

    dd.save_type_map(args.outfile + 'indi_drug_dose_meta.json')




