#!/usr/bin/env python3

from dtk.files import open_pipeline

class SparseMatrixBuilder:
    def __init__(self,row_map,add_missing_rows=False):
        self.row_map = row_map
        self.col_map = {}
        self.row_coord=[]
        self.col_coord=[]
        self.vals=[]
        self.skipped = 0
        self.add_missing_rows = add_missing_rows
    def add(self,row_key,col_name,val):
        try:
            row_idx = self.row_map[row_key]
        except KeyError:
            if self.add_missing_rows:
                row_idx = len(self.row_map)
                self.row_map[row_key] = row_idx
            else:
                self.skipped += 1
                return
        try:
            col_idx = self.col_map[col_name]
        except KeyError:
            col_idx = len(self.col_map)
            self.col_map[col_name] = col_idx
        self.row_coord.append(row_idx)
        self.col_coord.append(col_idx)
        self.vals.append(val)
    def get_matrix(self,dtype):
        from scipy import sparse
        return sparse.csr_matrix(
                (self.vals, (self.row_coord, self.col_coord)),
                dtype=dtype,
                )
    @staticmethod
    def map2list(d):
        return [
                x[0]
                for x in sorted(d.items(),key=lambda x:x[1])
                ]
    def col_names(self):
        return self.map2list(self.col_map)
    def row_keys(self):
        return self.map2list(self.row_map)

class PathManager:
    in_path = '%s.%s.tsv'
    fm_path = '%s.%s_mat.npz'
    col_path = '%s.%s_cols.txt'
    def __init__(self,in_stem,out_stem):
        self.in_stem = in_stem
        self.out_stem = out_stem
    def _in(self,label):
        return self.in_path%(self.in_stem,label)
    def indi_in(self): return self._in('indi')
    def drug_in(self): return self._in('drug')
    def demo_in(self): return self._in('demo')
    def date_in(self): return self._in('date')
    def _fm(self,label):
        return self.fm_path%(self.out_stem,label)
    def indi_fm(self): return self._fm('indi')
    def drug_fm(self): return self._fm('drug')
    def demo_fm(self): return self._fm('demo')
    def date_fm(self): return self._fm('date')
    def _col(self,label):
        return self.col_path%(self.out_stem,label)
    def indi_col(self): return self._col('indi')
    def drug_col(self): return self._col('drug')
        

def get_events(fn):
    from dtk.readtext import parse_delim
    with open(fn) as f:
        result = set()
        for event,_ in parse_delim(f):
            result.add(int(event))
    return result

def save_bool_matrix(row_map,in_fn,out_fm,out_col):
    b = SparseMatrixBuilder(row_map)
    from dtk.readtext import parse_delim
    with open(in_fn) as in_f:
        for event,col in parse_delim(in_f):
            b.add(int(event),col,1)
    print(len(b.row_map),'rows',len(b.col_map),'cols',len(b.vals),'values',b.skipped,'skipped')
    fm = b.get_matrix(bool)
    from scipy import sparse
    sparse.save_npz(out_fm, fm)
    with open(out_col,'w') as f:
        for name in b.col_names():
            f.write(name+'\n')

def save_demo_matrix(row_map,in_fn,out_fm):
    b = SparseMatrixBuilder(row_map)
    from dtk.faers import ClinicalEventCounts
    b.col_map = {
            name:i
            for i,name in enumerate(ClinicalEventCounts.demo_cols)
            }
    from dtk.readtext import parse_delim
    null = '\\N'
    def int_encode(strval,cutoff):
        if strval == null:
            return None
        v = int(float(strval))
        if v > cutoff:
            return None
        return 1 + v
    sex_code = {null:None,'m':1,'f':2}
    with open(in_fn) as in_f:
        for event,age_yr,wt_kg,sex,reporter in parse_delim(in_f):
            event = int(event)
            for label,code in [
                        ('age_yr',int_encode(age_yr,120)),
                        ('wt_kg',int_encode(wt_kg,300)),
                        ('sex',sex_code[sex]),
                        ('reporter',int_encode(reporter,2)),
                        ]:
                if code:
                    b.add(event,label,code)
    print(len(b.row_map),'rows',len(b.vals),'values',b.skipped,'skipped')
    fm = b.get_matrix(None)
    print(fm.dtype,fm.shape)
    from scipy import sparse
    sparse.save_npz(out_fm, fm)

def save_date_matrix(row_map,in_fn,out_fm):
    # This array isn't really sparse (it has one cell per row, all occupied),
    # but the sparse array builder easily puts the rows in the correct order,
    # and the npz file seems to be about 1/4 the size of calling np.save on
    # the dense version. Also, it makes reading the whole thing back in more
    # consistent.
    from dtk.faers import ClinicalEventCounts
    b = SparseMatrixBuilder(row_map)
    b.col_map = {ClinicalEventCounts.date_col:0}
    from dtk.readtext import parse_delim
    data = []
    import numpy as np
    def yrq_encode(strval):
        yr,q = strval.split('Q')
        return (int(yr)-ClinicalEventCounts.quarter_base_year)*4 + int(q)-1
    with open(in_fn) as in_f:
        for event,yrq in parse_delim(in_f):
            event = int(event)
            b.add(event,'date',yrq_encode(yrq))
    print(len(b.row_map),'rows',len(b.vals),'values',b.skipped,'skipped')
    fm = b.get_matrix(None)
    print(fm.dtype,fm.shape)
    from scipy import sparse
    sparse.save_npz(out_fm, fm)

def make_fvs(in_stem,out_stem):
    from dtk.timer import Timer
    tmr = Timer()
    pm=PathManager(in_stem,out_stem)
    # find events present for both drugs and disease, and
    # make a single event/col map
    indi_ev_set = get_events(pm.indi_in())
    print('indi ev load took',tmr.lap(),'got',len(indi_ev_set))
    drug_ev_set = get_events(pm.drug_in())
    print('drug ev load took',tmr.lap(),'got',len(drug_ev_set))
    date_ev_set = get_events(pm.date_in())
    print('date ev load took',tmr.lap(),'got',len(date_ev_set))
    common_evs = list(indi_ev_set & drug_ev_set & date_ev_set)
    print(len(common_evs),'common events')
    common_evs.sort()
    ev_map = {x:i for i,x in enumerate(common_evs)}
    print('map construction took',tmr.lap())
    # build and save fvs for drug and disease, skipping any events
    # not in map, and not saving a key column
    save_bool_matrix(ev_map,pm.indi_in(),pm.indi_fm(),pm.indi_col())
    print('indi fm took',tmr.lap())
    save_bool_matrix(ev_map,pm.drug_in(),pm.drug_fm(),pm.drug_col())
    print('drug fm took',tmr.lap())
    # build and save demo fv
    save_demo_matrix(ev_map,pm.demo_in(),pm.demo_fm())
    print('demo fm took',tmr.lap())
    # build and save date fv
    save_date_matrix(ev_map,pm.date_in(),pm.date_fm())
    print('date fm took',tmr.lap())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    description='create sparse feature matrix for FAERS',
                    )
    parser.add_argument('in_stem')
    parser.add_argument('out_stem')
    args=parser.parse_args()

    make_fvs(args.in_stem,args.out_stem)
