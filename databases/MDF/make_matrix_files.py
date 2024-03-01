#!/usr/bin/env python

try:
    from dtk.files import open_pipeline
except ImportError:
    import sys
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
    from dtk.files import open_pipeline

from dtk.features import SparseMatrixBuilder,ExpandingIndex

class PathManager:
    in_path = '%sFrom%s.%s.txt'
    fm_path = 'clin_ev.%s.%s_mat.npz'
    col_path = 'clin_ev.%s.%s_cols.txt'
    def __init__(self,cds):
        self.cds = cds
    def _in(self,label,dedup='dedup'):
        return self.in_path%(label,self.cds,dedup)
    def indi_in(self): return self._in('diseases')
    def drug_in(self): return self._in('drugs')
    def demo_in(self): return self._in('demo')
    def date_in(self): return self._in('date')
    def _fm(self,label):
        return self.fm_path%('FAERS+'+self.cds,label)
    def indi_fm(self): return self._fm('indi')
    def drug_fm(self): return self._fm('drug')
    def demo_fm(self): return self._fm('demo')
    def date_fm(self): return self._fm('date')
    def _col(self,label):
        return self.col_path%('FAERS+'+self.cds,label)
    def indi_col(self): return self._col('indi')
    def drug_col(self): return self._col('drug')
    def _base_col(self,label):
        return '../faers/'+self.col_path%('FAERS',label)
    def base_indi_col(self): return self._base_col('indi')
    def base_drug_col(self): return self._base_col('drug')

def get_events(fn):
    from dtk.readtext import parse_delim
    with open(fn) as f:
        result = set()
        for event,_ in parse_delim(f):
            result.add(event)
    return result

def save_bool_matrix(row_map,col_path,in_fn,out_fm,out_col):
    # These matrices will be appended to the standard FAERS drug and indi
    # matrices, so they need to start with all the existing columns in
    # the FAERS matrices, and add to them as necessary. So we pre-load
    # the column map into the SparseMatrixBuilder.
    from dtk.readtext import parse_delim
    col_map=ExpandingIndex()
    with open(col_path) as col_f:
        for fields in parse_delim(col_f):
            col_map[fields[0]]
    b = SparseMatrixBuilder(row_map,col_map)
    with open(in_fn) as in_f:
        for event,col in parse_delim(in_f):
            b.add(event,col,1)
    print len(b.row_map),'rows',len(b.col_map),'cols',len(b.vals),'values'
    fm = b.get_matrix(bool)
    from scipy import sparse
    sparse.save_npz(out_fm, fm)
    with open(out_col,'w') as f:
        for name in b.col_names():
            f.write(name+'\n')

def save_demo_matrix(row_map,in_fn,out_fm):
    b = SparseMatrixBuilder(row_map)
    from dtk.faers import ClinicalEventCounts
# TODO ClinicalEventCounts.demo_cols do not reflect what the columns are for XXX
# XXX also reports race, which is the next column, but we're ignoring that for now
    b.col_map = {
            name:i
            for i,name in enumerate(ClinicalEventCounts.demo_cols)
            }
    from dtk.readtext import parse_delim
    null = '\N'
    def int_encode(strval,cutoff):
        if strval == null:
            return None
        v = int(float(strval))
        if v > cutoff:
            return None
        return 1 + v
    sex_code = {null:None,'m':1,'f':2}
    with open(in_fn) as in_f:
        for event,age_yr,wt_kg,sex,race in parse_delim(in_f):
            event = event
            for label,code in [
### Note we are effectively cleaning up data here by requiring that they are under the value noted
                        ('age_yr',int_encode(age_yr,120)),
                        ('wt_kg',int_encode(wt_kg,300)),
                        ('sex',sex_code[sex]),
                        ]:
                if code:
                    b.add(event,label,code)
    print len(b.row_map),'rows',len(b.vals),'values'
    fm = b.get_matrix(None)
    print fm.dtype,fm.shape
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
            event = event
            b.add(event,'date',yrq_encode(yrq))
    print len(b.row_map),'rows',len(b.vals),'values'
    print len(b.row_map),'rows',len(b.vals),'values'
    fm = b.get_matrix(None)
    print fm.dtype,fm.shape
    from scipy import sparse
    sparse.save_npz(out_fm, fm)

def make_fvs(cds):
    from dtk.timer import Timer
    tmr = Timer()
    pm=PathManager(cds)
    base_pm=PathManager('faers')
    # find events present for both drugs and disease, and
    # make a single event/col map
    indi_ev_set = get_events(pm.indi_in())
    print 'indi ev load took',tmr.lap(),'got',len(indi_ev_set)
    drug_ev_set = get_events(pm.drug_in())
    print 'drug ev load took',tmr.lap(),'got',len(drug_ev_set)
    #date_ev_set = get_events(pm.date_in())
    #print 'date ev load took',tmr.lap(),'got',len(date_ev_set)
    common_evs = list(indi_ev_set & drug_ev_set)
    print len(common_evs),'common events'
    common_evs.sort()
    ev_map = {x:i for i,x in enumerate(common_evs)}
    print 'map construction took',tmr.lap()
    # build and save fvs for drug and disease, skipping any events
    # not in map, and not saving a key column
    save_bool_matrix(ev_map,pm.base_indi_col(),
            pm.indi_in(),pm.indi_fm(),pm.indi_col(),
            )
    print 'indi fm took',tmr.lap()
    save_bool_matrix(ev_map,pm.base_drug_col(),
            pm.drug_in(),pm.drug_fm(),pm.drug_col(),
            )
    print 'drug fm took',tmr.lap()
    # build and save demo fv
    save_demo_matrix(ev_map,pm.demo_in(),pm.demo_fm())
    print 'demo fm took',tmr.lap()
    # build and save date fv
    #save_date_matrix(ev_map,pm.date_in(),pm.date_fm())
    #print 'date fm took',tmr.lap()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    description='create sparse feature matrix for FAERS',
                    )
    parser.add_argument('dataset')
    args=parser.parse_args()

    make_fvs(args.dataset)
