#!/usr/bin/env python3

import sys
from dtk.files import open_pipeline

from datetime import datetime

class Date_Deduplicator:
    def __init__(self, out):
        self.out = out
        self.seen = set()
        self.last_key = None
        self.value = set()
        self.raw = []
    def flush(self):
        if self.last_key and self.value:
            self.seen.add(self.last_key) 
            if len(self.value) > 1:
                # This usually happens because one or more don't have the event date filled in.
                # Regardless, assume the earliest reported date is the true one, especially because
                # older records don't record initial fda received time, but instead latest.
                pass
            self.value = min(self.value)
            self.out.write('%s\t%s\n'%(self.last_key, self.value))
        self.value = set()
        self.raw = []
        self.last_key = None
    def add_rec(self, key, uniq_id, value):
        if key != self.last_key:
            self.flush()
            assert key not in self.seen, "key %s not contiguous"%key
            self.last_key = key
        self.value.add(value)
        self.raw.append((key, uniq_id, value))

cur_year = datetime.now().year

bad_event_year = 0

def convert_faers(rec):
    file_yr, file_qt, event_dt, fda_dt = rec

    if event_dt:
        yr = event_dt[:4]
        m = event_dt[4:6]
        if not m:
            m = fda_dt[4:6]

        if int(yr) > cur_year:
            yr = fda_dt[:4]
    else:
        yr = 0
    yr = int(yr)

    if yr and (yr < 1990 or yr > cur_year):
        # There are a lot of reports dated between 1900 and 1990, despite processing files from 2004+
        # maybe people accidentally filled in a birthday for the event date?
        # Too spammy to keep in.
        #print(f"Unexpected year from {event_dt}, falling back to fda time of {fda_dt}")
        global bad_event_year
        bad_event_year += 1
        yr = 0

    if not yr:
        yr = fda_dt[:4]
        m = fda_dt[4:6]

    yr = int(yr)
    assert yr >= 1900 and yr <= cur_year, f"Unexpected year value in {rec}"

    if m:
        qt = ((int(m)-1) // 3) + 1
    else:
        raise Exception(f"Unexpected fallback for {rec}")
        qt = file_qt


    return '%sQ%s' % (yr, qt)

m2q={
    k:1+(i/3)
    for i,k in enumerate(
            'jan feb mar apr may jun jul aug sep oct nov dec'.split()
            )
    }

def convert_cvarod(rec):
    # input is dd-mmm-yy in rec[0]; month is alpha
    from dtk.faers import ClinicalEventCounts
    base_year = ClinicalEventCounts.quarter_base_year
    # leave a 10-year margin; database goes back to the 60's,
    # but only 3 records before 1970 match drug and indi
    cutoff_decade = (base_year % 100) - 10
    parts = rec[0].split('-')
    if int(parts[2]) < cutoff_decade:
        yr = '20'+parts[2]
    else:
        yr = '19'+parts[2]
    if int(yr) < base_year:
        raise ValueError('year out of range: '+yr)
    return '%sQ%s' % (yr,m2q[parts[1]])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    description='remove duplicate records',
                    )
    parser.add_argument('--cvarod',action='store_true')
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args=parser.parse_args()

    from atomicwrites import atomic_write
    with atomic_write(args.outfile,overwrite=True) as fh:
        dd=Date_Deduplicator(fh)
        cvt = convert_cvarod if args.cvarod else convert_faers
        with open_pipeline([['sort','-n',args.infile]]) as p:
            from dtk.readtext import parse_delim
            for i, rec in enumerate(parse_delim(p)):
                dd.add_rec(rec[0], rec[1], cvt(rec[2:]))
                if i % 1000000 == 0:
                    # Expect <0.1% bad event years, with most of them in earlier data.
                    print(f"Processed {i}, with {bad_event_year} bad event years")
        dd.flush()
