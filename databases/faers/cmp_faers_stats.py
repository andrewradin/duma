#!/usr/bin/env python

import sys
try:
    import path_helper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")

import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")

import django
django.setup()

def human_readable(value):
    units=' KMGTP'
    idx = 0
    div = 1000
    while value > 10*div:
        value /= div
        idx += 1
    return ("%d%s" % (value,units[idx])).strip()

def get_pct(other,mine):
    if mine:
        pct = int((100*(other-mine))/mine)
        if pct > 0:
            pct = '+%d' % pct
        elif pct == 0:
            pct = '+/-0'
        else:
            pct = str(pct)
    elif other:
        pct = '***'
    else:
        pct = '+/-0'
    return pct

def show_cmp(label,other,mine):
    pct = get_pct(other,mine)
    print '%s vs %s %s (%s%%)' % (
            human_readable(other),
            human_readable(mine),
            label,
            pct,
            )

def show_cmpf(label,other,mine):
    pct = get_pct(other,mine)
    from tools import sci_fmt
    print '%s vs %s %s (%s%%)' % (
            sci_fmt(other),
            sci_fmt(mine),
            label,
            pct,
            )

class DbStats:
    def __init__(self,key):
        self.key = key
        self.events = []
        self.drugs = []
        self.indis = []
        for line in open('history/%s.db_stats'%key):
            if 'FromFAER distinct event' in line:
                self.events.append(int(line.split()[-1]))
            if 'FromFAER distinct drug' in line:
                self.total_drugs = int(line.split()[-1])
            if 'FromFAER distinct indi' in line:
                self.total_indis = int(line.split()[-1])
            if 'drugs have 1000' in line:
                self.drugs.append(int(line.split()[0]))
            if 'indis have 1000' in line:
                self.indis.append(int(line.split()[0]))
        self.events = min(self.events)
        self.drugs = sum(self.drugs)
        self.indis = sum(self.indis)
    def compare(self,other):
        show_cmp('events',other.events,self.events)
        show_cmp('distinct drugs',other.total_drugs,self.total_drugs)
        show_cmp('drugs with over 1000 events',other.drugs,self.drugs)
        show_cmp('distinct indications',other.total_indis,self.total_indis)
        show_cmp('indications with over 1000 events',other.indis,self.indis)

class WsStats:
    @classmethod
    def sorted_keys(cls,keys):
        return sorted(keys,key=lambda x:int(x[6:]))
    @classmethod
    def build_set(cls,key):
        result = {}
        from dtk.files import get_file_records
        for rec in get_file_records(
                    'history/%s.faers_ws_eval.tsv'%key,
                    keep_header=False,
                    ):
            result[rec[0]] = cls(rec)
        return result
    def __init__(self,rec):
        self.name = rec[0]
        self.indi_cnt = int(rec[2])
        self.cas_cnt = int(rec[4])
        self.dcoe = float(rec[5])
        self.dcop = float(rec[6])
        self.raw = float(rec[7])
    def compare(self,other):
        assert self.name==other.name
        print '-----',self.name
        show_cmp('events',other.indi_cnt,self.indi_cnt)
        #show_cmp('cas',other.cas_cnt,self.cas_cnt)
        show_cmpf('dcoe eval',other.dcoe,self.dcoe)
        show_cmpf('dcop eval',other.dcop,self.dcop)
        show_cmpf('raw eval',other.raw,self.raw)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                description='FAERS data stats',
                )
    parser.add_argument('old')
    parser.add_argument('new')
    args = parser.parse_args()

    old = DbStats(args.old)
    new = DbStats(args.new)
    old.compare(new)

    old = WsStats.build_set(args.old)
    new = WsStats.build_set(args.new)
    for key in WsStats.sorted_keys(old):
        if key in new:
            old[key].compare(new[key])

