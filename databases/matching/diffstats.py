#!/usr/bin/env python3

import sys
import os
import django_setup

def signature(l):
    l.sort()
    l=['\t'.join(x) for x in l]
    import hashlib
    m=hashlib.sha1()
    m.update(('\n'.join(l)).encode('utf8'))
    return m.digest()

def parse_1st_col_multi(fn):
    from dtk.files import get_file_records
    last_key=None
    props=[]
    for rec in get_file_records(fn,keep_header=False):
        if rec[0] != last_key:
            if props:
                yield last_key,signature(props)
            props = []
            last_key = rec[0]
        props.append(rec[1:])
    if props:
        yield last_key,signature(props)

def parse_grouped_2nd_col_multi(fn):
    from dtk.files import get_file_records
    last_key=None
    props=[]
    for rec in get_file_records(fn,keep_header=False):
        if rec[:2] != last_key:
            if props:
                yield last_key,signature(props)
            props = []
            last_key = rec[:2]
        props.append(rec[2:])
    if props:
        yield last_key,signature(props)

def diff(parser,old_path,new_path):
    old_data = {}
    try:
        for key,props in parser(old_path):
            assert key not in old_data,'duplicate keys parsing '+old_path
            old_data[key] = props
        # if the above assert fires, it probably means that the parser assumed
        # all the occurrances of a given key in old_path are contiguous, and
        # that assumption was violated
    except IOError:
        pass # handle new files (treat as if old was empty)
    from collections import Counter
    ctr = Counter()
    seen = set()
    for key,props in parser(new_path):
        assert key not in seen,'duplicate keys (e.g. %s) parsing %s'%(
                key,
                new_path,
                )
        seen.add(key)
        # see comment on assert above
        if key in old_data:
            if props == old_data[key]:
                ctr['unchanged'] += 1
            else:
                ctr['modified'] += 1
            del old_data[key]
        else:
            ctr['added'] += 1
    ctr['removed'] += len(old_data)
    return ctr

class Collection:
    _metrics = 'unchanged modified added removed'.split()
    @classmethod
    def compare_header(cls):
        return ['dataset']+cls._metrics
    def __init__(self):
        self._drugsets = []
        self._dpi = []
    def add_drugset(self,name):
        self._drugsets.append(name)
    def add_dpi(self,name):
        self._dpi.append(name)
    def get_compares(self,rows,m_files):
        from path_helper import PathHelper
        for name in sorted(self._drugsets):
            fname = 'create.%s.tsv'%name
            print('comparing',name)
            ctr = diff(
                    parse_1st_col_multi,
                    PathHelper.drugsets+fname,
                    'stage_drugsets/'+fname,
                    )
            rows.append([name]+[
                    str(ctr[x])
                    for x in self._metrics
                    ])
        for name in sorted(self._dpi):
            fname = 'dpi.%s.tsv'%name
            print('comparing',name)
            src = 'stage_dpi/'+fname+'_processed.tsv'
            if not os.path.exists(src):
                src = 'stage_dpi/c50.'+name+'.tsv_processed.tsv'
            if not os.path.exists(src):
                src = 'stage_dpi/ki.'+name+'.tsv_processed.tsv'
            ctr = diff(
                    parse_1st_col_multi,
                    PathHelper.dpi+fname,
                    src,
                    )
            rows.append(['(dpi) '+name]+[
                    str(ctr[x])
                    for x in self._metrics
                    ])
        if not m_files:
            return
        for name in sorted(self._drugsets):
            fname = 'm.%s.xref.tsv'%name
            print('comparing',name)
            ctr = diff(
                    parse_1st_col_multi,
                    PathHelper.drugsets+fname,
                    'stage_drugsets/'+fname,
                    )
            rows.append(['(m) '+name]+[
                    str(ctr[x])
                    for x in self._metrics
                    ])

def get_cluster_compares(rows):
    fname='base_drug_clusters.tsv'
    from path_helper import PathHelper
    old_path=PathHelper.drugsets+fname
    new_path='stage_drugsets/'+fname
    old_data = {}
    parser = parse_grouped_2nd_col_multi
    for (group,key),props in parser(old_path):
        d = old_data.setdefault(group,{})
        assert key not in d,'duplicate keys parsing '+old_path
        d[key] = props
    seen = set()
    from collections import Counter
    ctrs = {}
    for (group,key),props in parser(new_path):
        assert (group,key) not in seen,'duplicate keys parsing '+new_path
        ctr = ctrs.setdefault(group,Counter())
        d = old_data.setdefault(group,{})
        if key in d:
            if props == d[key]:
                ctr['unchanged'] += 1
            else:
                ctr['modified'] += 1
            del d[key]
        else:
            ctr['added'] += 1
    for group in sorted(ctrs):
        d = old_data[group]
        ctr = ctrs[group]
        ctr['removed'] += len(d)
        rows.append(['(cluster) '+group]+[
                    str(ctr[x])
                    for x in Collection._metrics
                    ])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description='show difference stats for a drug collection'
            )
    parser.add_argument('--drugsets',nargs='*')
    parser.add_argument('--dpi',nargs='*')
    parser.add_argument('--clusters',action='store_true')
    parser.add_argument('--m-files',action='store_true')
    parser.add_argument('--raw',nargs='*')
    args = parser.parse_args()

    if args.raw:
        # this is a special mode for before-after comparisons between 2
        # attributes files or evidence files, pending getting this fully
        # working with the new versioned directory layout
        ctr = diff(
                parse_1st_col_multi,
                args.raw[0],
                args.raw[1],
                )
        print(ctr)
        sys.exit(0)
    collections = {}
    for drugset in args.drugsets or []:
        coll_name = drugset.split('.')[0]
        c = collections.setdefault(coll_name,Collection())
        c.add_drugset(drugset)
    for dpi in args.dpi or []:
        coll_name = dpi.split('.')[0]
        c = collections.setdefault(coll_name,Collection())
        c.add_dpi(dpi)

    rows = [Collection.compare_header()]
    for name in sorted(collections):
        coll = collections[name]
        coll.get_compares(rows,args.m_files)
    if args.clusters:
        get_cluster_compares(rows)
    from dtk.text import print_table
    print_table(rows)

