#!/usr/bin/env python3

import sys
import os

from dtk.files import get_file_records
class RandomAccessDPI:
    def __init__(self,fn):
        self.fh = open(fn)
        self.build_index()
    def build_index(self):
        self.index = {}
        pos = 0
        self.fh.seek(pos)
        last_key=None
        for line in self.fh:
            key = line.split('\t')[0]
            if key != last_key:
                if last_key is not None:
                    if key in self.index:
                        raise RuntimeError(key+' data not contiguous')
                    self.index[key] = pos
                last_key=key
            pos += len(line)
    def get(self,key):
        start = self.index[key]
        self.fh.seek(start)
        for line in self.fh:
            rec = line.strip('\n').split('\t')
            if rec[0] != key:
                return
            yield rec

class Merger:
    coll_order = {
            x+'_id':i
            for i,x in enumerate((
                    'duma',
                    'drugbank',
                    'ncats',
                    'chembl',
                    'bindingdb',
                    ))
            }
    def __init__(self,clust_fn):
        self.cluster_list = [
                rec
                for rec in get_file_records(
                        clust_fn,
                        keep_header=None,
                        parse_type='tsv',
                        )
                ]
        print('loaded',len(self.cluster_list),'clusters')
        self.inputs={}
        self.show_disjoint=False
        self.falsehoods={}
        self.output_all_keys = False
        self.base_dpi={}
    def load_falsehoods(self,filename):
        active=None
        for rec in get_file_records(filename,keep_header=None):
            if rec[0]:
                active = self.falsehoods.setdefault(rec[0],(set(),dict()))
            if len(rec) == 4:
                # save a correction (uniprot, ev, direction)
                active[1][rec[1]] = (float(rec[2]),int(rec[3]))
            elif len(rec) == 2:
                # save a removal (uniprot only)
                active[0].add(rec[1])
            else:
                raise RuntimeError('bad falsehood format')
    def set_output(self,outfile):
        from dtk.files import FileDestination
        self.dest = FileDestination(outfile,header=[
                        'dpimerge_id',
                        'uniprot_id',
                        'evidence',
                        'direction',
                        ])
    def set_input(self,keyname,in_fn):
        rad = RandomAccessDPI(in_fn)
        print('loaded',len(rad.index),'keys for',keyname,'from',in_fn)
        if keyname not in self.inputs:
            self.inputs[keyname] = []
        self.inputs[keyname].append(rad)
    def merge(self):
        # first output a merged entry for each cluster
        from dtk.data import MultiMap
        from dtk.prot_map import MoleculeKeySet
        for rec in self.cluster_list:
            mks = MoleculeKeySet(rec)
            mm = self.assemble_cluster(mks)
            if mm:
                self.output_cluster(mks.best_key(),mm)
        # get a list of all keys that are in clusters, grouped by collection
        from dtk.data import MultiMap
        from dtk.drug_clusters import assemble_pairs
        clust_keys = MultiMap(
                (coll,key)
                for rec in self.cluster_list
                for coll,key in assemble_pairs(rec)
                if coll in self.inputs
                )
        # output drugs remaining in collection
        for coll in self.inputs:
            self.copy_collection(coll,clust_keys.fwd_map().get(coll,set()))
        # force a new group of set_output/set_input calls before next merge
        self.inputs={}
        self.dest.close()
        self.dest = None
    def get_coll_dpis(self,coll,key):
        '''Return a MultiMap from uniprot ids to (ev,dir) pairs.

        MultiMap holds all dpi data for a given drug and collection,
        possibly spanning multiple flavors of DPI files.
        '''
        from dtk.data import MultiMap
        result = MultiMap([])
        try:
            rads = self.inputs[coll]
        except KeyError:
            return result
        for rad in rads:
            try:
                dpis = list(rad.get(key))
            except KeyError:
                pass
            else:
                result.union(MultiMap(
                        (rec[1],(float(rec[2]),int(rec[3])))
                        for rec in dpis
                        ))
        return result
    def assemble_cluster(self,mks):
        '''Given a cluster definition, return DPI info.

        Return value is a MultiMap mapping from uniprot ids to (ev,dir) pairs
        '''
        from dtk.data import MultiMap
        mm = MultiMap([])
        from collections import defaultdict
        source2uniprots = defaultdict(set)
        # first check if a duma drug overrides consensus DPI
        coll = mks.priority_dpi_coll
        candidates = mks.keys(coll)
        if candidates:
            key = mks.priority_duma_id(candidates)
            return self.get_coll_dpis(coll,key)
        # no, construct consensus DPI instead
        for coll in mks.collections():
            for key in mks.keys(coll):
                coll_data = self.get_coll_dpis(coll,key)
                mm.union(coll_data)
                source2uniprots[(coll,key)] |= set(coll_data.fwd_map().keys())
        # issue a warning if a key's dpis are completely disjoint;
        # this may indicate a clustering error
        if self.show_disjoint and len(source2uniprots) > 1:
            disjoint = set()
            for k in source2uniprots:
                mine = source2uniprots[k]
                others = set()
                for k2 in source2uniprots:
                    if k != k2:
                        others |= source2uniprots[k2]
                if mine and others and not mine & others:
                    disjoint.add(k)
            if disjoint:
                print(disjoint,source2uniprots)
        if mm.fwd_map():
            return mm
        return None
    def correct_dpi(self,outkey,uniprot_list):
        removals,corrections = self.falsehoods[outkey]
        d = {
                uniprot:(ev,direction)
                for uniprot,ev,direction in uniprot_list
                }
        d.update(corrections)
        for uniprot in removals:
            d.pop(uniprot,None)
        return [
                (uniprot,ev,direction)
                for uniprot,(ev,direction) in d.items()
                ]
    def filter_and_output_one_drug(self,coll_and_key,uniprot_list):
        out_key = coll_and_key[1]
        if out_key in self.falsehoods:
            uniprot_list = self.correct_dpi(out_key,uniprot_list)
        for uniprot,ev,direction in uniprot_list:
            self.dest.append((
                out_key,
                uniprot,
                ev,
                direction,
                ))
    def output_cluster(self,best_key,mm):
        uniprot_dict = {
                uniprot:list(s)
                for uniprot,s in mm.fwd_map().items()
                }
        for uniprot,ev_dir in self.base_dpi.items():
            uniprot_dict.setdefault(uniprot,[]).append(ev_dir)
        uniprot_list = []
        for uniprot,s in uniprot_dict.items():
            ev,direction = combine_redundant_dpi(s)
            uniprot_list.append((uniprot,ev,direction))
        self.filter_and_output_one_drug(best_key,uniprot_list)
    def copy_collection(self,coll,exclude):
        rads = self.inputs[coll]
        all_keys = set([k for rad in rads
                          for k in rad.index
                          if k not in exclude
                       ])
        for key in all_keys:
            uniprot_dict={}
            for rad in rads:
                # not all keys will be in both recs
                try:
                    for rec in rad.get(key):
                        if rec[1] not in uniprot_dict:
                            uniprot_dict[rec[1]] = []
                        uniprot_dict[rec[1]].append((rec[2],rec[3]))
                except KeyError:
                    pass
            for uniprot,ev_dir in self.base_dpi.items():
                uniprot_dict.setdefault(uniprot,[]).append(ev_dir)
            uniprot_list = []
            for k,l in uniprot_dict.items():
# if we have multiple evidence values from a single collection,
# which would arise from, e.g. ki and c50 data from the same collection
# combine those values appropriately
# otherwise go with the one set of values, but get them in the appropriate format
                ev,dir = combine_redundant_dpi(l)
                uniprot_list.append((k,ev,dir))
            self.filter_and_output_one_drug((coll,key),uniprot_list)

def combine_redundant_dpi(s):
    from numpy import sign
    ev = max(float(x[0]) for x in s)
    dirs = [int(x[1]) for x in s]
# take the consensus direction
    direction = str(sign(sum(dirs)))
    return ev,direction

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='create a merged dpi file')
    parser.add_argument('--clusters')
    parser.add_argument('--combos',action='store_true')
    parser.add_argument('--falsehoods')
    parser.add_argument('--show-disjoint',action='store_true')
    parser.add_argument('file',nargs='+',
        help="multiple sets of <out> <inf> <inf> ... separated by '+'")
    args = parser.parse_args()

    if args.combos:
        # This is a special reprocessing mode that passes through a
        # previously-merged file and adds in a base drug. For simplicity
        # in the Makefile, this produce a separate output for each arg.
        assert not args.clusters
        assert not args.falsehoods
        assert not args.show_disjoint
        from dtk.files import VersionedFileName
        from dtk.combo import base_dpi_data
        for out_fn in args.file:
            if not out_fn.endswith('.tsv'):
                continue
            vfn = VersionedFileName(file_class='matching',name=out_fn)
            src_dpi,base_drug = vfn.flavor.split('+')
            mgr = Merger('/dev/null')
            mgr.base_dpi = base_dpi_data(base_drug)
            mgr.set_output(out_fn)
            vfn.flavor = src_dpi
            vfn.format = 'tsv' # Ensure we're not using the sqlsv
            in_fn = vfn.to_string()
            mgr.set_input('dpimerge_id',in_fn)
            mgr.merge()
            from dtk.tsv_alt import SqliteSv
            assert out_fn.endswith('.tsv'), f"Unexpected output without .tsv {out_fn}"
            sql_outf = out_fn.replace('.tsv', '.sqlsv')
            SqliteSv.write_from_tsv(sql_outf, out_fn, [str, str, float, int])
        sys.exit(0)

    # an empty cluster file is equivalent to no clustering (i.e. each
    # drug gets passed through independently), although the output file
    # will still say dpimerge_id in the header
    # XXX ideally, if no cluster file is supplied, the merger would check
    # XXX that all inputs are from the same collection, and output that
    # XXX collection key name in the header
    mgr = Merger(args.clusters or '/dev/null')
    if args.falsehoods:
        mgr.load_falsehoods(args.falsehoods)
    if args.show_disjoint:
        mgr.show_disjoint = True
    rest = args.file
    delim = '+'
    while rest:
        outf = rest[0]
        try:
            i = rest.index(delim)
            ins = rest[1:i]
            rest = rest[i+1:]
        except ValueError:
            ins = rest[1:]
            rest = []
        mgr.set_output(outf)
        mgr.output_all_keys = ('_allkeys' in outf)
        for inf in ins:
            # XXX This keyname assumption works for dpi files produced
            # XXX by per-collection ETL. If we need another alternative
            # XXX we could support a 'keyname:' prefix on each inf.
            parts = os.path.basename(inf).split('.')
            if parts[0] in ('dpi','c50','ki'):
                # legacy filename
                coll = parts[1]
            else:
                coll = parts[0]
            keyname = coll + '_id'
            mgr.set_input(keyname,inf)
        mgr.merge()
        from dtk.tsv_alt import SqliteSv
        assert '.tsv' in outf, f"Unexpected output without .tsv {outf}"
        sql_outf = outf.replace('.tsv', '.sqlsv')
        SqliteSv.write_from_tsv(sql_outf, outf, [str, str, float, int])
