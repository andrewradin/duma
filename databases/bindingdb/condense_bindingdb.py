#!/usr/bin/env python3

import dtk.rdkit_2019

################################################################################
# doc clustering tools
################################################################################
class BindingDocInfo(object):
    def __init__(self, srcs_fn, raw_c50_fn, raw_ki_fn, clust_fn):
        from dtk.data import MultiMap
        data = []
        with open(srcs_fn, 'r') as f:
            for line in f:
                parts = line.split('\t')
                id = parts[0]
                doc = '\t'.join(parts[1:])
                # For some attributes of some drugs, all we know is we
                # got them from chembl, we don't have any of the doc ids.
                # We used to err on the side of caution with these... but
                # if they came from chembl, we probably have them more
                # appropriately condensed in that import, so it should be OK
                # to be aggressive here (thus I've added the False here).
                if False and doc.strip().lower() == 'chembl':
                    doc = 'unclustered-' + id
                data.append((id, doc))
        self._id2doc = MultiMap(data)

        from collections import defaultdict
        self._id2matching = defaultdict(set)
        from dtk.files import get_file_records
        from dtk.drug_clusters import assemble_pairs
        for rec in get_file_records(clust_fn, keep_header=None):
            my_ids = []
            other_ids = []
            for pair in assemble_pairs(rec):
                if pair[0] == 'bindingdb_id':
                    my_ids.append(pair[1])
                else:
                    other_ids.append(pair[1])
            for my_id in my_ids:
                s = self._id2matching[my_id]
                for other_id in other_ids:
                    s.add(other_id)
        
        from collections import defaultdict, namedtuple
        self._measurements = defaultdict(int)
        self._min_str = defaultdict(lambda: 1e99)
        for fn in [raw_c50_fn, raw_ki_fn]:
            with open(fn, 'r') as f:
                Type = None
                for line in f:
                    parts = line.strip().split('\t')
                    if not Type:
                        Type = namedtuple('Rec', parts)
                        continue
                    data = Type(*parts) 
                    id, n = data.bindingdb_id, data.n_independent_measurements
                    self._measurements[id] += int(n)
                    self._min_str[id] = min(self._min_str[id], float(data[3]))



    def doc2label(self, doc_ids):
        return {x:x for x in doc_ids}

    def id2doc(self, ids):
        fwd = self._id2doc.fwd_map()
        kvs = []
        for id in ids:
            kvs.append((id, tuple(sorted(fwd[id]))))
        from dtk.data import MultiMap
        return MultiMap(kvs)

    def choose_exemplar(self, ids):
        def order_func(id):
            num_matching = len(self._id2matching[id])
            return (
                    num_matching,
                    self._measurements[id],
                    -self._min_str[id],
                    id, # lexical sort by id to disambiguate ties
                    )
        
        return max(ids, key=order_func)
    @staticmethod
    def required_ids():
        """IDs that will be kept even without DPI"""
        return []

################################################################################
# main
################################################################################
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='''Condense BindingDB Drugs\
''',
            )
    parser.add_argument('srcs')
    parser.add_argument('c50')
    parser.add_argument('ki')
    parser.add_argument('clust')
    parser.add_argument('dpi')
    parser.add_argument('output')
    args = parser.parse_args()

    from dtk.coll_condense import CollectionCondenser
    doc_info = BindingDocInfo(args.srcs, args.c50, args.ki, args.clust)
    out_marker = '_condensed'
    assert out_marker in args.output
    in_attr = args.output.replace(out_marker,'')
    temp_fn='condense.tmp'
    cc = CollectionCondenser(
            dpi_fn=args.dpi,
            create_fn=in_attr,
            doc_info=doc_info,
            )
    cc.condense(
            output_fn=temp_fn,
            shadow_attr_name='shadowed_bindingdb_id'
            )
    import os
    os.rename(
            temp_fn,
            args.output,
            )

