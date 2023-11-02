#!/usr/bin/env python3

import pwd,os
user=pwd.getpwuid(os.getuid())[0]
root='/home/%s/2xar/' % user

import sys
sys.path.insert(1,root+'twoxar-demo/web1/')

class BindingDocInfo(object):
    def __init__(self, srcs_fn, raw_c50_fn, raw_ki_fn, m_file_fn):
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
        for bdb_id, attr, val in get_file_records(m_file_fn, keep_header=False):
            self._id2matching[bdb_id].add(val)
            
        
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
            return (num_matching, self._measurements[id], -self._min_str[id])
        
        return max(ids, key=order_func)

################################################################################
# main
################################################################################

from condense_chembl import ChemblCondenser

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='''Condense BindingDB Drugs''',
            )
    args = parser.parse_args()
    temp_fn='bdb_condense.tmp'

    m_file_fn = 'stage_drugsets/m.bindingdb.full.xref.tsv'


    binding_doc_info = BindingDocInfo(
            "../bindingdb/srcs.bindingdb.tsv",
            "stage_dpi/c50.bindingdb.c50.tsv",
            "stage_dpi/ki.bindingdb.ki.tsv",
            m_file_fn
            )
    cc = ChemblCondenser(
            dpi_fn='stage_dpi/dpi.dpimerge.bindingdb_ki_allkeys.tsv',
            create_fn='stage_drugsets/create.bindingdb.full.tsv',
            doc_info=binding_doc_info
            )
    cc.condense(
            output_fn=temp_fn,
            shadow_attr_name='shadowed_bindingdb_id'
            )
    os.rename(
            temp_fn,
            'stage_drugsets/create.bindingdb.full_condensed.tsv',
            )
    cc.filter_m_file(
            input_fn=m_file_fn,
            output_fn='stage_drugsets/m.bindingdb.full_condensed.xref.tsv',
            )

