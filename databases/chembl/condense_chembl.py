#!/usr/bin/env python3

import sys
try:
    from path_helper import PathHelper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
    from path_helper import PathHelper

import dtk.rdkit_2019

################################################################################
# doc clustering tools
################################################################################
class ChemblDocInfo(object):

    @staticmethod
    def required_ids():
        """IDs that will be kept even without DPI"""
        return [x.chembl_id for x in
                ch.MoleculeDictionary.select(ch.MoleculeDictionary.chembl_id
                        ).where(ch.MoleculeDictionary.max_phase > 0)
                ]
                            

    @staticmethod
    def doc2label(doc_ids):
        result = dict(
                ch.Docs.select(
                                ch.Docs.doc_id,
                                ch.Docs.chembl,
                        ).where(
                                ch.Docs.doc_id << list(doc_ids)
                        ).tuples()
                )
        return result

    @staticmethod
    def id2doc(chembl_ids):
        molregno2chembl=dict(
                    ch.MoleculeDictionary.select(
                                    ch.MoleculeDictionary.molregno,
                                    ch.MoleculeDictionary.chembl_id,
                            ).where(
                                    ch.MoleculeDictionary.chembl << chembl_ids
                            ).tuples()
                    )
        #print molregno2chembl
        molregno2doc=ch.Activities.select(
                        ch.Activities.molregno,
                        ch.Activities.doc,
                ).where(
                        ch.Activities.molregno << list(molregno2chembl.keys())
                ).tuples()
        from dtk.data import MultiMap
        return MultiMap(
                (molregno2chembl[mrn],doc)
                for mrn,doc in molregno2doc
                )

    @staticmethod
    def choose_exemplar(chembl_ids):
        # count Activities for each drug
        from collections import Counter
        ctr = Counter(
                x[0]
                for x in ch.Activities.select(
                                ch.MoleculeDictionary.chembl_id
                        ).join(
                                ch.MoleculeDictionary
                        ).where(
                                ch.MoleculeDictionary.chembl << chembl_ids
                        ).tuples()
                )
        if len(ctr) == 0:
            print(chembl_ids)
            raise NotImplementedError('No Activities')
        # get all drugs tied for highest count
        highest_count = ctr.most_common(1)[0][1]
        ties = []
        for drug,count in ctr.most_common():
            if count < highest_count:
                break
            ties.append(drug)
        # as a canonical tie-breaker, use a lexical sort
        ties.sort()
        return ties[0]

################################################################################
# main
################################################################################
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='''Condense ChEMBL Drugs\
''',
            )
    parser.add_argument('chembl_version',
            )
    parser.add_argument('dpi',
            )
    parser.add_argument('output',
            )
    args = parser.parse_args()

    import importlib
    ch = importlib.import_module(args.chembl_version+'_schema')

    from dtk.coll_condense import CollectionCondenser
    out_marker = '_condensed'
    assert out_marker in args.output
    in_attr = args.output.replace(out_marker,'')
    temp_fn='condense.tmp'
    cc = CollectionCondenser(
            dpi_fn=args.dpi,
            create_fn=in_attr,
            doc_info=ChemblDocInfo
            )
    cc.condense(
            output_fn=temp_fn,
            shadow_attr_name='shadowed_chembl_id'
            )
    import os
    os.rename(
            temp_fn,
            args.output,
            )

