#!/usr/bin/env python3

import sys
from dtk.files import get_file_records

if __name__=='__main__':
    import argparse
    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="Parse Monarch evidence codes")
    arguments.add_argument("dis_file", help="input file for the disease to pheno data")
    arguments.add_argument("gen_file", help="input file for the genes to phenotypes")
    arguments.add_argument("evid_out", help="output file for the evidence types")
    args = arguments.parse_args()

    evidence_types={}
    save_for_later = []
    for fh in [args.dis_file, args.gen_file]:
        RowType=None
        for fields in get_file_records (fh):
            if not RowType:
                from collections import namedtuple
                RowType=namedtuple('Monarch',fields)
                continue
            rec = RowType(*fields)
            # for some awful reason the order of the codes and the labels are not assured of being the same
            # as a result we don't want to set a code-label relationship off a piped code
            all_codes = rec.evidence.split('|')
            all_labels = rec.evidence_label.split('|')
            assert len(all_codes) == len(all_labels)
            paired = zip(all_codes, all_labels)
            if len(all_codes) > 1:
                save_for_later.append(paired)
                continue
            for c,l in paired:
                if c not in evidence_types:
                    evidence_types[c] = l
                if l != evidence_types[c]:
                    print("\n".join([
                        f'code: {c}',
                        f'label: {l}',
                        f'previously seen label: {evidence_types[c]}',
                        f'Full code: {rec.evidence}',
                        f'Full label: {rec.evidence_label}'
                    ]))
                    assert False, 'Data consistency error'
    # now make sure that the evidences with multiple citations don't include a never before seen code
    skipped_codes=set()
    skipped_labels=set()
    for tup in save_for_later:
        new_codes=[]
        new_labels=[]
        for c,l in tup:
            if c not in evidence_types:
                new_codes.append(c)
            if l not in evidence_types.values():
                new_labels.append(l)
        if not new_codes and not new_labels:
            continue
        assert len(new_codes) == len(new_labels)
        if len(new_codes) > 1:
            skipped_codes.update(set(new_codes))
            skipped_labels.update(set(new_labels))
        evidence_types[new_codes[0]] = new_labels[0]

    assert len(skipped_codes) == 0, f'Still undocumented codes {skipped_codes}'

    with open(args.evid_out, 'w') as f:
        f.write("\t".join(['evidence_code', 'evidence_label'])+"\n")
        for k,v in evidence_types.items():
            if k and v:
                f.write("\t".join([k,v])+"\n")
