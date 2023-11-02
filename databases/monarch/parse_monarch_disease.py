#!/usr/bin/env python3

import sys
from dtk.files import get_file_records

class sources():
    def __init__(self):
        from license import credit_list,ignore_list
        self.wl = [x['abbrv'] for x in credit_list]
        self.bl = [x['abbrv'] for x in ignore_list]
    def is_safe(self,source):
        for single_source in source.split('|'):
            parts=single_source.split('#')
            if len(parts) < 2:
                continue
            abbrv=parts[1]
            if abbrv in self.wl:
                return abbrv
            if abbrv not in self.bl:
                print(f'WARNING: unrecognized source: {abbrv} from {single_source} from {source}')
        return False


if __name__=='__main__':
    import argparse
    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="Parse Monarch disease to phenotypes and genes to phenotypes")
    arguments.add_argument("dis_file", help="input file for the disease to pheno data")
    arguments.add_argument("dis_out", help="output file for the disease to phenotypes")
    args = arguments.parse_args()

    Sources = sources()

    RowType=None

    with open(args.dis_out, 'w') as f:
        f.write("\t".join([
                           'mondo_id',
                           'disease',
                           'pheno_id',
                           'phenotype',
                           'relation_type',
                           'frequency',
                           'onset',
                           'evidence_code',
                           'source'
                 ])+"\n")

        for fields in get_file_records (args.dis_file):
            if not RowType:
                from collections import namedtuple
                RowType=namedtuple('Monarch',fields)
                continue
            rec = RowType(*fields)
            source = Sources.is_safe(rec.is_defined_by)
            if not source:
                continue
            f.write("\t".join([
                           rec.subject,
                           rec.subject_label,
                           rec.object,
                           rec.object_label,
                           rec.relation_label,
                           rec.frequency_label,
                           rec.onset_label,
                           rec.evidence,
                           source
                 ])+"\n")

