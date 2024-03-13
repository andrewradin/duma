#!/usr/bin/env python

'''
A class to hold the ZINC related functions.
For our first pass we are just downloading the ZINC labels,
though we may want to extend this over time.
'''
from __future__ import print_function

class zinc:
    def __init__(self):
        self._set_labels()
    def _set_labels(self):
        self.label_string='biogenic+in-cells endogenous fda for-sale+in-cells in-man in-trials not-for-sale+in-vitro for-sale+in-vitro in-vivo not-for-sale+in-cells world'
        self.labels= self.label_string.split()
    def get_labels(self):
        return self.labels
    def print_labels(self):
        print(self.label_string)
    def no_label_description(self):
        return 'Zinc ID but no label'
    def get_zinc_id_set_for_label(self,label,version,ids=None):
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned(
                'zinc',
                version,
                role=label,
                format='tsv.gz'
                )
        s3f.fetch()
        from dtk.files import get_file_records
        if ids is None:
            data_gen = get_file_records(s3f.path(), keep_header=False)
        else:
            data_gen = get_file_records(s3f.path(), keep_header=False,
                                        select=(ids,0))
        return set(item[0] for item in data_gen)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ZINC utility')
    parser.add_argument('--get_labels',action='store_true', help='print the ZINC labels we are interested in')
    args = parser.parse_args()

    z = zinc()
    if args.get_labels:
        z.print_labels()
