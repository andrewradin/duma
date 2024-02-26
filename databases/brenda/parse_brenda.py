#!/usr/bin/env python
import sys

verbose = False
class brenda_data:
    def __init__(self, id):
        self.id=id.split()[0]
        self.syns=set()
        self.rec_name = None
        self.human_spec_number = None
    def dump(self):
        if not self.human_spec_number:
            if verbose:
                print 'No human data found', self.id
            return
        return '\n'.join([
                     '\t'.join([self.id, 'canonical', self.rec_name])
                     ] + [
                          '\t'.join([self.id, 'synonym', s])
                          for s in self.syns
                         ]
                    )
    def check_species(self,s):
        parts = s.split()
        if parts[1] == 'Homo' and parts[2] == 'sapiens':
            self.human_spec_number = parts[0].strip('#')
    def add_rec_name(self, s):
        self.rec_name = s
    def add_syns(self, s):
        if s.startswith('#'):
            parts = s.split()
            if parts[0].strip('#') == self.human_spec_number:
                s = ' '.join(parts[1:])
            else:
                return
        self.syns.add(s.split('<')[0].strip())

if __name__ == '__main__':
    import argparse

    arguments = argparse.ArgumentParser(description="Extract relevant data from the BRENDA download")
    arguments.add_argument("i", help="the raw download file")
    arguments.add_argument("o", help="the output file")
    args = arguments.parse_args()

    try:
        from dtk.files import get_file_records
    except ImportError:
        sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
        from dtk.files import get_file_records
    current = None
    with open(args.o, 'w') as f:
        f.write('\t'.join(['brenda_id', 'attribute', 'value']) + '\n')
        for frs in get_file_records(args.i, keep_header=True, parse_type='tsv'):
            if frs[0]=='ID':
                if current is not None:
                    to_report = current.dump()
                    if to_report:
                        f.write(to_report+'\n')
                current=brenda_data(frs[1])
            elif frs[0]=='PR':
                current.check_species(frs[1])
            elif frs[0]=='RN':
                current.add_rec_name(frs[1])
            elif frs[0]=='SY':
                current.add_syns(frs[1])
