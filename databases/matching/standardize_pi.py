#!/usr/bin/env python3

from numpy import sign
from statistics import median

class Stats:
    def __init__(self):
        self.no_cons = 0
        self.with_cons = 0
        self.total = 0
        self.all_same_dir = 0
    def output(self):
        sys.stderr.write(str(self.total) + " total distinct interactions\n")
        sys.stderr.write(str(self.all_same_dir) + " interactions with duplicate evidence, but all in the same direction\n")
        sys.stderr.write(str(self.no_cons) + " interactions without a consensus direction - set to 0\n")
        sys.stderr.write(str(self.with_cons) + " interactions with duplicate evidence and a consensus direction\n")

class Drug:
    def __init__(self,key):
        self.drug_id = key
        self.interactions = {}
    def add(self,prot,ev,direction):
        try:
            self.interactions[prot]['direction'].append(int(direction))
        except KeyError:
            self.interactions[prot] = {
                            'direction': list(),
                            'evidence': list(),
                            }
            self.interactions[prot]['direction'].append(int(direction))
        self.interactions[prot]['evidence'].append(float(ev))
    def output(self,stats):
        for prot in sorted(self.interactions):
            detail = self.interactions[prot]
            out = [self.drug_id, prot]
            if len(set(detail['direction'])) == 1:
                # the directions agree, report the median
                out += [str(median(detail['evidence']))
                        ,str(detail['direction'][0])
                       ]
                if len(detail['direction']) > 1:
                    stats.all_same_dir += 1
            else:
                consens_dir = sign(sum(detail['direction']))
                if consens_dir == 0:
                    # there's no consensus direction, leave it at zero and report the median
                    out += [str(median(detail['evidence']))
                            ,str(consens_dir)
                           ]
                    stats.no_cons += 1
                else:
                    relevant_evids = [
                            detail['evidence'][i]
                            for i in range(len(detail['direction']))
                            if sign(detail['direction'][i]) == consens_dir
                            ]
                    out += [str(median(relevant_evids))
                            ,str(consens_dir)
                           ]
                    stats.with_cons += 1
            print("\t".join(out))
            stats.total += 1

if __name__=='__main__':
    import argparse, sys
    from collections import defaultdict
    from numpy import sign
    from statistics import median

    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="Condense duplicate drug- or protein-protein interactions")
    arguments.add_argument("--is-sorted", help="assume input is sorted", action='store_true')
    arguments.add_argument("--pre-sort", help="pre-sort input", action='store_true')
    arguments.add_argument("--edits", help="falsehoods file")
    arguments.add_argument("i", help="input file")
    args = arguments.parse_args()

    from dtk.files import get_file_records
    if args.edits:
        # Create dict of edits from falsehood file. File contains drug_id
        # and uniprot in the first two columns, and updated evidence and
        # direction in the third and fourth. If evidence and direction are
        # blank or missing, the record is ignored (skipped).
        # The grep skips comments and blank lines.
        edits={
                tuple(x[:2]):tuple(x[2:] or ['',''])
                for x in get_file_records(
                        args.edits,
                        grep=['-E','-v','^#.*|^ *$'],
                        )
                }
    else:
        edits={}
    seen = set()
    header = None
    drug = None
    drugs = {}
    stats = Stats()
    # There are two basic modes; either all records can be read and buffered
    # in the drugs hash, or the records can be read in sorted order and
    # output one drug at a time.  The latter mode is faster on very large
    # files.  You can invoke sorted mode by sorting the file externally
    # and setting the is_sorted flag, or by setting the pre_sort flag and
    # invoking on an unsorted file.
    is_sorted = args.pre_sort or args.is_sorted
    for fields in get_file_records(args.i,sort=args.pre_sort):
        if not header:
            header = fields[:4] #only keep the fields we handle
            print("\t".join(header))
            continue
        try:
            edit = edits[(fields[0],fields[1])]
        except KeyError:
            pass
        else:
            if edit == ('',''):
                continue
            fields[2:]=edit
        if is_sorted:
            if drug and fields[0] != drug.drug_id:
                drug.output(stats)
                drug = None
            if not drug:
                drug = Drug(fields[0])
                # make sure we're in sorted order
                assert fields[0] not in seen
                seen.add(fields[0])
        else:
            drug = drugs.setdefault(fields[0],Drug(fields[0]))
        drug.add(fields[1],fields[2],fields[3])
    if is_sorted:
        if drug:
            drug.output(stats)
    else:
        for drug in drugs.values():
            drug.output(stats)
    stats.output()
