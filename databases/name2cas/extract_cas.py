#!/usr/bin/env python3

import collections

class CasMap:
    def __init__(self):
        self.name2cas = {}
        # With this set to True, we only keep names that map to a single CAS
        # (and generate lots of output about what we dropped).  The evidence
        # of that output strongly suggests that essentially all names that
        # map to multiple CAS numbers are in fact combination drugs, and
        # so both CAS numbers should be mapped.
        self.drop_multiples = False
        self.dropped_names = set()
        self.dropped_cas_count = collections.Counter()
        self.tracefile = open("extract_cas.log","w")
        from path_helper import PathHelper
        from dtk.drug_clusters import FactChecker
        self.fact_checker = FactChecker(
                PathHelper.repos_root+'databases/matching/falsehoods.tsv'
                )
    def write_output(self,fn):
        with open(fn,"w") as f:
            for key in sorted(self.name2cas.keys()):
                f.write('\t'.join([key]+sorted(self.name2cas[key]))+'\n')
    def load_one_create_file(self,fn):
        print('processing',fn+'...')
        drug2cas={}
        name2drug={}
        name_attrs = set([
                'canonical',
                'synonym',
                'brand',
                'mixture',
                ])
        # The following two substances appear in Drugbank as 'combinations'
        # with lots of other drugs, but are not interesting for our purposes,
        # as all humans are constantly ingesting them, whether reported or not
        exclude = set([
                '7647-14-5', # Sodium Chloride
                '7732-18-5', # Water
                ])
        from dtk.files import get_file_records
        header=None
        for rec in get_file_records(fn):
            if header is None:
                header = rec
                continue
            if rec[1] in name_attrs:
                # convert so it matches falsehoods file
                rec[1] = 'name'
                rec[2] = rec[2].lower()
            if not self.fact_checker.check_fact(header[:1]+rec):
                continue
            if rec[1] == 'cas':
                drug2cas[rec[0]] = rec[2]
            elif rec[1] == 'name':
                name2drug.setdefault(rec[2],set()).add(rec[0])
        print('  ',len(name2drug),'name records')
        print('  ',len(drug2cas),'cas records')
        for name,druglist in name2drug.items():
            cas_set = set()
            for drug in druglist:
                try:
                    cas = drug2cas[drug]
                except KeyError:
                    continue
                if cas in exclude:
                    continue
                cas_set.add(cas)
            if cas_set:
                existing = self.name2cas.setdefault(name,set())
                if existing and not (existing & cas_set):
                    self.tracefile.write('non-overlapping CAS: "%s" %s %s\n'%(
                                    name,
                                    repr(existing),
                                    repr(cas_set),
                                    ))
                    # XXX These are likely errors. They should be reviewed
                    # XXX and added to the falsehoods file, but there are
                    # XXX several hundred. To reduce the amount of manual
                    # XXX work, we should count agreement between the various
                    # XXX datasets, and flag outliers.
                existing |= cas_set
        print('   total of',len(self.name2cas),'names accumulated')
    def post_process_cas(self):
        accum = set()
        for s in self.name2cas.values():
            accum |= s
        print('total of',len(accum),'distinct cas numbers')
        if not self.drop_multiples:
            print('not dropping multiples')
            return
        for name in self.name2cas.keys():
            s = self.name2cas[name]
            if len(s) != 1:
                self.dropped_cas_count.update(s)
                self.dropped_names.add(name)
                self.tracefile.write('dropping "%s" %s\n'%(name,repr(s)))
                del(self.name2cas[name])
        print(len(self.name2cas),'names remaining after removing ambiguities')
    def report(self):
        for name,count in self.dropped_cas_count.most_common():
            self.tracefile.write("cas dropped %d %s\n"%(count,name))

if __name__ == "__main__":
    import argparse
    try:
        from dtk.files import get_file_lines
    except ImportError:
        import sys
        sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
        from dtk.files import get_file_lines
    parser = argparse.ArgumentParser(
                    description='assemble name2cas map',
                    )
    parser.add_argument('outfile')
    parser.add_argument('infile',nargs='+')
    args=parser.parse_args()

    cm = CasMap()
    for fn in args.infile:
        cm.load_one_create_file(fn)
    cm.post_process_cas()
    cm.write_output(args.outfile)
    cm.report()
