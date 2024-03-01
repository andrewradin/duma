#!/usr/bin/env python

import collections
import re

class DrugNameMapper:
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
        self.dropped_use_count = collections.Counter()
        self.tracefile = open("name_map_output.txt","w")
        #self.build_from_base_clusters()
        self.build_from_create_files()
        # re-write common FAERS variants to a base format
        self.regex_list = [
                ('period',re.compile(r'(.*)\.$')),
                ('with',re.compile(r'(.*) w$')),
                ('tablets',re.compile(r'(.*) tablets\b')),
                ('tablets',re.compile(r'(.*) tab\b')),
                ('mg',re.compile(r'(.*) [0-9]+ mg\b')),
                ('hcl',re.compile(r'(.*) hcl\b')),
                ('hcl',re.compile(r'(.*) hydrochloride\b')),
                ('hbr',re.compile(r'(.*) hbr\b')),
                ('hbr',re.compile(r'(.*) hydrobromide\b')),
                ('besylate',re.compile(r'(.*) besylate\b')),
                ('tartrate',re.compile(r'(.*) tartrate\b')),
                ('tartrate',re.compile(r'(.*) bitartrate\b')),
                ('sodium',re.compile(r'(.*) sodium\b')),
                ('calcium',re.compile(r'(.*) calcium\b')),
                ('succinate',re.compile(r'(.*) succinate\b')),
                ('acetate',re.compile(r'(.*) acetate\b')),
                ('fumarate',re.compile(r'(.*) fumarate\b')),
                ('carbonate',re.compile(r'(.*) carbonate\b')),
                ('potassium',re.compile(r'(.*) potassium\b')),
                ('potassium',re.compile(r'(.*) dipotassium\b')),
                ('depot',re.compile(r'(.*) depot\b')),
                ('xl',re.compile(r'(.*) xl\b')),
                ('xr',re.compile(r'(.*) xr\b')),
                ('srt',re.compile(r'(.*) srt\b')),
                ('lar',re.compile(r'(.*) lar\b')),
                ('hfa',re.compile(r'(.*) hfa\b')),
                ('cq',re.compile(r'(.*) cq\b')),
                ('fentanyl',re.compile(r'.*\b(fentanyl)\b')),
                ('supplement',re.compile(r'(.*) supplement\b')),
                ('intravenous',re.compile(r'(.*) intravenous\b')),
                ]
        self.substitution_list = [
                ('advair','advair 125'),
                ('combivent','aerovent'),
                ('insulin','insulin glargine'),
                ('solostar','insulin glargine'),
                ('humulin','insulin, isophane'),
                ('duragesic','duragesic'),
                ('risperdal','risperdal'),
                ('botox','botox'),
                ('plan b one-step','levonorgestrel'),
                ]
    def build_from_create_files(self):
        use = ('drugbank','chembl')
        from path_helper import PathHelper
        from dtk.drug_clusters import FactChecker
        self.fact_checker = FactChecker(
                PathHelper.repos_root+'databases/matching/falsehoods.tsv'
                )
        for fn in ('create.%s.full.tsv'%x for x in use):
            self.load_one_create_file(PathHelper.storage+"drugsets/"+fn)
        self.post_process_cas()
        self.dump_name2cas()
    def dump_name2cas(self):
        with open("name2cas_dump.tsv","w") as f:
            for key in sorted(self.name2cas.keys()):
                f.write('\t'.join([key]+sorted(self.name2cas[key]))+'\n')
    def load_one_create_file(self,fn):
        print 'processing',fn+'...'
        drug2cas={}
        name2drug={}
        name_attrs = set([
                'canonical',
                'synonym',
                'brand',
#                'mixture',
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
        print '  ',len(name2drug),'name records'
        print '  ',len(drug2cas),'cas records'
        for name,druglist in name2drug.iteritems():
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
                    # spot checking the above logs, it seems like most of
                    # them are due to errors; since we do drugbank before TTD
                    # and TTD is usually wrong, for now don't merge in
                    # conflicts
                    # XXX ultimately, these should be reviewed and added to
                    # XXX the falsehoods file, and that file consulted here;
                    # XXX that will help with drug clustering as well as FAERS
                else:
                    existing |= cas_set
    def post_process_cas(self):
        print 'total of',len(self.name2cas),'names accumulated'
        if not self.drop_multiples:
            return # hack to try keeping all CAS codes
        for name in self.name2cas.keys():
            s = self.name2cas[name]
            if len(s) != 1:
                self.dropped_cas_count.update(s)
                self.dropped_names.add(name)
                self.tracefile.write('dropping "%s" %s\n'%(name,repr(s)))
                del(self.name2cas[name])
        print len(self.name2cas),'names remaining after removing ambiguities'
    def build_from_base_clusters(self):
        # XXX the following doesn't really do the right thing because
        # XXX it doesn't find intermediate connections
        from path_helper import PathHelper
        fn=PathHelper.storage+"drugsets/base_drug_clusters.tsv"
        drug2cas = {}
        from dtk.files import get_file_records
        from dtk.drug_clusters import assemble_pairs
        for rec in get_file_records(fn,
                        grep=['-E','^(cas|name)'],
                        ):
            if rec[0] == 'cas':
                assert not self.name2cas
                cas = rec[1]
                drug_keys = assemble_pairs(rec[2:])
                for key in drug_keys:
                    assert key not in drug2cas
                    drug2cas[key] = cas
            elif rec[0] == 'name':
                name = rec[1]
                drug_keys = assemble_pairs(rec[2:])
                cas_list = set([
                        drug2cas[key]
                        for key in drug_keys
                        if key in drug2cas
                        ])
                if cas_list:
                    if len(cas_list) != 1:
                        print name,'ambiguous:',cas_list
                    else:
                        self.name2cas[name] = cas_list.pop()
        print 'loaded mappings for',len(self.name2cas),'names'
    def variations(self,name):
        yield name,'part'
        # try matching on special common names that aren't directly listed
        # as synonyms; if the 'how' part appears anywhere in the string,
        # the entire string is replaced with the 'subst' part (thus, it
        # makes sense for both how and subst to be the same); Note that
        # we only apply these post-splitting -- otherwise they would 
        # prevent other portions of the string from matching
        for how,subst in self.substitution_list:
            if how in name:
                yield subst,how
        # try regex extraction on whole name, and each part
        for how,regex in self.regex_list:
            m = regex.match(name)
            if m:
                yield m.group(1),how
    def translate(self,name,trace=False):
        result = []
        if trace:
            print 'got',repr(name)
        try:
            # first, try the whole name
            cas_list = self.name2cas[name]
            for cas in cas_list:
                result.append( (cas,'hit') )
        except KeyError:
            if name in self.dropped_names:
                self.dropped_use_count.update([name])
            # if that doesn't work, split the name into parts based
            # on punctuation, and try each part; for mixtures, there
            # may be multiple individual drug names on a line, and
            # they may be duplicated
            parts = [x.strip(' -') for x in re.split("[)(,+/]",name)]
            reported = set()
            for part in parts:
                if trace:
                    print 'handling part',repr(part)
                for variation,how in self.variations(part):
                    if variation in self.name2cas:
                        cas_list = self.name2cas[variation]
                        for cas in cas_list:
                            if cas not in reported:
                                result.append( (cas,how) )
                        self.tracefile.write(
                            "mapped %s => %s (%s) %s\n"%(
                                            name,
                                            variation,
                                            how,
                                            ','.join(cas_list),
                                            )
                            )
                        break
                    elif variation in self.dropped_names:
                        self.dropped_use_count.update([variation])
                    elif trace:
                        print 'tried',repr(variation),repr(how)
        if not result:
            self.tracefile.write("missed %s\n"%name)
            result.append( (None,'miss') )
        if trace:
            print 'result',repr(result)
        return result
    def report_dropped_detail(self):
        for name,count in self.dropped_use_count.most_common():
            self.tracefile.write("use dropped %d %s\n"%(count,name))
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
                    description='map fuzzy drug names to drugbank ids',
                    )
    parser.add_argument('--build-only',action='store_true')
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args=parser.parse_args()

    d = DrugNameMapper()
    if args.build_only:
        sys.exit(0)
    with open(args.outfile,"w") as out:
        c = collections.Counter()
        for line in get_file_lines(args.infile):
            c.update(["lines"])
            # lines are supposed to consist of an event code and a drug name,
            # separated by a tab.  But some drug names contain embedded tabs,
            # and some records are just '\r\n'
            fields = line.strip().split('\t',1)
            if len(fields) != 2:
                print 'skipping unexpected record:',fields
                continue
            for cas,how in d.translate(fields[1]):
                if cas:
                    out.write("%s\t%s\n" % (fields[0],cas))
                c.update([how])
            if c['lines'] % 100000 == 0:
                print c
        d.report_dropped_detail()
    print c
