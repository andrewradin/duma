#!/usr/bin/env python3

from dtk.files import get_file_records
from dtk.files import get_file_lines

import collections
import re

class DrugNameMapper:
    def __init__(self,name2cas_file):
        self.name2cas = {
                rec[0]:rec[1:]
                for rec in get_file_records(name2cas_file)
                }
        self.dropped_names = set()
        self.dropped_use_count = collections.Counter()
        self.missed = collections.Counter()
        self.tracefile = open("name_map_output.txt","w")
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
                ('besylate',re.compile(r'(.*) (besylate|besilate)\b')),
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
                ('24hr',re.compile(r'(.*)\b24 ?hr\b')),
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
            print('got',repr(name))
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
                    print('handling part',repr(part))
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
                        print('tried',repr(variation),repr(how))
        if not result:
            self.tracefile.write("missed %s\n"%name)
            self.missed.update([name])
            result.append( (None,'miss') )
        if trace:
            print('result',repr(result))
        return result
    def report_dropped_detail(self):
        for name,count in self.dropped_use_count.most_common():
            self.tracefile.write("use dropped %d %s\n"%(count,name))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    description='map fuzzy drug names to drugbank ids',
                    )
    parser.add_argument('name2cas_file')
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args=parser.parse_args()

    d = DrugNameMapper(args.name2cas_file)
    from atomicwrites import atomic_write
    with atomic_write(args.outfile,overwrite=True) as out:
        c = collections.Counter()
        for line in get_file_lines(args.infile, progress=True):
            c.update(["lines"])
            # lines are supposed to consist of an event code and a drug name,
            # separated by a tab.  But some drug names contain embedded tabs,
            # and some records are just '\r\n'
            fields = line.strip().split('\t',2)
            if len(fields) != 3:
                print('skipping unexpected record:',fields)
                continue
            for cas,how in d.translate(fields[2]):
                if cas:
                    out.write("%s\t%s\t%s\n" % (fields[0],fields[1],cas))
                c.update([how])
            if c['lines'] % 1000000 == 0:
                print(c)
                top_missed = sorted(d.missed.items(), key=lambda x: -x[1])
                print(f"Top missed: {top_missed[:20]}")
        d.report_dropped_detail()
    print(c)
