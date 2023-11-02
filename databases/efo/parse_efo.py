#!/usr/bin/env python
import sys, time, os
from collections import defaultdict

# XXX This would be cleaner if the parse just extracted the parent
# XXX keys and wrote those to the files rather than the child names.
# XXX The child relationships could be easily reconstructed on reading.

class parseObo:
    def __init__(self, **kwargs):
        self.filename = kwargs.get('filename', None)
        self.current_term = None
        self.parents = defaultdict(list)
        self.efos = {}
        self.children = {}
    def run(self):
        from dtk.disease_efo import Disease
        with open(self.filename, "r") as infile:
            line_gen = (l.rstrip("\n") for l in infile if l.rstrip("\n"))
            for line in line_gen:
                if line == "[Term]":
                    if self.current_term:
                        if not self.current_term.id.startswith('http:'):
                            self._dump()
                    else:
                        self.current_term = Disease()
                elif self.current_term:
                    self.parse_line(line)
                elif line == '[Typedef]':
                    break # all terms come before all Typedefs
        # keep only diseases
        diseases = self.get_descendent_ids('EFO:0000408')
        to_remove = set(self.efos.keys()) - diseases
        for k in to_remove:
            del self.efos[k]
        # clean up lists
        for d in self.efos.values():
            d.synonyms = sorted(set(
                    x
                    for x in d.synonyms
                    if not x.startswith('moved to ')
                    ))
            d.children = sorted(set(d.children))
    def get_descendent_ids(self,ancestor_id):
        name2id = {x.name:x.id for x in self.efos.values()}
        before = None
        closure = set([ancestor_id])
        while closure != before:
            before = set(closure)
            for k,v in self.efos.items():
                if k in closure:
                    closure |= set((name2id[x] for x in v.children))
        return closure
    def parse_line(self, line):
        from dtk.disease_efo import Disease
        key, _, val = line.partition(':')
        if key == 'id':
            self.current_term.id = val.strip()
        elif key == 'name':
            # usually just text, but can be:
            # text {other stuff}
            self.current_term.name = val.split('{')[0].strip().lower()
        elif key == 'synonym':
            # text in quotation marks, followed by other stuff
            self.current_term.synonyms.append(val.split('"')[1].lower())
        elif key == 'is_a':
            # an EFO id, separated from other stuff by a space
            p_id = val.strip().split(' ')[0]
            self.current_term.parents.append(p_id)
            try:
                self.efos[p_id].children += [self.current_term.name.lower()]
            except KeyError:
                self.efos[p_id] = Disease()
                self.efos[p_id].children += [self.current_term.name.lower()]
    def _dump(self):
        from dtk.disease_efo import Disease
        id = str(self.current_term.id)
# came across at least one where there was no name provided
        if not self.current_term.name:
            self.current_term.name = self.current_term.id
        if not self.current_term.name.startswith('obsolete'):
            try:
                prev_child = self.efos[id].children
                self.efos[id] = self.current_term
                self.efos[id].children = prev_child
            except KeyError:
                self.efos[id] = self.current_term
        self.current_term = Disease()
    def report_all_terms(self,ofile):
        with open(ofile, 'w') as f:
            for v in self.efos.values():
                if not v.id or v.id.startswith('http:'):
                    continue
                syns = "; ".join(v.synonyms) if v.synonyms else ''
                childs = "; ".join(v.children) if v.children else ''
                f.write("\t".join([v.id,v.name,
                                   syns,
                                   childs
                                  ]) + "\n")


if __name__ == '__main__':
    import argparse
    import pickle

    arguments = argparse.ArgumentParser(description="Parses AE .obo file to find synonyms and child terms")
    arguments.add_argument("file", help="obo file path")
    arguments.add_argument('out', help="output name")
    arguments.add_argument('terms', help="output name for all disease terms")
    args = arguments.parse_args()

    efos = parseObo(filename = args.file, all_terms_out=args.terms)
    efos.run()
    pickle.dump(efos.efos,open(args.out,'wb'))
    efos.report_all_terms(args.terms)
