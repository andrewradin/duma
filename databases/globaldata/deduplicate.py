#!/usr/bin/env python3

program_description='''\
Deduplicate collection files.
'''

from dtk.files import get_file_records
from dtk.data import MultiMap
from atomicwrites import atomic_write

def get_key_attr_pairs(in_fn,attr):
    for rec in get_file_records(in_fn,keep_header=False):
        if rec[1] == attr:
            yield (rec[0],rec[2])

def get_filtered_key_canonical_pairs(in_fn):
    import re
    # thing, action, process, condition
    # e.g. Small Molecules to Inhibit RAF1 for Unspecified Cancer
    pat1 = re.compile(r'(.*) to ([^ ]+) (.*) for (.*)')
    # thing, condition
    # e.g. Vaccine for Hepatocellular Carcinoma
    pat2 = re.compile(r'(.*) for (.*)')
    for key,name in get_key_attr_pairs(in_fn,'canonical'):
        name = name.rstrip('.') # get rid of any trailing period
        # skip if canonical name is a descriptive phrase
        if pat1.match(name):
            continue
        if pat2.match(name):
            continue
        # ok, keep this one
        yield (key,name)

def group_by_canonical(in_fn,out_fn):
    mm = MultiMap(get_filtered_key_canonical_pairs(in_fn))
    with atomic_write(out_fn,overwrite=True) as out_f:
        for name,s in mm.rev_map().items():
            out_f.write('\t'.join(sorted(s,reverse=True))+'\n')

def load_clusters(fn):
    result = {}
    for rec in get_file_records(fn,keep_header=None):
        for key in rec:
            result[key] = rec[0]
    return result

class AttrSet:
    header = 'globaldata_id\tattribute\tvalue'
    def __init__(self):
        self.keys = set()
        self.attrs = {}
    @classmethod
    def get_from_file(cls,in_fn):
        active_key = None
        active_obj = None
        for rec in get_file_records(in_fn,keep_header=False):
            if rec[0] != active_key:
                if active_obj:
                    yield active_obj
                active_obj = cls()
                active_obj.keys.add(rec[0])
                active_key = rec[0]
            if rec[1] == 'canonical':
                rec[2] = rec[2].rstrip('.')
            active_obj.attrs.setdefault(rec[1],set()).add(rec[2])
        if active_obj:
            yield active_obj
    def merge(self,drug):
        self.keys.update(drug.keys)
        for attr,s in drug.attrs.items():
            self.attrs.setdefault(attr,set()).update(s)
    def output(self,want_key,f):
        assert want_key in self.keys
        # get the canonical name
        s = self.attrs['canonical']
        assert len(s) == 1,'multiple canonical names'
        canonical = next(iter(s))
        # the synonyms supplied by GD are useless; toss them
        self.attrs['synonym'] = set()
        # but, if the canonical name contains dosage or salt suffixes,
        # strip them and add the base name as a synonym
        root = self.strip_suffixes(canonical)
        if root:
            self.attrs['synonym'].add(root)
        self.attrs['shadowed_globaldata_id'] = set(
                x for x in self.keys if x != want_key
                )
        # resolve any issues we can
        try:
            s = self.attrs['max_phase']
            if len(s) > 1:
                self.attrs['max_phase'] = str(max(int(x) for x in s))
        except KeyError:
            pass
        try:
            s = self.attrs['mol_formula']
            if len(s) > 1:
                # XXX do something smarter later
                del(self.attrs['mol_formula'])
        except KeyError:
            pass
        # validate values
        for attr,required,multi in (
                ('canonical',True,False),
                ('max_phase',False,False),
                ('mol_formula',False,False),
                ('shadowed_globaldata_id',False,True),
                # XXX add others? retrieve from prop_dictionary?
                ):
            count = len(self.attrs.get(attr,set()))
            if required:
                assert count >= 1,f'{count} {attr} values in {want_key}'
            if not multi:
                assert count <= 1,f'{count} {attr} values in {want_key}: {self.attrs}'
        attr_names = ['canonical']+sorted(x
                for x in self.attrs
                if x != 'canonical')
        for name in attr_names:
            for val in sorted(self.attrs[name]):
                f.write(f'{want_key}\t{name}\t{val}\n')

class DpiSet:
    header = 'globaldata_id\tuniprot_id\tevidence\tdirection'
    def __init__(self):
        self.keys = set()
        self.uniprots = {}
    @classmethod
    def get_from_file(cls,in_fn):
        active_key = None
        active_obj = None
        for rec in get_file_records(in_fn,keep_header=False):
            if rec[0] != active_key:
                if active_obj:
                    yield active_obj
                active_obj = cls()
                active_obj.keys.add(rec[0])
                active_key = rec[0]
            active_obj.uniprots.setdefault(rec[1],[]).append(rec[2:])
        if active_obj:
            yield active_obj
    def merge(self,drug):
        self.keys.update(drug.keys)
        for uniprot,l in drug.uniprots.items():
            self.uniprots.setdefault(uniprot,[]).extend(l)
    def output(self,want_key,f):
        # Note that want_key might not be in self.keys, because not all
        # drug records have targets
        from collections import Counter
        for name in sorted(self.uniprots):
            l = self.uniprots[name]
            most_common_dir = Counter([x[1] for x in l]).most_common()[0][0]
            max_ev = max(float(e) for e,d in l if d == most_common_dir)
            f.write(f'{want_key}\t{name}\t{max_ev}\t{most_common_dir}\n')

globaldata_keysort=lambda x:int(x[2:])

def dedup(WrapperType,in_fn,out_fn,cluster_fn):
    base_keys = load_clusters(cluster_fn)
    results = {}
    for drug in WrapperType.get_from_file(in_fn):
        assert len(drug.keys) == 1
        cur_key = next(iter(drug.keys))
        try:
            base_key = base_keys[cur_key]
            if base_key in results:
                results[base_key].merge(drug)
            else:
                results[base_key] = drug
        except KeyError:
            pass # skip any drug not in cluster file
    with atomic_write(out_fn,overwrite=True) as out_f:
        out_f.write(WrapperType.header+'\n')
        for key in sorted(results,key=globaldata_keysort):
            results[key].output(key,out_f)

def dedup_attr(in_fn,out_fn,cluster_fn):
    dosages = set([
            'ER',
            'SR',
            'ODT',
            'DR',
            'PR',
            'MR',
            'CR',
            'LA',
            'IR',
            'ODF',
            'XR',
            ])
    salts = set([
            'hydrochloride',
            'sodium',
            'acetate',
            'sulfate',
            'mesylate',
            'maleate',
            'phosphate',
            'citrate',
            'fumarate',
            'chloride',
            'bromide',
            'tartrate',
            'succinate',
            'dihydrochloride',
            'hydrobromide',
            'disodium',
            'tosylate',
            'nitrate',
            'besylate',
            'iodide',
            'carbonate',
            'salicylate',
            'propionate',
            'palmitate',
            'benzoate',
            'hydrochloride',
            'lactate',
            ])
    def strip_suffixes(self,orig):
        words = orig.split()
        if len(words) == 1:
            return
        stripped = False
        if words[-1] in dosages:
            stripped = True
            words = words[:-1]
        if len(words) > 1 and words[-1] in salts:
            stripped = True
            words = words[:-1]
        if not stripped:
            return
        return ' '.join(words)
    AttrSet.strip_suffixes = strip_suffixes
    dedup(AttrSet,in_fn,out_fn,cluster_fn)

def dedup_dpi(in_fn,out_fn,cluster_fn):
    dedup(DpiSet,in_fn,out_fn,cluster_fn)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=program_description,
            )
    parser.add_argument('--clusters')
    parser.add_argument('mode')
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    if args.mode == 'canonical':
        group_by_canonical(args.input,args.output)
    elif args.mode == 'attr':
        dedup_attr(args.input,args.output,args.clusters)
    elif args.mode == 'dpi':
        dedup_dpi(args.input,args.output,args.clusters)
    else:
        raise RuntimeError(f"unknown mode '{args.mode}'")
