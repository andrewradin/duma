#!/usr/bin/env python3

from dtk.files import open_pipeline
import sys
from collections import defaultdict

def dec_fmt(x):
    sfmt = format(x,'.10f').rstrip('0')
    if sfmt.endswith('.'):
        sfmt += '0'
    return sfmt

class Demo_Deduplicator:
    def __init__(self, out):
        self.out = out
        self.seen = set()
        self.last_key = None
        self.values = defaultdict(list)
        self.raw = []
        self.multi = defaultdict(int)
        self.version_counts = defaultdict(int)
        self.kg_dict = {
                'kilograms':1,
                'kg':1,
                'kgs':1,
                'lbs':0.453592,
                'pounds':0.453592,
                'gms':.001,
                'oz':0.0283495,
                'ounces':0.0283495,
                }
        self.year_dict = {
                'yr':1,
                'y':1,
                'mon':0.083,
                'dec':10,
                'dy':0.002738,
                'wk':0.019166,
                'hr':0.00011408,
                }
    def summarize(self):
        print(f"Seen multiple options: {dict(self.multi)}, out of total {len(self.seen)}")
        print(f"# of followup version counts: {sorted(self.version_counts.items(), key=lambda x: -x[1])}")
    missing = '\\N'
    def flush(self):
        def get_best(key):
            options = self.values[key]
            if not options:
                return self.missing
            options = sorted(options)
            if options[0] == options[-1]:
                return options[0]

            self.multi[key] += 1
            #print("Multiple options here: ", key, options, self.raw)

            if key == 'reporter':
                # Treat as health professional if any report was by one.
                return max(options)
            elif key == 'sex':
                # This will be the most common.
                # Biases lower sort on ties, but this is extremely rare.
                return options[len(options)//2]
            else:
                # These are all numeric, take the median.
                import numpy as np
                return np.median([float(x) for x in options])
        if self.last_key and self.values:
            age = get_best('age_yr')
            weight = get_best('wt_kg')
            sex = get_best('sex')
            reporter = get_best('reporter')
            self.out.write('%s\t%s\t%s\t%s\t%s\n'%(self.last_key, age, weight, sex, reporter))
            self.seen.add(self.last_key)
            self.version_counts[len(self.raw)] += 1
        self.values = defaultdict(list)
        self.last_key = None
        self.raw = []
    def add_rec(self, key, uniq_id, value_dict):
        if key != self.last_key:
            self.flush()
            assert key not in self.seen, "key %s not contiguous"%key
            self.last_key = key

        table = str.maketrans('','',' ~,/;\-')
        clean_dict = {k:v.translate(table) for k,v in value_dict.items()}
        norm_dict = self.normalize_measures(clean_dict)
        self.raw.append((key, uniq_id, value_dict))

        for k, v in norm_dict.items():
            if v != self.missing:
                self.values[k].append(v)

    def normalize_measures(self, value_dict):
        assert len(value_dict.keys()) == 6
        tmp_dict = dict()
        if value_dict['wt'] != '' and value_dict['wt_c'] != '':
            try:
                tmp_dict['wt_kg'] = dec_fmt(
                        self.kg_dict[value_dict['wt_c']]
                                * float(value_dict['wt'])
                        )
            except KeyError:
                print('could not find weight conversion for:', value_dict['wt_c'], value_dict['wt'])
                tmp_dict['wt_kg'] = self.missing
            except ValueError:
                print('could not convert weight to float:', value_dict['wt'])
                tmp_dict['wt_kg'] = self.missing
        else:
            tmp_dict['wt_kg'] = self.missing

        if value_dict['age'] != '' and value_dict['age_c'] != '':
            try:
                tmp_dict['age_yr'] = dec_fmt(
                        self.year_dict[value_dict['age_c']]
                                *float(value_dict['age'])
                        )
            except KeyError:
                print('could not find age conversion for:', value_dict['age_c'], value_dict['age'])
                tmp_dict['age_yr'] = self.missing
            except ValueError:
                print('could not convert age to float:', value_dict['age'])
                tmp_dict['age_yr'] = self.missing
        else:
            tmp_dict['age_yr'] = self.missing

        if value_dict['sex'] not in ['m', 'f']:
            tmp_dict['sex'] = self.missing
        else:
            tmp_dict['sex'] = value_dict['sex']

        if value_dict['reporter'].lower() in ['md', 'ph', 'ot']:
            # Health professional (physician, pharmacist, other health prof)
            tmp_dict['reporter'] = 1
        elif value_dict['reporter']:
            # Lump together lawyer, consumer.
            tmp_dict['reporter'] = 0
        else:
            tmp_dict['reporter'] = self.missing

        return tmp_dict

def faers_generator(p):
    from dtk.readtext import parse_delim
    for rec in parse_delim(p):
        assert len(rec) == 8, f"Unexpected rec {rec}"
        # Multiple prim IDs per case id, each a different followup version.
        case_id = rec[0]
        prim_id = rec[1]
        rec_dict = dict(zip(['age', 'age_c', 'sex', 'wt', 'wt_c', 'reporter'], rec[2:]))
        yield case_id,prim_id,rec_dict

def cvarod_generator(p):
    from dtk.readtext import parse_delim
    sex_map={'':'','1':'m','2':'f','3':'','4':''}
    for rec in parse_delim(p):
        if len(rec) < 5:
            continue
        event = rec[0]
        rec_dict = dict(zip(['sex', 'age', 'wt', 'wt_c'], rec[1:]))
        rec_dict['age_c'] = 'yr'
        rec_dict['sex'] = sex_map[rec_dict['sex']]
        yield event,event,rec_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    description='remove duplicate records',
                    )
    parser.add_argument('--cvarod',
            action='store_true',
            help='CVAROD input (default FAERS)',
            )
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args=parser.parse_args()

    from atomicwrites import atomic_write
    with atomic_write(args.outfile,overwrite=True) as fh:
        dd=Demo_Deduplicator(fh)
        gen = cvarod_generator if args.cvarod else faers_generator
        with open_pipeline([['sort','-n',args.infile]]) as p:
            for i, (case_id, prim_id, rec_dict) in enumerate(gen(p)):
                assert case_id, f"Unexpected input - {case_id} {prim_id} {rec_dict}"
                dd.add_rec(case_id, prim_id, rec_dict)
                if i % 1000000 == 0:
                    dd.summarize()
            dd.flush()
            dd.summarize()
