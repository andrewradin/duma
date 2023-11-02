#!/usr/bin/env python

from dtk.files import get_file_records

def load_v2g(fn):
    out = []
    for rec in get_file_records(fn, keep_header=False, progress=True):
        # chr:pos
        key = rec[0]
        prot = rec[4]
        if prot != '-':
            out.append((key, prot))
    
    from dtk.data import MultiMap
    return MultiMap(out).fwd_map()

def load_v2d(fn):
    out = []
    for rec in get_file_records(fn, progress=True):
        # study name
        key = rec[0]
        chrm = rec[2]
        pos = rec[3]
        out.append((key, f'{chrm}:{pos}'))

    from dtk.data import MultiMap
    return MultiMap(out).fwd_map()


def run(v2g, v2d, output_summary):
    v2g_map = load_v2g(v2g)
    v2d_map = load_v2d(v2d)

    out = []
    from tqdm import tqdm
    for study, variants in tqdm(v2d_map.items()):
        num_variants = len(variants)
        prots = set()
        for variant in variants:
            prots.update(v2g_map.get(variant, []))
        
        out.append((study, num_variants, len(prots)))
        

    from dtk.tsv_alt import SqliteSv
    SqliteSv.write_from_data(output_summary, out, types=[str, int, int], header=['study', 'num_variants', 'num_prots'], index=['study']) 


if __name__=='__main__':
    import argparse
    arguments = argparse.ArgumentParser(description="Summarize d2g data")
    arguments.add_argument("--v2g", help="variant to gene data")
    arguments.add_argument("--v2d", help="variant to disease data")
    arguments.add_argument("--output-summary", help="Output file (sqlite)")
    args = arguments.parse_args()
    from dtk.log_setup import setupLogging
    setupLogging()
    
    run(**vars(args))


