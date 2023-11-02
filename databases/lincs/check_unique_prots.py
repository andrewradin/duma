#!/usr/bin/env python3

from atomicwrites import atomic_write

import logging
logger = logging.getLogger(__name__)



def make_uniprot_to_gene(uniprot_version):
    from dtk.s3_cache import S3File
    from dtk.files import get_file_records
    ver = f'HUMAN_9606.{uniprot_version}'
    uni_s3f = S3File.get_versioned('uniprot',ver,'Uniprot_data')
    uni_s3f.fetch()
    out = {}
    for rec in get_file_records(uni_s3f.path(), keep_header=False):
        if rec[1] == 'Gene_Name':
            out[rec[0]] = rec[2]
    return out


def load_dpi(fn):
    out = []
    from dtk.files import get_file_records
    for drug, prot, ev, dr in get_file_records(fn, progress=True, keep_header=False, parse_type='tsv'):
        if float(ev) < 0.5:
            continue
        out.append((drug, prot))
    from dtk.data import MultiMap
    return MultiMap(out)

def run(evidence, attributes, ref_evidence, uniprot_version):
    ev_mm = load_dpi(evidence)
    ref_ev_mm = load_dpi(ref_evidence)


    new_prots = []
    for prot in ev_mm.rev_map():
        if prot not in ref_ev_mm.rev_map():
            drugs = ev_mm.rev_map()[prot]
            new_prots.append((prot, drugs))

    fn = '/home/ubuntu/2xar/ws/downloads/lincs/LINCS2020/compoundinfo_beta.txt'
    cmp_lines = []
    for line in open(fn):
        line = line.strip()
        cmp_lines.append((line.split('\t')[0], line))
    from dtk.data import kvpairs_to_dict
    cmp_to_lines = kvpairs_to_dict(cmp_lines)

    u2g = make_uniprot_to_gene(uniprot_version)
    
    print(f"Found {len(new_prots)} new proteins")
    for prot, drugs in new_prots:
        print('-----------------')
        print(prot, u2g[prot], drugs)
        for drug in drugs:
            for line in cmp_to_lines[drug]:
                print(line)


if __name__ == "__main__":
    import argparse
    arguments = argparse.ArgumentParser(description="Creates a drugset for lincs data.")

    arguments.add_argument('-u', '--uniprot-version', help="Uniprot converter version")
    arguments.add_argument('-e', '--evidence',  help="Compound info file")
    arguments.add_argument('-a', '--attributes',  help="Compound info file")
    arguments.add_argument('-f', '--ref-evidence',  help="Compound info file")
    args = arguments.parse_args()
    from dtk.log_setup import setupLogging
    setupLogging()

    run(**vars(args))

