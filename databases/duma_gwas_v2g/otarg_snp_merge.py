#!/usr/bin/env python

from vep_out_to_v2g import make_uniprot_map

from dtk.files import get_file_records, get_file_lines
from collections import Counter
import logging
logger = logging.getLogger(__name__)

# This is only used when there is a consequence that is not scored by VEP
# see score_conseq, below
#
# Running Mar2022 there were three new variants
# see https://m.ensembl.org/info/docs/tools/vep/script/VEP_script_documentation.pdf
# search for "New in version 105 (October 2021)"
# Here is one of the new terms:
# e.g., http://www.sequenceontology.org/miso/current_release/term/SO:0002169
#
# I'm expecting that these will eventually get included in the vep file,
# but if it is not soon we should manually score by looking at parent scores

DEFAULT_SCORE=0.1
MISSING_CONSEQ = Counter()

def load_vep_consequences(fn):
    out = {}
    header = None
    for rec in get_file_records(fn, keep_header=True):
        if header is None:
            header = rec
            continue
        data = dict(zip(header, rec))
        # There is a v2g_score and an eco_score.
        # The v2g_score seems a bit harsh (lots of 0's), let's use the eco score instead for now.
        out[data['Term']] = float(data['eco_score'])
    return out

def score_conseq(conseq2score, conseqs):
    # Score as the maximum-scored consequence.
    out = 0
    for conseq in conseqs:
        if conseq not in conseq2score:
            con_score = DEFAULT_SCORE
            MISSING_CONSEQ.update([conseq])
        else:
            con_score = conseq2score[conseq]
        out = max(out, con_score)
    return out

def load_orig(orig_input, vep_consequences_fn):
    conseq2score = load_vep_consequences(vep_consequences_fn)
    from collections import defaultdict
    out_prots = defaultdict(dict)
    out_rs = defaultdict(set)
    for chrmpos, allele, rsids, conseq, prot in get_file_records(orig_input, keep_header=False, progress=True):
        conseqs = conseq.split(',')
        score = score_conseq(conseq2score, conseqs)
        out_prots[chrmpos][prot] = score
        if rsids.strip():
            out_rs[chrmpos].update(rsids.split(','))
    logger.info(f'Counts of missing consequences: {MISSING_CONSEQ}')
    return out_prots, out_rs


def merge_otarg(orig_data, otarg_input, output):
    import isal.igzip as gzip
    seen = set()
    missing = set()
    prev_id = None
    orig_prot_map, orig_rs_map = orig_data
    with gzip.open(output, 'wt') as out:
        row_header = ['chrm_and_pos', 'allele', 'rs_ids', 'consequences', 'uniprot', 'score']
        out.write('\t'.join(row_header) + '\n')

        for chrm, pos, score_json in get_file_records(otarg_input, keep_header=None, progress=True):
            import json
            otarg_score_data = json.loads(score_json)
            id = f'{chrm}:{pos}'

            if id not in orig_prot_map:
                missing.add(id)
            seen.add(id)

            orig_prots = orig_prot_map.get(id, {})
            orig_rsids = orig_rs_map.get(id, set())

            from collections import defaultdict
            out_prots = defaultdict(float)
            # Just take the max score across both prot sets for now.
            for prot, score in orig_prots.items():
                out_prots[prot] = max(out_prots[prot], score)

            for prot, score in otarg_score_data.items():
                out_prots[prot] = max(out_prots[prot], score)

            rsid_str = ','.join(orig_rsids)

            # We're leaving out the allele and consequences data, as it is unused, just kept for format compat.
            row_start = [id, '', rsid_str, '']
            for prot, score in out_prots.items():
                row = row_start + [prot, f'{score:.2f}']
                out.write('\t'.join(row) + '\n')

        # Handle snps that weren't found in opentargets, just re-transcribe them back out in the new format.
        from tqdm import tqdm
        for chrmpos, orig_prot_data in tqdm(orig_prot_map.items(), desc='Unmatched snps'):
            if chrmpos in seen:
                continue
            orig_rsids = orig_rs_map.get(id, set())
            rsid_str = ','.join(orig_rsids)
            row_start = [id, '', rsid_str, '']
            for prot, score in orig_prot_data.items():
                row = row_start + [prot, f'{score:.2f}']
                out.write('\t'.join(row) + '\n')

    # This comes from VEPd SNPs that either failed or didn't have any associated prots
    logger.info(f"Had {len(missing)} otarg snps that weren't in original... sample: {list(missing)[:10]}")

def run(orig_input, otarg_input, output, vep_consequences):
    orig_data = load_orig(orig_input, vep_consequences)
    merge_otarg(orig_data, otarg_input, output)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--orig-input", help="Output of the vep processing")
    parser.add_argument("--otarg-input", help="Output of the otarg processing")
    parser.add_argument("--vep-consequences", help="VEP consequence scoring")
    parser.add_argument("--output", help="Output file (tsv)")
    from dtk.log_setup import addLoggingArgs, setupLogging
    args = parser.parse_args()
    setupLogging()
    run(**vars(args))
