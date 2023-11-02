#!/usr/bin/env python

from vep_out_to_v2g import make_uniprot_map

from dtk.files import get_file_records, get_file_lines
import logging
logger = logging.getLogger(__name__)

def parse_variant_data(data):
    # The name of the quantile-score field for each score type.
    type_to_scorename = {
        'distance': 'distance_score_q',
        'eqtl': 'qtl_score_q',
        'pqtl': 'qtl_score_q',
        'sqtl': 'qtl_score_q',
        'dhscor': 'interval_score_q',
        'fantom5': 'interval_score_q',
        'pchic': 'interval_score_q',
        'fpred': 'fpred_max_score', # vep
    }

    score_type = data['type_id']
    try:
        scorename = type_to_scorename[score_type]
    except KeyError:
        logger.warning(f'Found an unrecognized score_type: {score_type}')
        raise
    if scorename not in data:
        return score_type, None, None

    score_val = data[scorename]
    gene = data['gene_id']

    assert score_val >= 0 and score_val <= 1.0, f"Unexpected score value {score_val} from {data}"
    return score_type, score_val, gene

def parse_variants_file(file):
    from collections import defaultdict
    # (chr,pos,gene) -> [(score_type, qscore)]
    var2scores = defaultdict(list)

    # (chr,pos) -> {(ref, alt)}
    var2meta = defaultdict(set)

    type_counts = defaultdict(int)
    fail_counts = defaultdict(int)

    # First, pull out the types & scores and group by variant
    import json
    for line in get_file_lines(file, keep_header=None, progress=False):
        data = json.loads(line)
        chrm = data['chr_id']
        pos = data['position']

        score_type, score_val, gene = parse_variant_data(data)
        type_counts[score_type] += 1
        if score_val is None:
            fail_counts[score_type] += 1
            continue

        var2scores[(chrm,pos,gene)].append((score_type, score_val))
        var2meta[(chrm,pos)].add((data['ref_allele'], data['alt_allele']))
    logger.info(f"Counts: {sorted(type_counts.items())} - Fails: {dict(fail_counts)}") 
    return var2scores, var2meta

def summarize_scores(var2scores):
    # The relative score type weights from https://genetics-docs.opentargets.org/our-approach/data-pipeline
    type_to_weight= {
        'fpred': 1.0, # vep
        'eqtl': 2/3.,
        'pqtl': 2/3.,
        'dhscor': 1/3.,
        'fantom5': 1/3., # enhancer-tss / andersson2014
        'pchic': 1/3.,
        'distance': 1/3.,
    }

    # There are a bunch of associations whose only evidence is transcription distance, with a very large window.
    # We threshold out the least confident of these to reduce noise.
    # In practice, a value of N will filter out roughly that fraction of the data, as these are quantile scores.
    # (e.g. 0.2 filters out a bit under 20% of the data, only keeping scores under 0.2 where they overlap with other evid)
    DISTANCE_ONLY_THRESHOLD = 0.2
    filtered_cnt = 0
    output_cnt = 0

    from collections import defaultdict
    out = defaultdict(dict)
    for (chrm, pos, gene), scores in var2scores.items():
        by_type = defaultdict(list)
        for scoretype, score in scores:
            by_type[scoretype].append(score)
        
        agg_by_type = {k: sum(v) / len(v) for k, v in by_type.items()}

        # Check if our only evidence is a far-away distance.
        if len(agg_by_type) == 1:
            key, val = list(agg_by_type.items())[0]
            if key == 'distance' and val <= DISTANCE_ONLY_THRESHOLD:
                filtered_cnt += 1
                continue
        
        num = 0
        den = 0
        # We score every score_type for every association; those that are missing
        # get implicit 0's.
        for score_type, weight in type_to_weight.items():
            score_val = agg_by_type.get(score_type, 0)
            num += score_val * weight
            den += weight
        
        weighted_score = num / den
        out[(chrm, pos)][gene] = weighted_score

    perc = filtered_cnt * 100 / len(var2scores)
    logger.info(f"Filtered out {filtered_cnt} / {len(var2scores)} ({perc:.1f}%)")
    return out

def process_file(variants_fn):
    var2scores, var2meta = parse_variants_file(variants_fn)
    return summarize_scores(var2scores), var2meta

def run(otarg_variants_files, output_v2g, output_alleles, uniprot):
    from tqdm import tqdm
    from atomicwrites import atomic_write
    from dtk.parallel import pmap
    unimap = make_uniprot_map(uniprot, 'Ensembl')
    missing_genes = set()

    import isal.igzip as gzip
    with gzip.open(output_v2g, 'wt') as out:
        from collections import defaultdict
        found_ids = defaultdict(int)
        out_allele_data = []
        for scores_output, var2meta in pmap(process_file, otarg_variants_files, progress=True):
            for (chrm, pos), gene2score in scores_output.items():
                prot2score = {}
                for gene, score in gene2score.items():
                    if gene not in unimap:
                        missing_genes.add(gene)
                        continue
                    prots = unimap[gene]
                    for prot in prots:
                        prot2score[prot] = score
                
                # This output format is convenient for merging, which wants all the data for a single snp at once.
                import json
                row = [chrm, str(pos), json.dumps(prot2score)]
                out.write('\t'.join(row) + '\n')
            

            for (chrm, pos), alleles in var2meta.items():
                for ref, alt in alleles:
                    out_allele_data.append([f'{chrm}:{pos}', ref, alt])
        logger.info("Write out allele ref data")
        from dtk.tsv_alt import SqliteSv
        SqliteSv.write_from_data(output_alleles, out_allele_data, [str, str, str], ['chrm_and_pos', 'ref', 'alt'], index=['chrm_and_pos'])

    logger.info(f"Missing uniprot-genes: {len(missing_genes)} - (samples) {list(missing_genes)[:10]}")
    logger.info(f"Expect roughly 14500 of these, mostly lncRNA")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("otarg_variants_files", nargs='+', help="Filtered/grouped otarg data")
    parser.add_argument('-o', "--output-v2g", help="Output v2g file (tsv.gz)")
    parser.add_argument("--output-alleles", help="Output alleles file (sqlsv)")
    parser.add_argument('-u', "--uniprot", help="Uniprot lookup file")
    from dtk.log_setup import addLoggingArgs, setupLogging
    args = parser.parse_args()
    setupLogging()
    run(**vars(args))
