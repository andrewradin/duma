#!/usr/bin/env python
import logging
logger = logging.getLogger(__name__)
import os

from dtk.gwas_filter import gwas_filter

from collections import namedtuple
def process(fn, threshold, manifest_data, p_secondary, p_secondary_prots):
    """
    We're doing some fancy things here to try to keep some diversity among both studies and SNPs, without keeping around
    an absurd amount of data.

    We keep all variants with p < threshold, where threshold is set pretty aggressively.
    
    For other variants where p < p_secondary, we keep the most signficant N variants that are 'near' unique proteins, with
    N=p_secondary_prots.  This helps ensure we keep at least some data even for the rarer diseases, and that the kept data
    isn't all clustered at a single region of a single chromosome.

    There are better ways to do this (i.e. looking for separated peaks) that probably should be implemented instead at
    some point, but this is an easy first pass.
    """
    Rec = namedtuple('Rec', 'chrom pos ref alt rsids nearest_genes pval beta sebeta maf maf_cases maf_controls n_hom_cases n_het_cases n_hom_controls n_het_controls')
    out_header = [
            'phenotype|study',
            'rsid',
            'chr',
            'pos',
            'pval',
            'maf',
            'flag'
        ]
    from dtk.files import get_file_records
    import os
    os.makedirs('filter_logs', exist_ok=True)
    gwfilter = gwas_filter('filter_logs/' + os.path.basename(fn) + '.filter.log')

    out = []
    basefn = os.path.basename(fn)
    study = manifest_data[basefn]['name']
    n_samples = manifest_data[basefn]['samples']

    best_gene_data = {}
    reported = set()

    def report(data):
        # NOTE: Some of these are more than 1 character... but the snp check will fail us if we pass that in.
        # We end up discarding the provided allele downstream anyway, so just pull off the first base...
        alt = data.alt[0]

        # RSIDs are stored as just the ids, without 'rs'
        rsids = [x.lstrip('rs') for x in data.rsids.split(',')]
        rsid_str = ','.join(rsids)

        snp_data = [rsid_str, data.chrom, data.pos, data.pval, f'{alt};{data.maf}', 'NA']
        if gwfilter.qc_snp(snp_data, n_samples, study):
            out.append('\t'.join([study] + snp_data) + '\n')
            reported.add(data.nearest_genes)

    for rec in get_file_records(fn, keep_header=False, parse_type='tsv'):
        data = Rec(*rec)
        if float(data.pval) > threshold:
            # Didn't hit our primary threshold, keep track anyway for secondary reporting.
            if data.nearest_genes not in best_gene_data or best_gene_data[data.nearest_genes].pval > data.pval:
                best_gene_data[data.nearest_genes] = data
        else:
            report(data)

    cnt_primary = len(out)
    
    others = sorted(best_gene_data.values(), key=lambda x: x.pval)
    for extra in others:
        if len(reported) >= p_secondary_prots:
            # We've collected enough data for this study
            break
        if float(data.pval) < p_secondary:
            report(extra)

    cnt_secondary = len(out) - cnt_primary
    logger.info(f"Collected {cnt_primary} primary and {cnt_secondary} secondary from {fn}")
        
    gwfilter.report()
    return out


def do_studies(manifest, output_studies):
    out_header = [
        'Phenotype|PMID',
        'TotalSamples(discovery+replication)',
        'GWASancestryDescription',
        'Platform [SNPs passing QC]',
        'DatePub',
        'IncludesMale/Female Only Analyses',
        'Exclusively Male/Female',
        'European Discovery',
        'African Discovery',
        'East Asian Discovery',
        'European Replication',
        'African Replication',
        'East Asian Replication',
    ]

    platform = "Illumina and Affymetrix"
    date = "N/A"

    from dtk.files import get_file_records
    datas = []
    logger.info("Processing manifest")
    from atomicwrites import atomic_write
    manifest_data = {}
    header = None

    from atomicwrites import atomic_write
    with atomic_write(output_studies, overwrite=True) as f:
        # name	category	n_cases	n_controls	path_bucket	path_https
        for rec in get_file_records(manifest, keep_header=True, progress=True):
            if header is None:
                header = rec
                cases_idx = rec.index('n_cases')
                controls_idx = rec.index('n_controls')
                name_idx = rec.index('name')
                url_idx = rec.index('path_https')
                continue

            study = rec[name_idx].lower() + '|finngen'
            study = study.replace(' ', '_')

            samples = int(rec[cases_idx]) + int(rec[controls_idx])
            out = [study, samples, 'finngen', platform, date, 'both_sexes', 'both_sexes', 'y', 0, 0, 0, 0, 0]
            f.write('\t'.join([str(x) for x in out]) + '\n')

            fn = os.path.basename(rec[url_idx])
            manifest_data[fn] = {
                'name': study,
                'samples': samples,
                }
    return manifest_data

def do_data(input_dir, output_data, pval, manifest_data, p_secondary, p_secondary_prots):
    import os
    input_names = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
    logger.info(f"Processing {len(input_names)} files")
    from dtk.parallel import pmap
    from atomicwrites import atomic_write
    import isal.igzip as gzip
    out_header = [
            'phenotype|study',
            'rsid',
            'chr',
            'pos',
            'pval',
            'maf',
            'flag',
            ]
    with atomic_write(output_data, overwrite=True, mode='wb') as f:
        with gzip.open(f, 'wt', compresslevel=2) as gzf:
            gzf.write('\t'.join(out_header) + '\n')
            static_args = dict(
                    threshold=pval,
                    manifest_data=manifest_data,
                    p_secondary=p_secondary,
                    p_secondary_prots=p_secondary_prots,
                    )
            for outputs in pmap(process, input_names, static_args=static_args, progress=True):
                for line in outputs:
                    gzf.write(line)

def run(input_dir, output_data,  output_studies, input_manifest, p_primary, p_secondary, p_secondary_prots):
    manifest_data = do_studies(input_manifest, output_studies)
    do_data(input_dir, output_data, p_primary, manifest_data, p_secondary, p_secondary_prots)



if __name__ == "__main__":
    import argparse
    arguments = argparse.ArgumentParser(description="")
    arguments.add_argument("-i", "--input-dir", required=True, help='Downloaded snps')
    arguments.add_argument("--input-manifest", required=True, help='Manifest file')
    arguments.add_argument("--output-data", required=True, help="Output data (.gz)")
    arguments.add_argument("--output-studies", required=True, help="Output studies")
    arguments.add_argument("--p-primary", type=float, required=True, help="All SNPs better than this will be included")
    # We need to set a very aggressive p-primary value to get the output size down.
    # But you end up with a lot of SNPs around a small number of proteins in a given study, and most of our data prevalent studies.
    # This secondary p-value filter ensures that we get a diversity of SNP regions for each study, and also that
    # we get a more even spread of data across studies.
    # See the comment at the top of "def process" above.
    arguments.add_argument("--p-secondary", type=float, required=True, help="For top N prots, only consider SNPs better than this")
    arguments.add_argument("--p-secondary-prots", type=int, required=True, help="Keep the top N SNPs near unique prots")
    args = arguments.parse_args()
    from dtk.log_setup import setupLogging
    setupLogging()
    run(**vars(args))
