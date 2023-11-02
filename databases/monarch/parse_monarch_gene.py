#!/usr/bin/env python3

import sys
from dtk.files import get_file_records

def make_gene_2_uniprot_map(file, attr_types=['Gene_Name', 'Gene_Synonym']):
    conv_d = {}
    for fields in get_file_records(file):
        if fields[1] not in attr_types:
            continue
        if fields[2] not in conv_d:
            conv_d[fields[2]] = set()
        conv_d[fields[2]].add(fields[0])
    return conv_d


if __name__=='__main__':
    import argparse
    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="Parse Monarch human genes to phenotypes")
    arguments.add_argument("gen_file", help="input file for the gene to pheno data")
    arguments.add_argument("uniprot_version")
    arguments.add_argument("gen_out", help="output file for the disease names, keyed by CUI")
    args = arguments.parse_args()

    from dtk.s3_cache import S3File
    uniprot_s3f=S3File.get_versioned(
            'uniprot',
            args.uniprot_version,
            role='Uniprot_data',
            )
    uniprot_s3f.fetch()

    conv_d = make_gene_2_uniprot_map(uniprot_s3f.path())

    from parse_monarch_disease import sources
    Sources = sources()

    RowType=None

    with open('geneids_without_uniprots.txt', 'w') as f:
        with open(args.gen_out, 'w') as f2:
            f2.write("\t".join([
                               'uniprot',
                               'pheno_id',
                               'phenotype',
                               'relation_type',
                               'evidence_code',
                               'source'
                     ])+"\n")

            for fields in get_file_records(args.gen_file):
                if not RowType:
                    from collections import namedtuple
                    RowType=namedtuple('Monarch',fields)
                    continue
                rec = RowType(*fields)
                source = Sources.is_safe(rec.is_defined_by)
                if not source:
                    continue
                try:
                    uniprot_list = conv_d[rec.subject_label]
                except KeyError:
                    f.write(rec.subject_label + "\n")
                    continue
                for u in uniprot_list:
                    f2.write("\t".join([
                        u,
                        rec.object,
                        rec.object_label,
                        rec.relation_label,
                        rec.evidence,
                        source
                    ]) + "\n")
