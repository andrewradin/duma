#!/usr/bin/env python3

import sys
try:
    from dtk.files import get_file_records
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
    from dtk.files import get_file_records

def make_hgnc_2_uniprot_map(file):
    conv_d = {}
    with open(file, 'r') as f:
        for l in f:
            fields = l.rstrip("\n").split("\t")
            if fields[1] == 'hgnc':
                try:
                    conv_d[fields[2]].add(fields[0])
                except KeyError:
                    conv_d[fields[2]] = set()
                    conv_d[fields[2]].add(fields[0])
    return conv_d

def print_out(k, l, u):
    print("\t".join([k, u, l.score, l.source]))

def dump_counter(f,c):
    for key,count in c.most_common():
        f.write("\t".join([key,str(count)])+"\n")

if __name__=='__main__':
    import argparse
    import operator
    from collections import Counter
    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="Parse DisGeNet disease associated gene lists")
    arguments.add_argument("input")
    arguments.add_argument("uniprot_version")
    arguments.add_argument("dis_out", help="output file for the disease names, keyed by disease ontology terms")
    arguments.add_argument("model_out", help="output file for the disease models, keyed by disease ontology terms")
    args = arguments.parse_args()

    from dtk.s3_cache import S3File
    uniprot_s3f=S3File.get_versioned(
            'uniprot',
            args.uniprot_version,
            role='Uniprot_data',
            )
    uniprot_s3f.fetch()
    conv_d = make_hgnc_2_uniprot_map(uniprot_s3f.path())


    missed_genes = set()
    header = False
    with open(args.dis_out, 'w') as f:
        with open(args.model_out, 'w') as f2:
            f.write("\t".join(['DiseaseName', 'DOID', 'Uniprot', 'AssociationType', 'EvidenceDescription', 'Reference'])+"\n")
            f2.write("\t".join(['DiseaseName', 'DOID', 'ModelSpecies', 'ModelType', 'AssociationType', 'EvidenceDescription', 'Reference'])+"\n")
            for fields in get_file_records(args.input):
                if fields[0].startswith('#'):
                    continue
                if not header:
                    header = fields
                    continue
                if fields[header.index('SpeciesName')] == 'Homo sapiens':
                    hgnc_id = fields[header.index('DBObjectID')].split(':')[1]
                    if hgnc_id not in conv_d:
                        missed_genes.add(fields[header.index('DBObjectSymbol')])
                        continue
                    uniprots = conv_d[hgnc_id]
                    # write out each line (no aggregating for now)
                    for u in uniprots:
                        f.write("\t".join([fields[header.index('DOtermName')],
                                       fields[header.index('DOID')],
                                       u,
                                       fields[header.index('AssociationType')],
                                       fields[header.index('EvidenceCodeName')],
                                       fields[header.index('Reference')]
                                      ])+"\n")
                elif fields[header.index('AssociationType')] == 'is_model_of':
                    f2.write("\t".join([fields[header.index('DOtermName')],
                                       fields[header.index('DOID')],
                                       fields[header.index('SpeciesName')],
                                       fields[header.index('DBObjectSymbol')],
                                       fields[header.index('AssociationType')],
                                       fields[header.index('EvidenceCodeName')],
                                       fields[header.index('Reference')]
                                      ])+"\n")
                else:
                    # At some point we may want to pull data from the model organisms
                    # We'd need to "translate" from the respective species to human
                    # the write out would otherwise be pretty similar to the human one above
                    continue
    print('Missing gene names:')
    print(", ".join(missed_genes))
