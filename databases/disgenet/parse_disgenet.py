#!/usr/bin/env python3

import sys
try:
    from dtk.files import get_file_records
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
    from dtk.files import get_file_records

def make_entrez_2_uniprot_map(file):
    conv_d = {}
    with open(file, 'r') as f:
        for l in f:
            fields = l.rstrip("\n").split("\t")
            if fields[1] == 'GeneID':
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
    arguments.add_argument("uniprot_version")
    arguments.add_argument("meddra_version")
    arguments.add_argument("umls_version")
    arguments.add_argument("dis_out", help="output file for the disease names, keyed by MedDRA terms")
    arguments.add_argument("dis_out_cui", help="output file for the disease names, keyed by CUI")
    arguments.add_argument("umls_out", help="output file for the curated UMLS keyed scores")
    args = arguments.parse_args()

    from dtk.s3_cache import S3File
    uniprot_s3f=S3File.get_versioned(
            'uniprot',
            args.uniprot_version,
            role='Uniprot_data',
            )
    uniprot_s3f.fetch()
    umls_s3f=S3File.get_versioned(
            'umls',
            args.umls_version,
            role='to_meddra',
            )
    umls_s3f.fetch()
    conv_d = make_entrez_2_uniprot_map(uniprot_s3f.path())
    from dtk.meddra import IndiMapper
    d = IndiMapper(args.meddra_version)
    umls2meddra = dict(get_file_records(umls_s3f.path()))
    c = Counter()
    gen = (l for l in sys.stdin if not l.startswith("#"))
    header = None
    disease_d={}
    cui_disease_d={}
    missed_umls_codes = Counter()
    with open('geneids_without_uniprots.txt', 'w') as f:
        with open(args.umls_out, 'w') as f2:
            for line in gen:
                fields = line.rstrip("\n").split("\t")
                # later versions of the input file right-justify some
                # fields with spaces; fix that here
                fields = [x.lstrip(' ') for x in fields]
                if not header:
                    header = fields
                    from collections import namedtuple
                    RowType=namedtuple('DisGeNet',header)
                    rec = RowType(*fields)
                    print_out('MedDRA_ID', rec, 'Uniprot')
                    f2.write("\t".join(['UMLS_ID','Uniprot',  rec.score, rec.source]) + "\n")
                    continue
                rec = RowType(*fields)
                try:
                    uniprot_list = conv_d[rec.geneId]
                except KeyError:
                    f.write(rec.geneId + "\n")
                    continue
                missed_umls = False
                for u in uniprot_list:
                    f2.write("\t".join([rec.diseaseId, u, rec.score, rec.source]) + "\n")
                    meddra_code = umls2meddra.get(rec.diseaseId)
                    try:
                        c.update(['hit'])
                        indi_mapper_name = d.code2name(meddra_code)
                        disease_d[meddra_code] = indi_mapper_name
                        print_out(meddra_code, rec, u)
                    except KeyError:
                        missed_umls_codes.update([rec.diseaseId])
                        c.update(['miss'])
                        missed_umls = True
                if missed_umls:
                    continue
                if rec.diseaseId in cui_disease_d and cui_disease_d[rec.diseaseId]!=rec.diseaseName:
                    assert False, " ".join(["Inconsistent CUI to disease name:",
                                            cui_disease_d[rec.diseaseId]!=rec.diseaseName,
                                            rec.diseaseName,
                                            rec.diseaseId
                                          ])
                cui_disease_d[rec.diseaseId]=rec.diseaseName
    with open(args.dis_out, 'w') as f:
        f.write("\t".join(["MedDRA_ID", "Disease_Name"]) + "\n")
        sorted_x = sorted(disease_d.items(), key=operator.itemgetter(1))
        for tup in sorted_x:
            f.write("\t".join(tup) + "\n")
    with open(args.dis_out_cui, 'w') as f:
        f.write("\t".join(["UMLS_CUI", "Disease_Name"]) + "\n")
        sorted_x = sorted(cui_disease_d.items(), key=operator.itemgetter(1))
        for tup in sorted_x:
            f.write("\t".join(tup) + "\n")

    with open('how_mapped.log', 'w') as f:
        dump_counter(f,c)
    with open('missed_umls_codes.log', 'w') as f:
        dump_counter(f,missed_umls_codes)
