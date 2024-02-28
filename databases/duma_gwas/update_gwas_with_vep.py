#!/usr/bin/env python3
import gzip
import sys
from dtk.files import get_file_records

def make_uniprot_map(file, search_term):
    '''Makes a uniprot converter map '''
    conv_d = {}
    for fields in get_file_records(file,
                                   keep_header = True,
                                   parse_type = 'tsv'):
        if fields[1] == search_term:
            try:
                conv_d[fields[2]].add(fields[0])
            except KeyError:
                conv_d[fields[2]] = set()
                conv_d[fields[2]].add(fields[0])
    return {k: frozenset(v) for k, v in conv_d.items()}

def create_vep_dict(uniprot_file, input_vep_file):
    ''' Creates a dict where keys are chrom: pos: allele '''
    uni_conv = make_uniprot_map(uniprot_file, 'Ensembl')
    fn = input_vep_file
    vep_gen = (x for x in get_file_records(fn, parse_type='tsv', progress=True) if x[0][0:2] != '##')
    header = next(vep_gen)
    header_inds = [header.index(x) for x in header]
    col2ind = dict(zip(header, header_inds))
    vep_dict = {}
    for item in vep_gen:
        snp_loc = item[col2ind['Location']] + ':' + item[col2ind['Allele']]
        if snp_loc not in vep_dict:
            exist_var = ','.join([x.strip() for x in item[col2ind['Existing_variation']].split(',') if x.startswith('rs')])
            try:
                vep_dict[snp_loc] = [item[col2ind['Consequence']], 
                                     exist_var, 
                                     set(uni_conv[item[col2ind['Gene']]])]
            except KeyError:
                tmp_set = set()
                tmp_set.add(item[col2ind['Gene']])
                vep_dict[snp_loc] = [item[col2ind['Consequence']],
                                     exist_var,
                                     tmp_set]
        elif snp_loc in vep_dict:
            try:
                vep_dict[snp_loc][2].update(uni_conv[item[col2ind['Gene']]])
            except KeyError:
                vep_dict[snp_loc][2].add(item[col2ind['Gene']])
        for name in list(vep_dict[snp_loc][2]):
            if name.startswith('ENSG'):
                vep_dict[snp_loc][2].remove(name)
                vep_dict[snp_loc][2].add('-')
    for k, v in vep_dict.items():
        if len(v[2])>1 and '-' in v[2]:
            vep_dict[k][2].remove('-')   
    return vep_dict
    
def update_gwas_file(input_file, vep_dict, output_file, failed_file):
    ''' Updates the combined GWAS file with data outputted by VEP '''
    gwas_gen = get_file_records(input_file, parse_type='tsv')
    with gzip.open(output_file, 'wt') as of, gzip.open(failed_file, 'wt') as log:
        for item in gwas_gen:
            snp_loc = item[2] + ':' + item[3] + ':' + item[-2].split(';')[0]
            if snp_loc in vep_dict:
                rs = item[1]
                if vep_dict[snp_loc][1] not in ['-', '', ' ']:
                    vep_rss = [x.lstrip('rs') for x in vep_dict[snp_loc][1].split(',')]
                    if rs not in vep_rss:
                        sys.stderr.write(':'.join(['mismatch detected b/t rs IDs',
                                                   rs,
                                                   vep_dict[snp_loc][1]
                                                  ])+"\n"
                                         )
                        rs = vep_rss[0]
                pref_to_report = [item[0], # key
                             rs,
                             item[2], # chr
                             item[3], # pos
                             item[4], # pval
                             ]
                suf_to_report = [vep_dict[snp_loc][0], # snp consequence
                                 item[5], #allele;MAF
                                 item[6] # dbSNP validated
                                ]
                for uniprot_id in vep_dict[snp_loc][2]:
                    of.write('\t'.join(pref_to_report + [uniprot_id] + suf_to_report)+'\n')
            elif snp_loc not in vep_dict:
                log.write('\t'.join(item +[snp_loc]) + '\n')

if __name__ == '__main__':
    import argparse
    arguments = argparse.ArgumentParser(description='''Reformats GWAS data to include VEP data.''')
    arguments.add_argument("uni", help = 'Input Uniprot file')
    arguments.add_argument("vep", help = 'Input VEP file')
    arguments.add_argument("gwas", help = 'Input combined GWAS file')
    arguments.add_argument("o", help = 'Output filename')
    arguments.add_argument("f", help = 'Failed filename')
    args = arguments.parse_args()
    vep_dict = create_vep_dict(args.uni, args.vep)
    update_gwas_file(args.gwas, vep_dict, args.o, args.f)
