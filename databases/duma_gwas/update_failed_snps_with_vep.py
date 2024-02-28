#!/usr/bin/env python
import sys
from filter_gwas import get_nsamples
from dtk.files import get_file_records

def _init_filter():
    import os
    try:
        from filter_gwas import gwas_filter
    except ImportError:
        sys.path.insert(1, (sys.path[0] or '.')+"/../duma_gwas")
        from filter_gwas import gwas_filter
    filter = gwas_filter(log=os.path.dirname(os.path.realpath(__file__))+'/filter.log')
    return filter

from update_gwas_with_vep import make_uniprot_map

def create_vep_dict(uniprot_file, input_vep_file):
    ''' Creates a dict where keys are rs IDs '''
    uni_conv = make_uniprot_map(uniprot_file, 'Ensembl')
    fn = input_vep_file
    vep_gen = (x for x in get_file_records(fn, progress=True) if x[0][0:2] != '##')
    header = next(vep_gen)
    header_inds = [header.index(x) for x in header]
    col2ind = dict(zip(header, header_inds))
    vep_dict = {}
    for item in vep_gen:
        rs_id = item[0]
        snp_loc = item[col2ind['Location']] + ':' + item[col2ind['Allele']]
        max_af = item[col2ind['MAX_AF']]
        consequence = item[col2ind['Consequence']]
        if max_af == '-':
            max_af = 0
        if rs_id not in vep_dict:
            try:
                vep_dict[rs_id] = [snp_loc,
                                   max_af,
                                   consequence,
                                   set(uni_conv[item[col2ind['Gene']]])]
            except KeyError:
                tmp_set = set()
                tmp_set.add(item[col2ind['Gene']])
                vep_dict[rs_id] = [snp_loc,
                                   max_af,
                                   consequence,
                                   tmp_set]
        elif rs_id in vep_dict:
            if float(vep_dict[rs_id][1]) < float(max_af):
                vep_dict[rs_id][0] = snp_loc
                vep_dict[rs_id][1] = max_af
                vep_dict[rs_id][2] = consequence
            try:
                vep_dict[rs_id][3].update(uni_conv[item[col2ind['Gene']]])
            except KeyError:
                vep_dict[rs_id][3].add(item[col2ind['Gene']])
        for name in list(vep_dict[rs_id][3]):
            if name.startswith('ENSG'):
                vep_dict[rs_id][3].remove(name)
                vep_dict[rs_id][3].add('-')
    for k, v in vep_dict.items():
        if len(v[3])>1 and '-' in v[3]:
            vep_dict[k][3].remove('-')
    return vep_dict

def update_failed_snps(failed_snps_file, vep_dict, filter, output_file, nsamples):
    ''' Updates the failed SNPS file with data outputted by VEP '''
    failed_snps_gen = get_file_records(failed_snps_file, progress=True)
    with open(output_file, 'w') as of, open('not_in_vep.log', 'w') as log:
        for item in failed_snps_gen:
            rs_id = 'rs' + item[1]
            if rs_id in vep_dict:
                key = item[0] # key
                rs = rs_id.lstrip('rs')
                # The value in the vep_dict is almost always an NC_ accession
                #chr_num = vep_dict[rs_id][0].split(':')[0] # chr
                chr_num = item[2]
                chr_pos = vep_dict[rs_id][0].split(':')[1] # pos
                p_val = item[4] # pval
                variant = vep_dict[rs_id][2] # variant
                allele = vep_dict[rs_id][0].split(':')[2] # allele
                maf = vep_dict[rs_id][1] # maf
                if maf == 0:
                    maf = '-'
                dbsnp = item[-1] # dnSNP validated
                qc_check = [rs_id, chr_num, chr_pos, p_val, allele + ';' + str(maf)]
                if not filter.qc_snp(qc_check, nsamples[key], key):
                    continue
                pref_to_report = [key, rs, chr_num, chr_pos, p_val]
                suf_to_report = [variant, allele + ';' + str(maf), dbsnp]
                for uniprot_id in vep_dict[rs_id][3]:
                    of.write('\t'.join(pref_to_report + [uniprot_id] + suf_to_report)+'\n')
            elif rs_id not in vep_dict:
                log.write(rs_id + '\t' + '\t'.join(item) + '\n')
    filter.report()

if __name__ == '__main__':
    import argparse
    arguments = argparse.ArgumentParser(description='''Reformats failed SNP data to include VEP data.''')
    arguments.add_argument("uni", help = 'Input Uniprot file')
    arguments.add_argument("vep", help = 'Input VEP file (which should have been run on the failed SNPs)')
    arguments.add_argument("failed", help = 'Input failed SNPs file')
    arguments.add_argument("studies", help = 'All studies file')
    arguments.add_argument("o", help = 'Output filename')
    args = arguments.parse_args()

    filter = _init_filter()
    nsamples = get_nsamples(args.studies)
    vep_dict = create_vep_dict(args.uni, args.vep)
    update_failed_snps(args.failed, vep_dict, filter, args.o, nsamples)
