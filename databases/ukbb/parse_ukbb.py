#!/usr/bin/env python

##Todo: this assumes that there is always at least one valid snp per phenotype

def get_filtered_data(in_file, p_val,ac):
    gen = get_file_records(in_file)
    try:
        header = next(gen)
    except StopIteration:
        return None,None
    header_inds = [header.index(x) for x in header]
    col2ind = dict(zip(header, header_inds))
    filtered_gen = (x for x in gen if ukbb_snp_filter(x, col2ind, ac, p_val))
    return (col2ind, filtered_gen)

def ukbb_snp_filter(x, col2ind, ac, p_val):
    myreturn = False
    try:
        value = float(x[col2ind['pval']])
    except ValueError:
        return myreturn
    if value < p_val:
        if x[col2ind['low_confidence_variant']] != 'true':
            try:
                if float(x[col2ind['expected_case_minor_AC']]) >= ac:
                    myreturn = True
            except:
                try:
                    if float(x[col2ind['expected_min_category_minor_AC']]) >= ac:
                        myreturn = True
                except:
                    if float(x[col2ind['minor_AF']])*float(x[col2ind['n_complete_samples']]) >= ac:
                        myreturn = True
    return myreturn


def write_filtered_data(col2ind, in_gen, study, pheno, out_file):
    import gzip
    with gzip.open(out_file, 'ab') as of:
        for x in in_gen:
            chr_num = x[0].split(':')[0]
            chr_pos = x[0].split(':')[1].split(':')[0]
            p_val = x[col2ind['pval']]
            maf = x[col2ind['minor_allele']] + ";" + str(np.round(float(x[col2ind['minor_AF']]), 2))
            for_writing = '\t'.join(['|'.join([pheno,study]).lower(), 'rs_tbd', chr_num, chr_pos, p_val, maf, 'NA'])+'\n'
            of.write(for_writing.encode('utf-8'))
            nsamples = str(x[col2ind['n_complete_samples']])
    return nsamples

def write_studies(in_gen, study, pheno, nsamples, sex, out_file):
    with open(out_file, 'a') as of:
        of.write('\t'.join(['|'.join([pheno,study]).lower(), nsamples, "ukbb", "Ilumina [10800000]", "07/31/2018", sex, sex, "y", "NA", "0", "0", "0", "0", "0", "0"])+'\n')

if __name__=='__main__':
    import numpy as np
    import argparse
    import sys
    try:
        from dtk.files import get_file_records
    except ImportError:
        sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
        from dtk.files import get_file_records
    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="Parse full UKBB into our format")
    arguments.add_argument("-p", help = "Desired p-value cutoff")
    arguments.add_argument("-i", help = "Input file name")
    arguments.add_argument("-osnp", help="Output snp file name")
    arguments.add_argument("-ostudy", help="Output study file name")
    arguments.add_argument("-pheno", help="UKBB Phenotype, ex: 'Worry_too_long_after_embarrassment' ")
    arguments.add_argument("-study", help="pubmed ID")
    arguments.add_argument("-sex", help="sex of study cohort: both_sexes, female, male")
    arguments.add_argument("-ac", help="minimum number for AF*2*people in smallest sample subpopulation, default=50", default=50)
    args = arguments.parse_args()
    # Run the script
    col2ind, filtered_gen = get_filtered_data(args.i, float(args.p),float(args.ac))
    if col2ind is not None and filtered_gen is not None:
        nsamples = write_filtered_data(col2ind, filtered_gen, args.study, args.pheno, args.osnp)
        write_studies(filtered_gen, args.study, args.pheno, nsamples, args.sex, args.ostudy)
