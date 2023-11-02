#!/usr/bin/env python

if __name__=='__main__':
    import sys
    try:
        from dtk.files import get_file_records
    except ImportError:
        sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
        from dtk.files import get_file_records
    import argparse
    from dtk.hgChrms import linear_hgChrms
    from dtk.etl import latest_ucsc_hg_in_gwas
    # Set up the argument parser
    arguments = argparse.ArgumentParser(description='Parse PheWAS into a .tsv file')
    arguments.add_argument("-dataset", help='''Desired PheWAS dataset to
        download and unzip. Default is "phewas-catalog.csv"''', default = 'phewas-catalog.csv')
    arguments.add_argument("-output1", help='''Output 1 file name. Default is
        "phewasData.tsv"''', default = "phewasData.tsv")
    arguments.add_argument("-output2", help='''Output 2 file name. Default is
        "phewasStudies.tsv"''', default = "phewasStudies.tsv")
    arguments.add_argument("--chain_file", help = '''File used to convert genome builds. Optional.''')
    args = arguments.parse_args()
    max_p = 0.05 # this is consistent with GRASP
    pubmed_id = '24270849'
    total_samples = '13835'
    GWAS_ancestry = 'European'
    platform = 'PheWAS'
    snps_qc_pass = '[3144]'
    date_pub = '2012-04-17'
    chrms = linear_hgChrms(latest_ucsc_hg_in_gwas())
    if args.chain_file:
        try:
            from our_CrossMap import CrossMap
        except ImportError:
            sys.path.insert(1, (sys.path[0] or '.')+"/../CrossMap")
            from our_CrossMap import CrossMap
        CrossMap = CrossMap(args.chain_file)
        unmapped = set()
    else:
        CrossMap = False

    with open(args.output1, 'w') as f1, open(args.output2, 'w') as f2:
        f2.write('\t'.join(['Phenotype|PMID', 'Total Samples (discovery+replication)', 
                           'GWASancestry Description', 'Platform [SNPs passing QC]',
                           'DatePub', 'Includes Male/Female Only Analyses', 'Exclusively Male/Female European',
                           'Discovery African Discovery East Asian', 'Discovery European Replication African',
                           'Replication East Asian Replication'])+'\n')
        phewas_gen = (x for x in get_file_records(str(args.dataset),
                       parse_type = 'csv', keep_header=False))
        seen = set()
        header = None
        for gen in get_file_records(args.dataset,
                       parse_type = 'csv_strict',
        ):
            if not header:
                header = gen
                pheno_ind = header.index("phewas phenotype")
                snp_ind = header.index("snp")
                # chromosome column actually has chr pos for most entries
                position_ind = header.index("chromosome")
                pvalue_ind = header.index("p-value")
                continue
            k = gen[pheno_ind].replace(' ', '_').strip('"').lower()+'|'+ pubmed_id
            snp_id=gen[snp_ind].strip('"').lstrip('rs')
            position = gen[position_ind].strip('"')
            try:
                #some entries are only the chr
                chr,pos=position.split()
            except:
                chr=position
                pos=0
            pvalue=gen[pvalue_ind]
            if float(pvalue) > max_p or not chrms.check_position(chr, pos):
                continue
            if CrossMap:
                input = ['chr' + chr,
                         int(pos),
                         int(pos),
                         '+'
                        ]
                new_locs = CrossMap.crossMap(input)
                if new_locs:
                    chr = new_locs[0].lstrip('chr')
                    pos = str(new_locs[1])
                else:
                    unmapped.add(' '.join([str(x) for x in input]))
                    continue
            
            f1.write('\t'.join([k, snp_id,
                                chr, pos, pvalue, 'NA', 'NA'])+'\n')
            if k not in seen:
                f2.write('\t'.join([k, 
                                total_samples, GWAS_ancestry, platform+' '+snps_qc_pass,
                                date_pub, 'NA', 'NA', 
                                '0','0', '0', '0', '0','0'])+'\n')

            seen.add(k)
    with open('unmapped.txt', 'w') as f:
        f.write('\n'.join(unmapped) + '\n')
