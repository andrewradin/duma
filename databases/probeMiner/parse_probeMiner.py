#!/usr/bin/env python

### IMPORTANT NOTE ###

# As of now, this script retrieves drug-protein specific data.        #
# There are values and binary thresholded values for most columns.    #
# In the case of both, the raw values are kept instead of the binary. #
# Also, the create file will not have INCHI or SMILES.                #

### END IMPORTANT NOTE ###

if __name__=='__main__':
    import sys
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
    import os
    from dtk.files import get_file_records
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    import django
    django.setup()
    from browse.models import WsAnnotation
    import gzip
    import argparse
    import ast

    # Set up the argument parser
    arguments = argparse.ArgumentParser(description='''Parse ProbeMiner data and
        output two files.''')
    arguments.add_argument("dataset", help='''Desired ProbeMiner dataset to
        download and unzip. For example: "probeminer_datadump_2018-09-19.txt"''')
    arguments.add_argument("-create_out", nargs = '?', help='''Output file name of
        probeMiner create. The default is: "probeMiner_create.tsv"''',
        default = "probeMiner_create.tsv")
    arguments.add_argument("-data_out", nargs = '?', help='''Output file name
        of probeMiner data. The default is: "probeMiner_data.tsv"''',
        default ="probeMiner_data.tsv")
    args = arguments.parse_args()
    
    # Description of some of the data coming in from ProbeMiner
    '''
    1) Cell potency - Denoted by a cell, it shows whether a compound binding to the 
       target of interest is active in a cell line with at least 10 uM potency.

    2) PAINS score - Pan-assay interference compounds (PAINS) are those that 
       interfere with the detection methods of screening assays and are thus 
       problematic artefacts that have been identified to be widely used in many 
       scientific publications as chemical tools, thus leading to the wrong 
       conclusions. PAINS alert by giving them a PAINS score of 0.

    3) Selectivity - Noted if the drug hitting this target has been screened 
       against other targets and had 10-fold specificity for this targetSAR score - 
       essentially have there been a series of compounds generated for this 
       target-scaffold pairing

    4) SAR - Structure-Activity Relationships (SAR) increase confidence that the 
       biological effect of a given chemical tool is achieved via the modulation of 
       the reference target. Accordingly, the SAR score is a binary score measuring 
       whether there are (SAR Score = 1) known SAR for the compound-reference target 
       pair. To calculate the SAR Score we first calculate the level 1 of the 
       scaffold tree for all compounds in canSAR as it has been described to have 
       advantages over other scaffold definitions (Langdon et al., 2011). 
       Next, we consider a compound-reference target pair has SAR (SAR Score = 1) if 
       there is at least another compound reported in the same publication 
       (identical PubMedID) with the same level 1 scaffold active against the 
       reference target (pActivity > 5).

    5) SIC - The SIC is calculated as the summary of the differences between the median
       pActivity of the reference target and the pActivity of each off-target minus 
       one.

    6) Global - The Global Chemical Probe Score is a combination of the 
       previous 6 Chemical Probe Scores with customizable weights to allow chemical 
       biologists to prioritize the best chemical tools for the specific 
       requirements of their experiments, with the 6 being: pains, interactive 
       analog score, sar score, cell score, selectivity score, potency score.
    '''
    
    # Convert the input file to a generator
    data = get_file_records(args.dataset, parse_type = 'tsv', keep_header=False)

    # Open the output files
    with open(args.create_out, 'w') as f1, open(args.data_out, 'w') as f2:
        f1.write('\t'.join(['canSAR_id', 'attribute', 'value\n']))
        f2.write('\t'.join(['canSAR_id', 'uniprot_id', 'is_cell_potency', 'is_pains', 'sar_raw',
                            'is_suitable_probe', 'global', 'sic_min', 'sic_max', 'selectivity\n']))
        # Read, parse, and write the input data to disk
        for line in data:
            d = ast.literal_eval(line[-1])
            if d.get('chembl'):
                f1.write('\t'.join((line[1], 'chembl_id', d.get('chembl')+'\n')))
            if d.get('drugbank'):
                f1.write('\t'.join((line[1], 'drugbank_id', d.get('drugbank')+'\n')))
            if d.get('chembl') or d.get('drugbank'):
                f2.write('\t'.join((line[1], line[0], line[12], line[32], line[5],
                                      line[38], line[26], line[37], line[21], line[4]+'\n')))
        
