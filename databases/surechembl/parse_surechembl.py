#!/usr/bin/env python

if __name__=='__main__':
    import sys
    try:
        from dtk.files import get_file_records
    except ImportError:
        sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
        from dtk.files import get_file_records
    import argparse
    import pandas as pd
    import timeit
    import gzip

    # Set up the argument parser
    arguments = argparse.ArgumentParser(description='''Parse SureChEMBL drug
        and patent data into two .tsv files. Make sure your input file has been
        sorted by SureChEMBL ID and then by patent id first!''')
    arguments.add_argument("dataset", help='''Desired SureChEMBL dataset to
        download and unzip. For example: "SureChEMBL_map_20180401"''')
    arguments.add_argument("databases", help='''Database IDs to include for
        matching to SureChEMBL IDs, comma-separated. For example:
        "chembl_to_surechembl.tsv.gz,drugbank_to_surechembl.tsv.gz"''')
    arguments.add_argument("subsets",  help='''Subsets of the IDs for the drugs
        we are interested in, comma-separated. For example:
        "our_chembl.tsv,our_drugbank.tsv"''')
    arguments.add_argument("-drug_out", nargs = '?', help='''Output file name of
        drug data. The default is: "SureChEMBL_drugProperties.tsv"''',
        default = "SureChEMBL_drugProperties.tsv")
    arguments.add_argument("-patent_out", nargs = '?', help='''Output file name
        of patent data. The default is: "SureChEMBL_patentProperties.tsv"''',
        default ="SureChEMBL_patentProperties.tsv")
    args = arguments.parse_args()

    # Make a list out of the databases and subsets arguments
    databases = str(args.databases).split(",")
    subsets = str(args.subsets).split(",")

    # Re-format the input file names for later use
    db_names = []
    for base in databases:
        db_names.append(base.split('_')[0].upper() + '_ID')

    # Read in the database ID dataframes
    # Unzip and store database ID to surechembl ID data
    dfs = []
    for i, name in enumerate(databases):
        with gzip.open(name) as f:
            df = pd.read_table(f, names = [db_names[i], 'SCHEMBL_ID'])
            df = df.loc[1:]
            dfs.append(df)

    # Subset the ID data on the drugs we are interested in
    # Merge the ID data and convert to dictionaries
    id_dicts = []
    for i, name in enumerate(subsets):
        id_df = pd.read_csv(name, delimiter='\t', header=None,\
            names=[db_names[i]])
        id_df = pd.merge(id_df, dfs[i], how = 'left', on = db_names[i])
        id_df.dropna(inplace=True)
        id_dict = dict(zip(id_df['SCHEMBL_ID'], id_df[db_names[i]]))
        id_dicts.append(id_dict)

    #Create all schembl IDs as key dictionary 
    masterid_dict = {}
    for d in id_dicts:
        for k, v in d.iteritems():
            masterid_dict.setdefault(k, v)

    # Open the schembl file of interest and ID dictionaries
    counter = True
    patent_data = [0,0,0]
    prev_patentdrug_combo = None
    lines_read = 0
    schembl_gen = (x for x in get_file_records(args.dataset, parse_type='tsv') if x[0] in masterid_dict)

    # Open new files to write drug data and patent data to
    with open(args.drug_out, 'w') as f1, open(args.patent_out, 'w') as f2:
        #Write the headers of each file
        f1.write('\t'.join(['schembl_id', 'attribute', 'value']) + '\n')
        f2.write('\t'.join(['schembl_id', 'patent_id', 'title_count',
            'abstract_count', 'claims_count', 'publication_date']) + '\n')
        start_time = timeit.default_timer()
        print '\nProgram has started!'
        print 'Remember: sort your data by SureChEMBL ID and patent ID first!'
        # Initiate for-loop through schembl data generator
        for iteration, item in enumerate(schembl_gen, 1):

            # Print out updates to the command line
            if iteration % 1000000 == 0:
                elapsed = timeit.default_timer() - start_time
                print 'Read {} total lines in: {} seconds.'.format(iteration, int(elapsed))

            # item[4] is the patent ID and item[0] is the surechembl ID
            patentdrug_combo = [item[4], item[0], item[5]]

            # Conditional to check if first run through of loop or new drug
            # Write to drug data file if drug is new and in our IDs of interest
            if counter == True or item[0] != prev_patentdrug_combo[1]:
                counter = False
                write_drug = []
                for i, d in enumerate(id_dicts):
                    try:
                        write_drug.append('\t'.join([item[0], db_names[i].lower(),
                            d[item[0]]]) + '\n')
                    except KeyError:
                        pass
                write_drug.append('\t'.join([item[0], 'smiles_code', item[1]]) + '\n'\
                    + '\t'.join([item[0], 'inchi_key', item[2]]) + '\n'\
                    + '\t'.join([item[0], 'surechembl_frequency', item[3]]) + '\n')
                f1.write(''.join(write_drug))

            # Conditional to check if new patent-drug combination
            # Write to patent data file if patent-drug combination contains field counts
            if patentdrug_combo != prev_patentdrug_combo:
                if sum(patent_data):
                    write_patent = ['\t'.join([prev_patentdrug_combo[1], prev_patentdrug_combo[0], str(patent_data[0]),
                        str(patent_data[1]), str(patent_data[2]), prev_patentdrug_combo[2]]) + '\n']
                    f2.write(''.join(write_patent))
                patent_data = [0,0,0]

            # Store relevant field count data
            # item[6] is field id: a value of 2 is claims, 3 is abstract, 4 is title
            # item[7] is the field frequency for that given field
            prev_patentdrug_combo = [item[4], item[0], item[5]]
            patent_field_map = {'2': 2, '3': 1, '4': 0}
            if item[6] in patent_field_map:
                patent_data[patent_field_map[item[6]]] = int(item[7])

        # Write the last iteration of the loop to disk if it satisfies the conditional
        if sum(patent_data):
            write_patent = ['\t'.join([prev_patentdrug_combo[1], prev_patentdrug_combo[0], str(patent_data[0]),
                str(patent_data[1]), str(patent_data[2]), prev_patentdrug_combo[2]]) + '\n']
            f2.write(''.join(write_patent))

    # Print out the total number of lines and run time to the command line
    elapsed = timeit.default_timer() - start_time
    print 'Congratulations!'

