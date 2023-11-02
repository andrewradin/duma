#!/usr/bin/env python

#Code written by Jeremy Hunt

######
#import packages
######
import numpy as np
import pandas as pd
from collections import Counter
import pprint as pp
import json

import sys
from dtk.files import get_file_lines

######
#functions
######

def print_dict(dictionary, ident = '', braces=1):
    """ Recursively prints nested dictionaries."""
    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            print('%s%s%s%s' %(ident,braces*'[',key,braces*']'))
            print_dict(value, ident+'  ', braces+1)
        else:
            print(ident+'%s = %s' %(key, value))

def make_ensembl_2_uniprot_map(file):
    conv_d = {}
    with open(file, 'r') as f:
        for l in f:
            fields = l.rstrip("\n").split("\t")
            if fields[1] == 'Ensembl':
                try:
                    conv_d[fields[2]].add(fields[0])
                except KeyError:
                    conv_d[fields[2]] = set()
                    conv_d[fields[2]].add(fields[0])
    return conv_d

def load_disease_data(data_dir):
    import pyarrow.parquet as pq
    from tqdm import tqdm
    import os

    cols = ['name', 'id']

    out = {}
    
    for fn in tqdm(os.listdir(data_dir), desc='Disease data'):
        fn = os.path.join(data_dir, fn)
        if not fn.endswith('.parquet'):
            continue
        data = pq.read_table(fn, columns=cols).to_pydict()
        for (name, id) in zip(*data.values()):
            out[id] = name

    return out


######
#parser
######

if __name__=='__main__':
    import argparse #for adding arguments to a .py script when running from terminal (treats args as strings)
    parser = argparse. ArgumentParser(description='Parse OpenTargets target-disease association scores')
    parser.add_argument('--otarg-disease-dir', help = 'Directory with opentargets disease parquet files')
    parser.add_argument('--otarg-overall-dir', help = 'Directory with opentargets overall associations parquet files')
    parser.add_argument('--otarg-bytype-dir', help = 'Directory with opentargets by-type associations parquet files')
    parser.add_argument('-u', '--uniprot_converter', help = '~/2xar/ws/HUMAN_9606_Uniprot_data.tsv')
    args = parser.parse_args()
    conv_d = make_ensembl_2_uniprot_map(args.uniprot_converter)

    dis_id_to_name = load_disease_data(args.otarg_disease_dir)

    #######
    #Association data
    #######
    import dtk.open_targets as ot
    wr = ot.Writer('temp')

    from tqdm import tqdm
    import os
    import pyarrow.parquet as pq


    dirname = args.otarg_overall_dir
    cols = ['targetId', 'diseaseId', 'score']
    for fn in tqdm(os.listdir(dirname), desc='Overall data'):
        fn = os.path.join(dirname, fn)
        if not fn.endswith('.parquet'):
            continue
        data = pq.read_table(fn, columns=cols).to_pydict()

        for targetId, diseaseId, overallScore in zip(*data.values()):
            scores = {}
            target_name_ensembl = targetId
            target_name_uniprot_set = conv_d.get(target_name_ensembl,[])
            disease_name = dis_id_to_name[diseaseId]
            disease_key = diseaseId
            scores['overall'] = overallScore

            wr.add(disease_name,disease_key,target_name_uniprot_set,scores)

    dirname = args.otarg_bytype_dir
    cols = ['targetId', 'diseaseId', 'datatypeId', 'score']
    for fn in tqdm(os.listdir(dirname), desc='By Type data'):
        fn = os.path.join(dirname, fn)
        if not fn.endswith('.parquet'):
            continue
        data = pq.read_table(fn, columns=cols).to_pydict()

        for targetId, diseaseId, datatypeId, score in zip(*data.values()):
            scores = {}
            target_name_ensembl = targetId
            target_name_uniprot_set = conv_d.get(target_name_ensembl,[])
            disease_name = dis_id_to_name[diseaseId]
            disease_key = diseaseId
            scores[datatypeId] = score
            wr.add(disease_name,disease_key,target_name_uniprot_set,scores)
    wr.close()
