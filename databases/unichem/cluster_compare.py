#!/usr/bin/env python

# This file allows a quick comparison of the performance of the drug
# clusterer and unichem between a pair of drug collections. Before
# running, unichem.py must be run by hand to download the pair file
# of interest, and all needed drugset files must be in ws/drugsets.

def get_ingredients(matching_ver):
    from dtk.s3_cache import S3File
    s3f=S3File('matching',f'matching.full.v{matching_ver}.ingredients.tsv')
    s3f.fetch()
    paths = open(s3f.path()).read().strip().split()
    split_fns = [x.split('/')[-1].split('.') for x in paths]
    return {
            parts[0]:int(parts[2][1:])
            for parts in split_fns
            }

from dtk.lazy_loader import LazyLoader
class CollectionInfo(LazyLoader):
    _kwargs=['file_class','version']
    def _keys_loader(self):
        from dtk.s3_cache import S3File
        s3f=S3File(
                self.file_class,
                f'{self.file_class}.full.v{self.version}.attributes.tsv',
                )
        s3f.fetch()
        from dtk.files import get_file_records
        return set(
                rec[0]
                for rec in get_file_records(s3f.path(),keep_header=False)
                )

def get_cluster_matches(matching_ver,col1info,col2info):
    from dtk.s3_cache import S3File
    s3f=S3File('matching',f'matching.full.v{matching_ver}.clusters.tsv')
    s3f.fetch()
    from dtk.files import get_file_records
    from dtk.drug_clusters import assemble_pairs
    for rec in get_file_records(s3f.path(),keep_header=False):
        col1keys=set()
        col2keys=set()
        for col,key in assemble_pairs(rec):
            if col == col1info.file_class + '_id':
                col1keys.add(key)
            elif col == col2info.file_class + '_id':
                col2keys.add(key)
        for k1 in col1keys:
            for k2 in col2keys:
                yield (k1,k2)

def get_unichem_matches(col1,col2):
    stems = (col1.file_class,col2.file_class)
    fn_pat = '%s_to_%s.tsv.gz'
    from dtk.files import get_file_records
    import os
    for ordering in (lambda x:x,reversed):
        cols = tuple(ordering(stems))
        path = fn_pat%cols
        if os.path.exists(path):
            for rec in get_file_records(path,keep_header=False):
                yield ordering(rec)
            return
    raise RuntimeError("Can't find a unichem file")

def filter_matches(matches,col1keys,col2keys):
    return set(
            (key1,key2)
            for key1,key2 in matches
            if key1 in col1keys and key2 in col2keys
            )

def show_one_sided(label,s):
    indent = '    '
    print(len(s),'found by',label,'only')
    n_examples = 10
    for example in list(s)[:n_examples]:
        print(indent,example)
    if len(s) > n_examples:
        print(indent,'...')


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='''\
''',
            )
    parser.add_argument('matching_ver',type=int)
    parser.add_argument('collection1')
    parser.add_argument('collection2')
    args = parser.parse_args()
    col1 = args.collection1
    col2 = args.collection2
    # get collection versions from ingredients file
    coll_versions = get_ingredients(args.matching_ver)
    col1info = CollectionInfo(file_class=col1,version=coll_versions[col1])
    print(f'got {len(col1info.keys)} from {col1info.file_class}')
    col2info = CollectionInfo(file_class=col2,version=coll_versions[col2])
    print(f'got {len(col2info.keys)} from {col2info.file_class}')
    raw_cluster_matches=set(
            get_cluster_matches(args.matching_ver,col1info,col2info)
            )
    print(f'got {len(raw_cluster_matches)} raw cluster matches')
    cluster_matches = filter_matches(
            raw_cluster_matches,
            col1info.keys,
            col2info.keys,
            )
    raw_unichem_matches=set(
            get_unichem_matches(col1info,col2info),
            )
    print(f'got {len(raw_unichem_matches)} raw unichem matches')
    unichem_matches = filter_matches(
            raw_unichem_matches,
            col1info.keys,
            col2info.keys,
            )
    # filtered matches are the ones where both keys are
    # in the subsets we extract
    print(len(cluster_matches),'filtered cluster matches')
    print(len(unichem_matches),'filtered unichem matches')
    print(len(cluster_matches & unichem_matches),'found by both')
    show_one_sided('cluster',cluster_matches - unichem_matches)
    show_one_sided('unichem',unichem_matches - cluster_matches)
