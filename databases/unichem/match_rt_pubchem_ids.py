#!/usr/bin/env python

import sys
sys.path.insert(1,"../../web1")
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
import django
django.setup()

import dtk.drug_clusters as dc

class UniChemMapping:
    def __init__(self,src1,src2):
        self.filename = "src%dsrc%d.txt.gz" % (src1,src2)
        self.fwd = {}
        self.rev = {}
        header = None
        import gzip
        for line in gzip.open(self.filename):
            if header is None:
                header = line
                continue
            fields = line.strip('\n').split('\t')
            self.fwd[fields[0]] = fields[1]
            self.rev[fields[1]] = fields[0]

def get_rt_pubchem_records(f):
    f.seek(0)
    f.readline() # discard header
    for line in f:
        fields = line.strip('\n').split('\t')
        cid = fields[0]
        if cid.startswith('CID'):
            cid=cid[3:]
        cid = cid.lstrip('0')
        fields[0] = cid
        yield fields

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='unichem utility')
    args = parser.parse_args()

    m_chembl = UniChemMapping(1,22)
    m_drugbank = UniChemMapping(2,22)
    clr = dc.Clusterer()
    clr.load_archive()
    matched_drugbank = 0
    mapping = {}
    f=open('../../../ws/drugsets/rt.pubchem.full.test.tsv')
    for fields in get_rt_pubchem_records(f):
        cid = fields[0]
        if cid in m_drugbank.rev:
            matched_drugbank += 1
        if cid not in mapping:
            d = {
                'seen':0,
                }
            if cid in m_chembl.rev:
                ch_id = m_chembl.rev[cid]
                d['chembl'] = ch_id
                ch_key = ('chembl_id',ch_id)
                if ch_key in clr.drugs_by_key:
                    d['mapped_chembl'] = clr.drugs_by_key[ch_key]
            if cid in m_drugbank.rev:
                d['drugbank'] = m_drugbank.rev[cid]
            mapping[cid] = d
        mapping[cid]['seen'] += 1
    print 'distinct cids in input:',len(mapping)
    print 'mapped to drugbank via unichem:',len(
            [x for x in mapping.values() if 'drugbank' in x]
            )
    print 'mapped to chembl via unichem:',len(
            [x for x in mapping.values() if 'chembl' in x]
            )
    clust_mappings = [x for x in mapping.values() if 'mapped_chembl' in x]
    print '... then mapped to cluster:',len(
            clust_mappings
            )
    clr.build_links()
    drugbank_clusters = 0
    new_drugbank_clusters = 0
    for d in clust_mappings:
        s = d['mapped_chembl'].get_cluster_as_set()
        for drug in s:
            if 'drugbank_id' in drug.key:
                d['mapped_drugbank'] = drug.key[1]
                drugbank_clusters += 1
                if 'drugbank' not in d:
                    new_drugbank_clusters += 1
                break
    print '... that included drugbank:',drugbank_clusters
    print '... and weren\'t found directly:',new_drugbank_clusters
    print 'not mapped by any method:',len(
            [x for x in mapping.values() if x.keys() == ['seen']]
            )
    print 'total input records:',sum(
            [x['seen'] for x in mapping.values()]
            )
    print 'total records not mapped by any method:',sum(
            [x['seen'] for x in mapping.values() if x.keys() == ['seen']]
            )
    out = open('rt.drugbank.full.test.tsv','w')
    out.write('drugbank_id\tattribute\tvalue\n')
    for fields in get_rt_pubchem_records(f):
        if fields[0] not in mapping:
            continue
        if fields[2] == '':
            continue # these are in the original for some reason
        d = mapping[fields[0]]
        if 'drugbank' in d:
            db_id = d['drugbank']
        elif 'mapped_drugbank' in d:
            db_id = d['mapped_drugbank']
        else:
            continue
        out.write('\t'.join([db_id]+fields[1:])+'\n')
