#!/usr/bin/env python

import pwd,os
user=pwd.getpwuid(os.getuid())[0]
root='/home/%s/2xar/' % user

import sys
sys.path.insert(1,root+'twoxar-demo/databases/chembl/')
sys.path.insert(1,root+'twoxar-demo/web1/')

def setup_django():
    if not 'django' in sys.modules:
        print 'loading django'
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
        import django
        django.setup()

def load_groups(fn,quiet=False):
    '''Return contents of DPI file as a dict.

    fn is the path to the DPI file.
    returns { <drug>:{<uniprot>:(ev,dir),...},...}
    '''
    result = dict()
    from dtk.files import get_file_records
    for fields in get_file_records(fn,keep_header=False):
        d = result.setdefault(fields[0],dict())
        val = (float(fields[2]),int(fields[3]))
        try:
            if d[fields[1]] != val and not quiet:
                print 'conflict',fields,d
                # one source of these is having different values for two
                # different uniprot aliases; this should get resolved in
                # an earlier processing stage
        except KeyError:
            pass
        d[fields[1]] = val
    return result

def load_structure(fn):
    from dtk.data import MultiMap
    from dtk.files import get_file_records
    mm = MultiMap(
            (fields[0],fields[2].split('/')[1])
            for fields in get_file_records(fn,keep_header=False)
            if fields[1] == 'inchi'
            )
    return dict(MultiMap.flatten(mm.fwd_map()))

def load_inchis(fn):
    from dtk.data import MultiMap
    from dtk.files import get_file_records
    mm = MultiMap(
            (fields[0],fields[2])
            for fields in get_file_records(fn,keep_header=False)
            if fields[1] == 'inchi'
            )
    return dict(MultiMap.flatten(mm.fwd_map()))

class ClusterAccumulator:
    def __init__(self):
        self.cur_labels = set()
        self.last_val = None
        self.clusters = []
        self.dpi = []
    def flush(self):
        if len(self.cur_labels) > 0:
            self.clusters.append(self.cur_labels)
            self.dpi.append(self.last_val)
        self.cur_labels = set()
        self.last_val = None
    def add(self,label,val):
        if val != self.last_val:
            self.flush()
        self.last_val=val
        self.cur_labels.add(label)

def compare_similarity(indent,inchi_map,drug_ids):
    # get fingerprint of each drug
    from rdkit import Chem
    from rdkit.Chem import AllChem
    radius = 2
    fps = {}
    for key in drug_ids:
        if key not in inchi_map:
            print "no inchi for",key
            continue
        mol = Chem.inchi.MolFromInchi(inchi_map[key])
        if not mol:
            print "conversion failed for",key
            continue
        fps[key] = AllChem.GetMorganFingerprint(mol,radius)
    # get pairwise similarity
    similarities = []
    from rdkit import DataStructs
    fps_list = fps.items()
    for i,fp in enumerate(fps_list):
        for j in range(i):
            sim = DataStructs.DiceSimilarity(fp[1],fps_list[j][1])
            similarities.append((sim,fp[0],fps_list[j][0]))
    similarities.sort()
    print indent,len(similarities),'pairwise comparisons'
    if not similarities:
        return
    print indent,'minimum',similarities[0]
    print indent,'maximum',similarities[-1]
    if False:
        bin_size = max(1,len(similarities)/10)
        i = bin_size/2
        while i < len(similarities):
            print indent,i,similarities[i]
            i += bin_size
    # build clusters
    # This is a modification of the clusters2 algorithm from dtk.similarity.
    # The idea here is to avoid pre-specifying a threshold by merging drugs
    # until every drug has at least one partner.
    # 
    # This will get confused by single outlier drugs (which will cause it to
    # keep merging beyond the natural cluster boundaries to pull in the
    # outlier), so maybe it should have a threshold cutoff as well (although
    # the output of this run doesn't indicate this is a big issue).
    #
    # It may also stop prematurely, if the gaps between drugs are somewhat
    # uneven, but again that doesn't seem too bad in this case.
    #
    # XXX eventually, if this is useful, move it to SimilarityMatrix class,
    # XXX and maybe add conversion functions between the matrix and the
    # XXX similarities list format
    clusters = {}
    revmap = {}
    for sim,drug1,drug2 in reversed(similarities):
        if len(revmap) == len(fps_list):
            break
        cid1 = revmap.get(drug1)
        cid2 = revmap.get(drug2)
        if not cid1 and not cid2:
            # make a new cluster
            cid = 1+len(clusters)
            s = set([drug1,drug2])
            clusters[cid] = set([drug1,drug2])
            revmap[drug1] = cid
            revmap[drug2] = cid
        elif not cid1:
            # add cid1 to cid2's cluster
            s = clusters[cid2]
            s.add(drug1)
            revmap[drug1] = cid2
        elif not cid2:
            # add cid2 to cid1's cluster
            s = clusters[cid1]
            s.add(drug2)
            revmap[drug2] = cid1
        elif cid1 == cid2:
            # already both known and in the same cluster
            pass
        else:
            #print 'merging',len(clusters[cid1]),len(clusters[cid2]),sim
            # merge the two clusters
            s = clusters[cid1]
            s |= clusters[cid2]
            for drug_id in clusters[cid2]:
                revmap[drug_id] = cid1
            # leave cluster2 in place so cid generation works;
            # it's no longer pointed to by revmap
    valid_cids = set(revmap.values())
    print indent,'%d structural subclusters found; max %d, cutoff %f'%(
            len(valid_cids),
            len(clusters),
            sim,
            )
    if len(valid_cids) > 1:
        for cid in valid_cids:
            print indent,clusters[cid]

def group_dpi(args):
    # construct DPI-level clusters
    groups = load_groups(args.path,quiet=True)
    l = list(groups.iteritems())
    l.sort(key=lambda x:x[1])
    ca = ClusterAccumulator()
    for drug,dpi in l:
        ca.add(drug,dpi)
    # order clusters by size
    l = zip(ca.dpi,ca.clusters)
    print len(l),'dpi clusters'
    l.sort(key=lambda x:len(x[1]),reverse=True)
    # show cluster size histogram
    print 'starting inchi load'
    inchi_map = load_inchis('stage_drugsets/create.chembl.full.tsv')
    print 'inchi load complete'
    if False:
        level = 1024
        count = 0
        for size in [len(x[1]) for x in l]:
            if size < level:
                print count,'>=',level
                level /= 2
                count = 0
            count += 1
        print count,'>=',level
    # show dpi info for top clusters
    if False:
        for dpi,s in l[:50]:
            print len(s),dpi
    # show structure for top cluster
    if False:
        struct = load_structure('stage_drugsets/create.chembl.full.tsv')
        from collections import Counter
        ctr = Counter()
        for key in l[0][1]:
            ctr[struct[key]] += 1
        print ctr
    if False:
        print list(l[0][1])[:10]
        group_by_doc(l[0][1])
        # the above shows a couple of docs with 300+ chemicals, one with 246,
        # a few with 50-80, several with around 10-30, and lots with 1
    # dump out all info, with doc_id subclusters within each dpi cluster
    for dpi,s in l:
        print '======',len(s),'drugs with dpi',dpi
        dg = DocGrouper2(s)
        print len(dg.leftover),'drugs not in doc clusters'
        for drug_id in dg.leftover:
            print "    ",drug_url(drug_id)
        for subcluster in dg.subclusters:
            print "  ---- %d (-%d) drugs for %s"%(
                    len(subcluster.drug_ids),
                    subcluster.removed,
                    doc_url(subcluster.doc_id),
                    )
            compare_similarity("  ",inchi_map,subcluster.drug_ids)
            for drug_id in subcluster.drug_ids:
                print "    ",drug_url(drug_id)
    # XXX To check clusters by associated documents:
    # XXX - to mirror what ACD does manually:
    # XXX   - check doc_id field in Activities table for matching molregno
    # XXX     - (should only be one record? or one doc_id across multiple
    # XXX       records?)
    # XXX     - (molecule w/ most records is probable lead molecule?)
    # XXX - there's also a (set of) doc_id(s) accessible through compound
    # XXX   records; these might be more promiscuous

def doc_url(chembl_id):
    return 'https://www.ebi.ac.uk/chembl/beta/document_report_card/'+chembl_id

def drug_url(chembl_id):
    return 'https://www.ebi.ac.uk/chembl/beta/compound_report_card/'+chembl_id

# The following classes perform alternative implementations of the a service
# to subcluster chembl ids by documents that they appear in. The API is:
# - a set of chembl ids is passed the the __init__ functions
# - the 'subcluster' member contains a list of objects with the following
#   properties:
#   - doc_id - the chembl document used to define the subcluster
#   - drug_ids - a set of chembl ids in the subcluster
#   - removed - a count of chembl ids associated with the doc, but not
#     included in the cluster because of other conditions imposed by the
#     clustering algorithm
# - the 'leftover' member contains a set of chembl ids not in any subcluster
#
# DocGrouper1 creates a subcluster for the doc containing the largest number
# of chembl ids in the set. It then repeats this process for the remaining
# chembl ids until there are no docs left containing more than one ungrouped
# chembl id.
class DocGrouper1:
    def __init__(self,chembl_ids):
        c2d = chembl2doc(chembl_ids)
        filtered = c2d.rev_map().iteritems()
        d2l = doc2label(c2d.rev_map().keys())
        from collections import Counter
        removed = Counter()
        self.subclusters = []
        class Dummy: pass
        while filtered:
            ordered = sorted(filtered,key=lambda x:-len(x[1]))
            key,content = ordered[0]
            sub = Dummy()
            self.subclusters.append(sub)
            sub.doc_id = d2l[key]
            sub.drug_ids = content
            sub.removed = removed[key]
            filtered = []
            for doc,drugs in ordered[1:]:
                drugs2 = drugs-content
                if drugs != drugs2:
                    removed[doc] += len(drugs-drugs2)
                if len(drugs2) > 1:
                    filtered.append((doc,drugs2))
        self.leftover = chembl_ids
        for sub in self.subclusters:
            self.leftover -= sub.drug_ids
#
# DocGrouper2 creates a subcluster for each doc containing more than one
# chembl id, where that chembl id is not contained in any other doc.
class DocGrouper2:
    def __init__(self,chembl_ids):
        c2d = chembl2doc(chembl_ids)
        d2l = doc2label(c2d.rev_map().keys())
        single_doc_drugs = set([
                drug
                for drug,docs in c2d.fwd_map().iteritems()
                if len(docs) == 1
                ])
        filtered = [
                (doc,drugs & single_doc_drugs)
                for doc,drugs in c2d.rev_map().iteritems()
                ]
        filtered.sort(key=lambda x:-len(x[1]))
        self.subclusters = []
        class Dummy: pass
        for doc,drugs in filtered:
            if len(drugs) <= 1:
                break
            sub = Dummy()
            self.subclusters.append(sub)
            sub.doc_id = d2l[doc]
            sub.drug_ids = drugs
            sub.removed = len(c2d.rev_map()[doc])-len(drugs)
        self.leftover = chembl_ids
        for sub in self.subclusters:
            self.leftover -= sub.drug_ids

def group_by_doc(chembl_ids):
    # extract the largest set, then filter the remaining sets
    # by the ones already grouped, and repeat until there are no more
    # sets over some minimum size
    c2d = chembl2doc(chembl_ids)
    filtered = c2d.rev_map().iteritems()
    while filtered:
        ordered = sorted(filtered,key=lambda x:-len(x[1]))
        print 'doc_id',ordered[0][0],'refs',len(ordered[0][1]),'new drugs'
        filtered = []
        for doc,drugs in ordered[1:]:
            drugs2 = drugs-ordered[0][1]
            if drugs != drugs2:
                print 'removing',len(drugs-drugs2),'of',len(drugs),'from',doc
            if len(drugs2) > 1:
                filtered.append((doc,drugs2))

def doc2label(doc_ids):
    import chembl_schema as ch
    result = dict(
            ch.Docs.select(
                            ch.Docs.doc,
                            ch.Docs.chembl,
                    ).where(
                            ch.Docs.doc << doc_ids
                    ).tuples()
            )
    return result

def chembl2doc(chembl_ids):
    import chembl_schema as ch
    molregno2chembl=dict(
                ch.MoleculeDictionary.select(
                                ch.MoleculeDictionary.molregno,
                                ch.MoleculeDictionary.chembl_id,
                        ).where(
                                ch.MoleculeDictionary.chembl << chembl_ids
                        ).tuples()
                )
    #print molregno2chembl
    molregno2doc=ch.Activities.select(
                    ch.Activities.molregno,
                    ch.Activities.doc,
            ).where(
                    ch.Activities.molregno << molregno2chembl.keys()
            ).tuples()
    from dtk.data import MultiMap
    return MultiMap(
            (molregno2chembl[mrn],doc)
            for mrn,doc in molregno2doc
            )

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='''\
''',
            )
    parser.add_argument('cmd',
            )
    parser.add_argument('--path',
            )
    args = parser.parse_args()
    locals()[args.cmd](args)

    
