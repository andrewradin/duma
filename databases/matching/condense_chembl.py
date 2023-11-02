#!/usr/bin/env python


import pwd,os
user=pwd.getpwuid(os.getuid())[0]
root='/home/%s/2xar/' % user

import sys
sys.path.insert(1,root+'twoxar-demo/databases/chembl/')
sys.path.insert(1,root+'twoxar-demo/web1/')

import dtk.rdkit_2019
################################################################################
# DPI clustering tools
################################################################################
def load_dpi_as_dict(fn,quiet=False):
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
                print('conflict',fields,d)
                # one source of these is having different values for two
                # different uniprot aliases; this should get resolved in
                # an earlier processing stage
        except KeyError:
            pass
        d[fields[1]] = val
    return result

def canonical_dpi(d):
    '''Return a consistent hashable tuple for a DPI dict.

    The DPI dict is assumed to be {<uniprot>:(ev,dir),...}.
    '''
    return tuple(
            (key,d[key])
            for key in sorted(d.keys())
            )

################################################################################
# doc clustering tools
################################################################################
class ChemblDocInfo(object):
    @staticmethod
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

    @staticmethod
    def id2doc(chembl_ids):
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
                        ch.Activities.molregno << list(molregno2chembl.keys())
                ).tuples()
        from dtk.data import MultiMap
        return MultiMap(
                (molregno2chembl[mrn],doc)
                for mrn,doc in molregno2doc
                )

    @staticmethod
    def choose_exemplar(chembl_ids):
        import chembl_schema as ch
        # count Activities for each drug
        from collections import Counter
        ctr = Counter(
                x[0]
                for x in ch.Activities.select(
                                ch.MoleculeDictionary.chembl_id
                        ).join(
                                ch.MoleculeDictionary
                        ).where(
                                ch.MoleculeDictionary.chembl << chembl_ids
                        ).tuples()
                )
        if len(ctr) == 0:
            print(chembl_ids)
            raise NotImplementedError('No Activities')
        # get all drugs tied for highest count
        highest_count = ctr.most_common(1)[0][1]
        ties = []
        for drug,count in ctr.most_common():
            if count < highest_count:
                break
            ties.append(drug)
        # as a canonical tie-breaker, use a lexical sort
        ties.sort()
        return ties[0]

# DocGrouper2 creates a subcluster for each doc containing more than one
# chembl id, where that chembl id is not contained in any other doc.
class DocGrouper2:
    def __init__(self,chembl_ids,doc_info):
        c2d = doc_info.id2doc(chembl_ids)
        d2l = doc_info.doc2label(c2d.rev_map().keys())
        single_doc_drugs = set([
                drug
                for drug,docs in c2d.fwd_map().items()
                if len(docs) == 1
                ])
        filtered = [
                (doc,drugs & single_doc_drugs)
                for doc,drugs in c2d.rev_map().items()
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

################################################################################
# create file scanning tools
################################################################################
def group_drugs(src):
    last_key = None
    last_batch = []
    for fields in src:
        if fields[0] != last_key:
            if last_key:
                yield (last_key,last_batch)
            last_key = fields[0]
            last_batch = []
        last_batch.append(fields)
    if last_key:
        yield (last_key,last_batch)

################################################################################
# structural clustering tools
################################################################################
def extract_similarities(inchi_map,drug_ids):
    '''Return [(similarity,drug_id_1,drug_id_2),...] for each pair of drugs.
    '''
    # get fingerprint of each drug
    from rdkit import Chem
    from rdkit.Chem import AllChem
    radius = 2
    fps = {}
    for key in drug_ids:
        if key not in inchi_map:
            continue
        mol = Chem.MolFromInchi(inchi_map[key])
        if not mol:
            continue
        fps[key] = AllChem.GetMorganFingerprint(mol,radius)
    # get pairwise similarity
    similarities = []
    from rdkit import DataStructs
    fps_list = list(fps.items())
    for i,fp in enumerate(fps_list):
        for j in range(i):
            sim = DataStructs.DiceSimilarity(fp[1],fps_list[j][1])
            similarities.append((sim,fp[0],fps_list[j][0]))
    return similarities

def cluster_by_similarity(similarities):
    '''Return a list of sets of drug ids.
    '''
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
    similarities.sort() # least to most similar pair
    all_drug_ids = set()
    for sim,drug1,drug2 in similarities:
        all_drug_ids.add(drug1)
        all_drug_ids.add(drug2)
    clusters = {}
    revmap = {}
    for sim,drug1,drug2 in reversed(similarities):
        if len(revmap) == len(all_drug_ids):
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
    return [clusters[cid] for cid in valid_cids]

def compare_similarity(inchi_map,drug_ids):
    #return [drug_ids] # XXX bypass structural clustering
    similarities = extract_similarities(inchi_map,drug_ids)
    return cluster_by_similarity(similarities)

################################################################################
# Top-level condenser class
################################################################################
class ChemblCondenser:
    def __init__(self,dpi_fn,create_fn,doc_info):
        self.create_fn = create_fn
        self.build_clusters(dpi_fn, doc_info)
    def build_clusters(self,dpi_fn, doc_info):
        from dtk.files import get_file_records
        drug_subset = set([
                fields[0]
                for fields in get_file_records(
                            self.create_fn,
                            keep_header=False,
                            )
                ])
        print(len(drug_subset),'drugs in collection subset')
        # load inchis
        from dtk.data import MultiMap
        mm = MultiMap(
                (fields[0],fields[2])
                for fields in get_file_records(self.create_fn,keep_header=False)
                if fields[1] == 'inchi'
                )
        inchi_map = dict(MultiMap.flatten(mm.fwd_map()))
        print(len(inchi_map),'drugs with inchis')
        # read DPI info and make DPI clusters
        drug2dpi = MultiMap(
                (drug,canonical_dpi(dpi))
                for drug,dpi in load_dpi_as_dict(dpi_fn).items()
                if drug in drug_subset
                )

        print(len(drug2dpi.rev_map()),'distinct DPI signatures')

        # In the condensed collections, we won't bother including anything
        # that we don't have DPI data for.
        # For chembl we pre-filter for c50/ki data, but that happens before
        # we convert to evidence values, so there are a lot of extra drugs
        # that drop out here.
        self.drugs_missing_dpi = set()
        for drug in drug_subset:
            if drug not in drug2dpi.fwd_map():
                self.drugs_missing_dpi.add(drug)

        # create doc subclusters within each DPI cluster; build a map
        # from subcluster members to subcluster exemplar
        self.drug2exemplar = {}

        print("%d drugs missing DPI, dropping" % len(self.drugs_missing_dpi))

        for dpi,drugset in drug2dpi.rev_map().items():
            if len(drugset) == 1:
                continue
            dg = DocGrouper2(drugset, doc_info)
            for subcluster in dg.subclusters:
                if len(subcluster.drug_ids) < 2:
                    continue
                for struct_subcluster in compare_similarity(
                            inchi_map,
                            subcluster.drug_ids,
                            ):
                    exemplar = doc_info.choose_exemplar(struct_subcluster)
                    for drug_id in struct_subcluster:
                        self.drug2exemplar[drug_id] = exemplar
        mm = MultiMap(self.drug2exemplar.items())
        self.exemplar2drugs = mm.rev_map()
        print('grouped %d drugs in %d clusters'%(
                len(self.drug2exemplar),
                len(self.exemplar2drugs),
                ))
    def condense(self,output_fn,shadow_attr_name):
        from dtk.files import FileDestination, get_file_records
        drugs_in = 0
        drugs_out = 0
        with FileDestination(output_fn) as outp:
            # note that the first 'drug' through the loop below is
            # actually the header line
            for key,recs in group_drugs(get_file_records(self.create_fn)):
                drugs_in += 1
                exemplar = self.drug2exemplar.get(key)
                if exemplar and exemplar != key:
                    continue # skip this drug
                if key in self.drugs_missing_dpi:
                    continue # skip this drug
                drugs_out += 1
                # copy drug record
                for rec in recs:
                    outp.append(rec)
                # if this is an exemplar, output the drugs being shadowed
                if exemplar:
                    for other in self.exemplar2drugs[exemplar]:
                        if other == key:
                            continue # don't link to self
                        outp.append([key,shadow_attr_name,other])
        print('reduced',drugs_in-1,'input drugs to',drugs_out-1,'output drugs')
    def filter_m_file(self,input_fn,output_fn):
        from dtk.files import FileDestination, get_file_records
        inp = get_file_records(input_fn)
        with FileDestination(output_fn,header=next(inp)) as outp:
            for fields in inp:
                exemplar = self.drug2exemplar.get(fields[0])
                if exemplar and exemplar != fields[0]:
                    continue # skip this drug
                if fields[0] in self.drugs_missing_dpi:
                    continue # skip this drug
                outp.append(fields)

################################################################################
# main
################################################################################
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='''Condense ChEMBL Drugs\
''',
            )
    # Currently, the Makefile just passes a collection name, and
    # all paths are constructed below. We could move this back
    # up into the Makefile and pass each filename as a separate
    # argument. The ChemblCondenser class makes no filename assumptions.
    parser.add_argument('collection',
            )
    args = parser.parse_args()
    temp_fn='condense.tmp'
    coll=args.collection
    cc = ChemblCondenser(
            dpi_fn='stage_dpi/dpi.dpimerge.chembl_ki_allkeys.tsv',
            create_fn='stage_drugsets/create.chembl.%s.tsv'%coll,
            doc_info=ChemblDocInfo
            )
    cc.condense(
            output_fn=temp_fn,
            shadow_attr_name='shadowed_chembl_id'
            )
    os.rename(
            temp_fn,
            'stage_drugsets/create.chembl.%s_condensed.tsv'%coll,
            )
    cc.filter_m_file(
            input_fn='stage_drugsets/m.chembl.%s.xref.tsv'%coll,
            output_fn='stage_drugsets/m.chembl.%s_condensed.xref.tsv'%coll,
            )

