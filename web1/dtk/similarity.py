from __future__ import print_function
import numpy as np

# XXX To do:
# XXX - share config between review page and flagging script (or maybe
# XXX   get rid of script and flag directly from review page)
# XXX   - factor out shared code into dtk, so we don't need to import the
# XXX     flagger itself in the view
# XXX - histogram of similarity matrix values?
# XXX - add PCA
# XXX - move some of the plotting over here. Some examples are in the DPIComp page

class SimilarityMatrix:
    def __init__(self,row_keys,matrix):
        self.row_keys = row_keys
        self.matrix = np.array(matrix)
        if len(self.row_keys) == 0:
            return
        self._sanity_checks()
    def _sanity_checks(self):
        # - square matrix, size matches row_keys
        assert self.matrix.shape == 2*(len(self.row_keys),), "Shape is %s rather than %dx%d" % (self.matrix.shape, 2*len(self.row_keys), 2*len(self.row_keys))
        # - matrix is symmetrical
        assert np.allclose(self.matrix,self.matrix.T)
        # - matrix diagonal is 1
        assert (self.matrix.diagonal()==1).all()
    def get(self,key1,key2):
        idx1 = self.row_keys.index(key1)
        idx2 = self.row_keys.index(key2)
        return self.matrix[idx1,idx2]
    def clusters(self,repulsion,damping=0.8,max_iter=1000):
        import sklearn.cluster
        model = sklearn.cluster.AffinityPropagation(
                affinity='precomputed',
                # Setting explicit equal preferences (rather than the
                # default calculated from the median) seems to function
                # a little better, and provides a parameter to tweak.
                # A repulsion value of 0 produces a small number of large
                # clusters, 1 puts each item in its own cluster; 0.5 seems
                # like a reasonable default value
                preference=[repulsion]*len(self.row_keys),
                # A damping factor around 0.8 seems to be most likely to
                # converge and yield good results. The default of 0.5
                # seems to fail to converge more often. 0.9 sometimes
                # reports odd results even if it says it converged.
                #
                # Very large datasets (e.g. 1000+ drugs) have trouble
                # converging. It's not clear if increasing max_iter will
                # help, but I've exposed it to allow experimentation.
                verbose=True,
                damping=damping,
                max_iter=max_iter,
                )
        # The affinity propagation algorithm may fail to converge. The only
        # indication of this is, if you set verbose mode, it prints a message
        # to that effect. So, capture stdout so that we can report this
        # programatically.
        from io import StringIO
        capture_out = StringIO()
        from dtk.files import Quiet
        with Quiet(replacement=capture_out) as tmp:
            cluster_ids = model.fit_predict(self.matrix)
        self.output_trace = capture_out.getvalue()
        print(self.output_trace)
        # assemble clusters as sets
        clusters = {}
        for cid,row_key in zip(cluster_ids,self.row_keys):
        ### a TypeError can be raised here if the model above did not
        ### identify any clusters. This is caused b/c cid ends up being
        ### array([nan]).
        ### I've only seen this when very few drugs are used - and that is
        ### not a use case we run into enough to deal with at this time.
            clusters.setdefault(cid,set()).add(row_key)
        # convert to list of sets and sort by size
        clusters = list(clusters.values())
        clusters.sort(key=lambda x:len(x),reverse=True)
        return clusters
    def clusters2(self,thresh=1):
        # a cluster is the transitive closure of everything with a
        # similarity >= thresh. With thresh==1, only identical things
        # cluster together
        clusters = {}
        revmap = {}
        for row_key,row in zip(self.row_keys,self.matrix):
            # make a new cid if this row_key hasn't been seen; else
            # add to the existing cluster this row_key is in
            try:
                cid = revmap[row_key]
            except KeyError:
                cid = 1+len(clusters)
                clusters[cid] = set()
            # find everything sufficiently similar to the row key
            neighbors = set([
                    self.row_keys[i]
                    for i,d in enumerate(row)
                    if d >= thresh
                    ])
            # add all those things to the cluster
            for key in neighbors:
                clusters[cid].add(key)
                revmap[key] = cid
        clusters = list(clusters.values())
        clusters.sort(key=lambda x:len(x),reverse=True)
        return clusters
    def mds(self):
        if self.matrix.shape[0] == 0:
            # running MDS will throw, just set the empty matrix.
            self.mds_matrix = self.matrix
            return

        from sklearn.manifold import MDS
        embedding = MDS(n_components=2, dissimilarity='precomputed')
        self.mds_matrix = list(embedding.fit_transform(1.-self.matrix))

def calc_jaccard(s1, s2):
    num = len(s1 & s2)
    # Early exit if num is 0, output will always be 0.
    if not num:
        return 0.0

    # denom = len(s1 | s2), but the below faster
    denom = len(s1) + len(s2) - num
    # XXX one might argue that all empty sets are identical,
    # XXX and so the following should really return 1 if
    # XXX denom is zero; the special case for num is
    # XXX unneccessary in either case
    if not denom or not num:
        return 0.0
    return float(num)/denom

def diff_clusters(base,new):
    result = []
    excluded = set()
    for k,c1 in base.items():
        scores = [
                (calc_jaccard(c1,c2),(k2,c2))
                for k2,c2 in new.items()
                if frozenset(c2) not in excluded
                ]
        scores.sort(key=lambda x:x[0],reverse=True)
        if scores and scores[0][0]:
            c2 = scores[0][1][1]
            k = scores[0][1][0]
            result.append((c1,c2,k))
            excluded.add(frozenset(c2))
        else:
            result.append((c1,None,None))
    result += [
            (None,c2,k2)
            for k2,c2 in new.items()
            if frozenset(c2) not in excluded
            ]
    return result

def build_asym_mol_prot_sim_matrix(d,row_keys,col_keys):
    import numpy as np
    return np.array([
               [
                 calc_jaccard(d[row_key],d[col_key])
                 for col_key in col_keys
               ]
               for row_key in row_keys
             ])

def build_mol_prot_sim_matrix(d,top=10, verbose=True):
    # ignore any empty clusters (those without any DPI)
    row_keys = [k for k,v in d.items() if v]
    print('found %d clusters with distinct DPI signatures'%len(row_keys))
    from collections import Counter
    ctr = Counter([frozenset(x) for x in d.values()])
    print(len(ctr),'distinct DPI signatures; top %d:'%top)
    if verbose:
        for s,cnt in ctr.most_common(top):
            print('   ',cnt,'cluster for',','.join(s))
    if False:
        target = ctr.most_common(1)[0][0]
        target_wsa_ids = list(set([
                wsa
                for wsa,s in d.items()
                if frozenset(s) == target
                ]))
        print('top signature wsa ids:',target_wsa_ids)
    # in order to take advatange of the symmetric matix, pre-create it
    matrix = [
               [
                 calc_jaccard(d[row_key],d[col_key])
                 for col_key in row_keys
               ]
               for row_key in row_keys
             ]
    return SimilarityMatrix(row_keys,matrix)
