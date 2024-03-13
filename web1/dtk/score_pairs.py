
from dtk.lazy_loader import LazyLoader

class FeaturePairs(LazyLoader):
    @classmethod
    def from_fm(cls,fm,mode='sc'):
        fp = cls()
        if mode == 'mm':
            # an in-place hack based on FMCalibrator; all the caveats
            # in the comments there apply here as well
            import numpy as np
            nda = fm.data_as_array()
            for col in range(nda.shape[1]):
                l = list(np.nan_to_num(nda[:,col]))
                lo = min(l)
                hi = max(l)
                delta = hi - lo
                if delta:
                    nda[:,col] = [(x-lo)/delta for x in l]
                else:
                    nda[:,col] = len(l) * [0]
            from scipy import sparse
            fm.data = sparse.csr_matrix(nda)
        else:
            assert mode in ('sc','scl')
            from dtk.score_calibration import FMCalibrator
            fmc = FMCalibrator(fm=fm)
            fmc.calibrate(logscale = (mode == 'scl'))
        fp.sources = fm.spec.get_codes()
        fp.feature_names = fm.feature_names
        fp.score_keys = fm.sample_keys
        fp.target_keys = set(k for k,t in zip(fp.score_keys,fm.target) if t)
        # not needed yet
        #fp.target_idxs = [i for i,t in enumerate(fm.target) if t]
        fp.data = fm.data_as_array().T
        return fp
    def _pairs_loader(self):
        return [(i,j)
                for i in range(len(self.feature_names))
                for j in range(i)
                ]
    def _Metric_loader(self):
        '''Return the instance-wide default metric class.'''
        from dtk.enrichment import SigmaOfRank1000
        return SigmaOfRank1000
    def vec2ord(self,vector):
        '''Return a data_catalog-style ordering from a FM column.'''
        return sorted(zip(self.score_keys,vector),key=lambda x:-x[1])
    def enrichment(self,vector):
        '''Return the enrichment for an FM column or similar vector.'''
        from dtk.enrichment import EMInput
        em = self.Metric()
        em.evaluate(EMInput(self.vec2ord(vector),self.target_keys))
        return em.rating
    ###
    # per-feature lists
    ###
    def _feat_base_loader(self):
        '''Return the base metric for each feature.'''
        return [
                self.enrichment(self.data[i])
                for i in range(len(self.feature_names))
                ]
    hit_thresh = 0.7
    def _feat_hits_loader(self):
        '''Return the set of over-threshold hits for each feature.

        Since we're working with calibrated scores, the threshold is
        a probability rather than a rank.
        '''
        return [
                set(k
                        for k,p in self.vec2ord(self.data[i])
                        if k in self.target_keys and p > self.hit_thresh
                        )
                for i in range(len(self.feature_names))
                ]
    ###
    # per-pair lists
    ###
    def _pair_labels_loader(self):
        return [
                ' '.join(sorted(self.feature_names[x] for x in pair))
                for pair in self.pairs
                ]
    def _pair_hit_overlap_loader(self):
        return [
                len(self.feat_hits[i1] & self.feat_hits[i2]) \
                        / len(self.target_keys)
                for i1,i2 in self.pairs
                ]
    def _pair_hit_corr_loader(self):
        mask=[x in self.target_keys for x in self.score_keys]
        subset = self.data[:,mask]
        pc=PairwiseCorrelation()
        corr = pc.cross_correlate(subset)
        return [
                corr[i1,i2]
                for i1,i2 in self.pairs
                ]
    def _pair_corr_loader(self):
        pc=PairwiseCorrelation()
        corr = pc.cross_correlate(self.data)
        return [
                corr[i1,i2]
                for i1,i2 in self.pairs
                ]
    def _pair_ex_corr_loader(self):
        return [h-b for h,b in zip(self.pair_hit_corr,self.pair_corr)]
    def _pair_base_loader(self):
        return [
                max(self.feat_base[i1],self.feat_base[i2])
                for i1,i2 in self.pairs
                ]
    def _pair_f1_metric_loader(self):
        return [self.feat_base[i1] for i1,i2 in self.pairs]
    def _pair_f2_metric_loader(self):
        return [self.feat_base[i2] for i1,i2 in self.pairs]
    def _add_boosts_loader(self):
        result = []
        for i,(i1,i2) in enumerate(self.pairs):
            combined = (self.data[i1] + self.data[i2])/2
            result.append(self.enrichment(combined)-self.pair_base[i])
        return result
    def _mult_boosts_loader(self):
        result = []
        for i,(i1,i2) in enumerate(self.pairs):
            # Compute the mult combination as 1 - (1-a)(1-b).
            # This should theoretically allow a score without support in
            # other CMs to remain near the top, and (unlike the first cut)
            # isn't strongly correlated with add_boosts. The below
            # calculation of a+b-ab is algebraically equivalent to
            # 1 - (1-a)(1-b).
            combined = (self.data[i1] + self.data[i2]) \
                    - (self.data[i1] * self.data[i2])
            result.append(self.enrichment(combined)-self.pair_base[i])
        return result

class PairwiseCorrelation:
    nonoverlap = 'omit'
    nonoverlap_choices = [(x,x) for x in [
            'zeros',
            'omit'
            ]]
    corr_type = 'spearman'
    corr_type_choices = [(x,x) for x in [
            'spearman',
            'pearson',
            'jaccard',
            ]]
    def cross_correlate(self, scoremat):
        '''Return a correlation matrix given a score matrix.

        The input scoremat is a matrix of score values with
        a row for each score and a column for each key. The
        matrix contains nans for unspecified scores, and options
        for nan handling are implemented below.

        The output is an NxN matrix of correlations, where N
        is the number of rows in the input scoremat. The ordering
        of rows/columns in the output correlation matrix is
        the same as the rows in the input scoremat.
        '''
        import numpy as np
        from scipy.stats import spearmanr, pearsonr

        N = len(scoremat)
        masks = []
        for row in scoremat:
            masks.append(~np.isnan(row))

        zdata = np.diagflat([1.0] * N)

        for i in range(N):
            for j in range(0, i):
                a = scoremat[i]
                b = scoremat[j]
                ma = masks[i]
                mb = masks[j]

                if self.nonoverlap == 'omit':
                    mask = ma & mb
                    a = a[mask]
                    b = b[mask]
                elif self.nonoverlap == 'zeros':
                    mask = ma | mb
                    a = np.nan_to_num(a[mask])
                    b = np.nan_to_num(b[mask])
                else:
                    raise RuntimeError(f"invalid nonovermap {self.nonoverlap}")

                if not mask.any():
                    # If there are no elements in common, all scores return 0
                    # (though some will complain about it, hence the special
                    # case)
                    corr = 0
                elif self.corr_type == 'spearman':
                    corr, p = spearmanr(a, b)
                elif self.corr_type == 'pearson':
                    corr, p = pearsonr(a, b)
                elif self.corr_type == 'jaccard':
                    corr = (ma & mb).sum() / (ma | mb).sum()
                else:
                    raise RuntimeError(f"invalid corr_type {self.corr_type}")

                zdata[i,j] = corr
                zdata[j,i] = corr

        zdata = np.nan_to_num(zdata)
        return zdata
    @classmethod
    def scoremat_from_orderings(cls,orderings):
        # Build out the full score matrix.
        # Scores are rows, wsas are columns, nan's where missing.
        all_keys = set.union(*[
                set(x[0] for x in ordering)
                for ordering in orderings
                ])
        key2idx = {key:i for i,key in enumerate(all_keys)}
        import numpy as np
        scoremat = np.full(
                (len(orderings), len(key2idx)),
                np.nan,
                dtype=np.float,
                )
        for row, ordering in enumerate(orderings):
            for key, value in ordering:
                scoremat[row, key2idx[key]] = value
        return scoremat

class RankTracer(LazyLoader):
    # For the weighted average of 2 scores, figure out the score
    # corresponding to a particular rank at each weight from 0 to 1.
    #
    # Every key in x_ord and y_ord defines a straight line between
    # (0,x_ord_score) and (1,y_ord_score). Start on the nth line
    # down at w==0, and each time another line crosses your path,
    # switch to that line.
    _kwargs = ['x_ord','y_ord']
    rank = 1000
    def _x_lookup_loader(self):
        return dict(self.x_ord)
    def _y_lookup_loader(self):
        return dict(self.y_ord)
    class State:
        def __init__(self,w,path,above=0,below=0):
            self.w = w # how far we've moved to the right
            self.path = path # the (intercept,slope) of the line we're on
            # If multiple score keys have identical scores in both x_ord
            # and y_ord, the line they define actually represents a group
            # of paths, so we need to track what our ranking is in the
            # group. For symmetry of coding, we do this by counting our
            # offset from both the upper and lower edge of the group.
            # (These are both zero if there's only one key in the group.)
            self.above = above
            self.below = below
        def point(self):
            return (self.w,self.path[0]+self.w*self.path[1])
        def __repr__(self):
            return f'w:{self.w} path:{self.path} ({self.above}/{self.below})'
    def _path_array_loader(self):
        import numpy as np
        return np.array(self.paths)
    def filter_paths(self,state):
        #return self.paths
        # numpy speedup; uncomment above line to disable
        # This gives about a 25% speedup loading the trace.
        delta=self.path_array-(state.path) # each row is (s1-s0,m1-m0)
        # w_vec is cross_w for each path (nan if same slope)
        w_vec=-delta[:,0]/delta[:,1]
        valid=(w_vec>state.w) & (w_vec<=1)
        if not valid.any():
            return []
        # find the smallest valid w
        first_crossing=min(w_vec[valid])
        # since we don't do rounding here, and we count on it above,
        # return somewhat more than we need to
        margin=0.1
        near=(w_vec>(first_crossing-margin))&(w_vec<(first_crossing+margin))
        return self.path_array[near]
    def next_state(self,state):
        # The basic idea here is:
        # - state.path defines the line we're on, and state.w is our
        #   position on that line
        # - we check each other line in the set to find the one with the
        #   closest intersection to the right of our current position
        # - we then plot that intersection point and continue on the
        #   new intersecting line (whether it crosses us going up or
        #   down, it's now the k'th line in the set, replacing our
        #   original one)
        # All the real complication is for managing ties.
        if state.w >= 1:
            return None # we've reached the far end
        # cross_w is the weight for which our current path intersects
        # the candidate path. We're looking for the closest path, so
        # we're only interested if it crosses after state.w, and
        # before the best candidate so far (kept in result_w).
        # Initialize result_w to the maximum weight, because we're
        # initially interested in anything to our right.
        result_w = 1
        result_paths = []
        for c_a,c_delta in self.filter_paths(state):
            # c_a and c_delta are the y-intercept and slope for the
            # current candidate; get the relative slope for this candidate
            rel_delta = state.path[1] - c_delta
            # skip parallel paths, which should never meet;
            # this also prevents us from comparing against ourselves
            if rel_delta == 0:
                continue
            # we use rounding here so that crossing points are
            # calculated consistently between iterations; this
            # keeps paths from being skipped because they round
            # down on one iteration and up on the next
            #
            # but rounding is very expensive, so first do a rough
            # comparison to eliminate most paths, then round if
            # we need an exact comparison (saves ~50% page load time).
            cross_w = (c_a - state.path[0]) / rel_delta
            if cross_w <= 0.99*state.w or cross_w > 1.01*result_w:
                continue
            cross_w = round(cross_w,6)
            if cross_w <= state.w or cross_w > result_w:
                continue
            # we now know cross_w is better or equal to the best so far;
            # accumulate an additional crossing if it's equal, or start
            # over if it's better
            if cross_w == result_w:
                result_paths.append((c_a,c_delta))
            else:
                result_w = cross_w
                result_paths = [(c_a,c_delta)]
        # result_paths is now the list of all paths that cross us at the
        # nearest crossing point; put them in order by slope, and then
        # remove the outermost pairs where one crosses from above and
        # the other crosses from below (these have no net effect on rank)
        if result_paths:
            result_paths.sort(key=lambda x:-x[1])
            # paths crossing in opposite directions cancel out
            while result_paths[0][1] > state.path[1] \
                    and result_paths[0][-1] < state.path[1]:
                result_paths = result_paths[1:-1]
        # if everything cancels out, keep going on the same path,
        # but bump w so we only consider crossings to the right
        # on the next pass
        if not result_paths:
            return self.State(result_w,state.path,
                    above=state.above,
                    below=state.below,
                    )
        # All remaining paths are now crossing from the same direction;
        # if we get here, we need to switch paths. But if we're on one
        # of a group of identical paths, we might just switch to another
        # one of those. The logic below implements the following for
        # both directions:
        # - if there are fewer crossings than paths remaining between
        #   our current path and the edge of the group, just adjust
        #   the remaining counts
        # - otherwise, select the appropriate crossing, skipping one
        #   for each path remaining to reach the edge
        if result_paths[0][1] > state.path[1]:
            # crossing from below; choose the one with the smallest slope
            if state.above >= len(result_paths):
                # remain in same path (group) but adjust counts
                return self.State(result_w,state.path,
                        above=state.above - len(result_paths),
                        below=state.below + len(result_paths),
                        )
            idx = -1 - state.above
        else:
            # crossing from above; choose the one with the largest slope
            if state.below >= len(result_paths):
                # remain in same path (group) but adjust counts
                return self.State(result_w,state.path,
                        above=state.above + len(result_paths),
                        below=state.below - len(result_paths),
                        )
            idx = 0 + state.below
        # idx now indicates which path in result_paths we should switch to;
        # search for adjacent matches to set above and below
        matches = [x == result_paths[idx] for x in result_paths]
        below = sum(matches[:idx])
        # the idx+1 trick below works to select everything in the
        # list after idx, except if idx == -1
        above = 0 if idx == -1 else sum(matches[idx+1:])
        # return the selected new path
        return self.State(result_w,result_paths[idx],above=above,below=below)
    def _paths_loader(self):
        '''Return [(a_score,b_score-a_score),...] for each molecule.

        List is sorted highest to lowest, which really only matters in
        _start_state_loader. For a given path, its score with weight w
        is path[0]+w*path[1].
        '''
        d = {}
        for key,a_val in self.x_ord:
            b_val = self.y_lookup.get(key,0)
            d[key] = (a_val,b_val)
        for key,b_val in self.y_ord:
            if key not in d:
                d[key] = (0,b_val)
        return sorted(
                ((a_val,b_val-a_val) for key,(a_val,b_val) in d.items()),
                reverse=True,
                )
    def _start_state_loader(self):
        path = self.paths[self.rank]
        above = 0
        while above < self.rank and path == self.paths[self.rank-above-1]:
            above += 1
        below = 0
        max_below = len(self.paths) - self.rank - 1
        while below < max_below and path == self.paths[self.rank+below+1]:
            below += 1
        return self.State(0,self.paths[self.rank],above=above,below=below)
    def _trace_loader(self):
        state = self.start_state
        result = [state.point()]
        import numpy as np
        old_settings = np.geterr()
        np.seterr(all="ignore") # don't complain about nans
        while True:
            state = self.next_state(state)
            if state is None:
                np.seterr(**old_settings)
                return result
            result.append(state.point())

