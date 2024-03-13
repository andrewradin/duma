
class MRMR:
    '''Tool for MRMR feature selection.

    Maximum Relevance, Minimum Redundancy is an outline of an iterative
    feature selection methodology. You begin with no features selected,
    and then in each cycle you select the feature that provides the
    most relevance to the task at hand, while being minimally redundant
    with the features already selected. The actual definitions of
    relevance and redundancy can vary with implementation.

    This class defines an API to this class of algorithms, in order to:
    - allow client code to be written without needing to interact with
      the details of how relevance and redundancy are calculated
    - allow easy creation of different MRMR flavors by plugging in
      different relevance and redundancy functions

    A final usable class would look like:
    class UsableClass(MixIn,MixIn,...,MRMR): pass

    The mixins provide the relevance and redundacy functions, and
    also can listen to the add_feature method to gather and cache
    info needed for calculation; in this case they must implement
    super() calls to pass the info along to other mixins.
    '''
    @classmethod
    def from_fm(cls,fm):
        mrmr = cls()
        mrmr.set_targets(
                set([k for k,t in zip(fm.sample_keys,fm.target) if t])
                )
        for i,label in enumerate(fm.feature_names):
            unlabeled_values = fm.data.T[i].toarray()[0].tolist()
            ordering = [
                    (k,v)
                    for k,v in zip(fm.sample_keys,unlabeled_values)
                    if v > 0 # skip nan and zero
                    ]
            ordering.sort(key=lambda x:-x[1])
            mrmr.add_feature(label,ordering)
        return mrmr
    def __init__(self):
        self.unselected = []
        self.selected = []
        self.detail = []
    def get_ordered_labels(self):
        return list(self.selected)
    def set_targets(self,targets):
        # not used by the base class, but useful across many
        # possible relevance and redundancy measures
        self.targets = targets
    def add_feature(self,label,ordering):
        self.unselected.append(label)
    def cycle_to_end(self):
        while(self.unselected):
            self.cycle()
    def cycle(self):
        best = None
        for label in self.unselected:
            rel = self.relevance(label)
            red = self.redundancy(label,self.selected)
            score = rel/red
            if best is None or score > best[0]:
                best = (score,label,rel,red)
        self.detail.append(best)
        self.selected.append(best[1])
        self.unselected.remove(best[1])

class RelSoR1000:
    '''Package SoR1000 as a relevance function.'''
    def __init__(self):
        super().__init__()
        self.rel_cache = {}
        from dtk.enrichment import SigmaOfRank1000
        self.Metric = SigmaOfRank1000
    def add_feature(self,label,ordering):
        super().add_feature(label,ordering)
        em = self.Metric()
        from dtk.enrichment import EMInput
        em.evaluate(EMInput(ordering,self.targets))
        self.rel_cache[label] = em.rating
    def relevance(self,label):
        return self.rel_cache[label]

# XXX Other correlation options to explore:
# XXX - aggregate pairwise correlations using average instead of max
# XXX - use an overall correlation rather than aggregating up from
# XXX   pairwise ones (which might also be faster). For example,
# XXX   accumulate all the hits in the selected set (or create some
# XXX   other exemplar), and then measure the correlation of each
# XXX   unselected feature against that exemplar.

class RedMaxCorr:
    '''Implement a redundancy function based on pairwise correlation.'''
    def redundancy(self,label,selected):
        # The reference example calculated redundancy as the average
        # correlation with selected features, but it seems like max
        # should be better. We want something that isn't totally redundant
        # with any of the selected features.
        bias = 0.02
        if not selected:
            return bias
        # returns a value in the range [bias:1+bias], so the smaller the
        # bias, the larger the effect of redundancy on the total score
        return bias + max([abs(self.corr(label,x)) for x in selected])
    def add_feature(self,label,ordering):
        super().add_feature(label,ordering)

class CorrTopOverlap:
    '''Implement a pairwise correlation function.'''
    def __init__(self):
        super().__init__()
        self.hit_cache = {}
    def add_feature(self,label,ordering):
        super().add_feature(label,ordering)
        self.hit_cache[label] = set([
                k for k,v in ordering[:1000] if k in self.targets
                ])
    def corr(self,label1,label2):
        return len(
                self.hit_cache[label1] & self.hit_cache[label2]
                )/len(self.targets)

class MRMRBasic(RelSoR1000,RedMaxCorr,CorrTopOverlap,MRMR):
    pass
