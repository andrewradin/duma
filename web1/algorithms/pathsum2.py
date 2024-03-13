#!/usr/bin/env python3

# Generic Pathsum
# - limit memory use:
#   - process one layer at a time
#   - compact network representation
# - separate network construction from scoring
#   - networks can be re-scored easily
# - abstract:
#   - each level is a mapping between 2 namespaces
#   - each mapping has evidence and direction values
#     - evidence allows thresholding
#   - each mapping may contain additional values used for scoring
# - representation-independent:
#   - mappings by default come from tsv files
#   - but algorithm will handle any iterable source of tuples

################################################################################
# utilities
################################################################################

from functools import reduce
import gzip
def wrap_tsv(filename,thresh=None,dont_thresh_header=False):
    if filename.endswith('.gz'):
        f = gzip.open(filename,'rt')
    else:
        f = open(filename,"r")
    for line in f:
        fields = line.rstrip('\n').split('\t')
        if thresh:
            if dont_thresh_header:
                dont_thresh_header=False
            elif float(fields[thresh[0]]) < thresh[1]:
                continue
        yield fields

class PathGrouper:
    def __init__(self,ts,formats,group_col):
        from collections import OrderedDict,defaultdict
        self.colmap = OrderedDict()
        for fmt in formats:
            if fmt in ts.columns:
                # first entry in list is the grouping column
                self.colmap[fmt] = [ts.columns[fmt].index(group_col)]
                # remaining entries are score columns
                self.colmap[fmt] += ts.colsets('evidence',[fmt])[fmt]
        # create structure for accumulating results:
        # { wsa_id: { group: total,...}, ...}
        factory=lambda:defaultdict(float)
        self.totals = defaultdict(factory)
    def process_target(self,targ,wsa_list):
        for fmt,cols in self.colmap.items():
            key_col = cols[0]
            ev_cols = cols[1:]
            for row in targ.paths.get(fmt,[]):
                group = row[key_col]
                scores = [float(row[x]) for x in ev_cols]
                # XXX in the original, we take the average, but
                # XXX that seems wrong; most sigprot values will
                # XXX be near 0 or 1, and most DPI values will be
                # XXX 0.5 or 0.9, so each contribution will basically
                # XXX be 0.25, 0.5, 0.75, or 1
                row_score = sum(scores)/len(scores)
                # XXX But even though multiplication seems more
                # XXX reasonable, it pushes the tissue numbers down
                # XXX slightly in the info-gain table, and doesn't
                # XXX seem to make a huge difference in overall score
                #row_score = reduce(operator.mul,scores)
                for wsa_id in wsa_list:
                    self.totals[wsa_id][group] += row_score

################################################################################
# Core pathsum classes
################################################################################
class Target:
    '''Accumulate all paths for a single target (e.g. drug).
    '''
    def __init__(self,name):
        self.name = name
        self.paths = {}
    def add_paths(self,key,paths):
        self.paths[key] = paths
    def add_path(self,key,path):
        l = self.paths.setdefault(key,[])
        l.append(path)
    def path_count(self):
        return sum(map(len,list(self.paths.values())))

class TargetSet:
    '''Support iteration through targets.
    Yields targets individually, without requiring them all to be in
    memory at once.
    '''
    def __init__(self):
        self.net = None
        self.f = None
    def load_from_network(self,net,idxs):
        self.net = net
        self.idxs = list(idxs) # allow repeated use if generator, etc.
        self.columns = net.columns
    def load_from_file(self,filename):
        self.filename = filename
        self.f = wrap_tsv(filename)
        fields = next(self.f)
        assert len(fields) == 1
        key_count = int(fields[0])
        self.columns = {}
        for i in range(0,key_count):
            fields = next(self.f)
            self.columns[int(fields[0])] = fields[1:]
    def colsets(self,name,idxs):
        result = {}
        for idx in idxs:
            l = []
            result[idx] = l
            for i,h in enumerate(self.columns[idx]):
                if h.endswith(':'+name):
                    l.append(i)
        return result
    def __len__(self):
        if self.net:
            return len(self.net.get_target_list(self.idxs))
        else:
            assert(self.f)
            # this requires reading the entire file, which
            # may not be worth it
            raise Exception('not implemented')
    def get_target(self,name):
        if self.net:
            return self.net.get_target(name,self.idxs)
        else:
            assert(self.f)
            target = None
            idx = None
            from dtk.files import get_file_records
            for fields in get_file_records(
                            self.filename,
                            grep=[name],
                            ):
                if not target:
                    if fields[0] != name:
                        continue
                    target = Target(name)
                    # fall through to parse idx
                if fields[0] == '':
                    # next path for this idx
                    target.add_path(idx,fields[1:])
                elif fields[0] == target.name:
                    # next idx for this target
                    idx = int(fields[1])
                else:
                    # next target
                    break
            return target
    def get_next_target(self):
        if self.net:
            for name in self.net.get_target_list(self.idxs):
                yield self.get_target(name)
        else:
            assert(self.f)
            target = None
            key = None
            for fields in self.f:
                if fields[0] == '':
                    # next path for this key
                    target.add_path(key,fields[1:])
                elif target and fields[0] == target.name:
                    # next key for this target
                    key = int(fields[1])
                else:
                    # a new target
                    if target:
                        yield target
                    target = Target(fields[0])
                    key = int(fields[1])
            if target:
                yield target
                target = None
    def write_target(self,target,f):
        for key in target.paths:
            f.write("\t".join((target.name,str(key)))+"\n")
            for row in target.paths[key]:
                f.write("\t"+"\t".join([str(x) for x in row])+"\n")
    def prep_target_file(self,filename):
        assert(self.net)
        if filename.endswith('.gz'):
            f = gzip.open(filename,'wt')
        else:
            f= open(filename,"w")
        f.write(str(len(self.columns))+'\n')
        for k,v in self.columns.items():
            l = [str(x) for x in [k]+v]
            f.write('\t'.join(l)+'\n')
        return f
    def save(self,filename):
        f = self.prep_target_file(filename)
        for t in self.get_next_target():
            self.write_target(t,f)

class Network:
    '''Build a compact network representation.
    The network is built by applying mappings to an existing network
    layer in order to build another layer. A mapping is an iterable
    that yields a series of indexable objects. The first two elements
    in the object are the keys on the output and input layer respectively.
    The output key plus any remaining elements are stored as data, and
    are assembled into a path when the network is traversed.
    '''
    def __init__(self):
        self.stack=[]
        self.columns={}
    def _stack_to(self,idx):
        while len(self.stack) <= idx:
            self.stack.append({})
    def _handle_mapping_row(self
                    ,in_key
                    ,out_key
                    ,data
                    ,index
                    ,outdex
                    ,auto_add=False
                    ):
        if in_key not in index:
            if not auto_add:
                return
            target = [(None,[in_key])]
            index[in_key] = target
        else:
            target = index[in_key]
        outlist = outdex.setdefault(out_key,[])
        outlist.append((target,data))
    def run_mapping(self,
                name,
                mapping,
                src_dest,
                auto_add=False,
                header=None,
                ):
        if not header:
            header = next(mapping)
        col_count=len(header)
        for pair in src_dest:
            self._stack_to(pair[0])
            self._stack_to(pair[1])
            if auto_add:
                incols = [name+':'+header[1]]
                if pair[0] not in self.columns:
                    self.columns[pair[0]] = incols
                else:
                    assert self.columns[pair[0]] == incols
            #print header
            #print self.columns
            #print pair
            mycols=[name+':'+x for x in header[0:1]+header[2:]]
            outcols=self.columns[pair[0]]+mycols
            if pair[1] not in self.columns:
                self.columns[pair[1]] = outcols
            else:
                assert self.columns[pair[1]] == outcols
        for row in mapping:
            assert len(row) == col_count
            in_key = row[1]
            out_key = row[0]
            data = row[0:1]+row[2:]
            for pair in src_dest:
                self._handle_mapping_row(
                            in_key,
                            out_key,
                            data,
                            self.stack[pair[0]],
                            self.stack[pair[1]],
                            auto_add=auto_add,
                            )
    def get_target_list(self,idxs):
        '''Get the union of all keys in a list of indexes.
        '''
        s = set()
        for idx in idxs:
            s = s.union(set(self.stack[idx].keys()))
        return s
    def get_target(self,name,idxs):
        t = Target(name)
        for idx in idxs:
            d = self.stack[idx]
            if name in d:
                t.add_paths(idx,self._assemble(d[name]))
        return t
    def _assemble(self,l):
        # l is a list of tuples, as in the values of the 'stack' dicts.
        # Each tuple consists of a left and right side.  The left side
        # is None, or a list like l, one level down the stack.  The
        # right side is a row of values defining part of a path.
        # This function recursively assembles all the paths defined by
        # the passed-in list by combining the right side of each tuple
        # in the list with all the partial paths to its left.
        result = []
        for t in l:
            if t[0]:
                left_list = self._assemble(t[0])
            else:
                left_list = [[]]
            for left in left_list:
                result.append(left+t[1])
        return result
    def pretty_print(self,level,key):
        print('Network.stack level',level,'key',key)
        l=self.stack[level][key]
        # l is a list of tuples
        # the first element of each tuple is None, or a nested list
        # the second element is a list of values
        def helper(l,prefix):
            for t in l:
                print(prefix,t[1])
                if t[0] is not None:
                    helper(t[0],prefix+'  ')
        helper(l,'  ')

class BaseAccumulator(object):
    '''Base class for score accumulation.

    This class defines the base for a configurable object that calculates a
    score from each target in s target set. To be usable, a derived class
    must define an _score method that takes a vector of floats extracted
    from a path within the target subnet, and returns a float score for
    that path's contribution to the total score. But all aspects of the
    score accumulation are broken out into override-able methods, so they
    may be customized in various ways, while presenting the same interface
    to the ScoreSet object.

    By default, the columns specified in score_map are extracted, all paths
    matching a key in score_map are examined, and all individual path scores
    are totaled to get the final target score.
    '''
    def __init__(self,score_map):
        self.score_map = score_map
    def _reset(self):
        self.total = 0
    def _get_paths(self,t,k):
        for row in t.paths.get(k,[]):
            yield row
    def _accumulate(self,row,path_score):
        self.total += path_score
    def _final(self):
        return self.total
    def score(self,t):
        self._reset()
        for k,v in self.score_map.items():
            for row in self._get_paths(t,k):
                vec = [float(row[col]) for col in v]
                val = self._score(k,vec)
                self._accumulate(row,val)
        return self._final()

class BinningAccumulator(BaseAccumulator):
    '''Total only the best score in each bin.

    This accumulator supports the 'classic' pathsum direct/indirect score
    algorithm, where the total score is the sum of the best path score for
    each tissue (as opposed to the sum of all path scores, which is what
    the base class provides). A bin_col parameter specifies which detail
    column is used to do the binning.
    '''
    def __init__(self,score_map,bin_col):
        super(BinningAccumulator,self).__init__(score_map)
        self.bin_col = bin_col
    def _reset(self):
        self.bins = {}
    def _accumulate(self,row,path_score):
        key = row[self.bin_col]
        if key not in self.bins or self.bins[key] < path_score:
            self.bins[key] = path_score
    def _final(self):
        if self.bins:
            return sum(self.bins.values())
        return 0

class DirectionAccumulator(BaseAccumulator):
    def _score(self,k,vec):
        return -1 * reduce(lambda x,y:x*y, vec)

class EvidenceAccumulator(BinningAccumulator):
    def __init__(self,score_map,bin_col=0,weight_map={}):
        super(EvidenceAccumulator,self).__init__(score_map,bin_col)
        self.weight_map = weight_map
    def _score(self,k,vec):
        if k in self.weight_map:
            w = self.weight_map[k]
            vec = map(lambda x:x[0]*x[1], zip(vec,w))
            denom = sum(w)
        else:
            denom = len(vec)
        return sum(vec)/denom

class Accumulator:
    '''Accumulate score for a target.
    For each path, a vector of values is extracted as determined
    by the score_map.  In product mode, these values are multiplied
    together, and then multiplied by a passed-in factor (typically
    1 or -1).  In (default) weighted sum mode, each value is
    multiplied by a weight, the products are summed, and the total
    is divided by the sum of the weights.  Weights are passed in the
    weight_map; if none are supplied they are all assumed to be 1.
    The final score across all paths is the sum of the path scores
    (if bin_col is None) or the sum of the highest path score for
    each distinct value in the bin_col column of the path record.
    score_map and weight_map are dictionaries where the key is a
    path group number, and the value is a vector of record indexes
    or weights.
    (This implements all currently used variations of scoring, but
    might be replaced by a set of classes that implement the score()
    interface in various ways.)
    '''
    def __init__(self,score_map,bin_col=0,weight_map={},product=0):
        self.bin_col = bin_col
        self.score_map = score_map
        self.weight_map = weight_map
        self.product = product
    def _reset(self):
        if self.bin_col is None:
            self.total = 0
        else:
            self.bins = {}
    def _score(self,k,vec):
        if self.product:
            # XXX multiplicative weights don't make sense here, but
            # XXX we could apply an additive offset to each column
            return self.product * reduce(lambda x,y:x*y, vec)
        else:
            if k in self.weight_map:
                w = self.weight_map[k]
                vec = map(lambda x:x[0]*x[1], zip(vec,w))
                denom = sum(w)
            else:
                denom = len(vec)
            return sum(vec)/denom
    def _accumulate(self,row,path_score):
        if self.bin_col is None:
            self.total += path_score
        else:
            key = row[self.bin_col]
            if key not in self.bins or self.bins[key] < path_score:
                self.bins[key] = path_score
    def _final(self):
        if self.bin_col is None:
            return self.total
        else:
            if self.bins:
                return sum(self.bins.values())
            return 0
    def score(self,t):
        self._reset()
        for k,v in self.score_map.items():
            for row in t.paths.get(k,[]):
                vec = [float(row[col]) for col in v]
                val = self._score(k,vec)
                self._accumulate(row,val)
        return self._final()

class ScoreSet:
    '''Accumulate scores across a set of targets.
    '''
    def __init__(self,ts,metrics,filename=None):
        self.paths = 0
        self.scores = {}
        if filename:
            f = ts.prep_target_file(filename)
        for target in ts.get_next_target():
            self.paths += target.path_count()
            self.scores[target.name] = [m.score(target) for m in metrics]
            if filename:
                ts.write_target(target,f)
        if filename:
            f.close()

################################################################################
# simulation of original algorithm
################################################################################
def legacy(t2p,p2p,p2d,filename):
    '''Simulate the original pathsum implementation.
    The first (t2p) mapping generates index 0 to hold tissues and 1 to hold
    proteins.
    If implementing indirect, the p2p mapping maps from index 1 to index 2.
    The p2d mapping maps from index 1 to 3, and (if available) from 2 to 4.
    Direct score is calculated on paths terminating in index 3, and Indirect
    score on paths terminating in index 4.
    '''
    tissue_idx=0
    t2p_idx=1
    p2p_idx=2
    p2d_dir_idx=3
    p2d_indir_idx=4
    net = Network()
    # XXX to implement per-tissue thresholds, use a replacement for
    # XXX wrap_tsv that implements a different threshold for each tissue
    net.run_mapping('t2p'
                ,wrap_tsv(t2p,thresh=(2,0.95))
                ,[(tissue_idx,t2p_idx)]
                ,auto_add=True
                ,header='protein tissue evidence direction fold'.split()
                )
    print(map(len,net.stack))
    p2d_src_dest = [(t2p_idx,p2d_dir_idx)]
    if p2p:
        net.run_mapping('p2p'
                    ,wrap_tsv(p2p)
                    ,[(t2p_idx,p2p_idx)]
                    ,header='prot2 prot1 evidence direction'.split()
                    )
        print(map(len,net.stack))
        p2d_src_dest.append((p2p_idx,p2d_indir_idx))
    net.run_mapping('p2d'
                ,wrap_tsv(p2d)
                ,p2d_src_dest
                )
    print(map(len,net.stack))
    ts = TargetSet()
    ts.load_from_network(net,(x[1] for x in p2d_src_dest))
    print(ts.columns)
    print(len(ts),'targets')
    if p2p:
        ss = ScoreSet(ts,[
                Accumulator(ts.colsets('direction',[p2d_dir_idx,p2d_indir_idx])
                        ,bin_col=None
                        ,product=-1
                        ), # direction
                Accumulator(ts.colsets('evidence',[p2d_dir_idx])), # direct
                Accumulator(ts.colsets('evidence',[p2d_indir_idx])), # indirect
                ])
    else:
        ss = ScoreSet(ts,[
                Accumulator(ts.colsets('direction',[p2d_dir_idx])
                        ,bin_col=None
                        ,product=-1
                        ), # direction
                Accumulator(ts.colsets('evidence',[p2d_dir_idx])), # direct
                ])
    print(len(ss.scores),'scores')
    l = sorted(
            [[k]+v for k,v in ss.scores.items()],
            key=lambda x: x[2], # direct score
            reverse=True,
            )
    print('top 10 targets')
    for v in l[:10]:
        print(' ',v)
    t = ts.get_target(l[0][0])
    print('%d direct paths for top target %s' % (
                            len(t.paths[p2d_dir_idx]),
                            t.name,
                            ))
    #for row in sorted(t.paths[p2d_dir_idx],key=lambda x:x[0]):
    #    print " ",row,float(row[2])*float(row[6])
    if p2p:
        print('%d indirect paths for top target %s' % (
                            len(t.paths[p2d_indir_idx]),
                            t.name,
                            ))
        #for row in t.paths[p2d_indir_idx]:
        #    print " ",row
    ts.save(filename)
    print('save complete')
    ts2 = TargetSet()
    ts2.load_from_file(filename)
    count = 0
    for target in ts2.get_next_target():
        count += 1
    print('read',count,'records')
    print(ts2.columns)

def demo():
    legacy(
        'sig_prot-0923.tsv',
        None,
        '../../ws/dpi/dpi.drugbank.default.tsv',
        'direct_only_paths.tsv.gz',
        )
    legacy(
        'sig_prot-0923.tsv',
        'ppi.drpias.tsv',
        '../../ws/dpi/dpi.drugbank.default.tsv',
        'both_paths.tsv.gz',
        )

################################################################################
# unit test
################################################################################
def near(a,b,error=1E-6):
    return abs(float(a)-float(b)) < error
