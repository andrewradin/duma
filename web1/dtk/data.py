# Compound data structures used in many places, or common tools for
# manipulating data structures

def modify_dict(target,template):
    '''Variant of dict.update, supporting key deletion.'''
    for k,v in template.items():
        if v is None:
            try:
                del(target[k])
            except KeyError:
                pass
        else:
            target[k] = v

def merge_dicts(*dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result

def dict_subset(d,keys):
    return {k:d[k] for k in keys if k in d}

def dict_subtract(d,keys):
    return {k:d[k] for k in d if k not in keys}

# XXX Maybe add a chain() method where values in self are replaced by the
# XXX union of all values found in a second MultiMap using the original
# XXX values as keys?
class MultiMap:
    '''Handle 1-to-n mappings.

    This is initialized by a bunch of (key,value) pairs, where the keys
    may occur more than once. The fwd_map and rev_map methods return a
    dict from keys to sets of values. The update() method is analogous to
    dict.update -- it takes a MultiMap, and overrides any keys in self
    with the set from that MultiMap. The union() method merges a second
    MultiMap such that a key has all the values from either of the original
    maps. The flatten() method can wrap either fwd_map or rev_map to get
    all the 1-to-1 (key,value) pairs.
    '''
    def __init__(self,kv_pairs):
        self._fwd = {}
        self._rev = None
        for k,v in kv_pairs:
            self._fwd.setdefault(k,set()).add(v)

    @classmethod
    def from_fwd_map(cls, fwd_map):
        """Builds a MultiMap given an existing fwd_map."""
        mm = MultiMap([])
        mm._fwd = fwd_map
        return mm

    def fwd_map(self):
        return self._fwd
    def rev_map(self):
        if self._rev is None:
            self._rev = {}
            for k,s in self._fwd.items():
                for v in s:
                    self._rev.setdefault(v,set()).add(k)
        return self._rev
    def update(self,mm):
        # any mapping in mm replaces the corresponding one in self;
        # _rev will be recalculated if needed
        self._fwd.update(mm._fwd)
        self._rev=None
    def union(self,mm):
        # each key gets the union of values in self and mm
        for k,s in mm._fwd.items():
            try:
                self._fwd[k] |= s
            except KeyError:
                self._fwd[k] = s
        self._rev=None
    def all_pairs(self):
        for k, s in self._fwd.items():
            for v in s:
                yield k, v
    @staticmethod
    def flatten(d,selector=None):
        # use this as a wrapper around fwd_map or rev_map to get all valid
        # 1-to-1 mappings; it can be used to initialize a dict(), or in a
        # loop for a one-time scan
        #
        # If selector is None, ambiguous mappings are dropped; otherwise
        # selector is a function that chooses one of the alternate mappings.
        # If ambiguous mappings are always an error, selector can raise an
        # exception. What's not supported is letting selector decide to
        # bypass particular keys.
        if selector:
            return (
                    (k, next(iter(s)) if len(s) == 1 else selector(s))
                    for k,s in d.items()
                    )
        return (
                (k, next(iter(s)))
                for k,s in d.items()
                if len(s) == 1
                )

class RandomKeyset:
    ''' Implement 'in' operator for a randomly selected fraction of a keyspace.

    usage:
    rks = RandomKeyset(25,100)
    if key in rks:
        pass # this will happen for ~25/100 of the keys
    the seed parameter will alter which keys are selected
    '''
    def __init__(self,num,denom,seed=''):
        self._num = num
        self._denom = denom
        self._seed = seed
    def __contains__(self,key):
        import hashlib
        h = hashlib.new('md5')
        h.update(self._seed+key)
        return (int(h.hexdigest()[-30:],16) % self._denom) < self._num

# append integers to names as needed to make them unique
def uniq_name_list(l):
    # copy input as a starting point
    result = l[:]
    # get the offsets of all distinct values in input
    from dtk.data import MultiMap
    mm = MultiMap(enumerate(l))
    for val,offsets in mm.rev_map().items():
        # if there's more than one offset, value isn't unique
        if len(offsets) > 1:
            # replace value at each offset by adding suffix
            for i,offset in enumerate(sorted(offsets)):
                result[offset] += '_%d'%(i+1)
    return result

# This class can be passed as a key function parameter when comparing things
# of mixed type.
class TypesafeKey:
    def __init__(self,val):
        self._val = val
    def __eq__(self,rhs):
        return self._val == rhs._val
    def __lt__(self,rhs):
        # Note something a bit subtle here: if ordered comparison between
        # the underlying values is possible, we want to do that even if
        # they're not the same type (e.g. comparing floats and ints). That's
        # why we need the class rather than just a function returning a
        # (typename,val) tuple.
        try:
            return self._val < rhs._val
        except TypeError:
            return type(self._val).__name__ < type(rhs._val).__name__
    def __le__(self,rhs):
        if self.__eq__(rhs):
            return True
        return self.__lt__(rhs)

def cond_int_key(s):
    """Return a sort key for a string that may be an int.

    Ints will sort in numeric order following alphabetics.
    """
    try:
        return (1,int(s))
    except ValueError:
        return (0,s)

def assemble_attribute_records(recs, one_to_one=False):
    """Takes records from src and groups them by key.

    Assumes each record is in the format:
        (ID, ATTR_ID, *everything else)

    If not one_to_one, yields a sequence of:
        ID, {ATTR_ID: {RECS}}
    otherwise:
        ID, {ATTR_ID: REC}
    """

    from collections import defaultdict
    rec_pairs = defaultdict(list)
    from dtk.data import MultiMap

    # Note that it would be more efficient to assume all records
    # for the same ID are adjacent, but for now we also handle the
    # out-of-order case.

    for rec in recs:
        rec_id, attr_id, *rest = rec
        if isinstance(rec, list):
            rec = tuple(rec)
        rec_pairs[rec_id].append((attr_id, rec))

    for rec_id, pairs in rec_pairs.items():
        if one_to_one:
            yield rec_id, dict(pairs)
        else:
            yield rec_id, MultiMap(pairs).fwd_map()

def kvpairs_to_dict(kvpairs):
    """Converts key-value pairs into {key:list(values)}."""
    from collections import defaultdict
    out = defaultdict(list)
    for k, v in kvpairs:
        out[k].append(v)
    return dict(out)
