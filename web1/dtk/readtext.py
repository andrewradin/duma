from builtins import range
def parse_delim(stream,delim='\t'):
    for line in stream:
        yield line.strip('\r\n').split(delim)

def comment_stripper(iterable):
    for line in iterable:
        if not line.strip() or line.startswith('#'):
            continue
        yield line

# Code that processes files using get_file_records can access data
# by using numeric indexes and ad-hoc conversions. This is very
# fast and easy to code, but less maintainable because the meanings
# of each field are not clear ('rec[3]' is not self-documenting).
#
# There are various ways to address this:
# - do it anyway (ok for simple loops, esp. w/ a comment documenting
#   what's expected in each column)
# - store indexes in named variables (rec[score_idx])
# - convert the list of fields to an object that can be accessed via
#   attributes
#
# The RecordMapper class provides one implementation of this third option.
# This class performs near-optimal conversion of a list of strings
# into an object supporting named access to each field, with optional
# per-field type conversion. The 'make_from' classmethods are convenience
# constructor alternatives that accept various descriptions of what
# mapping you want to perform. Once a RecordMapper object exists,
# the wrap() method takes a field list and returns an object that
# can be accessed by attribute.
#
# Usage:
#   rm = RecordMapper.make_from...
#   for rec in get_file_records(...):
#      m = rm.wrap(rec)
#      if m.field > 1 ...
#
# Optimizations and alternatives:
# - for situations where performance is absolutely critical, you can get
#   about a 10% improvement by inlining the wrap function:
#   rm = RecordMapper.make_from...
#   plan=rm._plan
#   nt=rm._nt
#   for rec in get_file_records(...):
#      m = nt(*[cast(rec[idx]) for idx,cast in plan])
#      if m.field > 1 ...
# - for situations where performance is absolutely critical, and you're only
#   accessing a fraction of the data, you can do lazy in-place conversion
#   (note that this will not work with a field named 'wrap', or any other
#   name in use by the RecordMapper class):
#   rm = RecordMapper.make_from...
#   m = rm
#   for rec in get_file_records(...):
#      m._rec = rec
#      if m.field > 1 ...
#
class RecordMapper:
    def _convert_and_wrap(self,rec):
        return self._nt(*[cast(rec[idx]) for idx,cast in self._plan])
    def _wrap_only(self,rec):
        # this assumes 'names' map 1-to-1 to record columns
        return self._nt(*rec)
    def __init__(self,names,plan):
        from collections import namedtuple
        self._nt=namedtuple('Rec',names)
        self._plan=plan
        if plan:
            self._attr_map=dict(zip(names,plan))
            self.wrap = self._convert_and_wrap
        else:
            self._attr_map={
                    name:(i,lambda x:x)
                    for i,name in enumerate(names)
                    }
            self.wrap = self._wrap_only
    @classmethod
    def make_from_idxmap(cls,idx_map):
        names = [x[0] for x in idx_map]
        plan = [(x[1],x[2] or (lambda x:x)) for x in idx_map]
        return cls(names,plan)
    @classmethod
    def make_from_colmap_and_header(cls,col_map,header):
        names = [x[0] for x in col_map]
        plan = [(header.index(x[1]),x[2] or (lambda x:x)) for x in col_map]
        return cls(names,plan)
    @classmethod
    def make_from_header(cls,header):
        return cls(header,None)
    def __getattr__(self,attr):
        try:
            idx,cast = self._attr_map[attr]
        except KeyError:
            raise AttributeError(
                    "unknown attribute '%s'; options are '%s'"
                    % (attr, "','".join(sorted(self._attr_map.keys())))
                    )
        return cast(self._rec[idx])

# stackable generator for RecordMapper, using header only
def convert_records_using_header(src,header=None):
    if header is None:
        header = next(src)
    rm = RecordMapper.make_from_header(header)
    for rec in src:
        yield rm.wrap(rec)

# stackable generator for RecordMapper, using colmap and header
# - colmap is (name,h_name,cast) triples
# - h_name looked up in header (first record in src if not provided)
def convert_records_using_colmap(src,colmap,header=None):
    if header is None:
        header = next(src)
    rm = RecordMapper.make_from_colmap_and_header(colmap,header)
    plan=rm._plan
    nt=rm._nt
    for i,rec in enumerate(src):
        try:
            yield nt(*[cast(rec[idx]) for idx,cast in plan])
        except IndexError:
            raise ValueError(f'syntax error in record {i+1}: {rec}')

def dc_file_fetcher(keyset,rec_src,
            key_idx=0,
            key_mapper=None,
            data_idxs=None,
            data_mappers=None,
            data_mapper=float,
            ):
    '''Reformat a stream of tuples as required by Data Catalog.'''
    data_cols = None
    for rec in rec_src:
        # supply any defaults the first time through the loop
        if data_cols is None:
            # either a list of column indicies is passed in, or we return
            # all columns except the key column, in order
            if data_idxs is None:
                data_idxs = [x for x in range(0,len(rec)) if x != key_idx]
            # translate data from string to expected type; the caller can
            # pass a list of mappers for each column, or a single mapper
            # to be used for all columns (which defaults to float())
            if data_mappers is None:
                data_mappers = [data_mapper] * len(data_idxs)
            # restructure per-column stuff for convenient access
            data_cols = list(zip(data_idxs,data_mappers))
        # get key, map it, and filter if we're doing key filtering
        key = rec[key_idx]
        if key_mapper:
            key = key_mapper(key)
        if keyset and key not in keyset:
            continue
        # extract and map data, and return (key,data) tuple
        data = [mapper(rec[idx]) for idx,mapper in data_cols]
        yield (key, tuple(data))

