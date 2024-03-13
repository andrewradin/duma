import logging
logger = logging.getLogger(__name__)

class Code:
    _type_by_name = {
            x.__name__:x
            for x in [
                    float,
                    int,
                    bool,
                    str,
                    ]
            }
    def __init__(self,code,**kwargs):
        assert '_' not in code
        self._code = code
        self._args = kwargs
    def label(self): return self._args.get('label',self._code.upper())
    def fv_label(self): return self._args.get('fv_label',self._code.upper())
    def valtype(self): return self._args.get('valtype','float')
    def valtype_as_type(self): return self._type_by_name[self.valtype()]
    def is_stored(self):
        return self.valtype() != 'alias' and 'calc' not in self._args
    # A keyset is used to obtain a subset of wsa or uniprot values
    def is_keyset(self): return self.valtype() == 'bool'
    # A score is something that can reasonably appear on the scoreboard.  All
    # floats are scores unless they explicitly say they aren't.
    def is_score(self):
        return self.valtype() == 'float' and self._args.get('score',True)
    # Setting efficacy to false
    # - excludes scores from DEA
    # - disables scores and features by default in meta methods
    def is_efficacy(self): return self._args.get('efficacy',True)
    # Other score categories for Feature Matrix filtering
    def is_novelty(self): return self._args.get('novelty',False)
    # This is set to indicate meta method outputs
    def is_meta_out(self): return self._args.get('meta_out',False)
    # Setting fv to false excludes things from meta inputs. This is used
    # for redundant codes (like negative versions of a score).
    def is_feature(self): return self._args.get('fv',True)
    def _get_val(self,row):
        try:
            idx = self._args['index']
            return row[idx]
        except KeyError:
            # not stored; must be calculated
            l = self._args['calc']
            return l[0](*[cp._get_val(row) for cp in l[1:]])
    def __repr__(self): return "CP(%s:%s)"%(self._code,repr(self._args),)

# If a CodeGroup fetcher is specified as a string, it implies that
# the code group follows a set of conventions that simplify the
# coding of score data delivery for a plugin:
# - all scores for the code group reside in a single tsv file
# - the string passed as the fetcher is the full path to the file
# - the file is opened and the header retrieved; it's expected that
#   the column headers will match codes or the keyname
# - the file may have more or fewer header columns than defined in
#   the CodeGroup:
#   - columns missing from the CodeGroup will not be extracted
#   - columns missing from the file will be removed from the CodeGroup,
#     along with any calculated Codes depending on those columns
#   These rules mean that the plugin normally won't need any special logic
#   for handling legacy files.
# - the fetcher string will be replaced with a function which returns
#   the selected data from the file
# Note that on unbound instances, the fetcher is never used.  Plugins using
# these new conventions will typically pass None in this case, which allows
# the fully-configured set of code groups to be passed through.
class CodeGroup:
    def __init__(self,keyname,fetcher,*args):
        self._cache = None
        self._fetcher = fetcher
        self._keyname = keyname
        from collections import OrderedDict
        import dtk.readtext as rt
        self._codes = OrderedDict()
        # load codes into hash
        for cp in args:
            assert cp._code not in self._codes
            self._codes[cp._code] = cp
        # check for standard fetcher case
        std_fetch = isinstance(self._fetcher,str)
        if std_fetch:
            # it's a filename; retrieve header from file
            fn = self._fetcher
            try:
                with open(fn) as f:
                    header = next(rt.parse_delim(f))
                assert self._keyname in header,'key missing'
            except (IOError,StopIteration,AssertionError) as ex:
                # data file is missing or unreadable; create a dummy header
                # that causes all data to be stripped
                logger.info("error reading '%s': %s",fn,str(ex))
                header = [self._keyname]
            key_idx = header.index(self._keyname)
            key_mapper = {
                    'wsa':int,
                    }.get(self._keyname)
            data_idxs = []
            data_mappers = []
            # remove any stored codes not present in file
            for cp in list(self._codes.values()):
                if cp.is_stored():
                    try:
                        idx = header.index(cp._code)
                    except ValueError:
                        del self._codes[cp._code]
                        continue
                    data_idxs.append(idx)
                    data_mappers.append(cp.valtype_as_type())
            # replace fetcher
            def fetcher_wrapper(keyset):
                f = open(fn)
                next(f)
                return rt.dc_file_fetcher(keyset,rt.parse_delim(f),
                        key_mapper=key_mapper,
                        key_idx=key_idx,
                        data_idxs=data_idxs,
                        data_mappers=data_mappers,
                        )
            self._fetcher = fetcher_wrapper
        # link children under aliases
        active_list = None
        for cp in self._codes.values():
            if cp.valtype() == 'alias':
                active_list = []
                cp._args['subcodes'] = active_list
            elif active_list is not None:
                # this is a subcode; add to parent's list
                active_list.append(cp)
                # and make hidden by default
                if 'hidden' not in cp._args:
                    cp._args['hidden'] = True
        if std_fetch:
            # remove any calculated codes with missing inputs
            # (this is done repeatedly so that nested references
            # can bubble up from the bottom)
            any_removed=True # force first time through loop
            while any_removed:
                any_removed = False
                for cp,key,l in self._codelist_generator():
                    for name in l[1:]:
                        if name not in self._codes:
                            del self._codes[cp._code]
                            any_removed = True
                            # once a code is removed, don't check any
                            # more names
                            break
        # calculate indexes for stored codes
        next_idx = 0
        for cp in self._codes.values():
            if cp.is_stored():
                cp._args['index'] = next_idx
                next_idx += 1
        self.data_length = next_idx
        # Translate arglists from names to CodeProxy references. This must
        # be done after self._codes is finalized.
        for cp,key,l in self._codelist_generator():
            cp_list = self.flat_proxy_list(*l[1:])
            cp._args[key] = tuple(l[:1])+tuple(cp_list)
    def _codelist_generator(self):
        # Some CodeProxy args consist of a template or function followed
        # by a list of arguments (passed originally as code names, then
        # converted to CodeProxy references).  This function locates and
        # returns all such arguments for processing and validation.
        codelist_args = ('href','calc')
        for cp in list(self._codes.values()):
            for key in codelist_args:
                try:
                    l = cp._args[key]
                except KeyError:
                    continue
                yield cp,key,l
    def dump(self):
        lines = [
                "CodeGroup %s %s" % (self._keyname,self._fetcher)
                ]
        for cp in self._codes.values():
            lines.append("  %s" % repr(cp))
        return lines
    def _codeset(self): return set(self._codes.keys())
    def get_code_proxies(self,typename='',include_hidden=False):
        result = []
        for cp in self._codes.values():
            if cp._args.get('hidden',False) and not include_hidden:
                continue
            if typename and not getattr(cp,'is_'+typename)():
                continue
            result.append(cp)
        return result
    def get_code_proxy(self,code):
        return self._codes[code]
    def flat_proxy_list(self,*args):
        result = []
        for cp in args:
            if isinstance(cp,str):
                cp = self.get_code_proxy(cp)
            if cp.valtype() == 'alias':
                result += self.flat_proxy_list(*cp._args['subcodes'])
            else:
                result.append(cp)
        return result
    def _fetcher_integrity_wrapper(self,fetcher):
        for key,vec in fetcher:
            assert len(vec) == self.data_length, (
                'fetcher returned %d items, expected %d: %s' % (
                        len(vec),
                        self.data_length,
                        repr(self._fetcher),
                        )
                )
            yield key,vec
    def load_cache(self,keyset=None):
        if self._cache is None:
            # if the first call is restricted to a keyset, only load
            # those keys into the cache (the assumption is the client
            # will never ask for anything else)
            try:
                self._cache = dict(
                        self._fetcher_integrity_wrapper(
                                self._fetcher(keyset)
                                )
                        )
            except IOError as ex:
                logger.error('got fetcher exception %s',ex)
                self._cache = {}
    def data(self,keyset,keysort=False):
        if self._cache is None:
            if keysort:
                # if we're sorting by key, it's a feature vector build;
                # avoid caching to save memory if underlying fetcher can
                # handle the sorting
                try:
                    return self._fetcher_integrity_wrapper(
                            self._fetcher(keyset,keysort)
                            )
                except TypeError:
                    pass
            # else, load the cache (possibly only for a limited number of keys)
            self.load_cache(keyset)
        # if we get here, everything (of interest) is in the cache; in the
        # keyset case, if the cache was loaded earlier, it might contain
        # everything, so we still need to filter
        result = self._cache.items()
        if keyset:
            result = [x for x in result if x[0] in keyset]
        if keysort:
            if type(result) != list:
                result = list(result)
            result.sort(key=lambda x:x[0])
        return result
    def _get_val_row(self,key):
        self.load_cache()
        return self._cache[key]

def stdprefix(job_id):
    return "j%d_"%job_id

def parse_stdcode(full_code):
    parts=full_code.split('_')
    if len(parts) != 2 or parts[0][0] != 'j':
        return None
    try:
        job_id = int(parts[0][1:])
        return (job_id,parts[1])
    except ValueError:
        return None

class Catalog:
    def __init__(self):
        self._cgs = []
    def dump(self):
        lines = ['Catalog:']
        for prefix,cg,label in self._cgs:
            glines = cg.dump()
            if prefix:
                glines[0] = prefix+' '+glines[0]
            lines += [
                '  '+line
                for line in glines
                ]
        return lines
    def codes_by_key(self):
        # return {key:([<normal_code>,...],[<special_code>,...]),...}
        result = {}
        for prefix,cg,label in self._cgs:
            for cp in cg._codes.values():
                t=result.setdefault(cg._keyname,([],[]))
                ok = (cp.is_score()
                        and cp.is_efficacy()
                        and not cp._args.get('hidden',False)
                        )
                t[0 if ok else 1].append(cp._code)
        return result
    def add_group(self,prefix,cg,label=None):
        # first, do uniqueness check
        newcodes = cg._codeset()
        for existing in self._cgs:
            if existing[0] == prefix:
                assert existing[1]._codeset().isdisjoint(newcodes)
        # ok, now add group
        self._cgs.append( (prefix,cg,label) )
    def has_codes(self,keys=None,typename=''):
        return any(self.get_codes(keys,typename))
    def get_codes(self,keyname,typename,include_hidden=False):
        # note this returns a generator, not a list!
        for prefix,cg,label in self._cgs:
            if keyname and cg._keyname != keyname:
                continue
            for cp in cg.get_code_proxies(typename,include_hidden):
                yield prefix+cp._code
    def get_uniprot_keyset_codes(self):
        return list(self.get_codes('uniprot','keyset'))
    def _find_code(self,code):
        for prefix,cg,label in self._cgs:
            if code.startswith(prefix):
                try:
                    cp = cg.get_code_proxy(code[len(prefix):])
                except KeyError:
                    continue
                return (prefix,label,cg,cp)
        raise ValueError('%s not a valid code' % code)
    def is_type(self,code,typename):
        prefix,prelabel,cg,cp = self._find_code(code)
        return getattr(cp,'is_'+typename)()
    def get_label(self,code,add_prefix=True):
        prefix,prelabel,cg,cp = self._find_code(code)
        label = cp.label()
        if not add_prefix:
            return label
        if prelabel:
            return prelabel+' '+label
        if prefix:
            return prefix.rstrip('_').upper()+' '+label
        return label
    def get_keyname(self,code):
        prefix,prelabel,cg,cp = self._find_code(code)
        return cg._keyname
    def get_valtype(self,code):
        prefix,prelabel,cg,cp = self._find_code(code)
        return cp.valtype()
    def get_arff_label(self,code):
        # Create a label that's as much like the user labels in the
        # scoreboard as we can get, while not causing problems for
        # weka or the info gain script (which doesn't seem to like
        # spaces in feature names).  At the job level, these use a
        # cleaned-up version of whatever the client supplied.  At the
        # code level, they use the code directly, unless the plugin
        # supplies something more meaningful for this particular
        # application (the fv_label).
        #
        # The feature_set script checks that this results in a unique
        # set of codes across all plugins, and fails if not.  The
        # user can then ask to fall back to the pure codes, which are
        # guaranteed unique, or they can adjust their job labels.
        purify = lambda x:''.join(x.split()) # removes all whitespace
        prefix,prelabel,cg,cp = self._find_code(code)
        label = cp.fv_label()
        if prelabel:
            return purify(prelabel)+'_'+label
        if prefix:
            return prefix.rstrip('_').upper()+'_'+label
        return label
    def get_arff_type(self,code):
        return {
                'float':'REAL',
                'bool':'{True, False}',
                }[self.get_valtype(code)]
    def get_keyset(self,code):
        prefix,prelabel,cg,cp = self._find_code(code)
        result = set()
        for key,buf in cg.data(None):
            if cp._get_val(buf):
                result.add(key)
        return result
    def get_cell(self,code,key):
        prefix,prelabel,cg,cp = self._find_code(code)
        try:
            row = cg._get_val_row(key)
        except KeyError:
            return (None, {})
        val = cp._get_val(row)
        from dtk.data import dict_subset
        attrs = dict_subset(cp._args,['fmt'])
        try:
            href = cp._args['href']
            fill = tuple([cp._get_val(row) for cp in href[1:]])
            attrs['href'] = href[0] % fill
        except KeyError:
            pass
        if hasattr(val, 'calc'):
            val = val.calc()
        return (val,attrs)
    def get_ordering(self,code,desc):
        prefix,prelabel,cg,cp = self._find_code(code)
        result = []
        for key,buf in cg.data(None):
            val = cp._get_val(buf)
            if val is not None:
                result.append( (key,val) )
        # In case we have something convertible to floats instead of floats,
        # explicitly do the conversion here.  In particular this is for
        # liveselectability.
        result = [(key, float(val)) for key, val in result]
        result.sort(key=lambda x: x[1],reverse=desc)
        return result
    def flat_cp_generator(self,*codes):
        for code in codes:
            prefix,prelabel,cg,cp = self._find_code(code)
            vt = cp.valtype()
            if vt == 'alias':
                for x in cg.flat_proxy_list(*cp._args['subcodes']):
                    yield (prefix+x._code,x)
            else:
                yield (code,cp)
    def get_alias_expansion(self,code,key):
        prefix,prelabel,cg,cp = self._find_code(code)
        try:
            cg.load_cache([key])
            row = cg._get_val_row(key)
        except KeyError:
            return []
        expand = cg.flat_proxy_list(code[len(prefix):])
        all_bools = all([x.valtype() == 'bool' for x in expand])
        result = []
        for cp2 in expand:
            val = cp2._get_val(row)
            if all_bools:
                if val:
                    result.append(cp2.fv_label())
            else:
                if val is not None:
                    result.append( (cp2.label(),val) )
        if all_bools:
            return [(cp.label(),', '.join(result))]
        return sorted(result,key=lambda x:x[1],reverse=True)
    def get_feature_vectors(self,*args,**kwargs):
        plug_unknowns=kwargs.pop('plug_unknowns',None)
        key_whitelist=kwargs.pop('key_whitelist',None)
        exclude_pure_unknowns=kwargs.pop('exclude_pure_unknowns',False)
        if kwargs:
            raise TypeError(
                    'unexpected keyword arguments: '+' '.join(list(kwargs.keys()))
                    )
        # get a list of all unique cgs, and a flattened list of (cg,cp) pairs
        cgs = []
        cols = []
        for code in args:
            prefix,prelabel,cg,cp = self._find_code(code)
            if cg not in cgs:
                cgs.append(cg)
            subcode = code[len(prefix):]
            cols += [
                    (prefix,cg,x)
                    for x in cg.flat_proxy_list(subcode)
                    ]
        # verify that all cgs have the same underlying key
        keys = set([x._keyname for x in cgs])
        assert len(keys) == 1
        # build a tracking structure for each cg, with
        # a data buffer holding the next record (None == EOF), and an
        # iterator for accessing all records in key order
        srcs = {}
        for cg in cgs:
            iterator = iter(cg.data(None,keysort=True))
            try:
                buf = next(iterator)
            except StopIteration:
                buf = None
            srcs[cg] = [buf,iterator]
        # define a generator to yield merged results
        def generator():
            while True:
                # find the next key to output; if none left, we're done
                keys = [x[0][0] for x in srcs.values() if x[0] is not None]
                if not keys:
                    break # all done
                key = min(keys)
                # extract all the relevant data values for this key
                result = []
                for prefix,cg,cp in cols:
                    src = srcs[cg]
                    if src[0] is not None and src[0][0] == key:
                        v = cp._get_val(src[0][1])
                    else:
                        v = None
                    result.append(v)
                # now bump the iterator for each cg matching the current key
                for src in srcs.values():
                    if src[0] is not None and src[0][0] == key:
                        try:
                            src[0] = next(src[1])
                        except StopIteration:
                            src[0] = None
                # do special unknown handling
                if exclude_pure_unknowns and set(result) == set([None]):
                    continue
                if plug_unknowns is not None:
                    result = [
                            plug_unknowns if x is None else x
                            for x in result
                            ]
                # apply whitelist
                if key_whitelist and key not in key_whitelist:
                    continue
                # yield results for this key
                yield (key,result)
        return (
                [prefix+cp._code for prefix,cg,cp in cols],
                generator(),
                )

