import numpy as np
import math
import scipy

import logging
logger = logging.getLogger(__name__)

################################################################################
# This file contains code for managing feature matrices, with the goals of:
# - uniform access, so tools can work across many use cases
# - efficient storage, so algorithms are fast, and large matrices can be
#   handled
#
# This is achieved by using mix-ins to control FM behavior. So, there isn't
# a single FM class. Instead, an FMBase base class is combined with a
# 'Repr' mixin to determine data representation (numpy array vs. sparse array),
# and an optional 'Recipe' mixin to build the matrix from an abstract
# description (e.g. a data catalog and list of codes) rather than needing
# to supply all the individual data values. Any such class is considered an
# 'FM' class.
#
# An FM class instance has a save() method that writes the matrix in a special
# archive format based on numpy's .npz file. When an FM with a Recipe mixin is
# saved, only the recipe is written, so it's very space-efficient.
# Alternatively, a save_flat() method will create a totally self-contained
# matrix archive (which could be re-instantiated on another machine where the
# data catalog wasn't available). The classmethod FMBase.load_from_file
# creates an FM instance of the correct type from an archive.
#
# There are 3 use cases for creating FM objects:
# - they can be created directly by algorithms where data is already available.
#   For example, code that runs a PCA on an existing FM could report results
#   in a new FM. The class could FMBase by itself, or with a Repr mixin. Data
#   would be passed in to the ctor.
# - they can be recovered from a file. This could be an archive file, as
#   described above, with possibly other cases added later
#   - as a special case, they can be recovered from an old ML-style ARFF file
#     and corresponding druglist.csv file
# - they can be created from a recipe. The initial case is creation from
#   the data catalog. There may be a use case for recording a sequence of
#   simple, fast transforms like subsetting, hstacking, etc, but we can wait
#   for a specific requirement before implementing
#
# XXX Remaining steps:
# XXX - handle and document all standard attributes, especially:
# XXX   - 'feature_key' and 'feature_keys' attributes
# XXX - HStack and heterogeneous recipes; load_from_parts
# XXX   - note that the experiments/struct implementation of HStack as
# XXX     a recipe is probably wrong. Instead, there should be some
# XXX     sort of transform recipe layer that combines other FMs, and
# XXX     an HStackRepr that provides storage for them
# XXX   - we defer the need for this by always flattening DCSpec
# XXX     FMs into a single sparse array of floats (throwing an
# XXX     exception if we encounter something incompatible)
################################################################################
# FMBase
# - an FM class is constructed by combining this class with two mix-ins:
#   - a 'Repr' mixin defines the actual in-memory format used (e.g. a
#     numpy array vs. a sparse array)
#   - an optional 'Recipe' mixin supports defining the FM in terms of
#     instructions for assembling the data, rather than the data itself;
#     the obvious use case is building the FM from the data catalog
# - a save() method allows saving an FM to disk (in a special archive based
#   on numpy .npz format)
################################################################################
from sklearn.utils import Bunch
import six
# From Bunch, this class inherits the ability to use property or dict syntax
# to access properties, and to initialize properties through ctor keyword args.
class FMBase(Bunch):
    '''Encapsulate a feature matrix.

    Access API follows scikit_learn:
    - can be accessed as dictionary or as object
    - 'data' member/key returns n_samples x n_attributes array
    - 'target' member/key returns n_samples array of integer class labels
    - 'target_names' member/key returns list of class names (indexed by the
      integers in 'target')
    - 'feature_names' member/key returns list of attribute labels

    In addition, an optional 'sample_keys' member can hold a list of ids
    tying each row to some external object, and the 'sample_key' member
    can hold a string identifying the type of object (for example,
    a sample key of 'wsa' means the rows are drugs, identified by a
    wsa_id).
    '''
    # Extend the Bunch functionality to support dynamic loading of missing
    # attributes. _attr_loader should set the requested attribute as a
    # property of 'self'. It can be overridden so that mixins can specify
    # their own defaults or dynamic loading.
    def __getitem__(self,key):
        # Need to special case for archive to avoid infinite recursion.
        # _attr_loader calls hasattr which, in sklearn.Bunch calls through to
        # __getitem__.
        # This works in py2 because you'd hit a recursion limit exception
        # and hasattr returned false on any exceptions.
        # py3 changed hasattr to propagate unexpected exceptions.
        if key != 'archive' and key not in self:
            self._attr_loader(key)
        return super(FMBase,self).__getitem__(key)
    ######
    # _attr_loader provides backstop defaults for missing attributes,
    # possibly deriving them from other attributes, including loading
    # them from an archive. Note that even if this function did nothing,
    # it must be present to prevent __getitem__ loops from the code above.
    ######
    _casts = {
            'feature_names':list,
            'sample_keys':list,
            'target':list,
            'target_names':list,
            }
    def _attr_loader(self,attr):
        if hasattr(self,'archive'):
            # If an FM is backed by an archive, and the archive has an
            # element with the name of the attribute we're looking for,
            # recover it. If that attribute has a special Repr or Recipe
            # specific format, that should be intercepted in a mixin
            # _attr_loader before we get here.
            if attr in self.archive.content():
                data = self.archive.get(attr)
                # npz format converts anything it can into an ndarray;
                # if we expect a different list-like format, cast it
                # back here.
                if attr in self._casts:
                    data = self._casts[attr](data)
                setattr(self,attr,data)
                return
        if attr == 'python_type':
            self.python_type = float
    ######
    # access data in specific format
    ######
    # XXX this could be delegated to the repr layer, but that doesn't
    # XXX handle cases where FMBase is used alone, or 'data' is assigned
    # XXX manually to a different representation
    def data_as_array(self):
        import scipy.sparse
        if scipy.sparse.isspmatrix_csr(self.data):
            return self.data.toarray()
        if isinstance(self.data,np.ndarray):
            return self.data
        return np.ndarray(self.data)
    def extract_features(self,col_list):
        return self.data[:,col_list]
    def replace_features(self,col_list,src):
        self.data[:,col_list] = src
    ######
    # count missing values
    ######
    def feature_missing_values(self):
        return [
                sum([np.isnan(v) for v in row])
                for row in self.data.T
                ]
    def sample_missing_values(self):
        from collections import Counter
        return Counter([
                sum([np.isnan(v) for v in row])
                for row in self.data
                ])
    ######
    # Save matrix to archive
    # - there are two flavors: save and save_flat; save_flat forces a store
    #   of the underlying data, rather than saving a recipe
    ######
    def save(self,stem):
        store = FMArchive()
        store.put('typestack',[x.__name__ for x in type(self).__bases__])
        self._store_to_archive(store)
        store.write(stem)
    def save_flat(self,stem):
        from dtk.features import FMArchive
        store = FMArchive()
        basenames = [x.__name__ for x in type(self).__bases__]
        if len(basenames) == 3:
            # strip out recipe class
            # XXX there's probably a more robust way to do this
            basenames = basenames[1:]
        store.put('typestack',basenames)
        self._store_content_to_archive(store)
        store.write(stem)
    # recipe classes should override this to save recipes rather than
    # content; they can call _store_content_to_archive() with an
    # exclude list to store data they don't handle specially; classes
    # overriding this method will typically need _attr_loader code
    # to restore the data they're saving specially
    def _store_to_archive(self,store):
        self._store_content_to_archive(store)
    # Repr classes may override this if they use attributes that don't
    # save and restore properly via native FMArchive methods; they can
    # call the base class _store_content_to_archive() with an exclude list
    # to store data they don't handle specially; classes overriding this
    # method will typically need _attr_loader code to restore the data
    # they're saving specially
    def _store_content_to_archive(self,store,exclude=None):
        if exclude is None:
            exclude = set()
        store_list = [
                'data',
                'sample_keys',
                'sample_key',
                #'feature_keys',
                #'feature_key',
                'feature_names',
                'typenames',
                'target',
                'target_names',
                ]
        for attr in store_list:
            if hasattr(self,attr) and attr not in exclude:
                store.put(attr,getattr(self,attr))
    ######
    # Save matrix to ARFF file
    ######
    def save_to_arff(self,fh_or_path,desc='',sparse=True):
        dest = ArffDestination(
                fh_or_path,
                desc,
                zip(self.feature_names,self.typenames),
                sparse=sparse,
                )
        for row in self.data:
            dest.append(FeatureVector(row))
    ######
    # alternative constructors / recovery from file
    ######
    @classmethod
    def load_from_file(cls,stem):
        store = FMArchive(path=stem)
        def load(x):
            # In PY3 if we load FMArchive's generated by PY2, any strings
            # are interpreted as byte strings.  These don't look up properly
            # from globals(), so decode them.
            # (PY2 will allow either globals()['blah'] or globals()[u'blah'])
            if hasattr(x, 'decode'):
                x = x.decode('utf8')
            return globals()[x]
        ancestors=tuple(load(x) for x in store.get('typestack'))
        FM = type('FM',ancestors,{})
        return FM(archive=store)
    @classmethod
    def load_from_recipe(cls,spec):
        if spec.subspecs:
            raise NotImplementedError() # XXX need HStackRepr
            # the spec needs multiple types of storage; build a 'pure' FM
            # for each type, and hstack them together
            parts = [cls.load_from_recipe(x) for x in spec.subspecs]
            # XXX How should we decide whether the consolidated data should
            # XXX be in a sparse or dense array? Maybe use dense if over
            # XXX 1/3 of the attributes are dense?
            #class FM(HstackRecipe,NDArrayRepr,FMBase): pass
            #class FM(HstackRecipe,SKSparseRepr,FMBase): pass
            return FM(parts=parts)
        # it's a pure type; instantiate a single FM
        if spec.type_info.is_sparse():
            class FM(DCRecipe,SKSparseRepr,FMBase): pass
        else:
            class FM(DCRecipe,NDArrayRepr,FMBase): pass # pylint: disable=function-redefined
        return FM(spec=spec,python_type=spec.type_info.python_type())
    @classmethod
    def load_from_arff(cls,path,druglist_path=None,default=float('nan')):
        import arff
        from dtk.files import get_file_lines
        parsed = arff.load(get_file_lines(path))
        # this loads a feature matrix from an ARFF file, making some
        # assumptions based on the format output by run_ml.py:
        kwargs = dict(
                # all but the last column is data
                data = np.array([
                        [default if v is None else v for v in x[:-1]]
                        for x in parsed['data']
                        ]),
                # the last column is target info, stored as either
                # 'True' or 'False'
                target = np.array([
                        {'True':1,'False':0}[x[-1]]
                        for x in parsed['data']
                        ]),
                target_names = ['False','True'],
                feature_names = np.array([
                        x[0]
                        for x in parsed['attributes'][:-1]
                        ]),
                )
        if druglist_path:
            # if a druglist_path is specified, it expects the format
            # of druglist.csv: one wsa_id per line, in the same order
            # as the ARFF rows
            kwargs['sample_key'] = 'wsa'
            kwargs['sample_keys'] = [
                    int(x)
                    for x in open(druglist_path)
                    ]
        return FMBase(**kwargs)

################################################################################
# Repr mix-ins
################################################################################
class NDArrayRepr(object):
    # FMBase attr_loader and store_to_archive handle NDArrays
    def load_from_row_stream(self,data_src,cols):
        self.feature_names = cols
        sample_keys = []
        matrix = []
        for key,vec in data_src:
            sample_keys.append(key)
            matrix.append(vec)
        self.sample_keys = sample_keys
        self.data = np.array(matrix,dtype=self.python_type)
        # XXX The above implementation requires enough memory for both
        # XXX a numpy and python copy of the array. It could be reworked
        # XXX to build the numpy array up as vstacks, which would require
        # XXX only 2 numpy copies, which is smaller.

class SKSparseRepr(object):
    def _attr_loader(self,attr):
        if attr == 'data' and hasattr(self,'archive'):
            self.data = scipy.sparse.csr_matrix(
                    (
                        self.archive.get('fm_data'),
                        self.archive.get('fm_indices'),
                        self.archive.get('fm_indptr'),
                        ),
                    shape=self.archive.get('fm_shape'),
                    )
        else:
            super(SKSparseRepr,self)._attr_loader(attr)
    def _store_to_archive(self,store):
        self._store_content_to_archive(store)
    def _store_content_to_archive(self,store,exclude=None):
        if exclude is None:
            exclude = set()
        # A sparse array appears to get stored by a direct savez_compressed,
        # but it comes back as a slightly mangled ndarray. The scipy function
        # to save and restore these correctly (scipy.sparse.save_npz) seems to
        # only support a single item per file. This URL describes how
        # to save the parts of a csr matrix and re-assemble it:
        # https://stackoverflow.com/questions/8955448/
        if 'data' not in exclude:
            store.put('fm_data',self.data.data)
            store.put('fm_indices',self.data.indices)
            store.put('fm_indptr',self.data.indptr)
            store.put('fm_shape',self.data.shape)
            exclude.add('data')
        super(SKSparseRepr,self)._store_content_to_archive(store,exclude)
    def load_from_row_stream(self,data_src,cols):
        smb = SparseMatrixBuilder(
                col_map={n:i for i,n in enumerate(cols)},
                )
        used_keys=set()
        seen_keys=set()
        for key,vec in data_src:
            seen_keys.add(key)
            for col,val in zip(cols,vec):
                if val is None:
                    # convert to numpy convention for unknowns
                    val = float('nan')
                if val:
                    smb.add(key,col,val)
                    used_keys.add(key)
        self.data = smb.get_matrix(self.python_type)
        self.feature_names = smb.col_names()
        self.sample_keys = smb.row_keys()

################################################################################
# A shim for working around issues iterating through fm.data rows
################################################################################
class FeatureVector:
    def __init__(self,row,target=None):
        # row should be either a list-like object, or a csr_sparse matrix;
        # target is an optional extra value output as a final column
        self._row = row
        self._target = target
    def get_sparse_content(self):
        'Return an iterable containing (index,values) pairs for the row'
        row = self._row
        if not scipy.sparse.isspmatrix_csr(row):
            row = scipy.sparse.csr_matrix(row)
        result = zip(row.indices,row.data)
        if self._target is not None:
            result += [(row.shape[1],self._target)]
        return result
    def get_dense_content(self):
        'Return an iterable containing all row values in order'
        row = self._row
        if isinstance(row,scipy.sparse.spmatrix):
            # numpy.matrix also?
            row = row.toarray()[0]
        if self._target is not None:
            row = list(row) + [self._target]
        return row

################################################################################
# Arff file output
################################################################################
class ArffDestination:
    # All the actual formatting pieces are written as classmethods,
    # so they can be invoked in any context where arff output is
    # needed. The class instance interface wraps this so it looks
    # like creating a python list.
    @classmethod
    def arff_type(cls,typename):
        return {
                'float':'REAL',
                'bool':'{True, False}',
                }[typename]
    @classmethod
    def arff_formatter(cls,typename):
        def bool_fmt(v):
            if v:
                return 'True'
            return 'False'
        return {
                'bool':bool_fmt,
                }.get(typename,str)
    @classmethod
    def arff_val(cls,val,cast):
        if val is None or math.isnan(val):
            return '?'
        return cast(val)
    @classmethod
    def write_header(cls,fh,title,attrs):
        fh.write('@RELATION "%s"\n' % title)
        fh.write('\n')
        for name,typename in attrs:
            fh.write('@ATTRIBUTE "%s" %s\n'%(name,cls.arff_type(typename)))
        fh.write('\n')
        fh.write('@DATA\n')
    @classmethod
    def write_sparse(cls,fh,fv,fmts):
        vals = [
                ' '.join([str(idx),cls.arff_val(val,fmts[idx])])
                for idx,val in fv.get_sparse_content()
                ]
        fh.write('{'+','.join(vals)+'}\n')
    @classmethod
    def write_dense(cls,fh,fv,fmts):
        fh.write(','.join([
                cls.arff_val(x,cast)
                for x,cast in zip(fv.get_dense_content(),fmts)
                ])+'\n')
    def __init__(self,fh_or_path,title,attrs,sparse=False):
        if isinstance(fh_or_path,str):
            self.fh = open(fh_or_path,'w')
        else:
            self.fh = fh_or_path
        self.fmts = [
                self.arff_formatter(typename)
                for _,typename in attrs
                ]
        self.write_header(self.fh,title,attrs)
        self.writer = self.write_sparse if sparse else self.write_dense
    def append(self,fv):
        self.writer(self.fh,fv,self.fmts)

################################################################################
# Data Catalog wrapper
################################################################################
class AttrType:
    _type_by_name = {
                x.__name__:x
                for x in [float,int,bool,str]
                }
    def __init__(self,typestr,sparse=False):
        self._typestr = typestr
        self._sparse = sparse
    def is_sparse(self): return self._sparse
    def python_type(self): return self._type_by_name[self._typestr]

class DCSpec:
    @classmethod
    def from_joblist(cls,ws_id,joblist):
        full_codes = []
        for job_id,codes in joblist:
            full_codes += [
                    '%d_%s'%(job_id,code)
                    for code in codes
                    ]
        return cls(ws_id,full_codes)
    def __init__(self,ws_id,full_codes,catalog=None,**options):
        self._args = dict(ws_id=ws_id,full_codes=full_codes,**options)
        self._ws_id = ws_id
        self._full_codes = full_codes
        self._cat = catalog
        self.plug_unknowns = options.pop('plug_unknowns',None)
        self.key_whitelist = options.pop('key_whitelist',None)
        self.job_labels = options.pop('job_labels',{})
        # pop() and process any valid options before reaching here;
        # only the invalid ones will be left
        if options:
            raise TypeError(
                    'unexpected keyword arguments: '+' '.join(list(options.keys()))
                    )
        cat = self.get_data_catalog()
        from dtk.data import MultiMap
        # When the actual data is retrieved inside get_feature_vectors,
        # the code args get expanded to include all subcodes. So, we
        # need to do the same here to extract type information.
        type_map = MultiMap([
                (full_code,(
                        cp.valtype(),
                        # the following is the 'sparse' flag; we initially
                        # default it to True, because some codes definitely
                        # benefit from sparse storage, and it's never a
                        # terrible idea, and (initially) we don't handle
                        # more than one storage type for the entire matrix
                        # XXX when we support an HStackRepr, maybe set this
                        # XXX via a new dc Code attribute
                        True,
                        ))
                for full_code,cp in cat.flat_cp_generator(*full_codes)
                ])
        self.subspecs = []
        # XXX special case prior to HStackRepr: if both bool and float,
        # XXX represent bool as float
        if set(type_map.rev_map()) == set([('float',True),('bool',True)]):
            type_map = MultiMap([
                    (x,('float',True))
                    for x in type_map.fwd_map()
                    ])
        if len(type_map.rev_map()) == 1:
            # there's only one type of data; set shared attributes
            self.type_info = AttrType(*list(type_map.rev_map().keys())[0])
        else:
            raise NotImplementedError(
                    "unsupported type combo: %s"%(
                            ' '.join(list(type_map.rev_map().keys())),
                            )
                    )
            # create 'pure' subspecs for each type of data
            for key,s in six.iteritems(type_map.rev_map()):
                my_codes = list(s)
                # all subspecs share one data catalog copy
                ss = DCSpec(self._ws_id,my_codes,catalog=self._cat)
                self.subspecs.append(ss)
    def get_data_catalog(self):
        if self._cat is None:
            import runner.data_catalog as dc
            from runner.process_info import JobInfo
            self._cat = dc.Catalog()
            all_jobs = set([int(x.split('_')[0]) for x in self._full_codes])
            for job_id in all_jobs:
                bji = JobInfo.get_bound(self._ws_id,job_id)
                bji.fetch_lts_data()
                for cg in bji.get_data_code_groups():
                    self._cat.add_group(
                            str(job_id)+'_',
                            cg,
                            self.job_labels.get(job_id,''),
                            )
        return self._cat
    def get_codes(self): return self._full_codes

################################################################################
# Recipe mix-ins
################################################################################
class DCRecipe(object):
    _my_attrs = (
            'data',
            'sample_keys',
            'sample_key',
            'feature_names',
            )
    def _attr_loader(self,attr):
        # this expects either 'spec' is pre-loaded with a DCSpec object,
        # or 'archive' is pre-loaded with an FMArchive previously saved
        # from a DCRecipe FM
        if attr == 'spec':
            # Note: the archive wraps everything as an ndarray, so saving
            # a dict results in an array with no shape, but containing
            # one object. tolist() unwraps it, returning a dict rather
            # than a list
            specargs = self.archive.get('specargs').tolist()
            self.spec = DCSpec(**specargs)
        elif attr == 'python_type':
            self.python_type = self.spec.type_info.python_type()
        elif attr == 'typenames':
            self.typenames = [self.python_type.__name__]*len(self.feature_names)
        elif attr in self._my_attrs:
            # assemble data catalog from jobs
            cat = self.spec.get_data_catalog()
            # assemble consolidated code list
            codes = self.spec.get_codes()
            cols,data_src = cat.get_feature_vectors(*codes,
                    plug_unknowns=self.spec.plug_unknowns,
                    key_whitelist=self.spec.key_whitelist,
                    exclude_pure_unknowns=True
                    )
            if self.spec.job_labels:
                labels = [cat.get_arff_label(col) for col in cols]
                seen = set()
                any_dups = False
                for x in labels:
                    if x in seen:
                        any_dups = True
                        logger.error(f'{x} is used for multiple features')
                    seen.add(x)
                if any_dups:
                    raise ValueError('labels not distinct')
            else:
                labels = ['j'+x for x in cols]
            self.load_from_row_stream(data_src,labels)
            if codes:
                # get_feature_vectors above checks that the keyname is
                # consistent across all codes, so just get it from the
                # first one
                self.sample_key = cat.get_keyname(codes[0])
        else:
            super(DCRecipe,self)._attr_loader(attr)
    def _store_to_archive(self,store):
        store.put('specargs',self.spec._args)
        self._store_content_to_archive(store, exclude=set(self._my_attrs))
    def exclude_by_key(self,filter_keys):
        # This is a special-case function to allow the fdf plugin to
        # remove rows from an existing feature matrix. It is currently
        # only needed for FM objects built from a DCRecipe, so it's
        # implemented here. If it's needed for flat FMs in the future,
        # it will need to be implemented at the Repr level as well.
        whitelist = self.spec.key_whitelist
        if whitelist is None:
            whitelist = set(self.sample_keys)
        whitelist -= filter_keys
        self.spec.key_whitelist = whitelist
        self.spec._args['key_whitelist'] = whitelist
        # clear out dynamically loaded attributes so that they re-load
        # based on the new whitelist; because FMBase is really a
        # sklearn Bunch, you need to do this using dict syntax rather
        # than attr syntax
        for attr in self._my_attrs:
            if attr in self:
                del self[attr]

################################################################################
# FMArchive
################################################################################
import contextlib
@contextlib.contextmanager
def npz_archive(path):
    store=np.load(path, allow_pickle=True)
    yield store
    store.close()

class FMArchive:
    """Stores named numpy arrays in a .npz file.

    Similar functionality is also available in dtk.arraystore using a more
    configurable library/format than npz, consider switching at some point.
    """
    suffix='.npz'
    def __init__(self,**kwargs):
        self.path = kwargs.pop('path',None)
        self.parent = kwargs.pop('parent',None)
        self.prefix = kwargs.pop('prefix',None)
        assert not kwargs
        assert not (bool(self.path) and bool(self.parent))
        assert bool(self.prefix) == bool(self.parent) # both or nothing
        if self.prefix:
            assert self._part_nbr(self.prefix) is not None
        self._write_buf = {}
    def _part_nbr(self,element):
        import re
        m = re.match('P([0-9]+)',element)
        if m:
            return int(m.group(1))
        return None
    def _root_prefix(self):
        n = self
        prefix = ''
        while n.parent:
            prefix = n.prefix + prefix
            n = n.parent
        return n,prefix
    def _put(self,element,content):
        if self.parent:
            self.parent._put(self.prefix+element,content)
        else:
            self._write_buf[element] = content
    def put(self,element,content):
        assert self._part_nbr(element) is None
        self._put(element,content)
    def write(self,path=None):
        assert not self.parent
        stem = path or self.path
        assert stem
        #print self._write_buf
        np.savez_compressed(stem+self.suffix,**self._write_buf)
    def content(self):
        root,prefix = self._root_prefix()
        with npz_archive(root.path+root.suffix) as store:
            s = set()
            for element in store.files:
                if not element.startswith(prefix):
                    continue
                element = element[len(prefix):]
                if self._part_nbr(element) is None:
                    s.add(element)
            return s
    def get(self,element):
        assert self._part_nbr(element) is None
        root,prefix = self._root_prefix()
        with npz_archive(root.path+root.suffix) as store:
            return getattr(store.f,prefix+element)
    def partitions(self):
        root,prefix = self._root_prefix()
        with npz_archive(root.path+root.suffix) as store:
            s = set()
            for element in store.files:
                if not element.startswith(prefix):
                    continue
                element = element[len(prefix):]
                part = self._part_nbr(element)
                if part is not None:
                    s.add(part)
            return s
    def partition(self,num):
        return self.__class__(parent=self,prefix='P%d'%num)
    def _dump(self,indent):
        for element in self.content():
            print(indent,element,self.get(element))
        for part in self.partitions():
            sub = self.partition(part)
            print(indent,'Partition',part,':')
            sub._dump(indent+'   ')
    def dump(self):
        print('Base:')
        self._dump('  ')

################################################################################
# Reformat data for sparse matrices
################################################################################
class ExpandingIndex:
    # This class acts like a dict where, each time an undefined key is
    # accessed, it gets the next consecutive value (starting from 0).
    def __init__(self):
        self.__impl = {}
    def __getitem__(self,key):
        if key not in self.__impl:
            self.__impl[key] = len(self.__impl)
        return self.__impl[key]
    def __getattr__(self,attr):
        return getattr(self.__impl,attr)

class SparseMatrixBuilder:
    '''Accumulate cells to define a sparse matrix.

    Cells are identified by row keys and column names, which can be
    anything. These are mapped to consective integers to provide row
    and column indexes. This may be done automatically, or pre-defined
    maps may be passed in. If a predefined map is passed, cells not
    in the maps will be ignored.

    Once the data is assembled, you can extract the sparse matrix
    itself, and ordered lists of the row keys and column names.
    '''
    # This class accumulates individual cells into a sparse matrix.
    def __init__(self,row_map=None,col_map=None):
        self.row_map = ExpandingIndex() if row_map is None else row_map
        self.col_map = ExpandingIndex() if col_map is None else col_map
        self.row_coord=[]
        self.col_coord=[]
        self.vals=[]
    def add(self,row_key,col_name,val):
        try:
            row_idx = self.row_map[row_key]
            col_idx = self.col_map[col_name]
        except KeyError:
            return
        self.row_coord.append(row_idx)
        self.col_coord.append(col_idx)
        self.vals.append(val)
    def get_matrix(self,dtype):
        from scipy import sparse
        return sparse.csr_matrix(
                (self.vals, (self.row_coord, self.col_coord)),
                dtype=dtype,
                )
    @staticmethod
    def _map2list(d):
        return [
                x[0]
                for x in sorted(six.iteritems(d),key=lambda x:x[1])
                ]
    def col_names(self):
        return self._map2list(self.col_map)
    def row_keys(self):
        return self._map2list(self.row_map)
    def to_wrapper(self, dtype):
        return SparseMatrixWrapper(self.get_matrix(dtype), self.row_keys(), self.col_names())


class SparseMatrixWrapper:
    def __init__(self, csr_matrix, row_names, col_names):
        self.mat = csr_matrix
        self.row_names = row_names
        self.col_names = col_names
        self.row_to_idx = {row: idx for idx, row in enumerate(row_names)}
        self.col_to_idx = {col: idx for idx, col in enumerate(col_names)}

    def __getitem__(self, arg):
        if isinstance(arg, tuple):
            row_name, col_name = arg
            row_idx = self.row_to_idx[row_name]
            col_idx = self.col_to_idx[col_name]
            return self.mat[row_idx, col_idx]
        else:
            row_name = arg
            row_idx = self.row_to_idx[row_name]

            ind_start = self.mat.indptr[row_idx]
            ind_end = self.mat.indptr[row_idx+1]
            col_idxs = self.mat.indices[ind_start:ind_end]
            cell_values = self.mat.data[ind_start:ind_end]

            col_names = [self.col_names[i] for i in col_idxs]
            return dict(zip(col_names, cell_values))