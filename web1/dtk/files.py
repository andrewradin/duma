import logging
logger = logging.getLogger(__name__)


class NoMatchesError(RuntimeError):
    pass

class MultipleMatchesError(RuntimeError):
    pass

################################################################################
# Get a file's date in a django-friendly TZ-aware datetime
################################################################################
def modification_date(filename):
    import os
    from django.utils.timezone import utc
    import datetime
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t,utc)
################################################################################
# Convenience operations on the filesystem
################################################################################
def remove_if_present(path):
    import os
    try:
        os.remove(path)
    except OSError as e:
        import errno
        if e.errno != errno.ENOENT:
            raise

def remove_tree_if_present(path):
    import shutil
    try:
        shutil.rmtree(path)
    except OSError as e:
        import errno
        if e.errno != errno.ENOENT:
            raise

def rm_readonly_tree(root):
    import os
    if not os.path.exists(root):
        return
    # shutil.rmtree will fail if tree contains any read-only files
    # or directories; assure that isn't the case
    import stat
    filemode=stat.S_IREAD+stat.S_IWRITE
    dirmode=stat.S_IREAD+stat.S_IWRITE+stat.S_IEXEC
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames:
            os.chmod(os.path.join(dirpath,name),dirmode)
        for name in filenames:
            try:
                os.chmod(os.path.join(dirpath,name),filemode)
            except OSError:
                # tolerate case where file is a broken symlink
                pass
        import shutil
    shutil.rmtree(root)

################################################################################
# Tools for writing text files
################################################################################
class FileDestination:
    '''Accept tuples and place them in a tsv file.

    The client interface allows a generic python list object to be passed
    instead, and is meant to be generic enough that shims like this can
    be implemented for other destinations, and interchangably paired up
    with other data sources.

    The close() method allows the caller to force a flush of data to the
    file; it can also be invoked by using FileDestination as a context
    manager ("with FileDestination(...) as dest:").
    '''
    # This is loosely based on the old FAERS flow-through query stuff:
    # https://github.com/andrewradin/twoxar-demo/blob/46009c1dfcc0373223e906efed3ffb9f7b9efc44/web1/dtk/faers.py#L665
    # XXX This could grow more features, like determining compression and
    # XXX delimiter from the filename. It could also maybe gain some
    # XXX efficiency by using an output pipeline corresponding to
    # XXX get_file_lines below.
    # There's a bit of assymmetry here, since this function handles converting
    # python types to text, but get_file_records can't convert back to
    # non-text types (since prior knowledge of the desired types is needed).
    def __init__(self,path,header=None,delim='\t',gzip=False):
        if gzip:
            import gzip
            self._fh = gzip.open(path,'wb')
        else:
            self._fh = open(path,'wb')
        self._delim = delim
        if header:
            self.append(header)
    def append(self,vec):
        try:
            self._fh.write((self._delim.join([
                    str(x)
                    for x in vec
                    ])+'\n').encode())
        except UnicodeEncodeError as ex:
            raise UnicodeEncodeError(
                    ex.args[0],
                    ex.args[1],
                    ex.args[2],
                    ex.args[3],
                    ex.args[4]+'; vec='+repr(vec),
                    )
    def close(self):
        self._fh.close()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

################################################################################
# Tools for reading text files
################################################################################
keep_header_default = True

def get_file_lines(filename,
            zcat=None, # to auto-detect, or True or False to force
            grep=None, # or a list of grep options
            sort=False, # or True
            keep_header=keep_header_default,
                    # or False to strip or None if none expected
            externalize_zcat = True, # or False to use gzip module if possible
            cut = 0,
            tempfiles = [], # a hook for deleting NamedTemporaryFiles
            progress = False, # Show a progress bar (only for non-pipelined)
            zstd=None, # force zstd decompression
            ):
    '''Return an iterator over lines from a text file.

    By default, returns all lines in order, unzipping the file if necessary.
    May perform an optional grep and/or sort on file content.  If the file
    is expected to contain a header line, that line can be stripped, or
    can be passed through untouched as the first line, independently of any
    grep or sort.
    '''
    if zcat is None:
        zcat = filename.endswith('.gz') or filename.endswith('.bgz')
    
    if zstd is None:
        zstd = filename.endswith('.zst')

    if progress:
        # Hard to get current progress with external zcat, and it's not wildly faster
        # anymore.
        externalize_zcat = False
    # XXX Header handling could maybe be a bit simpler. Right now, the
    # XXX header line bypasses grep and sort, and (conceptually) gets
    # XXX routed through zcat and cut. But the zcat'ing happens out-of-line
    # XXX anyway, and the cut code could maybe be done out-of-line as well
    # XXX (and in any event is kind of hacked-in anyway, and could use
    # XXX some review). If it all happened out-of-line up here, the
    # XXX pipeline would only need to strip the first line or not.
    if grep or sort or (zcat and externalize_zcat) or cut:
        pipeline = build_zcat_grep_sort_pipeline(
                            filename,zcat,grep,sort,keep_header,cut
                            )
        fh = open_pipeline(pipeline)
        # Unsupported
        progress = False
    else:
        if progress:
            import os
            size_bytes = os.path.getsize(filename)
            from tqdm import tqdm
            progress_bar = tqdm(total=size_bytes, unit='B', unit_scale=True, unit_divisor=1024)
            desc = progress if isinstance(progress,str) else os.path.basename(filename)
            progress_bar.set_description(desc)
            progress_bar.current = 0

        # just open the file, and strip the header if asked
        import io
        if zcat:
            import isal.igzip as igzip
            if progress:
                # Need the underlying compressed handle so we know where
                # we are in the file for progress.
                raw_fh = open(filename, 'rb')
                fh = io.TextIOWrapper(igzip.GzipFile(fileobj=raw_fh, mode='r'))
            else:
                fh = igzip.open(filename, 'r')
        elif zstd:
            import zstandard
            if progress:
                # Need the underlying compressed handle so we know where
                # we are in the file for progress.
                raw_fh = open(filename, 'rb')
                fh = zstandard.open(raw_fh, mode='rt')
            else:
                fh = zstandard.open(filename, 'rt')
        else:
            if progress:
                raw_fh = open(filename, 'rb')
                fh = io.TextIOWrapper(raw_fh)
            else:
                fh = open(filename)

        def update_progress():
            spot = raw_fh.tell()
            progress_bar.update(spot - progress_bar.current)
            progress_bar.current = spot

        if keep_header is False:
            next(fh)

    # by yielding each line here, rather than returning fh, we assure that
    # the context manager gets invoked; if we leave this to the caller, there's
    # an inconsistency between get_file_lines and get_file_records
    with fh as src:
        for i, line in enumerate(src):
            yield line
            if progress and i % 100 == 0: # only update progress every N lines
                update_progress()
    # Once we know the pipeline is complete, close any temporary files
    # the client code has asked us to manage. This provides an easy hook
    # for passing data to pipeline components in NamedTemporaryFiles,
    # and assuring those files exist while the pipeline is running, but
    # get deleted afterwards.
    for fh in tempfiles:
        fh.close()

    if progress:
        progress_bar.close()

# TODO allow for specifying a delimiter to cut
def build_zcat_grep_sort_pipeline(filename,zcat,grep,sort,keep_header,cut):
    pipeline = []
    merge_header = False
    if zcat:
        # even if we're doing nothing else, this is faster than
        # the python gzip library
        pipeline.append(['zcat'])
    if (grep or sort) and keep_header:
        # we need special processing to preserve the header line;
        # first, grab the header line locally
        if zcat:
            import gzip
            fh = gzip.open(filename, 'rt')
        else:
            fh = open(filename)
        header = next(fh).rstrip('\n')
        # add a pipeline step to strip it out of the grep/sort data
        pipeline.append(['tail','-n','+2'])
        # set a flag to merge our stashed copy back in at the end
        merge_header=True
    if keep_header is False:
        pipeline.append(['tail','-n','+2'])
    if grep:
        # do this before the sort, if any, to reduce sorting time
        pipeline.append(['grep']+grep)
    if cut:
        # do this before the sort, if any, to reduce sorting time
        pipeline.append(['cut', '-f', cut])
    if sort:
        # XXX allow passing sort parameters?
        # use the C locale to avoid unicode overhead
        import os
        d=dict(os.environ)
        d['LC_ALL'] = 'C'
        cmd = [d,'sort']
        # If the length of the pipeline is non-zero prior to adding
        # the sort stage, sort's input will be a pipe rather than
        # a file that it opens itself.  In this case, sort does some
        # non-optimal things (not using as much memory as it might, and
        # not using multiple cores).  In measurements on dev01, sorting
        # a 2.7G file, this increases the sort time from 1:07 minutes to
        # 1:49 minutes.  On both dev01 (2 cores) and my laptop (8 cores)
        # we can get most of this back by adding the -S25% option.
        if pipeline:
            cmd.append('-S25%')
        pipeline.append(cmd)
    if merge_header:
        # put the header back in ahead of the pipeline data
        if "'" not in header:
            pipeline.append(['bash','-c',"(cat <<<'%s'; cat -)"%header])
        else:
            # we can try switching to double quotes, but that has issues
            # of its own; just die if it's going to be a problem
            import re
            if re.search(r'[$`"!]',header):
                raise ValueError('''
                headers with single quotes can't also have special
                shell characters when you're doing sort or grep processing
                ''')
            pipeline.append(['bash','-c',"(cat <<<\"%s\"; cat -)"%header])
    # now check for potential optimizations
    first_2_cmds = [x[0] for x in pipeline][:2]
    if first_2_cmds == ['zcat','grep']:
        # replace with a single zgrep
        pipeline=pipeline[1:]
        pipeline[0][0] = 'zgrep'
    # first command in pipeline needs filename
    pipeline[0].append(filename)
    return pipeline

def open_pipeline(pipeline):
    '''Return a file-like object for reading from a pipeline.

    pipeline - a list with one element per pipeline stage.  Each element
    is a list defining a command in the style of subprocess, except that
    if the element list begins with a dict, it is extracted and used
    as the environment for that pipeline stage.
    '''
    prev = None
    procs = []
    import subprocess
    for stage in pipeline:
        if type(stage[0]) == dict:
            env = stage[0]
            stage = stage[1:]
        else:
            env = None
        proc = subprocess.Popen(stage,
                    stdin=prev,
                    env=env,
                    stdout=subprocess.PIPE,
                    bufsize=-1,
                    )
        procs.append(proc)
        prev = proc.stdout
    # return file handle to pipe from last stage, wrapped
    # in something that will clean up at the end; by doing this instead of
    # using contextwrapper, we get an oject that can be used with or
    # without a 'with' statement.
    class Wrapper:
        def __init__(self,fh,procs):
            self._fh=fh;
            self._procs=procs
        def __getattr__(self,attr):
            return getattr(self._fh,attr)
        def __enter__(self):
            # Subprocess output is a bytestring in python3.  Decode it to
            # a normal string like consumers of this class expect.
            import codecs
            return codecs.iterdecode(self._fh, 'utf8')
        def __exit__(self,*args):
            self.close()
            return self._fh.__exit__(self._fh,*args)
        def close(self):
            self._fh.close()
            while self._procs:
                self._procs[-1].kill()
                self._procs[-1].wait()
                self._procs.pop()
    return Wrapper(prev,procs)

# TODO add the ability to set a deliminter
def get_file_columns(fn,columns,**kwargs):
    return get_file_lines(fn
                          ,cut=','.join([str(c) for c in columns])
                          ,**kwargs
                         )

def filter_rows_wrapper(src,select,keep_header):
    src = iter(src)
    if keep_header == True:
        header = next(src)
        yield header # don't filter first line
    # else it's already been stripped, or it's not special
    patterns,col = select

    # If col is the name of a column instead of the index, convert to index.
    if col is not None and isinstance(col, str):
        col = header.index(col)

    if col is None:
        patterns = set(patterns)
        for row in src:
            for entry in row:
                if entry in patterns:
                    yield row
                    break
    elif len(patterns) == 1:
        pattern = list(patterns)[0]
        for row in src:
            if row[col] == pattern:
                yield row
    else:
        patterns = set(patterns)
        for row in src:
            if row[col] in patterns:
                yield row

def get_file_records(fn,parse_type=None,select=None,progress=False,allow_suspicious=False,**kwargs):
    """
    select is a tuple ([patterns], col), indicating that we should return
    only the rows that match a pattern in the specified column number.  col can
    be None, which will look in any column.
        This makes use of grep functionality to be particularly fast and able
        to handle arbitrarily large pattern lists.
    
    if progress is specified, will display tqdm progress bar as it iterates if possible.

    allow_suspicious is currently unused and will be removed.
    """
    if parse_type == None:
        parse_type = fn.split('.')[-1]
        if parse_type == 'gz':
            parse_type = fn.split('.')[-2]
        elif parse_type == 'bgz':
            parse_type = fn.split('.')[-2]
    if parse_type == 'sqlsv':
        unsupported = {'cut', 'sort', 'grep'} & kwargs.keys()
        assert not unsupported, f'{unsupported} not supported in sqlsv yet for {fn}'
        from dtk.tsv_alt import SqliteSv
        sql = SqliteSv(fn)
        rec_args = {}
        if select:
            # The parameter can be either a column index or the name of the column.
            if isinstance(select[1], str):
                col_name = select[1]
            else:
                col_name = sql.get_header()[select[1]]
            key = f'filter__{col_name}__in'

            def run_chunks():
                # To support querying large (>30k) numbers of elements, we have to break it into chunks.
                # For anything less than the max-per-query, this just does 1 chunk.
                from dtk.parallel import chunker
                keep_header = kwargs.get('keep_header', False)
                for chunk in chunker(list(select[0]), chunk_size=SqliteSv.MAX_VARS):
                    rec_args[key] = chunk
                    yield from sql.get_records(keep_header=keep_header, **rec_args)
                    # Only keep the header if requested for the first chunk, don't insert into the middle.
                    keep_header=False
            return run_chunks()

        return sql.get_records(keep_header=kwargs.get('keep_header', False), **rec_args)

    if select and 'grep' not in kwargs:
        # Note that the 'real' selection happens below in filter_rows_wrapper,
        # but if the client code didn't specify a grep of its own, use the
        # grep parameter to toss rows that clearly don't match.
        # XXX it's often the case that the column specified in select[1] is
        # XXX 0; it may be faster in that case to construct a regex that only
        # XXX looks at the beginning of the line (although this may not work
        # XXX well with lots of patterns in select[0])
        import tempfile
        fh = tempfile.NamedTemporaryFile(mode='wt+', encoding='utf8')
        for value in select[0]:
            fh.write(value+'\n')
        fh.flush()
        kwargs['grep'] = ['-w','-F','-f',fh.name]
        try:
            oldlist = list(kwargs['tempfiles'])
            oldlist.append(fh)
        except KeyError:
            oldlist = [fh]
        kwargs['tempfiles'] = oldlist
    src = get_file_lines(fn,progress=progress,**kwargs)
    from dtk.readtext import parse_delim
    if parse_type == 'tsv':
        src = parse_delim(src)
    elif parse_type == 'csv':
        src = parse_delim(src,delim=',')
    elif parse_type == 'csv_strict':
        import csv
        src = csv.reader(src)
    elif parse_type == 'ssv':
        src = parse_delim(src,delim=' ')
    elif parse_type == 'psv':
        src = parse_delim(src,delim='|')
    else:
        raise NotImplementedError("parse_type='%s'"%repr(parse_type))
    if select:
        keep_header = kwargs.get('keep_header',keep_header_default)
        # grep result isn't guaranteed -- it's just a speedup; so implement
        # fallback filtering here
        src = filter_rows_wrapper(src,select,keep_header)
    return src

################################################################################
# Tools for scanning directories
################################################################################
def scan_dir(path,filters=[],output=lambda x:x.full_path):
    '''Return directory content, with configurable filtering and output.

    For each item in the directory, an object is constructed with 2 attributes:
    - 'filename' is the name within the directory
    - 'full_path' is 'filename' with 'path' pre-pended
    'filters' is a list of callables which take the above object as a parameter
    and return a bool. Any falseish return from any filter excludes that item
    from consideration.
    'output' is a callable that takes the above object and returns a
    representation of the item to be yielded by scan_dir.
    '''
    from collections import namedtuple
    Item=namedtuple('Item','full_path filename')
    import os
    for name in os.listdir(path):
        i=Item(os.path.join(path,name),name)
        for f in filters:
            if not f(i):
                i = None
                break
        if i:
            yield output(i)

def name_match(regex):
    '''Return a scan_dir filter applying a regex to the filename.'''
    import re
    return lambda item:re.match(regex,item.filename)

def is_file(item):
    '''scan_dir filter selecting only files (i.e. not subdirs).'''
    import os
    return os.path.isfile(item.full_path)

def is_dir(item):
    '''scan_dir filter selecting only files (i.e. not subdirs).'''
    import os
    return os.path.isdir(item.full_path)

def get_dir_file_names(path):
    '''wrap scan_dir for the common case of retrieving file names.'''
    return scan_dir(
                path,
                filters=[is_file],
                output=lambda x:x.filename,
                )

def get_dir_matching_paths(path,regex):
    '''wrap scan_dir for the common case of retrieving regex matches.'''
    return scan_dir(
                path,
                filters=[name_match(regex)],
                )

################################################################################
# A context manager to suppress output from noisy functions
#
# There are a couple of places this can get used in several places at the same
# time (particularly in the refresh workflow test).  It doesn't really
# matter which replacement wins in that case, but we do want to make sure
# that at the very end stdout does get restored correctly.
################################################################################
import sys
class Quiet:
    import threading
    _fh_stack = []
    _lock = threading.RLock()
    _devnull = None

    def __init__(self,replacement=None):
        self.replacement = replacement
    def __enter__(self):
        import sys
        with self._lock:
            self._push(sys.stdout)
            sys.stdout = self.replacement or self._get_devnull()

    def __exit__(self,type,value,traceback):
        import sys
        with self._lock:
            # This might restore you to something unexpected if you have
            # partially overlapping Quiets.  But when the last one goes
            # away, you should get proper stdout back.
            sys.stdout = self._pop()

    @classmethod
    def _get_devnull(cls):
        with cls._lock:
            if not cls._devnull:
                import os
                cls._devnull = open(os.devnull, 'w')
        return cls._devnull

    @classmethod
    def _push(cls, fh):
        cls._fh_stack.append(fh)

    @classmethod
    def _pop(cls):
        return cls._fh_stack.pop()

################################################################################
# File name conventions
################################################################################
def safe_name(s):
    import re
    pattern = re.compile(r'[\W_]+')
    return pattern.sub('_', s)

class AttrFileName(object):
    def set_name(self,name):
        parts = name.split(self._sep)
        if parts[-1] != self.format:
            raise ValueError('illegal file name')
        if parts[0] == 'create' and len(parts) == 4:
            self.detail = parts[2]
        elif parts[0] in ('patch','pt','rt','m') and len(parts) == 5:
            self.detail = parts[3]
        else:
            raise ValueError('illegal file name')
        self.use = parts[0]
        self.collection = self._sep.join(parts[1:3])
    def get_name(self):
        parts = [self.use,self.collection]
        if self.use == 'patch':
            parts.append(self.detail)
        parts.append(self.format)
        return self._sep.join(parts)
    def set_create(self,key_name,detail):
        self.use = 'create'
        self.collection = self._sep.join((key_name,detail))
        self.detail = detail
    def set_patch(self,collection,detail):
        self.use = 'patch'
        self.collection = collection
        self.detail = detail
    def __init__(self,file_name='create...tsv'):
        self._sep = '.'
        self.format = 'tsv'
        self.set_name(file_name)
    def key_name(self):
        parts = self.collection.split(self._sep)
        return parts[0] + '_id'
    def is_master_create(self):
        # for create files, the convention is that '<keyname>.full'
        # contains all drugs of interest from the collection,
        # and others are subsets; it's ok for drugs to be excluded
        # from full if they're excluded from all others as well
        return self.use == 'create' and self.detail == 'full'
    def ok(self):
        parts = self.collection.split(self._sep)
        return parts[0] and parts[1]

from dtk.lazy_loader import LazyLoader

class UnknownRoleError(Exception):
    pass

class VersionedFileName:
    class Meta(LazyLoader):
        # defaults
        prefix=''
        flavor_len = 1
        roles = []
        no_compress = set()
        no_decompress = set()
        # everything defined up to here can be overridden by init keyword args
        _kwargs=[x for x in locals().keys() if not x.startswith('__')]
        def _role_len_loader(self):
            if self.roles:
                return len(self.roles[0].split('.'))
            return 0
        def compress_on_write(self,filename):
            vfn = VersionedFileName(meta=self,name=filename)
            return vfn.role not in self.no_compress
        def decompress_on_read(self,filename):
            vfn = VersionedFileName(meta=self,name=filename)
            return vfn.role not in self.no_decompress
    @classmethod
    def _get_meta(cls,meta=None,file_class=None):
        # meta can be set explicitly or via (registered) file_class
        if file_class:
            assert meta is None, f"Couldn't find meta for {file_class}"
            return cls.meta_lookup[file_class]
        assert meta is not None
        return meta
    def __init__(self,meta=None,file_class=None,name=None):
        self.meta = self._get_meta(meta,file_class)
        if name:
            self.from_string(name)
    def from_string(self,name):
        def parse(count,parts):
            return '.'.join(parts[:count]),parts[count:]
        parts = name.split('.')
        # get prefix
        prefix_len = 1 if self.meta.prefix else 0
        self.prefix, parts = parse(prefix_len,parts)
        assert self.prefix == self.meta.prefix, f'Expected "{self.prefix}" to be "{self.meta.prefix}" in "{name}"'
        # get flavor
        self.flavor, parts = parse(self.meta.flavor_len,parts)
        # get version
        vstr, parts = parse(1,parts)
        assert vstr and vstr[0] == 'v',f"bad version format in '{name}'"
        self.version = int(vstr[1:])
        # get role
        self.role, parts = parse(self.meta.role_len,parts)
        if self.meta.role_len:
            if self.role not in self.meta.roles:
                # We don't want to let you construct this.  But if we're
                # listing out all files in a directory, you probably want
                # to just skip this one, as it's probably just new data this
                # branch doesn't know about yet.
                raise UnknownRoleError(f"in '{name}' (role: {self.role} known: {self.meta.roles})")
        # get format
        # XXX we could verify that all remaining parts are from some
        # XXX controlled vocabulary, but just trust for now
        assert len(parts) >= 1
        self.format = '.'.join(parts)
    def to_string(self):
        parts=[]
        if self.meta.prefix:
            parts.append(self.meta.prefix)
        if self.meta.flavor_len:
            assert len(self.flavor.split('.')) == self.meta.flavor_len
            parts.append(self.flavor)
        parts.append('v%d'%self.version)
        if self.meta.roles:
            assert self.role in self.meta.roles, f"Couldn't find '{self.role}' in {self.meta.roles}"
            assert len(self.role.split('.')) == self.meta.role_len
            parts.append(self.role)
        parts.append(self.format)
        return '.'.join(parts)
    @staticmethod
    def label(flavor,version):
        parts=[]
        if flavor:
            parts.append(flavor)
        parts.append('v%d'%version)
        return '.'.join(parts)
    # in both this and the function below, an optional path is stripped
    # from the list of files (only the filename part is relevant)
    @classmethod
    def get_choices(cls,meta=None,file_class=None,paths=None,keep_roles=None):
        meta = cls._get_meta(meta,file_class)
        by_flavor = {}
        if not keep_roles and meta.roles:
            keep_roles = meta.roles[:1]
        import os
        for path in paths:
            dirpath,name = os.path.split(path)
            try:
                vfn = cls(meta=meta,name=name)
            except UnknownRoleError:
                logger.warning("Skipping %s, unknown role", name)
                continue
            if keep_roles and vfn.role not in keep_roles:
                continue # only keep main role name
            # Use a set here in case there are multiple formats available for
            # the same dataset, suppress duplicates.
            by_flavor.setdefault(vfn.flavor,set()).add(vfn.version)
        sorted_flavors = sorted(by_flavor.keys())
        # first include most recent version of each flavor
        result = []
        for flavor in sorted_flavors:
            by_flavor[flavor] = list(sorted(by_flavor[flavor], reverse=True))
            l = by_flavor[flavor]
            result.append(cls.label(flavor,l[0]))
        # then include all older versions of each flavor
        for flavor in sorted_flavors:
            for v in by_flavor[flavor][1:]:
                result.append(cls.label(flavor,v))
        return [(x,x) for x in result]
    # this class can't fully reconstruct a file name from a choice string
    # because:
    # - the format information isn't available
    # - a single choice string potentially represents several files varying
    #   by role
    # so, this is implemented as a filter
    @classmethod
    def get_matching_files(cls,
            meta=None,
            file_class=None,
            choice=None,
            paths=None,
            role=None,
            format=None,
            ):
        assert choice
        parts = choice.split('.')
        assert parts[-1][0] == 'v', f'Expected {parts[-1]} to be a version'
        flavor = '.'.join(parts[:-1])
        version = int(parts[-1][1:])
        result = []
        meta = cls._get_meta(meta,file_class)
        import os
        for path in paths:
            dirpath,name = os.path.split(path)
            try:
                vfn = cls(meta=meta,name=name)
            except UnknownRoleError:
                logger.warning("Skipping %s, unknown role", name)
                continue
            if vfn.flavor == flavor and vfn.version == version:
                if ((role is None or vfn.role == role)
                    and (format is None or vfn.format == format)):
                    result.append(path)
        return result
    @classmethod
    def get_matching_path(cls,**kwargs):
        l = cls.get_matching_files(**kwargs)
        if len(l) == 0:
            raise NoMatchesError(f'no unique match: {repr(l)} for {kwargs}')
        elif len(l) > 1:
            raise MultipleMatchesError('no unique match: '+repr(l))
        return l[0]

from path_helper import PathHelper,AutoCompressMode
def spec_to_parms(spec):
    roles=[]
    no_compress = set()
    no_decompress = set()
    for role in spec[2]:
        if isinstance(role,str):
            mode = AutoCompressMode.COMP_AND_DECOMP
        else:
            role,mode = role
        if mode == AutoCompressMode.NO_COMPRESS:
            no_compress.add(role)
            no_decompress.add(role)
        elif mode == AutoCompressMode.COMPRESS_ONLY:
            no_decompress.add(role)
        roles.append(role)
    return dict(
            prefix=spec[0],
            flavor_len=spec[1],
            roles=roles,
            no_compress=no_compress,
            no_decompress=no_decompress,
            )
VersionedFileName.meta_lookup = {
        spec[0]:VersionedFileName.Meta(**spec_to_parms(spec))
        for spec in PathHelper.vbucket_specs
        }


def gzip_file(fn_in, fn_out):
    import gzip
    import shutil
    with gzip.GzipFile(fn_out, 'wb') as f_out:
        with open(fn_in, 'rb') as f_in:
                shutil.copyfileobj(f_in, f_out)

def cmp_uncmp_fn(fn):
    """Returns the .gz and non-.gz versions of a filename"""
    if fn.endswith('.gz'):
        cmp_fn = fn
        uncmp_fn = fn[:-3]
    else:
        uncmp_fn = fn
        cmp_fn = fn + '.gz'
    return cmp_fn, uncmp_fn

def open_cmp(fn, mode='r', *args, **kwargs):
    """Convenience function for dealing with paths that may or may not be gzipped.

    This often comes up when dealing with outputs that once were written uncompressed and are now written compressed.

    When reading, will open either the .gz or the uncompressed version of the file,
    whichever exists (preferring uncompressed).

    When writing, will always write compressed (and append a .gz suffix if the filename doesn't provide one)
    """
    cmp_fn, uncmp_fn = cmp_uncmp_fn(fn) 

    import os
    import gzip
    if 'r' in mode:
        if os.path.exists(uncmp_fn):
            return open(uncmp_fn, mode, *args, **kwargs)
        elif os.path.exists(cmp_fn):
            if 'b' not in mode and 't' not in mode:
                # Paper over the usual differences between gzip and non-gzip.
                mode += 't'

            return gzip.open(cmp_fn, mode, *args, **kwargs)
        else:
            raise FileNotFoundError(f"Neither {cmp_fn} nor {uncmp_fn} exist.")
    else:
        if 'b' not in mode and 't' not in mode:
            # Paper over the usual differences between gzip and non-gzip.
            mode += 't'

        return gzip.open(cmp_fn, mode, *args, **kwargs)

def path_exists_cmp(fn):
    """Returns whether this path or its compressed equivalent exist."""
    import os
    cmp_fn, uncmp_fn = cmp_uncmp_fn(fn) 
    return os.path.exists(cmp_fn) or os.path.exists(uncmp_fn)
