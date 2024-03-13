
import logging
logger = logging.getLogger(__name__)
class S3Bucket:
    list_cache_fn = '__list_cache'
    lock_fn='__lock'
    def __init__(self,file_class):
        from path_helper import PathHelper
        try:
            self.cache_path = PathHelper.s3_cache_path(file_class)
            from dtk.aws_api import Bucket
            self.bucket = Bucket(
                    name='2xar-duma-'+file_class,
                    cache_path=self.cache_path,
                    )
        except AttributeError:
            # if the file_class isn't a 'classic' s3 cache bucket, assume
            # we're dealing with a virtual/versioned bucket
            from dtk.files import VersionedFileName
            self.meta = VersionedFileName.meta_lookup[file_class]
            from dtk.aws_api import VBucket
            self.bucket = VBucket(file_class=file_class)
            self.cache_path = self.bucket.cache_path
    # autocompress support
    def compress_on_write(self,filename):
        try:
            return self.meta.compress_on_write(filename)
        except AttributeError:
            # non-versioned case
            return False
    def decompress_on_read(self,filename):
        try:
            return self.meta.decompress_on_read(filename)
        except AttributeError:
            # non-versioned case
            return False
    def _get_s3_list(self,dezip=False,cache_ok=False):
        # This must be called with the lock asserted!
        s = set()
        need_s3_fetch = True
        if cache_ok:
            try:
                f = open(self.cache_path+'/'+self.list_cache_fn)
                for name in f:
                    s.add(name.strip())
                f.close()
                need_s3_fetch = False
            except IOError:
                pass
        if need_s3_fetch:
            f = open(self.cache_path+'/'+self.list_cache_fn,'w')
            for name in self.bucket.namelist():
                if dezip and name.endswith('.gz'):
                    name = name[:-3]
                f.write(name+'\n')
                s.add(name)
            f.close()
        return s
    # XXX The dezip option is problematic because a call by one client
    # XXX will affect the contents of the cache file for other clients,
    # XXX so all clients of the same bucket need to be coordinated.
    # XXX It is currently used only for d2ps and dpi.
    def list(self,dezip=False,cache_ok=False):
        # Typically this gets created via build_(dev|prod)_symlinks in install.sh,
        # but let's just create it if it doesn't already exist.  Particularly useful
        # for tests, or if you've pulled in a new branch without install.sh'ing yet.
        import os
        os.makedirs(self.cache_path, exist_ok=True)
        from dtk.lock import FLock
        with FLock(self.cache_path+"/"+self.lock_fn) as lock:
            s = self._get_s3_list(dezip,cache_ok)
            # Strip .gz suffix from anything that will be auto-decompressed.
            # Also, if files should be compressed on S3, but aren't, make
            # them look correct (they'll be compressed on fetch).
            s2 = set()
            for name in s:
                from dtk.files import UnknownRoleError
                try:
                    if self.decompress_on_read(name) and name.endswith('.gz'):
                        name = name[:-3]
                    # XXX maybe push this conditional down?
                    if self.compress_on_write(name) \
                            and not self.decompress_on_read(name) \
                            and not name.endswith('.gz'):
                        name += '.gz'
                    s2.add(name)
                except UnknownRoleError:
                    logger.warning("Ignoring unknown role of file %s, don't know how to decompress", name)

            s = s2
            import os
            for file in os.listdir(self.cache_path):
                if file in (self.list_cache_fn,self.lock_fn):
                    continue
                if file.startswith('.'):
                    continue # skip hidden files, e.g. .swp files
                s.add(file)
            l = list(s)
            l.sort()
            return l

class S3MiscBucket(S3Bucket):
    '''Allow S3File fetch from duma-datasets bucket.

    This bucket doesn't correspond to the bucket and directory naming
    conventions assumed by S3Bucket. Just override init to set paths
    differently.
    '''
    def __init__(self):
        from path_helper import PathHelper
        self.cache_path = PathHelper.s3_cache_root
        from dtk.aws_api import Bucket
        self.bucket = Bucket(name='duma-datasets',cache_path=self.cache_path)

class S3File:
    def __init__(self,bucket,file_name):
        if isinstance(bucket,str):
            self.s3 = S3Bucket(bucket)
        else:
            self.s3 = bucket
        self.file_name = file_name
    @classmethod
    def get_versioned(cls,file_class,choice,role=None,format=None):
        s3b = S3Bucket(file_class)
        from dtk.files import VersionedFileName
        try:
            path = VersionedFileName.get_matching_path(
                    file_class=file_class,
                    choice=choice,
                    role=role,
                    format=format,
                    paths=s3b.list(cache_ok=True),
                    )
        except RuntimeError:
            # Try again but force refresh cache.
            path = VersionedFileName.get_matching_path(
                    file_class=file_class,
                    choice=choice,
                    role=role,
                    format=format,
                    paths=s3b.list(cache_ok=False),
                    )
        import os
        dirname,filename = os.path.split(path)
        return S3File(s3b,filename)
    def path(self):
        return self.s3.cache_path+self.file_name
    def fetch(self,unzip=False,force_refetch=False):
        from dtk.lock import FLock
        lock = FLock(self.s3.cache_path+"/"+self.s3.lock_fn)
        lock.acquire()
        try:
            import os
            if force_refetch or not os.path.isfile(self.path()):
                post_process = None
                if unzip:
                    # legacy option: it's on the client to pass in the unzipped
                    # name, and to know that the file on S3 is zipped
                    s3_name = self.file_name+'.gz'
                    post_process = 'unzip'
                elif self.s3.compress_on_write(self.file_name):
                    if self.file_name.endswith('.gz'):
                        zipped_name = self.file_name
                        unzipped_name = self.file_name[:-3]
                    else:
                        unzipped_name = self.file_name
                        zipped_name = self.file_name + '.gz'
                    # try to avoid fetching from s3
                    s3_filenames = self.s3._get_s3_list(cache_ok=True)
                    if zipped_name not in s3_filenames \
                            and unzipped_name not in s3_filenames:
                        # list might be stale; refresh
                        s3_filenames = self.s3._get_s3_list(cache_ok=False)
                    if self.s3.decompress_on_read(self.file_name):
                        assert self.file_name == unzipped_name
                        s3_name = zipped_name
                        if s3_name in s3_filenames:
                            post_process = 'unzip'
                        else:
                            # the zipped file isn't on S3; maybe this is a
                            # legacy file that was never zipped; no post
                            # processing needed
                            s3_name = self.file_name
                            # don't assert, so we get IOError below,
                            # as expected by clients
                            #assert s3_name in s3_filenames
                    else:
                        # cache copy should be compressed, and read in as-is
                        assert self.file_name == zipped_name
                        s3_name = zipped_name
                        if s3_name not in s3_filenames:
                            # compressed copy not on S3; assume it's an older
                            # uncompressed copy, and zip after retrieval
                            s3_name = unzipped_name
                            # don't assert, so we get IOError below,
                            # as expected by clients
                            #assert s3_name in s3_filenames
                            post_process = 'zip'
                else:
                    # read in exactly as on S3
                    s3_name = self.file_name
                try:
                    self.s3.bucket.get_file(s3_name)
                except Exception as ex:
                    # The original code here looked for S3ResponseError;
                    # in testing I also saw ClientError; just catch everything
                    # and look for the 404 as some confirmation that it's
                    # an S3 error.
                    # Make sure the target file doesn't exist after a fetch
                    # failure, as that will screw up caching. This used to
                    # be an issue prior to boto3; it may no longer be a
                    # problem, but leaving the code here shouldn't hurt.
                    import os
                    try:
                        os.remove(self.path())
                    except FileNotFoundError:
                        pass
                    if '404' in str(ex):
                        # the file was not found on S3; there may be code
                        # up the stack that's prepared to handle an IOError,
                        # which is roughly equivalent, so let's throw one of
                        # those instead
                        raise IOError(s3_name+' not found on S3')
                    raise
                if post_process:
                    cmd = {'zip':'gzip','unzip':'gunzip'}[post_process]
                    import subprocess
                    subprocess.check_call([cmd, self.s3.cache_path+s3_name])
        finally:
            lock.release()

# shared helper functions for versions.py files
def attribute_file_name(file_class,version,flavor='full'):
    from dtk.files import VersionedFileName
    vfn = VersionedFileName(file_class=file_class)
    vfn.prefix = file_class
    vfn.flavor = flavor
    vfn.version = int(version[1:])
    vfn.role = 'attributes'
    vfn.format = 'tsv'
    return vfn.to_string()

def attribute_file_path(file_class,version,flavor='full'):
    afn = attribute_file_name(file_class,version,flavor)
    s3b = S3Bucket(file_class)
    import os
    return os.path.join(s3b.cache_path,afn)

