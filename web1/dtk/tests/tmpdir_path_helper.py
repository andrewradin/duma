import pytest
from dtk.lts import LtsRepo
import os

def s3cachedir(fn):
    def decorator(func):
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            print("Overriding S3 cache to", fn)
            os.environ['S3_CACHE_DIR'] = fn
            out = func(*args, **kwargs)
            print("Removing S3 cache override")
            del os.environ['S3_CACHE_DIR']
            return out
        return wrapper
    return decorator

    

@pytest.fixture(autouse=True)
def tmpdir_path_helper(request, tmpdir):
    from path_helper import PathHelper, make_directory
    tmpdir = tmpdir.join('test_home')
    tmpdir.mkdir()

    s3env_change = False
    if hasattr(request, 'param'):
        if 'S3_CACHE_DIR' in request.param:
            fn = request.param['S3_CACHE_DIR']
            print("Overriding S3 cache to", fn)
            os.environ['S3_CACHE_DIR'] = fn
            s3env_change = True

    # We want to reference everything into the tmpdir unless it is looking
    # for something in the repo.
    orig_home = PathHelper.home
    PathHelper.home = str(tmpdir) + "/"
    repos_root = PathHelper.repos_root
    orig_attrs = {}

    # PathHelper probably got loaded before our test set S3_CACHE_DIR, so
    # make sure we respect it now.
    from path_helper import PathHelper
    orig_s3_cache_root = PathHelper.s3_cache_root
    s3_cache_root = os.environ.get("S3_CACHE_DIR", PathHelper.home+ 'ws/')
    print("Setting us up with cache root", s3_cache_root)

    for attr_name, attr_val in PathHelper.__dict__.items():
        if not isinstance(attr_val, str):
            continue

        if attr_name.startswith('R_'):
            # These are paths to our R libs and binaries, don't change them.
            continue

        if attr_val.startswith(orig_home) and not attr_val.startswith(repos_root):
            orig_attrs[attr_name] = attr_val
            if attr_val in PathHelper.s3_cache_dirs or attr_name == 's3_cache_root':
                new_val = attr_val.replace(orig_s3_cache_root, s3_cache_root)
            else:
                new_val = attr_val.replace(orig_home, PathHelper.home)
            setattr(PathHelper, attr_name, new_val)

    for d in PathHelper.create_dirs:
        if d in PathHelper.s3_cache_dirs:
            d = d.replace(orig_s3_cache_root, s3_cache_root)
        else:
            d = d.replace(orig_home, PathHelper.home)
        make_directory(d)

    from .static_reset import static_reset

    # Anything using a tmpdir path helper probably needs static state reset,
    # as lots of static state bakes in the PathHelper directories.
    static_reset()


    yield PathHelper

    if s3env_change:
        print("Restoring S3 cache")
        del os.environ['S3_CACHE_DIR']

    # Restore pathhelper to what it was before we started.
    PathHelper.home = orig_home
    for attr_name, orig_val in orig_attrs.items():
        setattr(PathHelper, attr_name, orig_val)

    # We do this before and after the test, it is cheap.
    static_reset()
