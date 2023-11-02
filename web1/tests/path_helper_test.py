import pytest

def test_legacy_s3_cache_paths():
    from path_helper import PathHelper
    root = PathHelper.s3_cache_root
    file_classes = PathHelper.legacy_s3_file_classes
    for file_class in file_classes:
        ph_dir = getattr(PathHelper,file_class)
        assert ph_dir == root+file_class+'/'
        assert PathHelper.s3_cache_path(file_class) == ph_dir
        assert ph_dir in PathHelper.create_dirs

def test_vbucket_s3_cache_paths():
    from path_helper import PathHelper
    root = PathHelper.s3_cache_root
    file_classes = [x[0] for x in PathHelper.vbucket_specs]
    assert 'test' in file_classes
    for file_class in file_classes:
        ph_dir = root+file_class+'/'
        assert ph_dir in PathHelper.s3_cache_dirs
