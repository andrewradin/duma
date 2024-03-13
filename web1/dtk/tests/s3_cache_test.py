import pytest

def test_versioned_file_name():
    from dtk.files import VersionedFileName
    # test Meta defaults
    vfn = VersionedFileName(
            meta=VersionedFileName.Meta(),
            name='one_part_flavor.v10.tsv',
            )
    assert vfn.prefix == ''
    assert vfn.flavor == 'one_part_flavor'
    assert vfn.version == 10
    assert vfn.role == ''
    assert vfn.format == 'tsv'

    # test customized filename format
    vfn = VersionedFileName(
            meta=VersionedFileName.Meta(
                    prefix = 'my_prefix',
                    flavor_len = 3,
                    roles = ['main_role','aux_role1','aux_role2']
                    ),
            name='my_prefix.three.part.flavor.v1.main_role.tsv',
            )
    assert vfn.prefix == 'my_prefix'
    assert vfn.flavor == 'three.part.flavor'
    assert vfn.version == 1
    assert vfn.role == 'main_role'
    assert vfn.format == 'tsv'

    # test file name construction
    vfn = VersionedFileName(meta=VersionedFileName.Meta())
    vfn.flavor='my_flavor'
    vfn.version=2
    vfn.format='tsv'
    assert vfn.to_string() == 'my_flavor.v2.tsv'

    # test multi-part file format designators
    vfn = VersionedFileName(meta=VersionedFileName.Meta())
    vfn.flavor='my_flavor'
    vfn.version=3
    vfn.format='tsv.gz'
    assert vfn.to_string() == 'my_flavor.v3.tsv.gz'
    vfn = VersionedFileName(
            meta=VersionedFileName.Meta(),
            name='another_flavor.v44.tsv.gz',
            )
    assert vfn.prefix == ''
    assert vfn.flavor == 'another_flavor'
    assert vfn.version == 44
    assert vfn.role == ''
    assert vfn.format == 'tsv.gz'

    # test file_class format registration
    vfn = VersionedFileName(file_class='test')

    # test choice list construction
    meta=VersionedFileName.Meta(
                prefix = 'my_prefix',
                flavor_len = 1,
                roles = ['main','aux']
                )
    paths=[
            'some_path/my_prefix.flavor1.v1.main.tsv',
            'some_path/my_prefix.flavor1.v1.aux.tsv',
            'some_path/my_prefix.flavor1.v2.main.tsv',
            'some_path/my_prefix.flavor1.v2.aux.tsv',
            'some_path/my_prefix.flavor1.v3.main.tsv',
            'some_path/my_prefix.flavor1.v3.aux.tsv',
            'some_path/my_prefix.flavor2.v1.main.tsv',
            'some_path/my_prefix.flavor2.v1.aux.tsv',
            'some_path/my_prefix.flavor2.v2.main.tsv',
            'some_path/my_prefix.flavor2.v2.aux.tsv',
            'some_path/my_prefix.flavor2.v3.main.tsv',
            'some_path/my_prefix.flavor2.v3.aux.tsv',
            'some_path/my_prefix.flavor2.v4.main.tsv',
            'some_path/my_prefix.flavor2.v4.aux.tsv',
            ]
    choices = VersionedFileName.get_choices(meta=meta,paths=paths)
    assert choices == [(x,x) for x in [
            'flavor1.v3',
            'flavor2.v4',
            'flavor1.v2',
            'flavor1.v1',
            'flavor2.v3',
            'flavor2.v2',
            'flavor2.v1',
            ]]
    files = VersionedFileName.get_matching_files(
            meta=meta,
            choice='flavor2.v2',
            paths=paths,
            )
    assert files == [
            'some_path/my_prefix.flavor2.v2.main.tsv',
            'some_path/my_prefix.flavor2.v2.aux.tsv',
            ]
    files = VersionedFileName.get_matching_files(
            meta=meta,
            choice='flavor2.v2',
            role='main',
            paths=paths,
            )
    assert files == [
            'some_path/my_prefix.flavor2.v2.main.tsv',
            ]
    path = VersionedFileName.get_matching_path(
            meta=meta,
            choice='flavor2.v2',
            role='main',
            paths=paths,
            )
    assert path == 'some_path/my_prefix.flavor2.v2.main.tsv'
    with pytest.raises(RuntimeError):
        VersionedFileName.get_matching_path(
                meta=meta,
                choice='flavor2.v2',
                role='main',
                paths=[],
                )
    with pytest.raises(RuntimeError):
        VersionedFileName.get_matching_path(
                meta=meta,
                choice='flavor2.v666',
                role='main',
                paths=paths,
                )
    with pytest.raises(RuntimeError):
        VersionedFileName.get_matching_path(
                meta=meta,
                choice='flavor2.v2',
                paths=paths,
                )
    with pytest.raises(RuntimeError):
        VersionedFileName.get_matching_path(
                meta=meta,
                choice='flavor2.v2',
                role='not_really_a_role',
                paths=paths,
                )

def make_versioned_s3_test_objects():
    from dtk.s3_cache import S3Bucket
    file_class = 'test'
    s3b = S3Bucket(file_class)
    from dtk.files import VersionedFileName
    import os
    import subprocess
    for flavor in ('flavor1','flavor2'):
        for role in ('both','comp_only','neither'):
            for version in range(1,4):
                vfn=VersionedFileName(file_class=file_class)
                vfn.flavor = flavor
                vfn.version = version
                vfn.role = role
                vfn.format = 'txt'
                fn = vfn.to_string()
                path = os.path.join(s3b.cache_path,fn)
                with open(path,'w') as fh:
                    fh.write('%s version %d\n'%(flavor,version))
                if role in ('both','comp_only'):
                    if flavor == 'flavor1' and version == 1:
                        # force some files to always be uncompressed on S3,
                        # so we can verify that AutoCompress retrieves
                        # them properly
                        pass
                    else:
                        subprocess.check_call(["gzip",path])
                        path += '.gz'
                        fn += '.gz'
                s3b.bucket.put_file(fn)
                os.remove(path)

def test_auto_compress_meta():
    from dtk.s3_cache import S3Bucket
    file_class = 'test'
    s3b = S3Bucket(file_class)
    assert s3b.compress_on_write('test.flavor1.v1.both.txt')
    assert s3b.compress_on_write('test.flavor1.v1.comp_only.txt')
    assert not s3b.compress_on_write('test.flavor1.v1.neither.txt')
    assert s3b.decompress_on_read('test.flavor1.v1.both.txt')
    assert not s3b.decompress_on_read('test.flavor1.v1.comp_only.txt')
    assert not s3b.decompress_on_read('test.flavor1.v1.neither.txt')
    from dtk.files import VersionedFileName
    meta = VersionedFileName.Meta(prefix='dummy')
    assert meta.compress_on_write('dummy.flavor.v1.txt')
    assert meta.decompress_on_read('dummy.flavor.v1.txt')

def test_vbucket():
    from dtk.s3_cache import S3Bucket,S3File

    with pytest.raises(KeyError):
        s3b = S3Bucket('no_such_directory')
    s3b = S3Bucket('test')
    # clean out any junk in cache directory
    import os
    for fn in os.listdir(s3b.cache_path):
        os.remove(os.path.join(s3b.cache_path,fn))
    # verify list matches expected S3 files
    l = s3b.list()
    expect = [
            x+('.gz' if 'comp_only' in x else '')
            for x in [
                    f'test.{flavor}.{version}.{role}.txt'
                    for flavor in ('flavor1','flavor2')
                    for version in ('v1','v2','v3')
                    for role in ('both','comp_only','neither')
                    ]
            ]
    assert l==expect
    # verify fetch
    from dtk.files import remove_if_present
    testfile_list = (
            'test.flavor1.v1.both.txt', # uncompressed on S3
            'test.flavor1.v2.both.txt', # compressed on S3
            'test.flavor1.v1.comp_only.txt.gz', # uncompressed on S3
            'test.flavor1.v2.comp_only.txt.gz', # compressed on S3
            )
    import os
    for filename in testfile_list:
        assert filename in l
        s3f = S3File(s3b,filename)
        remove_if_present(s3f.path())
        assert not os.path.exists(s3f.path())
        s3f.fetch()
        assert os.path.exists(s3f.path())
    # verify versioned file shortcut
    s3f = S3File.get_versioned('test','flavor1.v3',role='both')
    dirname,filename = os.path.split(s3f.path())
    assert filename == 'test.flavor1.v3.both.txt'
    # verify plausible but non-existent file raises error
    fn = 'test.flavor4.v3.both.txt'
    s3f = S3File(s3b,fn)
    remove_if_present(s3f.path())
    with pytest.raises(IOError) as excinfo:
        s3f.fetch()
    assert not os.path.exists(s3f.path())
    # ...but creating file makes fetch happy
    with open(s3f.path(),'w') as fh:
        fh.write('local-only file')
    s3f.fetch()
    # ...and new filename shows up in list
    assert fn in s3b.list()
    # verify stale list cache doesn't break fetch
    for filename in testfile_list:
        s3f = S3File(s3b,filename)
        remove_if_present(s3f.path())
        open(os.path.join(s3b.cache_path,s3b.list_cache_fn),'w')
        assert not os.path.exists(s3f.path())
        s3f.fetch()
        assert os.path.exists(s3f.path())

