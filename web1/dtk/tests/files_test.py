

import pytest

def test_get_file_lines(tmpdir):
    fn = str(tmpdir/'testfile.txt.gz')
    import gzip
    with gzip.open(fn, 'wb') as f:
        f.write(b'hello test')
    
    from dtk.files import get_file_lines
    assert list(get_file_lines(fn, progress=True)) == ['hello test']
    assert list(get_file_lines(fn)) == ['hello test']
    assert list(get_file_lines(fn, keep_header=False)) == []

    # Now without gz
    fn = str(tmpdir/'testfile.txt')
    with open(fn, 'wb') as f:
        f.write(b'hello test')
    
    from dtk.files import get_file_lines
    assert list(get_file_lines(fn, progress=True)) == ['hello test']
    assert list(get_file_lines(fn)) == ['hello test']
    assert list(get_file_lines(fn, keep_header=False)) == []



    # multiline
    fn = str(tmpdir/'testfile.txt.gz')
    import gzip
    with gzip.open(fn, 'wb') as f:
        f.write(b'hello test\nb multiline\na file')
    
    from dtk.files import get_file_lines
    assert list(get_file_lines(fn, progress=True)) == ['hello test\n', 'b multiline\n', 'a file']
    assert list(get_file_lines(fn)) == ['hello test\n', 'b multiline\n', 'a file']
    assert list(get_file_lines(fn, keep_header=False)) == ['b multiline\n', 'a file']

    # Sort seems to add back in the trailing \n to the final line.
    assert list(get_file_lines(fn, sort=True)) == ['hello test\n', 'a file\n', 'b multiline\n']
    assert list(get_file_lines(fn, sort=True, grep=['e'])) == ['hello test\n', 'a file\n', 'b multiline\n']
    assert list(get_file_lines(fn, sort=True, grep=['m'])) == ['hello test\n', 'b multiline\n']

