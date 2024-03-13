



from builtins import range
def setup_db(tmp_path, data, return_fn=False):
    from dtk.tsv_alt import SqliteSv
    fn = str(tmp_path / 'data.sqlsv')
    SqliteSv.write_from_data(fn, data, [str, float, str])
    if return_fn:
        return SqliteSv(fn), fn
    else:
        return SqliteSv(fn)

def test_write_from_tsv(tmp_path):
    from dtk.tsv_alt import SqliteSv
    data = [
        ('uniprot', 'score', 'note'),
        ('prot1', -2, ''),
        ('prot2', 0, ''),
        ]
    tsv_fn = str(tmp_path / 'data.tsv')
    with open(tsv_fn, 'w') as f:
        f.write('\n'.join(('\t'.join((str(x) for x in row)) for row in data)))

    sql_fn = str(tmp_path / 'data.sqlsv')
    SqliteSv.write_from_tsv(sql_fn, tsv_fn, [str, float, str])
    db = SqliteSv(sql_fn)

    for orig_row, sv_row in zip(data[1:], db.get_records()):
        assert len(orig_row) == 3
        assert len(sv_row) == 3
        for i in range(len(sv_row)):
            assert orig_row[i] == sv_row[i]

        assert orig_row[0] == sv_row['uniprot']
        assert orig_row[1] == sv_row['score']
        assert orig_row[2] == sv_row['note']


def test_sqlitesv_all_cols(tmp_path):
    data = [
        ('uniprot', 'score', 'note'),
        ('prot1', -2, ''),
        ('prot2', 0, ''),
        ('prot3', 0.229, ''),
        ('prot4', 0.230, ''),
        ('prot0', 0.1, 'Out of order'),
        ]
    db = setup_db(tmp_path, data)


    from collections import namedtuple
    assert db.get_namedtuple('Test')._fields == ('uniprot', 'score', 'note')

    for orig_row, sv_row in zip(data[1:], db.get_records(insertion_ordered=True)):
        assert len(orig_row) == 3
        assert len(sv_row) == 3
        for i in range(len(sv_row)):
            assert orig_row[i] == sv_row[i]

        assert orig_row[0] == sv_row['uniprot']
        assert orig_row[1] == sv_row['score']
        assert orig_row[2] == sv_row['note']


def test_sqlitesv_filter(tmp_path):
    data = [
        ('uniprot', 'score', 'note'),
        ('prot1', -2, ''),
        ('prot2', 0, ''),
        ('prot3', 0.229, ''),
        ('prot4', 0.230, ''),
        ('prot0', 0.1, 'Out of order'),
        ]
    db = setup_db(tmp_path, data)

    for expected, row in zip(data[1:], db.get_records(columns=['score'], insertion_ordered=True)):
        assert len(row) == 1
        assert row[0] == expected[1]
        assert row['score'] == expected[1]

    
    data = list(db.get_records(filter__uniprot__eq='prot0'))
    assert len(data) == 1
    assert tuple(data[0]) == ('prot0', 0.1, 'Out of order')

    data = list(db.get_records(filter__uniprot__eq='bad'))
    assert len(data) == 0

    data = list(db.get_records(filter__uniprot__gt='prot2'))
    assert len(data) == 2
    assert data[0][0] == 'prot3'
    assert data[1][0] == 'prot4'

    data = list(db.get_records(filter__score__lte=0.1))
    assert len(data) == 3
    assert data[0][0] == 'prot1'
    assert data[1][0] == 'prot2'
    assert data[2][0] == 'prot0'

    data = list(db.get_records(filter__uniprot__in=['prot3', 'prot2']))
    assert len(data) == 2
    assert data[0][0] == 'prot2'
    assert data[1][0] == 'prot3'


def test_sqlitesv_get_file_records(tmp_path):
    data = [
        ('uniprot', 'score', 'note'),
        ('prot1', -2, ''),
        ('prot2', 0, ''),
        ('prot3', 0.229, ''),
        ('prot4', 0.230, ''),
        ('prot0', 0.1, 'Out of order'),
        ]
    db, db_fn = setup_db(tmp_path, data, return_fn=True)
    from dtk.files import get_file_records
    recs = list(get_file_records(db_fn, keep_header=True))
    assert len(recs) == len(data)
    for rec, datum in zip(recs, data):
        assert list(rec) == list(datum)

    recs = [list(x) for x in get_file_records(db_fn, select=(['prot1'], 0))]
    assert recs == [['prot1', -2, '']]

    recs = [tuple(x) for x in get_file_records(db_fn, select=(['prot1', 'prot2', 'prot3'], 0))]
    assert recs == [data[1], data[2], data[3]]

    
    from mock import patch
    with patch('dtk.tsv_alt.SqliteSv.MAX_VARS', 1):
        from dtk.tsv_alt import SqliteSv
        assert SqliteSv.MAX_VARS == 1
        recs = [tuple(x) for x in get_file_records(db_fn, select=(['prot1', 'prot2', 'prot3'], 0))]
        assert recs == [data[1], data[2], data[3]]

        recs = [tuple(x) for x in get_file_records(db_fn, keep_header=True, select=(['prot1', 'prot2', 'prot3'], 0))]
        assert recs == [data[0], data[1], data[2], data[3]]





