from __future__ import print_function

import os

class SqliteSv(object):
    """TSV-like data stored in an SQLite database."""
    # https://www.sqlite.org/limits.html, SQLITE_MAX_VARIABLE_NUMBER - 1
    MAX_VARS=32765
    def __init__(self, fn):
        import sqlite3
        self.conn = sqlite3.connect(fn, timeout=60.0)
        # Makes sqlite use mmap up to file sizes of (very large).
        self.conn.execute("PRAGMA mmap_size=1000000000;")
        self.conn.row_factory = sqlite3.Row

    def get_namedtuple(self, class_name):
        """Returns a namedtuple whose fields are the columns of this dataset."""
        from collections import namedtuple
        return namedtuple(class_name, self.get_header())

    def get_header(self):
        cursor = self.conn.cursor()
        # This won't actually execute because we don't fetch the results.
        cursor.execute("select * from data WHERE 1=0")
        cols = [x[0] for x in cursor.description]
        return cols

    def get_records(self, columns=None, keep_header=False, insertion_ordered=False,**kwargs):
        """Fetch data from this file.

        If you specify no parameters, it will yield every column of every row.

        Specify columns as a list of column names to limit the set of columns.

        To filter the rows returned, add parameters in the form of:
            filter__<colname>__<op>=<val>, similar to Django querysets.

        e.g.
            get_records(columns=['uniprot1', 'direction'],
                        filter__uniprot1__eq='Q1234',
                        filter__evidence__gte=0.9)
        """

        if keep_header:
            yield self.get_header()
        if columns is None:
            selector = '*'
        else:
            selector = ','.join(columns)

        if kwargs.pop('unique', False):
            selector = " DISTINCT " + selector

        op_map = {
                'lte': '<=',
                'lt': '<',
                'eq': '=',
                'gt': '>',
                'gte': '>=',
                }
        parm_values = []
        where_parts = []

        for argname, argvalue in kwargs.items():
            if argname.startswith('filter__'):
                parts = argname.split('__')
                op = parts[-1]
                name = '`' + '__'.join(parts[1:-1]) + '`'
                if op == 'in':
                    placeholders = ','.join(['?'] * len(argvalue))
                    where_parts.append(name + " IN (%s)" % placeholders)
                    parm_values.extend(argvalue)
                else:
                    op = op_map[op]
                    where_parts.append(name + op + "?")
                    parm_values.append(argvalue)
            else:
                raise ValueError("Unknown kwarg %s" % argname)


        where = ' AND '.join(where_parts)
        if where:
            where = "WHERE " + where
        
        if insertion_ordered:
            order_by = "ORDER BY rowid"
        else:
            order_by = ""
        stmt = "SELECT {selector} FROM data {where} {order_by}".format(
                    selector=selector,
                    where=where,
                    order_by=order_by,
                    )
        # print("Executing", stmt, parm_values)
        yield from self.conn.execute(stmt, parm_values)


    @staticmethod
    def write_from_tsv(out_fn, in_tsv_fn, types, header=None, index=None):
        from dtk.files import get_file_records
        data = list(get_file_records(in_tsv_fn))
        SqliteSv.write_from_data(out_fn, data, types, header, index)

    @staticmethod
    def write_from_data(out_fn, data, types, header=None, index=None):
        """Writes out a sqlite file.

        data: List of data rows (tuples or lists)
        types: List of types for each column (str, float, int)
        header: List of column names (if None, taken from data[0]).
        index: List of column names to index (if None, all columns).
        """
        # Remove any previous temporary file.
        try:
            os.unlink(out_fn + ".tmp")
        except OSError:
            pass
        typemap = {
                str: 'text',
                float: 'real',
                int: 'integer',
                }

        sqltypes = [typemap[type] for type in types]

        if header is None:
            assert len(data) >= 1, f"No header? {data}"
            header = data[0]
            data = data[1:]
        
        if index is not None:
            for to_index in index:
                assert to_index in header, f'Missing column to index "{to_index}" in {header}'
        
        for i, row in enumerate(data):
            assert len(row) == len(header), f'Row {i} had {len(row)} columns, vs {len(header)}: {row}'

        import sqlite3
        conn = sqlite3.connect(out_fn + ".tmp")
        with conn:
            assert len(header) == len(sqltypes), "File header didn't match types"

            cmd = ','.join([('`%s` %s' % (x, t)) for x, t in zip(header, sqltypes)])

            cmd = "CREATE TABLE data (%s)" % cmd
            conn.execute(cmd)

            qs = ', '.join(['?'] * len(header))

            conn.executemany("INSERT INTO data VALUES(%s)" % qs, data)

            for col in header:
                if index is not None and col not in index:
                    continue
                cmd = "CREATE INDEX `%s_idx` on data(`%s`)" % (col, col)
                conn.execute(cmd)
        conn.close()
        os.rename(out_fn + ".tmp", out_fn)
    def insert(self, records):
        # If we're inserting on the fly, we probably want WAL mode for better
        # concurrency.
        self.conn.execute("PRAGMA journal_mode=WAL;")
        header = self.get_header()
        qs = ', '.join(['?'] * len(header))
        with self.conn:
            self.conn.executemany("INSERT INTO data VALUES(%s)" % qs, records)
