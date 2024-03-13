def yield_assembled_recs(line_src,quote='"',delim='|'):
    '''Return arrays of fields for each input record.

    AACT records are pipe-delimited, but fields may be quoted, and within
    a quoted field there may be pipes or newlines. Given an iterator over
    lines (with newlines removed), this function reformats them into
    arrays of fields. Quotes are removed, and quoted pipes and newlines
    are converted to spaces.

    For efficiency, this is done using a simple split on lines that don't
    contain a quote, and only resorts to detailed delimiter searching on
    quoted lines.
    '''
    def partial_quote(f):
        return f.startswith(quote) and not f.endswith(quote)

    def clean_field(f):
        f = f.replace(delim,' ')
        f = f.replace(quote,'')
        return f

    for line in line_src:
        if not quote in line:
            yield line.split(delim)
            continue
        # this line contains a quote; process it field-by-field
        result = []
        field_src = iter(line.split(delim))
        while True:
            # this is basically 'for field in field_src', but it allows
            # field_src to be changed lower in the loop, so we can advance
            # over continuation lines
            try:
                field = next(field_src)
            except StopIteration:
                break
            result.append(field)
            while partial_quote(result[-1]):
                # keep concatenating fields until we find the ending quote;
                # advance to the next line if needed; replace the field
                # separator with a space
                try:
                    field = next(field_src)
                except StopIteration:
                    field_src = iter(next(line_src).split(delim))
                    field = next(field_src)
                result[-1] += f' {field}'
        # yield back the assembled record, after cleaning each field
        yield [clean_field(x) for x in result]

def line_src(fn):
    fd = open(fn)
    for line in fd:
        yield line.rstrip('\n')

def aact_file_records(fn):
    for rec in yield_assembled_recs(line_src(fn)):
        yield rec

def get_study_condition_lists():
    result = {}
    fn = 'browse_conditions.txt'
    header = None
    for fields in aact_file_records(fn):
        if not header:
            header = [x.upper() for x in fields]
            assert header[1:3] == ['NCT_ID','MESH_TERM']
            continue
        result.setdefault(fields[1],[]).append(fields[2])
    return result

