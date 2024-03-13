import six
def codify(string):
    if six.PY2:
        string = string.encode('ascii','ignore')
    string = string.lower()
    import re
    string = re.sub('[^a-z0-9]+','_',string)
    return string

################################################################################
# Class for creating HTML tables
# - defines a template-friendly interface for extracting HTML; can be
#   rendered with the _table.html template, or custom template code
# - Column objects simplify specifying how column data is extracted from
#   an abstract row, and how it is formatted.
# - provides hooks for sorting when rendering column headers
# - a "row source" object provides an abstract sequence of rows; this
#   can be a django queryset, or any type of list
################################################################################
class Table(object):
    class Column(object):
        def __init__(self,label,**kwargs):
            self.label = label
            # set up defaults here
            self.code = None
            self.sort = None
            self.extract = None
            self.idx = None
            self.cell_fmt = None
            self.cell_html_hook = None
            self.cell_attr_hook = None
            self.cell_attrs = {}
            # cycle through supplied args, overriding defaults
            for k,v in six.iteritems(kwargs):
                # here, handle any 'special' args that are shorthand
                # for combinations of more basic underlying attributes
                if k=='decimal':
                    if v:
                        self.cell_attrs = dict(style="text-align:right")
                    continue
                # anything left should match an existing attribute from above
                if not hasattr(self,k):
                    raise TypeError("unexpected keyword argument '%s'"%k)
                setattr(self,k,v)
            # supply any 'dynamic' defaults for things not overridden
            if not self.code:
                self.code = codify(self.label)
        def header_html(self,sort_handler,url_builder):
            from dtk.html import tag_wrap
            label = self.label
            if sort_handler and self.sort:
                label = sort_handler.header_html(
                        self.code,
                        self.sort,
                        label,
                        url_builder,
                        )
            return tag_wrap('th',label,self.cell_attrs)
        def cell_value(self,row):
            if self.extract:
                return self.extract(row)
            if self.idx is not None:
                try:
                    return row[self.idx]
                except KeyError:
                    # we only handle KeyError here, and not IndexError,
                    # because we expect that if the row is a dict(), it
                    # might be sparcely populated, but if the row is
                    # an array, it should be complete
                    return None
            try:
                return getattr(row,self.code)
            except AttributeError:
                return '???'
        def cell_data(self,row):
            data = self.cell_value(row)
            if self.cell_fmt is not None:
                # this is for simple string formatting
                data = self.cell_fmt(data)
            if self.cell_html_hook:
                # this is for complex data rewrites
                data = self.cell_html_hook(data,row,self)
            if data is None:
                data = ''
            return data

        def cell_html(self,row):
            data = self.cell_data(row)
            if self.cell_attr_hook:
                attrs = self.cell_attr_hook(data,row,self)
            else:
                attrs = self.cell_attrs
            from dtk.html import tag_wrap
            return tag_wrap('td',data,attrs)
    def __init__(self,row_source,columns,url_builder=None,sort_handler=None):
        '''Initialize a Table object.

        row_source is an iterable that returns rows.  A row can be anything
        from which a value can be extracted by a cell_value method.  By
        default, it is expected to have an attribute whose name matches
        col.code, but arrays and hashes can be handled by setting col.idx,
        and any other situation can be handled by setting col.extract to
        a callable.  row_source is expected to already be aware of any
        sorting, filtering, or paging.

        columns is a list of Column-like objects defining what should be
        displayed in the header, and for each row.

        url_builder is a function that builds a URL back to the current
        page, with GET parameters modified as defined by **kwargs.

        sort_handler is a SortHandler-like object that helps in rendering
        column headings to allow users to see the current sort, and change
        sort options by clicking. As noted above, the Table object relies
        on the row_source to do any actual sorting.
        '''
        self._row_source = row_source
        self._columns = columns
        self._url_builder = url_builder
        self._sort_handler = sort_handler
    def headers(self):
        return [
                column.header_html(self._sort_handler,self._url_builder)
                for column in self._columns
                ]
    def headers_data(self):
        return [{'title': col.label} for col in self._columns]
    def remove_columns(self, labels):
        self._columns = [x for x in self._columns if x.label not in labels]
    def rows(self):
        class RowWrapper(list):
            def __init__(self,src,data):
                self.source_row = src
                super(RowWrapper,self).__init__(data)
        return [
                RowWrapper(
                        row,
                        [column.cell_html(row) for column in self._columns],
                        )
                for row in self._row_source
                ]

    def rows_data(self):
        return [[column.cell_data(row) for column in self._columns]
                   for row in self._row_source]
    
    @classmethod
    def from_dataframe(cls, df):
        col_labels = df.columns.tolist()
        columns = [Table.Column(x, idx=i) for i, x in enumerate(col_labels)]
        data = df.to_numpy(dtype=str)
        return Table(data, columns)

################################################################################
# Class for managing paging
# This does page limit calculation, and renders page controls.  The actual
# selection of rows is the responsibility of the row_source object passed
# to the Table class.
################################################################################
class Pager:
    def __init__(self,url_builder,total_rows,page_size,page):
        self.url=url_builder
        self.page_size = page_size
        total_pages = (total_rows + self.page_size - 1)//self.page_size
        if total_pages < 1:
            total_pages = 1
        self.last_page = total_pages
        self.total_rows = total_rows
        self.set_page(page)
    def set_page(self,page):
        self.page_num = max(1,min(page,self.last_page))
        self.prev_page = max(1,self.page_num - 1)
        self.next_page = min(self.last_page,self.page_num + 1)
        self.page_start = self.page_size * (self.page_num-1)
        self.page_end = min(self.total_rows, self.page_size * self.page_num)
    def page_of_idx(self,idx):
        return 1 + idx//self.page_size
    def html(self):
        from dtk.html import tag_wrap,link,glyph_icon
        from django.utils.html import format_html
        return tag_wrap(
                'span',
                format_html(
                        '{} {} Page {} of {} {} {}',
                        link(
                                glyph_icon('fast-backward'),
                                self.url(page=1),
                                ),
                        link(
                                glyph_icon('step-backward'),
                                self.url(page=self.prev_page),
                                ),
                        self.page_num,
                        self.last_page,
                        link(
                                glyph_icon('step-forward'),
                                self.url(page=self.next_page),
                                ),
                        link(
                                glyph_icon('fast-forward'),
                                self.url(page=self.last_page),
                                ),
                        ),
                {'class':'pull-right'}
                )

################################################################################
# class for managing sorting
# Again, parallel to Pager above, this handles the UI, and relies on the
# row_source to implement the actual sorting.
#
# Quick checklist for adding sorting to an existing table (assuming you're in
# a DumaView):
# - pass url_builder=self.here_url to the Table ctor
# - add a sort specifier qparm (e.g. add
#   'sort':(SortHandler,code_of_default_sort),
#   to GET_parms)
# - pass the qparm to the Table ctor (e.g. sort_handler=self.sort)
# - add sort='l2h' (or 'h2l') parms to the ctor of each sortable column
# - write bespoke code to query the sort handler and apply the appropriate
#   sort to the row source
# Note:
# - after the first 4 steps, the web UI for the sort will appear to function,
#   but no actual sorting will take place
# - if you have multiple sortable things on the same page, each one needs a
#   different qparm name
################################################################################
class SortHandler:
    sort_parm='sort'
    page_parm='page'
    reverse_mode='h2l'
    def __init__(self,spec=''):
        self.minus = spec[:1] == '-'
        if self.minus:
            spec = spec[1:]
        self.colspec = spec
    def to_string(self):
        minus = '-' if self.minus else ''
        return minus+self.colspec
    def header_html(self,colspec,mode,label,url_builder):
        from dtk.html import glyph_icon,link
        from django.utils.html import format_html
        if colspec == self.colspec:
            if not self.minus:
                colspec = '-'+colspec
                icon = glyph_icon("sort-by-attributes")
            else:
                icon = glyph_icon("sort-by-attributes-alt")
        else:
            icon = ""
            if mode == self.reverse_mode:
                colspec = '-'+colspec
        return format_html('{}{}',
                        link(
                                label,
                                url_builder(**{
                                            self.sort_parm:colspec,
                                            self.page_parm:None,
                                            }),
                                ),
                        icon,
                        )

################################################################################
# Support classes for building a key-based row source
# This is a case where data is assembled from multiple independent data
# sources, where each data source represents a column, and data is grouped
# into rows based on a common key.
#
# KeyFilter - combines multiple filters, and supplies an 'ok' function
#   reporting whether a key passed the filters; also provides a consolidated,
#   human-readable filter description
# ScoreFilter - one possible input to KeyFilter; generates filters based on
#   numerical scores associated with a key; provides a form for editing
#   filters, and supports encoding and decoding filters in URL query strings.
#   expects an ordering() function that can retrieve ordered data for a
#   specified column
# IdRowSource - assembles data to be passed to Table class.  Relies on an
#   ordering() function as above to retrieve the sort column, then applies
#   a KeyFilter.  A get_page() method assembles the actual data page to
#   be rendered, using a get_value() function to retrieve cell data.  Makes
#   length and rank data available.  The get_page method is separate from the
#   filtering in the ctor so that the total row count from the ctor can be
#   used to build a pager object, which is then used by get_page
################################################################################
class KeyFilter:
    '''Accumulates multiple key-based filters.
    '''
    from collections import namedtuple
    FilterDetail=namedtuple('FilterDetail','index colname value')
    def __init__(self,copy_from=None):
        if copy_from:
            self._excluding = copy_from._excluding
            self._fs = set(copy_from._fs)
            self._parts = dict(copy_from._parts)
        else:
            self._excluding = True
            self._fs = set()
            self._parts = {}
    def excluding(self):
        return self._excluding
    def merge(self,col_key,col_label,col_spec,keys,exclude=False):
        self._parts[col_key] = self.FilterDetail(col_key,col_label,col_spec)
        if self._excluding:
            if exclude:
                self._fs |= set(keys)
            else:
                self._fs = set(keys) - self._fs
                self._excluding = False
        else:
            if exclude:
                self._fs -= set(keys)
            else:
                self._fs &= set(keys)
    def ok(self,key):
        if self._excluding:
            return key not in self._fs
        return key in self._fs
    def count(self):
        if self._excluding:
            return -len(self._fs)
        return len(self._fs)
    def get_description(self):
        '''Return internal filter condiguration as text for humans.
        '''
        filts = list(self._parts.values())
        filts.sort(key=lambda x:x.colname)
        return ' & '.join([
                x.colname+':'+x.value
                for x in filts
                ])

class ScoreFilter:
    '''Manage filtering based on score values.
    '''
    filt_delim=':'
    filt_list_delim=','
    form_prefix='filt_'
    def get_filter_config(self):
        '''Return internal filter condiguration as text for query string.
        '''
        return self.filt_list_delim.join([
                self.filt_delim.join([k,v])
                for k,v in six.iteritems(self._filt)
                ])
    def __init__(self,filt_cols,ordering,filt='',labeler=None):
        '''Construct, optionally from query string.

        filt_cols is a list of column codes
        filt is a string of the form built by get_filter_config
        ordering is a function taking a column code and a boolean indicating
            sort direction, and returning an ordered iterable of (key,value)
            for the column
        labeler is a function taking a column code and returning a
            human-friendly label
        '''
        self._filt_cols = filt_cols
        self._ordering = ordering
        self._filt = {}
        if filt:
            for item in filt.split(self.filt_list_delim):
                colkey,crit = item.split(self.filt_delim)
                if colkey in filt_cols and crit:
                    self._filt[colkey] = crit
        self._label = labeler or (lambda x:x)
    def get_filter_form_class(self):
        '''Return a form allowing the user to edit the filter.

        POST data can be handled by update_filter().
        '''
        from dtk.dynaform import FormFactory
        from django import forms
        ff = FormFactory()
        for code in self._filt_cols:
            field_code = self.form_prefix + code
            ff.add_field(field_code,forms.CharField(
                    label=self._label(code),
                    initial=self._filt.get(code),
                    required=False,
                    ))
        return ff.get_form_class()
    def update_filter(self,p):
        '''Update internal filter condiguration from POST data.
        '''
        new = {}
        for field_code in p:
            if field_code.startswith(self.form_prefix):
                colkey = field_code[len(self.form_prefix):]
                val = p[field_code]
                if val:
                    new[colkey] = val
        self._filt=new
    def add_to_key_filter(self,key_filter):
        '''Update a KeyFilter object to include this filter.
        '''
        # calculate filter
        for colkey,crit in six.iteritems(self._filt):
            s = set()
            if crit[0] == '<':
                limit = float(crit[1:])
                for row_key,val in self._ordering(colkey,False):
                    if val >= limit:
                        break
                    s.add(row_key)
            elif crit[0] == '>':
                limit = float(crit[1:])
                for row_key,val in self._ordering(colkey,True):
                    if val <= limit:
                        break
                    s.add(row_key)
            elif crit.startswith('top'):
                l = self._ordering(colkey,True)
                if crit[-1] == '%':
                    limit = int(crit[3:-1])
                    limit = (len(l)*limit)//100
                else:
                    limit = int(crit[3:])
                for row_key,val in l[:limit]:
                    s.add(row_key)
            else:
                raise Exception("unknown filter spec: '%s'" % crit)
            key_filter.merge(
                    colkey,
                    self._label(colkey),
                    crit,
                    s,
                    )

# XXX Next steps:
# XXX - reimplement IndFilter to have a parallel structure to ScoreFilter
class IdRowSource():
    '''Assemble data for table display.
    '''
    def __init__(self,ordering,sort,key_filter,condense=False):
        '''Determine keys in table and their order.

        ordering is a function taking a column code and a boolean indicating
            sort direction, and returning an ordered iterable of (key,value)
            for the column
        sort is a SortHandler object or equivalent
        key_filter is a KeyFilter object or equivalent, with all filters
            loaded

        Here, the following are calculated:
        _row_keys has all keys ordered by the sort spec
        _ranker has the ranks for all keys in _row_keys
        _matched_keys is the subset of _row_keys with the filter applied
        '''
        # calculate filter
        self._row_keys = ordering(sort.colspec,sort.minus)

        if condense:
            from dtk.enrichment import make_dpi_group_mapping, condense_ordering_with_mapping
            ids = [x[0] for x in self._row_keys]
            row_to_group, _ = make_dpi_group_mapping(ids)
            condensed, row_condensed = condense_ordering_with_mapping(self._row_keys,
                                                       row_to_group)
            self._row_keys = condensed
            self._row_condensed = row_condensed
        else:
            self._row_to_group = None


        from dtk.scores import Ranker
        self._ranker = Ranker(self._row_keys)
        self._matched_keys = [
                    x
                    for x in self._row_keys
                    if key_filter.ok(x[0])
                    ]

    def condensed_group(self, row_id):
        if self._row_condensed:
            return self._row_condensed[row_id]
        else:
            return []

    def row_count(self):
        '''Return the number of rows matching the filter.
        '''
        return len(self._matched_keys)
    def ordered_ids(self):
        for x in self._matched_keys:
            yield x[0]
    def get_page(self,get_value,pager,colkeys):
        '''Return data for one page of results.

        The returned object is a list of dicts, one per row.  Each dict
        contains a value for each key in colkeys, plus the value of the
        rowkey under the key '_pk', and the zero-based index of the row in
        _matched_keys as '_idx'.  This data structure may be freely
        modified by the caller, for example to add more data.
        '''
        result = []
        for idx,(rowkey,value) in enumerate(
                    self._matched_keys[pager.page_start:pager.page_end]
                    ):
            d = {
                    colkey:get_value(colkey,rowkey)
                    for colkey in colkeys
                    }
            d.update(
                    _pk=rowkey,
                    _idx=pager.page_start+idx,
                    )
            result.append(d)
        return result
    def get_ranker(self):
        '''Return the positions of the keys in the unfiltered list.
        '''
        return self._ranker
