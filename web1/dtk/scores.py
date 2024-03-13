from __future__ import print_function
from django.utils.html import format_html

import logging
logger = logging.getLogger(__name__)

def get_ranked_groups(ordering):
    '''Given an ordering sorted in descending order, return rank information.

    For every distinct score, return a pair like:
    (number of higher scores,list of keys of all scores tied at this level)
    '''
    ahead = 0
    tied = []
    last_score = None
    for label,score in ordering:
        if score != last_score:
            if tied:
                yield (ahead,tied)
            ahead += len(tied)
            tied = []
            last_score = score
        tied.append(label)
    if tied:
        yield (ahead,tied)

class Ranker:
    '''given an ordering, create a ranking, with tie handling.

    ctor input is assumed to be sorted highest to lowest.
    '''
    def __init__(self,ordering,none_if_missing=False):
        self.result = {}
        for ahead,tied in get_ranked_groups(ordering):
            for label in tied:
                self.result[label] = (ahead,len(tied))
        self.none_if_missing = none_if_missing
        self.total = len(self.result)
    def keys(self):
        return set(self.result.keys())
    def get(self,label):
        if label in self.result:
            ahead,tied = self.result[label]
            return 1+ahead+tied//2
        return self.total+1 if not self.none_if_missing else None
    def get_pct(self,label):
        try:
            ahead,tied = self.result[label]
        except KeyError:
            return 100.0
        # Note that tied includes itself, so we increment by 1 to get an ordinal rank starting
        # at 1, but then subtract 1 to remove ourself from the tied count.
        # Also note, this is a pessimistic definition of percentile in the case of ties,
        # as we are treating all tied items as ahead of ourself.
        return 100*(ahead+tied + 1 - 1)/self.total
    def get_details(self,label):
        """Returns # of items ahead, tied and behind.

        NOTE: Tied includes itself, so will always be at least 1.
        ahead + tied + behind will sum to the total number of items.
        """
        if label in self.result:
            ahead,tied = self.result[label]
            return (ahead, tied, self.total - ahead - tied)
        return (len(self.result), 0, 0)
    def get_all(self,labels):
        return [self.get(l) for l in labels]
    def avg(self,labels):
        return sum(self.get_all(labels))/len(labels)
    def sigma(self,labels):
        def score(rank):
            width=100
            center=200
            from dtk.num import sigma
            return sigma((center-rank)/(0.5*width))
        return sum(map(score,self.get_all(labels)))/len(labels)

class JobCodeSelector:
    '''Allow easy extraction of single data codes.

    This class holds a method for building a choices list from a SourceList
    object, and a method that, given a code from that choices list, returns
    a data catalog the code can be used with.
    '''
    @classmethod
    def get_choices(cls,sources,keyname,typename):
        choices = []
        for source in sources.sources():
            bji = source.bji()
            cat = bji.get_data_catalog()
            choices += [
                        (
                            '%d_%s'%(bji.job.id,x),
                            '%s %s'%(source.label(),cat.get_label(x)),
                        )
                        for x in cat.get_codes(keyname,typename)
                    ]
        if not choices:
            choices = (
                    ('','No %s %s in selected sources'%(keyname,typename)),
                    )
        return choices
    @classmethod
    def get_catalog(cls,ws,choice):
        job_id,subcode = choice.split('_')
        import runner.data_catalog as dc
        cat = dc.Catalog()
        from runner.process_info import JobInfo
        bji = JobInfo.get_bound(ws,int(job_id))
        bji.fetch_lts_data()
        for cg in bji.get_data_code_groups():
            cat.add_group(job_id+'_',cg)
        return cat

# Design:
# - SourceList manages a list of 'jobs of interest' providing scores, and
#   allows assignment of custom labels to jobs.  The list can be serialized
#   and deserialized for saving in the session object, and other places.
#   Individual score data can be retrieved through a 'global code' consisting
#   of a job id and score code separated by an underbar.
# - SourceListWithEnables is a derived class which adds tracking of individual
#   score enables within each source
# - SourceTable can be instantiated from either of the above, and supports
#   editing the underlying attributes with a table-like form.  The columns
#   are configurable.  It supports editing of labels and enables, multiple
#   score selection columns, and static display of underlying job information.
#   It's designed for easy hookup with DumaView's form/post/context model.
# XXX Additionally, job_start should probably be using SourceTable, rather
# XXX than custom code, for the 'sources' section.
# XXX
# Boilerplate still required on score selection views:
# - define mode for score select subview
# - custom_setup:
#   - initialize and load self.sources
#   - force select subview if necessary
#   - if select subview, set up button_map and source_table
# - custom_context:
#   - place source_table rendering data in context
# - make score form from source table
# - make form for any additional info on select subview
# - move data to query parms and redirect on display post
# - handle defaults post
# XXX some of the above is pretty generic, and could probably live in the
# XXX view base class
class SourceList(object):
    '''Track score sources of current interest.

    Initialization:
    - load_from_string()
    - load_from_session() (also enables storing updates to session)
    - load_defaults() (determine most recent job of each type)
    Extraction:
    - to_string()
    - sources() - returns list of proxy objects
    Modification:
    - add_from_string() - add more jobs
    - remove()
    - set_label() - assign custom job label
    '''
    rec_sep = '|'
    field_sep = ':'
    source_key_prefix = 'score_source_data'
    def src_key(self):
        return "%s_%d" % (self.source_key_prefix,self.ws.id)
    class SourceProxy(object):
        def __init__(self,ws,
                    job_id=None, job=None, # one of these 2 is required
                    label=None,
                    bji=None,
                    ):
            self.ws = ws
            self._job_id = job.id if job is not None else job_id
            self._label = label
            self._bji = bji
            self._default_label = None
        def job_id(self): return self._job_id
        def bji(self):
            if self._bji is None:
                from runner.process_info import JobInfo
                self._bji = JobInfo.get_bound(self.ws,self._job_id)
            return self._bji
        def default_label(self):
            if self._default_label is None:
                bji = self.bji()
                self._default_label = bji.role_label()
            return self._default_label
        def label(self):
            if self._label is None:
                return self.default_label()
            return self._label
        def __repr__(self):
            return "<dtk.scores.SourceProxy %d (%s)>"%(
                    self._job_id,
                    self.label(),
                    )
    def __init__(self,ws,jcc=None,session=None):
        self.ws = ws
        self.SourceClass = self.SourceProxy # allows override
        self._session = session
        self._sources = []
        from runner.process_info import JobCrossChecker
        self._jcc = JobCrossChecker() if jcc is None else jcc
    def _write_session(self):
        if self._session is not None:
            self._session[self.src_key()] = self.to_string()
    def _parse_source_string(self,s):
        job_id = s.split(self.field_sep)[0]
        label = s[len(job_id)+1:]
        if not label:
            label = None
        job_id = int(job_id)
        dup = job_id in [x.job_id() for x in self._sources]
        return job_id,label,dup
    def _add_source(self,pos,**kwargs):
        # NOTE: caller must call _write_session afterwards
        new_src = self.SourceClass(self.ws,**kwargs)
        if pos is None:
            self._sources.append(new_src)
        else:
            self._sources.insert(pos,new_src)
    def get_source_types(self):
        # get all workspace names, ordered by level
        order={x[1]:x[0] for x in enumerate(self._jcc.level_names)}
        namelist = []
        for name in self._jcc.ws_jobnames(self.ws):
            stem = name.split('_')[0]
            if stem in order:
                namelist.append( (name,order[stem]) )
        namelist.sort(key=lambda x:x[1])
        # get corresponding labels for names offering scores
        result = []
        from runner.process_info import JobInfo
        for name,order in namelist:
            info = JobInfo.get_unbound(name)
            if info.get_data_catalog().has_codes():
                result.append( (name,info.source_label(name)) )
        return result
    def to_string(self,partial=None):
        return self.rec_sep.join([
                    str(x.job_id())+self.field_sep+x.label()
                    for x in self._sources
                    if not partial or x.job_id() in partial
                    ])
    #
    # modifiers
    #
    def add_from_string(self,s,pos=None):
        job_id,label,dup = self._parse_source_string(s)
        if dup:
            return
        self._add_source(pos,job_id=job_id,label=label)
        self._write_session()
    def remove(self,src):
        self._sources.remove(src)
        self._write_session()
    def set_label(self,source,label):
        source._label = label
        self._write_session()
    def sort(self):
        level2int={x[1]:x[0] for x in enumerate(self._jcc.level_names)}
        l = [
                (
                    (level2int[x.bji().job_type], x.label()), # key
                    x,
                )
                for x in self._sources
            ]
        l.sort(key=lambda x: x[0])
        self._sources = [x[1] for x in l]
        self._write_session()
    #
    # initializers
    #
    def load_from_string(self,s):
        self._sources = []
        self._write_session()
        for item in s.split(self.rec_sep):
            try:
                job_id,label,dup = self._parse_source_string(item)
                if dup:
                    continue
                self._add_source(None,job_id=job_id,label=label)
            except ImportError:
                # in development, it's possible to develop a new plugin
                # and then switch to a branch where the plugin doesn't
                # yet exist, but its name is still in the session; silently
                # skip that session entry without throwing an error
                pass
        self._write_session()
    def load_from_session(self,session):
        self._session = session
        config_string = self._session.get(self.src_key(),'')
        if config_string:
            self.load_from_string(config_string)
        else:
            self.load_defaults()
    def load_from_scoreset(self, scoreset_id):
        from browse.models import ScoreSet
        ss = ScoreSet.objects.get(pk=scoreset_id)
        self.load_from_string(ss.source_list_source_string())

    def load_defaults(self):
        from runner.process_info import JobInfo
        from runner.models import Process
        self._sources = []
        self._write_session()
        jobnames = [name
                for name in self._jcc.ws_jobnames(self.ws)
                if JobInfo.get_unbound(name).is_a_source()
                ]
        qs = Process.objects.filter(
                name__in=jobnames,
                status=Process.status_vals.SUCCEEDED,
                )
        from django.db.models import Max
        job_ids = [
                job_id
                for role,job_id in qs.values_list('role').annotate(Max('id'))
                if role
                ]
        for job in Process.objects.filter(pk__in=job_ids):
            self._add_source(
                        pos=None,
                        job=job,
                        )
        self._write_session()
        self.sort()
    #
    # access to underlying source objects
    #
    def sources(self):
        return self._sources
    def sources_bound_jobs(self):
        from runner.process_info import JobInfo
        unbound_ids = [src.job_id() for src in self._sources if not src._bji]
        bjis = JobInfo.get_all_bound(self.ws,unbound_ids)
        bji_lookup = dict(zip(unbound_ids, bjis))
        for src in self._sources:
            if not src._bji:
                src._bji = bji_lookup[src.job_id()]
        return [src.bji() for src in self._sources]

    def parse_global_code(self,global_code):
        job_id,code = global_code.split('_')
        job_id = int(job_id)
        for src in self._sources:
            if src.job_id() == job_id:
                cat = src.bji().get_data_catalog()
                return (src,cat,code)
        # We've been asked for a score that isn't in the source list
        #raise ValueError("Invalid Global Code '%s'"%global_code)
        # XXX Originally, this threw an exception, but for development,
        # XXX it's nice to be able to type a new global code into a
        # XXX URL. So, the code below fakes things up by returning date
        # XXX from a temporary source object that isn't in the list.
        # XXX If this violates some assumption somewhere, it might be
        # XXX worth adding the new source to the list as well.
        new_src = self.SourceClass(self.ws,job_id=job_id)
        cat = new_src.bji().get_data_catalog()
        return (new_src,cat,code)
    #
    # convenience functions for accessing scores
    #
    def get_ordering_from_code(self,global_code,desc):
        src,cat,code = self.parse_global_code(global_code)
        return cat.get_ordering(code,desc)
    def get_label_from_code(self,global_code):
        src,cat,code = self.parse_global_code(global_code)
        return ' '.join([src.label(),cat.get_label(code)])
    def get_value_from_code(self,global_code,key):
        src,cat,code = self.parse_global_code(global_code)
        # Note that this returns a (value,attr) tuple; the column
        # formatters need to deal with this
        return cat.get_cell(code,key)

# convenience function for getting SourceList from url components
def get_sourcelist(ws,session,joblist=None,prescreen=None):
    sl = SourceList(ws)
    if joblist:
        sl.load_from_string(joblist)
    elif prescreen:
        sl.load_from_string(prescreen.source_list_jobs())
    else:
        sl.load_from_session(session)
    return sl

class SourceListWithEnables(SourceList):
    enable_key_prefix = 'score_enable_data'
    def en_key(self):
        return "%s_%d" % (self.enable_key_prefix,self.ws.id)
    class SourceProxy(SourceList.SourceProxy):
        def __init__(self,ws,**kwargs):
            super(SourceListWithEnables.SourceProxy,self).__init__(ws,**kwargs)
            all_codes=set([x[0] for x in self.get_enable_choices()])
            self._set_enables(all_codes)
        def _set_enables(self,s):
            self._enables=set()
            cat=self.bji().get_data_catalog()
            for code in cat.get_codes('wsa','score'):
                if code in s:
                    self._enables.add(code)
        def enabled_codes(self):
            cat=self.bji().get_data_catalog()
            return [
                code
                for code in cat.get_codes('wsa','score')
                if code in self._enables
                ]
        def get_enable_choices(self):
            cat=self.bji().get_data_catalog()
            return [
                (code,cat.get_label(code))
                for code in cat.get_codes('wsa','score')
                ]
    def __init__(self,ws,**kwargs):
        super(SourceListWithEnables,self).__init__(ws,**kwargs)
        self.SourceClass = self.SourceProxy
    def _write_session_enables(self):
        if self._session is not None:
            self._session[self.en_key()] = self.get_string_from_enables()
    def load_from_session(self,session):
        super(SourceListWithEnables,self).load_from_session(session)
        try:
            s = self._session[self.en_key()]
            if s:
                self.set_enables_from_string(s)
        except KeyError:
            pass
        self._write_session_enables()
    def load_defaults(self):
        super(SourceListWithEnables,self).load_defaults()
        self._write_session_enables()
    def get_string_from_enables(self):
        result = []
        for src in self._sources:
            l = [str(src.job_id())]+src.enabled_codes()
            result.append(self.field_sep.join(l))
        return self.rec_sep.join(result)
    def get_code_list_from_enables(self):
        result = []
        for src in self._sources:
            job_id = src.bji().job.id
            for code in src.enabled_codes():
                result.append('%d_%s'%(job_id,code))
        return result
    def set_enables_from_string(self,s):
        new_vals = {}
        for part in s.split(self.rec_sep):
            if part:
                subparts = part.split(self.field_sep)
                new_vals[int(subparts[0])] = set(subparts[1:])
        for src in self._sources:
            src._set_enables(new_vals.get(src.job_id(),set()))
        self._write_session_enables()
    def set_enables(self,source,enables):
        source._set_enables(enables)
        self._write_session_enables()
    def get_string_from_active_sources(self):
        used_jobs = set()
        for src in self._sources:
            if src.enabled_codes():
                used_jobs.add(src.job_id())
        return self.to_string(used_jobs)

class ColType:
    def add_edit_widget(self,row,form): return
    def add_form_fields(self,ff,sl): return
    def update_source(self,post,sl): return

class LabelColType(ColType):
    base_key='lab_'
    def _make_key(self,src):
        return self.base_key+str(src.job_id())
    def add_edit_widget(self,row,form):
        key=self._make_key(row.src)
        row.label_field=form[key]
    def add_form_fields(self,ff,sl):
        from django import forms
        for src in sl.sources():
            key=self._make_key(src)
            f = forms.CharField(
                        initial=src.label(),
                        required=False,
                        )
            f.widget.attrs['size']=ff.source_label_field_size
            ff.add_field(key,f)
    @staticmethod
    def job_type_tooltip(data,row,col):
        from django.utils.html import format_html
        icon = ''
        if row.src.label() != row.src.default_label():
            from dtk.html import glyph_icon
            icon=glyph_icon('info-sign',hover=row.src.default_label())
        return format_html(u'{}{}',data,icon)
    def add_column(self,cols):
        from dtk.table import Table
        cols.append(Table.Column('Prefix',
                        code='label_field',
                        cell_html_hook=self.job_type_tooltip,
                        ))
    def update_source(self,post,sl):
        # make local copy of list, so iterator isn't affected by deletions
        for src in list(sl.sources()):
            key=self._make_key(src)
            if key not in post:
                continue
            label = post[key]
            if label:
                sl.set_label(src,label)
            else:
                sl.remove(src)

class EnablesColType(ColType):
    base_key='en_'
    def _make_key(self,src):
        return self.base_key+str(src.job_id())
    def add_edit_widget(self,row,form):
        key=self._make_key(row.src)
        row.enable_field=form[key]
    def add_form_fields(self,ff,sl):
        from django import forms
        from dtk.html import WrappingCheckboxSelectMultiple
        for src in sl.sources():
            key=self._make_key(src)
            ff.add_field(key,forms.MultipleChoiceField(
                    choices=src.get_enable_choices(),
                    initial=src.enabled_codes(),
                    widget=WrappingCheckboxSelectMultiple,
                    required=False,
                    ))
    def add_column(self,cols):
        from dtk.table import Table
        cols.append(Table.Column('Show Score',
                        code='enable_field',
                        ))
    def update_source(self,post,sl):
        for src in sl.sources():
            key=self._make_key(src)
            sl.set_enables(src,post[key])

class SelectColType(ColType):
    def __init__(self,key,label,initial=None,dct='wsa'):
        self._key = key
        self._label = label
        self._errors = []
        self._initial = initial
        self._dct = dct
    def add_edit_widget(self,row,form):
        # if the form is bound, and there are errors on the field,
        # stash them so they can be put into the column header;
        # this executed redundantly on every row, but this is
        # the only place we have access to the form; this is a
        # bit of a special case, since we're spreading pieces of
        # a single form field across all the rows
        self._errors = form[self._key].errors
        # render the portion of the radio buttons that apply to
        # the source associated with this row
        from dtk.html import radio_cell
        setattr(row,self._key,radio_cell(
                self._key,
                row.src.bji().get_score_choices(dtc=self._dct),
                checked=self._initial,
                ))
    def add_form_fields(self,ff,sl):
        from django import forms
        # This will handle posts from the HTML constructed by
        # dtk.html.radio_cell(bji.get_score_choices)
        choices = []
        for source in sl.sources():
            cat = source.bji().get_data_catalog()
            for code in cat.get_codes(self._dct,'score'):
                choices.append((
                        '%d_%s'%(source.job_id(),code),
                        cat.get_label(code),
                        ))
        ff.add_field(self._key,forms.ChoiceField(
                        choices=choices,
                        widget=forms.RadioSelect,
                        ))
    def add_column(self,cols):
        from dtk.table import Table
        label = self._label
        if self._errors:
            from dtk.html import alert
            from django.utils.html import format_html
            label = format_html(u'{} ({})',
                        label,
                        alert(','.join([x for x in self._errors])),
                        )
        cols.append(Table.Column(label,
                        code=self._key,
                        ))
    # this has no update_source() method, because Select columns
    # are used as parameters by the view, not saved in a SourceList

class JobTimeColType(ColType):
    def add_column(self,cols):
        from dtk.table import Table
        from dtk.text import fmt_time
        cols.append(Table.Column('Run Time',
                        extract=lambda x:x.src.bji().job.completed,
                        cell_fmt=fmt_time,
                        ))

class JobStatusColType(ColType):
    def __init__(self,context):
        self._context = context
    def add_column(self,cols):
        from dtk.table import Table
        from runner.templatetags.process import job_summary_impl
        from django.utils.safestring import mark_safe
        def job_summary(job):
            return mark_safe(job_summary_impl(self._context,job))
        cols.append(Table.Column('Job Status',
                        extract=lambda x:x.src.bji().job,
                        cell_fmt=job_summary,
                        ))

class SourceTable:
    def __init__(self,sl,recipe):
        self.recipe=recipe
        self.sl = sl
    def get_table(self,form):
        # XXX this might be a little prettier if DummyClass took the original
        # XXX row object as a ctor param, and forwarded all unknown attr
        # XXX requests
        class DummyClass:
            pass
        rows=[]
        for source in self.sl.sources():
            row = DummyClass()
            rows.append(row)
            row.src = source
            for ct in self.recipe:
                ct.add_edit_widget(row,form)
        from dtk.table import Table
        columns = []
        for ct in self.recipe:
            ct.add_column(columns)
        return Table(rows,columns)
    def make_form_class(self):
        from dtk.dynaform import FormFactory
        ff = FormFactory()
        for ct in self.recipe:
            ct.add_form_fields(ff,self.sl)
        return ff.get_form_class()
    def update_source_data(self,post):
        for ct in self.recipe:
            ct.update_source(post,self.sl)

################################################################################
# Norm classes map a score into a range
# - the ctor takes a single ordering parameter like [(key,score),...]
#   - the score in this list may be NaN, which is treated as 0
# - the get() method takes a key and returns a mapped score value
# - the fmt() method takes a value (as returned by get()) and returns a
#   compact string representation
# - the range() method returns a pair of values indicating the min and max
#   values returned by get(). It might either be the actual range, or the
#   theoretical range -- e.g. a percent norming could return (0,100) even
#   if no value in the ordering hit the 100% criteria.
################################################################################
class BaseNorm:
    '''Provides a common default fmt() method for norm classes.'''
    def fmt(self, val):
        return f'{val:.3g}' if val is not None else 'None'

class BaseScoreNorm(BaseNorm):
    '''Base class for statistical manipulation of score values.'''
    def __init__(self,ordering):
        from math import isnan
        self.scores = {
                x[0]:float(x[1])
                for x in ordering
                if not isnan(float(x[1]))
                }
        self._cached_actual_range = None
    def get(self,key):
        return self.scores.get(key,0.0)
    def _actual_range(self):
        if self._cached_actual_range is None:
            # find largest and smallest values; note that 0 is a
            # possible value even if it isn't in the loaded scores
            if self.scores:
                self._cached_actual_range = (
                        min(0,min(self.scores.values())),
                        max(0,max(self.scores.values())),
                        )
            else:
                self._cached_actual_range = (0,0)
        return self._cached_actual_range

class ZNorm(BaseScoreNorm):
    '''Norm to avg 0 sd 1.'''
    def __init__(self,ordering):
        super().__init__(ordering)
        if self.scores:
            from dtk.num import avg_sd
            l = list(self.scores.values())
            self.avg,self.sd = avg_sd(l)
            if self.sd == 0:
                # This means all the scores are the same. We need to avoid
                # a divide-by-zero error in get(). Note that the base-class
                # get() will only return either the single constant value,
                # or 0.
                if l[0]:
                    # the single constant value is non-zero; arrange sd so
                    # get() returns 0 or +/-1
                    self.sd = l[0]
                else:
                    # get() will only ever return 0, doesn't matter what
                    # we divide by
                    self.sd=1
        else:
            self.avg=0
            self.sd=1
    def get(self,key):
        val = super().get(key)
        return (val - self.avg)/self.sd

class MMNorm(BaseScoreNorm):
    '''Norm into 0 to 1 range based on min and max values.'''
    def __init__(self,ordering):
        super().__init__(ordering)
        self.min,self.max = self._actual_range()
        self.spread = self.max-self.min
    def get(self,key):
        if self.spread == 0.0:
            return 0.0
        val = super().get(key)
        result = (val-self.min)/self.spread
        assert 0 <= result <= 1, \
            f"Got {result} from {self.max} ({val} - {self.min})/{self.spread}"
        return result
    def range(self):
        return (0, 1)

class LMMNorm(MMNorm):
    def get(self,key):
        # replicate the scaling in dtk.scores.FMCalibrator
        # XXX this is just copy/pasted since it's currently experimental
        top_val = 0.9999
        import math
        top_log = -math.log10(1-top_val)
        return -math.log10(1 - super().get(key)*top_val)/top_log
class RankNorm:
    '''Norm based on rank rather than score.'''
    def __init__(self, ordering, none_if_missing=False):
        self.ranker = Ranker(ordering, none_if_missing=none_if_missing)
    def get(self, key):
        return self.ranker.get(key)
    def fmt(self, val):
        return f'{val:0d}' if val is not None else 'None'
    def range(self):
        return (1, self.ranker.total)

class PctNorm(RankNorm):
    '''Norm based on mapping rank to percent.'''
    def get(self, key):
        # Prefer 'bigger is better' style values.
        rank = self.ranker.get(key)
        if rank is None:
            return None
        return 100 - rank * 100 / self.ranker.total
    def fmt(self, val):
        return f'{val:.1f}%' if val is not None else 'None'
    def range(self):
        return (0, 100)

class NoNorm(BaseScoreNorm):
    '''Pass un-normed score without modification.

    This just provides the norm API without altering values.
    '''
    def range(self):
        return self._actual_range()

class WzsNorm():
    """This one is different from the others, it is persistent and can switch between jobs.

    Mostly this is because it has a lot more heavy-weight init pulling data from the wzs job.
    """
    def __init__(self, weights, norms):
        self.weights = weights
        self.norms = norms
    
    def load(self, bji, code):
        if code == 'wzs':
            ordering = bji.get_data_catalog().get_ordering(code, True)
            h = max(x[1] for x in ordering)
            # Rescale wzs to 0-10, to make visualization easier.
            scale = 10 / h
            self.scores = {id:val * scale for id, val in ordering}
            self._cur_weight = 1
        else:
            self.scores, self._cur_weight = self.find_entry(bji, code)

        self.low = min(self.scores.values()) * self._cur_weight
        self.high = max(self.scores.values()) * self._cur_weight
    
    def find_entry(self, bji, code):
        bji_name = f'{bji.job.role}_{code}'
        for (name, weight), norm in zip(self.weights, self.norms):
            if name.lower() == bji_name.lower():
                return norm, weight
        raise Exception(f"Couldn't find {bji_name} {bji.job.id} in {self.weights}")

    def get(self,key):
        return self.scores.get(key, 0.0) * self._cur_weight

    def fmt(self, val):
        return f'{val:.3g}' if val is not None else 'None'

    def range(self):
        return (self.low, self.high)



class TargetNorm:
    """This is still something of a work in progress.  The normalization/scoring may change."""
    def __init__(self, ordering, none_if_missing):
        ids = [x[0] for x in ordering]
        from browse.models import WsAnnotation, Workspace
        ws = Workspace.objects.get(wsannotation=ids[0])
        wsas = WsAnnotation.all_objects.filter(pk__in=ids)
        id2agent = {id:agent for (id, agent) in wsas.values_list('id', 'agent_id')}
        from dtk.prot_map import AgentTargetCache
        atc = AgentTargetCache.atc_for_wsas(wsas, ws=ws)

        from collections import defaultdict
        target_ranks = defaultdict(list)

        for id, score in ordering:
            agent = id2agent[id]
            for uni, gene, direc in atc.info_for_agent(agent):
                target_ranks[uni].append(score)
        
        import numpy as np
        target_scores = [(k,np.min(v)) for k, v in target_ranks.items()]
        target_scores.sort(key=lambda x: -x[1])
        self.target_ranker = RankNorm(target_scores, none_if_missing=none_if_missing)
        self.target_scores = target_scores
        self.atc = atc
        self.id2agent = id2agent
    
    def get(self, key):
        agent = self.id2agent.get(key, None)
        if not agent:
            return None
        ranks = []
        for uni, gene, direc in self.atc.info_for_agent(agent):
            ranks.append(self.target_ranker.get(uni))
        ranks = [x for x in ranks if x is not None]
        if not ranks:
            return None
        import numpy as np
        return np.min(ranks)
    
    def range(self):
        return self.target_ranker.range()

    def fmt(self, val):
        return f'{val:0d}' if val is not None else 'None'

        






def check_and_fix_negs(ordering):
     negs = sum(1 for x in ordering if x[1] < 0.)
     if negs > 0:
         print(f'{negs} of {len(ordering)} total values were negative. Absolute value is being applied.')
         return [(x[0],abs(x[1])) for x in ordering]
     return ordering





def condense_scores(root_node, ancestors_func, descendants_func, ordering):
    # Dist to root for each item in ordering.
    # Sort desc (score, -dist to root)
    # Iterate:
    #   If subsumed, continue
    #   Add to output
    #   Mark off all ancestors
    #   Mark off all descendants
    
    dist_to_root = {}
    def vis(node, dist):
        if node in dist_to_root:
            return
        dist_to_root[node] = dist
        for next_node in descendants_func(node):
            vis(next_node, dist + 1)
    vis(root_node, 0)

    ordering = sorted(ordering, key=lambda x: (-x[1], dist_to_root.get(x[0], 0)))


    subs_anc = set()
    subs_desc = set()

    def mark(node, marked_set, neighbor_func):
        if node in marked_set:
            return set()
        
        marked_set.add(node)
        out = set([node])
        for next_node in neighbor_func(node):
            out |= mark(next_node, marked_set, neighbor_func)
        return out
    
    output = []
    node_subsumes = {}
    for node, score in ordering:
        if node in subs_anc or node in subs_desc:
            continue

        marked = set()
        marked |= mark(node, subs_anc, ancestors_func)
        marked |= mark(node, subs_desc, descendants_func)
        marked -= {node}

        if marked:
            node_subsumes[node] = marked
        output.append((node, score))
    
    return output, node_subsumes
