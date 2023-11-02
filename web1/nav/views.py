from __future__ import print_function
from builtins import range
from django.shortcuts import render

from django.db import transaction
from django.http import HttpResponseRedirect
from django.contrib.auth.decorators import login_required
import datetime
from django import forms

from browse.models import Workspace
from path_helper import PathHelper
from runner.models import Process
from dtk.url import UrlConfig
from dtk.duma_view import DumaView,list_of,boolean
from dtk.table import Table,SortHandler,KeyFilter
from functools import reduce

from django.utils.html import format_html
from django.urls import reverse

import logging
import six
logger = logging.getLogger(__name__)

################################################################################
# shared utilities
################################################################################
from browse.views import make_ctx,is_demo,post_ok

################################################################################
# non-workspace views
################################################################################
@login_required
def ws_cmp(request):
    url_config=UrlConfig(request,
                defaults={
                    'all':False,
                })
    # get workspace list
    show_all=url_config.as_bool('all')
    queryset = Workspace.objects.all()
    if not show_all:
        queryset = queryset.filter(active=True)
    queryset = queryset.order_by('name')
    # get jcc
    from runner.process_info import JobCrossChecker
    jcc=JobCrossChecker()
    results = []
    # see if we should be doing a recalculate
    recalc = False
    if request.method == 'POST' and post_ok(request):
        if 'recalc_btn' in request.POST:
            recalc = True
    # build an initial list of information about the scores in each
    # workspace
    from dtk.html import link
    from runner.models import Process
    from runner.process_info import JobInfo
    order={x[1]:x[0] for x in enumerate(jcc.level_names)}
    for ws in queryset:
        row = {}
        results.append(row)
        row['ref'] = ws
        row['link'] = link(ws.name,ws.reverse('nav_scoreboard'))
        # for every jobname in the workspace that produces scores,
        # get the label and level for the job, and the most recent
        # successful job id
        scores = {}
        from dtk.scores import SourceList
        sl = SourceList(ws,jcc)
        sl.load_defaults()
        row['source_list'] = sl
        for src in sl.sources():
            key = (order[src.bji().job_type],src.label())
            scores[key] = src
        row['scores'] = scores
    # Now, build a superset of all score types across all workspaces,
    # and sort it by level to get the column order
    header = set()
    for row in results:
        header |= set(row['scores'].keys())
    header = list(header)
    header.sort()
    # get sort specs
    sorter = SortController(url_config)
    # collect all possible score subtypes for each header entry
    l = []
    for i,item in enumerate(header,start=1):
        plugin = jcc.level_names[item[0]]
        info = JobInfo.get_unbound(plugin)
        cat = info.get_data_catalog()
        choices = []
        # there isn't a great natural key for the columns that we've
        # grouped by label, so use an index, and stash the composite
        # key here so it doesn't need to be recalculated in 2 places
        for code in cat.get_codes('wsa','score'):
            if not cat.is_type(code,'efficacy'):
                continue
            colspec = "%d_%s" % (i, code)
            choices.append( {
                    'code':code,
                    'sortkey':colspec,
                    'label':sorter.col_html(colspec,cat.get_label(code))
                    } )
        if choices:
            l.append( (item,choices) )
    header = l
    # At this point, header looks like:
    # [
    #    # for each jobname
    #    (
    #       (level,JobLabel),
    #       [
    #           # for each score type within the job
    #           {
    #               'code':scorecode1, # like 'dcoe'
    #               'label':scorelabel1, # like 'Disease CO+'
    #               'sortkey':column_sort_key, # like '3_dcoe'
    #           },
    #           ...
    #       ]
    #    ),
    #    ...
    # ]
    # now reorganize scores list to match header
    from dtk.html import decimal_cell
    for row in results:
        row['groups'] = []
        ws = row['ref']
        row['sortkey'] = None
        for col in header:
            l = []
            row['groups'].append(l)
            for score in col[1]:
                try:
                    src = row['scores'][col[0]]
                    cat = src.bji().get_data_catalog()
                    if recalc:
                        print('recalc',ws,src.label(),src.job_id(),score['code'])
                        try:
                            import dea
                            dea_options = dea.Options(ws)
                            code = score['code']
                            ordering = cat.get_ordering(code,True)
                            fhf = dea.FileHandleFactory()
                            from path_helper import make_directory
                            fhf.dirpath = src.bji().get_dea_path()
                            make_directory(fhf.dirpath)
                            fhf.set_score(code)
                            dea_options.run(ordering,fhf)
                        except (ValueError,) as ex:
                            print('got exception',ex)
                    from dea import EnrichmentResult
                    er = EnrichmentResult(src.bji(),score['code'])
                    if score['sortkey'] == sorter.colspec:
                        row['sortkey'] = er.get_value()
                    l.append(decimal_cell(er.dea_link()))
                except KeyError:
                    l.append(decimal_cell(''))
    results.sort(key=lambda x:(x['sortkey'] is None, x['sortkey']),reverse=not sorter.minus)
    return render(request
                ,'nav/ws_cmp.html'
                ,make_ctx(request,None,'',{
                     'workspaces': results,
                     'show_all': show_all,
                     'header': header,
                     })
                )

################################################################################
# workspace views
################################################################################
def get_keys_for_wsa(wsa,col):
    if col == 'drug':
        # this is a special case for padre_final_predictions.tsv,
        # which is keyed by the agent_id
        text_key = str(wsa.agent_id)
        return set([text_key])
    if col == 'stitch_id':
        # this is a special case of trying to reconstruct the old
        # stitch id from a pubchem cid (which may be stored, or
        # may in turn be constructed from a new-style stitch id
        template = 'CID000000000'
        return set([
                template[:len(template)-len(x)]+x
                for x in wsa.agent.pubchem_cids(version=wsa.ws.get_dpi_version())
                ])
    drug_ids = getattr(wsa.agent,'m_'+col+'_set')
    native = getattr(wsa.agent,col)
    if native:
        drug_ids.add(native)
    return drug_ids

def process_adr_prefix(s):
    return s.split("_")[-1]

def extract_adr_vals(fn,keyset):
    from dtk.files import get_file_records
    # This functionality is somewhat out-of-date, and I'm not sure
    # whether all the files that come through here have headers or not.
    # But the code below was at least in some cases tossing the header
    # line on a mismatch to the keyset, so setting keep_header=None will
    # retain that functionality, with any header line being mismatched
    # and tossed inside get_file_records.
    src = get_file_records(fn,select=(keyset,0),keep_header=None)
    for fields in src:
        if len(fields) == 2:
            val = 1.0
        else:
            val = float(fields[2])
        adr = process_adr_prefix(fields[1])
        yield adr,val

def get_adr_vals_for_drug(wsa,fn):
    f = open(fn)
    header = next(f).strip('\n').split('\t')
    keys = get_keys_for_wsa(wsa,header[0])
    result = {}
    for adr,val in extract_adr_vals(fn,keys):
        result[adr] = val
    return result

def get_adr_vals_for_drugset(wsa_set,fn):
    f = open(fn)
    header = next(f).strip('\n').split('\t')
    keys = set()
    for wsa in wsa_set:
        keys |= get_keys_for_wsa(wsa,header[0])
    result = {}
    for adr,val in extract_adr_vals(fn,keys):
        if adr in result:
            result[adr] += val
        else:
            result[adr] = val
    return {k:v/len(wsa_set) for k,v in six.iteritems(result)}

def read_gottlieb(adr_ids):
    from dtk.s3_cache import S3File
    s3f = S3File('tox','gottliebSideEffectsWithMedDRA.tsv')
    s3f.fetch()
    f = open(s3f.path())
    for line in f:
        fields = line.strip('\n').split('\t')
        adr_id = fields[1].upper()
        if adr_id in adr_ids:
            yield (adr_id,fields[0],float(fields[2]))

class PadreDataSource:
    def ordering(self,colkey,desc):
        self._load_data()
        l = []
        for k,v in six.iteritems(self._data):
            try:
                val = v[colkey]
                l.append( (k,val) )
            except KeyError:
                # show all ADRs, not just those in sort column
                l.append( (k,None) )
        l.sort(key=lambda x:x[1] or 0,reverse=desc)
        return l
    def label(self,colkey):
        return self.def_by_code[colkey][2]
    def value(self,colkey,rowkey):
        self._load_data()
        return self._data[rowkey].get(colkey)
    def get_columns_form_class(self):
        from dtk.dynaform import FormFactory
        ff = FormFactory()
        for code,label,selected in self.all_column_specs():
            ff.add_field(code,forms.BooleanField(
                    label=label,
                    initial=selected,
                    required=False,
                    ))
        return ff.get_form_class()
    hidden = '-'
    def __init__(self,wsa,colorder):
        self._data = None
        self._wsa = wsa
        self._coldefs = [
                (None,'sev','Severity','%0.2f'),
                ('padre/padre_final_predictions.tsv'
                        ,'padre'
                        ,'Predicted SIDER Present'
                        ,'%0.4f'
                        ),
                ('padre/padre_final_predictions.tsv'
                        ,'ktpadre'
                        ,'KT Avg. Pred. SIDER Pres.'
                        ,'%0.4f'
                        ),
                ('adr.sider.default_aggregated.tsv'
                        ,'siderdef'
                        ,'SIDER Present'
                        ,'%d'
                        ),
                ('adr.sider.default_aggregated.tsv'
                        ,'ktsiderdef'
                        ,'SIDER KT Fraction Present'
                        ,'%0.2f'
                        ),
                ('adr.sider.portion_aggregated.tsv'
                        ,'siderprev'
                        ,'SIDER Prevalence'
                        ,'%0.4f'
                        ),
                ('adr.sider.portion_aggregated.tsv'
                        ,'ktsiderprev'
                        ,'SIDER KT Avg. Prevalence'
                        ,'%0.4f'
                        ),
                ('adr.sider.odds_ratio_aggregated.tsv'
                        ,'siderodds'
                        ,'SIDER Odds Ratio'
                        ,'%0.2f'
                        ),
                ('adr.offsides.odds_ratio_aggregated.tsv'
                        ,'offodds'
                        ,'OFFSIDES Odds Ratio'
                        ,'%0.2f'
                        ),
                ('adr.adrecs.portion_aggregated.tsv'
                        ,'adrecprev'
                        ,'ADRECS Prevalence'
                        ,'%0.4f'
                        ),
                ]
        self.default_order = []
        self.def_by_code = {}
        for coldef in self._coldefs:
            key = coldef[1]
            self.def_by_code[key] = coldef
            self.default_order.append(key)
        self._colorder = []
        seen = set()
        for code in colorder:
            key = code
            if key[0] == self.hidden:
                key = key[1:]
            assert key in self.def_by_code
            self._colorder.append(code)
            seen.add(key)
        any_seen = bool(seen)
        for key in self.default_order:
            if key not in seen:
                if any_seen:
                    self._colorder.append(self.hidden+key)
                else:
                    # we're reverting to defaults; by excluding the KT
                    # numbers and Padre predictions, the page loads
                    # much faster, and focuses attention on actual
                    # documented side effects for the drug
                    if key == 'padre' or key.startswith('kt'):
                        key = self.hidden+key
                    self._colorder.append(key)
    def all_column_specs(self):
        result = []
        for key in self._colorder:
            selected = key[0] != self.hidden
            if not selected:
                key = key[1:]
            result.append( (key, self.label(key), selected) )
        return result
    def active_column_codes(self):
        return [
                key
                for key in self._colorder
                if key[0] != self.hidden
                ]
    def _load_data(self):
        if self._data is not None:
            return # already loaded
        self._data = {} # { adr_id: { col_code: val, ...}, ...}
        from browse.models import WsAnnotation
        qs = WsAnnotation.objects.filter(
                    pk__in=self._wsa.ws.get_wsa_id_set('kts'),
                    )
        kt_list = list(qs)
        from dtk.s3_cache import S3File
        for key in self._colorder:
            if key[0] == self.hidden:
                continue
            fn,code,label,fmt = self.def_by_code[key]
            if not fn:
                continue
            if '/' in fn:
                # explicitly passing bucket
                bucket,fn = fn.split('/')
            else:
                bucket = 'tox'
            f = S3File(bucket,fn)
            f.fetch()
            if code.startswith('kt'):
                vals = get_adr_vals_for_drugset(kt_list,f.path())
            else:
                vals = get_adr_vals_for_drug(self._wsa,f.path())
            for k,v in six.iteritems(vals):
                self._data.setdefault(k,{})[code] = v
        # load names and severities for all side effects fetched above
        for adr_id,name,severity in read_gottlieb(set(self._data.keys())):
            self._data[adr_id]['name'] = name
            self._data[adr_id]['sev'] = severity

def handle_filter_edit(post_data,url_config):
    filt_list = []
    prefix='flt_'
    for key in post_data:
        if key.startswith(prefix):
            code = key[len(prefix):]
            crit = post_data[key]
            crit.strip()
            if crit:
                filt_list.append("%s:%s" % (code,crit))
        if key == 'page_size':
            url_config.modify({'page_size':post_data[key].strip()})
    return HttpResponseRedirect(
            url_config.here_url({'filter':','.join(filt_list)})
            )

def append_padre_files(result,adr_id,templates):
    from dtk.s3_cache import S3File,S3Bucket
    from path_helper import PathHelper
    # make a special bucket that caches into a publish subdir
    s3b = S3Bucket('padre')
    # XXX this is annoying -- both these levels should share a path
    s3b.cache_path = PathHelper.padre_pub
    s3b.bucket.cache_path = s3b.cache_path
    from dtk.html import link
    try:
        for label,template in templates:
            fn = template % adr_id
            s3f = S3File(s3b,fn)
            s3f.fetch()
            result.append(link(
                    label,
                    PathHelper.url_of_file(s3f.path()),
                    ))
    except IOError:
        pass
    return

def get_padre_stats_table(adr_id):
    from dtk.s3_cache import S3File
    s3f = S3File('padre','padre_attrs_selected.tsv')
    s3f.fetch()
    f = open(s3f.path())
    for line in f:
        adr,attr_list = line.strip('\n').split('\t')
        if adr == adr_id:
            pairs = attr_list.split(';')
            recs = [x.split(',') for x in pairs]
            recs.sort(key = lambda x:float(x[1]),reverse=True)
            return recs
    return None

@login_required
def adr_info(request,ws_id,adr_id):
    ws = Workspace.objects.get(pk=ws_id)
    name = ''
    for got_id,name,severity in read_gottlieb([adr_id]):
        pass
    if name:
        headline = name.title() +' ('+adr_id+')'
    else:
        headline = 'MedDRA id '+adr_id
    from dtk.html import link
    links = [
            link('MedDRA on bioportal'
                    ,'http://bioportal.bioontology.org/ontologies'
                        '/MEDDRA?p=classes&conceptid='+adr_id
                    ,new_tab=True
                    ),
            ]
    append_padre_files(links,adr_id,[
            ('Cross-validation stats','padre_cv_stats_%s_boxplot.png'),
            ])
    stats_tab = get_padre_stats_table(adr_id)
    return render(request
                ,'nav/adr_info.html'
                ,make_ctx(request,ws,'rvw:review',{
                     'headline':headline,
                     'links':links,
                     'stats_tab':stats_tab,
                     })
                )

class AdrsView(DumaView):
    template_name='nav/adrs.html'
    index_dropdown_stem='rvw:review'
    GET_parms = {
            'config':(str,''),
            'page':(int,1),
            'page_size':(int,25),
            'sort':(SortHandler,'-sev'),
            'filt':(str,''),
            'col_order':(list_of(str),''),
            }
    filt_mode='filt'
    cols_mode='cols'
    def custom_setup(self):
        self.context_alias(
                data_source = PadreDataSource(
                                    self.wsa,
                                    self.col_order,
                                    ),
                )
        from dtk.table import ScoreFilter
        self.context_alias(
                score_filter = ScoreFilter(
                                    self.data_source.active_column_codes(),
                                    self.data_source.ordering,
                                    self.filt,
                                    self.data_source.label,
                                    )
                )
        if self.config == self.filt_mode:
            self.button_map={
                    'filter':['filter','page_size'],
                    }
        elif self.config == self.cols_mode:
            self.button_map={
                    'columns':['columns'],
                    }
    def make_columns_form(self,data):
        FormClass = self.data_source.get_columns_form_class()
        return FormClass(data)
    def columns_post_valid(self):
        p = self.columns_form.cleaned_data
        order = [
                code if p[code] else '-'+code
                for code in self.request.POST['order'].split('|')
                ]
        return HttpResponseRedirect(self.here_url(
                    col_order=','.join(order),
                    config=None,
                    ))
    def make_filter_form(self,data):
        FormClass = self.score_filter.get_filter_form_class()
        return FormClass(data)
    def make_page_size_form(self,data):
        class FormClass(forms.Form):
            page_size = forms.IntegerField(
                    initial=self.page_size,
                    )
        return FormClass(data)
    def filter_post_valid(self):
        self.score_filter.update_filter(
                self.filter_form.cleaned_data
                )
        self.page_size = self.page_size_form.cleaned_data['page_size']
        return HttpResponseRedirect(self.here_url(
                    filt=self.score_filter.get_filter_config(),
                    page_size=self.page_size,
                    page=None,
                    config=None,
                    ))
    def custom_context(self):
        if self.config == self.filt_mode:
            pass
        elif self.config == self.cols_mode:
            pass
        else:
            self.table_context()
    def table_context(self):
        self.context_alias(
                key_filter=KeyFilter(),
                )
        self.score_filter.add_to_key_filter(self.key_filter)
        from dtk.table import IdRowSource
        self.context_alias(
                row_source=IdRowSource(
                            self.data_source.ordering,
                            self.sort,
                            self.key_filter,
                            )
                )
        # set up pager
        from dtk.table import Pager
        self.context_alias(
                pager=Pager(
                        self.here_url,
                        self.row_source.row_count(),
                        self.page_size,
                        self.page,
                        ),
                )
        # get one page of row data
        rows = self.row_source.get_page(
                        self.data_source.value,
                        self.pager,
                        self.data_source.active_column_codes(),
                        )
        ranker = self.row_source.get_ranker()
        # lay out the three fixed columns
        from dtk.table import Table
        from dtk.html import link,tie_icon

        columns = [
                Table.Column('Rank',
                        idx='_pk',
                        cell_fmt=lambda x:format_html('{}{}'
                                                    ,ranker.get(x)
                                                    ,tie_icon(ranker,x)
                                                    ),

                        ),
                Table.Column('medDRA',
                        idx='_pk',
                        cell_fmt=lambda x: link(
                                        x,
                                        self.ws.reverse('nav_adr_info',x),
                                        )
                        ),
                Table.Column('name',
                        idx='_pk',
                        cell_fmt=lambda x: self.data_source.value('name',x)
                        ),
                ]
        # lay out the dynamic columns
        def formatter_factory(code):
            fmt = self.data_source.def_by_code[code][3]
            def formatter(v):
                if v is None:
                    return ''
                return fmt % v
            return formatter
        columns += [
                Table.Column(label,
                        code=code,
                        idx=code,
                        cell_fmt=formatter_factory(code),
                        decimal=True,
                        sort=SortHandler.reverse_mode,
                        )
                for code,label,selected in self.data_source.all_column_specs()
                if selected
                ]
        self.context['main_table']=Table(
                        rows,
                        columns,
                        sort_handler=self.sort,
                        url_builder=self.here_url,
                        )

@login_required
def url_test(request,ws_id):
    ws = Workspace.objects.get(pk=ws_id)
    url_config=UrlConfig(request,
                defaults={
                })
    links = []
    links.append( ('Push',url_config.here_url({
                            'done':url_config.here_url({}),
                            })))
    done = url_config.as_string('done')
    if done:
        links.append( ('Pop',done) )
    return render(request
                ,'nav/placeholder.html'
                ,make_ctx(request,ws,'nav_url_test',{
                     'headline':'URL Testing',
                     'note': '''
                         Test for components that parse URLs and querystrings.
                         ''',
                     'links':links,
                     })
                )

@login_required
def textdiff(request,ws_id):
    ws = Workspace.objects.get(pk=ws_id)
    url_config=UrlConfig(request,
                defaults={
                    'minmatch':0,
                })
    if request.method == 'POST':
        return HttpResponseRedirect(url_config.here_url({
                                'txt1': request.POST['txt1'],
                                'txt2': request.POST['txt2'],
                                'minmatch': request.POST['minmatch'],
                                }))
    txt1 = url_config.as_string('txt1')
    txt2 = url_config.as_string('txt2')
    minmatch=url_config.as_int('minmatch')
    from dtk.text import diffstr
    diff = diffstr(txt1,txt2,
                minmatch=minmatch,
                )
    return render(request
                ,'nav/textdiff.html'
                ,make_ctx(request,ws,'nav_url_test',{
                     'txt1':txt1,
                     'txt2':txt2,
                     'minmatch':minmatch,
                     'diff':diff,
                     })
                )

class SplitAlignView(DumaView):
    index_dropdown_stem='nav_split_align'
    template_name='nav/split_align.html'
    # XXX - add gene-level information?
    # XXX - report items in p2ts/p3ts that aren't in the union of the
    # XXX   corresponding pass/fail/ongo CT sets?
    # XXX - if a split doesn't yet exist, do an align_with_base after
    # XXX   the autosplit?
    def custom_context(self):
        from dtk.kt_split import EditableKtSplit
        self.base_name = 'p2ts'
        self.code2label = dict(self.ws.get_wsa_id_set_choices())
        self.cmp_names = [self.base_name,'p3ts']+[
                x[0]
                for x in self.ws.get_ct_drugset_choices()
                ]
        if True:
            # XXX for now, force to always use MOA drugsets
            self.cmp_names = [
                    'moa-'+x
                    for x in self.cmp_names
                    ]
            self.base_name = self.cmp_names[0]
        self.base_split = EditableKtSplit(self.ws,self.base_name)
        self.all_base_ids = set(
                self.base_split.test_wsa_ids
                + self.base_split.train_wsa_ids
                )
        self.make_table()
    def make_table(self):
        view = self # easy access to view in Row class definition
        from dtk.kt_split import EditableKtSplit
        class Row:
            def __init__(self,ds_name,ws):
                self.ws = ws
                self.ds_code = ds_name
                self.drugset = view.code2label[ds_name]
                self.split = EditableKtSplit(view.ws,ds_name)
                test_ids = set(self.split.test_wsa_ids)
                train_ids = set(self.split.train_wsa_ids)
                all_ids = test_ids | train_ids
                self.size = len(all_ids)
                self.test_drugs = len(test_ids)
                self.train_drugs = len(train_ids)
                to_test,to_train = self.split.get_misaligned(view.base_split)
                self.misaligned_drugs = to_test + to_train
                self.misaligned_prots = self._get_misaligned_prots(test_ids, train_ids)
                self.extra_drugs = all_ids - view.all_base_ids
            def _get_misaligned_prots(self,test_wsas, train_wsas):
                test_prots = self._get_prots(test_wsas)
                train_prots = self._get_prots(train_wsas)
                print(test_prots,train_prots)
                overlap = test_prots & train_prots
                return overlap
            def _get_prots(self,wsas):
                from browse.models import WsAnnotation
                wsa_list = list(WsAnnotation.objects.filter(id__in=wsas))
                from dtk.prot_map import DpiMapping, AgentTargetCache
                targ_cache = AgentTargetCache(
                    mapping=DpiMapping(self.ws.get_dpi_default()),
                    agent_ids=[x.agent_id for x in wsa_list],
                    dpi_thresh=self.ws.get_dpi_thresh_default(),
                )
                prots = set([uniprot for wsa in wsa_list for uniprot,_,_ in targ_cache.info_for_agent(wsa.agent.id)])
                return prots
        def prot_link_list_fmt(data):
            from browse.models import Protein
            uniprot2gene_map = Protein.get_uniprot_gene_map(data)
            urls = [self.ws.reverse('protein',prot) for prot in data]
            genes = [uniprot2gene_map.get(prot, prot) for prot in data]
            from dtk.html import join,link
            return join(sep=', ', *[
                    link(g,u)
                    for g,u in zip(genes,urls)
                    ])
        def annotate_link_list_fmt(data):
            from browse.models import WsAnnotation
            wsa_list = WsAnnotation.objects.filter(id__in=data)
            from dtk.html import join,link
            return join(sep=', ', *[
                    link(wsa.get_name(view.is_demo()),wsa.drug_url())
                    for wsa in wsa_list
                    ])
        def drugset_link_hook(data,row,col):
            from dtk.html import link,join
            return link(data,join(
                    view.ws.reverse('drugset'),
                    '?drugset=',
                    row.ds_code,
                    sep='',
                    ))
        rows = [ Row(x, self.ws) for x in self.cmp_names ]
        from dtk.table import Table
        base_label = self.code2label[self.base_name]
        self.context_alias(table=Table(
                rows,
                [
                        Table.Column('Drugset',
                                cell_html_hook=drugset_link_hook,
                                ),
                        Table.Column('Size',
                                ),
                        Table.Column('Test Drugs',
                                ),
                        Table.Column('Train Drugs',
                                ),
                        Table.Column('Misaligned Drugs',
                                code='misaligned_drugs',
                                cell_fmt=annotate_link_list_fmt,
                                ),
                        Table.Column('Drugs not in '+base_label,
                                code='extra_drugs',
                                cell_fmt=annotate_link_list_fmt,
                                ),
                        Table.Column('Misaligned Proteins',
                                code='misaligned_prots',
                                cell_fmt=prot_link_list_fmt,
                                ),
                        ],
                ))

class DiseaseNameEditView(DumaView):
    template_name='nav/disease_name_edit.html'
    button_map={
            'save':['pattern'],
            }
    def make_pattern_form(self,data):
        initial=self.ws.get_disease_default(self.disease_vocab.name())
        class MyForm(forms.Form):
            pattern = forms.CharField(
                         required=False,
                         initial=initial,
                         widget=forms.TextInput(attrs={'size':'100'}),
                         )
        return MyForm(data)
    def save_post_valid(self):
        p = self.pattern_form.cleaned_data
        self.ws.set_disease_default(
                self.disease_vocab.name(),
                p['pattern'],
                self.username(),
                )
        return HttpResponseRedirect(self.ws.reverse('nav_disease_names'))

class DiseaseNamesView(DumaView):
    template_name='nav/disease_names.html'
    def custom_context(self):
        from dtk.vocab_match import DiseaseVocab
        from dtk.html import link,join,glyph_icon,nowrap
        view = self # so Row methods can access more easily
        class Row:
            def __init__(self,name,dv):
                self.vocabulary=name
                initial,dd=view.ws.get_disease_default(
                                name,
                                return_detail=True,
                                )
                self.pattern=join(
                        initial,
                        link(
                            glyph_icon('pencil'),
                            view.ws.reverse('nav_disease_name_edit',name),
                            ),
                        )
                if dd:
                    if dd.user:
                        from dtk.text import fmt_time
                        self.status = '%s @ %s'%(dd.user,fmt_time(dd.timestamp))
                    else:
                        self.status = 'OK'
                else:
                    self.status='Not explicitly set'
                if dv.lookup_href:
                    self.link=nowrap(link(
                            'lookup',
                            dv.lookup_href,
                            new_tab=True,
                            ))
                else:
                    self.link=link(
                            'browse',
                            view.ws.reverse('nav_ontobrowse',name)
                                    +'?src=nav_disease_names',
                            )
        rows = [
                Row(k,v)
                for k,v in DiseaseVocab.get_subclasses()
                ]
        from dtk.table import Table
        self.context_alias(vocab_table=Table(
                rows,
                [
                        Table.Column('Vocabulary',
                                ),
                        Table.Column('Pattern',
                                ),
                        Table.Column('When_set_and_by_whom',
                                code='status',
                                ),
                        Table.Column('',
                                code='link',
                                ),
                        ],
                ))

class OntoListView(DumaView):
    template_name='nav/ontolist.html'
    GET_parms = {
            'sort':(SortHandler,'indi'),
            'cutoff':(int,None),
            }
    def custom_setup(self):
        if self.disease_vocab.filter_idx is None:
            self.filter_col=''
        else:
            if self.cutoff is None:
                self.cutoff = self.disease_vocab.default_cutoff
            header = self.disease_vocab.header()
            self.filter_col = header[self.disease_vocab.filter_idx]
            self.button_map={
                    'redisplay':['cutoff'],
                    }
    def custom_context(self):
        self.context_alias(page_label=self.disease_vocab.name()+' List')
        vcab_items = self.disease_vocab.items()
        if vcab_items is None:
            msg, vcab_items = self.disease_vocab.no_items()
            self.message(msg)
            return
        unfiltered = [
                (text,code,self.disease_vocab.detail(code))
                for code,text in vcab_items
                ]
        header = self.disease_vocab.header()
        filter_idx = self.disease_vocab.filter_idx
        self.row_source = [
                (text,)+detail
                for text,code,detail in unfiltered
                if filter_idx is None or detail[filter_idx] >= self.cutoff
                ]
        columns = [
                Table.Column('Name',
                        code='name',
                        idx=0,
                        sort='l2h',
                        ),
                ]+[
                Table.Column(hdr,
                        code='det%d'%(i+1),
                        idx=i+1,
                        sort='h2l',
                        )
                for i,hdr in enumerate(self.disease_vocab.header())
                ]
        sort_idx=0
        for x in columns:
            if x.code == self.sort.colspec:
                sort_idx=x.idx
        self.row_source.sort(key=lambda x:x[sort_idx],reverse=self.sort.minus)
        self.context_alias(table=Table(
                self.row_source,
                columns,
                url_builder=self.here_url,
                sort_handler=self.sort,
                ))
    def make_cutoff_form(self, data):
        class MyForm(forms.Form):
            cutoff = forms.IntegerField(
                         required=False,
                         label='Minimum '+self.filter_col,
                         initial=self.cutoff,
                         )
        return MyForm(data)
    def redisplay_post_valid(self):
        p = self.context['cutoff_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))

class OntoBrowseView(DumaView):
    template_name='nav/ontobrowse.html'
    GET_parms={
            'phrase':(str,None),
            'targ_words':(list_of(list_of(str),delim='|'),None),
            'src':(str,None),
            }
    button_map={
            'findwords':['phrase'],
            'findmatch':['wordlist'],
            }
    # Eventually, we could cover more use cases through some combination of
    # generalizing this view, and developing other views using the same tools:
    # - KT search: NDFRT, OB (what is full NDFRT list?)
    # - AACT
    # - AE
    # - GRASP/GWAS
    # - CM parameters: opentargets, faers/cvarod, uphd
    # - CAPP: (link faers to DisGeNet -- both mapped to MedDRA off-line)
    # Non-disease applications are also possible.
    #
    # Some thoughts:
    # - for OB in particular, where the targets are long strings, the scoring
    #   prefers shorter items even if they're less correct. For example, in a
    #   search for "type 2 diabetes",
    #     "treatment of gaucher disease type 1"
    #   gets ranked above
    #     "treatment of type 2 diabetes by administering bromocriptine
    #      mesylate and a first phase insulin secretagogue wherein the
    #      combined therapeutic effect is greater than the additive effect
    #      of administering each agent alone"
    #   I assume what's happening is the first has 1 of 6 words correct, where
    #   the latter has 4 of 31, a lower ratio. In this case what's more
    #   important is the fraction of probe words matched, not the fraction
    #   of probe words in the target.
    def custom_setup(self):
        if not self.phrase:
            self.phrase = self.ws.name
        from dtk.vocab_match import VocabMatcher
        vcab_items = self.disease_vocab.items()
        # XXX right now the only the vocab that uses this is DisGeNet,
        # XXX but others may want to adopt it in the future.
        if vcab_items is None:
            msg, vcab_items = self.disease_vocab.no_items()
            self.message(msg)
        self.vm = VocabMatcher(list(vcab_items))
        self.word_map = self.vm.map_words(self.phrase)
        if self.targ_words:
            # XXX There are lots of potential use cases for this page with
            # XXX different disease vocabularies and contexts. Maybe all
            # XXX the selection code below could be delegated in a more
            # XXX flexible way, with different implementations for different
            # XXX use cases.
            # XXX
            # XXX For example, OB KT search could list drugs rather than use
            # XXX codes, with relevant use codes grouped under each drug.
            self.matches = [
                    (score,text,code,self.disease_vocab.detail(code))
                    for score,text,code
                    in self.vm.score_phrases(self.targ_words)
                    ]
            filt_idx = self.disease_vocab.filter_idx
            if filt_idx is not None:
                # re-sort with filter column as a minor key
                self.matches.sort(
                        key=lambda x:(x[0],x[3][filt_idx]),
                        reverse=True,
                        )
            if self.matches and self.src:
                # src being set indicates we should support a postback
                if self.disease_vocab.multi_select:
                    # template implements a wrapper form, each row
                    # supplies a checkbox
                    self.select_multiple = True
                    self.button_map = dict(self.button_map)
                    self.button_map['multi_select'] = ['match_check']
                else:
                    # one button per entry, rendered entirely in template
                    self.select_single = True
                    self.button_map = dict(self.button_map)
                    self.button_map['select'] = []
    cb_prefix = 'checkbox_'
    tw_cb_prefix = 'tw_checkbox_'
    def custom_context(self):
        from dtk.html import bulk_update_links
        kwargs=dict(tw_bulk_links=bulk_update_links(self.tw_cb_prefix,attr='id'))
        if hasattr(self,'select_multiple'):
            # re-write self.matches to include match_check form field per row
            self.matches = [
                    row+(self.match_check_form[row[2]],)
                    for row in self.matches
                    ]
            kwargs['bulk_links']=bulk_update_links(self.cb_prefix,attr='id')
        self.context_alias(**kwargs)
    def make_match_check_form(self,data):
        from dtk.dynaform import FormFactory
        ff = FormFactory()
        for score,text,code,detail in self.matches:
            ff.add_field(
                    code,
                    forms.BooleanField(
                            label='',
                            required=False,
                            )
                    )
        MyForm = ff.get_form_class()
        return MyForm(data,auto_id=self.cb_prefix+'%s')
    def multi_select_post_valid(self):
        p = self.context['match_check_form'].cleaned_data
        code_list = [
                k
                for k,v in p.items()
                if v
                ]
        if code_list:
            pattern = self.disease_vocab.build_pattern(code_list)
            self.ws.set_disease_default(
                        self.disease_vocab.name(),
                        pattern,
                        self.username(),
                        )
            # XXX (eventually support save to session, controlled
            # XXX by a second GET_parm)
        return HttpResponseRedirect(self.ws.reverse(self.src))
    def select_post_valid(self):
        row_val = self.request.POST['row_val']
        pattern = self.disease_vocab.build_pattern(row_val)
        self.ws.set_disease_default(
                    self.disease_vocab.name(),
                    pattern,
                    self.username(),
                    )
        # XXX (eventually support save to session, controlled
        # XXX by a second GET_parm)
        return HttpResponseRedirect(self.ws.reverse(self.src))
    def make_phrase_form(self,data):
        class MyForm(forms.Form):
            phrase = forms.CharField(
                max_length=512,
                initial=self.phrase,
                )
        return MyForm(data)
    def findwords_post_valid(self):
        p = self.context['phrase_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(targ_words=None,**p))
    def make_wordlist_form(self,data):
        from dtk.dynaform import FormFactory
        ff = FormFactory()
        from dtk.html import WrappingCheckboxSelectMultiple
        for idx,(src,targs) in enumerate(self.word_map):
            choices=list(enumerate(targs))
            if self.targ_words:
                initial = [
                        x[0]
                        for x in choices
                        if x[1] in self.targ_words[idx]
                        ]
            elif src in targs:
                initial = [
                        x[0]
                        for x in choices
                        if x[1] == src
                        ]
            else:
                initial = None
            ff.add_field(
                    src,
                    forms.MultipleChoiceField(
                            label=src,
                            choices=choices,
                            initial=initial,
                            widget=WrappingCheckboxSelectMultiple,
                            required=False,
                            ),
                    )
        MyForm = ff.get_form_class()
        return MyForm(data,auto_id=self.tw_cb_prefix+'%s')
    def findmatch_post_valid(self):
        p = self.context['wordlist_form'].cleaned_data
        result = []
        for src,targs in self.word_map:
            idxs = [int(x) for x in p[src]]
            targ_words = [targs[x] for x in idxs]
            result.append(','.join(targ_words))
        return HttpResponseRedirect(self.here_url(
                targ_words='|'.join(result),
                ))

class DiseaseSumView(DumaView):
    template_name='nav/disease_sum.html'
    def custom_context(self):
        self.vdefaults=self.ws.get_versioned_file_defaults()
        self.build_disease_table()
        self.build_agr_table()
        self.build_pheno_table()
    def build_disease_table(self):
        from dtk.table import Table
        rows = []
        from dtk.vocab_match import DiseaseVocab
        for key,Subclass in DiseaseVocab.get_subclasses():
            if not hasattr(Subclass,'disease_info_link'):
                # some of the subclasses are slow to instantiate, so only
                # do it if there's disease info available
                continue
            vm = Subclass(version_defaults=self.vdefaults)
            pattern,dd=self.ws.get_disease_default(key,return_detail=True)
            if not dd:
                rows.append((vm.name(),'disease name not confirmed'))
                continue
            for code in vm.pattern_elements(pattern):
                try:
                    rows.append((vm.name(),vm.disease_info_link(code)))
                except KeyError:
                    self.message(f"Error finding making link for {key} - '{code}'")
        self.context_alias(disease_table=Table(
                rows,
                [
                        Table.Column('Vocabulary',
                                idx=0,
                                ),
                        Table.Column('Disease Designation',
                                idx=1,
                                ),
                        ],
                ))
    def build_agr_table(self):
        file_class = 'agr'
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned(
                file_class,
                self.vdefaults[file_class],
                role='models',
                )
        s3f.fetch()
        patterns=self.ws.get_disease_default('AGR').split('|')
        lookup={}
        from dtk.files import get_file_records
        from collections import Counter
        for rec in get_file_records(
                s3f.path(),
                keep_header=False,
                select=(patterns,1),
                ):
            l = lookup.setdefault(rec[1],[rec[0],Counter()])
            l[1][rec[2]] += 1
        rows = [
                [doid]+lookup[doid]
                for doid in patterns
                if doid in lookup
                ]
        rows.sort(key=lambda x:sum(x[2].values()), reverse=True)
        def link_fmt(s):
            from dtk.html import link
            from dtk.url import agr_url
            return link(s, agr_url(s), new_tab=True)
        def count_fmt(ctr):
            return '; '.join([k+':'+str(v) for k,v in ctr.most_common()])
        from dtk.table import Table
        self.context_alias(agr_table=Table(
                rows,
                [
                        Table.Column('Disease Name',
                                idx=1,
                                ),
                        Table.Column('AGR page',
                                idx=0,
                                cell_fmt=link_fmt,
                                ),
                        Table.Column('Model Counts',
                                idx=2,
                                cell_fmt=count_fmt,
                                ),
                        ],
                ))
    def build_pheno_table(self):
        file_class = 'monarch'
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned(
                file_class,
                self.vdefaults[file_class],
                role='disease',
                )
        s3f.fetch()
        patterns=self.ws.get_disease_default('Monarch').split('|')
        lookup={}
        from dtk.files import get_file_records
        from collections import Counter
        for rec in get_file_records(
                s3f.path(),
                keep_header=False,
                select=(patterns,0),
                ):
            if rec[0] not in lookup:
                lookup[rec[0]]=[]
            lookup[rec[0]].append(rec[1:])
        rows = [
                [id]+x
                for id in patterns
                if id in lookup
                for x in lookup[id]
                ]
        rows.sort()
        def pheno_link_fmt(s):
            from dtk.html import link
            from dtk.url import monarch_pheno_url
            return link(s, monarch_pheno_url(s), new_tab=True)
        def disease_link_fmt(s):
            from dtk.html import link
            from dtk.url import monarch_disease_url
            return link(s, monarch_disease_url(s), new_tab=True)
        def evid_fmt(s):
            from dtk.s3_cache import S3File
            file_class='monarch'
            s3f = S3File.get_versioned(
                    file_class,
                    self.vdefaults[file_class],
                    role='evidence',
                )
            s3f.fetch()
            patterns=s.split('|')
            return " | ".join([rec[1] for rec in get_file_records(
                    s3f.path(),
                    keep_header=False,
                    select=(patterns,0),
                    )])
        from dtk.table import Table
        self.context_alias(pheno_table=Table(
                rows,
                [
                        Table.Column('Monarch page',
                                idx=0,
                                cell_fmt=disease_link_fmt,
                                ),
                        Table.Column('Disease Name',
                                idx=1,
                                ),
                        Table.Column('Relation',
                                idx=4,
                                ),
                        Table.Column('Phenotypes',
                                idx=3,
                                ),
                        Table.Column('Phenotype page',
                                idx=2,
                                cell_fmt=pheno_link_fmt,
                                ),
                        Table.Column('Frequency',
                                idx=5,
                                ),
                        Table.Column('Onset',
                                idx=6,
                                ),
                        Table.Column('Evidence',
                                idx=7,
                                cell_fmt=evid_fmt,
                                ),
                        ],
                ))

# This provides bulk pre-screening, so it lives with scoreboards and
# prescreens, rather than in rvw
class RvwClustersView(DumaView):
    template_name='nav/rvw_clusters.html'
    from dtk.duma_view import boolean
    GET_parms={
            'save':(boolean,False),
            'sort':(SortHandler, 'reference'),
            'job_id':(int,None),
            'score':(str,None),
            'start':(int,0),
            'count':(int,200),
            'dpi':(str,None),
            'dpi_threshold':(float,0.5),
            'ppi':(str,None),
            'ppi_threshold':(float,0.9),
            'repulsion':(float,0.5),
            'damping':(float,0.8),
            'max_iter':(int,1000),
            'method':(str,'ST'),
            'st_dir_thresh':(float,0.7),
            'st_ind_thresh':(float,0.2),
            }
    button_map={
            'redisplay':['config'],
            }
    def custom_context(self):
        self.settings={'ws_id':self.ws.id}
        for key in self.GET_parms:
            if key in ['sort', 'save']:
                continue
            if not hasattr(self,key):
                return
            value = getattr(self,key)
            if value is None:
                return
            self.settings[key] = value
        self.setup()
        if self.save:
            self.flgr.flag_drugs()
        else:
            self.extract_similarity_and_clusters()
        from dtk.similarity import diff_clusters
        self.diff = diff_clusters(
                self.flgr.do_cluster(self.flgr.sm.clusters2()),
                self.flgr.group2members,
                )
        self.build_table()
    def make_config_form(self,data):
        class MyForm(forms.Form):
            job_id = forms.IntegerField(
                    label='Drug Ordering Job',
                    initial=self.job_id,
                    )
            score = forms.CharField(
                    label='Drug Ordering Score',
                    initial=self.score,
                    )
            start = forms.IntegerField(
                    label='Initial Drugs to skip',
                    initial=self.start,
                    )
            count = forms.IntegerField(
                    label='Drugs to examine',
                    initial=self.count,
                    )
            from dtk.prot_map import DpiMapping
            dpi = forms.ChoiceField(
                    label='DPI mapping',
                    choices=DpiMapping.choices(self.ws),
                    initial=self.dpi or self.ws.get_dpi_default(),
                    )
            dpi_threshold = forms.FloatField(
                    label='DPI evidence threshold',
                    initial=self.ws.get_dpi_thresh_default()
                            if self.dpi_threshold is None else self.dpi_threshold,
                    )
            from dtk.prot_map import PpiMapping
            ppi = forms.ChoiceField(
                    label='PPI mapping',
                    choices=PpiMapping.choices(),
                    initial=self.ppi or self.ws.get_ppi_default(),
                    )
            ppi_threshold = forms.FloatField(
                    label='PPI evidence threshold',
                    initial=self.ws.get_ppi_thresh_default()
                            if self.ppi_threshold is None else self.ppi_threshold,
                    )
            method = forms.ChoiceField(
                    label='Clustering method',
                    choices=[
                            ('AP','Affinity Propagation'),
                            ('ST','Similarity Threshold'),
                            ],
                    initial=self.method,
                    )
            st_dir_thresh = forms.FloatField(
                    label='Similarity Threshold cutoff for direct targets',
                    initial=self.st_dir_thresh,
                    )
            st_ind_thresh = forms.FloatField(
                    label='Similarity Threshold cutoff for indirect targets',
                    initial=self.st_ind_thresh,
                    )
            repulsion = forms.FloatField(
                    label='Affinity Propagation repulsion',
                    initial=self.repulsion,
                    )
            damping = forms.FloatField(
                    label='Affinity Propagation damping',
                    initial=self.damping,
                    )
            max_iter = forms.IntegerField(
                    label='Affinity Propagation max iterations',
                    initial=self.max_iter,
                    )
            save = forms.BooleanField(
                    label='Save cluster results as a flag',
                    initial=False,
                    required=False
                    )
        return MyForm(data)
    def redisplay_post_valid(self):
        p = self.context['config_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))
    def _cluster_uniprots(self,c):
        uniprots = set()
        for wsa_id in c:
            uniprots |= self.flgr.wsa2dpi[int(wsa_id)]
        return uniprots
    def build_table(self):
        from dtk.html import link,join
        from dtk.duma_view import qstr
        from browse.models import Protein
        from django.utils.safestring import mark_safe
        ugm = Protein.get_uniprot_gene_map()
        def fmt_ind_cluster(k):
            if k == 'none' or k is None:
                return ''
            all_drugs = [w for dk
                         in self.flgr.meta_group2members[k]
                         for w in self.flgr.group2members[int(dk)]
                        ]
            return link(
                    k,
                    self.ws.reverse('clust_screen')+qstr({},
                            dpi_t=self.settings['dpi_threshold'],
                            dpi=self.settings['dpi'],
                            ids=','.join([str(x) for x in all_drugs]),
                            ),
                    new_tab=True,
                    )
        def fmt_cluster(c):
            if not c:
                return ''
            return link(
                    str(len(c)),
                    self.ws.reverse('clust_screen')+qstr({},
                            dpi_t=self.settings['dpi_threshold'],
                            dpi=self.settings['dpi'],
                            ids=','.join([str(x) for x in c]),
                            ),
                    new_tab=True,
                    )
        def fmt_uniprot(c):
            if c is None:
                return ''
            uniprots = [
                    (p,ugm.get(p,'(%s)'%p))
                    for p in self._cluster_uniprots(c)
                    ]
            uniprots.sort(key=lambda x:x[1])
            return join(*[
                    link(g,self.ws.reverse('protein',p))
                    for p,g in uniprots
                    ])
        def fmt_diff(pair):
            c1,c2,_,_=pair
            if c1 is None:
                 c1 = set()
            if c2 is None:
                return 'Entire cluster merged'
            if c1 == c2:
                return 'No change'
            diffs = []
            if c2-c1:
                diffs.append(mark_safe('added %d'%len(c2-c1)))
            if c1-c2:
                diffs.append(mark_safe('removed %d'%len(c1-c2)))
            u1=self._cluster_uniprots(c1)
            u2=self._cluster_uniprots(c2)
            if u2-u1:
                uniprots = [
                        (p,ugm.get(p,'(%s)'%p))
                        for p in u2-u1
                        ]
                uniprots.sort(key=lambda x:x[1])
                diffs.append(join(mark_safe('new genes'),*[
                    link(g,self.ws.reverse('protein',p))
                    for p,g in uniprots
                    ]))
            return join(*diffs,sep='; ')
        from dtk.table import Table,SortHandler
        columns = [
                Table.Column('Indirect cluster',
                        idx=3,
                        code='indirect_cluster',
                        sort='l2h',
                        cell_fmt=fmt_ind_cluster,
                        ),
                Table.Column('Direct cluster',
                        idx=2,
                        code='direct_cluster',
                        sort='l2h',
                        ),
                Table.Column('Ref uniprots',
                        idx=0,
                        cell_fmt=fmt_uniprot,
                        ),
                Table.Column('Reference',
                        idx=0,
                        code='reference',
                        cell_fmt=fmt_cluster,
                        sort='l2h',
                        ),
                Table.Column('Actual',
                        idx=1,
                        code='actual',
                        cell_fmt=fmt_cluster,
                        sort='l2h',
                        ),
                Table.Column('Difference',
                        extract=lambda x:x,
                        cell_fmt=fmt_diff,
                        ),
                ]
        for x in columns:
            if x.code == self.sort.colspec:
                sort_idx=x.idx
        rows = []
        for c1,c2,k in self.diff:
            outer_k = None
            if c2 is None and len(c1) == 1:
                # don't show unique items that got merged away
                continue
            if c1 == c2 and len(c1) == 1:
                # don't show unique items that are unchanged
                continue
            for ok,s in six.iteritems(self.flgr.meta_group2members):
                if str(k) in s:
                    outer_k = ok
                    break
            if c1 is None:
                c1 = set()
            rows.append((c1,c2,k,outer_k))
        rows.sort(key=lambda x:x[sort_idx],
                                 reverse=self.sort.minus
                                 )
        self.context['table']=Table(
                        rows,
                        columns,
                        url_builder=self.here_url,
                        sort_handler=self.sort
                        )
    def extract_settings(self):
        from flagging.models import FlagSet
        fs=FlagSet.objects.get(id=self.flagging_set)
        import json
        self.settings = json.loads(fs.settings)
    def extract_similarity_and_clusters(self):
        self.flgr.setup()
        self.flgr.build_clusters()
        if self.method == 'AP' and 'Did not converge' in self.flgr.sm.output_trace:
            self.message('Did not converge')
    def setup(self):
        from scripts.flag_drugs_for_review_similarity import Flagger
        self.flgr = Flagger(**self.settings)
    def extract_recorded_clusters(self):
        from flagging.models import Flag
        clusters = {}
        for flag in Flag.objects.filter(run_id=self.flagging_set):
            qparms = dict([
                    pair.split('=')
                    for pair in flag.href.split('?')[1].split('&')
                    ])
            wsa_ids = [int(x) for x in qparms['ids'].split('%2C')]
            if flag.detail in clusters:
                assert set(wsa_ids) == clusters[flag.detail]
            else:
                clusters[flag.detail] = set(wsa_ids)
        self.clusters=list(clusters.values())
        self.clusters.sort(key=lambda x:len(x),reverse=True)


# XXX This view has been removed from the menu, because we no longer use
# XXX the 'struct' CM that drives it. It could be re-instated if there
# XXX were some other source for a KT similarity matrix. Since the number
# XXX of KTs is relatively small, this could probably be done on the fly,
# XXX maybe providing the option of protein or jaccard similarity.
class KtClustersView(DumaView):
    template_name='nav/kt_clusters.html'
    index_dropdown_stem='nav_kt_clusters'
    GET_parms={
            'struct_job':(int,None),
            'thresh':(float,0.5),
            }
    button_map={
            'redisplay':['config'],
            }
    def custom_setup(self):
        from runner.process_info import JobInfo
        ubi = JobInfo.get_unbound('struct')
        names = ubi.get_jobnames(self.ws)
        from dtk.text import fmt_time
        from runner.models import Process
        self.struct_choices = [
                (p.id,'%s %d %s'%(
                        p.job_type(),
                        p.id,
                        fmt_time(p.completed),
                ))
                for p in Process.objects.filter(
                        name__in=names,
                        status=Process.status_vals.SUCCEEDED,
                        ).order_by('-id')
                ]
        if self.struct_choices and not self.struct_job:
            self.struct_job = self.struct_choices[0][0]
    def custom_context(self):
        if self.struct_job:
            self.build_cluster_table()
        else:
            self.message('No structure jobs available')
    def make_config_form(self,data):
        class MyForm(forms.Form):
            struct_job = forms.ChoiceField(
                choices = self.struct_choices,
                )
            thresh = forms.DecimalField(
                initial = self.thresh,
                max_value = 1,
                min_value = 0,
                decimal_places = 1,
                )
        return MyForm(data)
    def redisplay_post_valid(self):
        p = self.context['config_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))
    def build_cluster_table(self):
        from runner.process_info import JobInfo
        bji=JobInfo.get_bound(self.ws,self.struct_job)
        sm = bji.get_kt_similarity_matrix()
        if not sm:
            self.message("No similarity data for job %d"%self.struct_job)
            return
        clusters = sm.clusters(self.thresh)
        m = sm.matrix
        treatments = sm.row_keys
        # build description of each cluster
        import numpy as np
        from dtk.num import median,avg
        rows = []
        class Dummy: pass
        for members in clusters:
            row = Dummy()
            rows.append(row)
            # choose exemplar(s) (max avg similarity with other members)
            mask = [x in members for x in treatments]
            interior = m[mask,:][:,mask]
            ordered_labels = [x for x in treatments if x in members]
            centrality = [avg(x) for x in interior]
            row.exemplar = [
                    label
                    for score,label in zip(centrality,ordered_labels)
                    if score == max(centrality)
                    ]
            # get measure of similarity within cluster
            row.cohesion = median(interior.ravel())
            # list other members
            row.others = members-set(row.exemplar)
            # find nearest drug(s) outside cluster
            # XXX show which interior drug it's paired with?
            # XXX show all drugs inside some threshold?
            # XXX note that one exterior drug could pair with multiple
            # XXX members above a threshold
            non_members = set(treatments)-members
            nearness = [
                    (label,max(m[mask,:][:,treatments.index(label)]))
                    for label in non_members
                    ]
            nearness.sort(key=lambda x:x[1],reverse=True)
            row.outside = nearness[:1]
        # render output
        from browse.models import WsAnnotation
        wsa_map = {
                wsa.id:wsa
                for wsa in WsAnnotation.objects.filter(
                        id__in=[int(x) for x in treatments]
                        )
                }
        from dtk.html import join,link
        from django.utils.safestring import mark_safe
        def format_ids(l):
            wsas = [wsa_map[x] for x in l]
            return join(*[
                    link(wsa.get_name(self.is_demo()),wsa.drug_url())
                    for wsa in wsas
                    ],sep=mark_safe('<br>'))
        def format_ids_plus_similarity(l):
            wsa_list = [(wsa_map[key],v) for key,v in l]
            return join(*[
                    join(
                        link(wsa.get_name(self.is_demo()),wsa.drug_url()),
                        '%.4f'%v,
                        )
                    for wsa,v in wsa_list
                    ],sep=mark_safe('<br>'))
        from dtk.table import Table
        columns = [
                Table.Column('Exemplar',
                        cell_fmt=format_ids,
                        ),
                Table.Column('Other Members',
                        code='others',
                        cell_fmt=format_ids,
                        ),
                Table.Column('Median Internal Similarity',
                        code='cohesion',
                        cell_fmt=lambda x:'%.4f'%x,
                        ),
                Table.Column('Nearest Outside Treatment',
                        code='outside',
                        cell_fmt=format_ids_plus_similarity,
                        ),
                ]
        self.context['table']=Table(
                        rows,
                        columns,
                        )

class ImpStatView(DumaView):
    template_name='nav/impstat.html'
    index_dropdown_stem='nav_impstat'
    GET_parms={
            'full':(boolean,False),
            }
    def custom_context(self):
        from browse.models import WsCollectionInfo
        self.context_alias(
                wci = WsCollectionInfo(self.ws.id),
                )
        self.valid_molecules = sum(self.wci.ws_agent_counts.values())
        self.find_wsa_miscounts()
        self.find_cluster_mates()
        self.build_collection_table()
    def find_cluster_mates(self):
        wci = self.wci
        ws_collections = wci.coll_to_ws_agents.keys()
        from browse.models import WsAnnotation
        cluster_reps=[]
        for s in wci.clust_keys_for_collection_set(ws_collections).values():
            clust_agents = s & wci.ws_agent_id_set
            if len(clust_agents) > 1:
                cluster_reps.append(next(iter(clust_agents)))
        qs = WsAnnotation.objects.filter(
                ws=self.ws,
                agent_id__in=cluster_reps,
                )
        self.context_alias(multi_wsa_clusters = self.prep_wsas(qs))
        self.multi_wsa_clust_count=len(self.multi_wsa_clusters)
    def prep_wsas(self,qs):
        from browse.models import WsAnnotation
        qs = WsAnnotation.prefetch_agent_attributes(qs)
        for wsa in qs:
            wsa.prepared_name = wsa.get_name(self.is_demo())
        return qs
    def find_wsa_miscounts(self):
        from browse.models import WsAnnotation
        ws_qs = WsAnnotation.objects.filter(ws=self.ws)
        self.wsa_count = ws_qs.count()
        removed_qs = ws_qs.filter(agent__removed=True)
        self.removed_count = removed_qs.count()
        self.context_alias(removed_ind_agent_wsas = self.prep_wsas(
                removed_qs.exclude(indication=0)
                ))
        self.removed_ind_count = len(self.removed_ind_agent_wsas)
        self.duplicate_count = (
                self.wsa_count - self.valid_molecules - self.removed_count
                )
            # XXX if the above is non-zero, we can retrieve the actual groups
            # XXX to render links; prep_wsas might be useful for retrieval,
            # XXX but we'd need to restructure the output to display
            # XXX each duplicated group separately
    def build_collection_table(self):
        wci = self.wci
        rows = [c
                for c in wci.all_collections
                if self.full or c.id in wci.coll_to_ws_agents.keys()
                ]
        for c in rows:
            c.import_counts = wci.import_counts(c)
        from dtk.table import Table
        cols = [
                Table.Column('Collection',
                        code='name',
                        ),
                Table.Column('In use',
                        extract=lambda c:wci.count_used_drugs(c),
                        ),
                Table.Column('Other imported',
                        extract=lambda c:wci.count_unused_drugs(c),
                        ),
                Table.Column('All imported',
                        extract=lambda c:c.import_counts[1],
                        ),
                Table.Column('Blocked',
                        extract=lambda c:c.import_counts[3],
                        ),
                Table.Column('Available',
                        extract=lambda c:c.import_counts[2],
                        ),
                Table.Column('Total in collection',
                        extract=lambda c:c.import_counts[0],
                        ),
                ]
        self.collections_table = Table(rows,cols)

class Col2View(DumaView):
    template_name='nav/col2.html'
    index_dropdown_stem='nav_col2'
    button_map={
            'imports':['collections'],
            'clear':['collections'],
            }
    require_version = True
    def custom_setup(self):
        # verify needed cluster info has been uploaded
        from browse.models import WsCollectionInfo
        wci = WsCollectionInfo(self.ws.id)
        from drugs.tools import CollectionUploadStatus
        self.version_ok = False
        # the code only support drug clustering for versioned DPI files;
        # prevent imports that won't do the right thing
        if not wci.version:
            self.message(f'Import not allowed; DPI default must be versioned')
            logger.info(f'Import not allowed; DPI default must be versioned')
        else:
            us = CollectionUploadStatus()
            missing_versions = [x.version for x in us.needed_clusters]
            if wci.version in missing_versions:
                self.message(f'Import not allowed; version {wci.version} clusters have not been uploaded')
                logger.info(f'Import not allowed; version {wci.version} clusters have not been uploaded')
            else:
                self.version_ok = True
        # gather counts
        ws_counts = wci.ws_agent_counts
        all_counts = wci.all_agent_counts
        self.all_collections = wci.all_collections
        for c in self.all_collections:
            c.all_count = all_counts.get(c.id,0)
            c.ws_count = ws_counts.get(c.id,0)
            c.out_count = c.all_count - c.ws_count
            c.form_field_name = 'col%d'%c.id
    def custom_context(self):
        self.field_with_col = zip(self.collections_form,self.all_collections)
    def make_collections_form(self,data):
        from dtk.dynaform import FormFactory
        ff = FormFactory()
        from drugs.models import Collection
        default_collections = Collection.default_collections
        for c in self.all_collections:
            is_default = c.name in default_collections
            ff.add_field(c.form_field_name,forms.BooleanField(
                    label=c.name + (' (*)' if is_default else ''),
                    required=False,
                    initial=is_default,
                    ))
        FormClass = ff.get_form_class()
        return FormClass(data)
    def imports_post_valid(self):
        return self.update_workspace(True)
    def clear_post_valid(self):
        return self.update_workspace(False)
    def update_workspace(self,reimport):
        # Guard against problematic import. The assert should never happen
        # because the template removes the submit button in this case.
        if self.require_version:
            assert self.version_ok
        p = self.context['collections_form'].cleaned_data
        import_collections = [
            c
            for c in self.all_collections
            if p[c.form_field_name]
            ]
        from browse.models import WsCollectionInfo
        wci = WsCollectionInfo(self.ws.id)
        for c in import_collections:
            wci.clear_unused_drugs(c,self.username())
        if reimport:
            for c in import_collections:
                self.ws.import_collection(c,self.username())
        return HttpResponseRedirect(self.here_url())

class ProtsetForm(forms.Form):
    name = forms.CharField(
            max_length=60,
            required=True,
            )
    description = forms.CharField(
            label="Description",
            widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
            required=False,
            )
    proteins = forms.CharField(
            label="Enter uniprot ids, one per line",
            widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
            required=True,
            help_text="Anything on a line after a # character is ignored, preloading will display gene names as such.",
            )

class PsFromDsForm(forms.Form):
    from dtk.prot_map import DpiMapping
    drugset = forms.ChoiceField(
                    required=True,
                    )
    mapping = forms.ChoiceField(
                    required=True,
                    )
    threshold = forms.FloatField(
                    )
    def __init__(self, ws, *args, **kwargs):
        super(PsFromDsForm,self).__init__(*args, **kwargs)
        from dtk.prot_map import DpiMapping
        # reload choices on each form load
        self.fields['drugset'].choices = ws.get_wsa_id_set_choices()
        self.fields['threshold'].initial = ws.get_dpi_thresh_default()
        f = self.fields['mapping']
        f.choices = DpiMapping.choices(ws)
        f.initial = ws.get_dpi_default()

class PsCmpForm(forms.Form):
    from dtk.prot_map import PpiMapping
    protset1 = forms.ChoiceField(
                    required=True,
                    )
    protset2 = forms.ChoiceField(
                    required=True,
                    )
    mapping = forms.ChoiceField(
                    label='PPI dataset',
                    required=True,
                    )
    min_ppi_evid = forms.FloatField(
                    label='Min PPI evidence',
                    initial = PpiMapping.default_evidence,
                    )
    path_n = forms.IntegerField(
                    label="Minimum # of proteins overlapping a pathway",
                    initial = 2,
                    )
    topN = forms.IntegerField(
                    label='Pathways per table to show',
                    initial = 10,
                    )
    def __init__(self, ws, *args, **kwargs):
        super(PsCmpForm,self).__init__(*args, **kwargs)
        from dtk.prot_map import PpiMapping
        # reload choices on each form load
        choices = ws.get_uniprot_set_choices()
        self.fields['protset1'].choices = choices
        self.fields['protset2'].choices = choices
        f = self.fields['mapping']
        f.choices = PpiMapping.choices()
        from browse.default_settings import PpiDataset
        f.initial = PpiDataset.value(ws=ws)

@login_required
def ps_cmp(request,ws_id):
    ws = Workspace.objects.get(pk=ws_id)
    url_config=UrlConfig(request,
                defaults={
                })
    protset1 = url_config.as_string('protset1')
    protset2 = url_config.as_string('protset2')
    mapping = url_config.as_string('mapping')
    min_path_entries = url_config.as_int('path_n')
    topN = url_config.as_int('topN')
    min_ppi_evid = url_config.as_float('min_ppi_evid')
    from dtk.prot_map import PpiMapping
    if protset1 or protset2 or mapping:
        ps_cmp_form = PsCmpForm(ws,{
                    'protset1':protset1,
                    'protset2':protset2,
                    'mapping':mapping,
                    'path_n':min_path_entries,
                    'topN':topN,
                    'min_ppi_evid':min_ppi_evid,
                    })
    else:
        ps_cmp_form = PsCmpForm(ws)
    from dtk.data import merge_dicts,dict_subset
    if request.method == 'POST':
        if 'display_btn' in request.POST:
            try:
                return HttpResponseRedirect(
                                url_config.here_url(merge_dicts(
                                        dict_subset(request.POST,[
                                                'protset1',
                                                'protset2',
                                                'mapping',
                                                'path_n',
                                                'topN',
                                                'min_ppi_evid',
                                                ]),
                                        ))
                                )
            except KeyError:
                ps_cmp_form = PsCmpForm(ws,request.POST)
    summaries = []
    png_plots = []
    pathway_enrichments = []
    if protset1 and protset2 and mapping:
        from path_helper import PathHelper,make_directory
        import os, operator
        from browse.models import Protein
        ps_names = dict(ws.get_uniprot_set_choices())
        ps1_name = ps_names[protset1]
        ps2_name = ps_names[protset2]
        pm = PpiMapping(mapping)
        ps1 = ws.get_uniprot_set(protset1)
        ps2 = ws.get_uniprot_set(protset2)
        path = PathHelper.ws_publish(ws.id)+'psCmp/'
        make_directory(path)
        base = os.path.join(path, "_".join([ps1_name, ps2_name]))
        total_prot_cnt = Protein.objects.count()
        summaries.append(overlap_description(
                         'proteins'
                         ,ps1
                         ,ps1_name
                         ,ps2
                         ,ps2_name
                         ,total_prot_cnt
                        ))
        dir_file = base + "_direct.png"
        plot_venn('Protein overlap'
                  ,ps1
                  ,ps1_name
                  ,ps2
                  ,ps2_name
                  ,dir_file
                  )
        png_plots.append(PathHelper.url_of_file(dir_file))
        # get neighbor protein info for all proteins
        prot2prot = {u:set() for u in list(ps1|ps2)}
        all_ppi_gen = ((r[0], r[1]) for r in pm.get_ppi_info_for_keys(prot2prot)
                       if float(r[2]) >= min_ppi_evid
                      )
        for prot1, prot2 in all_ppi_gen:
            try:
                prot2prot[prot1].add(prot2)
            except KeyError:
                pass
            try:
                prot2prot[prot2].add(prot1)
            except KeyError:
                pass
        all_proteins = reduce(operator.or_, prot2prot.values(), set())
        # calculate prevalence among partners and non-partners
        prev_list = [
                (
                    prot,
                    sum([prot in prot2prot[x] for x in ps1]),
                    sum([prot in prot2prot[x] for x in ps2]),
                )
                for prot in all_proteins
                ]
        ips1 = set([x[0] for x in prev_list if x[1]])
        ips2 = set([x[0] for x in prev_list if x[2]])
        summaries.append(overlap_description('indirect proteins, evid >= ' + str(min_ppi_evid),
                ips1,
                ps1_name,
                ips2,
                ps2_name,
                pm.get_uniq_target_cnt()
                ))
        indir_file = base + "_indirect.png"
        png_plots.append(PathHelper.url_of_file(indir_file))
        plot_venn('PPI-neighbor overlap'
                  ,ips1
                  ,ps1_name
                  ,ips2
                  ,ps2_name
                  ,indir_file
                  )


        from dtk.gene_sets import get_gene_set_file, get_pathway_prot_mm
        from browse.default_settings import GeneSets
        gsf = get_gene_set_file(GeneSets.value(ws=None))
        pwy_prot_mm = get_pathway_prot_mm(gsf.path())

        # get pathways for the direct prots
        prot2path = {
                uni:pwy_prot_mm.rev_map().get(uni, set())
                for uni in list(ps1|ps2)
                }
        all_paths = reduce(operator.or_, prot2path.values(), set())
        # calculate prevalence among partners and non-partners
        psX = ps2 & ps1
        psU1 = ps1 - ps2
        psU2 = ps2 - ps1
        path_prev_list = [
                (
                    path,
                    sum([path in prot2path[x] for x in ps1]),
                    sum([path in prot2path[x] for x in ps2]),
                    sum([path in prot2path[x] for x in psX]),
                    sum([path in prot2path[x] for x in psU1]),
                    sum([path in prot2path[x] for x in psU2]),
                )
                for path in all_paths
                ]
        paths1 = set([x[0] for x in path_prev_list if x[1] >= min_path_entries])
        paths2 = set([x[0] for x in path_prev_list if x[2] >= min_path_entries])
        pathsX = set([x[0] for x in path_prev_list if x[3] >= min_path_entries])
        pathsU1 = set([x[0] for x in path_prev_list if x[4] >= min_path_entries])
        pathsU2 = set([x[0] for x in path_prev_list if x[5] >= min_path_entries])
        summaries.append(overlap_description('pathways with more than ' + str(min_path_entries) + ' protein(s)',
                paths1,
                ps1_name,
                paths2,
                ps2_name,
                len(Protein.get_all_pathways())
                ))
        pathway_file = base + "_pathways.png"
        png_plots.append(PathHelper.url_of_file(pathway_file))
        plot_venn('Pathways'
                  ,paths1
                  ,ps1_name
                  ,paths2
                  ,ps2_name
                  ,pathway_file
                  )
        from dtk.gene_sets import get_pathway_id_name_map
        id2name = get_pathway_id_name_map()
        pathway_enrichments.append(get_pathway_enrichment(ps1_name
                              , paths1
                              , ps1
                              , total_prot_cnt
                              , pwy_prot_mm.fwd_map()
                              , id2name
                              , topN
                              )
                             )
        pathway_enrichments.append(get_pathway_enrichment(ps2_name
                              , paths2
                              , ps2
                              , total_prot_cnt
                              , pwy_prot_mm.fwd_map()
                              , id2name
                              , topN
                              )
                             )
        pathway_enrichments.append(get_pathway_enrichment('the intersection of both protSets'
                              , pathsX
                              , psX
                              , total_prot_cnt
                              , pwy_prot_mm.fwd_map()
                              , id2name
                              , topN
                              )
                             )
        pathway_enrichments.append(get_pathway_enrichment(ps1_name + ' only proteins'
                              , pathsU1
                              , psU1
                              , total_prot_cnt
                              , pwy_prot_mm.fwd_map()
                              , id2name
                              , topN
                              )
                             )
        pathway_enrichments.append(get_pathway_enrichment(ps2_name + ' only proteins'
                              , pathsU2
                              , psU2
                              , total_prot_cnt
                              , pwy_prot_mm.fwd_map()
                              , id2name
                              , topN
                              )
                             )
    sums_plots = zip(summaries, png_plots)
    return render(request
                ,'nav/ps_cmp.html'
                ,make_ctx(request,ws,'nav_ps_cmp',merge_dicts(
                        dict_subset(locals(),[
                                 'ps_cmp_form',
                                 'sums_plots',
                                 'pathway_enrichments',
                                 ]),
                         )
                ))

@login_required
def ps(request,ws_id):
    ws = Workspace.objects.get(pk=ws_id)
    url_config=UrlConfig(request,
                defaults={
                })
    from browse.models import ProtSet,Protein
    protset_form = ProtsetForm()
    d2p_form = PsFromDsForm(ws)

    import dtk.dynaform as df
    ff = df.FormFactory()
    df.Election({'ws_id':ws_id}).add_to_form(ff)
    df.P2DFileFieldType({'ws_id':ws_id}).add_to_form(ff)
    df.P2DMinFieldType({'ws_id':ws_id}).add_to_form(ff)
    reviewround_form = ff.get_form_class()()


    def gene_annotate(uniprots):
        from browse.models import Protein
        u2g = Protein.get_uniprot_gene_map(uniprots=uniprots)
        return [f'{uniprot} # {u2g.get(uniprot)}' for uniprot in uniprots]
    
    presets = url_config.as_list('uniprots')
    if presets:
        protset_form.initial['proteins'] = "\n".join(gene_annotate(presets))
    qs = ProtSet.objects.filter(ws=ws).order_by('name')
    if qs.exists():
        from dtk.html import checklist
        partslist_html = checklist([
                            ( ('parts%s'%x[0]),x[1])
                            for x in ws.get_uniprot_set_choices(auto_dpi_ps=False)
                            ],
                    set(),
                    )
    else:
        partslist_html = ''
    if request.method == 'POST' and post_ok(request):
        if 'create_btn' in request.POST:
            protset_form = ProtsetForm(request.POST)
            if protset_form.is_valid():
                p = protset_form.cleaned_data
                term_list = [x.split('#')[0].strip() for x in p['proteins'].split('\n')]
                prots = []
                any_errors = False
                for term in term_list:
                    if not term:
                        continue
                    prot = Protein.get_canonical_of_uniprot(uniprot=term.upper())
                    if prot:
                        uniprot = prot.uniprot
                        prots.append(prot)
                        # If someone provided a non-canonical uniprot, let them know that we converted their entry
                        # to the canonical one.
                        if uniprot.upper() != term.upper():
                            from django.contrib import messages
                            from dtk.html import link, join
                            url = link(prot.gene, ws.reverse('protein', uniprot))
                            messages.add_message(request, messages.INFO, join(f'Converted {term} to {uniprot} - ', url))
                    else:
                        any_errors = True
                        protset_form.add_error('proteins'
                                        ,'%s is not a valid uniprot' % term
                                        )
                # create protset
                if not any_errors:
                    if prots:
                        ps = ProtSet.objects.create(
                                ws=ws,
                                name=p['name'],
                                description=p['description'],
                                created_by=request.user.username,
                                )
                        for p in prots:
                            ps.proteins.add(p)
                        return HttpResponseRedirect(ws.reverse('nav_ps'))
                    else:
                        protset_form.add_error('proteins'
                                        ,'you must specify some proteins'
                                        )
        elif 'delete_btn' in request.POST:
            ProtSet.objects.filter(pk=request.POST['ps_id']).delete()
            return HttpResponseRedirect(ws.reverse('nav_ps'))
        elif 'union_preload_btn' in request.POST:
            name_set = set()
            prefix = 'parts'
            for key in request.POST:
                if key.startswith(prefix):
                    name_set.add((key[len(prefix):]))
            names=set()
            for name in name_set:
                names.update(ws.get_uniprot_set(name))
            names=sorted(names)
            protset_form.initial['proteins'] = "\n".join(gene_annotate(names))
        elif 'dpi_preload_btn' in request.POST:
            uniprots = ws.get_uniprots_from_drugset(
                    drugset_name=request.POST['drugset'],
                    threshold=request.POST['threshold'],
                    mapping=request.POST['mapping']
                    )
            protset_form.initial['proteins'] = "\n".join(sorted(gene_annotate(uniprots)))
        elif 'reviewround_preload_btn' in request.POST:
            from browse.models import WsAnnotation
            from dtk.prot_map import AgentTargetCache, DpiMapping
            election_id = request.POST['election']
            mapping = DpiMapping(request.POST['p2d_file'])
            thresh = request.POST['p2d_min']
            wsas = list(WsAnnotation.objects.filter(vote__election_id=election_id))
            atc = AgentTargetCache.atc_for_wsas(wsas=wsas, dpi_mapping=mapping, dpi_thresh=thresh)
            uniprots = atc.all_prots
            protset_form.initial['proteins'] = "\n".join(sorted(gene_annotate(uniprots)))
        else:
            raise NotImplementedError('unsupported POST')
    return render(request
                ,'nav/ps.html'
                ,make_ctx(request,ws,'nav_ps',{
                        'protset_form':protset_form,
                        'protset_list':ProtSet.objects.filter(ws=ws),
                        'partslist_html':partslist_html,
                        'd2p_form':d2p_form,
                        'reviewround_form':reviewround_form,
                         })
                )

# adapted from: http://matthiaseisen.com/pp/patterns/p0144/
def plot_venn(title,s1,s1_name,s2,s2_name, filename):
    from matplotlib import pyplot as plt
    from matplotlib_venn import venn2, venn2_circles
    fig = plt.figure()
    # Subset sizes
    s = (
        len(s1-s2),  # Ab
        len(s2-s1),  # aB
        len(s1&s2),  # AB
    )

    v = venn2(subsets=s, set_labels=(s1_name, s2_name))

    # Subset labels, colors, alphas
    if s[0]:
        v.get_label_by_id('10').set_text(str(s[0]))
        v.get_patch_by_id('10').set_color("#1A71A8")
        v.get_patch_by_id('10').set_alpha(0.75)
    if s[1]:
        v.get_label_by_id('01').set_text(str(s[1]))
        v.get_patch_by_id('01').set_color("#F48725")
        v.get_patch_by_id('01').set_alpha(0.75)
    if s[2]:
        v.get_label_by_id('11').set_text(str(s[2]))
        v.get_patch_by_id('11').set_color('#877C66')
        v.get_patch_by_id('11').set_alpha(1.0)

    plt.title(title)

    fig.savefig(filename)

def get_pathway_enrichment(title, paths, prots, bg, pwy2prot, id2name, topN = 10 ):
    from browse.models import Protein
    import scipy.stats as sps
    rows = []
    from dtk.url import pathway_url_factory
    pathway_url = pathway_url_factory()
    from dtk.html import link
    for path in paths:
        path_prots = set(pwy2prot[path])
        total = len(prots | path_prots)
        intersect = len(prots & path_prots)
        set_not_path = len(prots - path_prots)
        path_not_set = len(path_prots - prots)
        neither = bg - total
        oddsr, p = sps.fisher_exact([[intersect, set_not_path],
                                 [path_not_set, neither]
                                ],
                               )

        url = pathway_url(path)
        name = id2name.get(path, path)
        rows.append([link(name, url), oddsr, p])
    rows.sort(key=lambda x: (x[2], x[1]*-1.0))
    return ['Pathways enriched in ' + title, rows[:topN]]

def overlap_description(obj_type,s1,s1_name,s2,s2_name, bg):
    summary = []
    total = len(s1 | s2)
    intersect = len(s1&s2)
    only_s1 = len(s1-s2)
    only_s2 = len(s2-s1)
    neither = bg - total
    summary.append('%d distinct %s, of a possible %d'%(total,obj_type, bg))
    summary += [
            '%d (%d%%) in %s' % (
                    s,
                    int(0.5 + 100.0 * (s/total if total else 0)),
                    name
                    )
                for s,name in (
                        (only_s1,s1_name+' only'),
                        (only_s2,s2_name+' only'),
                        (intersect,'both'),
                        )
                ]
    try:
        import scipy.stats as sps
        import numpy as np
        oddsr, p = sps.fisher_exact([[intersect, only_s1],
                                 [only_s2, neither]
                                ],
                               )
        summary.append('That overlap has a %.2f odds ratio, with a p-value of %.2e'%(oddsr, p))
    except ValueError:
        # in some cases with bad data, the total set size may exceed the
        # background size, which throws an exception inside numpy; just
        # skip the output line in this case
        pass
    return 	summary

@login_required
def plot(request,ws_id):
    ws = Workspace.objects.get(pk=ws_id)
    url_config=UrlConfig(request,
                defaults={
                    'headline':'Plot',
                })
    from dtk.data import merge_dicts,dict_subset
    headline = url_config.as_string('headline')
    path = url_config.as_string('path')
    from dtk.plot import PlotlyPlot
    root,relpath  = PathHelper.path_of_pubfile(path)
    src = PlotlyPlot.build_from_file(root+relpath)
    return render(request
                ,'nav/plot.html'
                ,make_ctx(request,ws,'nav_scoreboard',merge_dicts(
                        dict_subset(locals(),[
                                 'headline',
                                 'src',
                                 ]),
                         {'div_id':'plot0'},
                         )
                ))

class dsDpiCmpView(DumaView):
    template_name='nav/ds_dpi_cmp.html'
    index_dropdown_stem='workflow'
    GET_parms={
            'drugset1':(str,None),
            'drugset2':(str,None),
            'dpi':(str,None),
            'ppi':(str,None),
            'dpi_t':(float,None),
            'ppi_t':(float,None)
            }
    button_map={
            'compare':['compare'],
            }
    def make_compare_form(self, data):
        from dtk.prot_map import DpiMapping,PpiMapping
        ds_choices = self.ws.get_wsa_id_set_choices(retro=True)
        class MyForm(forms.Form):
            drugset1 = forms.ChoiceField(
                label = 'Dugset1',
                choices = ds_choices,
                initial = self.drugset1
                )
            drugset2 = forms.ChoiceField(
                label = 'Dugset2',
                choices = ds_choices,
                initial = self.drugset2
                )
            dpi = forms.ChoiceField(
                label = 'DPI dataset',
                choices = DpiMapping.choices(self.ws),
                initial = self.dpi if self.dpi else self.ws.get_dpi_default(),
                )
            dpi_t = forms.FloatField(
                label = 'Min DPI evidence',
                initial = self.dpi_t if self.dpi_t else self.ws.get_dpi_thresh_default(),
                )
            ppi = forms.ChoiceField(
                label = 'PPI dataset',
                choices = PpiMapping.choices(),
                initial = self.ppi if self.ppi else self.ws.get_ppi_default(),
                )
            ppi_t = forms.FloatField(
                label = 'Min PPI evidence',
                initial = self.ppi_t if self.ppi_t else self.ws.get_ppi_thresh_default(),
                )
        return MyForm(data)
    def compare_post_valid(self):
        p = self.context['compare_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))
    def custom_context(self):
        if (not self.drugset1
            or not self.drugset2
            or not self.dpi
            or not self.dpi_t
            or not self.ppi_t
            or not self.ppi
           ):
            return
        self.setup()
        self.dpi_overlaps()
        self.sums_plots = zip(self.summary, self.png_plots)
        self.get_ranks()
        self._plot_boxplot()
        self.context_alias(
                sums_plots=self.sums_plots,
                plotly_plots=self.plotly_plots
             )
        self._plot_atc_freq()
        self._dir_prot_by_drug()
        self._plot_dt_boxplot()
        self._plot_indt_boxplot()
    def get_ranks(self):
        from dtk.scores import Ranker,SourceList
        import runner.data_catalog as dc
        self.all_ranks = {}
        sl = SourceList(self.ws)
        sl.load_from_session(self.request.session)
        self.all_ranks={x:{'cm':[], 'vals':[]}
                        for x in [self.ds1_name,self.ds2_name]
                       }
        for src in sl.sources():
            bji = src.bji()
            cat = bji.get_data_catalog()
            for code in cat.get_codes('wsa','score'):
                ranker = Ranker(cat.get_ordering(code, True))
                k = " ".join([src.label(), code])
                ranks1 = [ranker.get(wsa) for wsa in self.ds1]
                ranks2 = [ranker.get(wsa) for wsa in self.ds2]
                if sum(ranks1) == len(ranks1) and sum(ranks2) == len(ranks2):
                    continue
                self.all_ranks[self.ds1_name]['cm'] += [k]*len(self.ds1)
                self.all_ranks[self.ds1_name]['vals'] += ranks1
                self.all_ranks[self.ds2_name]['cm'] += [k]*len(self.ds2)
                self.all_ranks[self.ds2_name]['vals'] += ranks2

    def _plot_atc_freq(self):
        '''
        Fetch frequency of each ATC category present in each drug set ATC frequency is stored in
        dictionary of dictionary, first keyed by ATC level, then by ATC code, with an inner value that
        signifies the proportion of drugs with the code in drug set.
        '''
        self._get_atcs()
        from dtk.plot import PlotlyPlot

        # Create four grouped bar plots, one for each ATC category
        # TODO: Replace ATC codes on xaxis with readable string
        atc_levels = [1,2,3,4]
        bar_names = [self.ds1_name, self.ds2_name]
        for level in atc_levels:
            go_bars = []

            # Sorting logic to get the bar plots sorted in decending order of ATC frequency in drug set 1.
            ordered_atcs = sorted(self.ds1_atc_counts[level], reverse=True,
                                                            key=self.ds1_atc_counts[level].get)
            atc_counts_1 = [self.ds1_atc_counts[level][k] if k in self.ds1_atc_counts[level]
                                                            else 0 for k in ordered_atcs]
            atc_counts_2 = [self.ds2_atc_counts[level][k] if k in self.ds2_atc_counts[level]
                                                            else 0 for k in ordered_atcs]
            ordered_atcs.extend(list(set(ordered_atcs)-set(self.ds2_atc_counts.keys())))
            atc_map = self._get_atc_code_map()
            max_tick_lengths = []
            for i, ds_counts in enumerate([atc_counts_1, atc_counts_2]):
                x_tick_labels = [atc_map[code] if code in atc_map else code for code in ordered_atcs]
                try:
                    max_tick_lengths.append(max([len(label) for label in x_tick_labels]))
                except ValueError:
                    continue
                go_bars.append(dict(x = x_tick_labels,
                               type = 'bar',
                               y =  ds_counts,
                               name = bar_names[i]))
            if not go_bars:
                self.message('No ATC data')
            else:
                self.plotly_plots.append(('atc'+" "+str(level), PlotlyPlot(go_bars,
                           {'title':'Comparison of Level {0} ATC Code Frequencies'.format(level),
                            'yaxis':{'title':'Proportion of Drugs Level {0} ATC Code'.format(level)},
                            'xaxis':{'title':'Level {0} ATC Code'.format(level)},
                            'width':max([600,
                                len(ordered_atcs)*25
                                ]),
                            'height':600+max(max_tick_lengths)*10,
                             'margin':dict(
                             b=max(max_tick_lengths)*10
                             ),
                           }
                    )))
    @staticmethod
    def _get_atc_code_map():
         '''
         Static method that pulls down the atc_code.txt file, which maps codes to readable strings from
         S3. To refresh this file, run use Makefile in databases/atc.
         '''
         from dtk.s3_cache import S3MiscBucket,S3File
         from dtk.files import get_file_lines
         s3_file = S3File(S3MiscBucket(),'atc_code.txt')
         s3_file.fetch()
         codes = []
         terms = []
         for line in get_file_lines(s3_file.path()):
             codes.append(line.split("\t")[0].strip())
             terms.append(line.split("\t")[1].strip().title())
         return dict(zip(codes, terms))

    def _plot_boxplot(self):
        from dtk.plot import PlotlyPlot
        self.plotly_plots.append(('boxplot', PlotlyPlot(
                   [dict(type='box'
                      , y = self.all_ranks[ds]['vals']
                      , x = self.all_ranks[ds]['cm']
                      , name = ds
                      , marker = dict(size = 3, opacity = 0.5)
                     )
                    for ds in self.all_ranks
                   ]
            ,
            {'title':'CM Rank Distribution Comparison',
             'yaxis':{'title':'Rank',
                      'type':'log'},
             'boxmode': 'group',
             'width': max([600,
                          len(list(self.all_ranks.values())[0]['cm'])*3
                         ]
                         )
            }
           ))
        )
    def setup(self):
        self.plotly_plots = []
        self.png_plots = []
        self.summary = []
        self._load_drugs()
        self._load_prots()
        self.indirect_prots_by_drug()
        self._get_sim_matrices()
        self._calc_prev()
        self._get_protSets()
        self._plot_prep()
    def _load_drugs(self):
        from browse.models import WsAnnotation
        ds_names = dict(self.ws.get_wsa_id_set_choices(retro=True))
        self.ds1_name = ds_names[self.drugset1]
        self.ds2_name = ds_names[self.drugset2]
        self.ds1 = self.ws.get_wsa_id_set(self.drugset1)
        self.ds2 = self.ws.get_wsa_id_set(self.drugset2)
        if not self.ds1:
            self.message('The first drugset is empty - not all plotting is possible.')
        if not self.ds2:
            self.message('The second drugset is empty - not all plotting is possible.')
        self.total_wsa_ids = WsAnnotation.objects.filter(ws=self.ws.id).count()
        self.both_ds = self.ds1|self.ds2
        self.wsaid_to_wsa = {wsa.id:wsa
                             for wsa in self.ws.wsannotation_set.filter(
                                     pk__in=self.both_ds
                              )
                            }
    def _load_prots(self):
        from dtk.prot_map import DpiMapping
        import operator
        self.dm = DpiMapping(self.dpi)
        self.wsa_id2prot = {
                wsa_id:set([x.uniprot_id
                        for x in self.dm.get_dpi_info(wsa.agent, min_evid = self.dpi_t)
                        ])
                for wsa_id,wsa in self.wsaid_to_wsa.items()
                }
        self.all_proteins = reduce(operator.or_,list(self.wsa_id2prot.values()))
    def _get_sim_matrices(self):
        from dtk.similarity import build_mol_prot_sim_matrix, build_asym_mol_prot_sim_matrix
        self.dir_sm = build_mol_prot_sim_matrix(self.wsa_id2prot, verbose=False)
        self.dir_asm = build_asym_mol_prot_sim_matrix(self.wsa_id2prot, self.ds1, self.ds2)
        self.dir_sm.mds()
        self.ind_sm = build_mol_prot_sim_matrix(self.wsa_id2ind_prot, verbose=False)
        self.ind_asm = build_asym_mol_prot_sim_matrix(self.wsa_id2ind_prot, self.ds1, self.ds2)
        self.ind_sm.mds()
    def indirect_prots_by_drug(self):
        self.ds1_ind_prot_counts = {}
        self.ds2_ind_prot_counts = {}
        self.wsa_id2ind_prot = {}
        drug_ind_prot_count_dicts = [self.ds1_ind_prot_counts, self.ds2_ind_prot_counts]
        from dtk.prot_map import PpiMapping
        pm = PpiMapping(self.ppi)
        for i, ds in enumerate([self.ds1, self.ds2]):
            for drug in ds:
                ind_prots = [x[1] for x in
                             pm.get_ppi_info_for_keys(self.wsa_id2prot[drug],
                                                      self.ppi_t
                                                     )
                            ]
                drug_ind_prot_count_dicts[i][drug] = len(ind_prots)
                self.wsa_id2ind_prot[drug] = set(ind_prots)
    def _get_atcs(self):
        # Partially based on /home/ubuntu/2xar/twoxar-demo/experiments/IBD_MTs/aggregate_paths_atcs_per_scoreboard.py
        atc_cut_pts = [1,3,4,5]
        atc_levels = [1,2,3,4]
        ds1_atcs = {n:[] for n in atc_levels}
        ds2_atcs = {n:[] for n in atc_levels}
        from algorithms.run_orfex import get_atc_cache
        self.atc_cache = get_atc_cache()
        from browse.models import WsAnnotation
        from collections import Counter
        ds_atcs = [ds1_atcs, ds2_atcs]
        drug_sets = [self.ds1, self.ds2]
        # For each drug set, get each drug's ATC codes at each level, store in a temporary dictionary (ds*_atcs)
        ds1_atc_count = 0
        ds2_atc_count = 0
        atc_counters = [ds1_atc_count, ds2_atc_count]
        for ix, ds in enumerate(drug_sets):
            for drug_id in ds:
                wsa = WsAnnotation.objects.get(pk=drug_id)
                l = self.atc_cache.get(wsa.agent_id, None)
                if l is None:
                    l = []
                for atc in l:
                    atc_counters[ix] += 1
                    for i,lev in enumerate(atc_levels):
                         ds_atcs[ix][lev].append(atc[0:atc_cut_pts[i]])
        # For each drug set, aggregate occurances of ATC codes at each level, then transform into proportion,
        # storing each in self.ds*_atc_counts, a dictionary of dictionaries. This will be used to create a grouped
        # bar plot in _plot_atc_freq(self)
        self.ds1_atc_counts = {}
        self.ds2_atc_counts = {}
        ds_atc_counts = [self.ds1_atc_counts, self.ds2_atc_counts]
        for ix, ds_atc in enumerate(ds_atcs):
            for n in ds_atc:
                ds_atc_counts[ix][n] = {k:float(v)/atc_counters[ix] for k,v in Counter(ds_atc[n]).most_common()}
    def _calc_prev(self):
        self.prev_list = [
                (
                    prot,
                    sum([prot in self.wsa_id2prot[x] for x in self.ds1]),
                    sum([prot in self.wsa_id2prot[x] for x in self.ds2]),
                )
                for prot in self.all_proteins
                ]
    def _dir_prot_by_drug(self):
        self.ds1_prot_counts = {}
        self.ds2_prot_counts = {}
        drug_prot_count_dicts = [self.ds1_prot_counts, self.ds2_prot_counts]
        for i, ds in enumerate([self.ds1, self.ds2]):
            for drug in ds:
                drug_prot_count_dicts[i][drug] = len(self.wsa_id2prot[drug])
    def _get_protSets(self):
        self.ds1_ps = set([x[0] for x in self.prev_list if x[1]])
        self.ds2_ps = set([x[0] for x in self.prev_list if x[2]])
    def _plot_prep(self):
        from path_helper import make_directory
        import os
        path = PathHelper.ws_publish(self.ws.id)+'dsCmp/'
        make_directory(path)
        self.base_dir = os.path.join(path, "_".join([self.ds1_name, self.ds2_name]))
    def dpi_overlaps(self):
        self._summarise_drug_overlap()
        self._plot_drug_overlap()
        self._summarise_dt_overlap()
        self._plot_dt_overlap()
        self._plot_mds()
        self._plot_heatmaps()
        if len(self.ds1)>0 and len(self.ds2)>0:
            self._plot_scatter()
    def _summarise_drug_overlap(self):
        self.summary.append(overlap_description(
                     'drugs'
                     ,self.ds1
                     ,self.ds1_name
                     ,self.ds2
                     ,self.ds2_name
                     ,self.total_wsa_ids
                    ))
    def _plot_drug_overlap(self):
        drug_file = self.base_dir + "_drug.png"
        plot_venn('Drug overlap'
                  ,self.ds1
                  ,self.ds1_name
                  ,self.ds2
                  ,self.ds2_name
                  ,drug_file
                  )
        self.png_plots.append(PathHelper.url_of_file(drug_file))
    def _summarise_dt_overlap(self):
        self.summary.append(overlap_description(
                'proteins',
                self.ds1_ps,
                self.ds1_name,
                self.ds2_ps,
                self.ds2_name,
                self.dm.get_uniq_target_cnt()
                ))
    def _plot_dt_overlap(self):
        prot_file = self.base_dir + "_prot.png"
        plot_venn('Protein overlap'
                  ,self.ds1_ps
                  ,self.ds1_name
                  ,self.ds2_ps
                  ,self.ds2_name
                  ,prot_file
                 )
        self.png_plots.append(PathHelper.url_of_file(prot_file))
    def _plot_scatter(self):
        from dtk.plot import scatter2d
        import random
        self.plotly_plots.append(('scat',scatter2d(
                'fraction of ' + self.ds1_name,
                'fraction of ' + self.ds2_name,
                [
                    (
                        (x[1]+0.15*random.random())/len(self.ds1),
                        (x[2]+0.15*random.random())/len(self.ds2),
                    )
                    for x in self.prev_list
                    ],
                text=[x[0] for x in self.prev_list],
                ids=('protpage',[x[0] for x in self.prev_list]),
                title='prevalence of each DPI protein'
                )))
    def _plot_heatmaps(self):
        self._general_plot_asym_heatmaps(self.dir_asm,self.ds1,self.ds2,'Direct target heatmap', 'dirAHM')
        self._general_plot_asym_heatmaps(self.ind_asm,self.ds1,self.ds2,'Indirect target heatmap', 'indAHM')
    def _plot_mds(self):
        self._general_plot_mds(self.dir_sm,'Direct target MDS', 'dirMDS')
        self._general_plot_mds(self.ind_sm,'Indirect target MDS', 'indMDS')
    def _general_plot_mds(self, sm,title,file):
        from dtk.plot import scatter2d, Color
        class_idx = []
        for i in sm.row_keys:
            if i in self.ds1 and i in self.ds2:
                class_idx.append(2)
            elif i in self.ds1:
                class_idx.append(0)
            elif i in self.ds2:
                class_idx.append(1)
            else:
                self.message('missing idx')
        self.plotly_plots.append((file,scatter2d(
                'MDS Axis 1',
                'MDS Axis 2',
                sm.mds_matrix,
                classes=[
                    (self.ds1_name,{'color':Color.default}),
                    (self.ds2_name,{'color':Color.highlight}),
                    ('Both',{'color':Color.highlight2})
                 ],
                 class_idx = class_idx,
                text=[self.wsaid_to_wsa[i].get_name(self.is_demo())
                      for i in sm.row_keys
                     ],
                ids=('drugpage',sm.row_keys),
                title=title,
                refline=False
                )))
    def _general_plot_asym_heatmaps(self, sm,row_keys,col_keys,title,file):
        from dtk.plot import plotly_heatmap
        row_names = [self.wsaid_to_wsa[i].get_name(self.is_demo())
                     for i in row_keys
                    ]
        col_names = [self.wsaid_to_wsa[i].get_name(self.is_demo())
                     for i in col_keys
                    ]
        self.plotly_plots.append((file,plotly_heatmap(
                sm,
                row_names,
                col_labels=col_names,
                color_zero_centered = True,
                Title = title,
                reorder_cols=True
              )))
    def _plot_dt_boxplot(self):
        from dtk.plot import PlotlyPlot
        self.plotly_plots.append(('dt_boxplot', PlotlyPlot(
                   [dict(y = list(self.ds1_prot_counts.values())
                      , type = 'box'
                      , boxpoints='all'
                      , jitter=0.3
                      , pointpos=-1.8
                      , name = self.ds1_name
                      , marker = dict(size = 5, opacity = 0.7)
                     ),
                    dict(y = list(self.ds2_prot_counts.values())
                      , type = 'box'
                      , name = self.ds2_name
                      , boxpoints='all'
                      , jitter=0.3
                      , pointpos=-1.8
                      , marker = dict(size = 5, opacity = 0.7)
                     )
                   ]
            ,
            {'title':'Drug Set Direct Target Distributions',
             'yaxis':{'title':'Number of Direct Targets'},
             'boxmode': 'group'
            }
           ))
        )
    def _plot_indt_boxplot(self):
        from dtk.plot import PlotlyPlot
        self.plotly_plots.append(('indt_boxplot', PlotlyPlot(
                   [dict(y = list(self.ds1_ind_prot_counts.values())
                      , type = 'box'
                      , boxpoints='all'
                      , jitter=0.3
                      , pointpos=-1.8
                      , name = self.ds1_name
                      , marker = dict(size = 5, opacity = 0.7)
                     ),
                    dict(y = list(self.ds2_ind_prot_counts.values())
                      , type = 'box'
                      , name = self.ds2_name
                      , boxpoints='all'
                      , jitter=0.3
                      , pointpos=-1.8
                      , marker = dict(size = 5, opacity = 0.7)
                     )
                   ]
            ,
            {'title':'Drug Set Indirect Target Distributions',
             'yaxis':{'title':'Number of Indirect Targets'},
             'boxmode': 'group'
            }
           ))
        )

class DrugsetForm(forms.Form):
    name = forms.CharField(
            max_length=60,
            required=True,
            )
    description = forms.CharField(
            label="Description",
            widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
            required=False,
            )
    drugs = forms.CharField(
            label="Enter drugs, one per line",
            widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
            required=True,
            )


class DrugLookup:
    def __init__(self,ws,href_lookup={}):
        self.ws = ws
        self.errors = []
        self.wsa_list = []
        self.href_lookup = href_lookup
    def _add_error(self,term,matches=None):
        from django.utils.html import format_html,format_html_join
        if not matches:
            detail = "didn't match any drug"
        else:
            match_str = format_html_join(
                    "', '",
                    '<a term="{}" id="{}" href="{}">{}</a>',
                    [ (term, x.id, x.drug_url(), x.agent.canonical) for x in matches ],
                    )
            detail = format_html("ambiguous; matches: '{}'",match_str)
        if term in self.href_lookup:
            err = format_html('<a href="{}">{}</a> {}'
                    ,self.href_lookup[term]
                    ,term
                    ,detail
                    )
        else:
            err = format_html('<b>{}</b> {}'
                    ,term
                    ,detail
                    )
        self.errors.append(err)
    def lookup_single(self,term,exclude=[]):
        from browse.models import WsAnnotation
        from drugs.models import Tag
        tag_qs = Tag.objects.filter(value=term)
        if exclude:
            tag_qs = tag_qs.exclude(prop_id__in=exclude)
        wsa_qs = WsAnnotation.objects.filter(
                    ws=self.ws,
                    agent_id__in=[x.drug_id for x in tag_qs],
                    )
        cnt = wsa_qs.count()
        if cnt == 0:
            self._add_error(term)
        elif cnt == 1:
            self.wsa_list.append(wsa_qs[0])
        else:
            self._add_error(term,wsa_qs)

    def lookup_multiple(self,terms,exclude=[]):
        from browse.models import WsAnnotation
        for term in terms:
            if not term:
                continue # ignore blank lines
            if '|#' in term:
                wsa_id = int(term.split('|#')[-1])
                self.wsa_list.append(WsAnnotation.objects.get(pk=wsa_id))
                continue


            self.lookup_single(term,exclude)
class dsView(DumaView):
    template_name='nav/ds.html'
    GET_parms = {
            'edit':(int,None),
            }
    def custom_setup(self):
        self.preload_ids=set()
        self.button_map={
                'create':['drugset'],
                'preload':['ind','demerit','atc','union','drugsetname'],
                }
        from browse.models import DrugSet
        self.context_alias(drugset_list=DrugSet.objects.filter(ws=self.ws))
        if self.edit:
            self.context_alias(edit_ds=DrugSet.objects.get(pk=self.edit))
            self.button_map['save'] = ['edit']
            self.button_map['delete'] = []
    def make_drugset_form(self,data):
        if self.preload_ids:
            name_list='\n'.join(sorted([
                    '%s|#%d' % (name, wsa_id)
                    for wsa_id,name in self.ws.wsa_prop_pairs('canonical')
                    if wsa_id in self.preload_ids
                    ]))
        else:
            name_list=''
        return DrugsetForm(data,initial={'drugs':name_list})
    def create_post_valid(self):
        p = self.drugset_form.cleaned_data
        term_list = [x.strip() for x in p['drugs'].split('\n')]
        dl = DrugLookup(self.ws)
        dl.lookup_multiple(term_list)
        if dl.errors:
            # see if we can do better w/o the brand column
            dl2 = DrugLookup(self.ws)
            from drugs.models import Prop
            dl2.lookup_multiple(term_list,exclude=[Prop.get('brand')])
            if dl2.errors:
                # nope, just report original errors
                for err in dl.errors:
                    self.drugset_form.add_error('drugs',err)
            else:
                dl = dl2
        if not dl.errors:
            # create drugset
            from browse.models import DrugSet
            ds = DrugSet.objects.create(
                    ws=self.ws,
                    name=p['name'],
                    description=p['description'],
                    created_by=self.request.user.username,
                    )
            for wsa in dl.wsa_list:
                ds.drugs.add(wsa)
            return HttpResponseRedirect(self.here_url())
    def make_edit_form(self,data):
        class MyForm(forms.Form):
            name = forms.CharField(
                        label='Change Name',
                        initial=self.edit_ds.name,
                        )
        return MyForm(data)
    def save_post_valid(self):
        p = self.edit_form.cleaned_data
        self.edit_ds.name = p['name']
        self.edit_ds.save()
        return HttpResponseRedirect(self.here_url(edit=None))
    def delete_post_valid(self):
        self.edit_ds.delete()
        return HttpResponseRedirect(self.here_url(edit=None))
    def make_ind_form(self,data):
        from browse.models import WsAnnotation
        enum = WsAnnotation.indication_vals
        from dtk.html import WrappingCheckboxSelectMultiple
        class MyForm(forms.Form):
            ind = forms.MultipleChoiceField(
                        label='Indications',
                        choices=[x for x in enum.choices() if x[0]],
                        widget=WrappingCheckboxSelectMultiple,
                        required=False,
                        )
        return MyForm(data)
    def make_demerit_form(self,data):
        from browse.models import Demerit
        from dtk.html import WrappingCheckboxSelectMultiple
        choices = [
                (x.id,x.desc)
                for x in Demerit.objects.filter(active=True).order_by('desc')
                ]
        class MyForm(forms.Form):
            demerits = forms.MultipleChoiceField(
                        label='Rejection Reasons',
                        choices=choices,
                        widget=WrappingCheckboxSelectMultiple,
                        required=False,
                        )
        return MyForm(data)
    def make_atc_form(self,data):
        class MyForm(forms.Form):
            prefix = forms.CharField(
                        label='ATC Prefix',
                        required=False,
                        )
        return MyForm(data)
    def make_drugsetname_form(self,data):
        class MyForm(forms.Form):
            drugsetname = forms.CharField(
                        label='Drugset Name(s) (comma-separated)',
                        required=False,
                        )
        return MyForm(data)
    def make_union_form(self,data):
        from dtk.html import WrappingCheckboxSelectMultiple
        class MyForm(forms.Form):
            sets = forms.MultipleChoiceField(
                    label='',
                    choices=[(x.id,x.name)
                            for x in self.drugset_list.order_by('name')
                            ],
                    widget=WrappingCheckboxSelectMultiple,
                    required=False,
                    )
        return MyForm(data)
    def preload_post_valid(self):
        id_set = self.ind_form.cleaned_data['ind']
        from browse.models import WsAnnotation
        ws_qs = WsAnnotation.objects.filter(ws=self.ws)
        self.preload_ids |= set(
                ws_qs.filter(indication__in=id_set).values_list('id',flat=True)
                )
        dem_set=set(int(x) for x in self.demerit_form.cleaned_data['demerits'])
        for wsa in ws_qs.exclude(demerit_list=''):
            if wsa.demerits() & dem_set:
                self.preload_ids.add(wsa.id)
        prefix = self.atc_form.cleaned_data['prefix']
        if prefix:
            self.preload_ids |= set([
                    wsa_id
                    for wsa_id,atc in self.ws.wsa_prop_pairs('atc')
                    if atc.startswith(prefix)
                    ])
        id_set = self.union_form.cleaned_data['sets']
        from browse.models import DrugSet
        for ds in DrugSet.objects.filter(id__in=id_set):
            self.preload_ids |= set(ds.drugs.values_list('id',flat=True))


        dsnames_str = self.drugsetname_form.cleaned_data['drugsetname']
        if dsnames_str:
            dsnames = dsnames_str.strip().split(',')
            for dsname in dsnames:
                self.preload_ids |= self.ws.get_wsa_id_set(dsname.strip())


        if not self.preload_ids:
            self.message('no drugs selected for preload')

class SortController:
    sort_parm='sort'
    page_parm='page'
    def __init__(self,urlconfig):
        self.urlconfig = urlconfig
        spec = urlconfig.as_string(self.sort_parm)
        self.minus = spec[:1] == '-'
        if self.minus:
            spec = spec[1:]
        self.colspec = spec
    def col_html(self,colspec,label):
        from dtk.html import glyph_icon
        if colspec == self.colspec:
            if not self.minus:
                colspec = '-'+colspec
                icon = glyph_icon("sort-by-attributes-alt")
            else:
                icon = glyph_icon("sort-by-attributes")
        else:
            icon = ""
        return format_html('<a href="{}">{}</a>{}'
                        ,self.urlconfig.here_url({
                                    self.sort_parm:colspec,
                                    self.page_parm:None,
                                    })
                        ,label
                        ,icon
                        )

# The following is a base class for constructing classes for filtering
# based on a subset of enum values. This follows the model of ScoreFilter
# in dtk.scores; details of the method meanings are there. This factors
# out duplicate code from IndFilter and DemeritFilter
class EnumFilter(object):
    filt_list_delim=','
    def get_filter_config(self):
        return self.filt_list_delim.join([str(x) for x in self._exclude])
    def __init__(self,ws,querystring=''):
        self.ws=ws
        self.choices = self._get_choices()
        if querystring:
            self._exclude=set([
                    int(x)
                    for x in querystring.split(self.filt_list_delim)
                    ])
        else:
            self._exclude=set()
    def _all_ids(self):
        return set([i for i,label in self.choices])
    def get_filter_form_class(self):
        from dtk.html import WrappingCheckboxSelectMultiple
        from dtk.dynaform import FormFactory
        ff=FormFactory()
        field = forms.MultipleChoiceField(
                        label=self.list_label,
                        choices=self.choices,
                        widget=WrappingCheckboxSelectMultiple,
                        initial=self._all_ids() - self._exclude,
                        required=False,
                        )
        ff.add_field(self.field_key,field)
        return ff.get_form_class()
    def update_filter(self,p):
        all_ids = self._all_ids()
        conv = int if isinstance(next(iter(all_ids)), int) else lambda x: x
        self._exclude = self._all_ids()-set([
                conv(x)
                for x in p[self.field_key]
                ])
    def _short_description(self,plural):
        # build a descriptive label
        labels = []
        if len(self._exclude) > len(self.choices)/2:
            for i,label in self.choices:
                if i not in self._exclude:
                    labels.append(label)
            col_label = 'Only these '+plural
            if not labels:
                labels.append('(None)')
        else:
            for i,label in self.choices:
                if i in self._exclude:
                    labels.append(label)
            col_label = 'Exclude '+plural
        col_val = ','.join(labels)
        return col_label,col_val

class DemeritFilter(EnumFilter):
    field_key='deme'
    list_label='Reject reasons'
    def _get_choices(self):
        from browse.models import Demerit
        qs = Demerit.objects.filter(active=True).order_by('desc')
        return qs.values_list('id','desc')
    def add_to_key_filter(self,key_filter):
        if not self._exclude:
            return
        col_label,col_val=self._short_description('reject reasons')
        # You can't do a query on individual demerits. And, most drugs
        # don't have demerits. So, retrieve all drugs with demerits here,
        # and build a set of anything whose demerits overlap the exclude
        # set.
        from browse.models import WsAnnotation
        qs = WsAnnotation.objects.filter(ws=self.ws).exclude(demerit_list='')
        excluded_wsa_ids = set([
                wsa.id
                for wsa in qs
                if wsa.demerits() & self._exclude
                ])
        key_filter.merge(
                'demefilt',
                col_label,
                col_val,
                excluded_wsa_ids,
                exclude=True,
                )

class IndFilter(EnumFilter):
    field_key='ind'
    list_label='Drug indication'
    def _get_choices(self):
        from browse.models import WsAnnotation
        return WsAnnotation.indication_vals.choices()
    def add_to_key_filter(self,key_filter):
        if not self._exclude:
            return
        col_label,col_val=self._short_description('indications')
        from browse.models import WsAnnotation
        qs = WsAnnotation.objects.filter(ws=self.ws)
        # Most drugs have an indication value of 0 (Unclassified).  We want
        # to retrieve the fewest ids, so if 0 is among the included drugs,
        # we use a negative filter.
        if 0 in self._exclude:
            qs = qs.exclude(indication__in=self._exclude)
            exclude = False
        else:
            qs = qs.filter(indication__in=self._exclude)
            exclude = True
        key_filter.merge(
                'indfilt',
                col_label,
                col_val,
                set(qs.values_list('id',flat=True)),
                exclude=exclude,
                )

class DsFilter(EnumFilter):
    field_key='ds'
    list_label='MoleculeSet'
    def get_filter_config(self):
        return self.filt_list_delim.join([str(x) for x in self._include])
    def __init__(self,ws,querystring=''):
        self.ws=ws
        self.choices = self._get_choices()
        if querystring:
            self._include=set(querystring.split(self.filt_list_delim))
        else:
            self._include=set()
    def get_filter_form_class(self):
        from dtk.html import WrappingCheckboxSelectMultiple
        from dtk.dynaform import FormFactory
        ff=FormFactory()
        field = forms.MultipleChoiceField(
                        label=self.list_label,
                        choices=self.choices,
                        widget=WrappingCheckboxSelectMultiple,
                        initial=self._include,
                        required=False,
                        )
        ff.add_field(self.field_key,field)
        return ff.get_form_class()
    def update_filter(self,p):
        self._include = p[self.field_key]
        print("Set include to", self._include)
    def _get_choices(self):
        return self.ws.get_wsa_id_set_choices()
    def add_to_key_filter(self,key_filter):
        if not self._include:
            return
        col_label,col_val=self._short_description('molsets')

        all_ids = set()
        for choice in self._include:
            ids = self.ws.get_wsa_id_set(choice)
            all_ids |= set(ids)

        exclude = False

        key_filter.merge(
                'dsfilter',
                col_label,
                col_val,
                all_ids,
                exclude=exclude,
                )
    def _short_description(self,plural):
        # build a descriptive label
        labels = []
        for i,label in self.choices:
            if i in self._include:
                labels.append(label)
        col_label = 'Include'+plural
        col_val = ','.join(labels)
        return col_label,col_val

class ScoreMetricsView(DumaView):
    template_name='nav/score_metrics.html'
    index_dropdown_stem='nav_score_metrics'
    GET_parms = {
            'sort':(SortHandler,''),
            'ds':(str,None),
            }
    button_map = {
            'display':['config'],
            }
    def custom_setup(self):
        if self.ds is None:
            self.ds = self.ws.eval_drugset
        from dtk.scores import SourceListWithEnables
        self.sources = SourceListWithEnables(self.ws,jcc=self.jcc)
        self.sources.load_from_session(self.request.session)
    def custom_context(self):
        gcode_list = self.sources.get_code_list_from_enables()
        if not gcode_list:
            self.message('No scores enabled; please correct and try again.')
            return HttpResponseRedirect(self.ws.reverse('nav_scoreboard'))
        # set up first column to hold score name
        from dtk.table import Table
        page_url=self.ws.reverse('nav_scoreplot','wsa')
        from dtk.html import link
        from dtk.duma_view import qstr
        def fmt_link(data,row,col):
            return link(data,page_url+qstr(dict(
                                    score=row[0],
                                    )))
        cols=[Table.Column('',
                idx=1,
                cell_html_hook=fmt_link,
                )]
        # add metric columns
        metric_names = ['APS','AUR','DEA_AREA','FEBE',
                        'SigmaOfRank', 'SigmaOfRank1000', 'wFEBE']
        ds=self.ws.get_wsa_id_set(self.ds)
        from tools import sci_fmt
        for mn in metric_names:
            cols.append(Table.Column(mn,
                    idx=len(cols)+1,
                    code=str(len(cols)+1),
                    cell_fmt=sci_fmt,
                    sort=SortHandler.reverse_mode,
                    ))
        # assemble rows
        from dtk.enrichment import EnrichmentMetric,EMInput,fill_wsa_ordering
        metric_classes = [
                EnrichmentMetric.lookup(x)
                for x in metric_names
                ]
        rows=[]
        from dtk.files import Quiet
        # With MOA dpi, getting the list of ws_wsas is slower. Since many
        # jobs will likely share the same DPI, retrieve the list once per
        # dpi, rather than making fill_wsa_ordering retrieve it every job.
        ws_wsa_cache = {}
        for gcode in gcode_list:
            label = self.sources.get_label_from_code(gcode)
            ordering = self.sources.get_ordering_from_code(gcode,True)
            src, _, _ = self.sources.parse_global_code(gcode)
            dpi=src.bji().get_dpi_choice()
            if dpi not in ws_wsa_cache:
                ws_wsa_cache[dpi] = self.ws.get_default_wsas(
                        dpi
                        ).values_list('id', flat=True)
            ordering = fill_wsa_ordering(
                    ordering,
                    self.ws,
                    ws_wsas=ws_wsa_cache[dpi],
                    )
            emi = EMInput(ordering,ds)
            row = [gcode,label]
            for MetricClass in metric_classes:
                em = MetricClass()
                with Quiet() as tmp:
                    em.evaluate(emi)
                row.append(em.rating)
            rows.append(row)
        # sort data as requested
        if self.sort.colspec:
            rows.sort(
                    key=lambda x:x[int(self.sort.colspec)],
                    reverse=self.sort.minus,
                    )
        # save table for view
        self.metrics_table = Table(rows,cols,
                sort_handler=self.sort,
                url_builder=self.here_url,
                )
    def make_config_form(self,data):
        class MyForm(forms.Form):
            ds=forms.ChoiceField(
                    choices=self.ws.get_wsa_id_set_choices(
                        test_split=True,train_split=True),
                    label='Reference Drug Set',
                    initial=self.ds,
                    )
        return MyForm(data)
    def display_post_valid(self):
        p = self.context['config_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(
                ds=p['ds'],
                ))

class ScoreCorrView(DumaView):
    template_name='nav/score_corr.html'
    index_dropdown_stem='nav_score_corr'

    GET_parms = {
        'thres':(int,0),
        'nonoverlap':(str,'omit'),
        'display':(str,'spearman'),
        'scoreset':(int,None),
        'ds':(str,'none'),
        'wzs_filter':(boolean,True),
        }

    button_map={
        'save':['options'],
        }

    def make_options_form(self,data):
        from dtk.score_pairs import PairwiseCorrelation
        class MyForm(forms.Form):
            scoreset=forms.ChoiceField(
                label='Scoreset',
                choices=[(0, 'None')] + make_refresh_choices(self.ws),
                initial=self.scoreset,
                required=False,
                )
            thres=forms.IntegerField(
                label='Top n drugs to consider (0 is all)',
                initial=self.thres
                )
            ds=forms.ChoiceField(
                label='Filter by drugset',
                choices=[('none', 'None')] + self.ws.get_wsa_id_set_choices(),
                initial=self.ds,
                required=False,
                )
            wzs_filter = forms.BooleanField(
                initial=self.wzs_filter,
                label='Filter to WZS inputs',
                required=False,
                help_text="If from a scoreset with a WZS job, only use that job's inputs",
                )
            nonoverlap=forms.ChoiceField(
                label="Treat non-overlapping scores as:",
                choices=PairwiseCorrelation.nonoverlap_choices,
                help_text="How to handle WSAs present in only 1 of the 2 scores being pairwise-compared.",
                initial=self.nonoverlap,
            )
            display=forms.ChoiceField(
                label='Metric',
                choices=PairwiseCorrelation.corr_type_choices,
                help_text="Jaccard compares which WSAs are actually scored",
                initial=self.display,
            )

        return MyForm(data)

    def save_post_valid(self):
        p = self.context['options_form'].cleaned_data
        p['thres']=  0 if p['thres'] < 0 else p['thres']
        return HttpResponseRedirect(self.here_url(**p))

    def get_gcode_filter(self):
        if not self.scoreset:
            return None
        from browse.models import ScoreSetJob
        # try to find eff_fvs job
        try:
            ssj = ScoreSetJob.objects.get(
                    scoreset_id=self.scoreset,
                    job_type='eff_fvs',
                    )
        except (ScoreSetJob.DoesNotExist,ScoreSetJob.MultipleObjectsReturned):
            return None
        fm_code = f'fvs{ssj.job_id}'
        fm = self.ws.get_feature_matrix(fm_code)
        return set(fm.spec.get_codes())
    def custom_setup(self):
        from dtk.scores import SourceListWithEnables
        self.sources = SourceListWithEnables(self.ws,jcc=self.jcc)
        if self.scoreset:
            self.sources.load_from_scoreset(self.scoreset)
        else:
            self.sources.load_from_session(self.request.session)
        # scan through the scores, building table row and column descriptions,
        # and a list of dictionaries holding the actual score values
        from dtk.table import Table
        from dtk.html import tag_wrap
        from tools import sci_fmt
        import numpy as np
        rows=[]
        cols=[Table.Column('',
                extract=lambda x:x[1],
                cell_fmt=lambda x:tag_wrap('b',x),
                )]
        score_maps=[]
        corr_values=None
        def get_corr_factory(i):
            return lambda x:sci_fmt(corr_values[x[0], i])
        from dtk.html import link
        from dtk.duma_view import qstr
        comp_page=self.ws.reverse('nav_score_cmp', 'wsa')
        gcode_list = self.sources.get_code_list_from_enables()
        gcode_filter = self.get_gcode_filter()
        if gcode_filter:
            print(len(gcode_list),'gcodes before filter')
            gcode_list = [x for x in gcode_list if x in gcode_filter]
            print(len(gcode_list),'gcodes after filter')
        if not gcode_list:
            self.message('No scores enabled; please correct and try again.')
            return HttpResponseRedirect(self.ws.reverse('nav_scoreboard'))
        assert gcode_list
        def fmt_compare_link(data,row,col):
            if data in ('1','nan'):
                return data
            parms = dict(
                    x=col.score_gcode,
                    y=gcode_list[row[0]],
                    )
            if self.ds != 'none':
                parms['ds'] = self.ds
            return link(data,comp_page+qstr(parms))
        orderings = []
        if self.ds == 'none':
            keep_keys = None
        else:
            keep_keys = self.ws.get_wsa_id_set(self.ds)
        # build structures for data retrieval
        for i,gcode in enumerate(gcode_list):
            label = self.sources.get_label_from_code(gcode)
            # row has label for first column, and index for get_corr_factory
            # and fmt_compare_link
            rows.append((i,label))
            cols.append(Table.Column(label,
                    extract=get_corr_factory(i),
                    cell_html_hook=fmt_compare_link,
                    ))
            # col is annotated with gcode for fmt_compare_link
            cols[-1].score_gcode = gcode
            # orderings for each score are stashed later to get correlations;
            # here we pre-apply any subsetting
            ordering = list(self.sources.get_ordering_from_code(gcode,True))
            if self.thres:
                ordering = ordering[:self.thres]
            if keep_keys is not None:
                ordering = [x for x in ordering if x[0] in keep_keys]
            if not ordering:
                self.message('Warning: no qualified data points for '+label)
            orderings.append(ordering)

        from dtk.score_pairs import PairwiseCorrelation
        scoremat = PairwiseCorrelation.scoremat_from_orderings(orderings)
        pc = PairwiseCorrelation()
        pc.nonoverlap = self.nonoverlap
        pc.corr_type = self.display
        self.zdata = pc.cross_correlate(scoremat)
        corr_values = self.zdata
        self.score_names = [x[1] for x in rows]
        # save table for view
        self.corr_table = Table(rows,cols)
        self.plot()
    
    def plot(self):
        from dtk.plot import plotly_heatmap
        import numpy as np
        plot_data = self.zdata 
        score_names = self.score_names
        x,y = plot_data.shape
        padding_needed = max([len(i) for i in score_names])
        self.plotly = [('Heatmap',
                     plotly_heatmap(
                       plot_data
                       , score_names
                       , Title = 'CM score correlation'
                       , color_bar_title = "Spearman Cor."
                       , col_labels = score_names 
                       , height = y*13 + padding_needed*11
                       , width = x*13 + padding_needed*11
                       , zmin = -1
                       , zmax = 1
                       , reorder_cols = True
              ))]

def dpi_mapping_form_class(dpi_default=None,dpi_thresh_default=None,ws=None):
    from dtk.prot_map import DpiMapping
    class FormClass(forms.Form):
        dpi = forms.ChoiceField(
                choices = list(DpiMapping.choices(ws=ws)),
                initial = dpi_default,
                )
        dpi_thresh = forms.FloatField(
                initial = dpi_thresh_default,
                )
    return FormClass

class ReclassifyFormFactory:
    def __init__(self):
        from browse.models import Demerit
        qs = Demerit.objects.filter(active=True, stage=Demerit.stage_vals.REVIEW).order_by('desc')
        self.demerit_choices = qs.values_list('id','desc')
    def get_form_class(self,wsa,top=None):
        ind_choices = wsa.grouped_choices()
        if top:
            from dtk.html import move_to_top
            ind_choices = move_to_top(ind_choices,top)

        re_ind_choices = []
        for label, group in ind_choices:
            new_group = []
            for choice in group:
                if choice[0] != wsa.indication:
                    new_group.append((choice[0], '!' + choice[1]))
                else:
                    new_group.append(choice)
            re_ind_choices.append((label, new_group))

        ind_choices = re_ind_choices

        from dtk.html import WrappingCheckboxSelectMultiple
        class FormClass(forms.Form):
            wsa_id = forms.IntegerField(
                        initial=wsa.id,
                        widget=forms.HiddenInput(),
                        )
            demerit = forms.MultipleChoiceField(
                        label='Reject reasons',
                        choices=self.demerit_choices,
                        widget=WrappingCheckboxSelectMultiple,
                        initial=wsa.demerits(),
                        required=False,
                        )
            indication = forms.ChoiceField(
                        choices=ind_choices,
                        initial=wsa.indication,
                        )
            indication_href = forms.CharField(
                        initial=wsa.indication_href,
                        required=False,
                        )
        return FormClass

class ScoreboardView(DumaView):
    template_name='nav/scoreboard.html'
    index_dropdown_stem='nav_scoreboard'
    demo_buttons = [
            'columns',
            'defaults',
            'filter',
            ]
    GET_parms = {
            'config':(str,''),
            'page':(int,1),
            'page_size':(int,25),
            'sort':(SortHandler,None),
            'filt':(str,''),
            'indi':(str,''),
            'deme':(str,''),
            'ds':(str,''),
            'prescreen_id':(int,None),
            'dpi':(str,None),
            'dpi_thresh':(float,None),
            'scorejobs':(str,''),
            'condensed':(boolean,False),
            'show_drug_names':(boolean,True),
            }
    filt_mode='filt'
    cols_mode='cols'
    def prep_sources(self):
        from dtk.scores import SourceListWithEnables
        self.sources = SourceListWithEnables(
                self.ws,
                jcc=self.jcc,
                # enable write to session without requiring loading the
                # current session source list (which may just get tossed
                # in the load_scoreboard call); this can save lots of work;
                # also, arrange so that the scorejobs qparm doesn't change
                # the session joblist, so it can be used to create
                # special-purpose scoreboard links.
                session = None if self.scorejobs else self.request.session,
                )
        if self.prescreen_id:
            from browse.models import Prescreen,PrescreenFilterAudit
            self.pscr = Prescreen.objects.get(pk=self.prescreen_id)
            pjob_id = str(self.pscr.primary_job_id())
            job_list = [pjob_id]
            enables = {pjob_id:[self.pscr.primary_code()]}
            for ee in self.pscr.extra_scores.split('|'):
                items=ee.split(':')
                ejob_id = items[0]
                for ecode in items[1:]:
                    enables.setdefault(ejob_id,[]).append(ecode)
                    if ejob_id not in job_list:
                        job_list.append(ejob_id)
            saved_filter = PrescreenFilterAudit.latest_filter(self.prescreen_id)
            if saved_filter:
                # make sure any filter columns are shown
                from dtk.table import ScoreFilter
                for part in saved_filter.split(ScoreFilter.filt_list_delim):
                    fscore,fcond = part.split(ScoreFilter.filt_delim)
                    sjob,scode = fscore.split('_')
                    if sjob not in job_list:
                        job_list.append(sjob)
                    l = enables.setdefault(sjob,[])
                    if scode not in l:
                        l.append(scode)
            self.sources.load_from_string('|'.join(job_list))
            self.sources.set_enables_from_string('|'.join(
                    ':'.join([k]+v)
                    for k,v in enables.items()
                    ))
            self.sort = SortHandler('-'+self.pscr.primary_score)
            if saved_filter:
                self.filt = saved_filter
            elif self.pscr.filters:
                self.filt = self.pscr.filters
        elif self.scorejobs:
            self.sources.load_from_string(self.scorejobs)
        else:
            self.sources.load_from_session(self.request.session)
        self.code_list = self.sources.get_code_list_from_enables()
    def custom_setup(self):
        self.timelog('custom_setup starting')
        if self.dpi is None:
            self.dpi = self.ws.get_dpi_default()
        if self.dpi_thresh is None:
            self.dpi_thresh = 0.9 # intentionally higher than 'preferred'
        # cancel prescreening if sort in URL
        if self.sort:
            self.base_qparms.pop('prescreen_id',None)
            self.prescreen_id = False
        if self.prescreen_id:
            self.context['headline'] = 'Prescreen'
        else:
            self.context['headline'] = 'Scoreboard'
        self.prep_sources()
        if not self.code_list:
            # force config page if no scores enabled
            self.config=self.cols_mode
        elif self.sort is None:
            # sort on first column,descending if no sort specified
            self.sort = SortHandler('-'+self.code_list[0])
        self.timelog('custom_setup before filter setup')
        from dtk.table import ScoreFilter
        self.context_alias(
                score_filter = ScoreFilter(
                                    self.code_list,
                                    self.sources.get_ordering_from_code,
                                    self.filt,
                                    self.sources.get_label_from_code,
                                    ),
                indi_filter = IndFilter(
                                    self.ws,
                                    self.indi,
                                    ),
                deme_filter = DemeritFilter(
                                    self.ws,
                                    self.deme,
                                    ),
                ds_filter = DsFilter(
                                    self.ws,
                                    self.ds,
                                    ),
                )
        self.timelog('custom_setup after filter setup')
        if self.config == self.filt_mode:
            self.button_map={
                    'filter':['filter','indi','deme', 'ds', 'page_size','condense'],
                    }
        elif self.config == self.cols_mode:
            self.button_map={
                    'columns':['columns'],
                    'defaults':[],
                    }
            import dtk.scores as st
            self.source_table = st.SourceTable(self.sources,[
                    st.LabelColType(),
                    st.JobTimeColType(),
                    st.JobStatusColType(self.context),
                    st.EnablesColType(),
                    ])
        else:
            if self.prescreen_id:
                # set up form-based reclassify and reclassify shortcuts;
                # the shortcut buttons require the reclassify form in order
                # to extract the hidden wsa_id field as a cross-check
                # (this is ok because there are no required fields in that
                # form; if this ever becomes an issue the wsa_id can be
                # split into a separate form)
                self.context_alias(shortcut_list=[
                        ('primary','ct_shortcut','Initial Prediction'),
                        ('','',''), # line break
                        ])
                from browse.models import WsAnnotation
                enum=WsAnnotation.indication_vals
                self.add_shortcut_handler(
                        'ct_shortcut',
                        self.reclassify_post_valid,
                        overrides=dict(
                                indication=enum.INITIAL_PREDICTION,
                                demerit=[],
                                indication_href='',
                                )
                        )
                rff = ReclassifyFormFactory()
                for did,label in rff.demerit_choices:
                    button_name = 'demerit_shortcut_%d'%did
                    self.shortcut_list.append(('info',button_name,label))
                    self.add_shortcut_handler(
                            button_name,
                            self.reclassify_post_valid,
                            overrides=dict(
                                    indication=enum.INACTIVE_PREDICTION,
                                    demerit=[did],
                                    indication_href='',
                                    )
                            )
                self.button_map={
                        'reclassify':['reclassify'],
                        }
                for _,name,_ in self.shortcut_list:
                    self.button_map[name] = ['reclassify']
            else:
                self.button_map={
                        'save':['save'],
                        }
            self.button_map['redisplay']=['target']
            # lock in data ordering and filtering; this needs to be done
            # prior to form processing because the reclassify form needs
            # to know the first unclassified wsa on the page
            self.context_alias(
                    key_filter=KeyFilter(),
                    )
            self.score_filter.add_to_key_filter(self.key_filter)
            self.indi_filter.add_to_key_filter(self.key_filter)
            self.deme_filter.add_to_key_filter(self.key_filter)
            self.ds_filter.add_to_key_filter(self.key_filter)

            from browse.models import WsAnnotation
            hidden_drug_ids = set(
                        WsAnnotation.objects.filter(ws=self.ws
                                                ,agent__hide=True
                                                ).values_list('id',flat=True)
                        )
            self.timelog('custom_setup before row_source setup')



            def ordering_excluding_hidden(code,desc):
                return [
                    x
                    for x in self.sources.get_ordering_from_code(code,desc)
                    if x[0] not in hidden_drug_ids
                    ]

            from dtk.table import IdRowSource
            self.context_alias(
                    row_source=IdRowSource(
                                ordering_excluding_hidden,
                                self.sort,
                                self.key_filter,
                                condense=self.condensed,
                                )
                    )
            if self.prescreen_id:
                non_unk = self.ws.get_wsa_id_set('classified')
                for i,wsa_id in enumerate(self.row_source.ordered_ids()):
                    if wsa_id in non_unk:
                        continue
                    try:
                        self.prescreen_wsa = WsAnnotation.objects.get(pk=wsa_id)
                    except WsAnnotation.DoesNotExist:
                        # WSAs can be marked as invalid, in which case the
                        # WsAnnotation.objects manager pretends they don't
                        # exist. Any such objects are not candidates for
                        # pre-screening. (We could also use all_objects to
                        # retrieve the WSA unconditionally, and then check
                        # its attributes, but then any future filtering
                        # changes would need to be duplicated here.)
                        continue
                    self.prescreen_idx = i
                    break
                assert hasattr(self,'prescreen_idx'),"all drugs pre-screened"
                # build flagging HTML, with special re-writes for bulk updates
                from flagging.models import Flag
                flags = Flag.get_for_wsa(self.prescreen_wsa)
                self.unformatted_prescreen_flags = flags
                if flags:
                    for f in flags:
                        if 'clust_screen' in f.href:
                            f.href += '&prescreen=%d'%self.prescreen_id
                            f.new_tab = False
                    self.prescreen_flags = Flag.format_flags(flags)
        self.timelog('custom_setup complete')
    def custom_context(self):
        if self.config == self.filt_mode:
            pass
        elif self.config == self.cols_mode:
            self.column_context()
        else:
            self.table_context()
    def column_context(self):
        columns_form = self.context['columns_form']
        self.context['column_table']=self.source_table.get_table(columns_form)
    def table_context(self):
        from browse.models import WsAnnotation
        # set up pager
        if self.prescreen_id:
            # We want to allow paging during pre-screening, but we want to
            # disable prescreening (i.e. hide the form) if any non-standard
            # sort or filter is specified. So, use a special url builder
            # that passes only the paging parameters and the prescreen
            # flag.  We'll rely on the prescreen flag to pass all the
            # scoreboard configuration when we change pages, and we'll
            # use the presence of any sort or filter queryparm to end
            # prescreening.
            def url_builder(**kwargs):
                from dtk.duma_view import qstr
                return qstr({'prescreen_id':self.prescreen_id},**kwargs)
        else:
            url_builder = self.here_url
        from dtk.table import Pager
        self.context_alias(
                pager=Pager(
                        url_builder,
                        self.row_source.row_count(),
                        self.page_size,
                        self.page,
                        ),
                )
        jump_to_template = 'jump_to_%d'
        prescreen_page_offset = None
        if self.prescreen_id:
            page = self.pager.page_of_idx(self.prescreen_idx)
            if not self.base_qparms.get('page'):
                self.pager.set_page(page)
            if page == self.pager.page_num:
                # set up jump to
                idx = self.prescreen_idx
                prescreen_page_offset=idx % self.pager.page_size
                # we need to set the anchor at least one row above the actual
                # target, because the fixed header hides the first row; if we're
                # not at least this far down the page, don't try to scroll
                # (and, the UX is better with a little more context)
                margin = 2
                if prescreen_page_offset > margin:
                    self.context['jump_to'] = jump_to_template % (idx - margin)
        # get one page of row data
        self.timelog('table_context before get_page')
        rows = self.row_source.get_page(
                        self.sources.get_value_from_code,
                        self.pager,
                        self.code_list,
                        )
        # retrieve wsa objects for page
        qs = WsAnnotation.all_objects.filter(pk__in=[r['_pk'] for r in rows])
        qs = WsAnnotation.prefetch_agent_attributes(qs)
        wsa_cache = {wsa.id:wsa for wsa in qs}
        if any([x.invalid for x in wsa_cache.values()]):
            self.message('WARNING: some of these drugs were removed from the workspace')
        # retrieve target data
        self.timelog('table_context before target retrieval')
        from dtk.prot_map import DpiMapping, AgentTargetCache, protein_link
        targ_cache = AgentTargetCache(
                mapping=DpiMapping(self.dpi),
                agent_ids=[x.agent_id for x in wsa_cache.values()],
                dpi_thresh=self.dpi_thresh,
                )
        note_cache = targ_cache.build_note_cache(
                self.ws,
                self.request.user.username,
                )
        # annotate rows
        for row in rows:
            wsa = wsa_cache[row['_pk']]
            row['wsa'] = wsa
            row['targets'] = targ_cache.info_for_agent(wsa.agent_id)
        self.timelog('table_context after target retrieval')
        # lay out the fixed columns
        ranker = self.row_source.get_ranker()
        from dtk.table import Table
        from dtk.html import link,tie_icon,join,glyph_icon,popover
        def add_jump_to_target(data,row,col):
            return dict(col.cell_attrs,id=jump_to_template%row['_idx'])
        def fmt_protein_links(uniprots):
            return join(*[
                    protein_link(
                            uniprot,
                            gene,
                            self.ws,
                            note_cache=note_cache,
                            #direction=direction,
                            )
                    for uniprot,gene,direction in uniprots
                    ])
        from dtk.duma_view import qstr
        qparms={}
        if self.prescreen_id:
            qparms['prescreen_id'] = self.prescreen_id

        def drug_link(wsa):
            return link(
                    wsa.get_name(self.is_demo()),
                    wsa.drug_url()+qstr({},**qparms),
                    )

        def condensed_drugs_link(wsa):
            wsa_id = wsa.id
            others = self.row_source.condensed_group(wsa_id)
            overflow = False
            others_cnt = len(others)
            # If too many, just cut it off.  Threshold at 20 for now.
            if others_cnt > 20:
                others = others[:20]
                overflow = True
            wsas = WsAnnotation.objects.filter(pk__in=others)
            wsa_map = {wsa.id:wsa for wsa in wsas}
            links = []
            for other in others:
                other_wsa = wsa_map[other]
                link = '<li>' + drug_link(other_wsa)
                links.append(link)
            if overflow:
                links += "<li>...and more (%d total)" % others_cnt

            from django.utils.safestring import mark_safe
            return mark_safe(''.join(links))

        if self.condensed:
            drug_fmt = condensed_drugs_link
        else:
            drug_fmt = drug_link

        columns = [
                Table.Column('Rank',
                        idx='_pk',
                        cell_fmt=lambda x:format_html('{}{}'
                                                    ,ranker.get(x)
                                                    ,tie_icon(ranker,x)
                                                    ),
                        cell_attr_hook=add_jump_to_target,
                        ),
                Table.Column('Drug',
                        idx='wsa',
                        cell_fmt=drug_fmt,
                        ),
                Table.Column('Indication',
                        idx='wsa',
                        cell_fmt=lambda x:x.indication_link()
                        ),
                Table.Column('Targets',
                        idx='targets',
                        cell_fmt=fmt_protein_links,
                        ),
                ]
        if not self.show_drug_names:
            del columns[1]
        # lay out the dynamic columns
        def format_score_cell(x):
            val,attrs = x
            fmt = attrs.get('fmt','%0.2f')
            try:
                val = fmt % float(val)
            except (ValueError,TypeError):
                if val is None:
                    val = ''
            href = attrs.get('href')
            if href:
                val = link(val,href)
            return val
        for code in self.code_list:
            label = self.sources.get_label_from_code(code)
            columns.append(Table.Column(label,
                            code=code,
                            idx=code,
                            cell_fmt=format_score_cell,
                            decimal=True,
                            sort=SortHandler.reverse_mode,
                            ))
        self.context['main_table']=Table(
                        rows,
                        columns,
                        sort_handler=self.sort,
                        url_builder=self.here_url,
                        )
        self.timelog('table_context complete')
    def make_target_form(self,data):
        FormClass = dpi_mapping_form_class(self.dpi,self.dpi_thresh,self.ws)
        form=FormClass(data)
        return form
    def redisplay_post_valid(self):
        p = self.target_form.cleaned_data
        return HttpResponseRedirect(self.here_url(
                    **p
                    ))
    def make_save_form(self,data):
        class FormClass(forms.Form):
            name = forms.CharField(
                        )
        form=FormClass(data)
        form.fields['name'].widget.attrs['placeholder']='assign name'
        return form
    def save_post_valid(self):
        p = self.save_form.cleaned_data
        per_job_enables = {}
        for item in self.sources.get_string_from_enables().split('|'):
            parts = item.split(':')
            per_job_enables[int(parts[0])] = parts[1:]
        sources = []
        from runner.process_info import JobInfo
        for item in self.sources.get_string_from_active_sources().split('|'):
            parts = item.split(':')
            job_id = int(parts[0])
            bji = JobInfo.get_bound(self.ws,job_id)
            sources.append(dict(
                    job_id = job_id,
                    job_type = bji.job.role,
                    label = bji.role_label(),
                    parsed_enables = per_job_enables.get(job_id,[]),
                    ))
        from browse.models import ScoreSet, ScoreSetJob
        ss = ScoreSet(ws=self.ws,
                user=self.request.user.username,
                desc=p['name'],
                sort_score=to_string(self.sort),
                saved_filters=to_string(self.filt),
                )
        ss.save()
        for source in sources:
            ssj = ScoreSetJob(scoreset=ss,**source)
            ssj.save()
        return HttpResponseRedirect(self.here_url(
                    ))
    def make_reclassify_form(self,data):
        rff = ReclassifyFormFactory()
        from browse.models import WsAnnotation
        enum = WsAnnotation.indication_vals
        FormClass = rff.get_form_class(
                self.prescreen_wsa,
                top=(
                        enum.INITIAL_PREDICTION,
                        enum.INACTIVE_PREDICTION,
                        ),
                )
        # Build a list of demerits to pre-set based on drug flags.
        # Since demerits are defined dynamically through a table,
        # we can't hard-code ids here. Instead, create a list of
        # demerit names to set, and then convert them to ids.
        added_demerit_names = set()
        for flag in self.unformatted_prescreen_flags:
            # copy certain demerits from other workspaces
            if flag.category == 'Demerit':
                if flag.detail in (
                        'Ubiquitous',
                        'Unavailable',
                        ):
                    added_demerit_names.add(flag.detail)
            # flag unavailable if not for sale in ZINC
            if flag.category == 'ZINC labels':
                if flag.detail == 'Zinc ID but no label' \
                        or 'not-for-sale' in flag.detail:
                    added_demerit_names.add('Unavailable')
            # flag non-novel if there are unwanted important proteins
            if flag.category == 'Unwanted Important Protein':
                added_demerit_names.add('Non-novel class')
        initial_overrides = {}
        if added_demerit_names:
            initial_overrides['demerit'] = \
                    self.prescreen_wsa.demerits() | set([
                            code
                            for code,desc in rff.demerit_choices
                            if desc in added_demerit_names
                            ])
            initial_overrides['indication'] = enum.INACTIVE_PREDICTION
        return FormClass(
                data,
                initial=initial_overrides,
                )
    def reclassify_post_valid(self,overrides={}):
        p = self.reclassify_form.cleaned_data
        wsa=self.prescreen_wsa
        assert p['wsa_id'] == wsa.id # die if simultaeously edited elsewhere
        if overrides:
            p.update(overrides)
        try:
            wsa.update_indication(
                        p['indication'],
                        p['demerit'],
                        self.request.user.username,
                        self.pscr.marked_because(),
                        p['indication_href'],
                        from_prescreen=self.pscr,
                        )
        except ValueError as ex:
            self.context.update(
                     {'message':"Can't save record."
                     ,'detail':str(ex)
                        +'.  Hit Back, correct the error, and try again.'
                     }
                    )
            return render(self.request,'error.html',self.context)
        self.timelog('before reclassify post redirect')
        return HttpResponseRedirect(self.here_url(
                    prescreen_id=self.pscr.id,
                    ))
    def make_columns_form(self,data):
        FormClass = self.source_table.make_form_class()
        return FormClass(data)
    def columns_post_valid(self):
        p = self.columns_form.cleaned_data
        self.source_table.update_source_data(p)
        return HttpResponseRedirect(self.here_url(
                    config=None,
                    ))
    def defaults_post_valid(self):
        self.sources.load_defaults()
        return HttpResponseRedirect(self.here_url(
                    config='cols',
                    ))
    def make_filter_form(self,data):
        FormClass = self.score_filter.get_filter_form_class()
        return FormClass(data)
    def make_indi_form(self,data):
        FormClass = self.indi_filter.get_filter_form_class()
        return FormClass(data)
    def make_deme_form(self,data):
        FormClass = self.deme_filter.get_filter_form_class()
        return FormClass(data)
    def make_ds_form(self,data):
        FormClass = self.ds_filter.get_filter_form_class()
        return FormClass(data)
    def make_condense_form(self, data):
        class FormClass(forms.Form):
            condensed = forms.BooleanField(
                    initial=self.condensed,
                    label='MoA Condensed',
                    required=False,
                    )
            show_drug_names = forms.BooleanField(
                    initial=self.show_drug_names,
                    label='Show Drug Names',
                    required=False,
                    )
        return FormClass(data)

    def make_page_size_form(self,data):
        class FormClass(forms.Form):
            page_size = forms.IntegerField(
                    initial=self.page_size,
                    )
        return FormClass(data)
    def filter_post_valid(self):
        self.score_filter.update_filter(
                self.filter_form.cleaned_data
                )
        self.indi_filter.update_filter(
                self.indi_form.cleaned_data
                )
        self.deme_filter.update_filter(
                self.deme_form.cleaned_data
                )
        self.ds_filter.update_filter(
                self.ds_form.cleaned_data
                )
        self.page_size = self.page_size_form.cleaned_data['page_size']
        self.condensed = self.condense_form.cleaned_data['condensed']
        self.show_drug_names = self.condense_form.cleaned_data['show_drug_names']
        filt=self.score_filter.get_filter_config()
        if self.prescreen_id:
            from browse.models import PrescreenFilterAudit
            old_filt = PrescreenFilterAudit.latest_filter(self.prescreen_id)
            if filt != old_filt:
                pfa = PrescreenFilterAudit(
                        prescreen_id = self.prescreen_id,
                        saved_filters = filt,
                        user = self.request.user.username,
                        )
                pfa.save()
        return HttpResponseRedirect(self.here_url(
                    filt=filt,
                    indi=self.indi_filter.get_filter_config(),
                    deme=self.deme_filter.get_filter_config(),
                    ds=self.ds_filter.get_filter_config(),
                    page_size=self.page_size,
                    show_drug_names=self.show_drug_names,
                    condensed=self.condensed,
                    page=None,
                    config=None,
                    ))

class JobTypeTable:
    '''Show all jobs of a specified type in tabular form.

    Allows the user to page through all successful jobs with a particular
    job name. Client code can specify the table columns. Among the supported
    columns are parts of an interactive form for selecting and labeling
    jobs (for inclusion as score sources).
    '''
    def __init__(self,ws,jobname,url_builder,page_size,page,dtc='wsa'):
        self.ws = ws
        self.jobname = jobname
        job_qs = Process.objects.filter(
                                name=self.jobname,
                                status=Process.status_vals.SUCCEEDED,
                                ).order_by('-id')
        from dtk.table import Pager
        self.pager = Pager(url_builder,job_qs.count(),page_size,page)
        self._rows = list(job_qs[
                            self.pager.page_start:self.pager.page_end
                            ])
        self._cols = []
        self._table = None
        from runner.process_info import JobInfo
        self.unbound = JobInfo.get_unbound(jobname)
        self._annotate_config()
        self._dtc = dtc
    def page_label(self):
        return self.unbound.source_label(self.jobname)
    def _base_key(self,job): return 'j%d_'%job.id
    def _annotate_config(self):
        settings_defaults = self.unbound.settings_defaults(self.ws)
        for i,job in enumerate(self._rows):
            try:
                # try to get settings for previous job
                settings_defaults['prev'] = self._rows[i+1].settings()
            except IndexError:
                # no previous job; try to remove prev as an option
                try:
                    del(settings_defaults['prev'])
                except KeyError:
                    pass # happens when only one job in list
            config = None
            if job.settings():
                config = str(job.settings())
                config = '{%s}'%', '.join([
                        str(k)+':'+str(v)
                        for k,v in sorted(job.settings().items())
                        ])
                from dtk.text import compare_dict
                for k,v in six.iteritems(settings_defaults):
                    test = compare_dict(k,v,job.settings())
                    if len(test) < len(config):
                        config = test
            job.formatted_settings = config
    def get_select_form_class(self,sources):
        self.current_sources = {
                src.job_id():src.label()
                for src in sources.sources()
                }
        from dtk.dynaform import FormFactory
        from django import forms
        ff = FormFactory()
        for job in self._rows:
            if job.id in self.current_sources:
                continue
            key = self._base_key(job)
            ff.add_field(key+'sel',forms.BooleanField(
                    required=False,
                    ))
            f = forms.CharField(
                    required=False,
                    )
            f.widget.attrs['size']=ff.source_label_field_size
            ff.add_field(key+'lab',f)
        return ff.get_form_class()
    def update_from_post(self,p,sources):
        # extract job ids and labels from form
        adds = []
        labels = {}
        for k,v in six.iteritems(p):
            if k.endswith('_sel') and v:
                adds.append(int(k[1:-4]))
            elif k.endswith('_lab') and v.strip():
                labels[int(k[1:-4])] = v.strip()
        if adds:
            # update source list
            for job_id in adds:
                s = str(job_id)
                if job_id in labels:
                    s += (sources.field_sep+labels[job_id])
                sources.add_from_string(s)
    def add_select_column(self,form):
        from dtk.table import Table
        from dtk.html import button
        self._cols.append(Table.Column(button('Add',name='add'),
                code='sel_field',
                ))
        from dtk.html import glyph_icon
        for job in self._rows:
            if job.id in self.current_sources:
                job.sel_field = glyph_icon('ok')
            else:
                key = self._base_key(job)
                job.sel_field = form[key+'sel']
    def add_label_column(self,form):
        from dtk.table import Table
        self._cols.append(Table.Column('optional label',
                code='lab_field',
                ))
        for job in self._rows:
            if job.id in self.current_sources:
                job.lab_field = self.current_sources[job.id]
            else:
                key = self._base_key(job)
                job.lab_field = form[key+'lab']
    def add_eval_columns(self,dtc='wsa'):
        from dtk.table import Table
        cat = self.unbound.get_data_catalog()
        codes = list(cat.get_codes(dtc,'score'))
        prefix='score_'
        from dtk.enrichment import EnrichmentMetric, MetricProcessor

        metric_processor = MetricProcessor()

        for code in codes:
            for em_name in self.unbound.enrichment_metrics():
                EM = EnrichmentMetric.lookup(em_name)
                self._cols.append(Table.Column(
                        f'{cat.get_label(code)} {EM.label()}',
                        code=prefix+code+em_name,
                        ))
        for job in self._rows:
            for code in codes:
                for em_name in self.unbound.enrichment_metrics():
                    table_key = prefix + code + em_name
                    try:
                        metric_val = metric_processor.compute(
                                metric=em_name,
                                ws_or_id=self.ws,
                                job_or_id=job,
                                code=code
                                )
                        v = f'{metric_val:.2f}'
                    except ValueError:
                        # This can happen if e.g. the set of codes for a CM
                        # changed and old jobs are missing some.
                        v = ''
                    setattr(job,table_key,v)
    def add_settings_column(self):
        from dtk.table import Table
        self._cols.append(Table.Column('Settings',
                code='formatted_settings',
                ))
    def add_past_settings_column(self,qparms):
        from dtk.html import link,join
        from dtk.duma_view import qstr
        # create a default settings link
        self.load_defaults_html=link(
                "Load default settings",
                self.ws.reverse('nav_job_start',self.jobname)+qstr(qparms),
                )
        # define an extractor to make a link with the row as the
        # copy_job in the URL, and the settings as text
        def extract(row):
            path=self.ws.reverse('nav_job_start',self.jobname,row.id)
            return join(
                    row.role+':',
                    link(row.formatted_settings,path+qstr(qparms)),
                    )
        # define the column
        from dtk.table import Table
        self._cols.append(Table.Column('Settings',
                extract=extract
                ))
    def add_date_column(self):
        from dtk.table import Table
        from dtk.text import fmt_time
        self._cols.append(Table.Column('Date',
                code='completed',
                cell_fmt=lambda x:fmt_time(x,fmt='%Y-%m-%d %H:%M')
                ))
    def add_jobid_column(self):
        from dtk.table import Table
        def extract(row):
            return row.id
        self._cols.append(Table.Column('Job ID',
                 extract=extract
                ))
    def add_date_elapsed_column(self):
        from dtk.table import Table
        from dtk.text import fmt_time,fmt_delta
        def extract(row):
            return '%s (%s)'%(
                    fmt_time(row.completed,fmt='%Y-%m-%d %H:%M'),
                    fmt_delta(row.completed,row.started)
                    )
        self._cols.append(Table.Column('Date',
                extract=extract,
                ))
    def add_status_column(self,context):
        from dtk.table import Table
        from runner.templatetags.process import job_summary_impl
        from django.utils.safestring import mark_safe
        self._cols.append(Table.Column('Status',
                extract=lambda x:mark_safe(job_summary_impl(
                                            context,
                                            x,
                                            #'bottom',
                                            )),
                ))
    def get_table(self):
        if self._table is None:
            from dtk.table import Table
            self._table = Table(self._rows,self._cols)
        return self._table

from dtk.duma_view import from_string,to_string

# XXX this might still have pieces that can reasonably be factored out;
# XXX for example, there may be common code with labeled source lists
class MetricColumn(Table.Column):
    label_delim = ':'
    code_part_delim = '_'
    @classmethod
    def _default_label(cls,code):
        return code.replace(cls.code_part_delim,' ')
    def __init__(self,code,label=None):
        if not label:
            label = self._default_label(code)
        from tools import sci_fmt
        super(MetricColumn,self).__init__(
                            label=label,
                            code=code,
                            cell_fmt=sci_fmt,
                            sort=SortHandler.reverse_mode,
                            )
        code_parts = code.split(self.code_part_delim,2)
        self.score_code = code_parts[0]
        self.ds_name = code_parts[1]
        from dtk.enrichment import EnrichmentMetric
        self.MetricClass = EnrichmentMetric.lookup(code_parts[2])
    @classmethod
    def from_string(cls,s):
        if cls.label_delim in s:
            code,label = s.split(cls.label_delim)
        else:
            code = s
            label = None
        return cls(code,label)
    def to_string(self):
        code = self.code_part_delim.join([
                self.score_code,
                self.ds_name,
                self.MetricClass.__name__,
                ])
        if self.label == self._default_label(code):
            return code
        return self.label_delim.join([code,self.label])
    class MetricExtractor:
        def __init__(self,ws):
            self.ws = ws
            self.ds_cache = {}
        def extract_metrics(self,bji,metrics):
            from dtk.enrichment import EMInput
            from dtk.files import Quiet
            result = {}
            cat = bji.get_data_catalog()
            score_cache = {}
            for mcol in metrics:
                if mcol.score_code in score_cache:
                    ordering = score_cache[mcol.score_code]
                else:
                    try:
                        ordering = cat.get_ordering(mcol.score_code,True)
                    except ValueError:
                        # allow heterogeneous job lists, where not all jobs
                        # have the same score codes
                        ordering = []
                    score_cache[mcol.score_code] = ordering
                if mcol.ds_name in self.ds_cache:
                    ds = self.ds_cache[mcol.ds_name]
                else:
                    ds = self.ws.get_wsa_id_set(mcol.ds_name)
                    self.ds_cache[mcol.ds_name] = ds
                emi = EMInput(ordering,ds)
                em = mcol.MetricClass()
                with Quiet() as tmp:
                    em.evaluate(emi)
                result[mcol.code] = em.rating
            return result

class SettingColumn(Table.Column):
    label_delim = ':'
    def __init__(self,code,label=None):
        if not label:
            label = code
        super(SettingColumn,self).__init__(
                            label=label,
                            code=code,
                            sort=True,
                            )
    @classmethod
    def from_string(cls,s):
        if cls.label_delim in s:
            code,label = s.split(cls.label_delim)
        else:
            code = s
            label = None
        return cls(code,label)
    def to_string(self):
        if self.code == self.label:
            return self.code
        return self.label_delim.join([self.code,self.label])

def build_label_list_form_class(ll,code_template='label%d'):
    from dtk.dynaform import FormFactory
    ff = FormFactory()
    for i,item in enumerate(ll):
        field_code = code_template % i
        ff.add_field(field_code,forms.CharField(
                initial=item.label,
                required=False,
                ))
        item.field_code = field_code
    return ff.get_form_class()

def update_label_list_from_dict(ll,d):
    for i in range(len(ll)-1,-1,-1):
        rec = ll[i]
        v = d[rec.field_code]
        if v:
            rec.label = v
        else:
            del(ll[i])

class RefreshQCPlotsView(DumaView):
    template_name='nav/refresh_qc_plots.html'
    GET_parms={
            'group':(str,None),
            'jobs':(list_of(int),''),
            }
    def custom_setup(self):
        from runner.process_info import JobInfo
        self.job_info_list = []
        div_count=0
        for job_id in self.jobs:
            bji = JobInfo.get_bound(self.ws,job_id)
            plot_list=[]
            for plot in bji.qc_plots():
                plot_list.append((plot,'plotly_div'+str(div_count)))
                div_count += 1
            report_list = bji.get_reported_info()
            self.job_info_list.append((bji,plot_list,report_list))

def make_refresh_choices(ws):
    from dtk.text import fmt_time
    from browse.models import ScoreSet
    qs=ScoreSet.objects.filter(ws=ws,desc='RefreshFlow')
    ss_choices = [
            (ss.id,'(%s) %s @ %s'%(ss.id, ss.user,fmt_time(ss.created)))
            for ss in qs.order_by('-id')
            ]
    return ss_choices

class RefreshQCView(DumaView):
    template_name='nav/refresh_qc.html'
    button_map={
            'redisplay':['config'],
            }
    GET_parms={
            'scoreset':(int,None),
            }

    def custom_setup(self):
        self.ss_choices = make_refresh_choices(self.ws)
        if self.ss_choices and self.scoreset is None:
            self.scoreset = self.ss_choices[0][0]
    def custom_context(self):
        from browse.models import ScoreSet
        if not self.scoreset:
            return
        ss = ScoreSet.objects.get(pk=self.scoreset)
        from dtk.warnings import get_scoreset_warning_summary,output_warnings
        warnings = get_scoreset_warning_summary(self.ws,ss)
        if warnings:
            output_warnings(self.request,warnings)
        self._build_workflow_history(ss)
        self.output = []
        self._load_ge_qc_links()
        self._load_gwas_qc_links()
        job_roles = self._extract_job_roles()
        self._make_workflow_part_table(job_roles)
        self._report_dpi_settings(job_roles)

        from dtk.html import link
        self.output.append(link(
            "Score correlation",
            self.ws.reverse('nav_score_corr') + f'?scoreset={self.scoreset}',
            new_tab=True
            ))
    def _get_refresh_workflow_default_settings(self):
        from runner.process_info import JobInfo
        uji = JobInfo.get_unbound('wf')
        d = uji.settings_defaults(self.ws)
        l = [x for x in d.keys() if x.endswith('RefreshFlow')]
        if len(l) != 1:
            return {}
        result = d[l[0]]
        # avoid showing None vs '' diffs
        rss_key = 'resume_scoreset_id'
        if rss_key in result:
            if result[rss_key] is None:
                result[rss_key] = ''
        return result
    def _build_workflow_history(self,ss):
        from runner.templatetags.process import job_summary_impl
        from dtk.text import fmt_time,compare_refresh_wf_settings
        defaults = self._get_refresh_workflow_default_settings()
        rows = [
                (
                  f'({p.id}) {p.user}@{fmt_time(p.completed)}',
                  job_summary_impl(self.context,p,'log'),
                  compare_refresh_wf_settings('default',defaults,p.settings()),
                  )
                for p in ss.get_contributing_workflow_jobs()
                ]
        from dtk.table import Table
        self.context['workflow_jobs'] = Table(rows,columns=[
                Table.Column('Workflow Job',
                        idx=0,
                        ),
                Table.Column('Status',
                        idx=1,
                        ),
                Table.Column('Settings',
                        idx=2,
                        ),
                ])
    def _report_dpi_settings(self,job_roles):
        # from the joblist, get all the Process entries so we can examine
        # their settings
        plist = list(Process.objects.filter(pk__in=job_roles.keys()))
        # get [(job_id,dpi_file),...] for all jobs with a dpi file configured]
        dpi_choices = []
        for proc in plist:
            dpi = None
            for key in ['p2d_file','dpi_file']:
                if key in proc.settings():
                    dpi = proc.settings()[key]
                    break
            if not dpi:
                continue
            dpi_choices.append((proc.id,dpi))
        # now find most common choice, and report any differences
        if dpi_choices:
            # XXX force error for testing
            # dpi_choices[0] = (dpi_choices[0][0],'DNChBX_ki.v15')
            from collections import Counter
            counts = Counter([x[1] for x in dpi_choices]).most_common()
            base_dpi = counts[0][0]
            self.context['base_dpi'] = base_dpi
            other_dpi = []
            for job_id,dpi in dpi_choices:
                if dpi == base_dpi:
                    continue
                other_dpi.append((job_roles[job_id],job_id,dpi))
            self.context['other_dpi'] = other_dpi
    def _extract_job_roles(self):
        '''Return {job_id:role,...} while checking for duplicates.'''
        from browse.models import ScoreSetJob
        from dtk.data import MultiMap
        mm = MultiMap(ScoreSetJob.objects.filter(
                scoreset_id=self.scoreset,
                ).values_list('job_id','job_type'))
        job_roles = dict(MultiMap.flatten(mm.fwd_map()))
        # warn about duplicate roles or job_ids
        # XXX force error for testing
        #mm = MultiMap([ (1,'a'), (2,'b'), (2,'c'), (3,'d'), (4,'d') ])
        for k,s in mm.fwd_map().items():
            if len(s) > 1:
                self.message(f'job {k} has multiple roles: '+', '.join(s))
        for k,s in mm.rev_map().items():
            if len(s) > 1:
                self.message(f'role {k} has multiple job ids: '+', '.join(
                        str(x) for x in s
                        ))
        return job_roles
    def _make_workflow_part_table(self,job_roles):
        from workflows.refresh_workflow import StandardWorkflow
        swf = StandardWorkflow(ws=self.ws)
        # gather names of CMs that drive each workflow part
        #
        # customsig requires lots of special handling; a workspace can
        # have multiple customsigs, each of which is treated as if
        # the stem is customsig_jobid to make it unique. But that makes
        # the role parsing complicated. So:
        # - here, we assemble the full stem name
        # - in the stemset, we fold all those down to a single 'customsig'
        # - in the parsing, we re-expand the stem to include the jobid
        #   after successfully matching the 'customsig' part
        stems=[]
        for wf_part in swf.pre_eff_parts:
            cm_name = wf_part.__class__.__module__.split('_')[-1]
            if cm_name == 'customsig':
                stems.append(f'{cm_name}_{wf_part.jid}')
            else:
                stems.append(cm_name)
        enable_idxs = set(swf.get_refresh_part_initial())
        enables = [
                i in enable_idxs
                for i,x in enumerate(swf.pre_eff_parts)
                ]
        labels = [
                x.label
                for x in swf.pre_eff_parts
                ]
        # add in base FAERS, so it doesn't land in 'other'; stick this in
        # front of 'capp', which is the first faers-related part, and
        # copy capp's enable
        faers_idx = stems.index('capp')
        stems.insert(faers_idx,'faers')
        enables.insert(faers_idx,enables[faers_idx])
        labels.insert(faers_idx,'FAERS extraction')
        stemset = set(
                'customsig' if x.startswith('customsig_') else x
                for x in stems
                )
        by_stem = {}
        for job_id,role in job_roles.items():
            prefix,stem,suffix = self._parse_role(role,stemset)
            d = by_stem.setdefault(stem,{})
            d.setdefault(prefix,set()).add((suffix,job_id))
        from dtk.html import link,ulist
        from dtk.duma_view import qstr
        subpage = self.ws.reverse('nav_refresh_qc_plots')
        rows = [
                (label,enable,ulist([
                        link('_'.join([
                                x
                                for x in (prefix,stem)
                                if x
                                ]),
                                subpage+qstr(dict(
                                        group=prefix,
                                        jobs=','.join(str(x[1]) for x in s),
                                        )),
                                new_tab=True,
                                )
                        for prefix,s in by_stem.get(stem,{}).items()
                        ]))
                for stem,enable,label in zip(stems,enables,labels)
                ]
        from dtk.table import Table
        def flag_inconsistent_rows(data,row,table):
            data_present = row[2] != ulist([])
            if data_present != row[1]:
                # Match the message box class alert-danger. This is
                # coded as hex in the min.css file, and I can't
                # find a corresponding color name.
                return {'bgcolor':'#f2dede'}
            return {}
        self.context['workflow_parts'] = Table(rows,columns=[
                Table.Column('Workflow Part',
                        idx=0,
                        ),
                Table.Column('Enabled by default',
                        idx=1,
                        cell_attr_hook=flag_inconsistent_rows,
                        ),
                Table.Column('Outputs',
                        idx=2,
                        ),
                ])
        agg_jobs = []
        is_agg = lambda x:x.startswith('wzs') or x.startswith('eff')
        s = by_stem.get('other',{}).get('',set())
        for suffix,job_id in sorted(s,key=lambda x:x[1]):
            if is_agg(suffix):
                agg_jobs.append((suffix,job_id))
            else:
                self.message(f'Unexpected job type: {suffix} ({job_id})')
        self.context['agg_jobs'] = agg_jobs
        if False:
            print()
            for stem,enable,label in zip(stems,enables,labels):
                print(stem,enable,label)
                d = by_stem.get(stem,{})
                for prefix,s in d.items():
                    print(' ',prefix,s)
        if False:
            print('other')
            d = by_stem.get('other',{})
            for prefix,s in d.items():
                print(' ',prefix,s)
        if False:
            pipelines = {}
            for stem in stems+['other']:
                d = by_stem.get(stem,{})
                for prefix,s in d.items():
                    pipelines.setdefault(frozenset(x[0] for x in s),set()).add(
                            prefix+'_'+stem
                            )
            print()
            for p,s in pipelines.items():
                print(p,s)
    def _parse_role(self,role,stemset):
        '''Return (prefix,stem,suffix) for role.

        Role is split around a stem, which is the rightmost part of the
        role that matches something in stemset. In general:
        - stem is a data source CM
        - prefix identifies distinct incarnations of the source CM
        - suffix is a point in the downstream processing pipeline
        Special-case rewriting is done if necessary for CMs where this
        convention isn't followed based on a lexical split of the role
        name alone.
        '''
        parts = role.split('_')
        for idx,part in reversed(list(enumerate(parts))):
            if part in stemset:
                if part == 'customsig':
                    assert idx == 0
                    return (
                            '',
                            '_'.join(parts[:2]),
                            '_'.join(parts[2:]),
                            )
                idx2 = idx+1
                return (
                        '_'.join(parts[:idx]),
                        '_'.join(parts[idx:idx2]),
                        '_'.join(parts[idx2:]),
                        )
        return ('','other',role)
    def _load_gwas_qc_links(self):
        from dtk.html import link
        from dtk.duma_view import qstr
        self.output.append(link(
                "GWAS dataset table",
                self.ws.reverse('gwas_search')+qstr(dict(show='selected')),
                new_tab=True
                ))
    def _load_ge_qc_links(self):
        from dtk.html import link
        from dtk.duma_view import qstr
        for (id,label) in self.ws.get_tissue_set_choices():
            self.output.append(link(
                label+" GE dataset QC summary",
                self.ws.reverse('ge:tissue_set_analysis')+qstr(dict(tissue_set_id=id)),
                    new_tab=True
                ))
            self.output.append(link(
                label+" GE dataset correlation",
                self.ws.reverse('ge:tissue_corr')+qstr(dict(tissue_set_id=id)),
                new_tab=True
                ))
            self.output.append(link(
                label+" GE dataset stats",
                self.ws.reverse('ge:tissue_stats', id),
                new_tab=True
                ))
    def make_config_form(self,data):
        class MyForm(forms.Form):
            scoreset = forms.ChoiceField(
                    choices=self.ss_choices,
                    initial=self.scoreset,
                    )
        return MyForm(data)
    def redisplay_post_valid(self):
        p = self.config_form.cleaned_data
        return HttpResponseRedirect(self.here_url(**p))

class ScoresetView(DumaView):
    # XXX most of the complication of this view is due to the ability
    # XXX to dynamically add attribute and metric columns; this is
    # XXX not set up in a way that's very useful for score sets from
    # XXX multiple CMs; in particular, most of the links generated in
    # XXX the 'compare' tables at the bottom are broken
    template_name='nav/scoreset.html'
    index_dropdown_stem='nav_scoreset_list'
    GET_parms = dict(
            config=(int,0),
            metrics=(list_of(MetricColumn),''),
            attribs=(list_of(SettingColumn),''),
            sort=(SortHandler,''),
            )
    button_map={
            'display':['metric_labels','attrib_labels'],
            'new_metric':['metric','metric_labels','attrib_labels'],
            'new_attrib':['attrib','metric_labels','attrib_labels'],
            }
    def custom_setup(self):
        # build a list of ScoreSetJob rows, and annotate with job row
        self.score_data = list(self.scoreset.scoresetjob_set.all())
        if not self.score_data:
            # empty scoreset; disallow config, display empty table
            self.config=0
            self.button_map={}
            return
        from runner.models import Process
        for ssj in self.score_data:
            job = Process.objects.get(pk=ssj.job_id)
            setattr(ssj,'job',job)
        if self.config:
            # get all settings key/value pairs as a set;
            # some settings values aren't hashable, so
            # convert values to strings
            self.all_settings = [
                    set((k,str(v)) for k,v in x.job.settings().items())
                    for x in self.score_data
                    ]
            # get settings common to all jobs
            self.common_settings = set((
                    pair
                    for pair in self.all_settings[0]
                    if all((
                            pair in x
                            for x in self.all_settings[1:]
                            ))
                    ))
        else:
            # only set up for remove buttons
            self.button_map={'remove':[], 'removedeps': []}
        from dtk.html import link
        self.context['config_link'] = link('Configure',self.here_url(config=1))
    def custom_context(self):
        if self.config:
            self.custom_config_context()
        else:
            self.custom_display_context()
    def custom_display_context(self):
        # generate header data
        ei = []
        self.context['extra_info'] = ei
        if self.scoreset.sort_score:
            ei.append(('sort score',self.scoreset.sort_score))
        if self.scoreset.saved_filters:
            ei.append(('saved filters',self.scoreset.saved_filters))
        if self.scoreset.migrated_from:
            ei.append(('migrated from',self.scoreset.migrated_from))
        # Annotate runs with the requested metric data and attribute data.
        # Only enter the loop if we've got extra columns configured,
        # because otherwise we end up fetching LTS data unneccesarily.
        from runner.process_info import JobInfo
        if self.attribs or self.metrics:
            me = MetricColumn.MetricExtractor(self.ws)
            for ssj in self.score_data:
                job = ssj.job
                for scol in self.attribs:
                    setattr(ssj,scol.code,job.settings().get(scol.code))
                bji = JobInfo.get_bound(self.ws,job)
                for k,v in six.iteritems(me.extract_metrics(bji,self.metrics)):
                    setattr(ssj,k,v)
        # Patch in a cell_html_hook function to show metric scores as links.
        # Constructing the links requires information from the row
        # (job id), the MetricColumn (score_code, etc.) and the view
        # (ws), as well as the already-extracted text.  All of these
        # but the view are passed to this hook function, and the view can
        # be accessed via a closure.
        def formatter(v,row,mc):
            from dtk.enrichment import DEA_AREA
            opts = {
                    'score': '%d_%s'%(row.job.id,mc.score_code),
                    'ds': mc.ds_name,
                    }
            score_page='nav_scoreplot'
            from django.utils.http import urlencode
            url=self.ws.reverse(score_page,'wsa')+'?'+urlencode(opts,doseq=True)
            from dtk.html import link
            return link(v,url)
        for mc in self.metrics:
            mc.cell_html_hook = formatter
        # sort data as requested
        if self.sort.colspec:
            self.score_data.sort(
                    key=lambda x:getattr(x,self.sort.colspec),
                    reverse=self.sort.minus,
                    )
        # build main table
        from runner.templatetags.process import job_summary_impl
        from django.utils.safestring import mark_safe
        from dtk.html import tag_wrap,join
        from django.middleware.csrf import get_token
        token = get_token(self.request)
        def remove_button_hook(data,row,column):
            # return form w/ button + id in hidden field
            return tag_wrap('form',
                    join(
                            tag_wrap('input',None,attr={
                                    'type':'hidden',
                                    'name':'csrfmiddlewaretoken',
                                    'value':token,
                                    }),
                            tag_wrap('input',None,attr={
                                    'type':'hidden',
                                    'name':'row_id',
                                    'value':row.id,
                                    }),
                            tag_wrap('button','-',attr={
                                    'type':'submit',
                                    'name':'remove_btn',
                                    'class':'btn btn-danger',
                                    }),
                            tag_wrap('button','-Deps',attr={
                                    'type':'submit',
                                    'name':'removedeps_btn',
                                    'class':'btn btn-danger',
                                    }),
                            ),
                    attr={
                            'method':'post',
                            },
                    )
        columns=[
                Table.Column('Job Type',
                        sort='l2h',
                        ),
                Table.Column('Job Summary',
                        extract=lambda x:mark_safe(
                                job_summary_impl(
                                        self.context,
                                        x.job,
                                        'bottom',
                                        )
                                ),
                        ),
                ]
        if self.scoreset.desc == 'RefreshFlow':
            columns.insert(0,
                    Table.Column('- Remove',
                            cell_html_hook=remove_button_hook,
                            )
                    )
        self.context_alias(table=Table(
                self.score_data,
                columns+self.attribs+self.metrics,
                sort_handler=self.sort,
                url_builder=self.here_url,
                ))
        # build score cross-compare tables
        from dtk.html import tag_wrap,link
        from dtk.duma_view import qstr
        comp_page=self.ws.reverse('nav_score_cmp', 'wsa')
        def compare_cell(data,row,col):
            if col.position >= row.position:
                return ''
            return link('compare',comp_page+qstr(dict(
                                    x=col.score,
                                    y=row.score,
                                    )))
        def embolden(data):
            return tag_wrap('b',data)
        from collections import OrderedDict
        scores = OrderedDict()
        class DummyClass():
            pass
        for mc in self.metrics:
            key = mc.score_code
            if key in scores:
                continue
            info = DummyClass()
            scores[key] = info
            info.label = key
            run_cols = []
            for i,ssj in enumerate(self.score_data):
                col = Table.Column(ssj.job_type,
                        cell_html_hook=compare_cell,
                        )
                col.position = i
                col.score='%d_%s' % (ssj.job_id,key)
                run_cols.append(col)
            info.table = Table(list(reversed(run_cols))[:-1],[
                            Table.Column('Job Type',
                                    code='label',
                                    cell_fmt=embolden,
                                    ),
                            ]+run_cols[:-1]
                    )
        self.context_alias(compare_scores=list(scores.values()))
    def remove_post_valid(self):
        to_remove = self.request.POST['row_id']
        from browse.models import ScoreSetJob
        ssj = ScoreSetJob.objects.get(pk=to_remove)
        self.log('removing job %d from scoreset %d (type %s)',
                ssj.job_id,
                ssj.scoreset_id,
                ssj.job_type,
                )
        ssj.delete()
        return HttpResponseRedirect(self.here_url())
    @transaction.atomic
    def removedeps_post_valid(self):
        from browse.models import ScoreSetJob, ScoreSet
        to_remove = self.request.POST['row_id']
        ssj = ScoreSetJob.objects.get(pk=to_remove)
        ss = ssj.scoreset
        dep_jids = ss.get_dependents([ssj.job_id])
        to_deletes = [ssj] + list(ScoreSetJob.objects.filter(job_id__in=dep_jids, scoreset=ss))
        for to_delete in to_deletes:
            self.log('removing job %d from scoreset %d (type %s)',
                    to_delete.job_id,
                    to_delete.scoreset_id,
                    to_delete.job_type,
                    )
            to_delete.delete()
        return HttpResponseRedirect(self.here_url())
    def custom_config_context(self):
        mlabel_form = self.context['metric_labels_form']
        self.context_alias(metric_table=Table(self.metrics,[
                    Table.Column('Label',
                            extract=lambda x:mlabel_form[x.field_code],
                            ),
                    Table.Column('Code'),
                    Table.Column('Type',
                            extract=lambda x:'Metric',
                            ),
                    ]))
        alabel_form = self.context['attrib_labels_form']
        self.context_alias(attrib_table=Table(self.attribs,[
                    Table.Column('Label',
                            extract=lambda x:alabel_form[x.field_code],
                            ),
                    Table.Column('Code'),
                    Table.Column('Type',
                            extract=lambda x:'Attribute',
                            ),
                    ]))
    def make_metric_labels_form(self,data):
        FormClass = build_label_list_form_class(
                        self.metrics,
                        code_template='mlabel%d',
                        )
        return FormClass(data)
    def make_attrib_labels_form(self,data):
        FormClass = build_label_list_form_class(
                        self.attribs,
                        code_template='alabel%d',
                        )
        return FormClass(data)
    def update_labels(self):
        form = self.context['attrib_labels_form']
        update_label_list_from_dict(self.attribs,form.cleaned_data)
        form = self.context['metric_labels_form']
        update_label_list_from_dict(self.metrics,form.cleaned_data)
    def display_post_valid(self):
        self.update_labels()
        return HttpResponseRedirect(self.here_url(
                attribs=self.attribs,
                metrics=self.metrics,
                config=None,
                sort=None,
                ))
    def make_attrib_form(self,data):
        # find all settings keys which vary across runs
        import operator
        all_pairs = reduce(operator.or_,self.all_settings)
        attribs = set([
                pair[0]
                for pair in all_pairs
                if pair not in self.common_settings
                ])
        attribs = sorted(attribs)
        class MyForm(forms.Form):
            attrib=forms.ChoiceField(
                    choices=[(x,x) for x in attribs],
                    label='Job Settings Attribute',
                    )
        return MyForm(data)
    def new_attrib_post_valid(self):
        form = self.context['attrib_form']
        p = form.cleaned_data
        code = p['attrib']
        if code in [x.code for x in self.attribs]:
            form.add_error(None,'this column already exists')
            return
        self.update_labels()
        self.attribs.append(SettingColumn(code))
        return HttpResponseRedirect(self.here_url(
                attribs=self.attribs,
                metrics=self.metrics,
                ))
    def make_metric_form(self,data):
        from runner.process_info import JobInfo
        scores = set()
        for ssj in self.score_data:
            bji=JobInfo.get_bound(self.ws,ssj.job)
            cat = bji.get_data_catalog()
            for code in cat.get_codes('wsa','score'):
                scores.add(code)
        scores = sorted(scores)
        from dtk.enrichment import EnrichmentMetric
        metrics = sorted(EnrichmentMetric.get_all_names())
        class MyForm(forms.Form):
            score=forms.ChoiceField(
                    choices=[(x,x) for x in scores],
                    )
            metric=forms.ChoiceField(
                    choices=[(x,x) for x in metrics],
                    )
            ds=forms.ChoiceField(
                    choices=self.ws.get_wsa_id_set_choices(),
                    label='Reference Drug Set',
                    initial=self.ws.eval_drugset,
                    )
        return MyForm(data)
    def new_metric_post_valid(self):
        form = self.context['metric_form']
        p = form.cleaned_data
        code = MetricColumn.code_part_delim.join([
                p['score'],
                p['ds'],
                p['metric'],
                ])
        if code in [x.code for x in self.metrics]:
            form.add_error(None,'this column already exists')
            return
        self.update_labels()
        self.metrics.append(MetricColumn(code))
        return HttpResponseRedirect(self.here_url(
                attribs=self.attribs,
                metrics=self.metrics,
                ))

class ScoresetListView(DumaView):
    template_name='nav/scoreset_list.html'
    index_dropdown_stem='nav_scoreset_list'
    button_map={
            'load':[],
            }
    def custom_setup(self):
        from browse.models import ScoreSet
        qs = ScoreSet.objects.filter(ws=self.ws).order_by('-id')
        self.context['scoresets'] = qs
    def load_post_valid(self):
        from dtk.scores import SourceListWithEnables
        sl = SourceListWithEnables(self.ws)
        sl.load_from_session(self.request.session)
        from browse.models import ScoreSet
        ss = ScoreSet.objects.get(pk=self.request.POST['scoreset_id'])
        sl.load_from_string(ss.source_list_source_string())
        sl.set_enables_from_string(ss.source_list_enable_string())
        from dtk.duma_view import qstr
        d = {}
        if ss.sort_score:
            d['sort'] = ss.sort_score
        if ss.saved_filters:
            d['filt'] = ss.saved_filters
        return HttpResponseRedirect(
                self.ws.reverse('nav_scoreboard')+qstr(d)
                )

class ScoresetEditView(DumaView):
    template_name='nav/scoreset_edit.html'
    button_map={
            'save':['edit'],
            }
    def make_edit_form(self,data):
        from browse.models import ScoreSet
        class MyForm(forms.ModelForm):
            class Meta:
                model = ScoreSet
                fields = [
                        'desc',
                        'sort_score',
                        'saved_filters',
                        ]
                widgets = {
                        'desc':forms.TextInput(attrs={'size':'100'}),
                        'saved_filters':forms.TextInput(attrs={'size':'100'}),
                        }
                labels = {
                        'desc':'Name',
                        }
        return MyForm(data,instance=self.scoreset)
    def save_post_valid(self):
        self.edit_form.save()
        return HttpResponseRedirect(self.ws.reverse('nav_scoreset_list'))

class AddToScoresetView(DumaView):
    template_name='nav/add_to_scoreset.html'
    button_map={
            'add':['add'],
            }
    def custom_setup(self):
        from runner.process_info import JobInfo
        self.context_alias(job_info=JobInfo.get_bound(self.ws,self.job))
        from browse.models import ScoreSet
        from dtk.text import fmt_time
        qs=ScoreSet.objects.filter(ws=self.ws)
        self.ss_choices = [
                (ss.id,'%s (%s@%s)'%(ss.desc,ss.user,fmt_time(ss.created)))
                for ss in qs.order_by('-id')
                ]
    def make_add_form(self,data):
        class MyForm(forms.Form):
            scoreset = forms.ChoiceField(
                    choices=self.ss_choices,
                    label='Add to',
                    )
            job_type = forms.CharField(
                    initial=self.job.role,
                    widget=forms.TextInput(attrs={'size':40}),
                    )
            replace = forms.ChoiceField(
                    choices=[
                            ('deps','Job exactly matching this job type and its dependencies'),
                            ('exact','Jobs exactly matching job type'),
                            ('none',"Don't remove anything from scoreset"),
                            ],
                    )
        return MyForm(data)
    def add_post_valid(self):
        p = self.add_form.cleaned_data
        self.remove_old(p)
        from browse.models import ScoreSetJob
        ssj = ScoreSetJob(
                scoreset_id=p['scoreset'],
                job_id=self.job.id,
                job_type=p['job_type'],
                )
        self.log('adding job %d from scoreset %d (type %s)',
                ssj.job_id,
                int(ssj.scoreset_id),
                ssj.job_type,
                )
        ssj.save()
        return HttpResponseRedirect(
                self.ws.reverse('nav_scoreset',p['scoreset'])
                )
    def remove_old(self,p):
        mode = p['replace']
        if mode == 'none':
            return
        from browse.models import ScoreSetJob
        qs = ScoreSetJob.objects.filter(scoreset_id=p['scoreset'], job_type=p['job_type'])
        for ssj in qs:
            ss = ssj.scoreset
            dep_jids = ss.get_dependents([ssj.job_id])
            to_deletes = [ssj]
            if mode == 'deps':
                to_deletes += list(ScoreSetJob.objects.filter(job_id__in=dep_jids, scoreset=ss))
            for to_delete in to_deletes:
                self.log('removing job %d from scoreset %d (type %s)',
                        to_delete.job_id,
                        to_delete.scoreset_id,
                        to_delete.job_type,
                        )
                to_delete.delete()


class PrescreenListView(DumaView):
    extra_scores_help='Format is job_id:score:score|job_id:score...'
    template_name='nav/prescreen_list.html'
    button_map={
            'add':['add'],
            'moaupdate':[],
            }
    def custom_context(self):
        from browse.models import Prescreen
        from rvw.prescreen import PrescreenOrdering
        prescreens = Prescreen.objects.filter(ws=self.ws).order_by('-id')
        next_mols = [PrescreenOrdering.next_mol_to_prescreen(pscr) for pscr in prescreens]
        self.context_alias(
                data = zip(prescreens, next_mols),
                sb_url = self.ws.reverse('nav_scoreboard'),
                )
    def get_defaults(self):
        from runner.process_info import JobInfo
        prev = self.ws.get_prev_job_choices('selectability')
        if not prev:
            return '', '', ''
        prev_jid = prev[0][0]
        bji = JobInfo.get_bound(self.ws, prev_jid)
        settings = bji.job.settings()
        cnt = settings['count']
        unit = 'MOAs' if settings.get('condensed') else 'molecules'
        addition = f'{cnt} {unit}'
        wzs_jid = settings['wzs_jid']
        wzs_bji = JobInfo.get_bound(self.ws, wzs_jid)
        extras = f'{wzs_jid}:wzs'

        if 'apr_jid' in settings:
            extras += f'|{settings["apr_jid"]}:avg'

        if prev:
            return f'{prev_jid}_liveselectability',extras,f'{wzs_bji.job.get_note_text()}; {addition}'
        else:
            return '','',''

    def make_add_form(self,data):
        from browse.models import Prescreen
        view=self
        prim,es,note = self.get_defaults()
        class MyForm(forms.ModelForm):
            class Meta:
                model = Prescreen
                fields = [
                        'name',
                        'primary_score',
                        'extra_scores',
                        ]
            name = forms.CharField(
                    initial=note,
                    )
            primary_score = forms.CharField(
                    initial=prim,
                    )
            extra_scores = forms.CharField(
                    required=False,
                    initial=es,
                    help_text=PrescreenListView.extra_scores_help,
                    )
            def clean_primary_score(self):
                val = self.cleaned_data['primary_score']
                # there are many things that can go wrong with an incorrect
                # 'val' that will throw all kinds of exceptions; just catch
                # them all outside the checking function and return the same
                # error to the user
                try:
                    if self._primary_score_ok(val):
                        return val
                except:
                    pass
                raise forms.ValidationError('Not a valid score')
            def clean_extra_scores(self):
                val = self.cleaned_data['extra_scores']
                try:
                    if val.strip():
                        for v in val.split('|'):
                            assert self._secondary_score_ok(v)
                    return val
                except:
                    pass
                raise forms.ValidationError('Not a valid score')
            def _primary_score_ok(self,val):
                import re
                m = re.match(r'(\d+)_(.+)',val)
                return self._gen_score_ok(val, m)
            def _gen_score_ok(self,val, m):
                from runner.process_info import JobInfo
                bji=JobInfo.get_bound(view.ws,int(m.group(1)))
                cat=bji.get_data_catalog()
                return cat.get_keyname(m.group(2))=='wsa'
            def _secondary_score_ok(self,val):
                import re
                m = re.match(r'(\d+):(.+)',val)
                return self._gen_score_ok(val, m)
        return MyForm(data)
    def add_post_valid(self):
        obj = self.add_form.save(commit=False)
        obj.ws = self.ws
        obj.user = self.request.user.username
        obj.save()
        return HttpResponseRedirect(self.here_url())
    def moaupdate_post_valid(self):
        from dtk.moa import update_moa_indications
        update_moa_indications(self.ws)

class PrescreenEditView(DumaView):
    template_name='nav/prescreen_edit.html'
    button_map={
            'save':['edit'],
            }
    def make_edit_form(self,data):
        from browse.models import Prescreen
        class MyForm(forms.ModelForm):
            class Meta:
                model = Prescreen
                fields = [
                        'name',
                        'extra_scores',
                        ]
                widgets = {
                        'name':forms.TextInput(attrs={'size':'100'}),
                        'extra_scores':forms.TextInput(attrs={'size':'100'}),
                        }
                help_texts = {
                        'extra_scores':PrescreenListView.extra_scores_help,
                        }
            # XXX This should be factored out from here and PrescreenListView
            # XXX to eliminate duplication and share validation code
        return MyForm(data,instance=self.prescreen)
    def save_post_valid(self):
        self.edit_form.save()
        return HttpResponseRedirect(self.ws.reverse('nav_prescreen_list'))


class WorkflowTimelineView(DumaView):
    template_name='nav/workflow_timeline.html'
    def custom_context(self):
        from runner.process_info import JobInfo
        from runner.models import Process
        from runner.plot import plot_timeline

        bji = JobInfo.get_bound(self.ws, self.job)
        ss = bji.get_scoreset()
        jids = ss.job_type_to_id_map().values()
        procs = Process.objects.filter(pk__in=jids)
        plot = plot_timeline(procs)
        self.context_alias(plotly_plots=[("timeline", plot)])

class ScoreSelectView(DumaView):
    template_name='nav/score_select.html'
    GET_parms={
            'done':(str,None),
            'page_size':(int,15),
            'page':(int,1),
            }
    button_map={
            'add':['jobs'],
            }
    def custom_setup(self):
        from dtk.scores import SourceList
        self.sources = SourceList(self.ws,jcc=self.jcc)
        self.sources.load_from_session(self.request.session)
        self.context_alias(source_table=JobTypeTable(
                    self.ws,
                    self.jobname,
                    self.here_url,
                    self.page_size,
                    self.page,
                    ))
    def custom_context(self):
        self.source_table.add_select_column(self.jobs_form)
        self.source_table.add_label_column(self.jobs_form)
        self.source_table.add_eval_columns()
        self.source_table.add_settings_column()
        self.source_table.add_jobid_column()
        self.source_table.add_date_column()
        self.source_table.add_status_column(self.context)
        from dtk.html import link
        from dtk.duma_view import qstr
        self.context_alias(run_new_html=link('Run a new job',
                    self.ws.reverse('nav_job_start',self.jobname)+qstr({
                            'done':self.here_url()
                            })
                    ))
    def make_jobs_form(self,data):
        FormClass=self.source_table.get_select_form_class(self.sources)
        return FormClass(data)
    def add_post_valid(self):
        p = self.jobs_form.cleaned_data
        self.source_table.update_from_post(p,self.sources)
        # return to calling page, if any
        next_url=self.done or self.here_url()
        return HttpResponseRedirect(next_url)

class JobStartView(DumaView):
    template_name='nav/job_start.html'
    GET_parms={
            'page_size':(int,15),
            'page':(int,1),
            'done':(str,None),
            }
    def custom_setup(self):
        if not hasattr(self,'copy_job'):
            self.copy_job = None
        from runner.process_info import JobInfo
        self.uji = JobInfo.get_unbound(self.jobname)
        if self.uji.needs_sources:
            # this is a meta-method, and so we offer the user a chance to
            # customize the list of sources to be used for scores, prior to
            # submitting the configuration form.
            from dtk.scores import SourceListWithEnables
            sources = SourceListWithEnables(self.ws)
            sources.load_from_session(self.request.session)
            import dtk.scores as st
            self.source_table = st.SourceTable(sources,[
                    st.LabelColType(),
                    st.JobTimeColType(),
                    st.JobStatusColType(self.context),
                    ])
            self.button_map={
                    'save':['source'],
                    'dflt':[],
                    }
            if self.copy_job:
                self.button_map['copy']=[]
        else:
            sources = None
            self.button_map={}
        self.button_map['run']=[]
        self._add_buttons(self.uji.get_buttons())

        self.context_alias(
                sources=sources,
                config_html = self.uji.get_config_html(
                                        self.ws,
                                        self.jobname,
                                        self.copy_job,
                                        sources=sources,
                                        ),
                )
    def _add_buttons(self, buttons):
        for button_def in buttons:
            self.button_map[button_def['name']] = []
            setattr(self, button_def['name'] + '_post_valid', button_def['action'])

    def custom_context(self):
        self.context_alias(
                past_runs=JobTypeTable(
                        self.ws,
                        self.jobname,
                        self.here_url,
                        self.page_size,
                        self.page,
                        ),
                # for 'get more sources' links
                here=self.here_url(done=self.done),
                )
        self.past_runs.add_past_settings_column(self.base_qparms)
        self.past_runs.add_date_elapsed_column()
        self.past_runs.add_status_column(self.context)
        self.past_runs.add_eval_columns()
        if self.sources:
            self.context_alias(
                    src_table=self.source_table.get_table(self.source_form),
                    )
            if self.copy_job:
                self.context['copy_from']=self.copy_job.id
    def make_source_form(self,data):
        FormClass = self.source_table.make_form_class()
        return FormClass(data)
    def save_post_valid(self):
        p = self.source_form.cleaned_data
        self.source_table.update_source_data(p)
        return HttpResponseRedirect(self.here_url())
    def dflt_post_valid(self):
        self.sources.load_defaults()
        return HttpResponseRedirect(self.here_url())
    def copy_post_valid(self):
        from dtk.job_prefix import SourceRoleMapper
        copy_sl = SourceRoleMapper.get_source_list_from_settings(
                                                self.ws,
                                                self.copy_job.settings(),
                                                )
        s = copy_sl.to_string()
        copy_id = self.copy_job.id
        if not s:
            self.message('No sources defined in job %d'%copy_id)
        elif s == self.sources.to_string():
            self.message('Job %d sources match existing sources'%copy_id)
        else:
            self.sources.load_from_string(s)
        return HttpResponseRedirect(self.here_url())
    def run_post_valid(self):
        config_html, next_url = self.uji.handle_config_post(
                                    self.jobname,
                                    self.jcc,
                                    self.request.user,
                                    self.request.POST,
                                    self.ws,
                                    sources=self.sources,
                                    )
        if next_url:
            from dtk.duma_view import qstr
            Process.drive_background()
            if self.done:
                next_url += qstr({'done':self.done})
            return HttpResponseRedirect(next_url)
        self.context_alias(
            config_html=config_html,
            )

class JobNoteForm(forms.Form):
    note = forms.CharField(
            widget=forms.Textarea(attrs={'rows':'2','cols':'50'}),
            required=False,
            )

@login_required
def progress(request,ws_id,job_id):
    ws = Workspace.objects.get(pk=ws_id)
    url_config=UrlConfig(request,
                defaults={
                })
    # render progress display for specified job
    from runner.process_info import JobInfo
    info = JobInfo.get_bound(ws,job_id)
    if info.job.status == info.job.status_vals.SUCCEEDED:
        # if job succeeded, and we're in jump-away-on-complete mode,
        # redirect to completion URL (otherwise stay here, even if
        # complete, so user can see error information)
        next_url = url_config.as_string('done')
        if next_url:
            return HttpResponseRedirect(next_url)
        info.fetch_lts_data()
    # set up job note editing
    nf = JobNoteForm()
    nf.initial['note'] = info.job.get_note_text()
    if request.method == 'POST' and post_ok(request):
        op = request.GET.get('op','no operation set')
        if op == 'abort':
            Process.abort(info.job.id)
            return HttpResponseRedirect(ws.reverse('nav_progress',job_id))
        elif op == 'force_stop':
            Process.stop(info.job.id,0,start_all=False)
            Process.drive_background() # assure pending jobs will run
            return HttpResponseRedirect(ws.reverse('nav_progress',job_id))
        elif op == 'note':
            nf = JobNoteForm(request.POST)
            if nf.is_valid():
                p = nf.cleaned_data
                from notes.models import Note
                Note.set(info.job
                        ,'note'
                        ,request.user.username
                        ,p['note']
                        )
                return HttpResponseRedirect(ws.reverse('nav_progress',job_id))
        else:
            logger.error("Unknown POST operation '%s'",op)
    warnings = info.get_warnings()
    if warnings:
        from dtk.warnings import output_warnings
        output_warnings(request,warnings)
    pause = url_config.as_bool('pause')
    refresh = 0
    show_stop = False
    if info.job.active():
        if request.user.groups.filter(name='button_pushers').count() \
                and info.job.log_closed():
            show_stop = True
        if not pause:
            refresh = 5
    from browse.models import ScoreSetJob
    scoreset_ids = sorted(set(ScoreSetJob.objects.filter(
            job_id=job_id,
            ).values_list('scoreset_id',flat=True)))
    return render(request
                ,'nav/progress.html'
                ,make_ctx(request,ws,'nav_scoreboard',{
                        'page_tab':'run(%d) %s' % (ws.id, info.job_type),
                        'job_info':info,
                        'refresh': refresh,
                        'pause':pause,
                        'show_stop':show_stop,
                        'note_form':nf,
                        'note_id':info.job.note_id,
                        'scoreset_ids':scoreset_ids,
                        }
                ))

class ClusterCyGraph:
    def __init__(self):
        self.nodes = set()
        self.links = []
        self.props = set()
        self.positions = []
    def add_link(self,start,end,name=None,**kwargs):
        self.nodes.add(start)
        self.nodes.add(end)
        if not name:
            name = 'link%d' % len(self.links)
        self.links.append( dict(kwargs,id=name,source=start,target=end) )
    def elements(self):
        from django.utils.safestring import mark_safe
        result = [
                { 'data': {
                        'id':node,
                        'prop':'y' if node in self.props else 'n',
                        } }
                for node in self.nodes
                ]
        result += [
                { 'data': d }
                for d in self.links
                ]
        import json
        return mark_safe(json.dumps(result))
    def style(self):
        from django.utils.safestring import mark_safe
        return mark_safe('''[
                { selector: 'node', style: {
                        'label': 'data(id)',
                        'shape':'roundrectangle',
                        } },
                { selector: "[prop='y']", style: {
                        'background-color': 'red',
                        } },
                { selector: "[prop='n']", style: {
                        'background-color': 'green',
                        } },
                { selector: 'edge', style: {
                        'width': 3,
                        'line-color': 'grey',
                        'line-style':'dotted',
                        'curve-style': 'bezier',
                        } },
                ]''')
    def layout(self):
        from django.utils.safestring import mark_safe
        # 'cose' or 'cose-bilkent' seem like they should be good,
        # but the first just places all nodes on top of each other,
        # and the second doesn't display anything (the same happens
        # on the test page).
        # 'breadthfirst' gave a somewhat reasonable layout, but in a
        # vertical direction that didn't work well with long node labels.
        # So, instead require the client to lay out the nodes.
        import json
        return mark_safe('''{
                name:'preset',
                positions:'''+json.dumps({
                        item:{'x':x,'y':y}
                        for item,x,y in self.positions
                        })+''',
                spacingFactor: 1.00,
                padding:10,
                }''')

@login_required
def matching(request,ws_id,wsa_id):
    ws = Workspace.objects.get(pk=ws_id)
    name_mode = request.GET.get('name_mode')
    from browse.models import WsAnnotation
    drug_ws = WsAnnotation.objects.get(pk = wsa_id)
    agent = drug_ws.agent
    root_key = agent.get_key(with_key_name=True)
    dpi_ver = ws.get_dpi_version()
    from dtk.drug_clusters import Clusterer,RebuiltCluster
    clr = Clusterer()
    if dpi_ver is None:
        clr.load_archive()
    else:
        rc = RebuiltCluster(base_key=root_key,version=dpi_ver)
        for drug,prop in rc.filtered_drug_prop_pairs:
            clr.add_drug_prop_pair(prop,drug)
    clr.build_links()
    from django.utils.safestring import mark_safe
    cluster_html = mark_safe(clr.get_cluster_html(root_key))
    cy = ClusterCyGraph()
    clust = clr.get_drug(root_key).get_cluster_as_set()
    clust = list(clust)
    if name_mode == 'collapse':
        # combine all name props with identical attachments
        from dtk.data import MultiMap
        dp_mm = MultiMap((d,p) for d in clust for p in d.prop_set)
        dsp_mm = MultiMap((frozenset(ds),p) for p,ds in dp_mm.rev_map().items())
        for ds,ps in dsp_mm.fwd_map().items():
            ps = set(p for p in ps if p[0] == 'name')
            vals = ', '.join(sorted(p[1] for p in ps))
            for d in ds:
                d.prop_set -= ps
                d.prop_set.add(('name',vals))
    # set up to provide aliases for long property substrings
    prop_aliases = {}
    def get_alias(orig):
        if orig not in prop_aliases:
            if orig == '':
                return ''
            if orig == 'UHFFFAOYSA':
                return '-'
            if orig == '/':
                return '/'
            label = ''
            seed = len(prop_aliases)
            while seed:
                label = chr(ord('a')+(seed%26)) + label
                seed //= 26
            if not label:
                label = 'a'
            prop_aliases[orig] = label
        return prop_aliases[orig]
    # add drug and prop nodes for each drug in cluster; columns are:
    # - root drug
    # - std_smiles properties
    # - drugs with std_smiles properties
    # - (most) other properties
    # - drugs without std_smiles properties
    # - properties seen for the first time on non-std_smiles drugs
    from dtk.grid_layout import VBox,HBox
    root_col = VBox()
    ssprop_col = VBox()
    ssdrug_col = VBox()
    prop_col = VBox()
    drug_col = VBox()
    oprop_col = VBox()
    # grid_layout left-justifies nodes in a VBox (and top-justifies them
    # in an HBox). So to make props look like they're centered in a wide
    # column, increase the width of both prop_col and the col to its left.
    root_col.node_width = 400
    ssprop_col.node_width = 400
    ssdrug_col.node_width = 400
    prop_col.node_width = 400
    drug_col.node_width = 400
    frame = HBox(mid=[
            root_col,
            ssprop_col,
            ssdrug_col,
            prop_col,
            drug_col,
            oprop_col,
            ])
    # order the clusters so that:
    # - drugs are grouped by their std_smiles property
    # - drugs w/o a std_smiles property appear last
    clust = [
            ([x for x in d.prop_set if x[0] == 'std_smiles'],d)
            for d in clust
            ]
    clust.sort(key=lambda x:x[0],reverse=True)
    for ss,d1 in clust:
        drug_label = d1.key[1]
        if drug_label == root_key[1]:
            root_col.mid.append(drug_label)
        elif ss:
            ssdrug_col.mid.append(drug_label)
        else:
            drug_col.mid.append(drug_label)
        for prop in d1.prop_set:
            if prop[0] == 'name':
                if name_mode == 'hide':
                    continue
                # no need for 'name:' prefix
                prop_label = prop[1]
            elif prop[0]  == 'inchi_key':
                # alias both parts separately for easier matching;
                # alias function handles 'not specified' key for 2nd part
                parts = prop[1].split('-')
                prop_label='-'.join([
                        get_alias(parts[0]),
                        get_alias(parts[1]),
                        parts[2],
                        ])
            elif prop[0] == 'inchi':
                # pass through chemical formula; alias basic and optional
                # structural sections separately
                parts = prop[1].split('/')
                if len(parts) >= 4:
                    prop_label='/'.join([
                            parts[1],
                            get_alias(parts[2]+'/'+parts[3]),
                            get_alias('/'.join(parts[4:])),
                            ])
            else:
                # include both key name and value
                prop_label=prop[0]+':'+prop[1]
            if prop_label not in cy.props:
                if prop[0] == 'std_smiles':
                    ssprop_col.mid.append(prop_label)
                elif not ss:
                    oprop_col.mid.append(prop_label)
                else:
                    prop_col.mid.append(prop_label)
            cy.props.add(prop_label)
            cy.add_link(drug_label,prop_label)
    frame.layout(cy.positions,0,0)
    return render(request
                ,'nav/matching.html'
                ,make_ctx(request,ws,'nav_scoreboard',{
                     'headline':agent.canonical,
                     'cluster_html':cluster_html,
                     'cy':cy,
                     'name_mode':name_mode,
                     })
                )

class ScoreCmpView(DumaView):
    template_name= 'nav/score_cmp.html'
    #index_dropdown_stem='nav_score_cmp'
    from dtk.duma_view import boolean
    GET_parms = {
            'x':(str,None),
            'y':(str,None),
            'ds':(str,None),
            'mode':(str,''),
            'rank_linear':(boolean,''),
            }
    demo_buttons = [
            'display',
            'defaults',
            ]
    pick_mode='pick'
    def all_parms_present(self): return self.x and self.y and self.ds
    def custom_setup(self):
        if self.dtc != 'wsa' and self.dtc != 'uniprot':
            raise ValueError('Scoreplots only display wsa or uniprot datasets')
        if not self.ds:
            self.ds = self.ws.eval_drugset
        from dtk.scores import SourceList
        self.sources = SourceList(self.ws,jcc=self.jcc)
        self.sources.load_from_session(self.request.session)
        if not self.all_parms_present():
            # force setup page
            self.mode=self.pick_mode
        if self.mode == self.pick_mode:
            self.button_map={
                    'display':['score','other'],
                    'defaults':[],
                    }
            import dtk.scores as st
            self.source_table = st.SourceTable(self.sources,[
                    st.SelectColType('y','Y Score',self.y,dct = self.dtc),
                    st.SelectColType('x','X Score',self.x,dct = self.dtc),
                    st.LabelColType(),
                    st.JobTimeColType(),
                    st.JobStatusColType(self.context),
                    ])
    class ScoreData:
        def __init__(self,global_code,sl):
            self.gcode = global_code
            self.src,self.cat,self.code = sl.parse_global_code(global_code)
            self.bji = self.src.bji()
            self.job = self.bji.job
            self.config = self.src.bji().job.settings()
            self.label = sl.get_label_from_code(global_code)
            self.ordering = self.cat.get_ordering(self.code,True)
            self.non_zeros = len([1 for x in self.ordering if x[1]])
            self.count = len(self.ordering)
            from dtk.scores import ZNorm
            self.norm = ZNorm(self.ordering)
    def custom_context(self):
        if self.mode == self.pick_mode:
            self.context['table'] = self.source_table.get_table(self.score_form)
        else:
            self.x_data=self.ScoreData(self.x,self.sources)
            self.y_data=self.ScoreData(self.y,self.sources)
            self.build_drug_info()
            from dtk.text import compare_dict
            test = compare_dict('',self.y_data.config,self.x_data.config)
            if test and len(test) < len(str(self.x_data.config)):
                self.context['settings_diff'] = 'Y Score'+test
            # make plots and thumbnails
            self.validate_plotdir()
            plotlist = (
                    'score_curves',
                    'z_scatter',
                    'z_box',
                    'rank_plot',
                    'metric_bar',
                    'metric_scatter',
                    'kt_cdf',
                    )
            plotly_plots=[]
            import os
            from dtk.plot import PlotlyPlot
            for plot in plotlist:
                path=os.path.join(self.plotdir,plot+'.plotly')
                try:
                    # use existing plot file if available
                    pp = PlotlyPlot.build_from_file(path,thumbnail=True)
                except IOError:
                    # else call method to get PlotlyPlot
                    func = getattr(self,'build_'+plot)
                    pp = func()
                    # and save as plot file
                    pp.save(path,thumbnail=True)
                plotly_plots.append(pp)
            self.context['plotly_plots']=[
                                    ('pldiv_%d'%i,x)
                                    for i,x in enumerate(plotly_plots)
                                    ]
            self.add_ranking_tables()
    def validate_plotdir(self):
        import os
        self.plotdir = os.path.join(
                PathHelper.tmp_publish(self.ws.id,'score_cmp'),
                '.'.join([self.x,self.y,self.dtc,self.ds]),
                )
        from dtk.plot import validate_plotdir
        validate_plotdir(self.plotdir,[
                ('ds:'+self.ds,' '.join(str(x) for x in sorted(self.kts))),
                ])
    def build_drug_info(self):
        if self.dtc == 'wsa':
            self.kts = self.ws.get_wsa_id_set(self.ds)
        elif self.dtc == 'uniprot':
            self.kts = self.ws.get_uniprot_set(self.ds)
        all_ids = (self.kts
            | set(self.y_data.norm.scores.keys())
            | set(self.x_data.norm.scores.keys())
            )
        if self.dtc == 'wsa':
            name_map = self.ws.get_wsa2name_map()
        elif self.dtc == 'uniprot':
            from browse.views import get_prot_2_gene
            name_map = get_prot_2_gene(all_ids)
        self.drug_info = [
                [
                    drug,
                    self.y_data.norm.scores.get(drug),
                    self.x_data.norm.scores.get(drug),
                    drug in self.kts,
                    self.y_data.norm.get(drug),
                    self.x_data.norm.get(drug),
                    name_map[drug],
                ]
                for drug in all_ids
                ]
        self.pair_count = len(self.drug_info)
        self.dual_zeros = len([
                1
                for x in self.drug_info
                if not x[1] and not x[2]
                ])
        if self.is_demo() and self.dtc == 'wsa':
            # if demo mode, obfuscate names; don't rely on the drugset
            # selected in the UI -- always use the 'real' kts as what
            # not to obfuscate
            from tools import obfuscate
            actual_kts = self.ws.get_wsa_id_set('kts')
            for x in self.drug_info:
                if x[0] not in actual_kts:
                    x[6] = obfuscate(x[6])
    def build_score_curves(self):
        from dtk.plot import PlotlyPlot
        return PlotlyPlot([
                    dict(
                         y=[v[1] for v in self.y_data.ordering],
                         name=self.y_data.gcode,
                        ),
                    dict(
                         y=[v[1] for v in self.x_data.ordering],
                         name=self.x_data.gcode,
                        ),
                    ],
                    {'title':'Score curves'},
                )
    def build_z_scatter(self):
        from dtk.plot import scatter2d,annotations,Color
        xy_vals = []
        isKt = []
        drug_ids = []
        text = []
        for row in self.drug_info:
            if not row[1] and not row[2]:
                # don't plot points corresponding to (0,0) scores
                continue
            xy_vals.append((row[4], row[5]))
            drug_ids.append(row[0])
            text.append(row[6])
            if row[3]:
                isKt.append(1)
            else:
                isKt.append(0)
        from dtk.num import corr
        from tools import sci_fmt
        return scatter2d(
                self.y_data.gcode,
                self.x_data.gcode,
                xy_vals,
                text = text,
                ids = ('drugpage', drug_ids),
                title = "Consistency of Z scores",
                class_idx = isKt,
                classes = [
                        ('Unknown',{'color':Color.default, 'opacity' : 0.2}),
                        ('KT',{'color':Color.highlight, 'opacity' : 0.7})
                        ],
                annotations=annotations(
                        'pearson: '+sci_fmt(corr(xy_vals,method='pearson')),
                        'spearman: '+sci_fmt(corr(xy_vals,method='spearman')),
                        ),
                bins=True
                )
    def build_z_box(self):
        from dtk.plot import PlotlyPlot
        y_kt_vals = []
        x_kt_vals = []
        for row in self.drug_info:
            if row[3]:
                y_kt_vals.append(row[4])
                x_kt_vals.append(row[5])
        return PlotlyPlot([
            dict(y = y_kt_vals
                   , type = 'box'
                   , name = self.y_data.gcode
                   , boxpoints = 'all'
                   , jitter = 0.5
                   , boxmean = 'sd'
                   , marker = dict(size = 3, opacity = 0.5)
                  ),
            dict(y = x_kt_vals
                   , type = 'box'
                   , name = self.x_data.gcode
                   , boxpoints = 'all'
                   , jitter = 0.5
                   , boxmean = 'sd'
                   , marker = dict(size = 3, opacity = 0.5)
                  ),
            ],
            {'title':'KT Z-scores',
             'yaxis':{'title':'Z-score'}
            },
            )
    def build_rank_plot(self):
        from dtk.plot import scatter2d,Color
        from dtk.scores import Ranker
        x_rank_lookup = Ranker(self.x_data.ordering)
        y_rank_lookup = Ranker(self.y_data.ordering)
        xy = []
        class_idx = []
        text = []
        ids = []
        for row in self.drug_info:
            if not row[1] and not row[2]:
                # don't plot points corresponding to (0,0) scores
                continue
            key=row[0]
            xy.append( (x_rank_lookup.get(key),y_rank_lookup.get(key)) )
            class_idx.append( row[3] )
            text.append( row[6] )
            ids.append( key )
        return scatter2d(
                self.x_data.gcode,
                self.y_data.gcode,
                xy,
                classes=[
                        ('Unknown',{'color':Color.default, 'opacity' : 0.2}),
                        ('KT',{'color':Color.highlight, 'opacity' : 0.7})
                        ],
                class_idx=class_idx,
                text=text,
                ids=('drugpage',ids),
                width=800,
                height=700,
                title="Rank scatterplot (lower left is best)",
                logscale=not self.rank_linear,
                )
    def load_metric_info(self):
        if hasattr(self,'minfo'):
            return
        class Dummy: pass
        self.minfo = Dummy()
        from dtk.enrichment import EMInput,EnrichmentMetric
        self.minfo.y_emi = EMInput(self.y_data.ordering,self.kts)
        self.minfo.x_emi = EMInput(self.x_data.ordering,self.kts)
        self.minfo.metric_names = [
                "DEA_AREA",
                "SigmaOfRank",
                "SigmaOfRank1000",
                "FEBE",
                "wFEBE",
                #"DEA_ES",
                "AUR",
                "APS",
                #"DPI_bg_corr",
                #"EcdfPoly",
                #"OLD_DEA_ES",
                ]
        self.minfo.xy = []
        floor = 1E-6 # keep logplot looking reasonable
        self.minfo.dif_por = []
        for name in self.minfo.metric_names:
            Type = EnrichmentMetric.lookup(name)
            x_em = Type()
            x_em.evaluate(self.minfo.x_emi)
            y_em = Type()
            y_em.evaluate(self.minfo.y_emi)
            self.minfo.xy.append( (
                    max(x_em.rating,floor),
                    max(y_em.rating,floor),
                    ) )
            if x_em.rating:
                self.minfo.dif_por.append(
                        (y_em.rating - x_em.rating) / x_em.rating
                        )
            else:
                self.minfo.dif_por.append(0)
    def build_metric_bar(self):
        self.load_metric_info()
        from dtk.plot import PlotlyPlot
        return PlotlyPlot([dict(
                    type='bar',
                    x=self.minfo.metric_names,
                    y=self.minfo.dif_por,
                    )],
                    {'title':'Evalulation metrics - portion change',
                     'yaxis':{'title':'(Score2 - Score1)/ Score1'}
                    }
                    )
    def build_metric_scatter(self):
        self.load_metric_info()
        from dtk.plot import scatter2d
        return scatter2d(
                    self.x_data.gcode,
                    self.y_data.gcode,
                    self.minfo.xy,
                    text=self.minfo.metric_names,
                    title="Evaluation metrics - scatter",
                    logscale=True,
                    )
    def build_kt_cdf(self):
        self.load_metric_info()
        from dtk.plot import PlotlyPlot
        x_cdf = self.minfo.x_emi.get_hit_cdf()
        y_cdf = self.minfo.y_emi.get_hit_cdf()
        return PlotlyPlot([
                    dict(y=y_cdf,name=self.y_data.gcode),
                    dict(y=x_cdf,name=self.x_data.gcode),
                    ],
                    {'title':'KT cumulative distribution'},
                    )
    def add_ranking_tables(self):
        # append rank info
        # These two functions assign the same rank number to all drugs in
        # a block of identical scores.  The second uses the offset of the
        # last drug, which may be a more accurate way to represent the
        # potentially large block of unscored drugs at the bottom.
        def append_rank(src,col):
            src.sort(key=lambda x:x[col],reverse=True)
            last_score = None
            for i,row in enumerate(src,start=1):
                if row[col] != last_score:
                    last_rank = i
                    last_score = row[col]
                row.append(last_rank)
        def append_rank_from_bottom(src,col):
            src.sort(key=lambda x:x[col])
            last_score = src[-1][col]
            last_rank = len(src)
            for i,row in enumerate(src,start=0):
                if row[col] != last_score:
                    last_rank = len(src)-i
                    last_score = row[col]
                row.append(last_rank)
        drug_info = self.drug_info
        append_rank_from_bottom(drug_info,4)
        append_rank_from_bottom(drug_info,5)
        # append score and rank deltas
        def append_delta(src,col1,col2):
            for row in src:
                row.append(row[col2]-row[col1])
        append_delta(drug_info,4,5)
        append_delta(drug_info,8,7) # flip direction, because low rank == good
        # build up and down changes lists
        num = 20
        data=[]
        drug_info.sort(key=lambda x:(x[9] is None, x[9]))
        data.append( ('%d most increased scores'%num, drug_info[:-num:-1]) )
        data.append( ('%d most decreased scores'%num, drug_info[:num]) )
        # rank is opposite -- low == good
        drug_info.sort(key=lambda x:(x[10] is None, x[10]))
        data.append( ('%d most increased rank'%num, drug_info[:-num:-1]) )
        data.append( ('%d most decreased rank'%num, drug_info[:num]) )
        # sample top and bottom of list by origininal score
        drug_info.sort(key=lambda x:(x[1] is None, x[1]))
        data.append( ('%d highest Score 1 (Y)'%num, drug_info[:-num:-1]) )
        drug_info.sort(key=lambda x:(x[2] is None, x[2]))
        data.append( ('%d highest Score 2 (X)'%num, drug_info[:-num:-1]) )
        self.context['data'] = data
    def make_score_form(self,data):
        FormClass = self.source_table.make_form_class()
        return FormClass(data)
    def make_other_form(self,data):
        if self.dtc == 'wsa':
            ds_choices = self.ws.get_wsa_id_set_choices(
                    train_split=True, test_split=True)
            import dea
            class MyForm(forms.Form):
                ds = forms.ChoiceField(
                    label = 'Drugset',
                    choices = ds_choices,
                    initial = self.ds,
                    )
            return MyForm(data)
        elif self.dtc == 'uniprot':
            ds_choices = self.ws.get_uniprot_set_choices()
            import dea
            class MyForm(forms.Form):
                ds = forms.ChoiceField(
                    label = 'Protset',
                    choices = ds_choices,
                    initial = self.ds,
                    )
            return MyForm(data)
    def display_post_valid(self):
        p1 = self.score_form.cleaned_data
        self.source_table.update_source_data(p1)
        p2 = self.other_form.cleaned_data
        return HttpResponseRedirect(self.here_url(
                    mode=None,
                    x=p1['x'],
                    y=p1['y'],
                    ds=p2['ds'],
                    ))
    def defaults_post_valid(self):
        self.sources.load_defaults()
        return HttpResponseRedirect(self.here_url())

class XwsCmpView(DumaView):
    template_name= 'nav/xws_cmp.html'
    index_dropdown_stem='nav_xws_cmp'
    GET_parms = {
            'metric':(str,None),
            'score':(str,None),
            'ds':(str,None),
            'mode':(str,''),
            }
    pick_mode='pick'
    def all_parms_present(self): return self.metric and self.score and self.ds
    def custom_setup(self):
        from dtk.scores import SourceList
        self.sources = SourceList(self.ws,jcc=self.jcc)
        self.sources.load_from_session(self.request.session)
        if not self.all_parms_present():
            # force setup page
            self.mode=self.pick_mode
        if self.mode == self.pick_mode:
            self.button_map={
                    'display':['score','other'],
                    'defaults':[],
                    }
            import dtk.scores as st
            self.source_table = st.SourceTable(self.sources,[
                    st.SelectColType('score','',self.score),
                    st.LabelColType(),
                    st.JobTimeColType(),
                    st.JobStatusColType(self.context),
                    ])
    def custom_context(self):
        self.context['headline']='Cross-workspace Compare'
        if self.mode == self.pick_mode:
            self.context['table'] = self.source_table.get_table(self.score_form)
        else:
            self.generate_metrics()
            # load some stuff needed to generate labels at top
            label = self.sources.get_label_from_code(self.score)
            self.context['score_label'] = label
            src,cat,code = self.sources.parse_global_code(self.score)
            self.context['bji'] = src.bji()
            self.context['ds_name']=[
                    x[1]
                    for x in self.ws.get_wsa_id_set_choices(fixed_only=True)
                    if x[0] == self.ds
                    ][0]
    def generate_metrics(self):
        ws=self.ws
        # get score values
        ordering = self.sources.get_ordering_from_code(self.score,True)
        # get selected enrichment metric
        from dtk.enrichment import EnrichmentMetric,EMInput
        Metric = EnrichmentMetric.lookup(self.metric)
        # build a map from agent_ids to local wsa_ids
        agent2wsa=ws.get_agent2wsa_map()
        # get drugset from each ws and evaluate
        # - if the cross_compare field for a workspace is Null (the default),
        #   use a heuristic to decide whether to compare to the workspace
        # - if cross_compare is False, never cross compare
        # - if cross_compare is True, always compare (unless there's no overlap
        #   between the remote drugset and the local collection)
        all_ws = set(Workspace.objects.exclude(cross_compare=False))
        all_ws.add(ws) # make sure current ws is in list, even if not active
        ds = self.ds
        metric_scores = []
        from browse.models import WsAnnotation
        for other_ws in all_ws:
            remote_ids = other_ws.get_wsa_id_set(ds)
            if not remote_ids:
                continue
            qs = WsAnnotation.objects.filter(pk__in=remote_ids)
            agent_ids = qs.values_list('agent_id',flat=True)
            local_ids = set([
                    agent2wsa[agent]
                    for agent in agent_ids
                    if agent in agent2wsa
                    ])
            if not local_ids:
                continue
            if other_ws == ws or other_ws.cross_compare:
                pass # always include
            else: # apply heuristic
                if len(local_ids) < 2:
                    continue
            em = Metric()
            em.evaluate(EMInput(ordering,local_ids))
            metric_scores.append( (other_ws,em.rating,local_ids) )
        metric_scores.sort(key=lambda x:x[1],reverse=True)
        # produce bar graph w/ metric value and ws name (color current ws)
        from dtk.plot import PlotlyPlot
        colors = [
                'red' if x[0] == ws else 'blue'
                for x in metric_scores
                ]
        bars = PlotlyPlot(
                data=[dict(
                        type='bar',
                        y=[x[1] for x in metric_scores],
                        text=[
                                '%s<br>%d KTs' % (x[0].name,len(x[2]))
                                for x in metric_scores
                                ],
                        marker={'color':colors},
                        )],
                layout={'title':'Enrichment by workspace'},
                )
        self.context['bars'] = bars
    def make_score_form(self,data):
        FormClass = self.source_table.make_form_class()
        return FormClass(data)
    def make_other_form(self,data):
        ds_choices = self.ws.get_wsa_id_set_choices()
        from dtk.enrichment import EnrichmentMetric
        metric_choices=[
                (name,name)
                for name,cls in EnrichmentMetric.get_subclasses()
                ]
        class MyForm(forms.Form):
            ds = forms.ChoiceField(
                label = 'Drugset',
                choices = ds_choices,
                initial = self.ds,
                )
            metric = forms.ChoiceField(
                label = 'Metric',
                choices = metric_choices,
                initial = self.metric,
                )
        return MyForm(data)
    def display_post_valid(self):
        p1 = self.score_form.cleaned_data
        self.source_table.update_source_data(p1)
        p2 = self.other_form.cleaned_data
        return HttpResponseRedirect(self.here_url(
                    mode=None,
                    score=p1['score'],
                    ds=p2['ds'],
                    metric=p2['metric'],
                    ))
    def defaults_post_valid(self):
        self.sources.load_defaults()
        return HttpResponseRedirect(self.here_url())

class ScoreGrid(DumaView):
    template_name= 'nav/score_grid.html'
    GET_parms = {
            'cutoff':(int,1000),
            'wsas':(list_of(int),[]),
            'scores':(str,''),
            'norm':(int, 0),
            'ds':(str,''),
            'cluster_cols':(boolean,True),
            'cluster_rows':(boolean,True),
            'from_wzs':(int,None),
            'zmax':(float,None),
            'zmin':(float,None),
            }
    button_map={
            'config':['config'],
            }
    def custom_setup(self):
        if self.from_wzs:
            from runner.process_info import JobInfo
            bji = JobInfo.get_bound(self.ws, self.from_wzs)
            fm_code = bji.parms['fm_code']
            fm = self.ws.get_feature_matrix(fm_code)
            score_codes = fm.spec.get_codes()

            # Sort by name, so that we could compare.
            scores = [[y.strip() for y in x.split('_')] for x in score_codes]
            bjis = JobInfo.get_all_bound(self.ws, [jid for jid,code in scores])
            named_scores = list(zip([x.job.role for x in bjis], score_codes))
            named_scores.sort()
            score_codes = [x[1] for x in named_scores]

            # Tack on wzs at the end.
            score_codes.append(f'{bji.job.id}_wzs')
            score_codes = '\n'.join(score_codes)

            return HttpResponseRedirect(self.here_url(scores=score_codes, from_wzs=None))
    def norm_choices(self, as_names):
        import dtk.scores as ds
        types = [
                ds.RankNorm,
                ds.MMNorm,
                ds.PctNorm,
                ds.NoNorm,
                ds.TargetNorm,
                ds.WzsNorm,
        ]
        if as_names:
            return [(i, x.__name__) for i, x in enumerate(types)]
        else:
            return [(i, x) for i, x in enumerate(types)]

    def make_config_form(self,data):
        from dtk.html import MultiWsaField
        class MyForm(forms.Form):
            cutoff = forms.IntegerField(
                    label='Support threshold',
                    initial=self.cutoff,
                    required=False,
                    )
            ds = forms.ChoiceField(
                    choices=[(None, "None")] +self.ws.get_wsa_id_set_choices(),
                    initial=self.ds,
                    label='Molecule Set',
                    required=False,
                    )
            wsas = MultiWsaField(
                         ws_id=self.ws.id,
                         initial=self.wsas,
                         label='Extra Molecules',
                         required=False,
                         )
            scores = forms.CharField(
                    widget=forms.Textarea(attrs={'rows':'4','cols':'30'}),
                    required=False,
                    initial = self.scores,
                    )
            norm = forms.ChoiceField(
                    choices=self.norm_choices(as_names=True),
                    initial=self.norm,
                    )
            cluster_cols = forms.BooleanField(
                    label='Cluster Columns',
                    initial=self.cluster_cols,
                    required=False,
                    )
            cluster_rows = forms.BooleanField(
                    label='Cluster Rows',
                    initial=self.cluster_rows,
                    required=False,
                    )
            zmax = forms.FloatField(
                    label='Override zmax',
                    initial=self.zmax,
                    required=False,
                    )
            zmin = forms.FloatField(
                    label='Override zmin',
                    initial=self.zmin,
                    required=False,
                    )
        return MyForm(data)
    def config_post_valid(self):
        p = self.config_form.cleaned_data
        return HttpResponseRedirect(self.here_url(**p))
    def custom_context(self):
        wsas = list(self.wsas)
        if self.ds:
            wsas += self.ws.get_wsa_id_set(self.ds)
        if not wsas or not self.scores:
            return

        from runner.process_info import JobInfo
        scores = [[y.strip() for y in x.split('_')] for x in self.scores.split('\n')]

        bjis = JobInfo.get_all_bound(self.ws, [jid for jid,code in scores])
        codes = [code for jid,code in scores]

        def score_name(bji, code):
            name = f'{bji.job.role}_{code} ({bji.job.id})'
            if bji.job.role == 'wzs':
                from urllib.parse import unquote
                url = unquote(self.here_url(from_wzs=bji.job.id, scores=''))
                name = f'<a href="{url}">{name}</a>'
            return name
        score_names = [score_name(bji, code) for bji, code in zip(bjis, codes)]

        NormCls = self.norm_choices(as_names=False)[self.norm][1]

        import dtk.scores as ds
        if NormCls in [ds.RankNorm, ds.TargetNorm]:
            reversescale=True
        else:
            reversescale=False

        if NormCls == ds.WzsNorm:
            wzs_bjis = [bji for bji, code in zip(bjis, codes) if code == 'wzs']
            assert len(wzs_bjis) == 1, "Can't wzs norm without exactly 1 wzs job"
            wzs_bji = wzs_bjis[0]
            weights, sources = wzs_bji.get_score_weights_and_sources()
            norm = ds.WzsNorm(weights, sources)

        zmin, zmax = 1e99, -1e99
        grid = []
        hover_grid = []


        for jid, code in scores:
            bji = JobInfo.get_bound(self.ws, jid)
            cat = bji.get_data_catalog()
            if NormCls != ds.WzsNorm:
                # Don't need this for wzs norm.
                ordering = cat.get_ordering(code,True)
            if NormCls in [ds.RankNorm, ds.TargetNorm]:
                norm = NormCls(ordering, none_if_missing=True)
            elif NormCls == ds.WzsNorm:
                norm.load(bji, code)
            else:
                norm = NormCls(ordering)

            row = [norm.get(wsa) for wsa in wsas]
            hover_row = [norm.fmt(x) for x in row]
            grid.append(row)
            hover_grid.append(hover_row)
            cur_min, cur_max = norm.range()
            zmin = min(zmin, cur_min)
            zmax = max(zmax, cur_max)

        if self.zmin is not None:
            zmin = self.zmin
        if self.zmax is not None:
            zmax = self.zmax

        from dtk.plot import plotly_heatmap
        import numpy as np
        grid = np.array(grid).transpose()
        grid[grid == None] = zmax
        # as of numpy 1.7.1, an object dtype isn't handled; grid will be
        # object type if it contained any None values prior to the assignment
        # above; force to float now that Nones are gone
        grid = grid.astype('float64')
        hover_grid = np.array(hover_grid).transpose()


        from browse.models import WsAnnotation
        wsa_objs = WsAnnotation.objects.filter(pk__in=wsas)
        id2wsa = {wsa.id:wsa for wsa in wsa_objs}

        wsa_labels = [id2wsa[x].html_url() for x in wsas]

        width = max(800, 30*len(score_names))
        height = max(800, 30*len(wsa_labels))
        heatmap = plotly_heatmap(
                grid,
                col_labels=score_names,
                row_labels=wsa_labels,
                reorder_cols=self.cluster_cols,
                reorder_rows=self.cluster_rows,
                zmin=zmin,
                zmax=zmax,
                Title='ScoreGrid',
                width=width,
                height=height,
                hover_text=hover_grid,
                colorscale='Viridis',
                reversescale=reversescale,
                max_len_r=10,
                )
        plots=[
                ('Heatmap', heatmap),
                ]

        has_support = grid<self.cutoff
       # take advantage of the special coding we put in place for WZS up above as a hacky way to index the WZS job(s)
        wzs_starts = [i for i, sn in enumerate(score_names) if sn.startswith('<a href=')]
        wzs_support = np.zeros(len(wsa_labels))
        for i in wzs_starts:
            sc = has_support[:,i]
            for j, w in enumerate(wsa_labels):
                wzs_support[j] += sc[j]

        self.context_alias(
                plotly_plots=plots,
                alg_cols=[{'title':x} for x in ['Score', f'# supported WSAs (rank < {self.cutoff})']],
                alg_data=list(zip(score_names,np.sum(has_support, axis=0).tolist())),
                wsa_cols=[{'title':x} for x in ['WSA', f'# supporting CMs (rank < {self.cutoff})', 'Has WZS support']],
                wsa_data=list(zip(wsa_labels,np.sum(has_support, axis=1).tolist(), wzs_support)),
                )


class ScorePIOverlap(DumaView):
    template_name= 'nav/score_pi_overlap.html'
    #index_dropdown_stem='nav_scoreplot'
    GET_parms = {
            'score':(str,None),
            'ds':(str,None),
            }

    def custom_setup(self):
        from dtk.scores import SourceList
        self.sources = SourceList(self.ws,jcc=self.jcc)
        self.sources.load_from_session(self.request.session)

    def custom_context(self):
        self.get_score_data()
        dpi = self.check_dpi_overlap()
        ppi = self.check_ppi_overlap()
        zippedDict = [('DPI','Percentage of scores','Overlap / Total in DPI', True)] + dpi + [('PPI', 'Percentage of scores', 'Overlap / Total in PPI', True)] + ppi
        self.context['sample_sorter'] = zippedDict

    def get_score_data(self):
        sl = self.sources
        self.src,self.cat,self.code=sl.parse_global_code(self.score)
        self.bji=self.src.bji()
        self.job=self.bji.job
        self.label=sl.get_label_from_code(self.score)
        self.ordering=self.cat.get_ordering(self.code,True)
        if self.dtc == 'wsa':
            self.kts = self.ws.get_wsa_id_set(self.ds)
        elif self.dtc == 'uniprot':
            self.kts = self.ws.get_uniprot_set(self.ds)
        self.ids = [x[0] for x in self.ordering]
        self.vals = [x[1] for x in self.ordering]
        if self.dtc == 'uniprot':
            from browse.views import get_prot_2_gene
            name_map = get_prot_2_gene(self.ids)
        elif self.dtc == 'wsa':
            name_map = self.ws.get_wsa2name_map()
        self.names = [
                name_map[x]
                for x in self.ids
                ]

    def check_ppi_overlap(self):
        from dtk.prot_map import DpiMapping,PpiMapping
        choices = PpiMapping.choices()
        result = []
        for choice in choices:
           mapp = PpiMapping(choice[0])
           result_len = len(set(mapp.get_uniq_target()).intersection(self.ids))
           result_len_str = str(len(set(mapp.get_uniq_target()).intersection(self.ids))) + ' / ' + str(len(mapp.get_uniq_target()))
           result_per = result_len/float(len(self.ids))
           #res_string = str(result_len)+" ("+"{:.2%}".format(result_per) + ")"
           result.append((choice[0],"{:.2%}".format(result_per),result_len_str , False,result_per))
        result.sort(key=lambda tup: tup[-1])
        return result[::-1]

    def check_dpi_overlap(self):
        from dtk.prot_map import DpiMapping,PpiMapping
        choices = DpiMapping.choices(ws=self.ws)
        result = []
        for choice in choices:
           mapp = DpiMapping(choice[0])
           result_len = len(set(mapp.get_uniq_target()).intersection(self.ids))
           result_len_str = str(len(set(mapp.get_uniq_target()).intersection(self.ids))) + ' / ' + str(len(mapp.get_uniq_target()))
           result_per = result_len/float(len(self.ids))
           #res_string = str(result_len)+" ("+"{:.2%}".format(result_per) + ")"
           result.append((choice[0],"{:.2%}".format(result_per),result_len_str , False, result_per))
        result.sort(key=lambda tup: tup[-1])
        return result[::-1]

class ScorePlotSimpleView(DumaView):
    template_name= 'nav/scoreplot_simple.html'
    GET_parms = {
            'score':(str,None),
            'basecolor':(str,None),
            'ds':(list_of(str),[])
            }
    def custom_context(self):
        from dtk.scores import SourceList
        self.sources = SourceList(self.ws,jcc=self.jcc)
        self.sources.load_from_session(self.request.session)
        sl = self.sources
        self.src,self.cat,self.code=sl.parse_global_code(self.score)
        self.ordering=self.cat.get_ordering(self.code,True)

        if self.basecolor:
            self.basecolor = '#' + self.basecolor

        kt_sets = []
        kt_colors = []
        for ds_id in self.ds:
            name, color = ds_id.split('|')
            color = '#' + color
            kt_sets.append(self.ws.get_wsa_id_set(name))
            kt_colors.append(color)


        from dtk.plot import score_plot
        plot = score_plot(self.ordering,
                          kt_sets,
                          dtc=self.dtc,
                          base_color=self.basecolor,
                          marked_colors=kt_colors,
                          )
        self.context_alias(plotly_plots=[
            ("Score Plot", plot),
            ])

class ScorePlotView(DumaView):
    template_name= 'nav/scoreplot.html'
    #index_dropdown_stem='nav_scoreplot'
    GET_parms = {
            'score':(str,None),
            'ds':(str,None),
            'mode':(str,''),
            'raise_exc':(boolean,False),
            'cutoff':(int,1000),
            }
    pick_mode='pick'
    def all_parms_present(self): return self.score and self.ds
    def custom_setup(self):
        if not self.ds and not self.dtc == 'uniprot':
            self.ds = self.ws.eval_drugset
        from dtk.scores import SourceList
        self.sources = SourceList(self.ws,jcc=self.jcc)
        self.sources.load_from_session(self.request.session)
        if not self.all_parms_present():
            # force setup page
            self.mode=self.pick_mode
        if self.mode == self.pick_mode:
            self.button_map={
                    'display':['score','other'],
                    'defaults':[],
                    }
            import dtk.scores as st
            self.source_table = st.SourceTable(self.sources,[
                    st.SelectColType('score','',self.score, dct = self.dtc),
                    st.LabelColType(),
                    st.JobTimeColType(),
                    st.JobStatusColType(self.context),
                    ])
        else:
            self.button_map={
                    'redisplay':['cutoff'],
                    }
    def custom_context(self):
        if self.dtc != 'wsa' and self.dtc != 'uniprot':
            raise ValueError('Scoreplots only display wsa or uniprot datasets')
        if self.mode == self.pick_mode:
            self.context['table'] = self.source_table.get_table(self.score_form)
        else:
            self.context_alias(plotly_plots=[],other_metrics=[])
            self.get_score_data()
            self.calc_cutoff_stats()
            if self.dtc == 'wsa':
                self.calc_kts_table()
            self.validate_plotdir()
            # Most plots on this page also generate a score. The plot may be
            # cached, but the score is always recalculated. A name starting
            # with a '*' means no score is generated, and the call will not
            # be made if a cached plot is found. Other names are always called,
            # and passed an optional plot=False parameter if the plot isn't
            # needed.
            # A name starting with '!' is always called, and any cached plot
            # replaced. This is useful during development.
            plotlist = [
                    '*rank_plot',
                    '*condensed_rank_plot',
                    'cal_plot',
                    'prob_plot',
                    # This one relies on dpi bg files we no longer generate.
                    #'dbc_plot',
                    'dea_plot',
                    'febe_plot',
                    'wfebe_plot',
                    'wfebe_condensed_plot',
                    '*entropy_plot',
                    'aur_plot',
                    'aur_condensed_plot',
                    'aps_plot',
                    '*kde_plot',
                    'sor_plot',
                    'sor1000_plot',
                    'sor1000_condensed_plot',
                    ]
            if self.dtc != 'wsa':
                plotlist = [x for x in plotlist if 'condensed' not in x]
            import os
            from dtk.plot import PlotlyPlot
            for plot in plotlist:
                call_optional = plot.startswith('*')
                call_forced = plot.startswith('!')
                if call_optional or call_forced:
                    plot = plot[1:]
                path=os.path.join(self.plotdir,plot+'.plotly')
                try:
                    # use existing plot file if available
                    pp = PlotlyPlot.build_from_file(path,thumbnail=True)
                except IOError:
                    pp = None
                    pass
                func = getattr(self,'calc_'+plot)
                try:
                    if call_optional and pp:
                        pass
                    elif pp and not call_forced:
                        func(plot=False)
                    else:
                        pp = func()
                        if not pp:
                            continue
                        pp.save(path,thumbnail=True)
                    self.plotly_plots.append((plot,pp))
                except Exception as ex:
                    self.message(func.__name__+' got exception '+str(ex))
                    # Set raise_exc=1 in the url params for debugging.
                    if self.raise_exc:
                        raise
    def get_score_data(self):
        sl = self.sources
        self.src,self.cat,self.code=sl.parse_global_code(self.score)
        self.bji=self.src.bji()
        self.job=self.bji.job
        self.label=sl.get_label_from_code(self.score)
        self.ordering=self.cat.get_ordering(self.code,True)
        from dtk.enrichment import EMInput, fill_wsa_ordering
        if self.dtc == 'wsa':
            self.kts = self.ws.get_wsa_id_set(self.ds)
            self.ordering = fill_wsa_ordering(self.ordering, self.ws, dpi=self.bji.get_dpi_choice())
        elif self.dtc == 'uniprot':
            self.kts = self.ws.get_uniprot_set(self.ds)
        self.emi = EMInput(self.ordering,self.kts)
        self.ids = [x[0] for x in self.ordering]
        self.vals = [x[1] for x in self.ordering]
        if self.dtc == 'uniprot':
            from browse.views import get_prot_2_gene
            name_map = get_prot_2_gene(self.ids)
        elif self.dtc == 'wsa':
            name_map = self.ws.get_wsa2name_map()
        self.names = [
                name_map[x]
                for x in self.ids
                ]
        self.calibrated = None
        if self.dtc == 'wsa':
            from dtk.score_calibration import ScoreCalibrator
            from dtk.moa import is_moa_score
            is_moa = is_moa_score(self.ids)
            try:
                sc = ScoreCalibrator()
                self.calibrated = sc.calibrate(
                        self.job.role,
                        self.code,
                        is_moa,
                        [
                                x[1]
                                for x in self.bji.remove_workspace_scaling(
                                        self.code,
                                        self.ordering,
                                        )
                        ],
                        )
            except IOError:
                self.message('No score calibration file')
            except KeyError:
                self.message('Score not present in calibration file')
    def calc_cutoff_stats(self):
        if self.cutoff >= self.emi.n_scores():
            self.message('Cutoff exceeds number of scores')
            return
        kts_above = len([
            i for i in self.emi.get_raw_ranks()
            if i < self.cutoff
            ])
        other_kts = self.emi.n_kts()-kts_above
        non_kts_above = self.cutoff - 1 - kts_above
        other_non_kts = self.emi.n_scores() - non_kts_above - self.emi.n_kts()
        cells = [
                kts_above,
                other_kts,
                non_kts_above,
                other_non_kts,
                ]
        if not all(cells):
            self.message(f'Some cutoff categories are empty: KT above/below {kts_above}/{other_kts}, non-KTs above/below {non_kts_above}/{other_non_kts}')
            return
        confusion_matrix = [ cells[:2],cells[2:]]
        odds = [a/b for a,b in confusion_matrix]
        odds_ratio = odds[0]/odds[1]
        import math
        ci_exp = math.exp(1.96*math.sqrt(sum((1/x for x in cells))))
        import scipy.stats
        chi2,p,dof,expected = scipy.stats.chi2_contingency(confusion_matrix)
        from tools import sci_fmt
        self.cutoff_stats = dict(
                odds_ratio=sci_fmt(odds_ratio),
                p_value=sci_fmt(p),
                ci=f'[{sci_fmt(odds_ratio/ci_exp)}, {sci_fmt(odds_ratio*ci_exp)}]'
                )
    def calc_kts_table(self):
        kt_keys = self.emi.get_kt_set()
        rows = []
        for i,(k,v) in enumerate(self.emi.get_labeled_score_vector()):
            if k not in kt_keys:
                continue
            rows.append((k,i+1,v))
        from dtk.table import Table
        from dtk.html import link
        from tools import sci_fmt
        name_map = self.ws.get_wsa2name_map()
        cols = [
                Table.Column('Drug',
                        idx=0,
                        cell_fmt=lambda x:link(
                                name_map[x],
                                self.ws.reverse('moldata:annotate',x),
                                new_tab=True,
                                )
                        ),
                Table.Column('Rank',
                        idx=1,
                        ),
                Table.Column('Score',
                        idx=2,
                        cell_fmt=sci_fmt,
                        ),
                ]
        self.context['above_table'] = Table(
                [x for x in rows if x[1] < self.cutoff],
                cols,
                )
        self.context['below_table'] = Table(
                [x for x in rows if x[1] >= self.cutoff],
                cols,
                )
    def validate_plotdir(self):
        import os
        self.plotdir = os.path.join(
                PathHelper.tmp_publish(self.ws.id,'scoreplot'),
                '.'.join([self.score,self.dtc,self.ds]),
                )
        from dtk.plot import validate_plotdir
        validate_plotdir(self.plotdir,[
                ('ds:'+self.ds,' '.join(str(x) for x in sorted(self.kts))),
                ])
    def wsa_scatterplot_options(self):
        from dtk.plot import Color
        return dict(
                refline=False,
                ids=('drugpage',self.ids),
                text=self.names,
                class_idx = [
                        1 if x[0] in self.kts else 0
                        for x in self.ordering
                        ],
                classes = [
                        ('Unknown',{'color':Color.default, 'opacity' : 0.2}),
                        ('KT',{'color':Color.highlight, 'opacity' : 0.7})
                        ],
                )
    def calc_prob_plot(self,plot=True):
        if not self.calibrated:
            return
        from dtk.plot import scatter2d,Color
        return scatter2d('rank','probability',
                enumerate(self.calibrated),
                title='Probability by Rank',
                **self.wsa_scatterplot_options()
                )
    def calc_cal_plot(self,plot=True):
        if not self.calibrated:
            return
        from dtk.plot import scatter2d,Color
        return scatter2d('raw score','probability',
                zip(self.vals,self.calibrated),
                title='Probability vs Raw Score',
                **self.wsa_scatterplot_options()
                )
    def calc_enrichment_plot(self,metric_name,do_plot):
        from dtk.enrichment import EnrichmentMetric
        from tools import sci_fmt
        Metric = EnrichmentMetric.lookup(metric_name)
        m = Metric()
        m.evaluate(self.emi)
        self.other_metrics.append((m.label(),sci_fmt(m.rating)))
        if do_plot:
            return m.plot(self.emi)
    def calc_score_plot(self,plot=True):
        from dtk.plot import score_plot
        return score_plot(self.emi.get_labeled_score_vector(),
                          [self.emi.get_kt_set()],
                          dtc=self.dtc,
                          )

    def calc_dea_plot(self,plot=True):
        from dtk.enrichment import EnrichmentMetric
        from tools import sci_fmt
        metrics={}
        for metric in ('DEA_AREA','SigmaOfRank','SigmaOfRank1000', 'SigmaOfRankCondensed', 'SigmaOfRank1000Condensed'):
            if self.dtc != 'wsa' and 'Condensed' in metric:
                continue
            Metric = EnrichmentMetric.lookup(metric)
            m = Metric()
            metrics[metric] = m
            m.evaluate(self.emi)
            self.other_metrics.append((m.label(),sci_fmt(m.rating)))
        if not plot:
            return
        m = metrics['DEA_AREA']
        if self.dtc =='uniprot':
            return m.plot(self.emi,dtc=self.dtc,ids=self.ids)
        elif self.dtc =='wsa':
            return m.plot(self.emi)
    def calc_aur_plot(self,plot=True):
        return self.calc_enrichment_plot('AUR',plot)
    def calc_aur_condensed_plot(self,plot=True):
        return self.calc_enrichment_plot('AURCondensed',plot)
    def calc_aps_plot(self,plot=True):
        return self.calc_enrichment_plot('APS',plot)
    def calc_entropy_plot(self):
        # XXX If we treat the entropy at the 'best' split point as a score,
        # XXX this could be pushed into dtk.enrichments.
        from dtk.enrichment import entropy
        total_scores = self.emi.n_scores()
        total_hits = self.emi.n_kts()
        in_hits = 0
        from dtk.scores import get_ranked_groups
        result = []
        for ahead,tied in get_ranked_groups(self.emi._score):
            hits = len(set(tied) & self.emi._kt_set)
            # uncomment to only calculate if # of KTs changes
            # if not hits: continue
            in_hits += hits
            in_scores = ahead+len(tied)
            in_frac = in_hits/float(in_scores)
            out_scores = total_scores - in_scores
            if out_scores:
                out_frac = (total_hits-in_hits)/float(out_scores)
            else:
                out_frac = 1
            result.append((
                    in_scores,
                    sum([
                        entropy([x,1-x])
                        for x in (in_frac,out_frac)
                        ]),
                    ))
        from dtk.plot import scatter2d
        return scatter2d(
                        'Drug rank',
                        'sum of entropy for both subsets',
                        result,
                        title='Total Entropy vs. Split Point',
                        refline=False,
                        linestyle='lines',
                        )
    def calc_kde_plot(self):
        n_points = 1000
        bandwidth = 0.05
        from sklearn.neighbors import KernelDensity
        all_scores = [x[1] for x in self.ordering]
        base = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        base.fit([[x] for x in all_scores])
        kts = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kts.fit([[x[1]] for x in self.ordering if x[0] in self.kts])
        distinct_scores = list(set(all_scores))
        distinct_scores.sort()
        top = float(max(distinct_scores))
        bot = float(min(distinct_scores))
        delta = top-bot
        points = [bot + x*delta/n_points for x in range(n_points)]
        from dtk.plot import PlotlyPlot
        import math
        base_plot=[
                math.exp(y)/math.exp(1)
                for y in base.score_samples([[x] for x in points])
                ]
        kts_plot=[
                math.exp(y)/math.exp(1)
                for y in kts.score_samples([[x] for x in points])
                ]
        return PlotlyPlot([
                    dict(
                            x=points,
                            y=base_plot,
                            name='background',
                            ),
                    dict(
                            x=points,
                            y=kts_plot,
                            name='KTs',
                            ),
                    ],
                    layout={
                            'title':'Kernel Density Estimate',
                            'xaxis':{'title':'Score'},
                            'yaxis':{'title':'Density'},
                            },
                )
    def calc_febe_plot(self,plot=True):
        from scripts.febe import febe
        fe = febe(
                named_scores=self.ordering,
                names_of_interest=self.kts,
                filename = None,
                )
        fe.run()
        self.context['best_p_ind'] = fe.peak_ind
        from tools import sci_fmt
        self.context['best_p_val'] = sci_fmt(fe.best_q)
        if not plot:
            return
        import math
        return self.febe_plot(fe.inds,
                       [-1.0 * math.log(i, 10) for i in fe.qvals],
                       fe.peak_ind
                      )
    def calc_sor_plot(self, plot=True):
        kt_ranks = self.emi.get_tie_adjusted_ranks()
        from dtk.enrichment import SigmaOfRank
        return self.sor_plot(kt_ranks, SigmaOfRank(), "Sigma Of Rank")

    def calc_sor1000_plot(self, plot=True):
        kt_ranks = self.emi.get_tie_adjusted_ranks()
        from dtk.enrichment import SigmaOfRank1000
        return self.sor_plot(kt_ranks, SigmaOfRank1000(), "Sigma Of Rank 1000")

    def calc_sor1000_condensed_plot(self, plot=True):
        kt_ranks = self.emi.get_condensed_emi().get_tie_adjusted_ranks()
        from dtk.enrichment import SigmaOfRank1000
        return self.sor_plot(kt_ranks, SigmaOfRank1000(), "Sigma Of Rank 1000 (Condensed)")


    def sor_plot(self,kt_ranks,sor, title):
        max_rank = sor.center + sor.width * 20
        n_points = 100
        xs = []
        ys = []
        for pi in range(n_points):
            rank = int(pi * max_rank / n_points)
            y = sor._score(rank)
            xs.append(rank)
            ys.append(y)
        from dtk.plot import Color

        curve_trace = {
                'x': xs,
                'y': ys,
                'mode': 'lines',
                'marker_color':Color.default,
                'name': 'background',
                }
        kt_trace = {
                'x': [rank for rank in kt_ranks if rank <= max_rank],
                'y': [sor._score(rank) for rank in kt_ranks if rank <= max_rank],
                'mode': 'markers',
                'marker_color':Color.highlight,
                'name': 'KTs',
            }

        xtitle = 'Rank'
        ytitle = title

        scores = [sor._score(rank) for rank in kt_ranks]
        near1 = len([score for score in scores if score > 0.95])
        near0 = len([score for score in scores if score < 0.05])
        middle = len(scores) - near1 - near0
        from dtk.plot import annotations
        ann = annotations('KT sigmas (~1/mid/~0): %d / %d / %d' % (
                        near1, middle, near0))

        from dtk.plot import PlotlyPlot
        return PlotlyPlot([curve_trace, kt_trace],
                    layout={
                            'title':title,
                            'xaxis':{'title':xtitle},
                            'yaxis':{'title':ytitle},
                            'annotations': ann,
                            },
                )



    def calc_wfebe_plot(self,plot=True):
        from scripts.febe import wfebe
        wfe = wfebe(
                named_scores=self.ordering,
                names_of_interest=self.kts,
                filename = None,
                )
        wfe.run()
        from tools import sci_fmt
        self.other_metrics.append(('wFEBE:',sci_fmt(wfe.final_score)))
        if not plot:
            return
        return self.febe_plot(wfe.inds, wfe.wfebe_scores,
            wfe.peak_ind, 'wfebe',
            'Weighted Fishers Exact Based Evaluation',
            ytitle = '-Log10(ref. set enrichment FDR)<br>*1-rank percentile'
                      )
    def calc_wfebe_condensed_plot(self,plot=True):
        from scripts.febe import wfebe
        wfe = wfebe(
                named_scores=self.emi.get_condensed_emi().get_labeled_score_vector(),
                names_of_interest=self.kts,
                filename = None,
                )
        wfe.run()
        from tools import sci_fmt
        self.other_metrics.append(('wFEBE Condensed:',sci_fmt(wfe.final_score)))
        if not plot:
            return
        return self.febe_plot(wfe.inds, wfe.wfebe_scores,
            wfe.peak_ind, 'wfebe',
            'Weighted Fishers Exact Based Evaluation (Condensed)',
            ytitle = '-Log10(ref. set enrichment FDR)<br>*1-rank percentile'
                      )
    def febe_plot(self, x, y, best_x, type='febe',
                  title='Fishers Exact Based Evaluation',
                  xtitle = 'Drug rank',
                  ytitle = '-Log10(ref. set enrichment FDR)'
                  ):
        from dtk.plot import PlotlyPlot
        return PlotlyPlot([
                    dict(
                         x=x,
                         y=y
                         ),
                    ],
                    layout={
                            'title':title,
                            'xaxis':{'title':xtitle},
                            'yaxis':{'title':ytitle},
                            'shapes':[
                                {
                                 'type': 'line',
                                 'x0': best_x,
                                 'y0': 0,
                                 'x1': best_x,
                                 'y1': max(y)*1.1,
                                 'line': {
                                     'color': 'red',
                                     'width': 2,
                                     'dash': 'dot',
                                 },
                            }]},
                )
    def calc_dbc_plot(self,plot=True):
        if self.dtc != 'wsa':
            return
        from tools import sci_fmt
        from dtk.prot_map import DpiMapping
        settings = self.job.settings()
        for key in ('p2d_file','dpi_file'):
            if key in settings:
                score_dpi = settings[key]
                print('Found the following DPI file:', score_dpi)
                break
        else:
            score_dpi = self.ws.get_dpi_default()
            print('No DPI file found; using default', score_dpi)
            self.message(
                "couldn't locate DPI in settings; using %s for background"%(
                        score_dpi,
                        )
                )
        dpi_obj = DpiMapping(score_dpi)
        wsa_id_map = dpi_obj.get_wsa_id_map(self.ws)
        from dtk.enrichment import DPI_bg_corr
        dbc = DPI_bg_corr(self.emi, wsa_id_map, dpi=score_dpi)
        dbc.evaluate()
        self.other_metrics.append(('DPI bg. corr. score:',sci_fmt(dbc.rating)))
        if not plot:
            return
        return dbc.plot(self.ws.id)

    def calc_rank_plot(self):
        return self.rank_plot(self.ordering, 'Rank plot')

    def calc_condensed_rank_plot(self):
        ordering = self.emi.get_condensed_emi().get_labeled_score_vector()
        return self.rank_plot(ordering, '(Condensed) Rank plot')

    def rank_plot(self, ordering, title):
        # The following numbers are used for calculating overlay
        # bars for the kts. This shouldn't be necessary (they're already
        # displayed in a different color in the underlying graph), but
        # when looking at thousands of drugs, the dithering makes the
        # rare colors tend to disappear. Using the 'shapes' option to
        # do overlays forces them to show up.  The colors array is still
        # useful because it affects the hover box color.
        x0s = []
        y1s = []
        for i,x in enumerate(ordering):
            if x[0] in self.kts:
                x0s.append(i)
                y1s.append(x[1])
        from dtk.plot import PlotlyPlot,Color
        colors=[
                Color.highlight if x in self.kts else Color.default
                for x in self.ids
                ]
        marker=dict(
                color=colors,
                )
        return PlotlyPlot([
                    dict(
                           type = 'bar',
                           y=[1]*len(ordering),
                           text=self.names,
                           marker=marker,
                           textposition='none',
                           ),
                    ],
                    layout={
                            'height':250,
                            'title':title,
                            #'xaxis':{'title':'Drug rank'},
                            'shapes':[
                              {'type':'rect',
                               'x0': x0s[i] - 0.35,
                               'y0':0,
                               'x1':x0s[i] + 0.35,
                               'y1':1,
                               'line':{'color':Color.highlight},
                              }
                              for i in range(len(x0s))
                             ]
                            },
                )
    def make_score_form(self,data):
        FormClass = self.source_table.make_form_class()
        return FormClass(data)
    def make_other_form(self,data):
        if self.dtc == 'wsa':
            ds_choices = self.ws.get_wsa_id_set_choices(
                    train_split=True, test_split=True)
            import dea
            class MyForm(forms.Form):
                ds = forms.ChoiceField(
                    label = 'Drugset',
                    choices = ds_choices,
                    initial = self.ds,
                    )
            return MyForm(data)
        elif self.dtc == 'uniprot':
            ds_choices = self.ws.get_uniprot_set_choices()
            import dea
            class MyForm(forms.Form):
                ds = forms.ChoiceField(
                    label = 'Protset',
                    choices = ds_choices,
                    initial = self.ds,
                    )
            return MyForm(data)
    def make_cutoff_form(self,data):
        class MyForm(forms.Form):
            cutoff = forms.IntegerField(
                label = 'Cutoff',
                initial = self.cutoff,
                )
        return MyForm(data)
    def redisplay_post_valid(self):
        p = self.cutoff_form.cleaned_data
        return HttpResponseRedirect(self.here_url(
                    cutoff=p['cutoff'],
                    ))
    def display_post_valid(self):
        p1 = self.score_form.cleaned_data
        self.source_table.update_source_data(p1)
        p2 = self.other_form.cleaned_data
        return HttpResponseRedirect(self.here_url(
                    mode=None,
                    score=p1['score'],
                    ds=p2['ds'],
                    ))
    def defaults_post_valid(self):
        self.sources.load_defaults()
        return HttpResponseRedirect(self.here_url())

