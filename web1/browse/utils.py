from builtins import range
from path_helper import PathHelper,make_directory
import os

import logging
logger = logging.getLogger(__name__)



def drug_search(
        pattern, # Will (default) search for this at the start of value
        version, # None will default to latest, which should have best clustering
        pattern_anywhere=False, # Slow; matches pattern anywhere in string.
        restrict_props=['synonym', 'canonical', 'mixture', 'brand', 'cas', 'override_name', 'native_id'],
        ):
    """
    Returns a list of (dpimergekey, matched_prop, matched_value, drugid)
    """
    from drugs.models import DpiMergeKey
    version = version or DpiMergeKey.max_version()

    out = []
    seen_keys = set()
    def find_keys(pat):
        tags = drug_search_internal(pat, version, pattern_anywhere, restrict_props, search_blobs=False)
        blobs = drug_search_internal(pat, version, pattern_anywhere, restrict_props, search_blobs=True)

        # We end up resolving and sorting on the python side rather than SQL.
        # I think it's doable with raw SQL, but the django system gets tripped up when trying to combine the
        # blob and tag queries and then ordering on an annotation of them.
        # But this is fast enough anyway, at least for reasonable queries.
        for dpimergekey, matched_value, matched_prop, drugid in tags:
            if dpimergekey not in seen_keys:
                seen_keys.add(dpimergekey)
                out.append((dpimergekey, matched_value, matched_prop, drugid))

        for dpimergekey, matched_value, matched_prop, drugid in blobs:
            if dpimergekey not in seen_keys:
                seen_keys.add(dpimergekey)
                out.append((dpimergekey, matched_value, matched_prop, drugid))

    find_keys(pattern)

    # We know that there is often ambiguity in whether a name is expressed with
    # a space or a hyphen (or neither).  If the search includes a single one of
    # either character, expand it to include the other possibilities.
    SUBS = {
        '-': [' ', ''],
        ' ': ['-', ''],
        }
    for k, repls in SUBS.items():
        if pattern.count(k) == 1:
            for repl in repls:
                find_keys(pattern.replace(k, repl))

    # Sort primary by length of matched value; the shorter it is, the closer it
    # must be to the original string we searched for.
    out.sort(key=lambda x: (len(x[2]), x[2]))

    return out

def drug_search_internal(
        pattern, # Will (default) search for this at the start of value
        version, # We'll cluster according to DpiMergeKey from this version
        pattern_anywhere=False, # Slow; matches pattern anywhere in string.
        restrict_props=['synonym', 'canonical', 'mixture', 'brand'],
        search_blobs=False, # Search blobs or tags
        ):
    from drugs.models import Tag, Blob, Drug, DpiMergeKey
    if pattern_anywhere is True:
        searchtype = '__icontains'
    elif pattern_anywhere == 'exact':
        searchtype = ''
    else:
        searchtype = '__istartswith'

    if search_blobs:
        proptype = 'blob'
    else:
        proptype = 'tag'


    search = {
            'dpimergekey__version': version,
            f'{proptype}__value{searchtype}': pattern,
            f'{proptype}__prop__name__in': restrict_props,
            }

    qs = Drug.objects.filter(**search)
    qs = qs.values_list('dpimergekey__dpimerge_key').distinct()

    from django.db.models import Min, Value

    qs = qs.annotate(
            matched_prop=Min(f'{proptype}__prop__name'),
            matched_value=Min(f'{proptype}__value'),
            drugid=Min('pk'),
            )
    return qs

def drug_search_wsa_filter(qs_ws,pattern):
    from django.db.models import Q
    return qs_ws.filter(Q(agent__tag__value__icontains=pattern)
                        | Q(agent__blob__value__icontains=pattern)
                        ).distinct()

def prep_heatmap(heatmap_data):
    names = list(set([t for v in heatmap_data.values() for t in v.keys()]))
    if len(names) < 2:
        return None, None
    return [n.replace(",", "_") for n in names], build_cor_mat(heatmap_data, names)
def build_cor_mat(heatmap_data, names):
    from scipy.stats.stats import spearmanr
    dat_mat, _ = build_dat_mat(heatmap_data, names)
    cor_mat = []
    for i in range(len(names)):
        row_data = []
        for j in range(len(names)):
            if i == j :
                cor = '1.0'
            elif i > j:
                cor = cor_mat[j][i]
            else:
                cor = str(spearmanr(dat_mat[i], dat_mat[j])[0])
            row_data.append(cor)
        cor_mat.append(row_data)
    return cor_mat
def build_dat_mat(heatmap_data, names):
    dat_mat = []
    minnum = min(len(names) * 0.8, len(names) - 1)
    # Skip any proteins that don't show up in at least K of our datasets.
    prots = [p for p in heatmap_data.keys() if len(list(heatmap_data[p].keys())) >= minnum]
    print("number of prots used for heatmap:", len(prots), "out of a total of", len( list(heatmap_data.keys())))

    # heatmap data is signed evidence score for:
    # {prot -> {tissue -> score} }
    for t in names:
        row_data = []
        for p in prots:
            try:
                row_data.append(heatmap_data[p][t])
            except KeyError:
                row_data.append(0.0)
        dat_mat.append(row_data)

    # Now we have a matrix:
    # [[t1p1, t1p2, ... t1pn],
    #  [t2p1, t2p2, ... t2pn],
    #  ....
    return dat_mat, prots

# XXX this is only used in the legacy view dea page; once that's gone,
# XXX delete it
class EnrichmentResultsBase(object):
    def __init__(self,base_dir,job_id):
        self.dirpath=os.path.join(base_dir,str(job_id)+'/')
        self.summaries = {}
        self.names = None
    def get_names(self):
        if self.names is None:
            try:
                suffix = '_summaryStats.tsv'
                self.names = []
                for name in os.listdir(self.dirpath):
                    if name.endswith(suffix):
                        self.names.append(name[:-len(suffix)])
            except Exception as ex:
                logger.warning("reading '%s' got exception %s"
                            ,self.dirpath
                            ,str(ex)
                            )
                self.names = []
        return self.names
    def get_summary(self,name):
        if name not in self.summaries:
            try:
                filename = os.path.join(self.dirpath,name +'_summaryStats.tsv')
                f = open(filename)
                lines = f.read().split('\n')
                recs = [x.split('\t') for x in lines]
                self.summaries[name] = dict(zip(recs[0],recs[1]))
            except Exception as ex:
                logger.warning("reading '%s' got exception %s"
                            ,filename
                            ,str(ex)
                            )
                self.summaries[name] = None
        return self.summaries[name]
    def get_href(self,name):
        path = os.path.join(self.dirpath, name + '_DEAPlots.pdf')
        if not os.path.exists(path):
            return ""
        return PathHelper.url_of_file(path)
    def format_link(self,name,subtype,link,suffix='.pdf'):
        path = os.path.join(self.dirpath, name + '_' + subtype + suffix)
        if not os.path.exists(path):
            return ""
        return '&nbsp;&nbsp;&nbsp;<a href="%s">%s</a>' % (
                PathHelper.url_of_file(path),
                link
                )
    def format_single(self,name,suffix='.pdf',label=''):
        s = self.get_summary(name)
        if s is None:
            return ''
        if not label:
            label=name
            label = label.replace('Treatments','')
            label = label.replace('Background','Bg')
        from tools import sci_fmt
        r='%s NES: <b>%s</b>' % (label,sci_fmt(s['NES']),)
        if 'alpha' in s:
            ci = 100 * (1-float(s['alpha']))
            r += ' (%0.1f%% CI %s-%s, p=%s)' % (
                    ci,
                    sci_fmt(s['NES_lower']),
                    sci_fmt(s['NES_upper']),
                    sci_fmt(s['P-value']),
                    )
        r += self.format_link(name,'DEAPlots','DEA plots',suffix)
        from django.utils.safestring import mark_safe
        return mark_safe(r)
    def format_multiple(self,names):
        l = []
        for name in names:
            r=self.format_single(name)
            if r:
                l.append(r)
        from django.utils.safestring import mark_safe
        return mark_safe('<br>'.join(l))
    def single_val(self,name,column):
        s = self.get_summary(name)
        if s is None:
            return ''
        return s[column]

################################################################################
# The extract_something_option functions and the Option classes are an
# older attempt at managing user selections. The functions have mostly
# been superceded by the DumaView querystring parsing support. The
# classes combine the parsing with tools that allow template rendering
# of multiple option values as links.
################################################################################
def extract_string_option(request,parm,default=''):
    try:
        selected = request.GET[parm]
    except KeyError:
        selected = default
    return selected

def extract_int_option(request,parm,low_limit=0,high_limit=1,default=0):
    try:
        selected = int(request.GET[parm])
    except (KeyError,ValueError) as e:
        selected = default
    if not low_limit <= selected <= high_limit:
        selected = default
    return selected

def extract_float_option(request,parm,low_limit=0,high_limit=1,default=0.0):
    try:
        selected = float(request.GET[parm])
    except (KeyError,ValueError) as e:
        selected = default
    if not low_limit <= selected <= high_limit:
        selected = default
    return selected

def extract_enum_option(request,parm,num_choices,default=0):
    return extract_int_option(request,parm,0,num_choices-1,default)

def extract_bool_option(request,parm,default=0):
    return extract_int_option(request,parm,0,1,default)

def extract_list_option(request,parm):
    s = extract_string_option(request,parm)
    if s:
        return s.split(",")
    return []

class EnumOptionBase:
    def qparm_val_of(self,opt):
        try:
            index=self.options.index(opt)
        except ValueError:
            index=0
        return str(index)
    def is_selected(self,opt):
        return opt == self.options[self.selected]
    def label_of(self,opt):
        return str(opt)

class LiteralOptionBase:
    def qparm_val_of(self,opt):
        return str(opt)
    def is_selected(self,opt):
        return opt == self.selected
    def label_of(self,opt):
        return str(opt)

#######
# The Option classes below extract values from query strings, and work with
# navtools template tags to let the user select options from within the page.
#
# An option class should derive from one of the above base classes to fill
# in the access methods used by the option_ template tags.
#
# An Option class should define:
# 'parm_name' - the querystring parameter name
# 'options' - list of option labels, in enum order
# 'selected' - the selected option
# plus members holding other selection-dependent information
#######

class ProteinEvidenceOption(LiteralOptionBase):
    # demonstrates override
    def label_of(self,opt):
        if opt<=1.:
            return str(opt)
        return "Dataset-specific thresholds"
    def __init__(self,request,ws,options):
        self.parm_name = 'evidence'
        self.options=options
        self.selected = extract_float_option(request
                                , self.parm_name
# min and max value
                                , 0., 2.0
# Default, setting to 2 indicated tissue-specific values should be used
# this is interpretted in code in ge/ts_views,
# though ultimately acted on by verify_tissue_threshold in ge/utils
                                , 2.
                                )


class JobShowOption(EnumOptionBase):
    def __init__(self,request):
        self.parm_name = 'show'
        self.options = ['Active only', 'All Jobs', 'Failed only']
        self.selected = extract_enum_option(request
                                ,self.parm_name
                                ,len(self.options)
                                )
        self.repeat=0
        from runner.models import Process
        if self.selected == 1:
            # show all the latest jobs
            self.title = 'Latest'
            self.qs=Process.latest_jobs_qs()
        elif self.selected == 2:
            # show only jobs in error
            self.title = 'Failed'
            self.qs=Process.failed_jobs_qs()
        else:
            # default to showing only active jobs
            self.title = 'Active'
            self.qs=Process.active_jobs_qs()
            self.repeat=10

class JobWorkspaceOption(EnumOptionBase):
    def __init__(self,request,ws,qs):
        self.parm_name = 'ws_only'
        self.options = ['All Workspaces', 'This Workspace Only']
        self.selected = extract_enum_option(request
                                ,self.parm_name
                                ,len(self.options)
                                )
        if self.selected == 1:
            self.title = 'Workspace'
            from runner.process_info import JobCrossChecker
            jcc = JobCrossChecker()
            self.qs = qs.filter(name__in=jcc.ws_jobnames(ws))
        else:
            self.title = 'All'
            self.qs = qs

# XXX These probably belong somewhere else
def get_dpi_druggable_prots():
    from browse.default_settings import DpiDataset
    from dtk.prot_map import DpiMapping
    dpi = DpiMapping(DpiDataset.default_global_default())
    prots = set(dpi.get_filtered_uniq_target())
    return prots

def get_dpi_or_sm_druggable_prots():
    dpi_druggable_prots = get_dpi_druggable_prots()
    from dtk.open_targets import OpenTargets
    from browse.default_settings import openTargets
    otarg = OpenTargets(openTargets.latest_version())
    sm_druggable_prots = otarg.get_small_mol_druggable_prots()
    return sm_druggable_prots | dpi_druggable_prots
