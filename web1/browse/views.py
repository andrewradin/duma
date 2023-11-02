from __future__ import print_function
# Create your views here.
from builtins import range
from django.http import HttpResponse,HttpResponseRedirect,JsonResponse
from django.shortcuts import render
from django.template import RequestContext
from django.urls import reverse
from browse.models import Workspace,Tissue,Sample,WsAnnotation,Likelihood,TissueSet,DrugSet,Election,Vote
from browse.utils import extract_string_option,extract_list_option,extract_float_option,extract_bool_option
from browse.utils import JobShowOption,JobWorkspaceOption
from browse.utils import drug_search_wsa_filter
from browse.utils import EnrichmentResultsBase
from notes.models import Note
from dtk.duma_view import DumaView,list_of,boolean
from dtk.table import Table,SortHandler
from django import forms
from django.contrib.auth.decorators import login_required
from django.utils.safestring import mark_safe
from django.views.generic import TemplateView
import math
import os
import pwd
import requests
import json
import glob
import csv
import datetime
from lxml import html, etree
from tools import touch,ProgressReader,ProgressWriter,obfuscate
from path_helper import PathHelper,make_directory
from decimal import Decimal
from runner.process_info import JobInfo,JobCrossChecker
from runner.models import Process
from drugs.models import Prop,Collection
from algorithms.exit_codes import status_file_line,ExitCoder
from aws_op import Machine
from reserve import ResourceManager
import numpy as np

import logging
import six
logger = logging.getLogger(__name__)

def make_ctx(request,ws,function,others):
    # 'function' is where to go in the index dropdown
    ret = { 'ws':ws
             ,'function_root':function
             ,'now':datetime.datetime.now()
             ,'ph':PathHelper
             }
    ret.update(others)
    return ret

@login_required
def publish(request,path):
    # This is a quick-and-dirty solution to requiring that users be
    # authenticated in order to fetch image files.  It's generally not
    # recommended to pass static content through Django, but we're
    # low-volume enough that it shouldn't matter.
    # XXX if it turns out to be a problem, we can:
    # XXX - stream the data rather than buffering it:
    # XXX https://djangosnippets.org/snippets/365/
    # XXX - use X-Sendfile with Apache:
    # XXX https://djangosnippets.org/snippets/2226/
    # XXX http://francoisgaudin.com/2011/03/13/serving-static-files-with-apache-while-controlling-access-with-django/
    # XXX - maybe, have Apache authenticate the publish directory
    # XXX https://docs.djangoproject.com/en/dev/howto/deployment/wsgi/apache-auth/
    from django.views.static import serve

    root,relpath  = PathHelper.path_of_pubfile('publish/'+path)

    from dtk.plot import PlotlyPlot
    PlotlyPlot.block_if_thumbnailing(os.path.join(root, relpath))

    response = serve(request,relpath,root)

    # Most of our static resources are immutable so default caching is fine.
    # For main.js, though, we want to make sure the user is getting the latest version,
    # otherwise the page might appear broken during a new release (and during dev).
    if path == 'js/main.js':
        # Despite the name, no-cache still allows caching, but the browse must
        # revalidate before using it.
        # We use ConditionalGetMiddleware, so it will get a 304 Not Modified
        # and use the cache if nothing has changed.
        response['Cache-Control'] = 'no-cache'

    return response

@login_required
def protected_static(request,path):
    # Similar to above, serving static content behind django auth.
    from django.views.static import serve
    root = os.path.join(PathHelper.website_root, 'static')
    return serve(request,path,root)


@login_required
def index(request):
    show_all=extract_bool_option(request,'all')
    duma_admin=request.user.groups.filter(name='duma_admin').exists()
    queryset = Workspace.objects.all()
    if not show_all:
        queryset = queryset.filter(active=True)
    queryset = queryset.order_by('name')
    # Need to resolve the list for prefetch to be useful.
    queryset = list(queryset.prefetch_related('stagestatus_set'))
    results = []
    from dtk.html import link

    def make_date_str(dt):
        from django.utils.safestring import mark_safe
        # We use non-breaking hyphens, otherwise the page likes to split
        # these dates onto multiple lines.
        return mark_safe(dt.strftime("%Y&#8209;%m&#8209;%d"))

    for w in queryset:
        obj = {}
        results.append(obj)
        kt=WsAnnotation.indication_vals.KNOWN_TREATMENT
        obj['ref'] = w
        obj['link'] = link(w.name,w.reverse('workflow'))
        from stages import make_workflow, overall_progress

        workflow = make_workflow(w)
        from django.db.models import Min, Max
        from browse.models import Election
        elections = Election.objects.filter(ws=w)
        if not elections.exists():
            review_date = ''
        else:
            review_date = make_date_str(elections.aggregate(Max('due'))['due__max'])
        obj['review_date'] = review_date
        stage = None
        for i, (section, steps) in enumerate(workflow):
            for step in steps:
                if stage is None and not step.is_complete():
                    stage = f'{section} ({i+1}/{len(workflow)})'
                    stage_num = i
        if stage is None:
            stage = 'Done'
            stage_num = len(workflow)
        completion_frac = overall_progress(workflow)

        assert workflow[0][1][0]._name == 'Data Abundance'
        data_update_date = workflow[0][1][0].status_obj().changed_on

        from dtk.retrospective import selected_mols
        obj['hits'] = selected_mols(w).count()
        obj['completion'] = f'{int(completion_frac*100)}%'
        obj['completion_frac'] = completion_frac
        obj['stage'] = stage
        obj['stage_num'] = stage_num
        obj['created'] = make_date_str(w.created)
        obj['data_update_date'] = make_date_str(data_update_date)
    return render(request
                ,'browse/index.html'
                ,make_ctx(request,None,None
                  ,{
                   'workspaces': results,
                   'show_all': show_all,
                   'show_create_link':duma_admin,
                   }
                  )
                )


class WorkflowView(DumaView):
    template_name='browse/workflow.html'
    button_map={
            'update':[],
            'update_note':['note'],
            }
    def custom_context(self):
        import stages

        workflow = stages.make_workflow(self.ws)
        completion = {}
        for title, parts in workflow:
            completion[title] = f'{int(100*stages.completion(parts))}%'

        overall_completion = stages.overall_progress(workflow)
        completion['all'] = f'{int(100*overall_completion)}%'


        self.context_alias(
                flow = workflow,
                completion = completion,
                )

    def update_post_valid(self):
        from stages import WorkflowStage
        from browse.models import StageStatus
        import json
        data = json.loads(self.request.POST['query'])
        name = data['name']
        status = data['status']
        stage = WorkflowStage.get_or_create_obj(self.ws, name)
        stage.status = StageStatus.statuses.find('label', status)
        stage.save()

        from browse.models import StageStatusLog
        StageStatusLog.objects.create(
            stage=stage,
            status=stage.status,
            user=self.request.user.username
        )


        return JsonResponse({
            'button_classes': WorkflowStage.button_classes_for_status(stage.status),
            'status_text': WorkflowStage.status_text_for_status(stage.status),
            })
    def make_note_form(self,data):
        class NoteForm(forms.Form):
            note = forms.CharField(
                    widget=forms.Textarea(attrs={'rows':'1','cols':'50'}),
                    required=False,
                    initial=self.ws.get_disease_note_text(),
                    )
        return NoteForm(data)
    def update_note_post_valid(self):
        p = self.note_form.cleaned_data
        from notes.models import Note
        Note.set(self.ws,
                'disease_note',
                self.request.user.username,
                p['note'],
                )
        return HttpResponseRedirect(self.here_url())

class DictCoder:
    def __init__(self,sep='|'):
        self.sep = sep
    def encode(self,d):
        return self.sep.join([k+self.sep+v for k,v in six.iteritems(d)])
    def decode(self,s):
        l = [x for x in s.split(self.sep)]
        result = {}
        for i in range(0,len(l)/2):
            k = l[2*i]
            v = l[2*i+1]
            result[k] = v
        return result

class AttrFileForm(forms.Form):
    csv = forms.ChoiceField(label='Import Drug Attributes:'
            ,choices=(('','None'),)
            ,required=False
            )
    def __init__(self, *args, **kwargs):
        super(AttrFileForm,self).__init__(*args, **kwargs)
        # reload choices on each form load
        from drugs.models import Collection
        self.fields['csv'].choices = Collection.get_attr_upload_choices()

class PathFileForm(forms.Form):
    gmt = forms.ChoiceField(label='Import Protein Pathways:'
            ,choices=(('','None'),)
            ,required=False
            )
    pathway_file_suffix = '.uniprot.gmt'
    def is_pathway_fn(self,fn):
        return fn.endswith(self.pathway_file_suffix)
    def short_pathway_name(self,fn):
        assert self.is_pathway_fn(fn)
        return fn[:-len(self.pathway_file_suffix)]
    def __init__(self, *args, **kwargs):
        super(PathFileForm,self).__init__(*args, **kwargs)
        # reload choices on each form load
        from dtk.s3_cache import S3Bucket
        glee = S3Bucket('glee')
        self.fields['gmt'].choices = [('','None')] + [
                (x,self.short_pathway_name(x))
                for x in glee.list()
                if self.is_pathway_fn(x)
                ]

class UploadView(DumaView):
    template_name='browse/upload.html'
    index_dropdown_stem=''
    GET_parms = {
            'show':(str,None),
            }
    button_map={
            'prop_refresh':[],
            'versioned_drug_upload':[],
            # These are no longer needed, but the code is left in place
            # for now. There's a corresponding change in the
            # _drug_coll_section template
            #'drug_import':['attr'],
            #'drug_refresh_imported':[],
            'prot_refresh':['prot'],
            # XXX Pathway import is disabled for now. This code is based
            # XXX on the old gmt files built by the gene_sets directory.
            # XXX We now use data straight from reactome, and access it
            # XXX via the ETL files instead of via the database. This should
            # XXX all be removed once we're sure we're happy with the new
            # XXX approach.
            #'path_import':['path'],
            }
    def custom_setup(self):
        from drugs.tools import CollectionUploadStatus
        self.coll_stat = CollectionUploadStatus()
        from browse.models import ProteinUploadStatus
        self.last_prot_fn = ProteinUploadStatus.current_upload()
        from dtk.s3_cache import S3Bucket
        self.s3_prot_files = S3Bucket('uniprot').list(cache_ok=True)
    def custom_context(self):
        self.context['show']=self.show
        self.context['props']=Prop.objects.all()
        self.context['last_uploads_table']=self.format_uploads()
        self.context['coll_uploads']=self.coll_stat.needed_attrs
        self.context['match_uploads']=[
                x.to_string()
                for x in self.coll_stat.needed_clusters
                ]
        self.context['show']=self.show
    def path_import_post_valid(self):
        raise NotImplementedError('no versioned file support')
        gmt = self.path_form.cleaned_data['gmt']
        if gmt:
            from dtk.s3_cache import S3File
            s3f = S3File('glee',gmt)
            s3f.fetch()
            from browse.models import Protein
            Protein.upload_pathway_from_gmt(
                    self.path_form.short_pathway_name(gmt),
                    s3f.path(),
                    )
        return HttpResponseRedirect(self.here_url(show='prots'))
    def versioned_drug_upload_post_valid(self):
        from drugs.models import Collection,DpiMergeKey
        # upload the most recent version of any changed attribute files,
        # and all new cluster definition files
        # (note that, if a failure occurs part-way through, only the files
        # that weren't processed will be in the needed lists)
        any_attr_error=False
        from dtk.s3_cache import S3File
        for fn in self.coll_stat.needed_attrs:
            ok = Collection.attr_loader(fn,versioned=True)
            if not ok:
                any_attr_error=True

        # Also upload the cluster files into dpimergekey.
        # This relies on the corresponding collections already having been
        # imported (for fill_missing_dpimerge).
        added_dpimerge = False 
        if not any_attr_error:
            from dtk.files import get_file_records
            for vfn in self.coll_stat.needed_clusters:
                s3f = S3File(self.coll_stat.cluster_bucket,vfn.to_string())
                s3f.fetch()
                DpiMergeKey.load_from_keysets(
                        vfn.to_string(),
                        get_file_records(s3f.path(),keep_header=None),
                        vfn.version,
                        )
                DpiMergeKey.fill_missing_dpimerge(vfn.version)
                added_dpimerge = True

        if not added_dpimerge:
            # We just added a new collection without a corresponding new DpiMergeKey file.
            # The molecules from this collection won't have any entries in the table.
            # For convenience, we'll just make sure they're in the latest version of the table,
            # though in most cases there should be a new dpimerge file when there are new collections.
            DpiMergeKey.fill_missing_dpimerge(DpiMergeKey.max_version())
        return HttpResponseRedirect(self.here_url(show='coll'))
    def drug_import_post_valid(self):
        p = self.attr_form.cleaned_data
        if p['csv']:
            Collection.attr_loader(p['csv'])
        return HttpResponseRedirect(self.here_url(show='coll'))
    def drug_refresh_imported_post_valid(self):
        self.log("Refreshing all imported drug collections")
        # We sort to make sure that 'create' imports all occur before
        # 'm.' imports.  If we get any additional ordering constraints,
        # may need to make more sophisticated.
        for upload in sorted(self.last_uploads, key=lambda x: x.filename):
            self.log("Refreshing %s" % upload.filename)
            Collection.attr_loader(upload.filename)
        return HttpResponseRedirect(self.here_url(show='coll'))
    def prop_refresh_post_valid(self):
        l = sorted(self.coll_stat.props_vfns,key=lambda x:-x.version)
        self.log("Refreshing properties from "+l[0].to_string())
        from dtk.s3_cache import S3File
        f=S3File('matching',l[0].to_string())
        f.fetch()
        from dtk.files import get_file_records
        src=get_file_records(f.path())
        from dtk.readtext import convert_records_using_header
        Prop.load_from_records(convert_records_using_header(src))
        return HttpResponseRedirect(self.here_url(show='props'))
    def prot_refresh_post_valid(self):
        p = self.context['prot_form'].cleaned_data
        from browse.models import import_proteins
        import_proteins(p['prot_file'])
        return HttpResponseRedirect(self.here_url(show='prots'))
    def format_uploads(self):
        self.last_uploads = []
        from drugs.models import UploadAudit
        seen = set()
        for ua in UploadAudit.objects.all().order_by('-timestamp'):
            if ua.filename in seen:
                continue
            seen.add(ua.filename)
            self.last_uploads.append(ua)
        from dtk.html import pad_table
        from dtk.text import fmt_time
        return pad_table(['file','date','status'],[
                (ua.filename,fmt_time(ua.timestamp),'OK' if ua.ok else 'ERROR')
                for ua in self.last_uploads
                ])
    def make_attr_form(self,data):
        return AttrFileForm(data)
    def make_path_form(self,data):
        return PathFileForm(data)
    def make_prot_form(self,data):
        from dtk.files import VersionedFileName
        choices = VersionedFileName.get_choices(
                file_class='uniprot',
                paths=self.s3_prot_files,
                )
        class MyForm(forms.Form):
            prot_file = forms.ChoiceField(
                    label = 'Uniprot File',
                    choices = choices,
                    )
        return MyForm(data)

# tools
def is_demo(user):
    return DumaView.user_is_demo(user)

def post_ok(request):
    if not is_demo(request.user):
        return True
    from django.contrib import messages
    messages.add_message(request, messages.INFO,
            "Action disallowed in demo mode",
            )
    return False

class PcaView(DumaView):
    '''Display PCA results for a feature vector.'''
    template_name='browse/pca.html'
    GET_parms = {
            'fm_code':(str,None),
            'x':(str,'pc1'),
            'y':(str,'pc2'),
            }
    button_map={
            'display':['config'],
            }
    def make_config_form(self,data):
        feature_choices = [
                (x.code,x.label)
                for x in self.feature_data
                ]
        class MyForm(forms.Form):
            fm_code = forms.ChoiceField(
                    label = 'Feature Matrix',
                    choices = self.ws.get_feature_matrix_choices(),
                    initial = self.fm_code,
                    )
            if feature_choices:
                x = forms.ChoiceField(
                    label = 'X Axis',
                    choices = feature_choices,
                    initial = self.x,
                    )
                y = forms.ChoiceField(
                    label = 'Y Axis',
                    choices = feature_choices,
                    initial = self.y,
                    )
        return MyForm(data)
    def display_post_valid(self):
        p = self.context['config_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))
    def custom_setup(self):
        self.context_alias(plotly_plots=[])
        self.feature_data=[]
        if self.fm_code:
            self.fm = self.ws.get_feature_matrix(self.fm_code)
            self.set_counts()
            self.prep_raw_fdrs()
            self.calc_pca()
    def custom_context(self):
        if self.fm_code:
            self.scatterplot()
            self.variance_plot()
            self.source_plots()
            self.feature_plot()
            self.js_plot()
    def set_counts(self):
        # convert matrix to numpy array and fill in unknowns
        self.matrix = np.nan_to_num(self.fm.data_as_array())
        self.n_samples = len(self.fm.target)
        self.n_attributes = len(self.fm.feature_names)
        assert self.n_attributes > 1
        self.target_idxs = set([
                i
                for i,t in enumerate(self.fm.target)
                if t
                ])
        self.n_targets = len(self.target_idxs)


    def calculate_JS_div(self, p, q):
        import scipy
        p /= p.sum()
        q /= q.sum()
        m = (p + q) / 2
        return (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    from collections import namedtuple
    FeatureData=namedtuple('FeatureData',['code','label','fdr',
                                          'kt_scores', 'js_scores'])

    def record_feature(self,code,label,vec):
        from dtk.num import avg_sd
        import numpy as np
        import scipy.stats
        data = list(enumerate(vec))
        vals = np.array([x[1] for x in data]) #numpy array of all value
        hits = np.array([x[1] for x in data if x[0] in self.target_idxs])
        misses = np.array([x[1] for x in data if x[0] not in self.target_idxs])
        if len(hits) == 0 or len(misses) == 0:
            self.log('skipping %s; %d hits %d misses',
                        label,
                        len(hits),
                        len(misses),
                        )
            return
        mode = scipy.stats.mode(vals)[0]
        vals_nomode = vals[np.where(vals!=mode)[0]]
        if len(vals_nomode) == 0:
            self.log('skipping %s; all %d values are %f',
                        label,
                        len(vals),
                        mode,
                        )
            return
        u1,o1 = avg_sd(hits)
        u2,o2 = avg_sd(misses)
        fdr = (u1-u2)**2/(o1**2+o2**2)
        #Chooses number of bins based on the no mode IQR
        q1 = np.percentile(vals_nomode, 25)
        q3 = np.percentile(vals_nomode, 75)
        bin_width = (q3-q1)/10
        #This is for a pathological case if the bin size is 0
        if bin_width < np.finfo(np.float32).eps:
            bin_width = np.finfo(np.float32).eps*1000
        bins = np.arange(np.min(vals),np.max(vals),bin_width)
        hitassign = np.digitize(hits, bins)
        missassign = np.digitize(misses, bins)
        bin_means_hits = np.array([len(hits[hitassign == i])/float(len(hits)) for i in range(0, len(bins)+1)])
        bin_means_misses = np.array([len(misses[missassign == i])/float(len(hits)) for i in range(0, len(bins)+1)])
        u0,o0 = avg_sd(vec)
        js_div = self.calculate_JS_div(bin_means_hits, bin_means_misses)

        self.feature_data.append(self.FeatureData(
                code,
                label,
                fdr,
                [(x-u0)/o0 for x in hits],
                js_div
                ))
        if code == self.x:
            self.x_data = vec
            self.x_label = label
            self.x_bins = bins
            #print bins
            #print np.unique(missassign)
            #print np.unique(hitassign)
            #print np.unique(bin_means_hits)
            #print np.unique(bin_means_misses)
        if code == self.y:
            self.y_data = vec
            self.y_label = label
            self.y_bins = bins
            #print np.unique(missassign)
            #print np.unique(hitassign)
            #print np.unique(bin_means_hits)
            #print np.unique(bin_means_misses)
    def prep_raw_fdrs(self):
        transpose = self.matrix.T
        for i,data in enumerate(transpose):
            label = self.fm.feature_names[i]
            code = 'f%d' % (i+1)
            self.record_feature(code,label,data)
    def calc_pca(self):
        import sklearn
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(self.matrix)
        self.explained_variance = pca.explained_variance_ratio_
        # record underlying features for each PC in scatterplot
        self.components = []
        import re
        for code in (self.x,self.y):
            m = re.match(r'pc([0-9]+)$',code)
            if m:
                pc_idx = int(m.group(1))-1
                self.components.append((code,pca.components_[pc_idx]))
        # transform data; put one PC in each row
        pcs = pca.transform(self.matrix)
        transpose = zip(*pcs)
        for i,data in enumerate(transpose):
            label = 'PC%d' % (i+1)
            code = label.lower()
            self.record_feature(code,label,data)
    def scatterplot(self):
        xy = zip(self.x_data,self.y_data)
        extras = {}
        if self.fm.get('sample_key') == 'wsa':
            extras['ids'] = ('drugpage',self.fm.sample_keys)
            name_map = self.ws.get_wsa2name_map()
            extras['text'] = [
                    name_map[wsa_id]
                    for wsa_id in self.fm.sample_keys
                    ]
        from dtk.plot import scatter2d,Color
        plot = scatter2d(
                    self.x_label,
                    self.y_label,
                    xy,
                    refline=False,
                    classes=[
                            ('Unknown',{'color':Color.default, 'opacity':0.2}),
                            ('KT',{'color':Color.highlight, 'opacity':0.6}),
                            ],
                    class_idx=self.fm.target,
                    jitter=True,
                    bins=(self.x_bins, self.y_bins),
                    **extras
                    )
        self.plotly_plots.append(('pca', plot))
    def variance_plot(self):
        from dtk.plot import scatter2d
        # now plot the explained variance decay
        plot = scatter2d(
                    'component',
                    'explained variance',
                    enumerate(self.explained_variance),
                    refline=False,
                    )
        self.plotly_plots.append(('pca_variance', plot))
    def source_plots(self):
        from dtk.plot import scatter2d
        for code,data in self.components:
            label = code.upper()
            l = list(zip(self.fm.feature_names,data))
            l.sort(key=lambda x:abs(x[1]),reverse=True)
            plot = scatter2d(
                        'feature',
                        'contribution to '+label,
                        enumerate([x[1] for x in l]),
                        refline=False,
                        text=[x[0] for x in l],
                        )
            self.plotly_plots.append((code, plot))
    def feature_plot(self):
        self.feature_data.sort(key=lambda x:x.fdr)
        from dtk.plot import boxplot_stack
        plot = boxplot_stack(
                [(x.kt_scores,x.label) for x in self.feature_data],
                description='<br>'.join(['Calculates a z-normalized score of KTs',
                ' relative to all data points and plots the distribution ',
                ' boxes are ordered by an adjusted z test metric between KTs and unknown'])
                )
        self.plotly_plots.append(('boxplot', plot))

    def js_plot(self):
        from dtk.plot import barplot_stack
        '''Plots JS Divergence of components'''
        self.feature_data.sort(key=lambda x:x.fdr)
        plot = barplot_stack(
                [(x.js_scores,x.label) for x in self.feature_data],
                description='<br>'.join(['Calculates the Jensen-Shannon Score for KT and',
                'Unknown density distributions for each feature.',
                'Higher scores indicate similar histogram distrubutions',
                'between the two while lower scores indicate different distributions.'])
                )
        self.plotly_plots.append(('sepplot', plot))

# GwasSearchView has quite a complex flow.
#
# For individual datasets, there are two operations: select and reject.
# Select happens immediately, so the select ('+') button causes a POST.
# The '-' button is just a link back to this view, with the key in the
# 'rej' query parm.  That causes the view to re-render with a popup,
# and the Save button in that popup is the real 'reject' button.  The
# popup cancel button is just a link back to the view with the 'rej'
# field cleared.
#
# The same popup is used to allow editing of the note text with a similar
# flow.  The pencil icon is a link setting the 'edit' queryparm, which
# again renders the popup, but now the Save button is the 'edit' button.
#
# All the popup handlers update the GwasDataset object as specified, and
# then redirect back to the view, with the appropriate section open to
# show the action that just occurred.
#
# There's a parallel structure for PMID filtering; the 'X' button sets
# the filt_pmid queryparm, which renders the popup with 'filt_pmid' as
# the Save button. The listed filters have a filt_clear button ('<'), and a
# pencil link that sets the filt_edit queryparm, rendering a popup
# with 'filt_clear' as the Save button.
#
# In both paths, once a GwasDataset or GwasFilter record is created,
# it may be disabled, but never deleted.  So, any attached notes are
# in theory recoverable (although it may not be easy via the UI).
class GwasSearchView(DumaView):
    template_name='browse/gwas_search.html'
    GET_parms = {
            'search':(str,''),
            'show':(list_of(str),[]),
            'sort':(SortHandler,'0'),
            'rej':(str,None),
            'edit':(str,None),
            'filt_pmid':(str,None),
            'filt_edit':(str,None),
            }
    button_map={
            'search':['search'],
            'select':[],
            'reject':['note'],
            'edit':['note'],
            'filt_pmid':['note'],
            'filt_edit':['note'],
            'filt_clear':[],
            'regen':[],
            }
    def custom_setup(self):
        from dtk.gwas import GwasSearchFilter
        self.base_qparms.pop('rej',None)
        if self.rej:
            self.set_popup_dataset(self.rej)
            self.pop_headline='Reject phenotype'
            self.pop_button='reject'
        self.base_qparms.pop('edit',None)
        if self.edit:
            self.set_popup_dataset(self.edit)
            self.pop_headline='Edit phenotype note'
            self.pop_button='edit'
        self.base_qparms.pop('filt_pmid',None)
        if self.filt_pmid:
            self.set_popup_pmid(self.filt_pmid)
            self.pop_headline='Exclude entire article'
            self.pop_button='filt_pmid'
        self.base_qparms.pop('filt_edit',None)
        if self.filt_edit:
            self.set_popup_pmid(self.filt_edit)
            self.pop_headline='Edit article note'
            self.pop_button='filt_edit'
        self.ds_filter = GwasSearchFilter(self.ws)
    def custom_context(self):
        self.context['show']=self.show
        older_dates = [x[1] for x in self.ds_filter.ood_info()]
        from dtk.text import fmt_time
        self.context.update(dict(
                ood_count=len(older_dates),
                ood_oldest=fmt_time(min(older_dates)) if older_dates else None,
                ))
        from dtk.table import Table
        from dtk.url import pubmed_url
        from dtk.html import link
        from dtk.duma_view import qstr
        def sort_url(**kwargs):
            kwargs['show'] = 'search_results'
            return self.here_url(**kwargs)
        for key,rows in (
                ('selection_table',self.ds_filter.selects),
                ('rejection_table',self.ds_filter.rejects),
                ):
            if key == 'selection_table':
                for r in rows:
                    r.load_study_data()
                self.context[key] = Table(rows,[
                Table.Column('Phenotype',
                    ),
                Table.Column('GWDS ID',
                    code='id',
                    ),
                Table.Column('Pubmed Id',
                    cell_fmt=lambda x:link(x,pubmed_url(x),new_tab=True),
                    ),
                Table.Column('Total samples',
                    code='total_samples'
                    ),
                Table.Column('# Variants',
                    code='num_variants'
                    ),
                Table.Column('# Prots',
                    code='num_prots'
                    ),
                Table.Column('Ancestry',
                    code='ancestry'
                    ),
                Table.Column('Chip type [SNPs]',
                    code='chip_type'
                    ),
                Table.Column('Pub. Date',
                    code='pub_date'
                    ),
                Table.Column('GWAS QC plots',
                            code = 'id',
                            cell_fmt=lambda x:link('GWAS QC',
                                      self.ws.reverse('gwas_qc')+qstr({'id':x}),
                                      new_tab=True
                                     ),
                    ),
                ])
            else:
                self.context[key] = Table(rows,[
                            Table.Column('Phenotype',
                            ),
                        Table.Column('Pubmed Id',
                            cell_fmt=lambda x:link(x,pubmed_url(x),new_tab=True),
                            )
                        ])
        self.context['pmid_filt_table'] = Table(self.ds_filter.filters,[
                    Table.Column('Pubmed Id',
                        cell_fmt=lambda x:link(x,pubmed_url(x),new_tab=True),
                        ),
                    ])
        # retrieve and convert matched data
        self.load_search_matches()
        self.context['table'] = build_gwds_table(self.matches,
                                                 sort_url,
                                                 self.sort,
                                                )
    def set_popup_dataset(self,key):
        self.pop_key = key
        self.pop_phenotype,self.pop_pmid = key.split('|')
        from browse.models import GwasDataset
        qs = GwasDataset.objects.filter(
                ws=self.ws,
                phenotype=self.pop_phenotype,
                pubmed_id=self.pop_pmid,
                )
        self.pop_dataset = qs[0] if qs.exists() else None
    def set_popup_pmid(self,key):
        self.pop_key = key
        self.pop_pmid = key
        self.pop_phenotype = ''
        from browse.models import GwasFilter
        qs = GwasFilter.objects.filter(
                ws=self.ws,
                pubmed_id=self.pop_pmid,
                )
        self.pop_filt_rec = qs[0] if qs.exists() else None
    def load_search_matches(self):
        from dtk.gwas import search_gwas_studies_file
        if not self.search:
            self.matches = []
            return

        parts = self.search.split('"')
        # Even-numbered parts are unquoted, odd-numbered are quoted.
        # (assuming non-nested quotes)

        # Take quoted verbatim
        terms = [part for i, part in enumerate(parts) if i % 2 == 1]
        # Add everything else space-separated
        [terms.extend(x for x in part.split(' ') if x)
         for i, part in enumerate(parts) if i % 2 == 0]
        logger.info("Searching gwas for %s", terms)


        self.matches = search_gwas_studies_file(
                                        self.ws,
                                        terms,
                                        ds_filter=self.ds_filter,
                                        )
        # sort.colspec is just the integer record index as a string;
        # this is set below using the 'code' parameter to Table.Column;
        # since not all dates are converted, we need a compare function
        # that will sort columns with mixed types
        from dtk.data import TypesafeKey
        self.matches.sort(
                key=lambda x:TypesafeKey(x[int(self.sort.colspec)]),
                reverse=self.sort.minus,
                )
    def make_note_form(self,data):
        try:
            initial=self.pop_dataset.get_note_text()
        except AttributeError:
            try:
                initial=self.pop_filt_rec.get_note_text()
            except AttributeError:
                initial = None
        class MyForm(forms.Form):
            note = forms.CharField(
                    widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
                    required=False,
                    initial=initial,
                    )
        return MyForm(data)
    def make_search_form(self,data):
        class MyForm(forms.Form):
            terms = forms.CharField(
                    label='Search Terms'
                 )
        return MyForm(data)
    def search_post_valid(self):
        p = self.context['search_form'].cleaned_data
        terms = p['terms'].strip().lower()
        return HttpResponseRedirect(self.here_url(
                            search=terms,
                            show='search_results',
                            ))
    def select_post_valid(self):
        self.update_dataset(self.request.POST['key'],False)
        return HttpResponseRedirect(self.here_url(show='selected'))
    def reject_post_valid(self):
        self.update_dataset(
                self.request.POST['key'],
                True,
                note=self.note_form.cleaned_data['note'],
                )
        return HttpResponseRedirect(self.here_url(show='rejected'))
    def edit_post_valid(self):
        self.update_dataset(
                self.request.POST['key'],
                None,
                note=self.note_form.cleaned_data['note'],
                )
        return HttpResponseRedirect(self.here_url())
    def update_dataset(self,key,reject,note=None):
        phenotype,pubmed_id = key.split('|')
        from browse.models import GwasDataset
        gwds,new = GwasDataset.objects.get_or_create(
                ws=self.ws,
                phenotype=phenotype,
                pubmed_id=pubmed_id,
                )
        if note is not None:
            Note.set(gwds,'note',self.request.user.username,note)
        if new:
            assert reject is not None
            if reject:
                gwds.rejected = reject
                gwds.save()
                # we don't need to delete
            else:
                # flag is already correct, but we need to extract
                gwds.extract_data()
        else:
            if reject is not None:
                if reject != gwds.rejected:
                    gwds.rejected = reject
                    gwds.save()
                    if reject:
                        gwds.delete_data()
                    else:
                        gwds.extract_data()
    def filt_pmid_post_valid(self):
        self.update_pmid(
                int(self.request.POST['key']),
                False,
                note=self.note_form.cleaned_data['note'],
                )
        return HttpResponseRedirect(self.here_url(show='pmid_filt'))
    def filt_edit_post_valid(self):
        self.update_pmid(
                int(self.request.POST['key']),
                None,
                note=self.note_form.cleaned_data['note'],
                )
        return HttpResponseRedirect(self.here_url(show='pmid_filt'))
    def filt_clear_post_valid(self):
        self.update_pmid(
                int(self.request.POST['key']),
                True,
                note=None
                )
        return HttpResponseRedirect(self.here_url(show='pmid_filt'))
    def regen_post_valid(self):
        from dtk.files import remove_if_present
        for gwds,date in self.ds_filter.ood_info():
            remove_if_present(gwds.make_path())
        return HttpResponseRedirect(self.here_url())
    def update_pmid(self,key,reject,note=None):
        pubmed_id = int(self.request.POST['key'])
        from browse.models import GwasFilter
        frec,new = GwasFilter.objects.get_or_create(
                ws=self.ws,
                pubmed_id=pubmed_id,
                )
        if note is not None:
            Note.set(frec,'note',self.request.user.username,note)
        if reject is not None and reject != frec.rejected:
            frec.rejected = reject
            frec.save()

def build_gwds_table(matches, url, sorter, full_detail=False):
    from dtk.table import Table
    from dtk.html import link
    from dtk.url import pubmed_url

    remaining_columns = ["Total samples","Ancestry","SNPs","Pub Date"]
    if full_detail:
        remaining_columns += ["Includes Male/Female Only Analyses",
                              "Exclusively Male/Female",
                              "European Discovery Samples",
                              "African Discovery Samples",
                              "East Asian Discovery Samples",
                              "European Replication Samples",
                              "African Replication Samples",
                              "East Asian Replication Samples"
                             ]
    return Table(matches,[
            Table.Column('Phenotype',
                    idx=0,
                    code='0',
                    sort='l2h',
                    ),
            Table.Column('Pubmed Id',
                    idx=1,
                    code='1',
                    sort='l2h',
                    cell_fmt=lambda x:link(x,pubmed_url(x),new_tab=True),
                    ),
            Table.Column('Ptyps Used',
                    idx=2,
                    code='2',
                    sort='l2h',
                    ),
            Table.Column('# Variants',
                    idx=3,
                    code='3',
                    sort='l2h',
                    ),
            Table.Column('# Prots',
                    idx=4,
                    code='4',
                    sort='l2h',
                    ),
            ]+[
            Table.Column(label,idx=i,code=str(i),sort='l2h')
            for i,label in enumerate(
                    remaining_columns,
                    start=6,
                    )
            ],
            url_builder=url,
            sort_handler=sorter,
            )

class GwasQcView(DumaView):
    template_name='browse/gwas_qc.html'
    GET_parms = {
            'id':(str, None),
            'sort':(SortHandler,'0'),
            }
    button_map={
            'save':['thresholds'],
            }
    def custom_setup(self):
        self._load_data()
    def custom_context(self):
        self.plotly_plots = []
        self._plot()
        self.context['table'] = build_gwds_table(self.study_data,
                                                 self.here_url,
                                                 self.sort,
                                                 full_detail=True
                                                )
        self.context['plotly_plots'] = self.plotly_plots
        self.context['png_plots'] = self.png_plots
        self.context['snp_table'] = self._build_snp_table()
    def make_thresholds_form(self,data):
        class MyForm(forms.Form):
            v2d = forms.FloatField(
                    label='Variant to Disease Threshold',
                    required=True,
                    initial=self.gwds.v2d_threshold,
                    help_text='gPath will only use SNPs more significant than this threshold',
                    )
        return MyForm(data)
    def save_post_valid(self):
        p = self.thresholds_form.cleaned_data
        self.gwds.v2d_threshold = p['v2d']
        self.gwds.save()
        return HttpResponseRedirect(self.here_url())
    def _build_snp_table(self):
        from dtk.table import Table
        from browse.models import Protein
        from dtk.html import link
        from django.utils.safestring import mark_safe
        from dtk.url import dbsnp_url, otarg_genetics_url
        prots = [x.uniprot for x in self.full_snps]
        prot_objs = Protein.objects.filter(uniprot__in=prots)
        u2g = {p.uniprot: p.get_html_url(self.ws.id) for p in prot_objs}

        if len(self.full_snps) > 100000:
            # Chrome seems to get unhappy if this gets well beyond 100k... which is very rare, but we have a single
            # CKD dataset that hits this currently.
            self.message("Too many SNPs to display table, try browsing the raw gwds text file.  Table will only show a subset.")
            self.full_snps = self.full_snps[:100000]
        
        from dtk.gwas import lookup_otarg_alleles
        fmt_id = lambda x: f'{x.chromosome}:{x.base}'
        chr_and_pos_list = [fmt_id(x) for x in self.full_snps]
        otarg_alleles = lookup_otarg_alleles(chr_and_pos_list)
        def otarg_cell(data):
            id = fmt_id(data)
            links = []
            for entry in otarg_alleles.get(id, []):
                ref, alt = entry[1], entry[2]
                url = otarg_genetics_url(data.chromosome, data.base, ref, alt)
                links.append(link(f'{ref}/{alt}', url))
            
            return mark_safe(' '.join(links))
        
        def rsids_links(rsids):
            return ', '.join([link(x, dbsnp_url(x)) for x in rsids.split(',')])

        cols = [
            Table.Column('rsid', code='snp', cell_fmt=rsids_links),
            Table.Column('OpenTargets', extract=otarg_cell),
            Table.Column('chromosome', code='chromosome'),
            Table.Column('chr position', code='base'),
            Table.Column('v2d evidence', code='evidence'),
            Table.Column('v2g evidence', code='v2g_evidence'),
            Table.Column('gene', code='uniprot', cell_fmt=lambda x: u2g.get(x, x)),
        ]
        table = Table(self.full_snps, cols)
        return table
    def _plot(self):
        self.plot_prefix = self.gwds.make_plot_path_prefix()
        self._plot_hists()
        self._manhattan()
        self._qqplot()
    def _get_output_fn(self, desc, suffix='plotly'):
        fn = os.path.join(self.plot_prefix+'_'+desc+'.'+suffix)
        # now that we can refresh GWAS datasets, we need to make sure
        # that the dataset isn't newer than any cached plot files;
        # treat a missing dataset file as always newer (it will be
        # newer after it's regenerated)
        def mod_date(fn):
            from dtk.files import modification_date
            try:
                return modification_date(fn)
            except OSError:
                return None
        base_date = mod_date(self.gwds.make_path())
        fn_date = mod_date(fn)
        if fn_date and (not base_date or fn_date < base_date):
            os.remove(fn)
        return fn
    def _qqplot(self):
        qq_png = self._get_output_fn('qqplot','png')
        if not os.path.isfile(qq_png):
            import statsmodels.api as sm
            from matplotlib import pyplot as plt
            import numpy as np
            fig = sm.qqplot(np.array([max(l)
                                      for l
                                      in self.snpscores.values()
                                    ]),
                            line='45'
                           )
            plt.title('QQ-plot')
            fig.savefig(qq_png)
        self.png_plots=[PathHelper.url_of_file(qq_png)]
    def _manhattan(self):
        from dtk.plot import PlotlyPlot
        fn = self._get_output_fn('manhattan')
        if os.path.isfile(fn):
            manhat = PlotlyPlot.build_from_file(fn, thumbnail=True)
        else:
            self._setup_manhat()
            manhat = PlotlyPlot([{
                'x':self.manhat_x,
                'y':self.manhat_y,
                "marker": {
                    "color": self.manhat_cols,
                    "size": self.manhat_sizes,
                    "symbol": "circle"
                },
                "mode": "markers",
                "name": "",
                "showlegend": False,
                "text": self.manhat_text,
                "type": "scatter"
            }],
            {
                'title': 'Manhattan plot',
                'width': 1200,
                'yaxis':{'title':'max(-Log10(p))',
                         'range': [0,max(self.manhat_y, default=0)]
                        },
                'xaxis':{'title':'Chromosome',
                         'showgrid': False,
                         'showticklabels': False,
                         'zeroline': False
                        }
            })
            manhat.save(fn,thumbnail=True)
        self.plotly_plots.append(('manhattan', manhat))
    def _setup_manhat(self):
        from dtk.hgChrms import linear_hgChrms
        from browse.default_settings import ucsc_hg
        chrms = linear_hgChrms(ucsc_hg.value(self.ws))
        self.manhat_x = []
        self.manhat_y = []
        self.manhat_cols = []
        self.manhat_sizes = []
        self.manhat_text = []
        for snp,tup in self.positions.items():
            self.manhat_x.append(chrms.get_linear_pos(tup))
            self.manhat_y.append(max(self.snpscores[snp]))
            try:
                col = '#1A84C6' if int(tup[0]) % 2 == 0 else '#F48725'
            except  ValueError:
                col = '#1A84C6' if chrms.get_chrm_index(tup[0]) % 2 == 0 else '#F48725'
            prots = list(self.prots_per_snp[snp])
            self.manhat_sizes.append(5 if prots[0]=='-' else 9)
            self.manhat_cols.append(col)
            self.manhat_text.append("<br>".join(['chr'+tup[0]+':'+tup[1]] +
                                                prots
                                                )
                                   )
    def _plot_hists(self):
        self._get_subtitles()
        self._hist('score', self.snpscores, 'snp_pscores',
                   'SNP', self.subtitle[0]
        )
        self._hist('score', self.prot_pscores, 'prot_pscores',
                   'Protein', self.subtitle[1]
        )
        self._hist('count', self.snps_per_prot,
                   'snps_per_prot', 'SNP', self.subtitle[2], 'Proteins'
        )
        self._hist('count', self.prots_per_snp,
                   'prots_per_snp', 'Protein', self.subtitle[3], 'SNPs'
        )
    def _get_subtitles(self):
        self.subtitle=[
                       ['This histogram shows the scores associated '+
                        'with each SNP. It does include SNPs that are',
                        'not associated with a gene, and thus will not '+
                        'be included for scoring.'
                       ],
                       ['This histogram shows the scores associated '+
                        'with each protein. Thus, these are the scores',
                        'used for CM scoring.'
                       ],
                       ['It is easiest to read this histogram from the '+
                        'Y-axis as it plots the number of SNPs connected',
                        'to each protein.'
                       ],
                       ['Similar to the previous plot, but this histogram '+
                        'shows the number of proteins associated with a SNP.',
                        'Any values above 1 come from our translation of '+
                        'genes to UniProt IDs.'
                       ]
                      ]
    def _hist(self, type, d, s, t, subtitle = '', t2=''):
        from dtk.plot import PlotlyPlot
        fn = self._get_output_fn(s+'Hist')
        if os.path.isfile(fn):
            phist = PlotlyPlot.build_from_file(fn, thumbnail=True)
        else:
            from math import log
            n = len(list(d.keys()))
            if type == 'count':
                vals = [len(l) for l in d.values()]
                title = 'Number of '+t+' per '+t2
                xtitle = 'Number of '+t
                ytitle ='Number of '+t2+' (total = %d)' % n
            elif type == 'score':
                vals = [max(l) for l in d.values()]
                xtitle = 'max(-Log10(p))'
                ytitle ='Number of '+t+' (total = %d)' % n
                title = 'Histogram of scores for each '+t
            if not vals:
                return
            xlim = max(vals) * 1.1
            phist = PlotlyPlot([
                dict(
                      type='histogram',
                      x=vals
                     ,autobinx=False
                     ,xbins={'start':0.0,
                             'end':xlim,
                             'size':xlim/50
                      }
                )],
                {'title':title,
                 'xaxis':{'title':xtitle,
                          'range':[0,xlim]
                         },
                 'yaxis':{
                   'title':ytitle}
                }
               )
            phist._layout['annotations'] = [{
                    'xref' : 'paper'
                    , 'yref' : 'paper'
                    , 'showarrow' : False
                    , 'y' : -0.2
                    , 'yanchor' : 'top'
                    , 'x' : 0.5
                    , 'xanchor' : 'center'
                    , 'text' : '<br>'.join(subtitle)
                    }]
            phist._layout['margin']=dict(
                              l=60,
                              r=30,
                              b=120,
                              t=30,
                              pad=4
                              )
            phist.save(fn,thumbnail=True)
        self.plotly_plots.append((s, phist))
    def _load_data(self):
        self._load_snp_data()
        self._load_study_data()
    def _load_study_data(self):
        _=self.gwds.check_study_data()
        self.study_data = self.gwds.matches
    def _load_snp_data(self):
        self.positions = {}
        self.snpscores = {}
        self.prot_pscores = {}
        self.snps_per_prot = {}
        self.prots_per_snp = {}
        from browse.models import GwasDataset
        self.gwds = GwasDataset.objects.get(pk=self.id)
        from dtk.gwas import score_snp
        self.full_snps = []
        for rec in self.gwds.get_data():
            self.study_key=rec.study_key
            self.positions[rec.snp] = (rec.chromosome, rec.base)
            s = score_snp(rec.evidence)
            if rec.snp not in self.snpscores:
                self.snpscores[rec.snp] = []
                self.prots_per_snp[rec.snp] = set()
            self.snpscores[rec.snp].append(s)
            self.prots_per_snp[rec.snp].add(rec.uniprot)
            if rec.uniprot == '-':
                continue
            if rec.uniprot not in self.prot_pscores:
                self.prot_pscores[rec.uniprot] = []
                self.snps_per_prot[rec.uniprot] = set()
            self.prot_pscores[rec.uniprot].append(s)
            self.snps_per_prot[rec.uniprot].add(rec.snp)
            self.full_snps.append(rec)
        

# XXX The next three views are variations on using dtk.composer to assemble
# XXX and present drug information. Possibly these belong in rvw (with
# XXX AllReviewNotesView)

class ReviewNotesView(DumaView):
    template_name='browse/review_notes.html'
    def custom_context(self):
        from browse.models import Vote
        user = self.request.user.username
        drugs = {}
        from dtk.composer import DrugNoteCollection
        for vote in Vote.objects.filter(
                    drug__ws=self.ws,
                    reviewer=user,
                    disabled=False,
                    ).order_by(
                    'drug__agent',
                    '-election__due',
                    ):
            try:
                d=drugs[vote.drug_id]
            except KeyError:
                d=DrugNoteCollection(vote.drug,self.is_demo())
                drugs[vote.drug_id]=d
            d.add_vote(vote,self.request.user.username)
        from browse.models import WsAnnotation
        wsa_ids = list(drugs.keys())
        wsa_lookup = {
                x.id:x
                for x in WsAnnotation.objects.filter(pk__in=wsa_ids)
                }
        from dtk.prot_map import AgentTargetCache
        atc = AgentTargetCache.atc_for_wsas(
                list(wsa_lookup.values()),
                ws=self.ws,
                )
        prot_notes = atc.build_note_cache(self.ws,user)
        for wsa_id,d in drugs.items():
            wsa = wsa_lookup[wsa_id]
            prot_info = atc.full_info_for_agent(wsa.agent_id)
            prot_info.sort(key=lambda x:x[2])
            for _,uniprot,gene,_,_ in prot_info:
                try:
                    note_info = prot_notes[uniprot][user]
                except KeyError:
                    pass
                else:
                    d.add_note(f'{gene} ({uniprot})',note_info[1])
        druglist = list(drugs.values())
        druglist.sort(key=lambda x:x.name)
        self.context['druglist'] = druglist
        self.context['page_label'] = 'My Review Notes'

class PatentNotesView(DumaView):
    template_name='browse/review_notes.html'
    GET_parms = {
            'wsa_list':(list_of(int), []),
            'recompute':(boolean, False),
            'appendix':(boolean, False),
            }
    
    def add_prot_notes(self, uniprot, gene, dnc):
        from browse.models import TargetAnnotation
        note_cache = TargetAnnotation.batch_note_lookup(
                self.ws,
                [uniprot],
                ''
                )
        title = f'Target Notes: {gene} ({uniprot})'
        this_prot_notes = note_cache.get(uniprot,{})
        combined_text = ''
        for user,(note_id,text) in sorted(this_prot_notes.items()):
            combined_text += f'<u>{user} ({gene})</u>: <p>{text}</p>'
        
        dnc.add_note(title, combined_text)


    def custom_context(self):
        qs = WsAnnotation.objects.filter(ws=self.ws)
        if self.wsa_list:
            qs = qs.filter(id__in=self.wsa_list)
        else:
            enum=WsAnnotation.indication_vals
            qs = qs.filter(
                dispositionaudit__ignore=False,
                dispositionaudit__indication__in=(
                    enum.CANDIDATE_PATENTED,
                    enum.PATENT_PREP,
                    enum.HIT,
                    )).distinct()
        from dtk.composer import DrugNoteCollection,EvidenceComposer,DrugNoteCache
        ec = None
        druglist = []
        cache = DrugNoteCache
        def process_wsa(wsa, rename=None):
            def compute():
                nonlocal ec
                if not ec:
                    ec = EvidenceComposer(self.ws, appendix_only=self.appendix)
                d=DrugNoteCollection(wsa,self.is_demo(),name_override=rename)
                ec.extract_evidence(wsa,d,appendix=self.appendix)
                return d
               
            # We always force_recompute on appendix because the patents often get added later,
            # and it loads pretty fast.
            d = cache.get_or_compute(
                    wsa,
                    compute,
                    appendix=self.appendix,
                    force_compute=self.recompute or self.appendix)
            if rename:
                # We have some old names cached.
                d.name = rename
            
            druglist.append(d)
            d.add_note('Study note',wsa.get_study_text())
            for vote in wsa.vote_set.all():
                d.add_vote(vote,'')
            
            from dtk.prot_map import AgentTargetCache
            atc = AgentTargetCache.atc_for_wsas([wsa], ws=self.ws)
            prot_info = atc.full_info_for_agent(wsa.agent_id)
            prot_info.sort(key=lambda x:x[2])

            for _,uniprot,gene,_,_ in prot_info:
                self.add_prot_notes(uniprot, gene, d)

            name = wsa.get_name(self.is_demo())
            for repl_wsa in wsa.replacement_for.all():
                repl_name = repl_wsa.get_name(self.is_demo())
                rename_to = f'{repl_name} (replaced by {name})'
                process_wsa(repl_wsa, rename=rename_to)

        # We want to sort here, before we've processed, because replacements
        # are going to get inserted into the output just after their replacee.
        qs = sorted(qs, key=lambda x:x.get_name(self.is_demo()))
        for wsa in qs:
            process_wsa(wsa)
        self.context['druglist'] = druglist
        self.context['page_label'] = 'Hits Notes'

class CompetitionView(DumaView):
    template_name='browse/competition.html'
    button_map={
            'analyze':['upload','factors'],
            }
    order = ['Marketed']+['Phase '+x for x in ['III','II','I']]
    def custom_setup(self):
        # empty display table if no TSV
        self.drugs = []
        # empty display competition score if no factors
        self.factor_vec = []
    def custom_context(self):
        col_list = [x[1] for x in self.colmap]
        from dtk.text import fmt_english_list
        from dtk.html import join,tag_wrap
        self.context['paste_instructions'] = join(
                tag_wrap('b','NOTE: '),
                'Pasted TSV must include the columns ',
                tag_wrap('b',fmt_english_list(col_list)),
                )
        self.context['total_mols'] = len(self.drugs)
        self.context['unique_mols'] = len(set(x.drug_name for x in self.drugs))
        # accumulate unique targets
        unique_targs = set()
        for drug in self.drugs:
            unique_targs |= set(drug.targets)
        self.context['unique_targs'] = len(unique_targs)
        # construct map from targets to uniprots
        from dtk.prot_search import find_protein_for_global_data_target
        target_map = {}
        for t in unique_targs:
            p = find_protein_for_global_data_target(t)
            if p:
                target_map[t] = p.gene
            else:
                target_map[t] = t
        # roll up data to phase level
        phase_dict = {}
        for drug in self.drugs:
            l = phase_dict.setdefault(drug.stage,[set(),0,set(),set(),{}])
            l[0].add(drug.drug_name)
            if drug.targets:
                mapped_targets = [target_map[t] for t in drug.targets]
                canonical_moa = '; '.join(sorted(mapped_targets))
                l[2].add(canonical_moa) # moas
                l[3] |= set(mapped_targets) # unique targets
            else:
                l[1] += 1 # missing moas
            # collapse ROAs by drugname so counts are more meaningful
            s = l[4].setdefault(drug.drug_name,set())
            if drug.roas:
                s |= set(drug.roas)
            else:
                s.add('missing')
        # calculate competition scores
        if self.factor_vec:
            for idx,score_name in (
                    (0,'comp_drug_score'),
                    (3,'comp_targ_score'),
                    ):
                seen = set()
                value_vec = []
                for name in self.order:
                    try:
                        s = phase_dict[name][idx]
                    except KeyError:
                        s = set()
                    value_vec.append(len(s - seen))
                    seen |= s
                self.context[score_name] = sum([
                        f*v
                        for f,v in zip(
                                self.factor_vec,
                                value_vec,
                                )
                        ])
        # Collapse dictionary into a list, so we can pre-sort it.
        # At the same time, convert the per-drugname ROAs dict to a
        # list containing all ROAs with one repetition per drugname
        # the ROA appears in.
        from itertools import chain
        phase_list = [
                (k,)+tuple(l[:4])+(list(chain(*l[4].values())),)
                for k,l in phase_dict.items()
                ]
        def phase_key(phase):
            if phase in self.order:
                return (self.order.index(phase),phase)
            # report any unexpected phases at the end in alpha order
            return (len(self.order),phase)
        phase_list.sort(key=lambda x:phase_key(x[0]))
        # build table
        def hover_list(s):
            from dtk.html import join, glyph_icon
            return join(
                    str(len(s)),
                    glyph_icon(
                            'info-sign',
                            html=True,
                            hover='<br>'.join(sorted(s)),
                            ),
                    )
        def counted_list(l):
            from collections import Counter
            ctr=Counter(l)
            return '; '.join(f'{k}:{v}' for k,v in ctr.most_common())
        from dtk.table import Table
        self.context['phase_table'] = Table(phase_list,[
                Table.Column('Phase',
                        idx=0,
                        ),
                Table.Column('Unique Molecules',
                        idx=1,
                        cell_fmt=hover_list,
                        ),
                Table.Column('Missing MOAs',
                        idx=2,
                        ),
                Table.Column('Distinct MOAs',
                        idx=3,
                        cell_fmt=hover_list,
                        ),
                Table.Column('Unique Targets',
                        idx=4,
                        cell_fmt=hover_list,
                        ),
                Table.Column('Routes of Administration',
                        idx=5,
                        cell_fmt=counted_list,
                        ),
                ])
    def make_upload_form(self,data):
        class MyForm(forms.Form):
            text = forms.CharField(
                    label='Pasted TSV',
                    widget=forms.Textarea(attrs={'rows':'6','cols':'120'}),
                    strip=False,
                    )
        return MyForm(data)
    def make_factors_form(self,data):
        from dtk.dynaform import FormFactory
        ff = FormFactory()
        import re
        for name in self.order:
            m = re.match('Phase (I+)$',name)
            if not m:
                continue
            f = forms.FloatField(
                    required=False,
                    label=name+' to Market fraction',
                    )
            ff.add_field(m.group(1),f)
        MyForm = ff.get_form_class()
        return MyForm(data)
    cvt_list = lambda x:[y.strip() for y in x.split(';')] if x else []
    colmap = [
            ('drug_name','Drug Name',None),
            ('company','Company Name',None),
            ('stage','Development Stage',None),
            ('targets','Target',cvt_list),
            ('roas','Route of Administration',cvt_list),
            ]
    def analyze_post_valid(self):
        p = self.context['upload_form'].cleaned_data
        text_records = [
                x.rstrip('\r').split('\t')
                for x in p['text'].split('\n')
                ]
        # strip blank lines from end
        while text_records[-1] == ['']:
            text_records.pop(-1)
        from dtk.readtext import convert_records_using_colmap
        self.drugs = list(convert_records_using_colmap(
                iter(text_records),
                self.colmap,
                ))
        ff = self.context['factors_form']
        p = ff.cleaned_data
        fvec = [p[f.name] for f in ff]
        if None not in fvec:
            self.factor_vec = [1]+fvec

class CompEvidenceView(DumaView):
    template_name='browse/comp_evidence.html'
    GET_parms = {
            'wsa':(int, None),
            'wzs_jid':(int, None),
            'orig_wsa':(list_of(int),[]),
            'orig_wzs_jid':(list_of(int), []),
            'recompute':(boolean, False),
            'appendix':(boolean, False),
            }
    def custom_context(self):
        qs = WsAnnotation.objects.filter(ws=self.ws)
        wsa = qs.get(id=self.wsa)
        self.context['drug'] = self._get_dnc(wsa, self.wzs_jid)
        if self.orig_wsa:
            orig_wsas = qs.filter(id__in=self.orig_wsa)
            orig = [self._get_dnc(orig_wsa, orig_jid) for orig_wsa, orig_jid in zip(orig_wsas, self.orig_wzs_jid)]
        else:
            orig = []

        self.context['wsa'] = wsa
        self.context['origs'] = orig
        self.context['page_label'] = 'Computational Evidence'

    def _get_dnc(self, wsa, wzs_jid):
        from dtk.composer import DrugNoteCollection, EvidenceComposer, DrugNoteCache
        cache = DrugNoteCache
        def compute():
            ec = EvidenceComposer(self.ws, appendix_only=self.appendix)
            dnc=DrugNoteCollection(wsa,self.is_demo())
            ec.extract_evidence(wsa,dnc,wzs_jid=self.wzs_jid,appendix=self.appendix)
            return dnc
        d = cache.get_or_compute(
                wsa,
                compute,
                wzs_jid=wzs_jid,
                force_compute=self.recompute or self.appendix,
                appendix=self.appendix,
                )
        return d


class DataStatusView(DumaView):
    template_name='browse/data_status.html'
    GET_parms = {
            'to_ignore':(list_of(int), [])
            }
    def custom_context(self):
        from dtk.ws_data_status import get_score_status_types, DataStatus
        self.raw_scores = dict()
        self.norm_scores = dict()

        types = get_score_status_types()
        for typename, scoreattrname, weightname in types:
            Class = DataStatus.lookup(typename)
            inst = Class(self.ws)
            for message in inst.messages:
                self.message(message)
            scorename = getattr(inst, scoreattrname)
            norm_score = inst.scores()[scorename]
            self.norm_scores[scorename] = norm_score
            self.raw_scores[scorename] = inst.raw_scores[scorename]
        self._get_eval_drugs()
        self.zippedList = [('Evaluation drugs',
                             self.eval_cnt,
                             True
                          )]
        self.zippedList += [('Available targets',
                            round(self.raw_scores['Available Targets (Raw %)'], 2),
                            True
                           )]
        self.plot_order = [
                           'Case/Control'
                          ,'miRNA'
                          ,'GWAS Datasets'
                          ,'DisGeNET Values'
                          ,'AGR Values'
                          ,'Tumor Mutation Sigs'
                          ,'Integrated Target Data'
                          ,'Complete Clinical Values'
                          ,'Monarch Initiative Values'
                          ]
        for k in self.plot_order:
            self._update_zipd_list(k)
        self.context['sample_sorter'] = self.zippedList
        self._plot_gauges()

    def _update_zipd_list(self, key, alt_text='0'):
        text = self.raw_scores.get(key, alt_text)
        self.zippedList += [(key, text, True)]

    def _get_eval_drugs(self):
        from stages import IndicationsStage
        self.ind = IndicationsStage(self.ws)
        e_ds = self.ws.eval_drugset
        eds_to_set_name = {
                           'kts': 'Known Treatments',
                           'p3ts': 'Phase 3+',
                           'p2ts': 'Phase 2+',
                           'tts': 'Phase 1+'
                          }
        try:
            self.eval_label = eds_to_set_name[e_ds]
            self.eval_cnt = self.ind._quality_detail[self.eval_label]
        except KeyError:
            self.eval_label = e_ds
            self.eval_cnt = len(self.ws.get_wsa_id_set(e_ds))
    def _plot_gauges(self):
        data_to_plot = []
        plot_labels = []
        for k in reversed(self.plot_order):
            val = self.norm_scores.get(k,0)
            if str(val)=='nan':
                continue
            data_to_plot.append(val)
            plot_labels.append(k)
        self.context['gauge'] = ('g', self.make_gauges(data_to_plot, plot_labels
                                                       )                                 )
    def make_gauges(self, data, labels):
        from dtk.plot import scatter2d, PlotlyPlot, annotations
        cnt = len(data)
        scale_lim=10
# setup the gauges
        traces = [dict(
                    y=[.3],
                    x=[0],
                    text="<b>" + s + "  </b><br>",
                    textposition="left",
                    mode='text',
                    hoverinfo='none',
                    showlegend=False,
                    xaxis='x'+str(i+1),
                    yaxis='y'+str(i+1)
                  )
                  for i,s in enumerate(labels)
                 ]
        shapes = []
        chart_width = 600
        chart_height = 60*cnt+180
        max_color_val = 210 # smaller is darker
        color_scale_steps = max_color_val*2
        max_y = 1
        min_y = 0
        oor_adj = 2.8
# build the color range for each gauge
        for i in range(cnt):
            xref='x'+str(i+1)
            yref='y'+str(i+1)
            for j in range(color_scale_steps):
                x0 = j*((scale_lim+oor_adj)/float(color_scale_steps)) - oor_adj
                if x0 < 0:
                    continue
                x1 = (j)*((scale_lim+oor_adj)/float(color_scale_steps))- oor_adj
                color = 'rgb(%i, %i, 0)' % (max_color_val-(j-max_color_val)
                                                            if j > max_color_val
                                                            else max_color_val,
                                                            j if j <= max_color_val
                                                            else max_color_val
                                                           )
                shapes.append({'type': 'rect',
                           'y0': min_y, 'y1': max_y,
                           'x0': x0,
                           'x1': x1,
                           'xref':xref, 'yref':yref,
                           'line': {'color': color},
                           'fillcolor': color
                })
            color = 'rgb(%i, %i, 0)' % (max_color_val, max_color_val/4)
            shapes.append({'type': 'rect',
                       'y0': min_y, 'y1': max_y,
                       'x0': 0,
                       'x1': 1,
                       'xref':xref, 'yref':yref,
                       'line': {'color': color},
                       'fillcolor': color
                })
# add actual data bars
            if data[i] is None:
                continue
            y_adjust = 0.12
            y_above = 0.08
            inner_width = 0.008
            outer_width = 13*inner_width
            shapes.append({'type': 'rectangle',
                   'y0': min_y+y_adjust, 'y1': max_y-y_adjust,
                   'x0': data[i]-inner_width, 'x1': data[i]+inner_width,
                   'xref':xref, 'yref':yref,
                   'fillcolor': 'black'})
            shapes.append({'type': 'path',
                   'path': ' '.join(['M',
                                     str(data[i]-outer_width),
                                     str(max_y+y_above),
                                     'L',
                                     str(data[i]+outer_width),
                                     str(max_y+y_above),
                                     'L',
                                     str(data[i]),
                                     str(max_y-y_adjust),
                                     'Z'
                                   ]),
                   'xref':xref, 'yref':yref,
                   'fillcolor': 'black'})
            shapes.append({'type': 'path',
                   'path': ' '.join(['M',
                                     str(data[i]-outer_width),
                                     str(min_y-y_above),
                                     'L',
                                     str(data[i]+outer_width),
                                     str(min_y-y_above),
                                     'L',
                                     str(data[i]),
                                     str(min_y+y_adjust),
                                     'Z'
                                   ]),
                   'xref':xref, 'yref':yref,
                   'fillcolor': 'black'})
        increment = 1/float(cnt)
        x_domains = [ [i*increment,(i+1) * increment] for i in range(cnt)]
        yaxes = []
        for i in range(cnt):
            yaxes.append({'domain': x_domains[i], 'range':[-0.1, 1.4],
                          'showgrid': False, 'showline': False,
                          'zeroline': False, 'showticklabels': False})
        xaxes = []
        for i in range(cnt):
            xaxes.append({'anchor':'y'+str(i+1), 'range':[-1*scale_lim,scale_lim+0.5],
                          'showgrid': False, 'showline': False, 'zeroline': False,
                          'ticks':'inside', 'ticklen': 0,
                          'tickvals':[0., scale_lim],
                          'ticktext':['']*2
                         })
        layout = {'shapes': shapes,
                  'autosize': False,
                  'width': chart_width,
                  'height': chart_height
                  }
        for i in range(cnt):
            layout['xaxis%i' % (i+1)] = xaxes[i]
            layout['yaxis%i' % (i+1)] = yaxes[i]
        return PlotlyPlot(traces, layout)

    ### Leaving this here, though we no longer use it
    ### If/when we want to bring the subscores back, I thought this might be helpful
    def make_alt_row(self, zl):
        i = 0
        new_list = []
        for key,rec in zl:
            if i == 0 :
                new_list.append((key,rec,True))
            else:
                new_list.append((key,rec,False))
            i+=1
        return new_list

    ### Leaving this here, though we no longer use it
    ### If/when we want to bring the subscores back, I thought this might be helpful
    def extract_data(self):
        from collections import defaultdict
        # scan master file, looking for matches to disease
        d = defaultdict(lambda: defaultdict(float))
        for key in self.keys:
            cur = self.ot.get_disease_scores(key)
            for uniprot, name_to_value in six.iteritems(cur):
                for name, value in six.iteritems(name_to_value):
                    d[uniprot][name] = max(d[uniprot][name], value)
        # find all columns with at least one non-zero score
        populated = set()
        for score in d.values():
            populated |= set(score.keys())
        # write out any matches
        score_names = [
                'overall',
                'literature',
                'rna_expression',
                'somatic_mutation',
                'genetic_association',
                'known_drug',
                'animal_model',
                'affected_pathway'
                ]
        score_name_non_zeros = [0 for i in score_names]
        score_name_zeros_or_NAN = [0 for i in score_names]
        for k,v in six.iteritems(d):
            for i,score in enumerate(score_names):
                if score in v and v[score] > 0.:
                    score_name_non_zeros[i] += 1
                else:
                    score_name_zeros_or_NAN[i] += 1
        self.rec = d
        self.score_name_non_zeros = score_name_non_zeros
        self.score_name_zeros_or_NAN = score_name_zeros_or_NAN
        score_names = [
                'Open Targets Total (%s)'%self.ot_version,
                '--Literature',
                '--RNA Expression',
                '--Somatic Mutation',
                '--Genetic Association',
                '--Known Drug',
                '--Animal Model',
                '--Affected Pathway'
                ]
        self.score_names = score_names
        self.found = len(d)



class TargetDataView(DumaView):
    template_name='browse/target_data.html'
    index_dropdown_stem='rvw:review'
    GET_parms={
            'prots':(list_of(str),[]),
            'dpi':(str,None),
            'dpi_thresh':(float,None),
            'ts':(int,None),
            'gwas_sig_t':(float,0.05),
            'excluded_tissues':(list_of(str),[]),
            'excluded_gwds':(list_of(str),[]),
            }
    button_map={
            'display':['display'],
            }
    def make_display_form(self, data):
        from browse.models import Protein
        uni2gene = Protein.get_uniprot_gene_map(self.prots)
        from dtk.prot_map import DpiMapping,PpiMapping
        class MyForm(forms.Form):
            genes = forms.CharField(
                label = 'Comma separated list of genes to plot',
                max_length = 150,
                required = True,
                initial=', '.join(sorted(uni2gene.values())),
                widget=forms.Textarea(attrs={'rows':'2','cols':'60'}),
                )
            dpi = forms.ChoiceField(
                label = 'DPI dataset',
                choices = DpiMapping.choices(self.ws),
                initial = self.ws.get_dpi_default(),
                )
            dpi_thresh = forms.FloatField(
                label = 'Min DPI evidence',
                initial = self.ws.get_dpi_thresh_default()
                )
            ts = forms.ChoiceField(
                label = 'tissueSet ID',
                choices = self.ws.get_tissue_set_choices()
                )
            excluded_tissues = forms.CharField(
                label = 'Comma separated list of any tissue IDs to exclude',
                max_length = 150,
                required = False,
                )
            gwas_sig_t = forms.FloatField(
                label = 'Significant GWAS p-value threshold',
                initial = 0.05,
                )
            excluded_gwds = forms.CharField(
                label = 'Comma separated list of any GWDS IDs to exclude',
                max_length = 150,
                required = False,
                )
        return MyForm(data)
    def display_post_valid(self):
        p = self.context['display_form'].cleaned_data
        genes = [x.strip() for x in p.pop('genes').split(',')]
        from browse.models import Protein
        gene2uni = Protein.get_gene_uniprot_map(genes)
        prots = [gene2uni.get(x) for x in genes]
        p['prots'] = ','.join([x for x in prots if x])
        return HttpResponseRedirect(self.here_url(**p))
    def custom_context(self):
        self.setup()
        if self.ts and self.prots:
            self.plotly_plots = []
            self.tables = []
            self.build_tables_and_plots()
            self.finalize_plots()
        self.context_alias(
                cy = self.cy,
                table_stats = self.summary_tables,
                plotly_plots = self.plotly_plots,
                )
    def finalize_plots(self):
        self._net_diag()
        self.summary_tables.append(('Gene expression', self.ge_table_stats.table))
        self.summary_tables.append(('GWAS', self.gwas_table_stats.table))
    def setup(self):
        if self.ts is None:
            self.ts = self.ws.get_tissue_set_choices()[0]
        if self.dpi is None:
            self.dpi = self.ws.get_dpi_default()
        if self.dpi_thresh is None:
            self.dpi_thresh = self.ws.get_dpi_thresh_default()
        self.cy = None
        self.plotly_plots = None
        self.summary_tables = []
    def load_direct_targets(self):
        self.prots = set(self.prots)
        self.n_dts = len(self.prots)
        from dtk.prot_map import DpiMapping
        dm = DpiMapping(self.dpi)
        self.all_dpi_targs = dm.get_filtered_uniq_target(min_evid=self.dpi_thresh)
    def build_tables_and_plots(self):
        if not self.prots:
            return
        self.load_direct_targets()
        self.load_indirect_targets()
        self.finalize_targets()
        self.setup_tables()
        self.preload_data()
        self.direct_data()
        self.indirect_data()
# XXX I took the easy way out and just use the WS PPI defaults
# XXX We have not been straying from those so that seems reasonable,
# XXX but it might at sometime be nice to have more flexibility.
# XXX The most obvious way to do that would be to do make the PPI a
# XXX parameter from the URL
    def load_indirect_targets(self):
        from dtk.prot_map import PpiMapping
        ppi_thresh = self.ws.get_ppi_thresh_default()
        pm = PpiMapping(self.ws.get_ppi_default())
        self.all_ppi_targs = pm.get_filtered_uniq_target(min_evid=ppi_thresh)
        self.indirect_targets = {}
        self.all_indirect_targets = set()
        all_ind = pm.get_ppi_info_for_keys(self.prots,
                                           min_evid = ppi_thresh,
                                          )
        for tup in all_ind:
            if tup[0] not in self.prots:
                continue
            if tup[0] not in self.indirect_targets:
                self.indirect_targets[tup[0]] = set()
            self.indirect_targets[tup[0]].add(tup[1])
            self.all_indirect_targets.add(tup[1])
        self.all_possible_proteins = self.all_ppi_targs | self.all_dpi_targs
    def finalize_targets(self):
        self.all_targets = self.prots | self.all_indirect_targets
        self.uni2gene = get_prot_2_gene(self.all_targets)
    def setup_tables(self):
        self.ge_table_stats = GESumStats(
                                 excluded = self.excluded_tissues,
                                 all_targets = self.all_possible_proteins
                             )
        self.ge_table_stats.setup(self.ts)
        self.gwas_table_stats = GwasSumStats(
                                 sig_t = self.gwas_sig_t,
                                 excluded = self.excluded_gwds,
                                 all_targets = self.all_possible_proteins
                                )
        self.gwas_table_stats.setup(self.ws)
    def direct_data(self):
        self.direct_expr_data()
        self.direct_gwas_data()
    def direct_expr_data(self):
        all_d = self._subset_ge_data(self.prots)
        self.me_dprots = self._get_sigProts(all_d)
        self.ge_table_stats.build_row(len(self.me_dprots),
                                      self.n_dts,
                                      'Direct targets'
                                      )
        self.plotly_plots.append(('direct',
                                   self._fc_plotly_heatmap(
                                                      all_d,
                                                      self.prots,
                                                      'Direct targets'
                                                     )
                                  ))
    def direct_gwas_data(self):
        all_d = self._subset_gwas_data(self.prots)
        self.gwas_dprots = self._get_sigProts(all_d, self.gwas_t)
        self.gwas_table_stats.build_row(len(self.gwas_dprots),
                                      self.n_dts,
                                      'Direct targets'
                                      )
    def indirect_data(self):
        self.iProts_for_net = set()
        self.gwas_iProts = set()
        for dp, iProts in self.indirect_targets.items():
            gene = self.uni2gene[dp]
            if not len(iProts):
                logger.info(" ".join(["Unable to find indirect targets for"
                                  , gene
                                  , "in the provided PPI dataset."
                                ])
                    )
            else:
                self.indirect_expr_data(iProts, gene)
                self.indirect_gwas_data(iProts, gene)
        self.ge_table_stats.build_row(len(self.iProts_for_net),
                                      len(self.all_indirect_targets),
                                      'All indirect targets'
                                      )
        self.gwas_table_stats.build_row(len(self.gwas_iProts),
                                      len(self.all_indirect_targets),
                                      'All indirect targets'
                                      )
    def indirect_expr_data(self, iProts, gene):
        all_d = self._subset_ge_data(iProts)
        me_iProts = self._get_sigProts(all_d)
        self.iProts_for_net.update(set(me_iProts))
        self.ge_table_stats.build_row(len(me_iProts), len(iProts), gene + ' partners')
        self.plotly_plots.append((gene
                            ,self._fc_plotly_heatmap(
                                all_d
                                , iProts
                                , 'Indirect targets via ' + gene
                                , width = max([600, 15 * len(iProts)])
                                )
                            ))
    def indirect_gwas_data(self, iProts, gene):
        all_d = self._subset_gwas_data(iProts)
        gwas_iProts = self._get_sigProts(all_d, self.gwas_t)
        self.gwas_iProts.update(set(gwas_iProts))
        self.gwas_table_stats.build_row(len(gwas_iProts), len(iProts), gene + ' partners')
    def _get_sigProts(self, tpd, thresh = None, min_tis = 1):
        from collections import Counter
        if thresh is None:
            sp_cnts = Counter([p
                       for t,d in six.iteritems(tpd)
                       for p,v in d.items()
                       if (abs(v[0]) >= self.tis_threshs[t][0]
                           and abs(v[1]) >= self.tis_threshs[t][1]
                          )
                      ])
        else:
            sp_cnts = Counter([p
                       for d in tpd.values()
                       for p,v in d.items()
                       if abs(v[0]) >= thresh
                      ])
        return [p for p,c in sp_cnts.items() if c >= min_tis]
    def _fc_plotly_heatmap(self, all_d, prots, title, width = 800):
        from dtk.plot import plotly_heatmap
        from browse.utils import build_dat_mat
        import numpy as np
        zdata = {}
        hover_data = {}
        for prot in prots:
            zdata[prot] = {}
            hover_data[prot] = {}
            for tid in self.tis_names:
                zdata[prot][self.tis_names[tid]] = all_d[tid][prot][1]
                hover_data[prot][self.tis_names[tid]] = all_d[tid][prot][0]
        plot_data, col_names = build_dat_mat(
                    zdata,
                    list(self.tis_names.values())
                   )
        hover_text, _ = build_dat_mat(
                    hover_data,
                    list(self.tis_names.values())
                   )
        hover_text = [['Evid: %0.3f' % (x)
                        for x in l]
                       for l in hover_text
                     ]
        genes = [self.uni2gene[c] for c in col_names]
        return plotly_heatmap(
                     np.array(plot_data)
                       , list(self.tis_names.values())
                       , Title = title
                       , color_bar_title = "Log2(Fold Change)"
                       , col_labels = genes
                       , width = width
                       , height = 50*len(self.tis_names) + 200
                       , color_zero_centered = True
                       , hover_text = np.array(hover_text)
              )
    def preload_data(self):
        self.load_ge_data()
        self.load_gwas_data()
    def load_ge_data(self):
        from browse.models import Tissue
        self.all_ge_data = {}
        self.tis_names = {}
        self.tis_threshs = {}
        qs = Tissue.objects.filter(tissue_set_id=self.ts)
        for t in qs:
            _,ev,fc,total = t.sig_result_counts()
            if (not total or str(t.pk) in self.excluded_tissues):
                continue
            self.tis_names[t.id] = str(t.id) + " - " + t.concise_name()
            self.tis_threshs[t.id] = (ev,fc)
            # we know our prots already, so set defaults
            d = {p : [0.0, 0.0] for p in self.all_targets}
            gen = (r for r in t.sig_results(over_only=False)
                   if r.uniprot in self.all_targets
                  )
            for rec in gen:
                d[rec.uniprot] = [float(rec.evidence),
                                  float(rec.fold_change) * float(rec.direction)
                                 ]
            self.all_ge_data[t.id] = d
    def _subset_ge_data(self, prots):
        return {tid:{p:self.all_ge_data[tid][p]
                     for p in prots
                    }
                    for tid in self.all_ge_data
               }
    def load_gwas_data(self):
        from dtk.gwas import gwas_codes, scored_gwas, score_snp
        self.gwas_t = score_snp(self.gwas_sig_t)
        self.all_gwas_data = {}
        for ds in gwas_codes(self.ws):
            if ds.lstrip('gwds') in self.excluded_gwds:
                continue
            gen = ((u,s) for u,s in six.iteritems(scored_gwas(ds))
                   if u in self.all_targets
                  )
            d = {p : [0.0] for p in self.all_targets}
            for tup in gen:
                d[tup[0]] = [tup[1]]
            self.all_gwas_data[ds] = d
    def _subset_gwas_data(self, prots):
        return {gwds:{p:self.all_gwas_data[gwds][p]
                     for p in prots
                    }
                    for gwds in self.all_gwas_data
               }
    def _net_diag(self):
        sig_dps = [self.uni2gene[p] for p in self.me_dprots]
        ipd = {k : s & self.iProts_for_net
               for k,s in self.indirect_targets.items()
              }
        import operator
        from functools import reduce
        import numpy as np
        try:
            iProts = reduce(operator.or_, list(ipd.values()))
            iProts -= set(self.prots)
        except TypeError:
            iProts = set()
        dp = [self.uni2gene[p] for p in self.prots]
        ipd = {self.uni2gene[k]:set([
                            self.uni2gene[x]
                            for x in v
                            if x in iProts
                            ])
               for k,v in ipd.items()
              }
        drug_name='Your drug'
        iProts = set([self.uni2gene[p] for p in iProts])
        nIPs = len(iProts)
        self.cy = WorkflowCyGraph()
        self.cy.add_prop_type('drug')
        self.cy.add_prop(drug_name,'drug')
        self.cy.add_prop_type('not_sig')
        for p in set(dp) - set(sig_dps):
            self.cy.add_prop(p, 'not_sig')
        # indirect prots I alternate y to fit more of them in
        # and thus they take up less width; x == 0 is in the
        # center, y == 0 is at the top
        xspacing = 100
        width = max(nIPs/2, len(dp)) * xspacing
        yspacing = width/8
        position_list = [(drug_name, 0, yspacing*0)]
        # I want all of the Dp in a row, below the drug
        # followed by the indirects below at several heights
        xs = list(np.linspace(width/-2, width/2, len(dp)))
        for i,p in enumerate(dp):
            self.cy.add_link(drug_name, p)
            position_list += [(p, xs[i], yspacing*1)]
        # Now add positions for any indirect nodes to position_list
        if nIPs > 0:
            # get all indirects that have multiple connections
            multis = [x
                    for x in iProts
                    if sum([x in y for y in ipd.values()]) > 1
                    ]
            # generate x coords for all non-multi indirects, plus 2 extra
            # for each direct (which will be left empty for spacing)
            xs = list(np.linspace(width/-2, width/2,
                    nIPs+2*len(dp)-len(multis),
                    ))
            # generate alternating y coords (+1 handles odd numbers)
            ys = [yspacing*4,yspacing*3] * ((nIPs//2) + 1)
            # generate evenly-spaced x coords for multis
            xs2 = list(np.linspace(width/-2, width/2, len(multis)))
            seen_ip = []
            gen = (p for p in dp if p in ipd)
            for p in gen:
                gen2 = (ip for ip in ipd[p] if ip not in dp)
                xs.pop(0) # skip x for spacing
                for ip in gen2:
                    # always link
                    self.cy.add_link(p, ip)
                    # only need to store position once per node
                    if ip not in seen_ip:
                        if ip in multis:
                            # on line below directs, evenly spaced
                            position_list += [(ip, xs2.pop(0), yspacing*2)]
                        else:
                            # on 2 lines below that, alternating y's
                            position_list += [(ip, xs.pop(0), ys.pop(0))]
                        seen_ip.append(ip)
                if not xs:
                    break
                xs.pop(0) # skip x for spacing
        # set direct + any indirect positions in diagram
        self.cy.set_abs_positions(position_list)

def get_prot_2_gene(prot_list, uniq_names = True):
    from browse.models import Protein
    prot_qs = Protein.objects.filter(uniprot__in=prot_list)
    result = {
            x.uniprot:(x.gene if x.gene != '' else x.uniprot)
            for x in prot_qs
            }
    # make sure we pass through any old uniprot ids from before condensing
    for p in prot_list:
        if p not in result:
            result[p] = p
    if uniq_names and len(list(result.values())) != len(set(result.values())):
        # some times we get duplicate genes which will mess with things
        # so rename duplicates by appending a number to them
        pairs = list(result.items())
        from dtk.data import uniq_name_list
        uniqs = uniq_name_list([x[1] for x in pairs])
        result = {k:v2 for (k,v1),v2 in zip(pairs,uniqs)}
    return result

class SumStatsTable:
    def __init__(self, **kwargs):
        self.table = [[' ', 'Total', '% detected', 'Log2(Odds Ratio)', 'p-value'],[]]
        self.sig_thresh = kwargs.get('sig_t', None)
        self.min_ds = kwargs.get('min_ds', 1)
        self.ex = kwargs.get('excluded', [])
        self.usable_targets = kwargs.get('all_targets', set())
    def setup(self):
        '''
        Should be overridden by derived class
        '''
        raise NotImplementedError('run not overridden')
    def build_row(self, me_in, total_in, name):
        from dtk.html import decimal_cell
        from math import log
        odds_ratio, pval = self.calc(me_in, total_in)
        self.table[1].append([name
                             , decimal_cell(total_in, fmt="%d")
                             , decimal_cell((float(me_in) / total_in) * 100.0
                                             if total_in else 0.0
                                           )
                             , decimal_cell('-' if odds_ratio == 0 else log(odds_ratio, 2))
                             , decimal_cell(pval, fmt="%0.2e")
                            ])
    def get_counts(self, all_d):
        all_prots = set(all_d.keys()) | self.usable_targets
        self.total_prot = len(all_prots)
        self.total_me_prot = len([x for x in all_d.values()
                                  if x >= self.min_ds]
                                )
    def calc(self, me_in, total_in):
        import scipy.stats as sps
        nme_in = total_in - me_in
        return sps.fisher_exact([
                             [me_in, nme_in],
                             [self.total_me_prot - me_in
                              , self.total_prot - self.total_me_prot - nme_in
                             ]
                            ],
                            alternative = 'greater'
                           )
class GESumStats(SumStatsTable):
    def setup(self, ts):
        self.ts = ts
        from browse.models import Tissue
        all_d = {}
        qs = Tissue.objects.filter(tissue_set_id=self.ts)
        for t in qs:
            if str(t.pk) in self.ex:
                continue
            for rec in t.sig_results(over_only=False):
                if self.sig_thresh is None:
                    _, ev, fc, _ = t.sig_result_counts()
                    increment = 1 if rec.evidence >= ev and rec.fold_change >= fc else 0
                else:
                    increment = 1 if rec.evidence >= self.sig_thresh else 0
                try:
                    all_d[rec.uniprot] += increment
                except KeyError:
                    all_d[rec.uniprot] = increment
        self.get_counts(all_d)
class GwasSumStats(SumStatsTable):
    def setup(self, ws):
        from dtk.gwas import gwas_codes, scored_gwas, score_snp
        score_t = score_snp(self.sig_thresh)
        all_d = {}
        for ds in gwas_codes(ws):
            if ds.lstrip('gwds') in self.ex:
                continue
            gen = (u for u,s in six.iteritems(scored_gwas(ds))
                   if s >= score_t and u in self.usable_targets
                  )
            for u in gen:
                try:
                    all_d[u] += 1
                except KeyError:
                    all_d[u] = 1
        self.get_counts(all_d)

class RecentJobs:
    def __init__(self,start_id):
        from runner.models import Process
        self.job_qs = None
        if not start_id:
            self.start_job = Process.objects.order_by('-id')[0]
            self.start_id = self.start_job.id
        elif start_id < 0:
            # find lowest job id in same batch as start_id
            # if it's != start_id, return it
            # if it's == start_id, return the job before start_id
            start_job = Process.objects.get(pk=-start_id)
            check_time = start_job.created
            from django.db.models import Min,Q
            while True:
                d = Process.objects.filter(
                        Q(completed__gte=check_time)
                                | Q(status__in=Process.active_statuses),
                        created__lte=check_time,
                        ).aggregate(Min('created'))
                group_start = d['created__min']
                assert group_start is not None
                #if group_start is None:
                #    group_start = check_time
                if group_start == check_time:
                    break
                check_time = group_start
            d = Process.objects.filter(
                    created=group_start
                    ).aggregate(Min('id'))
            new_id = d['id__min']
            if new_id == start_job.id:
                # find the previous job
                from django.db.models import Max
                d = Process.objects.filter(
                        id__lt=new_id
                        ).aggregate(Max('id'))
                new_id = d['id__max']
            self.start_id = new_id
            self.start_job = Process.objects.get(pk=new_id)
        else:
            self.start_id = start_id
            self.start_job = Process.objects.get(pk=start_id)
            self.job_qs = Process.objects.filter(id__gte=self.start_id)

@login_required
def jobsum(request,ws_id=None):
    if ws_id:
        ws = Workspace.objects.get(pk=ws_id)
    else:
        ws = None
    from dtk.url import UrlConfig
    url_config=UrlConfig(request, defaults={
                        })
    from dtk.html import link,pad_table
    pause = url_config.as_bool('pause')
    pause_resume = link(
            ('resume' if pause else 'pause')+' refresh',
            url_config.here_url({'pause':0 if pause else 1}),
            )
    # to avoid a bunch of hassle with timezone conversion, we configure
    # the range of interest by a starting job number; that could get
    # fixed someday
    rj = RecentJobs(url_config.as_int('from'))
    if not rj.job_qs:
        return HttpResponseRedirect(url_config.here_url({'from':rj.start_id}))
    after = rj.start_job.created
    from django.db.models import Count
    if ws:
        detail_url=ws.reverse('job_detail')
    else:
        detail_url="/job_detail/"
    count_lookup = {
            d['status']:d['id__count']
            for d in rj.job_qs.values('status').annotate(Count('id'))
            }
    from runner.models import Process
    enum=Process.status_vals
    if enum.QUEUED in count_lookup and enum.RUNNING not in count_lookup:
        logger.info('drive background from jobsum page')
        Process.drive_background()
    rows = [
            (label,'0' if code not in count_lookup
                    else link(
                        count_lookup[code],
                        detail_url+'?from=%d&status=%d'%(rj.start_id,code),
                        )
            )
            for code,label in Process.status_vals.choices()
            ]
    stats_table = pad_table(['',''],rows)
    # AWS control button handling
    worker = Machine.name_index[PathHelper.cfg('worker_machine_name')]
    others = []
    for k,v in six.iteritems(Machine.name_index):
        if v == worker:
            continue
        if k == 'platform':
            continue
        inst = v.get_ec2_instance()
        # verify instance exists here, so we don't throw rendering template
        try:
            inst.instance_type
        except RuntimeError:
            inst = None
        others.append((v,inst))
    others.sort(key=lambda x:x[0].name)
    if request.method == 'POST' and post_ok(request):
        if 'killall' in request.POST:
            if 'check' in request.POST:
                # kill every active job
                Process.kill_all()
                # shut down worker (make sure all remote jobs really died)
                worker.do_stop()
        else:
            mch = Machine.name_index[request.POST['machine']]
            if 'start' in request.POST:
                mch.do_start()
            if 'stop' in request.POST:
                mch.do_stop()
            if 'upgrade' in request.POST:
                mch.do_upgrade()
            if 'downgrade' in request.POST:
                mch.do_downgrade()
    from dtk.table import Table
    instance_table=Table(Machine.upgrade_path,[
                Table.Column('',
                        extract=lambda x:x,
                        ),
                Table.Column('cores',
                        extract=lambda x:Machine.instance_properties[x][0],
                        ),
                Table.Column('ram (GB)',
                        extract=lambda x:Machine.instance_properties[x][1],
                        ),
                Table.Column('$/hr',
                        extract=lambda x:Machine.instance_properties[x][2],
                        ),
                ])
    maint_locks = sorted(x
            for x in Process.maint_status()
            if x.status != 'Idle'
            )
    if maint_locks:
        maint_locks = Table(maint_locks,[
                Table.Column('Task',
                        ),
                Table.Column('Status',
                        ),
                Table.Column('PID',
                        ),
                Table.Column('Progress',
                        ),
                ])
    return render(request
                ,'runner/jobsum.html'
                , make_ctx(request,ws,'jobsum',{
                        'page_tab':'' if ws else 'jobsum',
                        'worker':worker,
                        'others':others,
                        'refresh':0 if pause else 10,
                        'pause_resume':pause_resume,
                        'after':after,
                        'after_job':rj.start_id,
                        'stats_table':stats_table,
                        'instance_table':instance_table,
                        'maint_locks': maint_locks,
                        })
                )

@login_required
def job_detail(request,ws_id=None):
    if ws_id:
        ws = Workspace.objects.get(pk=ws_id)
    else:
        ws = None
    from dtk.url import UrlConfig
    url_config=UrlConfig(request, defaults={
                        })
    from runner.models import Process
    rj = RecentJobs(url_config.as_int('from'))
    status = url_config.as_int('status')
    if status is not None:
        heading = Process.status_vals.get('label',status) + ' Recent Jobs'
        qs = rj.job_qs.filter(status=status)
    else:
        heading =  'All Recent Jobs'
        qs = rj.job_qs
    if ws:
        jwo = JobWorkspaceOption(request,ws,qs)
    else:
        jwo=None
    return render(request
                ,'runner/jobs.html'
                , make_ctx(request,ws,'jobs',{
                        'page_tab':'' if ws else 'jobsum',
                        'heading':heading,
                        'jwo':jwo,
                        'qs':jwo.qs if jwo else qs,
                        'job_cross_checker':JobCrossChecker(),
                        'rm':ResourceManager(),
                        })
                )

@login_required
def jobs(request,ws_id):
    ws = Workspace.objects.get(pk=ws_id)
    jso = JobShowOption(request)
    jwo = JobWorkspaceOption(request,ws,jso.qs.order_by('-id'))
    return render(request
                ,'runner/jobs.html'
                , make_ctx(request,ws,'jobs'
                    ,{'ph':PathHelper
                     ,'qs':jwo.qs
                     ,'refresh': jso.repeat
                     ,'heading': " ".join([jwo.title,jso.title,'Jobs'])
                     ,'jso':jso
                     ,'jwo':jwo
                     ,'job_cross_checker':JobCrossChecker()
                     ,'rm':ResourceManager()
                     }
                    )
                )

class ProtDetailView(DumaView):
    template_name='browse/prot_detail.html'
    GET_parms = {
            'show':(list_of(str),[]),
            'expsort':(SortHandler,'-evidence'),
            'dpisort':(SortHandler,'-evidence'),
            'mapping':(str,None),
            'prescreen':(int,None),
            }
    button_map={
            'redisplay':['prescreen'],
            }

    def custom_setup(self):
        self.expsort.sort_parm='expsort'
        self.dpisort.sort_parm='dpisort'
    def custom_context(self):
        self.load_prot()
        self.context['show']=self.show
        self.load_expression_data()
        self.load_score_data()
    def load_prot(self):
        from browse.models import Protein
        self.context['prot']= Protein.get_canonical_of_uniprot(self.prot_id)
    def load_expression_data(self):
        ge_results = []
        for t in Tissue.objects.filter(ws=self.ws):
            p = t.one_sig_result(self.prot_id)
            if p:
                _,ev,fc,_ = t.sig_result_counts()
                ge_results.append((
                        p.evidence,
                        p.direction,
                        p.fold_change,
                        t,
                        t.set_name(), # simplifies sorting
                        p.evidence >= ev and p.fold_change >= fc,
                        ))
        from dtk.html import link,glyph_icon,join
        from dtk.plot import dpi_arrow
        columns = [
                Table.Column('Evidence',
                    idx=0,
                    cell_fmt=lambda x:"%0.5f"%x,
                    sort='h2l',
                    ),
                Table.Column('Called significant',
                    idx=5,
                    cell_fmt=lambda x:str(x),
                    sort='h2l',
                    ),
                Table.Column('Direction',
                    idx=1,
                    cell_fmt=lambda x:dpi_arrow(x),
                    ),
                Table.Column('Fold Change',
                    idx=2,
                    cell_fmt=lambda x:"%0.5f"%x,
                    sort='h2l',
                    ),
                Table.Column('Tissue',
                    idx=3,
                    ),
                Table.Column('Tissue Set',
                    sort='h2l',
                    idx=4,
                    ),
                Table.Column(Tissue.sig_count_heading(),
                    idx=3,
                    cell_fmt=lambda x:join(
                            x.sig_count_fmt(),
                            link(glyph_icon('stats'),
                                x.ws.reverse('ge:sigprot',x.id),
                                ),
                            )
                    ),
                Table.Column('Quality',
                    idx=3,
                    cell_fmt=lambda x:x.quality_links(self.jcc),
                    ),
                ]
        sort_idx=0
        for x in columns:
            if x.code == self.expsort.colspec:
                sort_idx=x.idx
        ge_results.sort(key=lambda x:x[sort_idx],reverse=self.expsort.minus)
        self.context_alias(
                ge_table=Table(
                        ge_results,
                        columns,
                        url_builder=self.url_builder_factory('ge'),
                        sort_handler=self.expsort,
                        )
                )
    def load_score_data(self):
        scores = []
        from dtk.html import decimal_cell
        from dtk.scores import SourceList
        import runner.data_catalog as dc
        sl = SourceList(self.ws)
        if self.prescreen:
            from browse.models import Prescreen
            pscr = Prescreen.objects.get(id=self.prescreen,ws=self.ws)
            sl.load_from_string(pscr.source_list_jobs())
        else:
            sl.load_from_session(self.request.session)
        for job_id in (int(x.job_id()) for x in sl.sources()):
            src = SourceList.SourceProxy(self.ws,job_id=job_id)
            bji = src.bji()
            cat = bji.get_data_catalog()
            job = bji.job
            for code in cat.get_codes('uniprot','score'):
                    val,_ = cat.get_cell(code,self.protein.uniprot)
                    if val is None:
                        continue
                    row = [
                        ' '.join([src.label(),cat.get_label(code)])
                        , job
                        , decimal_cell(val)
                          ]
                    job=None # only show job info on first score
                    l = cat.get_ordering(code,True)
                    higher = 0
                    tied = 0
                    for item in l:
                        if item[1] > val:
                            higher += 1
                        elif item[1] == val:
                            tied += 1
                        else:
                            break
                    # tied includes this drug, so subtract it (but if the score
                    # is zero, we're not really in the list, so don't subtract
                    # in that case)
                    if tied:
                        tied -= 1
                    lower = len(l) - higher - tied - 1 # the minus 1 is to not count itself
                    row += [
                            decimal_cell(l[0][1]),
                            decimal_cell(higher,fmt='%d'),
                            decimal_cell(tied,fmt='%d'),
                            decimal_cell(lower,fmt='%d'),
                            ]
                    row += (src.bji(),)
                    scores.append(row)
        self.context['scores'] = scores
    # XXX Ideally, the prescreen qparm would get automatically set to the
    # XXX proper prescreen for the drug you linked from. The minimum
    # XXX implementation would be:
    # XXX - the protein page would accept a prescreen parm and pass it to
    # XXX   the Detail link
    # XXX - the drug page would supply the applicable prescreen qparm in
    # XXX   all the links in the Drug Targets collapse section
    # XXX As a further enhancement, the protein page could propagate the
    # XXX qparm to links in the Protein Interactions collapse section.
    # XXX There may be more links in drug subpages or the protein page
    # XXX where this qparm could be supported and propagated.
    # XXX An alternative approach of stashing the prescreen in the session
    # XXX when the drug page is loaded would be too error prone (when drugs
    # XXX from different prescreens are examined simultaneously).
    def make_prescreen_form(self,data):
        class MyForm(forms.Form):
            prescreen = forms.ChoiceField(
                label = 'Score Source',
                choices = [
                        (0,'From session')
                        ]+self.ws.get_prescreen_choices(),
                initial = self.prescreen,
                )
        return MyForm(data)
    def redisplay_post_valid(self):
        p = self.context['prescreen_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))



class ProteinView(DumaView):
    template_name='browse/protein.html'
    GET_parms = {
            'show':(list_of(str),[]),
            'ppi':(str,None),
            'dpi':(str,None),
            }
    button_map={
            'save_note':['protein_note'],
            'ppi':['ppi'],
            'dpi':['dpi'],
            }
    def custom_setup(self):
        self.note_owner = self.request.user.username
        self.load_notes()
    def custom_context(self):
        self.check_druggability()
        self.load_aliases()
        self.load_other_prots_for_this_gene()
        self.context['show']=self.show
        from dtk.url import ext_prot_links
        for cat, cat_links in ext_prot_links(self.ws,self.protein).items():
            self.context[f'{cat}_links'] = cat_links
        self.load_pathways()
        self.load_gwas()
        self.load_phenos()
        self.load_ppi()
        self.load_ps()
        self.load_dpi()

        from dtk.open_targets import OpenTargets
        from browse.default_settings import openTargets
        # Generally you want the latest version when looking at safety and
        # tractability data.
        otarg = OpenTargets(openTargets.latest_version())
        self.load_safety(otarg)
        self.load_tractability(otarg)
        self.load_orthology()

        self.load_other_ws()
    def make_protein_note_form(self,data):
        class MyForm(forms.Form):
            global_note = forms.CharField(
                    widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
                    required=False,
                    label='Global protein note',
                    initial=self.global_note_text,
                    help_text="Non-disease-specific shared notes about this target, such as toxicity or tractability. Always visible.",
                    )
            note = forms.CharField(
                    widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
                    required=False,
                    label='Personal protein note for '+self.note_owner,
                    initial=self.note_text,
                    help_text="Disease-specific personal notes about this target. Visible to others during review.",
                    )
        return MyForm(data)
    def make_dpi_form(self,data):
        from dtk.prot_map import DpiMapping
        class MyForm(forms.Form):
            dpi = forms.ChoiceField(
                label = 'DPI dataset',
                choices = DpiMapping.choices(self.ws),
                initial = self.dpi or self.ws.get_dpi_default(),
                )
        return MyForm(data)
    def make_ppi_form(self,data):
        from dtk.prot_map import PpiMapping
        class MyForm(forms.Form):
            ppi = forms.ChoiceField(
                label = 'PPI dataset',
                choices = PpiMapping.choices(),
                initial = self.ppi or self.ws.get_ppi_default(),
                )
        return MyForm(data)
    def save_note_post_valid(self):
        p = self.protein_note_form.cleaned_data
        from browse.models import TargetReview
        from browse.models import GlobalTargetAnnotation
        TargetReview.save_note(
                self.ws,
                self.protein.uniprot,
                self.note_owner,
                p['note'],
                )

        anno, created = GlobalTargetAnnotation.objects.get_or_create(target=self.protein)
        Note.set(
            anno,
            'note',
            self.request.user.username,
            p['global_note'],
        )
        return HttpResponseRedirect(self.here_url())
    def dpi_post_valid(self):
        p = self.dpi_form.cleaned_data
        return HttpResponseRedirect(self.here_url(**p))
    def ppi_post_valid(self):
        p = self.ppi_form.cleaned_data
        return HttpResponseRedirect(self.here_url(**p))
    def check_druggability(self):
        text = []
        if self.protein.is_small_mol_druggable():
            text.append('Druggable by small molecule per OpenTargets')
        if self.protein.is_dpi_druggable():
            text.append('Druggable by our library')
        if not text:
            text.append('Not druggable by our library or other small molecules')
        self.context_alias(druggability=text)
    def load_aliases(self):
        # if we're rendering the page for an alias, we want the list of alias
        # links to include the main uniprot, and exclude the alias we're
        # already showing
        aliases = self.protein.get_aliases() if self.protein else []
        if self.prot_id != self.protein.uniprot:
            aliases = set(aliases)
            aliases.add(self.protein.uniprot)
            aliases.remove(self.prot_id)
            aliases = list(aliases)
        aliases.sort()
        self.context_alias(aliases=aliases)
    def load_orthology(self):
        from dtk.orthology import get_ortho_records
        data = []
        header = ['organism', 'organism_gene', 'similarity(%)', 'transcripts']
        for rec in get_ortho_records(self.ws, uniprots=[self.protein.uniprot]):
            data.append([rec[x] for x in header])

        from dtk.table import Table
        columns = [Table.Column(x, idx=i) for i, x in enumerate(header)]

        self.context_alias(ortho_table=Table(data, columns))

    def load_phenos(self):
        file_class = 'monarch'
        self.vdefaults=self.ws.get_versioned_file_defaults()
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned(
                file_class,
                self.vdefaults[file_class],
                role='gene',
                )
        s3f.fetch()
        from dtk.files import get_file_records
        data = []
        for rec in get_file_records(
            s3f.path(),
            keep_header=False,
            select=([self.protein.uniprot],0)
            ):
            data.append(rec[1:])

        def pheno_link_fmt(s):
            from dtk.html import link
            from dtk.url import monarch_pheno_url
            return link(s, monarch_pheno_url(s), new_tab=True)
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
        columns = [
                   Table.Column('Phenotype',
                           idx=1,
                           ),
                   Table.Column('Monarch link',
                           idx=0,
                           cell_fmt=pheno_link_fmt,
                           ),
                   Table.Column('Relation',
                           idx=2,
                           ),
                   Table.Column('Evidence',
                           idx=3,
                           cell_fmt=evid_fmt,
                           ),
                   Table.Column('Source',
                           idx=4,
                           ),
                  ]
        self.context_alias(pheno_table=Table(data, columns))

    def load_other_prots_for_this_gene(self):
        if self.protein.gene:
            from browse.models import Protein,TargetAnnotation
            qs=Protein.objects.filter(
                    gene=self.protein.gene,
                    ).exclude(
                    pk=self.protein.id,
                    ).order_by('uniprot')
            all_prots = [x.uniprot for x in qs]
            note_cache = TargetAnnotation.batch_note_lookup(
                    self.ws,
                    all_prots,
                    self.request.user.username,
                    )
            from dtk.prot_map import protein_link
            self.context['gene_other_prots']=[
                    protein_link(
                            x.uniprot,
                            x.gene,
                            self.ws,
                            note_cache=note_cache,
                            use_uniprot_label=True,
                            )
                    for x in qs
                    ]
    def load_pathways(self):
        from dtk.url import pathway_url_factory
        from dtk.html import link
        pathway_url = pathway_url_factory()
        self.context['pathways']=sorted([
                link(x, pathway_url(x))
                for x in self.protein.get_pathways()
                ])
    def load_safety(self, otarg):
        safety = list(otarg.get_safety_data(self.protein.uniprot))
        if safety:
            safety = [['<br>'.join(cell) for cell in row] for row in safety]
            from dtk.table import Table
            columns = [
                Table.Column('Safety', idx=2),
                Table.Column('Organs', idx=3),
                Table.Column('Ref Label', idx=1),
                Table.Column('Ref Link', idx=0),
            ]
            safety_table = Table(safety, columns)
            self.context['safety_data'] = safety_table
    def load_tractability(self, otarg):
        from dtk.table import Table
        from functools import partial
        from django.utils.safestring import mark_safe
        tract = otarg.get_tractability_data(self.protein.uniprot)
        if tract:

            bucket_descrs = {
                'sm': [
                    'Ph 4',
                    'Ph 2/3',
                    'Ph 0/1',
                    'PDB Ligands',
                    'ChEMBL Active',
                    'DrugEBI > 0.7',
                    'DrugEBI 0 to 0.7',
                    'Druggable Genome',
                    '',
                    ],
                'ab': [
                    'Ph 4',
                    'Ph 2/3',
                    'Ph 0/1',
                    'Uniprot Loc (High Conf)',
                    'GO cell component (High Conf)',
                    'Uniprot Loc (Low Conf)',
                    'Uniprot preidcted peptide or region',
                    'GO cell component (Med Conf)',
                    'Human Protein Atlas (High Conf)',
                ],
                'PROTAC': [
                    'Ph 4',
                    'Ph 2/3',
                    'Ph 0/1',
                    '?',
                    '?',
                    '?',
                    '?',
                    '?',
                    '?',
                    '?',
                ],
                'othercl': [
                    'Ph 4',
                    'Ph 2/3',
                    'Ph 0/1',
                ],
            }
            bucket_names = {
                'sm': 'Small Molecule',
                'ab': 'Antibody',
                'PROTAC': 'PROTAC',
                'othercl': 'Other (protein, enzyme, oligonuc, etc.)',
            }
            buckets = tract[0]
            def extract(bucket_num, bucket_type):
                active = bucket_num in buckets[bucket_type]
                el_class = 'active' if active else 'inactive'
                try:
                    descr = bucket_descrs[bucket_type][bucket_num - 1]
                except IndexError:
                    descr = ''
                return mark_safe(f'<span class="{el_class}">{descr}</span>')

            num_buckets = 10
            bucket_columns = [Table.Column('', extract=lambda x:mark_safe(f'<b>{bucket_names[x]}</b>'))] + [
                Table.Column(f'Bucket {i}', extract=partial(extract, i))
                for i in range(1, num_buckets)
            ]

            bucket_table = Table(['sm', 'ab', 'PROTAC', 'othercl'], bucket_columns)

            props = list(tract[1].items())
            columns = [
                Table.Column('Prop', idx=0),
                Table.Column('Value', idx=1),
            ]
            tract_prop_table = Table(props, columns)
            self.context['tract_prop_table'] = tract_prop_table
            self.context['tract_bucket_table'] = bucket_table

    def load_gwas(self):
        from dtk.html import decimal_cell, link
        from dtk.url import pubmed_url, dbsnp_url
        gwas_table = []
        for gwds in self.ws.get_gwas_dataset_qs():
            for rec in gwds.get_data():
                if rec.uniprot == self.protein.uniprot:
                    gwas_table.append([gwds.phenotype.replace("_", " "),
                                       link(gwds.pubmed_id,pubmed_url(gwds.pubmed_id),new_tab=True),
                                       link(rec.snp,dbsnp_url(rec.snp),new_tab=True),
                                       decimal_cell(float(rec.evidence))
                                      ])
        self.context['gwas_table']=gwas_table
    def load_ppi(self):
        from dtk.prot_map import PpiMapping
        ppi_opts = [x[0] for x in PpiMapping.choices()]
        ppi_opts.sort()
        pm = PpiMapping(self.ppi if self.ppi else self.ws.get_ppi_default())
        protlist = pm.get_ppi_info_for_keys([self.protein.uniprot])
        protlist.sort(key=lambda x:x.evidence,reverse=True)
        from dtk.plot import dpi_arrow
        self.context_alias(
                ppi=self.ppi,
                ppi_opts=ppi_opts,
                protlist = [
                        [x, dpi_arrow(float(x.direction))]
                        for x in protlist
                        ]
                )
    def load_dpi(self):
        from dtk.prot_map import DpiMapping
        dpi_map = DpiMapping(self.dpi if self.dpi else self.ws.get_dpi_default())
        bindings = dpi_map.get_drug_bindings_for_prot(prot=self.protein.uniprot)
        key2ev = {x[0]:x[2] for x in bindings}
        agentkeys = key2ev.keys()
        wsa2key = {x[1]:x[0] for x in dpi_map.get_key_wsa_pairs(self.ws, keyset=agentkeys)}
        wsa_ids = wsa2key.keys()
        num_results = len(wsa_ids)
        wsas = WsAnnotation.objects.filter(pk__in=wsa_ids)
        wsas = WsAnnotation.prefetch_agent_attributes(wsas)
        import dtk.molecule_table as MT
        from dtk.table import Table
        def other_col_filter(in_ref, idx, num_mols):
            return not in_ref and (idx >= 5 and num_mols < num_results // 3 or num_mols <= 1)
        col_groups = [
            MT.Name(),
            MT.DrugAttr('native_id'),
            MT.Indication(),
            Table.Column('Evidence', extract=lambda x: key2ev.get(wsa2key[x.id], '?')),
            MT.Dpi(None, ref_prots=[self.protein.uniprot], other_col_filter=other_col_filter, dpi=dpi_map),
        ]
        cols = MT.resolve_cols(col_groups, wsas)
        self.context_alias(
            binding_table=Table(wsas, cols),
        )



    def load_ps(self):
        from browse.models import ProtSet
        pss = []
        for x in ProtSet.objects.filter(proteins__uniprot=self.protein.uniprot, ws=self.ws):
            pss.append(x.name)
        ips = self.ws.get_intolerable_ps_default()
        if ips not in pss:
            ipset = self.ws.get_uniprot_set(ips)
            if self.protein.uniprot in ipset:
                if ips.startswith('globps_unwanted_tier'):
                    pss.append(f'Platform-wide intolerable, tier {ips[-1]}')
                else:
                    pss.append(ips)
        self.context['pss'] = pss
    def load_notes(self):
        uniprot = self.protein.uniprot
        from browse.models import TargetAnnotation
        note_cache = TargetAnnotation.batch_note_lookup(
                self.ws,
                [uniprot],
                self.note_owner,
                )
        self.other_notes = []
        this_prot_notes = note_cache.get(uniprot,{})
        self.note_id,self.note_text = None,'' # defaults if no match found
        for user,(note_id,text) in sorted(this_prot_notes.items()):
            if user == self.note_owner:
                self.note_id,self.note_text = note_id,text
            else:
                self.other_notes.append((user,text))
        # at this point, the current user's note text and id are in
        # self.note_id and self.note_text, and any other readable notes
        # are in self.other_notes as (user,text) pairs.

        from browse.models import GlobalTargetAnnotation
        try:
            global_anno = GlobalTargetAnnotation.objects.get(target=self.protein)
            self.global_note_id = global_anno.note_id
            self.global_note_text = Note.get(global_anno, 'note', '')
        except GlobalTargetAnnotation.DoesNotExist:
            self.global_note_id, self.global_note_text = None, ''

    def load_other_ws(self):
        uniprot = self.protein.uniprot
        other_ws = Workspace.objects.filter(
                targetannotation__uniprot=uniprot,
                targetannotation__targetreview__note__isnull=False,
                ).distinct()

        ws_urls = [ws.reverse('protein', uniprot) for ws in other_ws]
        from dtk.html import link
        self.other_ws = [
                link(ws.name, ws_url)
                for ws, ws_url in zip(other_ws, ws_urls)
                if ws != self.ws
                ]

        from moldata.models import MolSetMoa
        molset_moas = MolSetMoa.objects.filter(proteins=self.protein)
        msmoa_ds = [msmoa.ds for msmoa in molset_moas]
        hs_urls = [f'{ds.ws.reverse("moldata:hit_selection")}?ds=ds{ds.id}'
                   for ds in msmoa_ds]
        self.hit_sel_links = [
            link(f'{ds.name} [{ds.ws.name},  {ds.drugs.count()} molecules]' , hs_url)
            for ds, hs_url in zip(msmoa_ds, hs_urls)
        ]


@login_required
def prot_search(request,ws_id):
    ws = Workspace.objects.get(pk=ws_id)
    matches = []
    subtitle = ''
    if 'search' in request.GET:
        from browse.models import Protein
        pattern = request.GET['search']
        matches = [(p.uniprot, p.gene, p.get_name()) for p in Protein.search(pattern)]
        matches.sort()
        subtitle = "%d matches for '%s'" % (len(matches),pattern)
    return render(request
                ,'browse/prot_search.html'
                , make_ctx(request,ws,'prot_search'
                    ,{'headline':'Protein search'
                     ,'matches':matches
                     ,'subtitle':subtitle
                     }
                    )
                )

class ClustScreenView(DumaView):
    template_name='browse/clust_screen.html'
    GET_parms={
            'ids':(list_of(int),''),
            'dpi':(str,None),
            'dpi_t':(float,0.5),
            'prescreen':(int,None),
            }
    button_map={
            'reclassify':['reclassify','selection'],
            }
    def custom_setup(self):
### updated this when we started linking to indirect clusters
### so we could easily maintain the inner clusters together
        temp = {x.id:x
                for x in WsAnnotation.objects.filter(id__in=self.ids)
               }
        self.drugs = [temp[id] for id in self.ids if id in temp]
        removed_drugs = len(self.ids) - len(self.drugs)
        if removed_drugs:
            self.message(
                    'WARNING: %d drugs from this cluster have been removed'
                    % removed_drugs
                    )
        if self.prescreen:
            from dtk.duma_view import qstr
            self.prescreen_url=self.ws.reverse('nav_scoreboard')+qstr({},
                    prescreen_id=self.prescreen,
                    )
    def custom_context(self):
        self.load_prot_data()
        self.build_table()
    def drug_key(self,d):
        return 'sel_%d'%d.id
    def make_selection_form(self,data):
        from dtk.dynaform import FormFactory
        ff = FormFactory()
        for d in self.drugs:
            if not d.indication:
                f = forms.BooleanField(required=False)
                ff.add_field(self.drug_key(d),f)
        MyForm = ff.get_form_class()
        return MyForm(data)
    def make_reclassify_form(self,data):
        from browse.models import Demerit
        from dtk.html import WrappingCheckboxSelectMultiple
        qs=Demerit.objects.filter(active=True, stage=Demerit.stage_vals.REVIEW).order_by('desc')
        demerit_choices=qs.values_list('id','desc')
        class FormClass(forms.Form):
            demerit = forms.MultipleChoiceField(
                        label='Reject reasons',
                        choices=demerit_choices,
                        widget=WrappingCheckboxSelectMultiple,
                        required=False,
                        )
            indication_href = forms.CharField(
                        required=False,
                        )
        return FormClass(data)
    def reclassify_post_valid(self):
        p = self.selection_form.cleaned_data
        selected = [
                d
                for d in self.drugs
                if p.get(self.drug_key(d))
                ]
        if not selected:
            self.message('No drugs selected')
            return
        p = self.reclassify_form.cleaned_data
        # since we're going to call update_indication multiple times,
        # pre-check to provide nice handling for common errors; the
        # exception thrown by update_indication will report anything
        # we missed, just in an uglier way
        demerits = p['demerit']
        if not demerits:
            self.message('Specify one or more reject reasons')
            return
        if self.prescreen:
            to_url=self.prescreen_url
            from browse.models import Prescreen
            from_prescreen = Prescreen.objects.get(pk=self.prescreen)
        else:
            to_url=self.here_url()
            from_prescreen = None
        for d in selected:
            d.update_indication(
                        d.indication_vals.INACTIVE_PREDICTION,
                        demerits,
                        self.request.user.username,
                        "bulk prescreen",
                        p['indication_href'],
                        from_prescreen=from_prescreen,
                        )
        return HttpResponseRedirect(to_url)
    def load_prot_data(self):
        from dtk.prot_map import DpiMapping
        dpi_mapping = DpiMapping(self.dpi or self.ws.get_dpi_default())
        self.dpi_map = {}
        all_prots = set()
        for d in self.drugs:
            for m in dpi_mapping.get_dpi_info(d.agent):
                if float(m[2]) >= self.dpi_t:
                    self.dpi_map[(m[1],d.agent_id)] = float(m[3])
                    all_prots.add(m[1])
        prot_order = []
        import math
        for p in all_prots:
            score = sum([
                    math.pow(2,2-i)
                    for i,d in enumerate(self.drugs)
                    if (p,d.agent_id) in self.dpi_map
                    ])
            prot_order.append((p,score))
        prot_order.sort(key=lambda x:-x[1])
        self.prot_order = [x[0] for x in prot_order]
    def build_table(self):
        from dtk.html import link
        def sel_extract(d):
            form = self.context['selection_form']
            try:
                return form[self.drug_key(d)]
            except KeyError:
                return ''
        columns = [
                Table.Column('',
                        extract=sel_extract,
                        ),
                Table.Column('',
                        extract=lambda x:link(
                                x.get_name(self.is_demo()),
                                x.drug_url()
                                )
                        ),
                Table.Column('',
                        extract=lambda x:x.indication_link(),
                        ),
                ]
        def prot_extract_factory(prot):
            def extract(d):
                from dtk.plot import dpi_arrow
                try:
                    direction = self.dpi_map[(prot,d.agent_id)]
                except KeyError:
                    return
                return dpi_arrow(direction)
            return extract
        from browse.models import Protein
        for prot in self.prot_order:
            prot_url = self.ws.reverse('protein',prot)
            gene = Protein.get_gene_of_uniprot(prot)
            if not gene:
                gene = "("+prot+")"
            columns.append(Table.Column(link(gene,prot_url),
                    extract=prot_extract_factory(prot),
                    ))
        self.context_alias(table=Table(self.drugs,columns))

class DrugSetView(DumaView):
    template_name='browse/drugset_view.html'
    button_map={
            'select':['select'],
            'change_test_train':[],
            'autosplit':[],
            'base':['base'],
            }
    GET_parms={
            'drugset':(str,None),
            'dpi':(str,None),
            }
    def make_select_form(self, data):
        from dtk.prot_map import DpiMapping
        class MyForm(forms.Form):
            drugset = forms.ChoiceField(
                    label='Drugset',
                    choices = self.ws.get_wsa_id_set_choices(retro=True),
                    required=True,
                    initial=self.drugset
                 )
            dpi = forms.ChoiceField(
                label = 'DPI',
                choices = DpiMapping.choices(ws=self.ws),
                initial = self.dpi or self.ws.get_dpi_default(),
                )
        return MyForm(data)
    def make_base_form(self, data):
        from dtk.prot_map import DpiMapping
        class MyForm(forms.Form):
            base_drugset = forms.ChoiceField(
                    label='',
                    choices = self.ws.get_wsa_id_set_choices(retro=True),
                    required=True,
                    initial=self.drugset
                 )
        return MyForm(data)
    def select_post_valid(self):
        p = self.context['select_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))
    def autosplit_post_valid(self):
        from dtk.kt_split import EditableKtSplit
        split = EditableKtSplit(self.ws, self.drugset)
        split.redo_autosplit()
    def change_test_train_post_valid(self):
        from dtk.kt_split import EditableKtSplit
        wsa_to_test = []
        wsa_to_train = []
        for k, v in six.iteritems(self.request.POST):
            if v == 'on':
                prev_set, wsa_id = k.split('_')
                if prev_set == 'test':
                    wsa_to_train.append(wsa_id)
                elif prev_set == 'train':
                    wsa_to_test.append(wsa_id)
                else:
                    assert False, "Unknown set %s" % prev_set

        print("Sending %s to test, %s to train" % (wsa_to_test, wsa_to_train))
        split = EditableKtSplit(self.ws, self.drugset)
        split.modify_split(wsas_to_test=wsa_to_test, wsas_to_train=wsa_to_train)
    def base_post_valid(self):
        p = self.context['base_form'].cleaned_data
        from dtk.kt_split import EditableKtSplit
        split = EditableKtSplit(self.ws, self.drugset)
        split.align_with_base(p['base_drugset'])
        return HttpResponseRedirect(self.here_url())
    def make_table(self, wsa_ids, include_checkbox):
        qs = WsAnnotation.objects.filter(id__in=wsa_ids)
        from dtk.plot import dpi_arrow
        from dtk.html import link

        def name_column_extractor(wsa):
            return link(
                    wsa.get_name(self.is_demo()),
                    wsa.drug_url(),
                    )

        from moldata.models import ClinicalTrialAudit
        cta_cache = {
                cta.wsa_id:cta
                for cta in ClinicalTrialAudit.get_latest_ws_records(self.ws.id)
                }
        def ct_status_extractor(wsa):
            try:
                cta = cta_cache[wsa.id]
            except KeyError:
                return ""
            return "; ".join(cta.stat_summary())

        from collections import defaultdict
        directions = defaultdict(dict)
        def get_dpi_arrow(wsa, target_name):
            direction = directions[wsa.id].get(target_name, None)
            if direction != None:
                return dpi_arrow(direction)
            else:
                return ''
        from dtk.prot_map import DpiMapping, AgentTargetCache
        targ_cache = AgentTargetCache(
                mapping=DpiMapping(self.dpi),
                agent_ids=[x.agent_id for x in qs],
                dpi_thresh=self.ws.get_dpi_thresh_default(),
                )
        target_names = defaultdict(set)
        for wsa in qs:
            for uniprot, ev, direc in targ_cache.info_for_agent(wsa.agent.id):
                target_names[uniprot].add(wsa.id)
                directions[wsa.id][uniprot] = direc
        from browse.models import Protein
        uniprot2gene_map = Protein.get_uniprot_gene_map(
                                target_names
                                )
        def target_names_key(x):
            num_drugs = len(x[1])
            id = next(iter(sorted(x[1])))
            return (-num_drugs, id)
        target_names = [x[0] for x in sorted(six.iteritems(target_names),
                                             key=target_names_key)]
        def drug_clust_score(wsa):
            return -sum([
                math.pow(2, 2-i)
                for i, trg in enumerate(target_names)
                if trg in directions[wsa.id]
                ])
        qs = sorted(list(qs), key=drug_clust_score)
        from django.utils.html import format_html
        cols = []
        if include_checkbox:
            cols += Table.Column('',
                        extract=lambda x: format_html('<input name="%s_%s" type="checkbox" />' % (include_checkbox, x.id))
                        ),
        cols += [
                Table.Column('Drug name',
                        extract=name_column_extractor,
                        ),
                Table.Column('Indication',
                        extract=lambda x:x.indication_link(),
                        ),
                Table.Column('CT Status',
                        extract=ct_status_extractor,
                        ),
                ]

        urls = [self.ws.reverse('protein',prot) for prot in target_names]
        genes = [uniprot2gene_map.get(prot, prot) for prot in target_names]
        cols += [Table.Column(link(gene, url),
                              extract=lambda x, y=prot:get_dpi_arrow(x, y))
                 for prot, gene, url in zip(target_names, genes, urls)]

        return genes, Table(qs, cols)
    def custom_setup(self):
        if not self.drugset:
            from dtk.moa import un_moa_drugset_variant
            self.drugset = un_moa_drugset_variant(self.ws.eval_drugset)
        if not self.dpi:
            from dtk.moa import un_moa_dpi_variant
            self.dpi = un_moa_dpi_variant(self.ws.get_dpi_default())
    def custom_context(self):
        if not self.drugset:
            self.context_alias(tables=[])
            return
        ids = self.ws.get_wsa_id_set(self.drugset)
        targets, table = self.make_table(ids, include_checkbox=False)

        tables = [('Drugs', table)]

        from dtk.kt_split import is_split_drugset
        if not is_split_drugset(self.drugset):
            test_ids = self.ws.get_wsa_id_set('split-test-' + self.drugset)
            test_genes, test_table = self.make_table(test_ids, include_checkbox='test')
            train_ids = self.ws.get_wsa_id_set('split-train-' + self.drugset)
            train_genes, train_table = self.make_table(train_ids, include_checkbox='train')
            tables += [('Drugs (test)', test_table)]
            tables += [('Drugs (train)', train_table)]

            overlap = sorted(set(test_genes) & set(train_genes))
            split_sizing = [
                    ('Test',len(test_ids),len(test_genes)),
                    ('Train',len(train_ids),len(train_genes)),
                    ]
        else:
            overlap = []
            split_sizing = []


        self.context_alias(
                tables=tables,
                test_train_overlap=overlap,
                split_sizing=split_sizing,
                has_content=True,
                )

class WorkspaceVersionDefaultsView(DumaView):
    template_name='browse/ws_vdefaults.html'
    button_map={
            'save':['defaults'],
            }
    def custom_setup(self):
        try:
            self.ws_id = self.ws.id
            self.label = 'Workspace'
        except AttributeError:
            self.ws_id = None
            self.ws = None
            self.label = 'Global'
    def custom_context(self):
        from browse.models import VersionDefaultAudit, VersionDefault
        audit_qs=VersionDefaultAudit.objects.filter(ws_id=self.ws_id)
        self.context['version_history'] = audit_qs.order_by('-timestamp')
        if self.ws:
            global_defaults = VersionDefault.get_defaults(None)
            from browse.default_settings import Defaultable
            for key in list(global_defaults.keys()):
                cls = Defaultable.lookup(key)
                # We don't want to set anything with a workspace-specific
                # default back to global.
                # This is for things like EvalDrugset which don't make sense
                # to reset to global.
                if hasattr(cls, 'workspace_default'):
                    del global_defaults[key]

            self.context['global_defaults'] = global_defaults
            # Check for MOA mismatch. We do this here so it fires every time
            # the page loads. This means it will appear after the post that
            # creates the problem, and every time someone looks at the page
            # until it gets fixed.
            #
            # Note that it doesn't get invoked on the global page, because
            # that page doesn't support MOA values for EvalDrugset. It's not
            # clear if that's intentional, or just an artifact of how the
            # MOA drugsets are generated. If that gets changed, un-indenting
            # this code one tab will catch the global page as well.
            d = VersionDefault.get_defaults(self.ws_id)
            dpi_moa = '-moa.' in d['DpiDataset']
            eval_moa = d['EvalDrugset'].startswith('moa-')
            if dpi_moa != eval_moa:
                self.message(
                        "MOA setting of DpiDataset and EvalDrugset don't match"
                        )
            # while we're at it, check that the needed version of the
            # MOA collection has been loaded for the selected dpi dataset
            from dtk.moa import MoaLoadStatus,moa_dpi_variant
            mls = MoaLoadStatus()
            from dtk.prot_map import DpiMapping
            moa_mapping = DpiMapping(moa_dpi_variant(d['DpiDataset']))
            if not mls.ok(moa_mapping):
                self.message(
                        f"DpiDataset is v{moa_mapping.version}; last loaded MOA collection (v{mls.last_moa_version}) doesn't support mappings past v{mls.last_merge_version}."
                        )
    def make_defaults_form(self, data):
        from dtk.dynaform import FormFactory
        ff = FormFactory()
        from browse.models import VersionDefault,Workspace
        from browse.default_settings import Defaultable
        d = VersionDefault.get_defaults(self.ws_id)
        ordering = VersionDefault.ordering()
        for key,choice in sorted(d.items(), key=lambda x: ordering[x[0]]):
            cls = Defaultable.lookup(key)
            if not getattr(cls, 'visible', True):
                continue


            if not self.ws_id and not cls.has_global_default():
                # This is a per-workspace setting, no sensible global default.
                continue

            props = {
                    'label':key,
                    'help_text': getattr(cls, 'help_text', ''),
                    'required': getattr(cls, 'required', True),
                    }
            
            # Allow a workspace-dependent help_text function instead.
            if not isinstance(props['help_text'], str):
                props['help_text'] = props['help_text'](self.ws)

            form_type = cls.form_type
            if form_type == float:
                ff.add_field(key,forms.FloatField(
                        initial=choice,
                        **props,
                        ))
            elif form_type == str:
                ff.add_field(key,forms.CharField(
                        initial=choice,
                        **props,
                        ))
            elif form_type == bool:
                ff.add_field(key,forms.BooleanField(
                        initial=choice,
                        **props,
                        ))
            elif form_type == 'choice':
                choices = cls.choices(self.ws)
                if choices:
                    if not self.ws and hasattr(cls, 'latest_version'):
                        latest = cls.latest_version()
                        if latest != choice:
                            help = f'<span class="newer-ver">Latest: {latest}</span>'
                            if 'help_text' in props:
                                props['help_text'] += help
                            else:
                                props['help_text'] = help

                    ff.add_field(key,forms.ChoiceField(
                            choices=choices,
                            initial=choice,
                            **props,
                            ))
            else:
                raise Exception(f'Unexpected form type {form_type}')
        FormClass = ff.get_form_class()
        return FormClass(data)
    def save_post_valid(self):
        p = self.context['defaults_form'].cleaned_data
        altered = []
        from browse.models import VersionDefault
        d = VersionDefault.get_defaults(self.ws_id)
        for key,choice in d.items():
            if key not in p:
                continue # skip any fields excluded above
            if str(choice) != str(p[key]):
                altered.append((key,p[key]))
        if altered:
            VersionDefault.set_defaults(self.ws_id,altered,self.username())
        else:
            self.message('No changes made')
        return HttpResponseRedirect(self.here_url())


# The 'goterm' page was written for the old path storage mechanism, so
# it consistently crashed.  It was very slow and not particularly helpful
# anyway, so it's been removed pending some future rewrite.

# XXX Still to do:
# XXX - pull non-run-specific plots from Explorer page?
# XXX - add indi and drug filtering
# XXX   The original idea here was that these could just be added to
# XXX   DemoFilter, and at least for drugs could be specified using
# XXX   the drugset nomenclature. But converting drugset specifiers to
# XXX   CAS lists requires a database, so it can't be done on worker
# XXX   machines, which eliminates a big part of the reason for
# XXX   shoehorning them into the extended cds name. This is on hold
# XXX   pending a re-thinking. Maybe a new qparm here can take a bg
# XXX   filter drugset (in either include or exclude mode), which a
# XXX   new dtk.faers class could convert to a CAS list, which could
# XXX   then be passed as a new CEC ctor parameter. Something parallel
# XXX   could handle bg indi filters. These would then need to be
# XXX   marshalled and demarshalled if they were deemed useful in
# XXX   run_faers.
class FaersIndiView(DumaView):
    template_name = 'browse/faers_indi.html'
    index_dropdown_stem = 'faers_indi'
    GET_parms={
            'indi':(list_of(str,delim='|'),[]),
            'wsa_list':(list_of(int),[]),
            'ds':(str,None),
            'cds':(str,None),
            'out_type':(str,'i'),
            'plot_type':(str,'r'),
            'min_events':(int,None),
            }
    button_map={
            "redisplay":['bgfilt','config']
            }
    # This view graphs raw data from a clinical events dataset. The general
    # flow is:
    # - cds specifies a clinical dataset to use, with optional background
    #   filtering
    # - events of interest are specified by a list of indications or drugs;
    #   if 'indi' is set, it is used as a list of indications. If not, a
    #   list of drugs is extracted from 'wsa_list' or 'ds' or ws_eval_drugset.
    #   XXX note that wsa_list is just a convenience, as you could achieve the
    #   XXX same thing with ds=wsas#,#,#
    # - the events extracted are converted to either a list of indications
    #   or drugs (controlled by out_type)
    # - the results are plotted as an ordered FC list or FC/prevalence
    #   scatterplot (controlled by plot_type)
    def custom_setup(self):
        if self.ds is None:
            self.ds = self.ws.non_moa_eval_drugset
        # the cvarod case is disabled, but by allowing this to be specified
        # in the URL, we can access DemoFilter functionality
        if not self.cds:
            self.cds = self.ws.get_cds_default()
    def make_bgfilt_form(self,data):
        from dtk.dynaform import FormFactory
        ff = FormFactory()
        from dtk.faers import DemoFilter
        stem,row_filter = DemoFilter.split_full_cds(self.cds)
        filt = DemoFilter(row_filter)
        for filter_part in filt.filter_parts:
            ff.add_field(filter_part,filt.get_form_field(filter_part))
        MyForm = ff.get_form_class()
        return MyForm(data)
    def make_config_form(self,data):
        class MyForm(forms.Form):
            out_type = forms.ChoiceField(
                    choices=[
                            ('d','Drugs'),
                            ('i','Indications'),
                            ],
                    label='Show data for',
                    initial=self.out_type,
                    )
            plot_type = forms.ChoiceField(
                    choices=[
                            ('r','FC Rank'),
                            ('s','FC v Prevalence'),
                            ],
                    label='Plot type',
                    initial=self.plot_type,
                    )
        return MyForm(data)
    def redisplay_post_valid(self):
        # construct filter from bgfilt form
        p = self.context['bgfilt_form'].cleaned_data
        from dtk.faers import DemoFilter
        stem,row_filter = DemoFilter.split_full_cds(self.cds)
        filt = DemoFilter(row_filter)
        filt.load_from_dict(p)
        # update URL with config form plus filter-encoded cds
        p = self.context['config_form'].cleaned_data
        p['cds'] = stem+'?'+filt.as_string()
        return HttpResponseRedirect(self.here_url(**p))
    def get_drug_match_keys(self):
        # get CAS from drug(s)
        if not self.wsa_list:
            self.wsa_list = self.ws.get_wsa_id_set(self.ds)
        self.drugs=[wsa
                for wsa in WsAnnotation.objects.filter(pk__in=self.wsa_list)
                ]
        self.context['drugnames'] = sorted([
                    x.get_name(self.is_demo())
                    for x in self.drugs
                    if x.agent.cas_set
                    ])
        result = set()
        for wsa in self.drugs:
            result |= wsa.agent.cas_set
        return result
    def custom_context(self):
        assert self.out_type in ('i','d'),'Invalid out_type'
        indi_mode = self.out_type == 'i' # else drug mode
        assert self.plot_type in ('r','s'),'Invalid plot_type'
        rank_plot = self.plot_type == 'r' # else scatter plot
        try:
            from dtk.faers import ClinicalEventCounts,CASLookup
            from dtk.faers import ClinicalEventCounts
            cec=ClinicalEventCounts(self.cds)
            if self.indi:
                indi_set = set(self.indi)
                if indi_mode:
                    result = cec.get_disease_co_portions(indi_set)
                    title = 'FAERS Co-occuring Indications'
                else:
                    result = cec.get_drug_portions(indi_set)
                    title = 'FAERS Drugs'
            else:
                cas_set = self.get_drug_match_keys()
                if indi_mode:
                    result = cec.get_disease_portions(cas_set)
                    title = 'FAERS Indications'
                else:
                    result = cec.get_drug_co_portions(cas_set)
                    title = 'FAERS Co-occuring Drugs'
        except ValueError:
            self.context['match_total'] = 0
            return
        bg_ctr,bg_total,match_ctr,match_total=result
        self.context['match_total'] = match_total
        self.context['headline'] = title
        if not match_total:
            return
        # If the condition of interest pulls n% of the database, it should
        # on average pull n% of all uncorrelated conditions. The 'scale'
        # value here adjusts the fold change so that n% is a fold change
        # of 0.
        scale=float(bg_total)/match_total
        if self.min_events is None:
            self.min_events = match_total//100
        bg_ctr = [x for x in bg_ctr if x[1] >= self.min_events]
        match_ctr=dict(match_ctr)
        data=[]
        import math
        if not indi_mode:
            cas_lookup=CASLookup(self.ws.id)
        for key,bg_cnt in bg_ctr:
            if indi_mode:
                wsa_id = None
                label = key
            else:
                tmp = cas_lookup.get_name_and_wsa(key)
                if not tmp:
                    continue
                label,wsa = tmp
                if not wsa:
                    continue
                wsa_id = wsa.id
            match_cnt=match_ctr.get(key,0)
            fc=math.log(scale*max(1,match_cnt)/bg_cnt,2)
            data.append((label,fc,match_cnt,bg_cnt,wsa_id))
        data.sort(key=lambda x:x[1])
        text=[
                '%s<br>%d/%d'%(key,match_cnt,bg_cnt)
                for key,_,match_cnt,bg_cnt,_ in data
                ]
        if rank_plot:
            points = enumerate(x[1] for x in data)
            x_label = ''
        else:
            points = [(math.log(x[3],10),x[1]) for x in data]
            x_label = 'Log of BG event count'
        from dtk.plot import scatter2d
        kwargs=dict(
                x_label=x_label,
                y_label='Fold Change',
                points=points,
                refline=False,
                text=text,
                )
        if not indi_mode:
            kwargs['ids'] = ('drugpage',[x[4] for x in data])
            kwargs['classes'] = [
                    ('Unknown',{'color':'blue'}),
                    ('KT',{'color':'red'}),
                    ]
            kts = self.ws.get_wsa_id_set(self.ds)
            print(self.ds,kts)
            kwargs['class_idx'] = [1 if x[4] in kts else 0 for x in data]
        self.context['plotclass'] = scatter2d(**kwargs)

class FaersRunTable(DumaView):
    template_name = 'browse/faers_run_table.html'
    index_dropdown_stem = 'faers_base'
    GET_parms={
        'jid':(int, None),
        'jid2':(int, None),
        'abs_order':(boolean, True),
        'capp_jid':(int, None),
    }


    def custom_context(self):
        from dtk.faers import make_faers_run_table, make_faers_diff_table, make_faers_general_table
        if not self.jid:
            results_df, strata_df = make_faers_general_table(self.ws)
        elif not self.jid2:
            results_df, strata_df = make_faers_run_table(self.ws.id, self.jid, capp_jid=self.capp_jid)
        else:
            results_df1, _ = make_faers_run_table(self.ws.id, self.jid)
            results_df2, _ = make_faers_run_table(self.ws.id, self.jid2)
            results_df = make_faers_diff_table(results_df1, results_df2, abs_order=self.abs_order)
            strata_df = None

        results_df = results_df.round(3)
        
        if not strata_df.empty:
            strata_df = strata_df.round(2)
            cols = strata_df.columns.tolist()
            cols = [{'title': x} for x in cols]
            self.context_alias(
                strata_cols=cols,
                strata_data=strata_df.to_numpy(),
            )

        cols = results_df.columns.tolist()
        cols = [{'title': x} for x in cols]
        self.context_alias(
            table_cols=cols,
            table_data=results_df.to_numpy(),
        )

class FaersDemoView(DumaView):
    template_name = 'browse/faers_demo_view.html'
    GET_parms={
        'indis':(list_of(str), None),
        'cds':(str, None),
    }
    button_map = {'show': ['show']}
    def make_show_form(self, data):
        wses = Workspace.objects.filter(active=True)
        vocab = self._vocab
        terms = [ws.get_disease_default(vocab) for ws in wses]
        initial = self.indis

        choices=list(zip(terms, wses.values_list('name', flat=True)))
        class MyForm(forms.Form):
            indis = forms.MultipleChoiceField(
                        choices=choices,
                        label='Indications',
                        initial=initial,
                        widget=forms.SelectMultiple(
                                attrs={'size':len(choices)}
                                ),
                        )

        return MyForm(data)

    def show_post_valid(self):
        p = self.context['show_form'].cleaned_data
        p['indis'] = ','.join(p['indis'])
        return HttpResponseRedirect(self.here_url(**p))
    def custom_setup(self):
        from dtk.faers import get_vocab_for_cds_name
        self._cds_name = self.cds or self.ws.get_cds_default()
        self._vocab = get_vocab_for_cds_name(self._cds_name)
        if not self.indis:
            self.indis = [self.ws.get_disease_default(self._vocab)]
    
    def indi_dose_table(self):
        from dtk.faers import make_faers_indi_dose_data
        from dtk.table import Table 

        indi = self.ws.get_disease_default(self._vocab)
        out = make_faers_indi_dose_data(self._cds_name, indi, self.ws.id)

        columns = [
            Table.Column("Drug", idx=0),
            Table.Column("Count", idx=1),
            Table.Column("Indication", idx=2),
            Table.Column("Route", idx=3),
        ]
        return Table(out, columns)

    def custom_context(self):
        import numpy as np
        from dtk.faers import ClinicalEventCounts 
        from dtk.plot import PlotlyPlot, bar_histogram_overlay, scatter2d, fig_legend
        cec = ClinicalEventCounts(self._cds_name)
        sex_idx = ClinicalEventCounts.demo_cols.index('sex')
        age_idx = ClinicalEventCounts.demo_cols.index('age_yr')
        wt_idx = ClinicalEventCounts.demo_cols.index('wt_kg')

        import pandas as pd

        groups = [
            ('All', cec._demo_fm, cec._date_fm),
        ]

        drug_table = self.indi_dose_table()

        for indi in self.indis:
            indi_cols = cec._get_indi_target_col_list(indi.split('|'))
            if indi_cols:
                indi_rows = cec._get_target_mask(indi_cols, cec._indi_fm)
            else:
                indi_rows = []

            short_name = indi
            if len(short_name) > 50:
                short_name = short_name[:47] + '...'
            groups.append((short_name, cec._demo_fm[indi_rows], cec._date_fm[indi_rows]))

        rows = []

        age_x_data = []
        wt_x_data = []
        date_x_data = []

        annos = [[], [], []]
        for i, (group_name, group_fm, group_date_fm) in enumerate(groups):
            demo_col = group_fm[:, sex_idx]
            N = demo_col.data.shape[0]
            male = (demo_col == 1).sum() / N
            female = (demo_col == 2).sum() / N
            rows.append([group_name, 'Male', male])
            rows.append([group_name, 'Female', female])

            # Data gives only the non-empty values.
            age_x_data.append(group_fm[:, age_idx].data)
            wt_x_data.append(group_fm[:, wt_idx].data)
            date_x_data.append(group_date_fm[:, 0].data / 4 + 1970)

            for anno, data in zip(annos, [age_x_data[-1], wt_x_data[-1], date_x_data[-1]]):
                mean = np.mean(data)
                std = np.std(data)
                anno.append(f'[{i}: μ={mean:.1f} σ={std:.2f}]')

        df = pd.DataFrame(rows, columns=['Group', 'Value', 'Portion'])

        import plotly.express as px
        fig = px.bar(df, x='Group', y='Portion', color='Value', barmode='stack')
        fig=  fig.to_dict()
        sex_plot = PlotlyPlot(fig['data'], fig['layout'])

        annos = [[fig_legend([' | '.join(x)], -0.2)] for x in annos]
        data, layout = bar_histogram_overlay(age_x_data, [x[0] for x in groups], bins=25, x_range=[0, 100], density=True, annotations=annos[0])
        layout['width'] = 800
        age_hist = PlotlyPlot(data=data, layout=layout)

        data, layout = bar_histogram_overlay(wt_x_data, [x[0] for x in groups], bins=40, x_range=[0, 160], density=True, annotations=annos[1])
        layout['width'] = 800
        wt_hist = PlotlyPlot(data=data, layout=layout)


        # Avoid aliasing by using exact year binning here.
        max_year = int(np.amax(date_x_data[0])) + 1
        min_year = int(np.amin(date_x_data[0])) - 1
        bins = int(max_year - min_year)
        data, layout = bar_histogram_overlay(date_x_data, [x[0] for x in groups], bins=bins, x_range=[min_year, max_year], density=True, annotations=annos[2])
        layout['width'] = 800
        date_hist = PlotlyPlot(data=data, layout=layout)

        count_rows = []
        for group_name, group_fm, group_date_fm in groups:
            if group_name == 'All':
                continue
            count = group_fm.shape[0]
            has_age = group_fm[:, age_idx] != 0
            has_sex = group_fm[:, sex_idx] != 0
            has_wt = group_fm[:, wt_idx] != 0
            count_rows.append([group_name, 'Total', count])
            count_rows.append([group_name, 'Sex', has_sex.sum()])
            count_rows.append([group_name, 'Age', has_age.sum()])
            count_rows.append([group_name, 'Sex&Age', (has_age.multiply(has_sex)).sum()])
            count_rows.append([group_name, 'Weight', has_wt.sum()])
            count_rows.append([group_name, 'Sex&Age&Weight', (has_wt.multiply(has_age).multiply(has_sex)).sum()])
        df = pd.DataFrame(count_rows, columns=['Group', 'Value', 'Count'])
        fig = px.bar(df, x='Group', y='Count', color='Value', barmode='group')
        fig=  fig.to_dict()
        count_plot = PlotlyPlot(fig['data'], fig['layout'])

        self.context_alias(
            plots=[
                ('Age (yr)', age_hist),
                ('Sex', sex_plot),
                ('Weight (kg)', wt_hist),
                ('Date (yr)', date_hist),
                ('Count', count_plot),
                ],
            drug_table=drug_table,
        )


class FaersView(DumaView):
    template_name = 'browse/faers.html'
    index_dropdown_stem = 'faers_base'
    GET_parms={
            'mode':(int,0),
            'ds':(str,None),
            #'cds':(str,None),
            'overlap_query':(str,None),
            }
    button_map = {'compare': ['compare']}
    modes=(
            dict(src=0,x=1,y=2,
                    label='enrichment vs q-value',
                    ),
            dict(src=0,x=1,y=3,
                    label='portion vs q-value',
                    ),
            dict(src=0,x=2,y=3,
                    label='portion vs enrichment',
                    ),
            dict(src=1,scale=False,
                    label='raw event counts',
                    x='total events',
                    y='disease events',
                    ),
            dict(src=1,scale=True,ratio=False,
                    label='raw event ratios',
                    x='fraction of total events',
                    y='fraction of disease events',
                    ),
            dict(src=1,scale=True,ratio=True,
                    label='raw enrichment',
                    x='fraction of disease events',
                    y='LOG2 enrichment',
                    ),
            dict(src=2,
                    label='polar plot',
                    x='prevalence',
                    y='enrichment',
                    ),
            dict(src=2,
                    label='drug rank',
                    ranked=True,
                    x='',
                    y='enrichment',
                    ),
            )
    def handle_job_id_arg(self,job_id):
        self.job_id = int(job_id)
        # get indication string
        from runner.models import Process
        job = Process.objects.get(pk=self.job_id)
        self.job_settings = job.settings()
        self.context_alias(
                indi = self.job_settings['search_term']
                )
    def custom_setup(self):
        if self.ds is None:
            self.ds = self.ws.eval_drugset
        self.cds=None # force default; cvarod case disabled
    def custom_context(self):
        from runner.process_info import JobInfo
        ubi = JobInfo.get_unbound('faers')
        jobnames = ubi.get_jobnames(self.ws)
        from runner.models import Process
        all_jobs = Process.job_set_history_qs(jobnames).filter(
                                status=Process.status_vals.SUCCEEDED
                                )
        self.mode = self.modes[self.mode]
        from dtk.html import link
        from dtk.text import fmt_time
        if hasattr(self,'job_id'):
            if not self.cds:
                self.cds = self.job_settings.get(
                        'cds',
                        self.ws.get_cds_default(),
                        )
            self.refline=False
            self.ids=[]
            self.text=[]
            self.class_idx=[]
            if self.mode['src']==1:
                self.raw_context()
            elif self.mode['src']==2:
                self.polar_context()
            else:
                self.scored_context()
            if self.mode.get('ranked'):
                # put everything back together
                data=list(zip(self.xy,self.text,self.ids,self.class_idx))
                # sort by y
                data.sort(key=lambda x:x[0][1])
                # take it all apart again, replacing x with a counter
                self.xy=[]
                self.text=[]
                self.ids=[]
                self.class_idx=[]
                for i,((x,y),text,ids,class_idx) in enumerate(data):
                    self.xy.append((len(data)-i,y))
                    self.text.append(text)
                    self.ids.append(ids)
                    self.class_idx.append(class_idx)
            # scatterplot
            from dtk.plot import scatter2d
            self.context['plotclass'] = scatter2d(
                    self.x_label,
                    self.y_label,
                    self.xy,
                    refline=self.refline,
                    ids=('drugpage',self.ids),
                    text=self.text,
                    classes=[
                            ('Unknown',{'color':'blue'}),
                            ('KT',{'color':'red'}),
                            ],
                    class_idx=self.class_idx,
                    )
            self.context['modes'] = [
                    link(mode['label'], self.here_url(mode=i))
                            if mode != self.mode else mode['label']
                    for i,mode in enumerate(self.modes)
                    ]
            
            bji = JobInfo.get_bound(self.ws, self.job_id)
            if os.path.exists(bji.fn_model_output):
                url = f"{self.ws.reverse('faers_run_table')}?jid={self.job_id}"
                self.context['modes'].append(link('FAERS Run Stats Table', url))
        all_jobs = list(all_jobs)
        if all_jobs:
            my_job_id = getattr(self,'job_id',0)
            self.context['joblist'] = [
                    link(
                        fmt_time(job.completed),
                        self.ws.reverse('faers_base')+('%d/'%job.id),
                        )
                        if job.id != my_job_id else fmt_time(job.completed)
                    for job in all_jobs
                    ]
        else:
            self.context['joblist'] = ['No FAERS jobs']
        try:
            co_pattern = self.context['indi']
        except KeyError:
            co_pattern = '%'+self.ws.name+'%'
        from django.utils.http import urlencode
        self.context['co_pattern'] = urlencode({'indi':co_pattern},doseq=True)

        self.ws_compare()
    def ws_compare(self):
        if not self.overlap_query:
            return
        cds = self.cds
        if not cds:
            cds = self.ws.get_cds_default()
        from dtk.faers import ClinicalEventCounts, get_vocab_for_cds_name
        cec = ClinicalEventCounts(cds)
        self_query = self.ws.get_disease_default(
                                    get_vocab_for_cds_name(cds)
                                    )
        indi_set_a = self_query.split('|')
        indi_set_b = self.overlap_query.split('|')

        overlap_results = cec.get_indi_contingency(indi_set_a, indi_set_b)

        from scipy import stats
        import numpy as np
        oddsratio, pvalue = stats.fisher_exact(np.reshape(overlap_results, (2,2)))
        self.context['fisher'] = ['%.3g' % x for x in (oddsratio, pvalue)]
        row_sum = [sum(overlap_results[i&~1:(i&~1) + 2]) for i in range(4)]
        overlap_format = ['%d (%.1f%%)' % (x, x * 100 / row) for x, row in zip(overlap_results, row_sum)]
        self.context['overlap_results'] = overlap_format




    def make_compare_form(self, data):
        class MyForm(forms.Form):
            overlap_query = forms.CharField(
                    label='Indication Query',
                    required=True,
                    initial=self.overlap_query,
                    )
        return MyForm(data)

    def compare_post_valid(self):
        p = self.context['compare_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))

    def polar_context(self):
        # This is very much like raw_context below, except we want to
        # show even zero-scoring drugs, so we iterate through the
        # background counts instead of the CAS counts, and we
        # only support one flavor of score.
        from dtk.faers import ClinicalEventCounts,CASLookup
        cas_lookup=CASLookup(self.ws.id)
        kts = self.ws.get_wsa_id_set(self.ds)
        # get background event count for each drug
        # XXX consider adding a UI for setting the disease match pattern
        # XXX and the data source (or maybe split this into two pages,
        # XXX one to display job output, and the other to explore the
        # XXX data interactively)
        cec=ClinicalEventCounts(self.cds)
        bg_ctr,bg_total,indi_ctr,disease_total=cec.get_drug_portions(
                                                    self.indi.split('|')
                                                    )
        bg_per_cas = dict(bg_ctr)
        disease_per_cas = dict(indi_ctr)
        # build plotly input
        xy = []
        import math
        for cas,bg_cnt in six.iteritems(bg_per_cas):
            tmp = cas_lookup.get_name_and_wsa(cas)
            if not tmp:
                continue
            name,wsa = tmp
            if not wsa:
                continue
            disease_cnt = disease_per_cas.get(cas,0)
            rd = float(disease_cnt)/disease_total
            rb = float(bg_cnt)/bg_total
            xy.append((
                    math.sqrt(rd**2+rb**2),
                    math.atan2(rd,rb)-math.pi/4,
                    ))
            self.ids.append(wsa.id)
            self.text.append('%s<br>%d / %d' % (name,disease_cnt,bg_cnt) )
            self.class_idx.append( 1 if wsa.id in kts else 0 )
        print('got',len(xy),'plot points')
        self.x_label = self.mode['x']
        self.y_label = self.mode['y']
        self.xy = xy
        self.refline=False
    def raw_context(self):
        # this sets up for the case where we filter events on indication,
        # and plot one point per drug
        from dtk.faers import ClinicalEventCounts,CASLookup
        cas_lookup=CASLookup(self.ws.id)
        kts = self.ws.get_wsa_id_set(self.ds)
        # get portion data
        cec=ClinicalEventCounts(self.cds)
        bg_ctr,bg_total,indi_ctr,disease_total=cec.get_drug_portions(
                                                    self.indi.split('|')
                                                    )
        bg_per_cas = dict(bg_ctr)
        xy = []
        for cas,disease_cnt in indi_ctr:
            tmp = cas_lookup.get_name_and_wsa(cas)
            if not tmp:
                continue
            name,wsa = tmp
            if not wsa:
                continue
            if not disease_cnt:
                continue
            bg_cnt = bg_per_cas[cas]
            xy.append( (bg_cnt,disease_cnt) )
            self.ids.append(wsa.id)
            self.text.append('%s<br>%d / %d' % (name,disease_cnt,bg_cnt) )
            self.class_idx.append( 1 if wsa.id in kts else 0 )
        print('got',len(xy),'plot points')
        if self.mode['scale']:
            # divide into everything in xy
            xy = [
                    (float(bg)/bg_total, float(disease)/disease_total)
                    for bg,disease in xy
                    ]
            import math
            if self.mode['ratio']:
                xy = [
                    (disease,math.log(disease/bg,2))
                    for bg,disease in xy
                    ]
        self.x_label = self.mode['x']
        self.y_label = self.mode['y']
        self.xy = xy
        self.refline=self.mode['scale'] and not self.mode['ratio']
    def scored_context(self):
        # this sets up for the case where we extract a score from a FAERS
        # run output file
        from path_helper import PathHelper
        from runner.process_info import JobInfo
        bji = JobInfo.get_bound(self.ws, self.job_id)
        fn = bji.fn_enrichment
        from dtk.files import get_file_records
        raw = list(zip(*get_file_records(fn,keep_header=True)))
        # XXX q_values can be very small, and plotly seems to have
        # XXX trouble with stuff < E-10 (it graphs it correctly,
        # XXX but drops the exponent in the hover box)
        # XXX Converting to log scale fixes this, but it compresses
        # XXX the upper values too much.
        #import math
        #logged = [math.log(float(x),10) for x in raw[1][1:]]
        #raw[1] = [raw[1][0]] + logged
        self.ids = [int(x) for x in raw[0][1:]]
        data = list(zip(raw[self.mode['x']],raw[self.mode['y']]))
        name_map = self.ws.get_wsa2name_map()
        kts = self.ws.get_wsa_id_set(self.ds)
        for wsa_id in self.ids:
            try:
                name = name_map[wsa_id]
            except KeyError:
                name = "(%d)" % wsa_id
            self.text.append(name)
            self.class_idx.append( 1 if wsa_id in kts else 0 )
        self.x_label= data[0][0]
        self.y_label= data[0][1]
        self.xy = data[1:]

@login_required
def plotly(request,ws_id):
    ws = Workspace.objects.get(pk=ws_id)
    mode = request.GET.get('mode','mpl')
    save = request.GET.get('save','')
    y=[x*x/10000.0 for x in range(100)]
    from dtk.plot import PlotlyPlot,annotations
    if mode == 'mpl':
        # matplotlib case
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(y)
        fig = plt.figure(1)
        pc = PlotlyPlot.build_from_mpl(fig)
    elif mode=='direct':
        pc = PlotlyPlot(
                [{'y':y}],
                )
    elif mode=='plotly':
        # see https://plot.ly/python/
        # use graph_objs to build data and layout params,
        # but use PlotlyPlot class to do actual rendering (not plotly.(i)plot)
        pc = PlotlyPlot(
                data=[
                        dict(x=[1,2,3],y=[2,3,1]),
                        dict(type='bar',x=[5,6],y=[6,7],name='labeled'),
                        dict(y=y,mode='markers',marker={'size':2}),
                        ],
                layout={'title':'my title'},
                )
    elif mode=='scatter2':
        import random
        xy = [(float(x),random.uniform(x-15,x+15)) for x in range(100)]
        class_idx = [1 if int(v[0]) % 7 == 0 else 0 for v in xy]
        from dtk.plot import scatter2d
        pc = scatter2d('linear x','x + random',xy,
                title='A Scatterplot Example',
                classes=[
                        ('Unknown',{'color':'blue'}),
                        ('KT',{'color':'red'}),
                        ],
                class_idx=class_idx,
                annotations=annotations('a single annotation')
                )
    elif mode=='scatter':
        import random
        xy = [(float(x),random.uniform(x-15,x+15)) for x in range(100)]
        identity = [x[0] for x in xy]
        x1,y1 = zip(*[x for x in xy if int(x[1]) % 7 == 0])
        x2,y2 = zip(*[x for x in xy if int(x[1]) % 7 != 0])
        # explore some reasonable settings for scatterplots
        pc = PlotlyPlot(
                data=[
                        dict(
                                name='other',
                                y=y2,
                                x=x2,
                                mode='markers',
                                marker={
                                    'color':'blue',
                                    },
                                ),
                        dict(
                                name='KTs',
                                y=y1,
                                x=x1,
                                mode='markers',
                                marker={
                                    'color':'red',
                                    },
                                ),
                        dict(
                                y=identity,
                                x=identity,
                                mode='lines',
                                showlegend=False,
                                hoverinfo='skip',
                                ),
                        ],
                layout={
                        'title':'a scatterplot',
                        'hovermode':'closest',
                        'xaxis':{'title':'linear x'},
                        'yaxis':{'title':'x + random'},
                        'legend':{'traceorder':'reversed'},
                        'annotations':annotations('t1', 't2'),
                        },
                )
    elif mode == 'offline':
        pc = PlotlyPlot.build_from_file('/tmp/xxx')
    else:
        raise Exception("unimplemented mode: '%s'" % mode)
    if save:
        pc.save(save)
        pc = PlotlyPlot.build_from_file(save)
    return render(request
                ,'browse/plotly.html'
                , make_ctx(request,ws,'test'
                    ,{'page_label':'static'
                     ,'plotclass':pc
                     }
                    )
                )

class WorkflowCyGraph:
    def __init__(self):
        self._nodes = {}
        self._abs_list = None
        self._hover_info = {}
        self.props = {}
    def add_prop_type(self, k):
        self.props[k] = set()
    def add_prop(self, n, k):
        self.props[k].add(n)
    def node_keys(self):
        return list(self._nodes.keys())
    def _get_node(self,name):
        if name not in self._nodes:
            node = dict(name=name,inputs=[],nbr=len(self._nodes)+1)
            self._nodes[name] = node
        return self._nodes[name]
    def _fmt(self,data):
        from django.utils.safestring import mark_safe
        import json
        return mark_safe(json.dumps(data))
    def add_node(self,src):
        self._get_node(src)
    def add_link(self,src,targ):
        n1 = self._get_node(src)
        n2 = self._get_node(targ)
        n2['inputs'].append(n1)
    def set_abs_positions(self,abs_list):
        self._abs_list = abs_list
    def set_hover(self,d):
        self._hover_info = d
    def style(self):
        from django.utils.safestring import mark_safe
        return mark_safe('''[
                { selector: 'node', style: {
                        'label': 'data(id)',
                        'shape':'roundrectangle',
                        'color':'#F48725',
                        'background-color':'#F48725',
                        'opacity':'1.0'
                        } },
                { selector: "[not_sig='y']", style: {
                        'color': '#1A84C6',
                        'background-color': '#1A84C6',
                        'opacity':'0.5'
                        } },
                { selector: "[drug='y']", style: {
                        'color': 'black',
                        'background-color': 'black',
                        } },
                { selector: 'edge', style: {
                        'width': 3,
                        'line-color': 'grey',
                        'curve-style': 'bezier',
                        } },
                ]''')
    def elements(self):
        from dtk.data import merge_dicts
        result = [
                {'data':merge_dicts(
                             {'id':n},
                             {k:'y' if n in self.props[k] else 'n' for k in self.props.keys()}
                            )
                      }
                for n in self._nodes.keys()
                ]
        result += [
                {'data':{
                        'id': '%s_%s' % (n2['name'], n1['name']),
                        'source':n2['name'],
                        'target':n1['name'],
                        }}
                for n1 in self._nodes.values()
                for n2 in n1['inputs']
                ]
        return self._fmt(result)
    def tips(self):
        return self._fmt(self._hover_info)
    def positions(self):
        result = {}
        if self._abs_list:
            for label,x,y in self._abs_list:
                result[label] = dict(x=x,y=y)
        if not result:
            return None
        return self._fmt(result)

class WfDiagView(DumaView):
    template_name='browse/wf_diag.html'
    def custom_context(self):
        self.context_alias(cy = WorkflowCyGraph())
        self.build_diagram()
        self.scan_plugins()
    def build_diagram(self):
        from workflows.refresh_workflow import StandardWorkflow
        std_wf = StandardWorkflow(ws=self.ws)
        std_wf.diagram(self.cy)
    def scan_plugins(self):
        # scan plugins to get LTS status and popup description
        not_LTS = set()
        not_refresh = set()
        not_both = set()
        descs = {}
        from dtk.data import MultiMap
        plugin2node = MultiMap([
                (node.split('_')[-1],node)
                for node in self.cy.node_keys()
                ])
        for plugin in JobInfo.get_plugin_names():
            uji = JobInfo.get_unbound(plugin)
            if plugin in plugin2node.fwd_map():
                desc = uji.description()
                if desc:
                    for node in plugin2node.fwd_map()[plugin]:
                        descs[node] = desc
                if not uji.use_LTS:
                    not_LTS.add(plugin)
            else:
                if uji.use_LTS:
                    not_refresh.add(plugin)
                else:
                    not_both.add(plugin)
        if descs:
            self.cy.set_hover(descs)
        self.not_LTS = ' '.join(sorted(not_LTS))
        self.not_refresh = ' '.join(sorted(not_refresh))
        self.not_both = ' '.join(sorted(not_both))

# XXX The user monitoring mechanism might be a good candidate for breaking
# XXX out into a separate app.  It would require working out a lot of the
# XXX django introspection techniques.  Also, this is a candidate for any
# XXX generic table object.
@login_required
def users(request,ws_id=None):
    ws = Workspace.objects.get(pk=ws_id) if ws_id else None
    # build a map of active sessions for each user id
    # - _auth_user_id may not be present if session isn't yet authenticated
    # - only look at non-expired sessions
    # - install script cleans out session table periodically
    # - user may be logged in from multiple devices simultaneously; there's
    #   no useful identifying info in the session to distinguish devices,
    #   so cover them all with a single logout operation
    session_map = {}
    from django.contrib.sessions.models import Session
    from django.utils import timezone
    for s in Session.objects.filter(expire_date__gte=timezone.now()):
        u_id = s.get_decoded().get('_auth_user_id')
        if u_id:
            session_map.setdefault(int(u_id),[]).append(s.session_key)
    if request.method == 'POST':
        if not is_demo(request.user):
            from browse.models import UserAccess
            for key in request.POST:
                if not key.endswith('_btn'):
                    continue
                parts = key.split('_')
                assert len(parts) == 3
                if parts[0] == 'clear':
                    UserAccess.objects.filter(pk=parts[1]).delete()
                elif parts[0] == 'logout':
                    for key in session_map.get(int(parts[1]),[]):
                        Session.objects.filter(session_key=key).delete()
                elif parts[0] == 'bulk' and parts[1] == 'clear':
                    # re-arm all source hosts not mapped to names
                    for ua in UserAccess.objects.filter():
                        if ua.host == ua.mapped_host():
                            ua.delete()
                else:
                    raise NotImplementedError('unknown button '+key)
        return HttpResponseRedirect(request.path)
    from django.contrib.auth import get_user_model
    User = get_user_model()
    users = User.objects.filter(is_active=True).order_by('username')
    from two_factor.utils import default_device
    for u in users:
        for name in ('duma_admin','button_pushers'):
            if u.groups.filter(name=name).exists():
                setattr(u,'in_'+name,True)
        u.clean = not u.user_permissions.exists()
        u.two_factor = default_device(u)
        u.normal_access = u.useraccess_set.filter(access='normal')
        u.unverified_access = u.useraccess_set.filter(access='unverified')
        u.sessions = session_map.get(u.id,[])
    return render(request
                ,'browse/users.html'
                , make_ctx(request,ws,'users'
                    ,{
                    'page_tab':'' if ws else 'users',
                    'users': users,
                     }
                    )
                )

class S3CacheView(DumaView):
    template_name='browse/s3_cache.html'
    button_map={
            'clear':['selection'],
            }
    def custom_setup(self):
        from path_helper import PathHelper
        self.key2path = {
                d.split('/')[-2] : d+'__list_cache'
                for d in PathHelper.s3_cache_dirs
                }
    def clear_post_valid(self):
        p = self.selection_form.cleaned_data
        import os
        for k in p:
            if not p[k]:
                continue
            path = self.key2path[k]
            os.remove(path)
        return HttpResponseRedirect(self.here_url())
    def make_selection_form(self,data):
        from dtk.dynaform import FormFactory
        ff = FormFactory()
        import os,pytz,datetime
        from dtk.text import fmt_time
        for key,path in sorted(self.key2path.items()):
            try:
                stat = os.stat(path)
            except FileNotFoundError:
                continue
            mod_time=datetime.datetime.fromtimestamp(stat.st_mtime,tz=pytz.utc)
            ff.add_field(key, forms.BooleanField(
                    required=False,
                    label=f'{key} ({fmt_time(mod_time)})',
                    ))
        MyForm = ff.get_form_class()
        return MyForm(data)

class CollStatsView(DumaView):
    template_name='browse/coll_stats.html'
    GET_parms={
            'sort':(SortHandler,'name'),
            }
    # XXX - version history could maybe be a link in the table; this requires
    # XXX   loading a collection into CollStats from an attributes file
    # XXX - possible visualization: https://medium.com/plotly/4-interactive-sankey-diagram-made-in-python-3057b9ee8616
    def custom_context(self):
        from dtk.coll_stats import CollStats
        mol_cs = CollStats()
        targ_cs = CollStats()
        from browse.models import Collection
        coll_names = Collection.default_collections
        if False:
            # force subset of collections for faster testing
            coll_names = ['drugbank.full','ncats.full','duma.full']
        prop_names = [
                'cas',
                'std_smiles',
                'atc',
                ]
        # get per-collection molecule data
        found_names = set()
        for c in Collection.objects.filter(name__in=coll_names):
            mol_cs.add_from_model(c,prop_names)
            found_names.add(c.name)
        coll_names = [x for x in coll_names if x in found_names]
        # add in target data
        from browse.models import VersionDefault
        from dtk.prot_map import DpiMapping
        dm = DpiMapping(VersionDefault.get_defaults(None)['DpiDataset'])
        from dtk.data import MultiMap
        mol2targ_mm = MultiMap(dm.get_filtered_key_target_pairs())
        for name in coll_names:
            for key in mol_cs.keys(name):
                targ_cs.add_keys(name,mol2targ_mm.fwd_map().get(key,set()))
        # get maps for uniqueness/overlap determination
        collset2mol_mm = mol_cs.get_collset2key_mm()
        collset2targ_mm = targ_cs.get_collset2key_mm()
        if False:
            for k,s in collset2targ_mm.fwd_map().items():
                print(k,s)
        # build rows for table display
        class Row: pass
        rows = []
        for coll_name in coll_names:
            r = Row()
            rows.append(r)
            r.name = coll_name
            r.total_molecules = len(mol_cs.keys(coll_name))
            for prop_name in prop_names:
                attr_name = 'with_'+prop_name
                setattr(r,attr_name,len(mol_cs.filter(
                        coll_name,
                        lambda x:getattr(x,attr_name)
                        )))
            unique = collset2mol_mm.fwd_map().get(coll_name,set())
            r.unique_molecules = len(unique)
            important = mol_cs.filter(coll_name,lambda x:x.super_important)
            r.total_important = len(important)
            r.unique_important = len(unique & important)
            r.total_targets = len(targ_cs.keys(coll_name))
            r.unique_targets = len(
                    collset2targ_mm.fwd_map().get(coll_name,set())
                    )
        rows.sort(
                key=lambda x:getattr(x,self.sort.colspec),
                reverse=self.sort.minus,
                )
        from dtk.table import Table
        self.context_alias(table=Table(rows,[
                Table.Column('Name',sort='l2h',
                        ),
                Table.Column('Total Molecules',sort='h2l',
                        ),
                Table.Column('With Std Smiles',sort='h2l',
                        ),
                Table.Column('With CAS',sort='h2l',
                        ),
                Table.Column('With ATC',sort='h2l',
                        ),
                Table.Column('Unique Molecules',sort='h2l',
                        ),
                Table.Column('Total Important',sort='h2l',
                        ),
                Table.Column('Unique Important',sort='h2l',
                        ),
                Table.Column('Total Targets',sort='h2l',
                        ),
                Table.Column('Unique Targets',sort='h2l',
                        ),
                ],
                sort_handler = self.sort,
                url_builder = self.here_url,
                ))

class EtlStatusView(DumaView):
    template_name='browse/etl_status.html'
    GET_parms={
            'sort':(SortHandler,'name'),
            }
    def custom_context(self):
        from dtk.etl import get_all_etl_names, get_etl_data, etl_tiers
        import os
        self.all_etl_names = get_all_etl_names()
        self.name_lookup = {x.lower():x for x in self.all_etl_names}
        rows = [get_etl_data(x, name_lookup=self.name_lookup)
                for x in self.all_etl_names
               ]
        rows.sort(
                key=lambda x:(
                        getattr(x,self.sort.colspec),
                        x.name, # backup sort for disambiguation
                        ),
                reverse=self.sort.minus,
                )
        # cross-check against misspelling in tiers structure
        invalid_names = []
        for l in etl_tiers.values():
            invalid_names += [x for x in l if x not in self.all_etl_names]
        if invalid_names:
            raise KeyError(
                    'mis-configured names in tiers: '+repr(invalid_names)
                    )
        from dtk.url import chembl_assay_url, bindingdb_drug_url
        from dtk.table import Table
        from dtk.html import tag_wrap,lines_from_pairs
        def deprecated_html_hook(data,row,col):
            if row.deprecated:
                return 'DEPRECATED'
            return data
        self.context_alias(table=Table(rows,[
                Table.Column('Name',sort='l2h',
                        cell_fmt=lambda x:tag_wrap('a',x,attr={
                                'href':reverse('etl_history',args=[x]),
                                }),
                        ),
                Table.Column('Priority',sort='l2h',
                        code='tier',
                        cell_fmt=lambda x:x if x else 'MISSING',
                        cell_html_hook=deprecated_html_hook,
                        ),
                Table.Column('Latest Version',sort='l2h',
                        cell_fmt=lambda x:x if x else '', # don't show 0
                        ),
                Table.Column('Description',sort='l2h'),
                Table.Column('Published',sort='l2h'),
                Table.Column('Source Version'),
                Table.Column('Dependencies',
                        cell_fmt=lines_from_pairs,
                        ),
                Table.Column('Other Info',
                        cell_fmt=lines_from_pairs,
                        ),
                ],
                sort_handler = self.sort,
                url_builder = self.here_url,
                ))

class EtlHistoryView(DumaView):
    template_name='browse/etl_history.html'
    def handle_etl_dir_arg(self,etl_dir):
        self.etl_dir = etl_dir
        from dtk.etl import get_versions_namespace
        self.versions = get_versions_namespace(etl_dir)
    def custom_context(self):
        if not self.versions:
            return
        from dtk.etl import get_etl_name_lookup,get_etl_data,generate_all_stats,stats_to_plots
        name_lookup=get_etl_name_lookup()
        rows = [
                get_etl_data(
                        self.etl_dir,
                        name_lookup=name_lookup,
                        version=version,
                        ns=self.versions,
                        )
                for version in sorted(
                        self.versions['versions'].keys(),
                        reverse=True,
                        )
                ]
        # highlight dependency changes
        from dtk.text import compare_dict
        for i in range(len(rows)-1):
            if rows[i+1].dependencies:
                rows[i].diff = compare_dict('prev',
                        dict(rows[i+1].dependencies),
                        dict(rows[i].dependencies),
                        )
            else:
                rows[i].diff = ''
        rows[-1].diff = ''
        from dtk.table import Table
        from dtk.html import lines_from_pairs
        plots = []
        if False:
            # XXX this can take forever; disable it for now; maybe add a
            # XXX qparm and a 'show plots' link?
            cols, dfs = generate_all_stats(self.etl_dir)
            for col in cols:
                from dtk.plot import Controls
                plot = stats_to_plots(col, dfs)
                plot.add_control(Controls.log_linear_y)
                plots.append([col, plot])
        self.context_alias(
            plotly_plots=plots,
            table=Table(rows,[
                Table.Column('Version'),
                Table.Column('Description'),
                Table.Column('Published'),
                Table.Column('Source Version'),
                Table.Column('Dependencies',
                        cell_fmt=lines_from_pairs,
                        ),
                Table.Column('Diff'),
                Table.Column('Other Info',
                        cell_fmt=lines_from_pairs,
                        ),
                ],
                ))

class EtlOrderView(DumaView):
    template_name='browse/etl_order.html'
    def custom_context(self):
        from dtk.etl import get_all_versioned_etl, order_etl
        all_etl = get_all_versioned_etl()
        self.context_alias(etl_groups=[
                sorted(subset,key=lambda x:x.name)
                for subset in order_etl(all_etl.values())
                ])
        # now annotate rows for display
        import datetime as dt
        from dtk.html import join,tag_wrap,glyph_icon
        from django.utils.safestring import mark_safe
        from dtk.text import fmt_interval
        def last_check_parser(x):
            day,note = x
            base_date = dt.date.fromisoformat(day)
            return (
                    base_date,
                    glyph_icon('info-sign',hover=f'checked {day}: '+note),
                    )
        for wave in self.etl_groups:
                for src in wave:
                    if src.months_between_updates is None:
                        src.update_freq = 'Never'
                        src.next_update = ''
                    else:
                        interval = dt.timedelta(
                                days=365*(src.months_between_updates/12)
                                )
                        src.update_freq = f'{src.months_between_updates} months'
                        try:
                            base_date = dt.date.fromisoformat(src.published)
                        except ValueError:
                            # src.published is an error condition, not a date
                            base_date = None
                        if src.last_check:
                            # A global 'last_checked' value was specified in
                            # the versions.py file. Use this in place of the
                            # published date.
                            try:
                                base_date,note = last_check_parser(
                                        src.last_check
                                        )
                            except ValueError as ex:
                                # there's a format error in the versions.py file
                                base_date = None
                                note = '(Parse Error)'
                                self.message(
                                        'Parse Error: last_checked in %s (%s)'
                                        % (src.name,ex)
                                        )
                            src.update_info = note
                        # now calculate next_update column
                        if not base_date:
                            src.next_update = 'N/A'
                        else:
                            next_update = base_date+interval
                            src.next_update = fmt_interval(
                                    next_update-dt.date.today()
                                    )
                    outdated_inputs = []
                    for inp,ver in src.dependencies:
                        try:
                            ver = int(ver)
                        except ValueError:
                            # handle case like uniprot = HUMNAN9606.v7
                            ver = ver.split('.')[-1]
                            # handle case with leading 'v'
                            assert ver[0] == 'v'
                            ver = int(ver[1:])
                        avail = all_etl[inp].latest_version
                        if avail != ver:
                            arrow = '\u21e8'
                            outdated_inputs.append(tag_wrap('b',
                                    f'{inp}:v{ver}{arrow}v{avail}'
                                    ))
                        else:
                            outdated_inputs.append(
                                    f'{inp}:v{ver}'
                                    )
                    if outdated_inputs:
                        src.pending_inputs=join(
                                *outdated_inputs,
                                sep=mark_safe('<br>')
                                )

class CreditsView(DumaView):
    template_name='browse/credits.html'
    def custom_context(self):
        credits_fn = PathHelper.website_root+'credits/credits.json'
        with open(credits_fn) as f:
            import json
            result = json.load(f)
        self.context_alias(db_credits=result['db_credits'])

class MemView(DumaView):
    template_name='browse/mem.html'
    GET_parms={
            'save':(int,0),
            'filter':(str,None),
            }
    prev_level={}
    def custom_context(self):
        import gc
        print(gc.get_count())
        gc.collect()
        print(gc.get_count())
        import objgraph
        prev = self.prev_level
        by_type = objgraph.most_common_types(limit=None,shortnames=False)
        if self.filter:
            display = [x for x in by_type if self.filter in x[0]]
        else:
            display = by_type
        self.context_alias(table=Table(display,[
                Table.Column('Type',idx=0),
                Table.Column('Count',idx=1),
                Table.Column('Delta',
                        extract=lambda x:x[1] - prev.get(x[0],0)
                        ),
                ]))
        if self.save:
            MemView.prev_level = dict(by_type)

class DumaTestView(DumaView):
    template_name='browse/test.html'
    index_dropdown_stem='test'
    GET_parms = {
            'pop':(str,''),
            }
    def custom_context(self):
        import sys
        # verify that URLs can be nested
        self.context_alias(links=[])
        from dtk.html import link
        from dtk.duma_view import qstr
        self.links.append(link('push',qstr({'pop':self.here_url()})))
        if self.pop:
            self.links.append(link('pop',self.pop))
        # show some random stuff
        self.context['page_label'] = 'This is a class-based-view test'
        self.context['info'] = [
                            x+':'+os.environ.get(x,'NOT SET')
                            for x in (
                                    'PATH',
                                    'PYTHONPATH',
                                    'LD_LIBRARY_PATH',
                                    'DJANGO_SETTINGS_MODULE',
                                    )
                            ]+[
                            'sys.path:'+repr(sys.path),
                            ]

class HitsView(DumaView):
    template_name='browse/hits.html'
    def custom_context(self):
        wsas_by_ws, phases, ivals = extract_hits_table_plot_data()
        self.context_alias(
                wsas_by_ws=wsas_by_ws,
                phases=phases,
                ivals=ivals,
                )

# this seems like maybe it should go in rvw, but since it shares all its
# mechanism with HitsView, it should live with the cross-ws stuff instead
class WsHitsView(DumaView):
    template_name='browse/hits.html'
    def custom_context(self):
        wsas_by_ws, phases, ivals = extract_hits_table_plot_data(wss=[self.ws])
        self.context_alias(
                wsas_by_ws=wsas_by_ws,
                phases=phases,
                ivals=ivals,
                )

def extract_hits_table_plot_data(wss=None):
    from browse.models import WsAnnotation, DispositionAudit
    discovery_order = WsAnnotation.discovery_order
    start_idx = discovery_order.index(WsAnnotation.indication_vals.REVIEWED_PREDICTION)
    inds_of_interest = WsAnnotation.discovery_order[start_idx:]
    disps = DispositionAudit.objects.filter(
                indication__in=inds_of_interest)
    from collections import defaultdict
    max_indidx_per_wsa = defaultdict(int)
    all_ind_per_wsa = defaultdict(set)
    for disp in disps:
        discovery_idx = WsAnnotation.discovery_order_index(disp.indication)
        max_indidx_per_wsa[disp.wsa_id] = max(max_indidx_per_wsa[disp.wsa_id],
                                           discovery_idx)
        all_ind_per_wsa[disp.wsa_id].add(disp.indication)
    from django.db.models import Q
    q = Q(pk__in=max_indidx_per_wsa.keys())
    if wss:
        q &= Q(ws_id__in=wss)
    wsas = WsAnnotation.objects.filter(q)
    wsas_by_ws = defaultdict(list)
    for wsa in wsas:
        wsas_by_ws[wsa.ws.name].append((wsa, max_indidx_per_wsa[wsa.id], all_ind_per_wsa[wsa.id]))
    for ws, wsas in wsas_by_ws.items():
        wsas.sort(key=lambda x:(-x[1], -x[0].indication))
    ivals = WsAnnotation.indication_vals
    choices = ivals.choices()
    inds_of_interest_names = [choices[x][1] for x in inds_of_interest]
    inds_of_interest_idxs = [WsAnnotation.discovery_order_index(x)
                             for x in inds_of_interest]
    # Order by latest review round.
    def order_fn(ws_and_wsas):
        from django.db.models import Min, Max
        from browse.models import Election
        ws_name = ws_and_wsas[0]
        return Election.objects.filter(ws__name=ws_name).aggregate(Max('due'))['due__max']
    # Note that in modern python versions, dicts will retain the order
    # you insert items into them.
    wsas_by_ws = {k:v for k,v in sorted(wsas_by_ws.items(), key=order_fn, reverse=True)}
    return (wsas_by_ws,
            list(zip(inds_of_interest_idxs, inds_of_interest_names, inds_of_interest)),
            ivals
           )

class SelectabilityFeaturePlot(DumaView):
    GET_parms = {
            # Feature to plot
            'featuresets':(list_of(str),None),
            # Which workspaces to include
            'ws': (list_of(str), []),
            # Which inidcation groups to compare
            'inds': (list_of(str), []),
            # should we use original molecules if this is a replacement
            'unreplaced': (boolean, False),
            }
    button_map = { 'calc': ['calc'] }
    template_name='browse/selectability_feature_plot.html'

    def make_calc_form(self, data):
        from algorithms.run_selectabilitymodel import workspace_choices
        ws_choices = workspace_choices()
        from dtk.selectability import AVAILABLE_FEATURESETS as featuresets
        feature_choices = [(cls.__name__, cls.name()) for cls in featuresets]
        from .models import Workspace
        ws = Workspace()
        ind_choices = ws.get_wsa_id_set_choices(retro=True)
        class MyForm(forms.Form):
            ws = forms.MultipleChoiceField(
                    label='Workspaces',
                    choices=ws_choices,
                    required=True,
                    initial=self.ws,
                    widget=forms.SelectMultiple(
                            attrs={'size':20}
                            ),
                    )
            inds = forms.MultipleChoiceField(
                    label='Indications',
                    choices=ind_choices,
                    initial=self.inds,
                    required=True,
                    widget=forms.SelectMultiple(
                            attrs={'size':15}
                            ),
                    )
            unreplaced = forms.BooleanField(
                        label="Use 'unreplaced' molecules",
                        required=False,
                        initial=self.unreplaced,
                    )

            featuresets = forms.MultipleChoiceField(
                    label='Feature Sets',
                    choices=feature_choices,
                    required=True,
                    initial=self.featuresets,
                    widget=forms.SelectMultiple(
                            attrs={'size':10}
                            ),
                    )
        return MyForm(data)

    def calc_post_valid(self):
        p = self.context['calc_form'].cleaned_data
        p['inds'] = ','.join(p['inds'])
        p['ws'] = ','.join(p['ws'])
        p['featuresets'] = ','.join(p['featuresets'])
        return HttpResponseRedirect(self.here_url(**p))

    def plot(self, xs_and_names, name):
        from dtk.plot import bar_histogram_overlay, PlotlyPlot,fig_legend
        xs_and_names = list(xs_and_names)
        x_data = [x for x, name in xs_and_names]
        names = [name for x, name in xs_and_names]
        if len(x_data) > 1:
            annot = [fig_legend(self._calc_pairwise_ks(x_data), -0.2)]
        else:
            annot = None
        traces, layout = bar_histogram_overlay(
                x_data=x_data,
                names=names,
                density=True,
                annotations=annot
                )
        layout['title'] = name
        plot = PlotlyPlot(
                data=traces,
                layout=layout
                )
        return plot
    def _calc_pairwise_ks(self,x_data):
        from scipy.stats import ks_2samp
        results = []
        for i in range(len(x_data)):
            for j in range(len(x_data)):
                if j >= i:
                    continue
                ks_result = ks_2samp(x_data[i], x_data[j])
                results.append(f"{i+1} v {j+1}: p={ks_result[1]:.1e}")
        return ["; ".join(results)]
    def custom_context(self):
        if not self.featuresets:
            return
        import dtk.selectability as select
        from browse.models import WsAnnotation
        # Find the feature.

        workspaces = Workspace.objects.filter(pk__in=self.ws)
        input_featureset_classes = [getattr(select, fs) for fs in self.featuresets if fs !='CMRankFeatureSet']
        input_featuresets = select.instantiate_featuresets(input_featureset_classes)

        id_features = [
                       select.IndicationGroupFeatureSet(i, self.unreplaced)
                       for i in self.inds
                      ]
        grps = WsAnnotation.all_groups()
        lkup = {x[0]:x[1] for x in grps}
        id_names = [lkup[i] for i in self.inds]

        # Review whatever WSAs we've selected
        wsa_source = select.WsaGroupSource(self.inds, self.unreplaced)

        if 'CMRankFeatureSet' in self.featuresets:
            input_featuresets += self._build_cmrank_fts(workspaces,wsa_source)

        featuresets = input_featuresets + id_features

        s = select.Selectability()
        data, errors = s.generate_data(wsa_source, workspaces, featuresets, self.inds, collect_errors=True)
        feature_names = sum([fs.feature_names() for fs in input_featuresets], [])

        N = len(feature_names)
        id_names_cnts=[]
        id_N = len(id_features)
        for j in range(id_N):
            id_names_cnts.append(sum([1 for row in data if row[N+j] == 1.0]))
        plots=[]
        for i, name in enumerate(feature_names):
            xs = []
            name_w_nan_cnt = []
            sample_cnt=0
            for j in range(id_N):
                xj = [row[i] for row in data if row[N+j] == 1.0]
                xs.append(xj)
                nan_cnt = sum([np.isnan(x) for x in xj])
                if nan_cnt:
                    id_name_cnt = id_names_cnts[j]-nan_cnt
                    end = f", NaN={nan_cnt})"
                else:
                    id_name_cnt = id_names_cnts[j]
                    end = f")"
                sample_cnt+=id_name_cnt
                name_w_nan_cnt.append(f"{id_names[j]} (n={id_name_cnt}{end}")
            if sample_cnt > 1:
                plot = self.plot(zip(xs, name_w_nan_cnt), name)
                if plot is not None:
                    plots.append((f'Plot {i}', plot))

        for error in errors:
            exc = f'<pre>{error[2]}</pre>'
            from django.utils.safestring import mark_safe
            msg = mark_safe(f'Failure: {error[0]}, {error[1].__class__.name()}{exc}')
            self.message(msg)

        self.context_alias(
                plotly_plots = plots,
                )

    def _build_cmrank_fts(self, wss,wsa_source):
        import dtk.selectability as select
        from runner.process_info import JobInfo
        from dtk.scores import Ranker
        from browse.models import Prescreen
        from collections import defaultdict
        wsas = [wsa for ws in wss for wsa in wsa_source.get_wsas(ws)]

        jrc_2_jid_2_rnkr = defaultdict(dict)
        jrc_2_wsa_2_jid = defaultdict(dict)

        prescreen_ids = [wsa.marked_prescreen_id for wsa in wsas]
        prescreens = Prescreen.objects.filter(pk__in=set(prescreen_ids))

        prescreens_jid_map = {p.id:p.source_list_job_ids() for p in prescreens}

        wsa2psid = dict(zip(wsas, prescreen_ids))
        for_nonps=set()
        mult_copies=set()
        for wsa in wsas:
            pscr_id = wsa2psid[wsa]
            if pscr_id == None:
                for_nonps.add(self._format_name(wsa))
                continue
            for jid in prescreens_jid_map[pscr_id]:
                bji = JobInfo.get_bound(wsa.ws.id,jid)
                cat = bji.get_data_catalog()
                codes = set(cat.get_codes('wsa','score'))
                jr = bji.job.role
                for code in codes:
                    jrc=jr+code
                    if jrc in jrc_2_wsa_2_jid and wsa in jrc_2_wsa_2_jid[jrc]:
                        mult_copies.add(f"{jrc} for " + self._format_name(wsa))
                        continue
                    jrc_2_wsa_2_jid[jrc][wsa]=jid
                    if jrc in jrc_2_jid_2_rnkr and jid in jrc_2_jid_2_rnkr[jrc]:
                        continue
                    jrc_2_jid_2_rnkr[jrc][jid]= Ranker(cat.get_ordering(code,True))
        if for_nonps:
            self.message("Ignoring non-prescreened wsas: " + ", ".join(for_nonps))
        if mult_copies:
            self.message(f"Multiple copies of the following jobs for the specificed molecules. Only the first instance is used. " + ", ".join(mult_copies))
        return [
                select.CMRankFeatureSet(jrc,
                                        jrc_2_jid_2_rnkr[jrc],
                                        jrc_2_wsa_2_jid[jrc]
                ) for jrc in jrc_2_jid_2_rnkr
               ]
    def _format_name(self, wsa):
        return f"{wsa.get_name(self.is_demo())} in WS:{wsa.ws.id}"

class SelectabilityModelView(DumaView):
    template_name='browse/selectabilitymodel.html'
    GET_parms={
            'wsa_ids':(list_of(int),None),
            'fold_idx':(int, 0),
            'indication':(int, None),
            }
    button_map={'wsaselect':['wsaselect'],
                'foldselect':['foldselect'],
                }

    def make_wsaselect_form(self, data):
        class MyForm(forms.Form):
            wsa_ids = forms.CharField(
                    label="WSA IDs to display",
                    widget=forms.Textarea(attrs={'rows':'4','cols':'20'}),
                    required=True,
                    initial='\n'.join([str(x) for x in self.wsa_ids]) if self.wsa_ids else ''
                    )
        return MyForm(data)

    def wsaselect_post_valid(self):
        p = self.context['wsaselect_form'].cleaned_data
        from algorithms.run_trgscrimp import parse_wsa_list
        p['wsa_ids'] = [str(x) for x in parse_wsa_list(p['wsa_ids'])]
        p['wsa_ids'] = ','.join(p['wsa_ids'])
        return HttpResponseRedirect(self.here_url(**p))

    def make_foldselect_form(self, data):
        class MyForm(forms.Form):
            fold_idx = forms.ChoiceField(
                    choices=((x, x) for x in range(self.num_folds)),
                    label="Fold",
                    required=True,
                    initial=self.fold_idx
                    )
        return MyForm(data)

    def foldselect_post_valid(self):
        p = self.context['foldselect_form'].cleaned_data
        self.base_qparms={}
        return HttpResponseRedirect(self.here_url(**p))

    def custom_setup(self):
        bji = JobInfo.get_bound(None, self.job)
        self.num_folds = bji.parms['kfold']

    def custom_context(self):
        bji = JobInfo.get_bound(None, self.job)
        if not self.wsa_ids:
            xval_data = bji.load_xval_data()
        else:
            from dtk.selectability import Selectability, WsaIdSource, generate_eval_stats
            model = bji.load_trained_model()
            wsa_src = WsaIdSource(self.wsa_ids)
            workspaces = wsa_src.workspaces()
            featuresets = model.featuresets
            xval_data = [generate_eval_stats(model, wsa_src, workspaces, featuresets, [])]

        # xval_data is a list, each an entry per fold
        # each entry contains {'eval_metrics, 'prob_vals', 'wsas'}
        # Within eval_metrics, you have a list of {'fvs', 'roc_curve', ...}

        tables = []

        fold_data = xval_data[self.fold_idx]
        # Pick the most general one.
        fvs_cols, fvs_selected = fold_data['fvs']
        fvs_wsa_ids = set(int(x[0]) for x in fvs_selected)

        wsa_ids = [wsa.id for wsa in fold_data['wsas'] if wsa.id in fvs_wsa_ids]
        prob_vals = [prob for prob, wsa in zip(fold_data['prob_vals'],fold_data['wsas']) if wsa.id in fvs_wsa_ids]
        wsas = WsAnnotation.objects.filter(pk__in=wsa_ids).select_related('ws', 'agent').prefetch_related('dispositionaudit_set', 'replacement_for', 'replacements')
        wsas = WsAnnotation.prefetch_agent_attributes(wsas)
        wsa_map = {x.id:x for x in wsas}


        wsa_data = {}
        for wsa_id, prob in zip(wsa_ids, prob_vals):
            wsa = wsa_map[wsa_id]
            is_replacement = wsa.replacement_for.count() > 0
            is_replaced = wsa.replacements.count() > 0
            extra = ''
            # The replacement/replacee logic matches what we do in
            # dtk/selectability
            if is_replacement:
                max_ind = wsa.indication_vals.UNCLASSIFIED
                extra = ' (Rplc)'
            elif is_replaced:
                max_ind = wsa.indication_vals.HIT
                extra = ' (Orig)'
            else:
                max_ind = wsa.max_discovery_indication()
            max_ind_idx = wsa.discovery_order_index(max_ind)
            max_ind_label = f"({max_ind_idx}) {wsa.indication_vals.get('label', max_ind)}{extra}"
            wsa_data[wsa.id] = {
                    'name': wsa.agent.canonical,
                    'url': wsa.drug_url(),
                    'prob': prob,
                    'max_ind': max_ind_label,
                    }

        import json
        fvs_cols += ['prob']
        fvs_cols += ['Max Indication']
        fvs_cols = json.dumps([{'title': x} for x in fvs_cols])
        fvs_selected = json.dumps([x + [wsa_data[x[0]]['prob'], wsa_data[x[0]]['max_ind']] for x in fvs_selected])
        tables.append((
            0,
            "Title",
            #str(eval_metric['evalset']),
            fvs_cols,
            fvs_selected,
            json.dumps(wsa_data),
            ))

        # Table for each WSA, showing feature
        self.context_alias(
                tables=tables,
                )

class MetricScatterPlot(DumaView):
    template_name='browse/metric_scatter_plot.html'
    GET_parms={
            'roles':(list_of(str),None),
            'scoresets':(list_of(str),None),
            'x_metric':(str,'SigmaOfRank1000Condensed'),
            'y_metric':(str,'AUR'),
            'plot_type':(str,None),
            }
    button_map={
            'scoreplot':['scores'],
            'metricplot':['scores','metrics'],
            }
    def custom_setup(self):
        self.extra_metrics = ['top5','top50','top100','top500','sel500']
        # now supply defaults for use in forms
        if self.scoresets is None:
            # default to latest refresh workflow in each active workspace
            from dtk.score_calibration import get_recent_active_scoresets
            self.scoresets = [
                    str(ss.id)
                    for ss in get_recent_active_scoresets('DNChBX_ki-moa.v24')
                    ]
            self.scoresets.sort(key=int)
        from dtk.score_calibration import ScoreCalibrator
        self.sc = ScoreCalibrator()
    def custom_context(self):
        from dtk.enrichment import MetricProcessor
        self.mp = MetricProcessor()
        if self.plot_type == 'metric':
            self.build_metric_plot()
        elif self.plot_type == 'overlay':
            self.build_overlay_plot()
        elif self.plot_type == 'log_overlay':
            self.build_overlay_plot(log_scale=True)
        elif self.plot_type == 'check_scores':
            self.build_check_scores_plot()
        elif self.plot_type is not None:
            raise NotImplementedError(f"unknown plot type: '{self.plot_type}'")
    def get_role_job_ids(self):
        from browse.models import ScoreSetJob
        all_job_ids = ScoreSetJob.objects.filter(
                scoreset_id__in=self.scoresets,
                ).values_list('job_id',flat=True)
        from runner.models import Process
        return Process.objects.filter(
                pk__in=all_job_ids,
                role__in=self.roles,
                ).values_list('id',flat=True)
    def get_calibrated(self,bji,code):
        emi = self.mp.get_emi(bji,code)
        from dtk.moa import is_moa_score
        is_moa = is_moa_score(self.mp.get_ws_wsaids(bji.ws.id))
        descaled = [x[1] for x in bji.remove_workspace_scaling(
                code,
                emi.get_labeled_score_vector(),
                )]
        return self.sc.calibrate(bji.job.role,code,is_moa,descaled)
    def build_overlay_plot(self,log_scale=False):
        plot_data = []
        for job_id in self.get_role_job_ids():
            bji = JobInfo.get_bound(None,job_id)
            cat = bji.get_data_catalog()
            codes = list(cat.get_codes('wsa','efficacy'))
            for code in codes:
                #if self.mp.compute('AUR',bji.ws,bji.job,code) > 0.7:
                #    continue
                vec = self.get_calibrated(bji,code)
                if log_scale:
                    vec = [-math.log(1-x,10) for x in vec]
                plot_data.append(dict(
                        y=vec,
                        name=f'{bji.job.role} {code} {bji.ws.id} {bji.job.id}',
                        ))
        from dtk.plot import PlotlyPlot
        plot = PlotlyPlot(
                plot_data,
                {
                    'width':1500,
                    'hovermode':'closest',
                    'hoverlabel':{'namelength':-1},
                    'title':'Calibrated Score Compare',
                },
                )
        self.context['plotly_plots']=[('Calibration Overlay',plot)]
    def build_metric_plot(self):
        def get_metric_of_score(metric,bji,code):
            if metric in self.extra_metrics:
                if metric.startswith('top'):
                    cutoff = int(metric[3:])
                    vec = self.get_calibrated(bji,code)
                    vec.sort(reverse=True)
                    return vec[cutoff]
                if metric.startswith('sel'):
                    cutoff = int(metric[3:])
                    vec = self.get_calibrated(bji,code)
                    vec.sort(reverse=True)
                    return vec[0]*(vec[0]-vec[cutoff])
                else:
                    raise NotImplementedError(f"unknown metric '{metric}'")
            else:
                return self.mp.compute(metric,bji.ws,bji.job,code)
        point_data=[]
        for job_id in self.get_role_job_ids():
            bji = JobInfo.get_bound(None,job_id)
            cat = bji.get_data_catalog()
            codes = list(cat.get_codes('wsa','efficacy'))
            for code in codes:
                point_data.append(dict(
                        x = get_metric_of_score(self.x_metric,bji,code),
                        y = get_metric_of_score(self.y_metric,bji,code),
                        ws = bji.ws,
                        job_id = bji.job.id,
                        role = bji.job.role,
                        code = code,
                        ))
        self.context['point_count']=len(point_data)
        from dtk.plot import scatter2d
        plot = scatter2d(
                self.x_metric,
                self.y_metric,
                [(d['x'],d['y']) for d in point_data],
                refline=False,
                text=[
                        f'{d["ws"].id} {d["role"]} {d["code"]}'
                        for d in point_data
                        ],
                ids=['anyjobpage',[
                        [d["ws"].id,d["job_id"]]
                        for d in point_data
                        ]],
                )
        self.context['plotly_plots']=[('Metric Compare',plot)]
    def build_check_scores_plot(self):
        fn = PathHelper.repos_root \
                + 'experiments/score_calibration/check_scores.out'
        point_data = []
        import re
        for line in open(fn).readlines():
            m = re.match(r'\((.*), (.*), (.*), (.*), \'(.*)\'\)\n$',line)
            d = dict(
                    x=float(m.group(1)),
                    y=float(m.group(2)),
                    ws_id=int(m.group(3)),
                    job_id=int(m.group(4)),
                    code=m.group(5),
                    )
            point_data.append(d)
        from dtk.plot import scatter2d
        plot = scatter2d(
                'SigmaOfRank500',
                'unconformity',
                [(d['x'],d['y']) for d in point_data],
                refline=False,
                text=[
                        f'{d["ws_id"]} {d["job_id"]} {d["code"]}'
                        for d in point_data
                        ],
                ids=['anyjobpage',[
                        [d["ws_id"],d["job_id"]]
                        for d in point_data
                        ]],
                )
        self.context['plotly_plots']=[('Conformity',plot)]
    def make_scores_form(self, data):
        # XXX add another field to select moa/non-moa scores?
        # XXX filter by role/code combos instead?
        role_choices = [(x,x) for x in sorted(self.sc.all_roles())]
        class MyForm(forms.Form):
            scoresets = forms.CharField(
                        label='Source Score Sets',
                        initial=' '.join(self.scoresets),
                        widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
                        )
            roles = forms.MultipleChoiceField(
                        choices=role_choices,
                        label='Include these roles',
                        required=True,
                        widget=forms.SelectMultiple(
                                attrs={'size':6}
                                ),
                        initial=self.roles or [x[0] for x in role_choices],
                        )
        return MyForm(data)
    def make_metrics_form(self, data):
        from dtk.enrichment import EnrichmentMetric
        metric_choices = [
                (x,x)
                for x in self.extra_metrics + EnrichmentMetric.get_all_names()
                ]
        class MyForm(forms.Form):
            x_metric = forms.ChoiceField(
                        choices=metric_choices,
                        label='X-axis metric',
                        required=True,
                        initial=self.x_metric,
                        )
            y_metric = forms.ChoiceField(
                        choices=metric_choices,
                        label='Y-axis metric',
                        required=True,
                        initial=self.y_metric,
                        )
        return MyForm(data)
    def get_score_qparms(self):
        p = self.context['scores_form'].cleaned_data
        qp = dict(p)
        qp['roles']=','.join(p['roles'])
        qp['scoresets']=','.join(p['scoresets'].split())
        return qp
    def scoreplot_post_valid(self):
        qp = self.get_score_qparms()
        qp['plot_type'] = 'overlay'
        return HttpResponseRedirect(self.here_url(**qp))
    def metricplot_post_valid(self):
        qp = self.get_score_qparms()
        qp['plot_type'] = 'metric'
        qp.update(self.context['metrics_form'].cleaned_data)
        return HttpResponseRedirect(self.here_url(**qp))

class PathwaysView(DumaView):
    template_name='browse/pathways_view.html'
    GET_parms={
            'glf_jobs':(list_of(int),None),
            'refresh_jobs':(list_of(int),None),
            'condense_table':(boolean, False),
            'dedupe_table':(boolean, False),
            }
    button_map={'calc':['calc']}

    def make_calc_form(self, data):
        MAX_SZ = 15
        glf_choices = self.ws.get_prev_job_choices('glf')
        glf_sz = min(len(glf_choices), MAX_SZ)


        refresh_choices = []
        wf_choices = self.ws.get_prev_job_choices('wf')
        for jid, desc in wf_choices:
            bji = JobInfo.get_bound(self.ws, jid)
            if 'RefreshFlow' not in bji.job.name:
                continue

            refresh_choices.append((f'{jid}', f'RefreshWF {jid} {bji.job.started}'))

        class MyForm(forms.Form):
            glf_jobs = forms.MultipleChoiceField(
                        choices=glf_choices,
                        label='Include these GLF runs',
                        required=False,
                        widget=forms.SelectMultiple(
                                attrs={'size':glf_sz}
                                ),
                        initial=self.glf_jobs,
                        )
            refresh_jobs = forms.MultipleChoiceField(
                        choices=refresh_choices,
                        required=False,
                        label='Include all GLF from',
                        widget=forms.SelectMultiple(
                                attrs={'size':glf_sz}
                                ),
                        initial=self.refresh_jobs,
                        )
            condense_table = forms.BooleanField(
                label="Condense table scores",
                required=False,
                initial=self.condense_table,
                help_text='Condenses scores hierarchically (experimental)',
            )
            dedupe_table = forms.BooleanField(
                label="Dedupe table scores",
                required=False,
                initial=self.dedupe_table,
                help_text='Dedupes identical pathway sets in table (also required condense table currently)',
            )
        return MyForm(data)

    def calc_post_valid(self):
        p = self.context['calc_form'].cleaned_data
        glf_jobs = set()
        if 'glf_jobs' in p:
            glf_jobs.update(p['glf_jobs'])
        if 'refresh_jobs' in p:
            glf_choices = self.ws.get_prev_job_choices('glf')
            glf_jids = {x[0] for x in glf_choices}
            for jid in p['refresh_jobs']:
                ss = ProteinScoreView.get_scoreset_from_refreshwf(ws=self.ws, jid=jid)
                for name, ssjid in ss.job_type_to_id_map().items():
                    if ssjid in glf_jids:
                        glf_jobs.add(str(ssjid))
            p['refresh_jobs'] = ','.join(p['refresh_jobs'])

        if glf_jobs:
            p['glf_jobs'] = ','.join(glf_jobs)

        return HttpResponseRedirect(self.here_url(**p))
    def _reactome_id_to_name(self):
        from dtk.gene_sets import get_pathway_id_name_map
        return get_pathway_id_name_map()
    def pathway_link(self, id):
        from dtk.html import link
        if id.startswith('REACTOME'):
            return link(id, self.pathway_url(id))
        else:
            name = self._rct_id_to_name.get(id, id)
            return link(name, self.pathway_url(id))
    def custom_context(self):
        if not self.glf_jobs:
            return

        from collections import defaultdict
        table_data = defaultdict(lambda: defaultdict(dict))
        self._rct_id_to_name = self._reactome_id_to_name()
        from dtk.url import pathway_url_factory
        self.pathway_url = pathway_url_factory()

        glf_names = []
        for glf_jid in self.glf_jobs:
            bji = JobInfo.get_bound(self.ws, glf_jid)
            name = '%s %d' % (bji.job.role, glf_jid)
            glf_names.append(name)
            dc = bji.get_data_catalog()
            gen = dc.get_feature_vectors('wFEBE', 'febeQ')[1]
            import numpy as np
            for pathway, (wfebe, febeq) in gen:
                pathway_name = pathway
                table_data['wfebe'][pathway]['pathway'] = pathway_name
                table_data['wfebe'][pathway][name] = wfebe
                table_data['febeq'][pathway]['pathway'] = pathway_name
                table_data['febeq'][pathway][name] = -np.log10(febeq)
        glf_names.sort()
        import numpy as np
        plotly_plots = []
        from dtk.plot import PlotlyPlot
        for table_name, pathways in six.iteritems(table_data):
            plot_data = []
            for pathway, data in list(pathways.items()):
                # We sort so that values is in sorted order according to glf
                # name.
                values = [value for name, value in sorted(six.iteritems(data))
                          if name != 'pathway']
                data['mean'] = np.mean(values)
                data['median'] = np.median(values)
                data['values'] = values

                pathway_name = self._rct_id_to_name.get(pathway, pathway)

                plot_data.append((data['mean'], pathway_name, values))
            plot_data.sort(reverse=True)
            plot_data = plot_data[:20]
            plot = PlotlyPlot([{
                'x': [x[1] for x in plot_data],
                'y': [x[0] for x in plot_data],
                'type': 'bar'
                }], {
                    'title': table_name,
                    'width': 1000,
                    'height': 1000,
                    'margin': {
                        'b': 300,
                        'l': 200,
                        },
                    'xaxis': {
                        'tickangle': -35,
                        },
                    'yaxis': {
                        'title': "Mean " + table_name,
                        }
                })
            boxplot = PlotlyPlot([{
                    'y': data[2],
                    'type': 'box',
                    'name': data[1],
                    'boxpoints': 'all',
                } for data in plot_data], {
                    'title': "Box " + table_name,
                    'width': 1000,
                    'height': 1000,
                    'showlegend': False,
                }

                )

            from dtk.plot import plotly_heatmap
            data_array = np.array([data[2] for data in plot_data])
            data_array = np.transpose(data_array)
            col_labels = [data[1] for data in plot_data]
            row_labels = glf_names
            heatmap = plotly_heatmap(data_array, row_labels=row_labels, col_labels=col_labels,
                    width=1600, height=1200,
                    )

            plotly_plots.append((table_name, plot))
            plotly_plots.append((table_name + 'box', boxplot))
            plotly_plots.append((table_name + 'heatmap', heatmap))
        
        def fmt_name(x):
            name = self.pathway_link(x['pathway'])
            if x.get('subsumes', None):
                cover_text = mark_safe(f'<a>+{len(x["subsumes"])}</a>')
                covers_names = [self._rct_id_to_name.get(x, x) for x in x["subsumes"]]
                from dtk.html import hover 
                cover = hover(cover_text, '\n'.join(sorted(covers_names)))
                name = f'{name} ({cover})'
            return mark_safe(name)

        cols = [
                Table.Column('Pathway', extract=fmt_name),
                ]
        cols += [
                Table.Column('Mean', idx='mean',
                    cell_fmt=lambda x:"%0.3f"%x,
                    ),
                Table.Column('Median', idx='median',
                    cell_fmt=lambda x:"%0.3f"%x,
                    ),
                ]
        for glf_name in glf_names:
            cols += [
                    Table.Column(glf_name, idx=glf_name,
                        cell_fmt=lambda x:"%0.3f"%x,
                        ),
                    ]

        wfebe_rows = table_data['wfebe'].values()
        if self.condense_table:
            # Condensing has two parts, hierarchical condensing and deduping.
            # Hierarchical finds pathways with high scores and collapses their ancestors and descendants.
            # Deduping finds pathways with identical prot sets and collapses all-but-one copy.
            #
            # Combining them is a little bit tricky, as e.g. some-but-not-all of a given set of identical pathways
            # might have been suppressed via condensing.
            from dtk.scores import condense_scores
            from dtk.gene_sets import ROOT_ID, get_pathway_data, make_dedupe_map
            import networkx as nx
            from dtk.data import MultiMap
            protsets, pathways_data, hier = get_pathway_data()


            g = nx.DiGraph()
            for node, children in hier.items():
                for child in children:
                    g.add_edge(node, child)

            condensed, node_subsumes = condense_scores(ROOT_ID, g.predecessors, g.successors, ((x['pathway'], x['mean']) for x in wfebe_rows if x['pathway'] in g))

            kept = set(x[0] for x in condensed)

            if self.dedupe_table:
                dedupe_map = make_dedupe_map(protsets, hier, siblings_only=False)
                dedupe_mm = MultiMap(dedupe_map.items())
                for group_id, pws in dedupe_mm.rev_map().items():
                    group_kept_pws = pws & kept

                    if group_kept_pws:
                        canon_pw = next(iter(group_kept_pws))
                    else:
                        # Not much to do here...
                        continue

                    if canon_pw not in node_subsumes:
                        node_subsumes[canon_pw] = set()
                    
                    dupe_pws = pws - {canon_pw}

                    cur_subs = set(dupe_pws)
                    # Find anything subsumed by any of the pathways we're about to remove.
                    for dupe_pw in dupe_pws:
                        cur_subs.update(node_subsumes.get(dupe_pw, set()))

                    # Add them all to the canonical pathway subsume.
                    node_subsumes[canon_pw].update(cur_subs)

                    kept -= dupe_pws

            def add_subsumes(x):
                if x['pathway'] in node_subsumes:
                    x['subsumes'] = node_subsumes[x['pathway']]
                return x
            
            wfebe_rows = [add_subsumes(x) for x in wfebe_rows if x['pathway'] in kept]

        febeq_rows = table_data['febeq'].values()
        tables = []
        tables.append(('wFEBE', Table(wfebe_rows, cols)))
        tables.append(('FEBE Q (-log10)', Table(febeq_rows, cols)))
        self.context_alias(
                tables=tables,
                plotly_plots=plotly_plots,
                )


class SuitabilityView(DumaView):
    GET_parms={
            'ws':(list_of(str),[]),
            'disp_ws':(list_of(str),[]),
            }
    template_name = 'browse/suitability.html'

    button_map = {'calc': ['calc']}

    def make_calc_form(self, data):
        import dtk.retrospective as retro
        workspaces = retro.filter_workspaces()
        workspaces = sorted(workspaces, key=retro.ws_review_date_key)
        ws_choices = [(ws.id, ws.name) for ws in workspaces if ws.active]

        from browse.models import Workspace
        disp_workspaces = Workspace.objects.all().order_by('name')
        dispws_choices = [(ws.id, ws.name) for ws in disp_workspaces]

        class MyForm(forms.Form):
            ws = forms.MultipleChoiceField(
                    label='Weighting Workspaces',
                    choices=ws_choices,
                    required=True,
                    initial=self.ws,
                    widget=forms.SelectMultiple(
                            attrs={'size':20}
                            ),
                    )
            disp_ws = forms.MultipleChoiceField(
                    label='Display Workspaces',
                    choices=dispws_choices,
                    required=True,
                    initial=self.disp_ws,
                    widget=forms.SelectMultiple(
                            attrs={'size':20}
                            ),
                    )
        return MyForm(data)

    def calc_post_valid(self):
        p = self.context['calc_form'].cleaned_data
        p['ws'] = ','.join(p['ws'])
        p['disp_ws'] = ','.join(p['disp_ws'])
        return HttpResponseRedirect(self.here_url(**p))

    def get_weights(self, workspaces, keys):
        def is_defus_faers(out):
            return 'defus' in out or 'weightedzscore' in out or 'drtarget' in out or 'drstruct' in out or 'wzs' in out

        gwas_types = set(['gpath', 'esga', 'gwasig'])
        dgn_types = set(['dgn', 'dgns'])

        def grouper(x):
            if is_defus_faers(x[0]):
                return 'faers'
            if len(x) >= 2 and x[1] == 'otarg':
                return 'otarg'
            elif x[0] in gwas_types:
                return 'gwas'
            elif x[0] in dgn_types:
                return 'dgn'
            else:
                return x[0]

        from collections import defaultdict
        import dtk.retrospective as retro
        fns, cats = retro.make_category_funcs(
                workspaces,
                retro.unreplaced_selected_mols,
                lambda *args: retro.mol_score_imps(*args, score_imp_group_fn=grouper),
                )

        key_sums = defaultdict(int)
        for fn, cat in zip(fns, cats):
            if cat in keys:
                for ws in workspaces:
                    key_sums[cat] += fn(ws)
        import numpy as np
        norm_factor = np.sum(list(key_sums.values()))
        for cat in list(key_sums.keys()):
            key_sums[cat] /= norm_factor

        return key_sums

    def custom_context(self):
        if not self.ws or not self.disp_ws:
            return

        from dtk.ws_data_status import compute_suitabilities

        types, raw_scores, norm_scores = compute_suitabilities(self.ws, self.disp_ws)

        table_columns = [
                {'data': 'name', 'title': 'Name'},
                {'data': 'short_name', 'title': 'Short Name'},
                {'data': 'category', 'title': 'Category'},
                ]
        table_columns += [{'data': 'Combined', 'title': 'Combined'}]
        from dtk.ws_data_status import DataStatus
        for typename, scoreattrname, weightname in types:
            Class = DataStatus.lookup(typename)
            scorename = getattr(Class, scoreattrname)
            table_columns.append({
                'data': scorename,
                'title': scorename,
                })


        self.context_alias(
                norm_scores=list(norm_scores.values()),
                raw_scores=list(raw_scores.values()),
                table_columns=table_columns,
                )


from tools import Enum
class RetrospectiveView(DumaView):
    scoreimp_types = Enum([], [
        ('DROP_SCORETYPE',),
        ('NO_GROUPING',),
        ('SOURCE_ROLE',),
        ('OUTPUT_ROLE',),
        ])
    reviewed_types = Enum([], [
        ('FURTHEST_INDICATION',),
        ('DEMERIT',),
        ])
    button_map={
            'calc': ['calc'],
            'scoreimp':['scoreimp'],
            'reviewed':['reviewed'],
            'molecules':['molecules'],
            'targetcmp':['targetcmp'],
            }
    GET_parms={
            'plot':(str,None),
            'scoreimp_type':(int,scoreimp_types.DROP_SCORETYPE),
            'reviewed_type':(int,reviewed_types.FURTHEST_INDICATION),
            'mol_or_prot':(str,'mol'),
            'tc_type':(str,'tc'),
            'ws':(list_of(str),None),
            }
    @property
    def template_name(self):
        if not self.plot:
            return 'browse/retrospective.html'
        elif self.plot == 'molecules':
            return 'browse/_retro_tables.html'
        else:
            return 'browse/_retro_plot.html'

    def make_calc_form(self, data):
        import dtk.retrospective as retro
        workspaces = retro.filter_workspaces()
        workspaces = sorted(workspaces, key=retro.ws_review_date_key)
        ws_choices = [(ws.id, ws.name) for ws in workspaces]
        class MyForm(forms.Form):
            ws = forms.MultipleChoiceField(
                    label='Workspaces',
                    choices=ws_choices,
                    required=True,
                    initial=self.ws,
                    widget=forms.SelectMultiple(
                            attrs={'size':20}
                            ),
                    )
        return MyForm(data)

    def calc_post_valid(self):
        p = self.context['calc_form'].cleaned_data
        p['ws'] = ','.join(p['ws'])
        return HttpResponseRedirect(self.here_url(**p))

    def prescreened_plot(self):
        import dtk.retrospective as retro
        workspaces = retro.filter_workspaces(self.ws)
        demerit_labels, demerit_cats = retro.demerit_labels_and_cats()
        fns, _ = retro.make_category_funcs(
                workspaces,
                retro.prescreened_mols,
                retro.mol_demerits,
                demerit_cats,
                )

        data_order_key=lambda entry: -sum(entry['y']) if entry['name'] not in ('Hits', 'In Review') else -1e99

        return retro.cross_ws_plot(
                workspaces,
                order_key=retro.ws_review_date_key,
                y_keys=fns,
                y_names=demerit_labels,
                title='Molecules Prescreened',
                x_title='Disease (by date)',
                y_title='Molecules Prescreened',
                data_order_key=data_order_key,
                )


    def make_reviewed_form(self, data):
        choices = self.reviewed_types.choices()
        class MyForm(forms.Form):
            reviewed_type = forms.ChoiceField(
                        choices=choices,
                        label='Display:',
                        initial=self.reviewed_type,
                        )
        return MyForm(data)

    def reviewed_post_valid(self):
        p = self.context['reviewed_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))

    def reviewed_plot(self):
        import dtk.retrospective as retro
        workspaces = retro.filter_workspaces(self.ws)
        if self.reviewed_type == self.reviewed_types.FURTHEST_INDICATION:
            progress_cats = WsAnnotation.discovery_order
            names = [WsAnnotation.indication_vals.get('label', x) for x in progress_cats]
            fn = retro.mol_progress
            data_order_key = None
        else:
            names, progress_cats = retro.demerit_labels_and_cats()
            fn = retro.mol_demerits
            data_order_key=lambda entry: -sum(entry['y']) if entry['name'] not in ('Hits', 'In Review') else -1e99

        fns, _ = retro.make_category_funcs(
                workspaces,
                retro.reviewed_mols,
                fn,
                progress_cats,
                )
        return retro.cross_ws_plot(
                workspaces,
                order_key=retro.ws_review_date_key,
                y_keys=reversed(fns),
                y_names=reversed(names),
                title='Molecules Reviewed',
                x_title='Disease (by date)',
                y_title='Molecules Reviewed',
                data_order_key=data_order_key,
                )

    def make_molecules_form(self, data):
        mt_choices = (
                ('mol', 'Molecule'),
                ('prot', 'Protein')
                )

        class MyForm(forms.Form):
            mol_or_prot = forms.ChoiceField(
                        choices=mt_choices,
                        label='Mol/Prot:',
                        initial=self.mol_or_prot,
                        )
        return MyForm(data)

    def molecules_post_valid(self):
        p = self.context['molecules_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))

    def make_targetcmp_form(self, data):
        tc_choices = (
                ('dir_mds', 'Direct MDS'),
                ('dir_hm', 'Direct Heatmap'),
                ('ind_mds', 'Indirect MDS'),
                ('ind_hm', 'Indirect Heatmap'),
                )

        class MyForm(forms.Form):
            tc_type = forms.ChoiceField(
                        choices=tc_choices,
                        label='Plot type',
                        initial=self.tc_type,
                        )
        return MyForm(data)

    def targetcmp_post_valid(self):
        p = self.context['targetcmp_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))

    def _load_agents_mm(self):
        import dtk.retrospective as retro
        self.ws_agent_to_wsa = {}
        self.wsa_max_ind = {}
        agent_ws_pairs = []
        for ws in self.workspaces:
            ws_agents = set()
            wsas = list(retro.reviewed_mols(ws).prefetch_related('dispositionaudit_set'))
            for wsa in wsas:
                agent = wsa.agent.id
                ws_agents.add(agent)
                self.ws_agent_to_wsa[(ws.id, agent)] = wsa.id
                self.wsa_max_ind[wsa.id] = wsa.max_discovery_indication()

            agent_ws_pairs.extend((agent, ws.id) for agent in ws_agents)
        from dtk.data import MultiMap
        self.agents_mm = MultiMap(agent_ws_pairs).fwd_map()

    def _load_prot_to_ws_wsa(self, min_idx=None):
        from browse.models import WsAnnotation, VersionDefault
        from dtk.prot_map import DpiMapping, AgentTargetCache
        global_defaults = VersionDefault.get_defaults(None)
        dpi = DpiMapping(global_defaults['DpiDataset'])
        atc = AgentTargetCache(
                mapping=dpi,
                agent_ids=self.agent_map.keys(),
                dpi_thresh = global_defaults['DpiThreshold'],
                )
        prot_ws_wsas = []
        for agent_id, ws_list in self.agents_mm.items():
            for drug_key, prot, ev, dr in atc.raw_info_for_agent(agent_id):
                for ws_id in ws_list:
                    wsa_id = self.ws_agent_to_wsa[(ws_id, agent_id)]
                    max_ind = self.wsa_max_ind[wsa_id]
        ### The mol_prot function may be able to take better advantage of this
        ### it was added to support the targetcmp page (which came after)
                    max_ind_idx = WsAnnotation.discovery_order_index(max_ind)
                    if min_idx is None or max_ind_idx >= min_idx:
                        prot_ws_wsas.append((prot, (ws_id, wsa_id, max_ind)))
        from dtk.data import MultiMap
        self.prot_to_ws_wsa = MultiMap(prot_ws_wsas).fwd_map()

    def _load_target_data(self):
        import dtk.retrospective as retro
        self.workspaces = retro.filter_workspaces(self.ws)
        self._load_agents_mm()
        from drugs.models import Drug
        agent_objs = Drug.objects.filter(pk__in=self.agents_mm.keys())
        self.agent_map = {agent.id: agent for agent in agent_objs}
    def targetcmp_plot(self):
        self._load_target_data()
        ws_to_prot = {}
        ws_id_to_label={}
        ws_to_prot = {}
        ws_id_to_label = {ws.id:ws.get_short_name() for ws in self.workspaces}
        from browse.models import WsAnnotation
        ivals = WsAnnotation.indication_vals
# TODO would like to make this min indication a setting from the UI
        min_idx = WsAnnotation.discovery_order_index(ivals.HIT)
        self._load_prot_to_ws_wsa(min_idx=min_idx)
        for prot, data in self.prot_to_ws_wsa.items():
            for ws, wsa, max_ind in data:
                if ws not in ws_to_prot:
                    ws_to_prot[ws] = set()
                ws_to_prot[ws].add(prot)
        if self.tc_type.startswith("ind"):
            from dtk.prot_map import PpiMapping
            from browse.models import VersionDefault
            global_defaults = VersionDefault.get_defaults(None)
            pm = PpiMapping(global_defaults['PpiDataset'])
            new_ws_to_prot={}
            for ws,prots in ws_to_prot.items():
                new_ws_to_prot[ws] = set(
                            [x[1] for x in
                             pm.get_ppi_info_for_keys(prots,
                                 global_defaults['PpiThreshold']
                             )
                            ])
            ws_to_prot = new_ws_to_prot
            title = "Indirect target similarity"
        else:
            title = "Direct target similarity"
        from dtk.similarity import build_mol_prot_sim_matrix
        sm = build_mol_prot_sim_matrix(ws_to_prot)
        if self.tc_type.endswith("hm"):
            from dtk.plot import plotly_heatmap
            names = [ws_id_to_label[i]
                      for i in sm.row_keys
                     ]
            return plotly_heatmap(
                    sm.matrix,
                    names,
                    col_labels=names,
                    color_zero_centered = True,
                    Title = title
                  )
        else:
            sm.mds()
            from dtk.plot import scatter2d, Color
            return scatter2d(
                    'MDS Axis 1',
                    'MDS Axis 2',
                    sm.mds_matrix,
                    text=[ws_id_to_label[i]
                          for i in sm.row_keys
                         ],
                    title=title,
                    refline=False,
                    textposition='top center'
                    )

    def mols_and_targets(self):
        self._load_target_data()

        from browse.models import WsAnnotation
        ivals = WsAnnotation.indication_vals
        # The code below assumes this is ordered from earliest to latest.
        inds_of_interest = [
                ivals.INITIAL_PREDICTION,
                ivals.REVIEWED_PREDICTION,
                ivals.HIT
                ]
        idxs_of_interest = [WsAnnotation.discovery_order_index(ioi) for ioi in inds_of_interest]
        ind_labels = [
                '<span class="ini">Ini</span>',
                '<span class="rvw">Rvw</span>',
                '<span class="hit">Hit</span>',
                ]

        if self.mol_or_prot == 'prot':
            self._load_prot_to_ws_wsa()
            prot_to_best_ws_wsa = {}
            for prot, data in self.prot_to_ws_wsa.items():
                ws_max_ind = {}
                ws_max_wsa = {}
                for ws, wsa, max_ind in data:
                    if max_ind > ws_max_ind.get(ws, 0):
                        ws_max_ind[ws] = max_ind
                        ws_max_wsa[ws] = wsa
                prot_to_best_ws_wsa[prot] = ws_max_wsa
            # The final data for prots and mols is stored differently, so
            # we plaster over those differences with the lambdas below.
            output_col = prot_to_best_ws_wsa
            get_wsa = lambda ws_id, item_id, ws_set: ws_set[ws_id]
            from browse.models import Protein
            u2g = Protein.get_uniprot_gene_map()
            name = lambda item_id: u2g.get(item_id, f'({item_id})')
        else:
            output_col = self.agents_mm
            get_wsa = lambda ws_id, item_id, ws_set: self.ws_agent_to_wsa[(ws_id, item_id)]
            name = lambda item_id: self.agent_map[item_id].canonical

        from collections import defaultdict
        data = []
        for item_id, ws_set in output_col.items():
            if len(ws_set) <= 1:
                continue
            counts = defaultdict(int)
            row = []
            for ws in self.workspaces:
                if ws.id not in ws_set:
                    row.append('')
                    continue
                wsa_id = get_wsa(ws.id, item_id, ws_set)
                max_ind = self.wsa_max_ind[wsa_id]
                idx = WsAnnotation.discovery_order_index(max_ind)
                label = ''
                for i, targ_idx in enumerate(idxs_of_interest):
                    if idx >= targ_idx:
                        counts[inds_of_interest[i]] += 1
                        label = ind_labels[i]

                url = ws.reverse('moldata:annotate', wsa_id)
                link = f'<a href="{url}">{label}</a>'
                row.append(link)

            sum_row = [
                    name(item_id),
                    counts[ivals.INITIAL_PREDICTION],
                    counts[ivals.REVIEWED_PREDICTION],
                    counts[ivals.HIT],
                    ]
            row = sum_row + row
            data.append(row)

        cols = ['Name', 'Initial', 'Reviewed', 'Hit'] + [ws.get_short_name() for ws in self.workspaces]

        return cols, data


    def scoreimp_plot(self):
        import dtk.retrospective as retro
        workspaces = retro.filter_workspaces(self.ws)

        def is_defus_faers(out):
            return 'defus' in out or 'weightedzscore' in out or 'drtarget' in out or 'drstruct' in out or 'wzs' in out

        types = self.scoreimp_types
        if self.scoreimp_type == types.DROP_SCORETYPE:
            grouper = lambda x: '_'.join(x[:-1])
        elif self.scoreimp_type == types.NO_GROUPING:
            grouper = lambda x: '_'.join(x)
        elif self.scoreimp_type == types.SOURCE_ROLE:
            grouper = lambda x: x[0] if not is_defus_faers(x[0]) else 'faers'
        elif self.scoreimp_type == types.OUTPUT_ROLE:
            def grouper(x):
                out = x[-2]
                if is_defus_faers(out):
                    out = 'defus'
                return out
        else:
            assert False, "Implement %s" % self.scoreimp_type

        fns, cats = retro.make_category_funcs(
                workspaces,
                retro.unreplaced_selected_mols,
                lambda *args: retro.mol_score_imps(*args, score_imp_group_fn=grouper),
                )

        data_order_key=lambda entry: -sum(entry['y'])
        return retro.cross_ws_plot(
                workspaces,
                order_key=retro.ws_review_date_key,
                data_order_key=data_order_key,
                y_keys=fns,
                y_names=cats,
                title='Hit Molecules Score Importance',
                x_title='Disease (by date)',
                y_title='Score Importance',
                )

    def ranks_plot(self):
        import dtk.retrospective as retro
        workspaces = retro.filter_workspaces(self.ws)
        rank_func = retro.make_rank_func(retro.unreplaced_selected_mols)
        return retro.cross_ws_box_plot(
                workspaces,
                order_key=retro.ws_review_date_key,
                y_func=rank_func,
                title='Selected Ranks',
                x_title='Disease (by date)',
                y_title='Ranks',
                )

    def make_scoreimp_form(self, data):
        choices = self.scoreimp_types.choices()
        class MyForm(forms.Form):
            scoreimp_type = forms.ChoiceField(
                        choices=choices,
                        label='Group scores by:',
                        initial=self.scoreimp_type,
                        )
        return MyForm(data)

    def scoreimp_post_valid(self):
        p = self.context['scoreimp_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))

    def custom_context(self):
        if self.plot == 'molecules':

            cols, data = self.mols_and_targets()
            columns = [{'title': col} for col in cols]

            self.context_alias(
                    tables = [
                        ('mols', data, columns, [[1, 'desc']])
                        ],
                    )

        elif self.plot:
            plot_name = self.plot
            func = getattr(self, plot_name + "_plot")
            plot = func()
            plot_form = getattr(self, plot_name + '_form', None)
            if self.plot == 'scoreimp':
                decimal_places = 3
                import dtk.retrospective as retro
                workspaces = retro.filter_workspaces(self.ws)
                ws_mols = [wsa for ws in workspaces
                             for wsa in retro.unreplaced_selected_mols(ws)]
                extra_table_header = ['WS', 'Mol']
                demo = self.is_demo()
                def wsa_link(wsa):
                    return f'<a href="{wsa.drug_url()}">{wsa.get_name(demo)}</a>'
                extra_table = [
                        [wsa.ws.name, wsa_link(wsa)]
                        for wsa in ws_mols
                        ]
                self.context_alias(
                        extra_table = extra_table,
                        extra_table_header = extra_table_header,
                        extra_table_title = 'Score Source Molecules',
                        )
            else:
                decimal_places = 0
            if self.plot == 'ranks' or self.plot == 'targetcmp':
                plot_table = False
            else:
                plot_table = True

            self.context_alias(
                    div_id=plot_name,
                    src=plot,
                    plot_form=plot_form,
                    plot_form_btn=(plot_name, 'Plot'),
                    plot_table=plot_table,
                    decimal_places=decimal_places,
                    )
        else:
            self.context_alias(
                sections=[
                    ('Prescreened', 'prescreened'),
                    ('Reviewed', 'reviewed'),
                    ('Ranks', 'ranks'),
                    ('Score Sources', 'scoreimp'),
                    ('Molecules', 'molecules'),
                    ('Target Compare', 'targetcmp'),
                    ],
                url=self.request.path + '?' + self.request.GET.urlencode(),
                )


class DashboardView(DumaView):
    def custom_setup(self):
        from dtk.s3_cache import S3Bucket, S3File
        class S3DashboardBucket(S3Bucket):
            def __init__(self):
                from path_helper import PathHelper, make_directory
                self.cache_path = PathHelper.dashboard_publish()
                make_directory(os.path.join(self.cache_path, 'publish'))
                from dtk.aws_api import Bucket
                self.bucket = Bucket(
                        name='2xar-dashboard-data',
                        cache_path=self.cache_path,
                        )

        # Fetch the files.
        files = ['publish/dashboard.html', 'publish/dash-data.js']
        for f in files:
            s3f = S3File(S3DashboardBucket(),f)
            s3f.fetch(force_refetch=True)

        return HttpResponseRedirect('/publish/dashboard/publish/dashboard.html')


class ProteinScoreView(DumaView):
    GET_parms={
            'jobs_and_codes':(list_of(str),None),
            'norm':(str,None),
            'plot':(str,None),
            'wzs_job':(str,None),
            'flag_prots':(list_of(str), []),
            'flag_ps':(str,None),
            }
    button_map={'calc':['calc']}

    @property
    def template_name(self):
        if not self.plot:
            return 'browse/protein_score_view.html'
        elif self.plot == 'network':
            return 'browse/_protein_network.html'
        elif self.plot == 'scoreplot' or self.plot == 'heatmap':
            return '_plotly_div.html'
        elif self.plot == 'scoretable' or self.plot == 'metrics':
            return 'browse/_datatable.html'
        else:
            raise Exception("Implement %s" % self.plot)

    def make_calc_form(self, data):
        MAX_SZ = 15
        # Currently all of our protein signatures get fed through codes,
        # so use that to find them.
        codes_choices = self.ws.get_prev_job_choices('codes')

        choices = []

        # In addition to codes inputs, we can now also look at refresh workflows
        # that used uniprot scoring.
        wf_choices = self.ws.get_prev_job_choices('wf')
        for jid, desc in wf_choices:
            bji = JobInfo.get_bound(self.ws, jid)
            if 'RefreshFlow' not in bji.job.name:
                continue

            from dtk.prot_map import DpiMapping
            dpi = DpiMapping(bji.parms['p2d_file'])
            mapping_type = dpi.mapping_type()
            if mapping_type != 'uniprot':
                continue

            choices.append((f'{jid}_refreshwf', f'Uniprot RefreshWF {jid}'))



        for jid, _ in codes_choices:
            bji = JobInfo.get_bound(self.ws, jid)
            try:
                input_jid, input_code = bji.input_job_and_code()
            except KeyError:
                # This job doesn't know its own input score, skip it.
                self.message("Skipping CoDES job %s, no input"% jid)
                continue
            input_bji = JobInfo.get_bound(self.ws, input_jid)
            choice = (
                f'{input_jid}_{input_code}',
                f'{input_bji.role_label()} {input_code} ({input_jid})',
                )
            if choice not in choices:
                choices.append(choice)

        sz = min(len(codes_choices), MAX_SZ)

        norm_choices = [
                ('minmax', 'MinMax'),
                ('density', 'Density'),
                ('none', 'None'),
                ]

        wzs_choices = [(None, 'None')] + self.ws.get_prev_job_choices('wzs')

        ps_choices = self.ws.get_uniprot_set_choices()

        class MyForm(forms.Form):
            jobs_and_codes = forms.MultipleChoiceField(
                        choices=choices,
                        label='Include these runs',
                        initial=self.jobs_and_codes,
                        widget=forms.SelectMultiple(
                                attrs={'size':sz}
                                ),
                        )

            wzs_job = forms.ChoiceField(
                    choices=wzs_choices,
                    label='Norm via WZS',
                    initial=self.wzs_job,
                    required=False,
                    help_text="(Requires only Uniprot RefreshWF run)",
                )

            norm = forms.ChoiceField(
                        choices=norm_choices,
                        label='Normalization',
                        initial=self.norm,
                        )

            flag_ps = forms.ChoiceField(
                        choices=ps_choices,
                        label='Flag Protein Set',
                        initial=self.flag_ps,
            )
        return MyForm(data)

    def calc_post_valid(self):
        p = self.context['calc_form'].cleaned_data
        p['jobs_and_codes'] = ','.join(p['jobs_and_codes'])
        return HttpResponseRedirect(self.here_url(**p))

    @classmethod
    def get_scoreset_from_refreshwf(cls, ws, jid):
        from browse.models import ScoreSet
        from runner.models import Process
        try:
            return ScoreSet.objects.get(ws=ws, wf_job=jid)
        except ScoreSet.DoesNotExist:
            # If we did a resume, we won't have a scoreset pointing directly
            # at us, but we'll have the ID in our settings, go find it.
            p = Process.objects.get(pk=jid)
            ss_id = p.settings()['resume_scoreset_id']
            return ScoreSet.objects.get(pk=ss_id)

    def make_highlight_prots(self, show_indirect):
        from collections import defaultdict
        from browse.default_settings import PpiDataset
        from dtk.prot_map import PpiMapping
        highlights = defaultdict(list)
        ppi_map = PpiMapping(PpiDataset.value(ws=self.ws))
        if self.flag_ps:
            self.flag_prots = set(self.flag_prots)
            self.flag_prots.update(self.ws.get_uniprot_set(self.flag_ps))
        for prot in self.flag_prots:
            gene = self.uni2gene.get(prot, prot)
            highlights[prot].append(f'{gene}-direct')

            if show_indirect:
                for p1, p2, ev, dr in ppi_map.get_ppi_info_for_keys([prot], min_evid=0.9):
                    highlights[p2].append(f'{gene}-indirect')
        
        for prot in self.kt_prots:
            highlights[prot].append('NonNovel')
        
        return highlights

    def compute_score_data(self):
        import algorithms.run_compositesig as csig
        tags = self.make_highlight_prots(show_indirect=True)

        from collections import defaultdict
        table_data = defaultdict(lambda: defaultdict(dict))
        all_prots = set()


        is_refresh = all(['refreshwf' in x for x in self.jobs_and_codes])
        jids_and_codes = csig.extract_jids_and_codes(self.ws, self.jobs_and_codes)

        bjis = {}
        for jid, code in jids_and_codes:
            bji = JobInfo.get_bound(self.ws, jid)
            bjis[jid] = bji

        if self.wzs_job:
            assert is_refresh, "WZS norm only works with uniprot refresh runs"
            self.norm = None
            datas, names = csig.wzs_weighted_uniprot_scores(
                ws=self.ws,
                wzs_jid=self.wzs_job,
                jids_and_codes=jids_and_codes,
                )
        else:
            names = []
            datas = []
            for jid, code in jids_and_codes:
                bji = bjis[jid]
                name = '%s %s %s' % (bji.job.role, code, jid)
                names.append(name)
                dc = bji.get_data_catalog()
                gen = list(dc.get_feature_vectors(code)[1])
                datas.append(gen)

        for name, gen in zip(names, datas):
            import numpy as np
            max_ev = 0
            min_ev = 1
            sum_ev = 0
            for uniprot, (ev,) in gen:
                max_ev = max(max_ev, abs(ev))
                min_ev = min(min_ev, abs(ev))
                sum_ev += abs(ev)


            for uniprot, (ev,) in gen:
                table_data['score'][uniprot]['uniprot'] = uniprot
                table_data['score'][uniprot]['sm_druggable'] = uniprot in self.sm_druggable_prots
                table_data['score'][uniprot]['dpi_druggable'] = uniprot in self.dpi_druggable_prots
                table_data['score'][uniprot]['gene'] = self.uni2gene.get(uniprot, uniprot)
                table_data['score'][uniprot]['tags'] = ''.join(f'<tag>{tag}</tag>' for tag in tags[uniprot])

                if self.norm == 'minmax':
                    if min_ev == max_ev:
                        score = 0
                    else:
                        score = (abs(ev) - min_ev)/(max_ev-min_ev)
                elif self.norm == 'density':
                    score = abs(ev) / sum_ev
                else:
                    score = abs(ev)
                table_data['score'][uniprot][name] = score
                all_prots.add(uniprot)
        names.sort()

        plot_data = []
        import numpy as np
        prot_data = table_data['score']
        for prot in all_prots:
            data = prot_data[prot]
            values = [data.get(name, 0) for name in names]
            data['mean'] = np.mean(values)
            data['median'] = np.median(values)
            data['values'] = values

            name = f'{self.uni2gene.get(prot, "")} ({prot})'

            plot_data.append((data['mean'], name, values, prot))
        plot_data.sort(reverse=True)
        for rank, (mean, name, values, prot) in enumerate(plot_data):
            table_data['score'][prot]['rank_mean'] = str(rank + 1)

        return table_data, plot_data, names

    def get_kt_prots(self):
        kt_prots = self.ws.get_uniprot_set(self.ws.get_nonnovel_ps_default())
        if not kt_prots:
            tts_prots = self.ws.get_uniprot_set(f'ds_0.5_tts')
            kt_prots.update(tts_prots)
        return kt_prots

    def make_metrics(self):
        from dtk.enrichment import EnrichmentMetric, EMInput
        table_data, plot_data, source_names = self.compute_score_data()
        ordering = [(prot, mean) for mean, name, values, prot in plot_data]
        emi_dpi_druggable = EMInput(ordering, self.dpi_druggable_prots)
        emi_sm_druggable = EMInput(ordering, self.sm_druggable_prots)
        emi_nonnovel = EMInput(ordering, self.kt_prots)
        metric_names = ['SigmaOfRank1000', 'SigmaPortion1000', 'SigmaPortionWeighted1000', 'SigmaPortion100', 'SigmaPortionWeighted100', 'AUR']

        columns = [
                {"data": "name", "title": "Metric Name"},
                {"data": "dpi_druggable", "title": "DPI Druggable" },
                {"data": "sm_druggable", "title": "SM Druggable" },
                {"data": "nonnovel", "title": "NonNovel" },
                ]
        order = [[0, 'desc']]

        table_data = []
        for metric_name in metric_names:
            EMClass = EnrichmentMetric.lookup(metric_name)
            em = EMClass()
            em.evaluate(emi_dpi_druggable)
            dpi_druggable = em.rating
            em.evaluate(emi_sm_druggable)
            sm_druggable = em.rating

            em.evaluate(emi_nonnovel)
            nonnovel = em.rating
            table_data.append({'name': metric_name,
                               'dpi_druggable': dpi_druggable,
                               'sm_druggable': sm_druggable,
                               'nonnovel': nonnovel}
                             )

        import json
        self.context_alias(
                table_data=json.dumps(table_data),
                table_columns=json.dumps(columns),
                table_order=json.dumps(order),
                table_id='metrics_table',
                )

    def make_scoreplot(self):
        table_data, plot_data, source_names = self.compute_score_data()
        # Can't draw much more than this without things getting unhappy.
        plot_data = plot_data[:5000]
        kt_prots = self.kt_prots

        table_name = 'Score'
        from dtk.plot import Color
        import numpy as np
        plotly_plots = []
        from dtk.plot import PlotlyPlot
        def bar_col(x):
            if x[3] in kt_prots:
                return Color.highlight
            if x[3] in self.sm_druggable_prots:
                return Color.default
            elif x[3] in self.dpi_druggable_prots:
                return Color.highlight2
            else:
                return Color.highlight4
        plot = PlotlyPlot([{
            'x': [f'{i:5d} | {x[1]}' for i, x in enumerate(plot_data)],
            'y': [x[0] for x in plot_data],
            'marker': {'color': [bar_col(x) for x in plot_data]},
            'type': 'bar'
            }], {
                'title': table_name,
                'width': 1000,
                'height': 1000,
                'margin': {
                    'b': 300,
                    'l': 200,
                    },
                'xaxis': {
                    'tickangle': 40,
                    },
                'yaxis': {
                    'title': "Mean " + table_name,
                    }
            })
        self.context_alias(
                div_id='scoreplot',
                src=plot,
                )

    def make_heatmap(self):
        table_data, plot_data, source_names  = self.compute_score_data()
        # Need to truncate, heatmap can't handle this much.
        plot_data = plot_data[:2000]
        kt_prots = self.get_kt_prots()
        from dtk.plot import plotly_heatmap
        data_array = np.array([data[2] for data in plot_data])
        data_array = np.transpose(data_array)
        col_labels = ['**'+data[1] if data[3] in kt_prots else data[1] for data in plot_data]
        row_labels = source_names
        heatmap = plotly_heatmap(data_array, row_labels=row_labels, col_labels=col_labels,
                width=1600, height=1200,
                colorscale='Reds',
                )
        self.context_alias(
                div_id='heatmap',
                src=heatmap,
                )

    def make_scoretable(self):
        table_data, plot_data, names  = self.compute_score_data()

        columns = [
                {"data": "protlink", "title": "Uniprot"},
                {"data": "gene", "title": "Gene" },
                {"data": "mean", "title": "Mean" },
                {"data": "median", "title": "Median"},
                {"data": "dpi_druggable", "title": "DPI Druggable"},
                {"data": "sm_druggable", "title": "Small Mol Druggable"},
                {"data": "rank_mean", "title": "Rank (by mean)" },
                {"data": "tags", "title": "Tags" },
                ]
        columns += [{"data": name, "title": name, "defaultContent": 0} for name in names]
        order = [[2, 'desc']]

        score_table_data = list(table_data['score'].values())
        for entry in score_table_data:
            protlink = self.ws.reverse('protein', entry['uniprot'])
            entry['protlink'] = f'<a href="{protlink}">{entry["uniprot"]}</a>'

        import json
        self.context_alias(
                table_data=json.dumps(score_table_data),
                table_columns=json.dumps(columns),
                table_order=json.dumps(order),
                table_id='scoretable',
                )

    def make_network(self):
        table_data, plot_data, source_names  = self.compute_score_data()
        top_prots = [p[3] for p in plot_data[:100]]
        import json
        kt_prots = self.get_kt_prots()
        top_prots = mark_safe(json.dumps(top_prots))
        kt_prots = mark_safe(json.dumps(list(kt_prots)))

        self.context_alias(
                network_prots=top_prots,
                kt_prots=kt_prots,
                )
    def get_dpi_druggable_prots(self):
        from browse.utils import get_dpi_druggable_prots
        self.dpi_druggable_prots = get_dpi_druggable_prots()
    def get_sm_druggable_prots(self):
        from dtk.open_targets import OpenTargets
        from browse.default_settings import openTargets
        otarg = OpenTargets(openTargets.latest_version())
        self.sm_druggable_prots = otarg.get_small_mol_druggable_prots()
    def custom_context(self):
        if not self.jobs_and_codes:
            return
        from browse.models import Protein

        if self.plot:
            self.uni2gene = Protein.get_uniprot_gene_map()
            self.get_dpi_druggable_prots()
            self.get_sm_druggable_prots()
            self.kt_prots = self.get_kt_prots()
            plot_name = self.plot
            func = getattr(self, 'make_' + plot_name)
            func()
        else:
            sections = [
                    ('Metrics', 'metrics'),
                    ('Score Table', 'scoretable'),
                    ('Score Plot', 'scoreplot'),
                    ('Heatmap', 'heatmap'),
                    ('Network', 'network'),
                    ]
            self.context_alias(
                    sections=sections,
                    url=self.request.path + '?' + self.request.GET.urlencode(),
                    )



class PathwayNetworkView(DumaView):
    # See the Javascript Development Confluence page for an overview of
    # how this works.
    template_name='browse/pathway_network_view.html'
    GET_parms={
            'query':(str, None),
            }
    def score_query(self, query_data):
        ws_id = query_data['wsId']
        job_id = query_data['jobId']
        code = query_data['code']
        pathway_ids = query_data['pathwayIds']
        bji = JobInfo.get_bound(ws_id, job_id)
        score_list = bji.get_data_catalog().get_ordering(code, True)

        from dtk.reactome import Reactome, score_pathways
        rct = Reactome()
        pathways = rct.get_pathways(pathway_ids)
        scores = score_pathways(pathways=pathways, score_list=score_list)
        score_labels = ['id', 'pathway_prots', 'por_pathway_scored', 'best_q', 'best_or', 'peak_ind', 'final_score']

        labelled_scores = [dict(zip(score_labels, score)) for score in scores]

        return JsonResponse({
            'scores': labelled_scores,
            })


    def prot_query(self, query_data):
        from dtk.reactome import Reactome
        rct = Reactome()
        prots = query_data['prots']
        out_nodes = []
        edges = []
        for prot in prots:
            out_nodes.append({
                'id': prot,
                'name': prot, # TODO: Gene Name
                'type': 'prot',
                })

            pathways = rct.get_pathways_with_prot(prot)
            for pathway in pathways:
                out_nodes.append({
                    'id': pathway.id,
                    'name': pathway.name
                    })
                sub_pathways = pathway.get_sub_pathways()
                for sub_pathway in sub_pathways:
                    edges.append([pathway.id, sub_pathway.id])
                # TODO: This isn't the best way to check this.
                if not sub_pathways:
                    edges.append([pathway.id, prot])


        return JsonResponse({
            'edges': edges,
            'nodes': out_nodes,
            })



    def custom_setup(self):
        if self.query:
            import json
            query_data = json.loads(self.query)

            if query_data['type'] == 'prot':
                return self.prot_query(query_data)
            elif query_data['type'] == 'score':
                return self.score_query(query_data)


            from dtk.reactome import Reactome
            rct = Reactome()
            depth = query_data['depth']
            nodes = set(query_data['nodes'])

            out_nodes = []
            edges = []

            for node in nodes:
                pathway = rct.get_pathway(node)
                out_nodes.append({
                    'id': pathway.id,
                    'name': pathway.name,
                    'type': pathway.type,
                    'hasDiagram': pathway.hasDiagram,
                    })
                if pathway.type == 'pathway':
                    for sub_pathway in pathway.get_sub_pathways():
                        out_nodes.append({
                            'id': sub_pathway.id,
                            'name': sub_pathway.name,
                            'type': pathway.type,
                            'hasDiagram': sub_pathway.hasDiagram,
                            })
                        edges.append([pathway.id, sub_pathway.id])

                        for neighbor in sub_pathway.get_neighbors():
                            edges.append([sub_pathway.id, neighbor.id])
                            edges.append([neighbor.id, sub_pathway.id])
                elif pathway.type == 'event':
                    for prot in pathway.get_proteins():
                        out_nodes.append({
                            'id': prot.id,
                            'name': prot.name,
                            'type': 'prot',
                            'hasDiagram': pathway.hasDiagram,
                            })
                        edges.append([pathway.id, prot.id])


            return JsonResponse({
                'edges': edges,
                'nodes': out_nodes,
                })

class ProteinNetworkView(DumaView):
    template_name='browse/protein_network_view.html'
    GET_parms={
            'prots':(list_of(str),['Q8IWL2']),
            'query':(str, None),
            }
    button_map={'calc':['calc']}

    def make_calc_form(self, data):
        class MyForm(forms.Form):
            prots = forms.CharField(
                        label='Starting prots',
                        initial='\n'.join(self.prots),
                        widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
                        )

        return MyForm(data)

    def calc_post_valid(self):
        p = self.context['calc_form'].cleaned_data
        p['prots'] = ','.join(p['prots'])
        return HttpResponseRedirect(self.here_url(**p))

    def custom_setup(self):
        if self.query:
            import json
            query_data = json.loads(self.query)

            from dtk.prot_map import PpiMapping
            ppi = PpiMapping(self.ws.get_ppi_default())

            depth = query_data['depth']
            prots = set(query_data['prots'])
            shown_prots = set(query_data['shownProts'])
            edges = []
            cur_prots = set(prots)
            for cur_depth in range(depth + 1):
                next_prots = set()
                new_edges = ppi.get_ppi_info_for_keys(
                        cur_prots, min_evid=self.ws.get_ppi_thresh_default())
                if cur_depth < depth:
                    edges.extend(tuple(new_edges))
                    for p1, p2, ev, dr in new_edges:
                        if p2 not in prots:
                            prots.add(p2)
                            next_prots.add(p2)
                else:
                    for p1, p2, ev, dr in new_edges:
                        if p2 in prots or p2 in shown_prots:
                            edges.append((p1, p2, ev, dr))

                cur_prots = next_prots

            from browse.models import Protein
            uni2gene = Protein.get_uniprot_gene_map(prots)
            prots = [{'prot': prot, 'gene': uni2gene.get(prot, prot)}
                    for prot in prots]

            return JsonResponse({
                'edges': edges,
                'prots': prots,
                })

    def custom_context(self):
        import json
        self.context_alias(
                startProts=mark_safe(json.dumps(self.prots)),
                url=self.request.path,
                )


class ProtsetView(DumaView):
    template_name='browse/protset_view.html'
    GET_parms={
            'protsets':(list_of(str),[]),
            'all_prots_for_gene':(boolean,False),
            }
    button_map={'calc':['calc']}
    def make_calc_form(self, data):
        choices = self.ws.get_uniprot_set_choices()
        class MyForm(forms.Form):
            protsets = forms.MultipleChoiceField(
                    label='Prot Sets',
                    choices=choices,
                    required=True,
                    initial=self.protsets,
                    widget=forms.SelectMultiple(
                            attrs={'size':20}
                            ),
                    )
            all_prots_for_gene = forms.BooleanField(
                        label='Expand Genes',
                        required=False,
                        initial=self.all_prots_for_gene,
                    )

        return MyForm(data)
    def calc_post_valid(self):
        p = self.context['calc_form'].cleaned_data
        p['protsets'] = ','.join(p['protsets'])
        return HttpResponseRedirect(self.here_url(**p))
    def custom_context(self):
        from collections import defaultdict
        all_prots = defaultdict(int)
        ps_to_prots = {}
        ps_and_prots = []
        for ps in self.protsets:
            prots = self.ws.get_uniprot_set(ps)
            for prot in prots:
                all_prots[prot] += 1
            for prot in prots:
                ps_and_prots.append((ps, prot))

        from browse.models import Protein
        uniprot2gene_map = Protein.get_uniprot_gene_map(
                                all_prots.keys()
                                )

        from dtk.data import MultiMap
        from dtk.html import link
        mm = MultiMap(ps_and_prots)
        prot_datas = []
        prots_to_ps = mm.rev_map()
        genes_used = set()
        for prot, pses in prots_to_ps.items():
            gene = uniprot2gene_map.get(prot, f'({prot})')
            if gene in genes_used and not self.all_prots_for_gene:
                continue
            genes_used.add(gene)
            data = {ps: 'X' for ps in pses}
            data['uniprot'] = link(prot, self.ws.reverse('protein', prot))
            data['gene'] = gene
            prot_datas.append(data)

        prot_datas.sort(key=lambda x:(-len(x), x['gene'], x['uniprot']))

        columns = [
                Table.Column('Gene', idx='gene'),
                Table.Column('Uniprot', idx='uniprot'),
                ] + [
                    Table.Column(self.ws.get_uniprot_set_name(psid), idx=psid)
                    for psid in self.protsets
                ]
        table = Table(prot_datas, columns)
        self.context_alias(table=table)


class XwsScorePlots(DumaView):
    template_name='browse/xws_score_plots.html'
    GET_parms={
            'wf_jids':(list_of(str),[]),
            'job_type':(str, ''),
            'code':(str, ''),
            'ds':(str, 'tts'),
            'mode':(str, 'wzscmp'),
            }
    button_map={
        'calc':['calc'],
        'wzscalc':['wzscalc'],
        }
    def make_calc_form(self, data):
        class MyForm(forms.Form):
            wf_jids = forms.CharField(
                    widget=forms.Textarea(attrs={'rows':'6','cols':'50'}),
                    required=True,
                    initial='\n'.join(self.wf_jids),
                    )
            job_type = forms.CharField(
                required=True,
                initial=self.job_type,
                )
            code = forms.CharField(
                required=True,
                initial=self.code,
                )
            ds = forms.CharField(
                required=True,
                initial=self.ds,
                )

        return MyForm(data)
    def calc_post_valid(self):
        p = self.context['calc_form'].cleaned_data
        import re
        jids = re.split(r'[,\s]+', p['wf_jids'])
        p['wf_jids'] = ','.join(x.strip() for x in jids)
        p['mode'] = 'scorecmp'
        return HttpResponseRedirect(self.here_url(**p))
    def make_wzscalc_form(self, data):
        class MyForm(forms.Form):
            wf_jids = forms.CharField(
                    widget=forms.Textarea(attrs={'rows':'6','cols':'50'}),
                    required=True,
                    initial='\n'.join(self.wf_jids),
                    )
            ds = forms.CharField(
                required=True,
                initial=self.ds,
                )

        return MyForm(data)
    def wzscalc_post_valid(self):
        p = self.context['wzscalc_form'].cleaned_data
        import re
        jids = re.split(r'[,\s]+', p['wf_jids'])
        p['wf_jids'] = ','.join(x.strip() for x in jids)
        p['mode'] = 'wzscmp'
        return HttpResponseRedirect(self.here_url(**p))
    def get_score_groups(self, bji, kts, code):
        cat = bji.get_data_catalog()
        ordering = cat.get_ordering(code, True)
        N = 1
        if bji.job_type == 'path':
            N = len(bji.tissue_ids)
        # Map to wsa groups.
        kt_out = []
        other_out = []
        for wsa, val in ordering:
            val /= N
            if wsa in kts:
                kt_out.append(val)
            else:
                other_out.append(val)
        return kt_out, other_out

    def get_typed_bji(self, ws, wf_jobid, jobtype):
        from runner.process_info import JobInfo
        wf_bji = JobInfo.get_bound(ws, wf_jobid)
        scoreset = wf_bji.get_scoreset()
        jobtype_to_id = scoreset.job_type_to_id_map()
        print("Check types", jobtype_to_id)
        out_jid = jobtype_to_id[jobtype]
        return JobInfo.get_bound(wf_bji.ws, out_jid)

    def get_eff_fvs_fm(self, ws, wf_jobid):
        scoreset = JobInfo.get_bound(ws, wf_jobid).get_scoreset()
        jobtype_to_id = scoreset.job_type_to_id_map()
        # First try to find the flat WZS feature matrix, it is much
        # faster to work with as it has been flattened - but it might
        # not exist if you didn't run this here.
        wzs_jid = jobtype_to_id['wzs']
        wzs_bji = JobInfo.get_bound(ws, wzs_jid)
        import dtk.features as feat
        try:
            fm = feat.FMBase.load_from_file(os.path.join(wzs_bji.indir, 'fm'))
            return fm
        except FileNotFoundError:
            # Otherwise, fall back to the fvs one.
            fvs_jid = jobtype_to_id['eff_fvs']
            return ws.get_feature_matrix('fvs%s' % fvs_jid)
        
    def collect(self, wf_jid, job_type, code):
        from dtk.prot_map import DpiMapping, AgentTargetCache
        from dtk.scores import Ranker
        from browse.models import WsAnnotation, Workspace
        from collections import defaultdict, Counter
        
        bji = self.get_typed_bji(None, wf_jid, job_type)
        ws = bji.ws
        kts = ws.get_wsa_id_set(self.ds)

        # Exclude those missing DPI.
        dpi = DpiMapping(ws.get_dpi_default())
        treat_qs = WsAnnotation.objects.filter(pk__in=kts)
        atc = AgentTargetCache.atc_for_wsas(treat_qs, ws=ws, dpi_mapping=dpi)
        missing_dpi = set()
        good_kts = set()
        for wsa_id, agent in treat_qs.values_list('id', 'agent_id'):
            if not atc.info_for_agent(agent):
                missing_dpi.add(wsa_id)
            else:
                good_kts.add(wsa_id)
        
        kt_data, other_data = self.get_score_groups(bji, good_kts, code)
        
        return kt_data, other_data
    
    def do_scoreplots(self):
        if self.job_type and self.code and self.wf_jids:
            import plotly.io as pio
            pio.templates.default='none'
            from dtk.plot import PlotlyPlot, bar_histogram_overlay
            from plotly.subplots import make_subplots

            plots = []
            all_kt = []
            all_other = []
            workspaces = []
            for wf_jid in self.wf_jids:
                kt, other = self.collect(wf_jid, self.job_type, self.code)
                # TODO: Weight these 
                all_kt.extend(kt)
                all_other.extend(other)
                plt = PlotlyPlot(*bar_histogram_overlay([kt, other], names=['kt','other'], bins=8, density=True))
                plt.update_layout({"title": f'{wf_jid}'})
                plots.append((wf_jid, plt))

                bji = JobInfo.get_bound(None, wf_jid)
                workspaces.append(bji.ws)

            COLS = 3
            ROWS = (len(plots) + COLS - 1) // COLS
            titles = [f'{ws.name}' for ws in workspaces]
            fig = make_subplots(rows=ROWS, cols=COLS, subplot_titles=titles)
            for i, plot in enumerate(plots):
                row = (i // COLS) + 1
                col = (i % COLS) + 1
                for data in plot[1]._data:
                    data['legendgroup'] = data['name']
                    if i > 0:
                        data['showlegend'] = False
                    fig.add_trace(data, row=row, col=col)
            fig_d = fig.to_dict()
            pp = PlotlyPlot(fig_d['data'], fig_d['layout'])
            pp.update_layout({
                'title': 'Per-WS',
                'height': 240 * ROWS,
                'width': 320 * COLS,
                })
            plots = [('combined', pp)]

            plt = PlotlyPlot(*bar_histogram_overlay([all_kt, all_other], names=['kt','other'], bins=20, density=True))
            plt.update_layout({'title': "Combined"})
            plots.append(("Combined", plt))

            self.context_alias(plots=plots)
    
    def wzscmp_collect(self, wf_jid):
        from dtk.prot_map import DpiMapping, AgentTargetCache
        from dtk.scores import Ranker
        from browse.models import WsAnnotation, Workspace
        from collections import defaultdict, Counter
        wzs_bji = self.get_typed_bji(None, wf_jid, 'wzs')
        ws = wzs_bji.ws

        kts = ws.get_wsa_id_set(self.ds)

        # Count how many are just missing DPI.
        # TODO: For MoA-based workspaces, maybe look at non-moa like in stages.py?
        dpi = DpiMapping(ws.get_dpi_default())
        treat_qs = WsAnnotation.objects.filter(pk__in=kts)
        atc = AgentTargetCache.atc_for_wsas(treat_qs, ws=ws, dpi_mapping=dpi)
        missing_dpi = set()
        for wsa_id, agent in treat_qs.values_list('id', 'agent_id'):
            if not atc.info_for_agent(agent):
                missing_dpi.add(wsa_id)

        
        cat = wzs_bji.get_data_catalog()
        ordering = cat.get_ordering('wzs', True)
        ranker = Ranker(ordering)

        group_funcs = {
            '1_missing_dpi': lambda x: x in missing_dpi,
            '2_non_missing_dpi': lambda x: x not in missing_dpi,
            '3_bad_wzs_rank': lambda x: ranker.get(x) > 2000,
            '4_mid_wzs_rank': lambda x: ranker.get(x) > 1000 and ranker.get(x) <= 2000,
            '5_good_wzs_rank': lambda x: ranker.get(x) <= 1000,
        }


        groups = defaultdict(set)
        for kt in kts:
            for name, func in group_funcs.items():
                if func(kt):
                    groups[name].add(kt)

        fm = self.get_eff_fvs_fm(ws, wf_jid)
        sample_keys = list(fm.sample_keys)
        wsa_to_idx = {key:i for i, key in enumerate(sample_keys)}

        features = list(fm.feature_names)

        kt_stats = {kt: Counter(['1_total']) for kt in kts}
        cm_stats = {cm: Counter() for cm in features}
        for col, feat in list(enumerate(features)):
            col_data = fm.data[:, col].toarray()[:, 0]
            cm_ranker = Ranker(sorted(zip(sample_keys, col_data), key=lambda x: -x[1]))
            for kt in kts:
                row = wsa_to_idx[kt]
                score = col_data[row]
                rank = cm_ranker.get(kt)
                if score == 0 or np.isnan(score):
                    category = '5_unscored'
                elif rank <= 2000:
                    category = '2_good_cm_rank'
                elif rank <= 5000:
                    category = '3_mid_cm_rank'
                else:
                    category = '4_bad_cm_rank'
                kt_stats[kt].update([category])
                cm_stats[feat].update([category])

        import pandas as pd
        cm_df = pd.DataFrame(cm_stats).fillna(0).T.sort_index(axis=1)

        grouped_kts = {}
        grouped_kts_any = {}
        for name, groupkts in groups.items():
            group_cats = Counter()
            group_cats_any = Counter()
            for kt in groupkts: 
                group_cats.update(kt_stats[kt])
                group_cats_any.update([f'any_{cat}' for cat, cnt in kt_stats[kt].items() if cnt > 0])
            grouped_kts[name] = group_cats
            grouped_kts_any[name] = group_cats_any
        combined_df = pd.DataFrame(grouped_kts).fillna(0).T.sort_index(axis=0).sort_index(axis=1)
        combined_any_df = pd.DataFrame(grouped_kts_any).fillna(0).T.sort_index(axis=0).sort_index(axis=1)

        return cm_df, combined_df, combined_any_df, len(kts)

    def df_to_heatmap(self, df):
        from dtk.plot import PlotlyPlot, plotly_heatmap
        import plotly.figure_factory as ff

        data = df.to_numpy()


        cols = df.columns.tolist()
        rows = df.index.tolist()

        fig = ff.create_annotated_heatmap(data, x=cols, y=rows, colorscale='Viridis')
        fig_d = fig.to_dict()
        fig_d['layout']['height'] = len(rows) * 30 + 8*max([len(x) for x in cols]) + 40
        fig_d['layout']['width'] = len(cols) * 40 + 8*max([len(x) for x in rows]) + 40
        fig_d['layout']['yaxis']['automargin'] = True
        fig_d['layout']['xaxis']['automargin'] = True
        return PlotlyPlot(fig_d['data'], fig_d['layout'])


    def do_wzscmpplots(self):
        if self.wf_jids:
            import plotly.io as pio
            pio.templates.default='none'
            from dtk.plot import PlotlyPlot, plotly_heatmap
            from plotly.subplots import make_subplots

            plots_cm = []
            plots_comb = []
            plots_comb_any = []
            all_kt = []
            all_other = []
            workspaces = []

            ref_df = None
            ref_comb_df = None
            ref_comb_any_df = None
            for wf_jid in self.wf_jids:
                cm_df, combined_df, combined_any_df, num_kts = self.wzscmp_collect(wf_jid)

                bji = JobInfo.get_bound(None, wf_jid)
                ws = bji.ws
                workspaces.append(bji.ws)

                plot = self.df_to_heatmap(cm_df)
                plot.update_layout({"title": ws.name})
                plots_cm.append((wf_jid, plot))

                plot = self.df_to_heatmap(combined_df)
                plot.update_layout({"title": ws.name})
                plots_comb.append((f'{wf_jid}_comb', plot))

                plot = self.df_to_heatmap(combined_any_df)
                plot.update_layout({"title": f'{ws.name} ({num_kts} kts)'})
                plots_comb_any.append((f'{wf_jid}_comb_any', plot))

                if ref_df is None:
                    ref_df = cm_df / num_kts
                    ref_comb_df = combined_df / combined_df.max()
                    ref_comb_any_df = combined_any_df / num_kts
                else:
                    ref_df = ref_df.add(cm_df / num_kts, fill_value=0)
                    ref_comb_df = ref_comb_df.add(combined_df / combined_df.max(), fill_value=0)
                    ref_comb_any_df = ref_comb_any_df.add(combined_any_df / num_kts, fill_value=0)

            plot = self.df_to_heatmap(ref_df.round(2))
            plots_cm.append((f'comb_all', plot))
            plot = self.df_to_heatmap(ref_comb_df.round(2))
            plots_comb.append((f'comb_comb_all', plot))
            plot = self.df_to_heatmap(ref_comb_any_df.round(2))
            plots_comb_any.append((f'comb_comb_any_all', plot))

            plots = [*plots_cm, *plots_comb, *plots_comb_any]

            self.context_alias(plots=plots)

    def custom_context(self):
        if self.mode == 'scorecmp':
            self.do_scoreplots()
        elif self.mode == 'wzscmp':
            self.do_wzscmpplots()


class DumaFormTestView(DumaView):
    template_name='browse/test_form.html'
    index_dropdown_stem='test_form'
    button_map={'form1':['default']}
    def make_default_form(self,data):
        FormClass=form_factory2(6)
        return FormClass(data)
    def custom_context(self):
        # The following line is only required because the template expects
        # the name 'form', but DumaView puts all forms in the context as
        # '<something>_form'. And, that's the only reason we needed to
        # override this function, so we would have saved a few lines by
        # fixing the template, but I'm trying to investigate what workarounds
        # are available.
        self.context['form'] = self.context['default_form']
    def form1_post_valid(self):
        print(self.context['default_form'].cleaned_data)
        return HttpResponseRedirect('#')

# For processing multiple independent forms on the same page, we could
# use class-based views with something like this:
# https://gist.github.com/jamesbrobb/748c47f46b9bd224b07f

# Based on usage examples, all the validation typically happens
# in the form class, but the actual action is triggered by view code
# (in a form_valid method, for class-based views).  Some forms may
# have helper functions to simplfy the view code (like the save() method
# on a model form)

# for constructing forms dynamically and/or factoring out reusable sub-parts,
# we could use some combination of the following:
# - inclusion of separately-defined subparts via add_subform()
# - factory methods as alternatives to overriding __init__
# Things constructed by these methods could live in a forms.py file within
# the app, which encourages reuse and common sub-parts.

# the following allows composition of multiple forms, both
# static and from factory classes; setting a prefix will also
# alter defaulted labels
def add_subform(fields,sub,prefix=''):
    for name,fld in six.iteritems(sub.fields):
        fields[prefix+name] = fld

class TestForm1(forms.Form):
    number = forms.IntegerField()
    name = forms.CharField(label='Name1')

class TestForm2(forms.Form):
    name = forms.CharField()

class TestForm(forms.Form):
    due_date = forms.DateTimeField(label="Due Date:")
    def __init__(self, *args, **kwargs):
        super(TestForm,self).__init__(*args, **kwargs)
        # XXX individual tweaks to fields statically defined above
        # XXX can be hard-coded here
        #
        # XXX and/or, functions to dynamically insert form pieces
        # XXX can be called here
        add_subform(self.fields,TestForm1(*args, **kwargs),'sub1_')

# create form with a factory; contrary to published examples on the web,
# an OrderedDict is required to control field order; you can define
# methods on the form as well as setting fields and other attributes
def form_factory1():
    from collections import OrderedDict
    from django.core.validators import RegexValidator
    fields = OrderedDict(
            name = forms.CharField(),
            address = forms.CharField(validators=[
                                    RegexValidator(r'[0-9]+ [a-z]+'),
                                    ]),
            )
    fields['phone'] = forms.CharField()
    def phone_check(self):
        if self.cleaned_data['phone'].startswith('+'):
            raise forms.ValidationError('plus not allowed')
    return type(
            'InfoForm',
            (forms.BaseForm,),
            {
                    'base_fields': fields,
                    'clean_phone': phone_check,
                    },
            )

def form_factory2(cnt):
    from collections import OrderedDict
    fields = OrderedDict()
    for i in range(cnt):
        fields['field_%d'%i] = forms.CharField()
    return type('DynaForm', (forms.BaseForm,), {'base_fields': fields})

from django.views.generic.edit import FormView
class FormTestView(FormView):
    form_class=form_factory2(6)
    template_name='browse/test_form.html'
    success_url='#' # come back to same form
    def form_valid(self,form):
        print(form.cleaned_data)
        return super(FormTestView,self).form_valid(form)




import django.contrib.auth.views as av
class PasswordChangeView(av.PasswordChangeView):
    def form_valid(self, *args, **kwargs):
        rsp = super().form_valid(*args, **kwargs)
        import os
        import subprocess
        msg = f'User {self.request.user.username} changed their password on {os.uname()[1]}'
        from path_helper import PathHelper
        pgm = PathHelper.website_root + 'scripts/slack_send.sh'
        subprocess.check_call([pgm, msg])
        return rsp
