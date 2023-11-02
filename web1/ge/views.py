from dtk.duma_view import DumaView,list_of,boolean
from dtk.table import Table,SortHandler
from django import forms
from django.http import HttpResponseRedirect

# the following are needed for old-style views
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from browse.views import make_ctx
from browse.views import post_ok

import logging
logger = logging.getLogger(__name__)

# Create your views here.
@login_required
def tissues(request,ws_id):
    from dtk.timer import Timer
    ltmr = Timer()
    # After sprint 147, https://platform.twoxar.com/44/tissues/ loads in
    # under 1.5 seconds, so further optimization may not be necessary.
    # A bit over 1/3 of that time is in latest_jobs, a bit under 1/3 in
    # extended_status, and the rest in everything else.
    from .utils import TissueActionCache
    tac = TissueActionCache(ws_id)
    from browse.models import Workspace
    ws = Workspace.objects.get(pk=ws_id)
    ws.check_defaults()
    session_sort_key = 'tissuesort_%d' % ws.id
    from dtk.url import UrlConfig
    url_config=UrlConfig(request, defaults={
                        'sort':request.session.get(session_sort_key,'id'),
                        })
    sort_type = url_config.as_string('sort')
    from runner.process_info import JobCrossChecker
    jcc = JobCrossChecker()
    excluded = url_config.as_bool('excluded')
    from browse.models import Tissue
    tissue_qs = Tissue.objects.filter(ws=ws)
    from dtk.html import link
    if excluded:
        excluded_link = link("hide excluded",url_config.here_url({
                                'excluded':0,
                                }))
    else:
        tissue_qs=tissue_qs.exclude(tissue_set_id__isnull=True)
        excluded_link = link("show excluded",url_config.here_url({
                                'excluded':1,
                                }))
    tissue_list = []
    refresh_link = ""
    tissue_sets = []
    tissue_set_idx = {}
    for choice in ws.get_tissue_set_choices():
        class TS:
            def __init__(self,choice):
                self.ts_id = choice[0]
                self.label = choice[1]
                self.counts = [0,0,0,0]
                self.rnaseq_counts = [0,0,0,0]
            def count_tissue(self,tissue):
                for idx in tissue.next_action.batches():
                    self.counts[idx] += 1
                    if Tissue.methods[tissue.get_method_idx()] == 'RNAseq':
                        self.rnaseq_counts[idx] += 1
            def labeled_counts(self):
                for i,(val,rnaval) in enumerate(zip(self.counts,self.rnaseq_counts)):
                    stage = 'sig' if i >= 2 else 'meta'
                    label = stage + ('' if val == 1 else 's')
                    yield (
                            i,
                            val,
                            label,
                            ''
                            )
                    if i == 1 or i == 3:
                        yield (
                                i,
                                rnaval,
                                label,
                                '|rnaseq'
                                )
                        yield (
                                i,
                                val-rnaval,
                                label,
                                '|no-rnaseq'
                                )
        tmp = TS(choice)
        tissue_sets.append(tmp)
        tissue_set_idx[tmp.ts_id] = tmp
    logger.debug('tissues view top of tissue loop: %s',ltmr.get())
    from .utils import TissueAction
    for tissue in tissue_qs:
        setattr(tissue
            ,'next_action'
            ,TissueAction(tissue,jcc,tac)
            )
        if tissue.tissue_set_id:
            tissue_set_idx[tissue.tissue_set_id].count_tissue(tissue)
        if tissue.next_action.phase == 'abort':
            refresh_link = link("(refresh)",url_config.here_url({
                                'sort':None,
                                }))
        tissue_list.append(tissue)
    logger.debug('tissues view bottom of tissue loop: %s',ltmr.get())
    if sort_type == 'status':
        tissue_list.sort(key=lambda x: x.next_action.description)
    elif sort_type == 'ts':
        # Excluded will have ids of None, convert to 0 for sorting.
        tissue_list.sort(key=lambda x: x.tissue_set_id or 0)
    elif sort_type == 'run':
        tissue_list.sort(key=lambda x: x.next_action.last_run_time)
    elif sort_type == 'acc':
        tissue_list.sort(key=lambda x: x.geoID,reverse=True)
    elif sort_type == 'name':
        tissue_list.sort(key=lambda x: x.name.lower(),reverse=True)
    else:
        assert sort_type == 'id' # the default
    request.session[session_sort_key] = sort_type
    # pre-load empty forms; these will be selectively overridden in POST
    from .forms import WsTissueForm, TissueFileForm
    tissue_form = WsTissueForm(ws)
    tissue_file_form = TissueFileForm(ws)
    from .utils import prep_tissue_set_id,remove_old_meta_results
    from browse.models import Sample
    from runner.models import Process
    import json
    if request.method == 'POST' and post_ok(request):
        if 'op' in request.GET:
            if request.GET['op'] == 'new_tissue':
                tissue_form = WsTissueForm(ws,request.POST)
                if tissue_form.is_valid():
                    p = tissue_form.cleaned_data
                    # strip input fields
                    for fld in "geo_id tissue".split():
                        p[fld]=p[fld].strip()
                    # construct 'source' field
                    try:
                        db_idx = Tissue.get_db_idx(p['geo_id'])
                    except ValueError as ex:
                        tissue_form.add_error('geo_id'
                                    ,'Must begin with E-, GSE, or GDS'
                                    )
                        db_idx = None
                    # save
                    if db_idx is not None:
                        r = Tissue(ws=ws
                                ,name=p['tissue']
                                ,geoID=p['geo_id']
                                ,source=Tissue.build_normal_source(
                                                    db_idx,
                                                    int(p['source']),
                                                    )
                                ,tissue_set_id=prep_tissue_set_id(p),
                                )
                        r.save()

                        if p['tissue_note'].strip():
                            from notes.models import Note
                            Note.set(r,
                                     'note',
                                     request.user.username,
                                     p['tissue_note'],
                                     )


                        if r.is_bioproject():
                            # We need to pull down sample data from bigquery
                            # Normally this happens during the search process,
                            # as it is much more efficient to pull in bul, but
                            # if you're adding it manually we need to pull this
                            # one down explicitly.
                            from dtk.sra_bigquery import SraBigQuery
                            sbq = SraBigQuery()
                            sbq.search_projects([r.geoID])

                        # clear out any stale sample results
                        Sample.objects.filter(tissue=r).delete()
                        jcc.queue_job('meta',r.get_meta_jobname()
                                        ,user=request.user.username
                                        ,settings_json=json.dumps({
                                                            'tissue_id':r.id,
                                                            })
                                        )
                        Process.drive_background()
                        return HttpResponseRedirect(ws.reverse('ge:tissues'))
            elif request.GET['op'] == 'multiple':
                # update GEO id to include preferred platform
                r = Tissue.objects.get(id=request.GET['tissue_id'])
                plat = request.GET['platform'].strip()
                r.geoID += (':'+plat)
                r.save()
                # clear out any stale sample results
                Sample.objects.filter(tissue=r).delete()
                jcc.queue_job('meta',r.get_meta_jobname()
                        ,user=request.user.username
                        ,settings_json=json.dumps({
                                            'tissue_id':r.id,
                                            })
                        )
                Process.drive_background()
                return HttpResponseRedirect(ws.reverse('ge:tissues'))
            elif request.GET['op'] == 'import':
                tissue_file_form = TissueFileForm(ws,request.POST)
                if tissue_file_form.is_valid():
                    p = tissue_file_form.cleaned_data
                    from dtk.s3_cache import S3File
                    s3file = S3File('sigprot',p['tsv'])
                    s3file.fetch()
                    enum = Tissue.source_vals
                    r = Tissue(ws=ws
                            ,name=p['tsv'][:-4] # strip '.tsv'
                            ,geoID='ext'
                            ,source=enum.get('metaGEO_parm',enum.EXT)
                            ,tissue_set_id=prep_tissue_set_id(p),
                            )
                    r.save()
                    # The expected input file format matches the legacy
                    # browse_significantprotein table, where columns are:
                    # 1 - (ignored)
                    # 2 - uniprot
                    # 3 - (ignored)
                    # 4 - evidence
                    # 5 - direction
                    # 6 - (ignored)
                    # 7 - fold_change
                    # and no header.
                    # So, just process it using the same code that imports
                    # from sigGEO.
                    bji=r._prep_foreground_sig(request.user.username)
                    bji._convert_sig_result(s3file.path())
                    bji.finalize()
                    return HttpResponseRedirect(ws.reverse('ge:tissues'))
            elif request.GET['op'] == 'meta':
                tissue = Tissue.objects.get(pk=request.POST['tissue_id'])
                remove_old_meta_results(tissue)
                jcc.queue_job('meta',tissue.get_meta_jobname()
                        ,user=request.user.username
                        ,settings_json=json.dumps({
                                'tissue_id':tissue.id,
                                })
                        )
                Process.drive_background()
                return HttpResponseRedirect(ws.reverse('ge:tissues'))
            elif request.GET['op'] == 'edit':
                tissue = Tissue.objects.get(pk=request.POST['tissue_id'])
                return HttpResponseRedirect(ws.reverse('ge:classify',tissue.id))
            elif request.GET['op'] == 'abort':
                tissue = Tissue.objects.get(pk=request.POST['tissue_id'])
                latest_jobs = jcc.latest_jobs()
                for jobname in (
                            tissue.get_meta_jobname(),
                            tissue.get_sig_jobname(),
                            ):
                    if jobname in latest_jobs:
                        abort_job = latest_jobs[jobname]
                        if abort_job.status in Process.active_statuses:
                            Process.abort(abort_job.id)
                # do redirect to render page with updated status
                return HttpResponseRedirect(ws.reverse('ge:tissues'))
            else:
                raise NotImplementedError(
                        "unrecognized op:"+request.GET['op']
                        )
        else:
            btns = [x
                    for x in request.POST
                    if x.startswith('btn_batch_')
                    ]
            if len(btns) == 1:
                tmp1,tmp2,batch,ts_id = btns[0].split('_')
                if '|' in ts_id:
                    ts_id,subset = ts_id.split('|')
                else:
                    subset = None
                ts_id = int(ts_id)
                batch = int(batch)
                any = False
                # Track which IDs we've already queued, so that we don't double-run any dupes.
                geos_ran = set()
                for tissue in tissue_list:
                    if tissue.tissue_set_id != ts_id:
                        continue
                    if batch not in tissue.next_action.batches():
                        continue
                    if subset is not None:
                        subset_rnaseq = subset == 'rnaseq'
                        method_rnaseq = Tissue.methods[tissue.get_method_idx()] == 'RNAseq'
                        if subset_rnaseq != method_rnaseq:
                            logger.info(f"Skipping tissue {tissue.geoID} {tissue.id} because of method {method_rnaseq}")
                            continue
                    level = 'meta' if batch < 2 else 'sig'
                    if level == 'meta':
                        if tissue.geoID in geos_ran:
                            logger.info(f"Skipping tissue {tissue.geoID} {tissue.id} which is previously queued.")
                            continue
                        geos_ran.add(tissue.geoID)
                    logger.info("Doing tissue %s %s", tissue.id, tissue.source)
                    any = True
                    if level == 'meta':
                        remove_old_meta_results(tissue)
                    minimal_settings = {
                                    'tissue_id':tissue.id,
                                    }
                    if level == 'sig':
                        from .forms import sigGEO_settings
                        settings = sigGEO_settings(ws)
                        d = settings.as_dict()
                        d['miRNA'] = tissue.tissue_set.miRNA
                        minimal_settings.update(d)
                    kwargs = {
                        'user':request.user.username,
                        'settings_json': json.dumps(minimal_settings),
                        }
                    if level == 'meta':
                        jcc.queue_job(level,tissue.get_meta_jobname(),**kwargs)
                    elif level == 'sig':
                        from .utils import upload_samples
                        try:
                            upload_samples(tissue)
                        except IOError as ex:
                            from django.contrib import messages
                            messages.add_message(request, messages.INFO,
                                    'Need to run meta for %s (%d)'%(
                                            tissue.name,
                                            tissue.id,
                                    ))
                        else:
                            jcc.queue_job(
                                    level,
                                    tissue.get_sig_jobname(),
                                    **kwargs
                                    )
                    else:
                        raise NotImplementedError
                if any:
                    Process.drive_background()
            else:
                raise NotImplementedError(
                        "unrecognized POST:"+str(request.POST)
                        )
            return HttpResponseRedirect(ws.reverse('ge:tissues'))
    logger.debug('tissues view before render: %s',ltmr.get())
    result = render(request
                ,'ge/tissues.html'
                , make_ctx(request,ws,'ge:tissues',
                    {'tissue_list':reversed(tissue_list)
                    ,'tissue_form':tissue_form
                    ,'tissue_file_form':tissue_file_form
                    ,'refresh_link':refresh_link
                    ,'excluded_link':excluded_link
                    ,'result_heading':Tissue.sig_count_heading()
                    ,'tissue_sets':tissue_sets
                    ,'url_config':url_config
                    ,'sort_type':sort_type
                    ,'job_cross_checker':jcc
                    }
                  )
                )
    if hasattr(jcc,'bad_outliers'):
        from django.contrib import messages
        messages.add_message(request, messages.INFO,
                "check Quality column for bad outlier files",
                )
    logger.debug('tissues view total time: %s',ltmr.lap())
    for bucket,elapsed in tac.btmr.get():
        logger.debug('tissues view bucket %s: %s',bucket,elapsed)
    return result

class NoteTissueView(DumaView):
    template_name='ge/note_tissue.html'
    button_map={
            'modify':['tissue_edit'],
            'meta':['tissue_edit'],
            'delete':[],
            }
    def custom_context(self):
        self.context_alias(
                note_tissue=self.tissue,
                )
    def make_tissue_edit_form(self,data):
        from .forms import TissueEditForm
        return TissueEditForm(data,instance=self.tissue)
    def save_tissue_edits(self):
        p = self.tissue_edit_form.cleaned_data
        logger.info("tissue modify post %s %s",
                repr(p),
                self.tissue.id,
                )
        tissue = self.tissue
        if 'source' in p:
            # do this first, as it might throw an exception
            fallback_reason=p['fallback_reason'].strip()
            try:
                new_source = tissue.build_updated_source(
                        p['source'],
                        fallback_reason,
                        )
            except ValueError as ex:
                self.message('ERROR: '+str(ex)+'. Please correct form input.')
                return
            tissue.fallback_reason = fallback_reason
            if tissue.source != new_source:
                logger.info("modifying tissue source: %s %s %s",
                        repr(p['source']),
                        repr(tissue.source),
                        repr(new_source),
                        )
                from .utils import remove_old_meta_results
                remove_old_meta_results(tissue)
                # loop through all tissues matching this
                # source and geoID, and toggle all of them
                from browse.models import Tissue
                for t in Tissue.objects.filter(geoID=tissue.geoID):
                    if t.id != tissue.id:
                        t.source = new_source
                        t.fallback_reason = fallback_reason
                        t.save()
                    try:
                        jname = t.get_meta_jobname()
                        jrec = self.jcc.latest_jobs()[jname]
                        jrec.invalidated = True
                        jrec.save()
                    except KeyError:
                        pass
                tissue.source = new_source
        tissue.name = p['name'].strip()
        tissue.ignore_missing = p['ignore_missing']
        from .utils import prep_tissue_set_id
        tissue.tissue_set_id=prep_tissue_set_id(p)
        tissue.save()
        from notes.models import Note
        Note.set(tissue
                    ,'note'
                    ,self.request.user.username
                    ,p['note']
                    )
    def modify_post_valid(self):
        self.save_tissue_edits()
        return HttpResponseRedirect(self.ws.reverse('ge:tissues'))
    def meta_post_valid(self):
        self.save_tissue_edits()
        from .utils import remove_old_meta_results
        remove_old_meta_results(self.tissue)
        import json
        self.jcc.queue_job('meta',self.tissue.get_meta_jobname()
                        ,user=self.request.user.username
                        ,settings_json=json.dumps({
                                'tissue_id':self.tissue.id,
                                })
                        )
        from runner.models import Process
        Process.drive_background()
        return HttpResponseRedirect(self.ws.reverse('ge:tissues'))
    def delete_post_valid(self):
        if 'delete_confirm' in self.request.POST:
            self.tissue.invalidate()
            return HttpResponseRedirect(self.ws.reverse('ge:tissues'))
        self.message('You must check the confirm box to delete a tissue')

class ClassifyView(DumaView):
    template_name='ge/classify.html'
    index_dropdown_stem='ge:tissues'
    button_map={
            'reconf':[],
            'process':['sig','tissue_edit'],
            'modify':['tissue_edit']
            }
    def custom_setup(self):
        from .utils import upload_samples
        try:
            self.context_alias(outliers = upload_samples(self.tissue))
        except IOError as ex:
            from django.utils.html import format_html
            self.message(format_html(
                '<h3>Re-run meta, or wait for meta to complete.</h3><br>{}',
                str(ex),
                ))
            return render(self.request,self.template_name,self.context)
        from .utils import SampleSorter
        self.ss = SampleSorter(self.tissue)
        self.context['sample_sorter'] = self.ss
    def reconf_post_valid(self):
        prefix = "reconfig_"
        s = ['N'] * len(self.ss.selector)
        for key in self.request.POST:
            if key.startswith(prefix):
                s[int(key[len(prefix):])-1] = 'Y'
        s = "".join(s)
        self.tissue.cc_selected = s
        self.tissue.save()
        return HttpResponseRedirect(self.here_url())
    def make_tissue_edit_form(self,data):
        from .forms import TissueEditForm
        return TissueEditForm(data,instance=self.tissue,skip_source=True)
    def modify_post_valid(self):
        p = self.tissue_edit_form.cleaned_data
        self.tissue.name = p['name'].strip()
        from .utils import prep_tissue_set_id
        self.tissue.tissue_set_id=prep_tissue_set_id(p)
        self.tissue.save()
        from notes.models import Note
        Note.set(self.tissue
                    ,'note'
                    ,self.request.user.username
                    ,p['note']
                    )
        return HttpResponseRedirect(self.here_url())
    def make_sig_form(self,data):
        from .forms import sigGEO_settings
        from browse.models import Species
        opts = sigGEO_settings(self.ws)
        try:
            initial_species = self.tissue.tissue_set.species
        except AttributeError:
            initial_species = None # defaults to first in list
        class MyForm(forms.Form):
# We could clean this up with a for loop and setattr() if, in sigGEO_settings, we set which type of form it applied to
            species = forms.ChoiceField(
                    label = opts.species.label,
                    choices = Species.choices(),
                    initial = initial_species,
                    )
            algo = forms.ChoiceField(
                    label = opts.algo.label,
                    choices = opts.algo.choices,
                    initial = opts.algo.initial
                    )
            scRNAseq = forms.BooleanField(required=opts.scRNAseq.required,
                    label=opts.scRNAseq.label,
                    initial=opts.scRNAseq.initial,
                    )
            runSVA = forms.BooleanField(required=opts.runSVA.required,
                    label=opts.runSVA.label,
                    initial=opts.runSVA.initial,
                    )
            top1thresh = forms.FloatField(
                    label=opts.top1thresh.label,
                    initial=opts.top1thresh.initial,
                    )
            permut = forms.IntegerField(
                    label=opts.permut.label,
                    initial=opts.permut.initial,
                    )
            minUniPor = forms.FloatField(
                    label=opts.minUniPor.label,
                    initial=opts.minUniPor.initial,
                    )
            minDirPor = forms.FloatField(
                    label=opts.minDirPor.label,
                    initial=opts.minDirPor.initial,
                    )
            minCPM = forms.IntegerField(
                    label=opts.minCPM.label,
                    initial=opts.minCPM.initial,
                    )
            minReadPor = forms.FloatField(
                    label=opts.minReadPor.label,
                    initial=opts.minReadPor.initial,
                    )
            ignoreMissing = forms.BooleanField(
                    required=opts.ignoreMissing.required,
                    label=opts.ignoreMissing.label,
                    initial=opts.ignoreMissing.initial
                    )
            debug = forms.BooleanField(required=opts.debug.required,
                    label=opts.debug.label,
                    initial=opts.debug.initial
                    )
            mirMappingFile = forms.ChoiceField(
                    label = opts.mirMappingFile.label,
                    choices = opts.mirMappingFile.choices,
                    initial = opts.mirMappingFile.initial,
                    )
        return MyForm(data)
    def process_post_valid(self):
        # save any tissue modifications (ignore redirect return)
        self.modify_post_valid()
        # XXX the update below might go inside the form processing, so
        # XXX all buttons don't revert on an invalid form
        # iterate through all radio buttons and update database
        self.ss.build_selected(self.request.POST['selector'])
        for key in self.request.POST:
            self.ss.update_records_from_key(key,self.request.POST[key])
        # verify at least one each of case and control
        from browse.models import Sample
        cases = self.tissue.sample_set.filter(
                    classification=Sample.group_vals.CASE
                    ).count()
        controls = self.tissue.sample_set.filter(
                    classification=Sample.group_vals.CONTROL
                    ).count()
        if not (cases and controls):
            self.message(
                "ERROR: At least one case and one control are required"
                )
            return HttpResponseRedirect(self.here_url())
        # start background
        settings = self.sig_form.cleaned_data
        settings['tissue_id'] = self.tissue.id
        settings['miRNA'] = self.tissue.tissue_set.miRNA
        from browse.models import Species
        # run_sig wants the species name rather than index.
        settings['species'] = Species.get('label', int(settings['species']))
        import json
        self.jcc.queue_job('sig',self.tissue.get_sig_jobname()
                        ,user=self.request.user.username
                        ,settings_json=json.dumps(settings)
                        )
        from runner.models import Process
        Process.drive_background()
        return HttpResponseRedirect(self.ws.reverse('ge:tissues'))

