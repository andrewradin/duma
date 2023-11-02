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

class AeSearchView(DumaView):
    template_name = 'ge/ae_search.html'
    index_dropdown_stem = 'ge:ae_search'
    warn_level = 500
    def custom_context(self):
        from browse.models import AeSearch
        from dtk.url import ge_eval_link
        self.context_alias(
            search_list = AeSearch.objects.filter(ws=self.ws).order_by('-id'),
            ge_eval_url = ge_eval_link(),
            search_url = self.ws.reverse(
                    'nav_job_start',
                    f'aesearch_{self.ws.id}',
                    ),
            )

class AeBulkView(DumaView):
    template_name='ge/ae_bulk.html'
    button_map={
            'reject':[],
            'rejectall':[],
            }
    GET_parms={
            'limit':(int, 100),
            }
    def custom_context(self):
        from dtk.url import ge_eval_link
        score_list=self.get_score_list()
        got = len(score_list)
        if got > self.limit:
            label = f'First {self.limit} of {got}'
        else:
            label = f'All {got}'
        self.context_alias(
                score_list=score_list[:self.limit],
                ge_eval_url = ge_eval_link(),
                quant_label = label,
                )
    def get_score_list(self):
        imported = self.search._imported_geo_ids()
        return [
                x
                for x in self.search.aescore_set.order_by(
                        '-score',
                        '-accession_id',
                        )
                if x.accession.geoID not in imported
                and not x.accession.reject_text(self.search.mode)
                ]
    def get_rejections(self):
        prefix = 'rej_'
        result = []
        for k,v in self.request.POST.items():
            v = v.strip()
            if not k.startswith(prefix) or not v:
                continue
            result.append((int(k[len(prefix):]),v))
        return result
    def rejectall_post_valid(self):
        if self.get_rejections():
            self.message('''
                You had reject reasons pending. Use the browser 'back' button
                and hit 'Update All' to save them, or try again to reject
                everything.
                ''')
        elif 'confirm' not in self.request.POST:
            self.message('''
                You must check the 'Yes, really!' checkbox for this to take
                effect.
                ''')
        else:
            reason="low score, not individually reviewed"
            for item in self.get_score_list():
                self.reject_one(item,reason)
    def reject_post_valid(self):
        from browse.models import AeScore
        for aescore_id,reason in self.get_rejections():
            item=AeScore.objects.get(pk=aescore_id)
            self.reject_one(item,reason)
        return HttpResponseRedirect(self.here_url())
    def reject_one(self,item,reason):
        aed,new=item.accession.aedisposition_set.update_or_create(
                accession=item.accession,
                mode=self.search.mode,
                defaults=dict(
                        rejected = reason,
                        ),
                )
        logger.info("Bulk Rejecting '%s' mode %d (new %d): '%s'",
                aed.accession.geoID,
                aed.mode,
                new,
                aed.rejected,
                )

@login_required
def ae_list(request,ws_id,search_id):
    from browse.models import Workspace
    ws = Workspace.objects.get(pk=ws_id)
    from browse.models import AeScore, AeSearch
    score_list = AeScore.objects.filter(
                        search_id=search_id,
                        ).order_by('-score','-accession_id')
    search = AeSearch.objects.get(pk=search_id)
    imported = search._imported_geo_ids()
    detail = None

    targ_detail_id = request.GET.get('detail', None)

    from dtk.ae_parser import disease
    dis = disease(term=search.term, ws=ws_id)
    dis.get_highlights()

    for i,item in enumerate(score_list):
        import numpy as np
        item.score = np.round(item.score, 3)
        if item.accession.geoID == targ_detail_id:
            detail = item
            detail_rank = i+1
            detail_type = 'Selected'

        if item.accession.geoID in imported:
            item.status = "Imported"
        else:
            rejected = item.accession.reject_text(search.mode)
            if rejected:
                item.status = "Rejected: "+rejected
                item.clear_btn = True
            else:
                item.status = "Needs Review"
                if detail is None and targ_detail_id is None:
                    detail = item
                    detail_rank = i+1
                    detail_type = 'Unreviewed'
    if detail:
        from ge.forms import WsTissueForm
        tissue_form = WsTissueForm(ws)
        geoID = detail.accession.geoID
        if geoID.startswith('E-GEOD-'):
            geoID = "GSE"+geoID[7:]
        tissue_form.initial['geo_id'] = geoID
        tissue_form.initial['source'] = detail.accession.src_default

        samples = detail.accession.get_sample_attrs()
        import pandas as pd
        df = pd.DataFrame(samples)
        from dtk.table import Table
        detail_sample_table = Table.from_dataframe(df)
    if request.method == 'POST' and post_ok(request):
        from browse.models import AeDisposition
        if 'reject_btn' in request.POST:
            aed,new=AeDisposition.objects.update_or_create(
                    accession_id=request.POST['item_id'].strip(),
                    mode=search.mode,
                    defaults=dict(
                            rejected = request.POST['reason'].strip(),
                            ),
                    )
            logger.info("Rejecting '%s' mode %d (new %d): '%s'",
                    aed.accession.geoID,
                    aed.mode,
                    new,
                    aed.rejected,
                    )
            return HttpResponseRedirect(ws.reverse('ge:ae_list',search_id))
        elif 'clear_btn' in request.POST:
            AeDisposition.objects.filter(
                    accession_id=request.POST['item_id'],
                    mode=search.mode,
                    ).delete()
            return HttpResponseRedirect(ws.reverse('ge:ae_list',search_id))
        else:
            raise Exception("unimplemented post operation")
    from dtk.data import merge_dicts,dict_subset
    from dtk.url import ge_eval_link
    ge_eval_url = ge_eval_link()
    return render(request
                ,'ge/ae_list.html'
                ,make_ctx(request,ws,'ge:ae_search',
                        merge_dicts(
                                {
                                },
                                dict_subset(locals(),(
                                            'detail',
                                            'detail_rank',
                                            'search',
                                            'score_list',
                                            'tissue_form',
                                            'ge_eval_url',
                                            'dis',
                                            'detail_type',
                                            'detail_sample_table',
                                            )),
                                )
                        )
                )


class SearchModelView(DumaView):
    template_name='ge/searchmodel.html'
    GET_parms={
            'fold_idx':(int, 0),
            }
    button_map={
                'foldselect':['foldselect'],
                }
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
        from runner.process_info import JobInfo
        import gzip
        import pickle
        self.bji = JobInfo.get_bound(self.ws, self.job)
        self.data = self.bji.load_xval_data()
        self.num_folds = len(self.data)
    def custom_context(self):
        data = self.data[self.fold_idx]

        table_data = data['table'] 
        header = table_data[0]
        rows = table_data[1:]

        search_cache = {}

        for row in rows:
            vals = dict(zip(header, row))
            key, term, wsname = vals['Key'], vals['Search Term'], vals['WS']
            from browse.models import AeAccession, AeSearch
            if (term, wsname) not in search_cache:
                srch = AeSearch.objects.filter(term=term, ws__name=wsname)[0]
                search_cache[(term, wsname)] = srch
            srch = search_cache[(term, wsname)]
            from dtk.html import link
            url = srch.ws.reverse('ge:ae_list', srch.id) + f"?detail={key}"

            row[0] = link(key, url)

        from dtk.table import Table
        cols = [Table.Column(x, idx=i) for i, x in enumerate(header)]

        table = Table(rows, cols)




        # term, title, description, nsamples, score, label
        # Pre-computed for the fold models?
        # model explainer?  top pos/neg features?
        # example explainer?  feature importance?
        # label overrides.

        self.context_alias(tables=list(enumerate([table])))
