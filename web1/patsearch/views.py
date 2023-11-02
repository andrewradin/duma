from django.shortcuts import render
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST

from django import forms
from django.http import HttpResponseRedirect

from dtk.duma_view import DumaView,list_of,boolean

from dtk.table import Table
from dtk.html import link
from dtk.duma_view import qstr

from .models import PatentSearch
import logging
logger = logging.getLogger(__name__)

class SummaryView(DumaView):
    template_name='patsearch/summary.html'
    def handle_pat_search_id_arg(self,search_id):
        search = PatentSearch.objects.get(pk=search_id)
        self.context_alias(
                search=search,
                )

def make_resolve_opts():
    from patsearch.models import PatentSearchResult
    enum = PatentSearchResult.resolution_vals
    opts = [
        (enum.RELEVANT, 'Relevant To Drug+Disease', 'primary'),
        (enum.IRRELEVANT_DRUG, 'Irrelevant To Drug', 'info'),
        (enum.IRRELEVANT_DISEASE, 'Irrelevant To Disease', 'info'),
        (enum.IRRELEVANT_ALL, 'Totally Irrelevant', 'info'),
        (enum.NEEDS_MORE_REVIEW, 'Needs More Review', 'default'),
        (enum.SKIPPED, 'Skip', 'default'),
    ]
    return opts

class ResolveView(DumaView):
    template_name='patsearch/resolve.html'
    def handle_pat_dd_search_id_arg(self,search_id):
        from .models import DrugDiseasePatentSearch, PatentSearchResult
        import json
        search = DrugDiseasePatentSearch.objects.get(pk=search_id)
        resolve_opts = json.dumps(make_resolve_opts())
        self.context_alias(
                res_enum=PatentSearchResult.resolution_vals,
                search=search,
                resolve_opts=resolve_opts
                )

class SearchView(DumaView):
    template_name='patsearch/search.html'
    index_dropdown_stem='pats_search'
    button_map = {
            'search': []
            }
    def custom_context(self):
        from .patent_search import BQ_TABLES
        self.context_alias(
                past_searches = PatentSearch.objects.filter(
                        ws=self.ws,
                        ).order_by('-id'),
                table_choices = enumerate(BQ_TABLES),
                )

    def custom_setup(self):
        drug_names = None
        aliases = self.ws.get_disease_aliases()
        if isinstance(aliases, str):
            aliases = [aliases]
        default_search = {
                'diseaseNames': list(aliases),
                'drugNames': drug_names
                }
        import json
        self.default_search_json = json.dumps(default_search)


    def search_post_valid(self):
        import json
        query_str = self.request.POST['query']
        query = json.loads(query_str)

        settings = {
                'query': query,
                'ws_id': self.ws.id,
                }
        p_id = self.jcc.queue_job('patsearch', 'patsearch',
                user=self.username(),
                settings_json=json.dumps(settings))
        from runner.models import Process
        Process.drive_background()
        return JsonResponse({
            'next_url': self.ws.reverse('nav_progress',p_id)
            })



@login_required
def preview_search(request, ws_id):
    import json
    query_str = request.POST['query']
    query = json.loads(query_str)

    from .patent_search import patent_search
    search, search_results = patent_search([query['drugNames'],
                                            query['diseaseNames']], 10)

    from django.core import serializers
    from django.forms.models import model_to_dict
    search = model_to_dict(search)
    patents = {x.patent.pub_id: model_to_dict(x.patent) for x in search_results}
    search_results = [model_to_dict(x) for x in search_results]

    return JsonResponse({
        'search': search,
        'results': search_results,
        'patents': patents,
    })

@require_POST
@login_required
def resolve_patent(request, ws_id, patent_result_id, resolution):
    from patsearch.patent_search import apply_patent_resolution
    search_result = apply_patent_resolution(patent_result_id, resolution)

    return JsonResponse({'newStatus': search_result.resolution_text})


@login_required
def patent_details(request, ws_id, patent_result_id):
    from .models import PatentSearchResult
    patent_result = PatentSearchResult.objects.get(pk=patent_result_id)

    from .patent_search import PatentContentJob, PatentContentStorage
    storage = PatentContentStorage('/tmp/pats')

    best_patent_content = storage.find_best_content(patent_result.patent)
    if not best_patent_content:
        content = {
                'available': False
                }
    else:
        content = storage.load_patent_content(best_patent_content)
        content['available'] = True

    response = content
    import json
    response['evidence'] = json.loads(patent_result.evidence)

    return JsonResponse(response)


def _get_drug(wsa, targ_cache):
    from browse.models import WsAnnotation
    agent = wsa.agent
    target_terms = []
    for uniprot, gene, direction in targ_cache.info_for_agent(agent.id):
        from browse.models import Protein
        target_terms += [gene]
        try:
            prot = Protein.objects.get(uniprot=uniprot)
            all_prot_names = [prot.get_name()] + [x for x in prot.get_alt_names() if not x.startswith('EC ')]
            target_terms += all_prot_names
        except Protein.DoesNotExist:
            # Apparently we have dpi prots that aren't in our prot info.
            gene_name = None

    drug_terms = list(set([agent.canonical] + list(agent.synonym_set) + list(agent.brand_set)))
    # Canonical first; then alphabetical.
    drug_terms.sort(key=lambda x: (x != agent.canonical, x))
    out = {
        'wsa': wsa.id,
        'name': agent.canonical,
        'drug_terms': drug_terms,
        'target_terms': target_terms,
        'study_note': wsa.get_study_text(),
        'global_note': agent.get_bd_note_text()
        }
    return out

@login_required
def get_drug(request, ws_id, wsa_id):
    from dtk.prot_map import DpiMapping, AgentTargetCache
    from browse.models import Workspace, WsAnnotation
    ws = Workspace.objects.get(pk=ws_id)
    wsa = WsAnnotation.objects.get(pk=wsa_id)
    targ_cache = AgentTargetCache(
            mapping=DpiMapping(ws.get_dpi_default()).dpimap_for_wsa(wsa),
            agent_ids=[wsa.agent.id],
            dpi_thresh=ws.get_dpi_thresh_default(),
            )
    return JsonResponse(_get_drug(wsa, targ_cache))

@login_required
def get_drugset(request, ws_id, setname):
    from dtk.prot_map import DpiMapping, AgentTargetCache
    from browse.models import Workspace, WsAnnotation
    ws = Workspace.objects.get(pk=ws_id)
    wsa_ids = ws.get_wsa_id_set(setname)
    wsas = [WsAnnotation.objects.get(pk=x) for x in wsa_ids]

    targ_caches = [AgentTargetCache(
            mapping=DpiMapping(ws.get_dpi_default()).dpimap_for_wsa(wsa),
            agent_ids=[wsa.agent.id for wsa in wsas],
            dpi_thresh=ws.get_dpi_thresh_default(),
            ) for wsa in wsas]

    druginfo = [_get_drug(wsa, targ_cache) for wsa, targ_cache in zip(wsas, targ_caches)]
    return JsonResponse({'drugs': druginfo})


@login_required
def search_drugs(request, ws_id, query):
    from browse.models import WsAnnotation
    from browse.utils import drug_search_wsa_filter
    from django.db.models import Q

    data_for = lambda wsa: {'id': wsa.id, 'name': wsa.agent.canonical}

    # Start with drugs whose canonical matches our search.
    qs = WsAnnotation.objects.filter(
            ws=ws_id,
            agent__tag__prop__name='canonical',
            agent__tag__value__istartswith=query,
            )
    qs = qs[:10]
    qs = WsAnnotation.prefetch_agent_attributes(qs, use_id_prefetch=True)
    out = [data_for(wsa) for wsa in qs]
    used_wsa = set([wsa.id for wsa in qs])

    # If we can't fill out the list with those, then do the general search.
    if len(qs) < 10:
        N = 10 - len(qs)
        qs_ws = WsAnnotation.objects.filter(ws=ws_id)
        qs = drug_search_wsa_filter(qs_ws,query)
        qs = WsAnnotation.prefetch_agent_attributes(qs, use_id_prefetch=True)
        out += [data_for(wsa) for wsa in qs[:N] if wsa.id not in used_wsa]


    return JsonResponse({'names': out})
