from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
import sys
import logging

logger = logging.getLogger(__name__)

@login_required
def prot_search(request):
    from dtk.prot_search import search_by_any, search_geneprot_by_start
    pattern = request.GET['search'].strip()

    if request.GET.get('type', None) == 'start':
        searcher = search_geneprot_by_start
    else:
        searcher = search_by_any

    limit = int(request.GET.get('limit', 100))
    # We limit this because otherwise if you search for e.g. '', it will
    # bog down the system for several minutes as it returns every prot.
    results, reached_limit = searcher(pattern, limit=limit)

    matches = [{
        'uniprot': p.uniprot,
        'gene': p.gene,
        'name': p.get_name(),
        } for p in results]
    return JsonResponse({
        'matches': matches,
        'reached_limit': reached_limit
        })

@login_required
def uniprot_lookup(request, uniprot):
    from browse.models import Protein
    prot = Protein.objects.get(uniprot=uniprot)
    return JsonResponse({
        'gene': prot.gene
        })

@login_required
def wsa_lookup(request, wsa_id):
    from browse.models import WsAnnotation
    wsa = WsAnnotation.objects.get(pk=wsa_id)
    ws = wsa.ws
    from dtk.prot_map import DpiMapping, AgentTargetCache, MultiAgentTargetCache
    dpi = DpiMapping(ws.get_dpi_default()).get_baseline_dpi().dpimap_for_wsa(wsa)

    atc = AgentTargetCache(
                mapping=dpi,
                dpi_thresh=ws.get_dpi_thresh_default(),
                agent_ids=[wsa.agent.id]
                )
    matc = MultiAgentTargetCache(atc)
    out = drug_and_target_data(wsa.agent, matc)

    return JsonResponse({'data': out})


@login_required
def global_data_prot_search(request):
    from browse.models import Protein
    # Data is too big for a GET, even though this is just a lookup.
    query = request.POST['query']

    from dtk.prot_search import parse_global_data, bulk_find_by_gene
    parsed_query = parse_global_data(query)

    all_target_ids = set([target_id
                        for drug in parsed_query
                        for target in drug['targets']
                        for target_id in target['ids']
                        ])

    # Right now the only thing we do with identifiers is assume they're a
    # gene name and search for them.
    found_prots = bulk_find_by_gene(all_target_ids)
    from collections import defaultdict
    out_genes = defaultdict(list)
    for p in found_prots:
        out_genes[p.gene].append([p.uniprot, p.get_name()])

    return JsonResponse({
        'parsed': parsed_query,
        'targetData': out_genes
    })

def drug_and_target_data(drug, matc):
    """Consistent formatting of drug->target data."""
    atc = matc.atc_for_agent(drug.id)
    if atc is None:
        bindings = []
    else:
        bindings = atc.info_for_agent(drug.id)
    return {
            'name': drug.canonical,
            'id': drug.id,
            'targets': bindings,
            }

@login_required
def search_drugs(request, name):
    if len(name) < 2:
        # No single letter searches, can bog down system.
        return JsonResponse({'data': []})
    from django.db.models import Q, Min
    from drugs.models import Drug, DpiMergeKey
    from dtk.prot_map import DpiMapping, AgentTargetCache, MultiAgentTargetCache
    from browse.default_settings import DpiDataset
    dpi = DpiMapping(DpiDataset.value(ws=None)).get_baseline_dpi()

    # Start with drugs whose canonical matches our search.
    # We use DpiMergeKey and some annotate magic to de-dupe.
    qs = DpiMergeKey.objects.filter(
            Q(drug__tag__prop__name='canonical') | Q(drug__tag__prop__name='override_name'),
            drug__tag__value__istartswith=name,
            ).values('dpimerge_key').distinct().annotate(drug=Min('drug')).values_list('drug', flat=True)
    qs = list(qs[:10])
    qs = Drug.objects.filter(pk__in=qs)

    atc = AgentTargetCache(
                mapping=dpi,
                dpi_thresh=dpi.default_evidence,
                agent_ids=[x.id for x in qs]
                )
    matc = MultiAgentTargetCache(atc)
    out = [drug_and_target_data(drug, matc) for drug in qs]

    return JsonResponse({'data': out})

@login_required
def search_wsas(request, ws_id, name):
    if len(name) < 2:
        # No single letter searches, can bog down system.
        return JsonResponse({'data': []})
    from django.db.models import Q
    from browse.models import WsAnnotation, Workspace
    ws = Workspace.objects.get(pk=ws_id)

    from dtk.prot_map import DpiMapping, AgentTargetCache, MultiAgentTargetCache
    dpi = DpiMapping(ws.get_dpi_default()).get_baseline_dpi()

    # Start with drugs whose canonical matches our search.
    qs = WsAnnotation.objects.filter(
            Q(agent__tag__prop__name='canonical') | Q(agent__tag__prop__name='override_name'),
            agent__tag__value__istartswith=name,
            ws_id=ws_id,
            )
    qs = list(qs[:10])

    atc = AgentTargetCache(
                mapping=dpi,
                dpi_thresh=dpi.default_evidence,
                agent_ids=[x.agent.id for x in qs]
                )
    matc = MultiAgentTargetCache(atc)
    out = [{**drug_and_target_data(wsa.agent, matc), "wsa_id": wsa.id}
            for wsa in qs]

    return JsonResponse({'data': out})

@login_required
def list_workspaces(request):
    active_only = request.GET.get('active_only', False) == 'true'
    from browse.models import Workspace
    if active_only:
        ws_qs = Workspace.objects.filter(active=True).order_by('name')
    else:
        ws_qs = Workspace.objects.all().order_by('name')

    return JsonResponse({'data': list(ws_qs.values('name', 'id', 'active'))})

@login_required
def list_jobs(request):
    ws_id = request.GET['ws']
    job_type = request.GET['job_type']

    from browse.models import Workspace
    ws = Workspace.objects.get(pk=ws_id)

    jobs = ws.get_prev_job_choices(job_type)
    job_ids = [job[0] for job in jobs]

    from runner.models import Process
    procs = Process.objects.filter(pk__in=job_ids)
    proc_map = {p.id:p for p in procs}

    all_inputs_map = {}
    input_weights = {}
    all_mapping_type = {}
    # We could pull this for other job types too, but this
    # is all we use it for right now.
    if job_type == 'wzs':
        for jid in job_ids:
            from runner.process_info import JobInfo
            bji = JobInfo.get_bound(ws_id, jid)
            all_inputs_map[jid] = list(bji.get_all_input_job_ids())

    if job_type == 'wf':
        for jid in job_ids:
            from runner.process_info import JobInfo
            bji = JobInfo.get_bound(ws_id, jid)
            if 'RefreshFlow' in bji.job.name or 'AnimalModelFlow' in bji.job.name:
                ss = bji.get_scoreset()
                if ss:
                    all_inputs_map[jid] = list(ss.job_type_to_id_map().values())
                    input_weights[jid] = bji.get_scoreset_weights()

                from dtk.prot_map import DpiMapping
                dpi = DpiMapping(bji.parms['p2d_file'])
                mapping_type = dpi.mapping_type()
                if 'AnimalModelFlow' in bji.job.name:
                    mapping_type += ', animal'
                all_mapping_type[jid] = mapping_type


    def make_label(job):
        name = ws.get_short_name()
        mapping = all_mapping_type.get(job[0], '')
        if mapping:
            mapping = '(' + mapping + ')'
        return f'{name} {job[1]} {mapping}'

    return JsonResponse({'data': [{
                'label': make_label(job),
                'id': job[0],
                'parms': proc_map[job[0]].settings(),
                'all_input_jids': all_inputs_map.get(job[0], []),
                'input_weights': input_weights.get(job[0], {}),
            } for job in jobs]
        })

@login_required
def fetch_scores(request, ws_id, job_id):

    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws_id, job_id)
    dc = bji.get_data_catalog()
    if 'codes' in request.GET:
        codes = request.GET['codes'].split(',')
    else:
        codes = list(dc.get_codes('', ''))
    fvs = [list(x) for x in dc.get_feature_vectors(*codes)]


    # If this is a glf job...
    if bji.job_type == 'glf':
        from dtk.gene_sets import LEGACY_FN, legacy_genesets_name_to_id
        if bji.parms.get('std_gene_list_set') == LEGACY_FN:
            name2id = legacy_genesets_name_to_id()
            score_types, rows = fvs
            found = 0
            missing = 0
            new_rows = []
            for name, scores in rows:
                # Convert Inf's out; JSON doesn't support Infinity, old datasets sometimes have it.
                scores = [score if score != float('Inf') else 1e99 for score in scores]
                if name in name2id:
                    new_rows.append((name2id[name], scores))
                    found += 1
                else:
                    new_rows.append((name, scores))
                    missing += 1
            logger.info("Missing %d/%d legacy geneset conversions", missing, missing+found)
            fvs = (score_types, new_rows)

    return JsonResponse({
        'data': fvs,
        })

@login_required
def ws_molsets(request, ws_id):
    from browse.models import Workspace
    ws = Workspace.objects.get(pk=ws_id)
    data = ws.get_wsa_id_set_choices()
    return JsonResponse({'data': data})

@login_required
def ws_protsets(request, ws_id):
    from browse.models import Workspace
    ws = Workspace.objects.get(pk=ws_id)
    data = ws.get_uniprot_set_choices()
    return JsonResponse({'data': data})


@login_required
def molset(request, ws_id, molset_id):
    from browse.models import Workspace, WsAnnotation
    from dtk.prot_map import AgentTargetCache, MultiAgentTargetCache
    ws = Workspace.objects.get(pk=ws_id)
    wsa_ids= ws.get_wsa_id_set(molset_id)
    wsas = WsAnnotation.objects.filter(pk__in=wsa_ids)
    atc = AgentTargetCache.atc_for_wsas(wsas, ws=ws)
    matc = MultiAgentTargetCache(atc)
    out = [drug_and_target_data(wsa.agent, matc) for wsa in wsas]
    return JsonResponse({'data': out})

@login_required
def protset(request, ws_id, protset_id):
    from browse.models import Workspace, Protein
    ws = Workspace.objects.get(pk=ws_id)
    unis = ws.get_uniprot_set(protset_id)
    genes = Protein.make_gene_list_from_uniprot_list(unis)
    return JsonResponse({'data': list(zip(unis, genes))})

@login_required
def indirect_targets(request, target_list):
    target_list = target_list.split(',')
    from dtk.prot_map import PpiMapping
    from browse.default_settings import PpiDataset
    from browse.models import Protein
    ppi = PpiMapping(PpiDataset.value(ws=None))

    ind_prots = [x[1] for x in ppi.get_ppi_info_for_keys(target_list, min_evid=0.9)]
    genes = Protein.make_gene_list_from_uniprot_list(ind_prots)

    out = [{'gene': gene, 'uniprot': prot} for prot, gene in zip(ind_prots, genes)]

    return JsonResponse({'targets': out})

@login_required
def pathway_data(request):
    from browse.models import Protein

    p2g = Protein.get_uniprot_gene_map()
    from dtk.gene_sets import get_pathway_data

    protsets, pathways_data, hier = get_pathway_data()
    pathway_to_name = {p['id']: p['name'] for p in pathways_data.values()}

    out = JsonResponse({'data': {
        'protsets': protsets,
        'idToName': pathway_to_name,
        'hierarchy': hier,
        'pathways': pathways_data,
        'prot2gene': p2g,
        }
    })

    return out

@login_required
def validate_score_json(request):
    from algorithms.run_customsig import parse_score_json
    data = request.POST['data']
    try:
        mapped_data, full_table = parse_score_json(data)
    except Exception as e:
        return HttpResponse("Errors:\n" + str(e))

    return HttpResponse("Valid scores!\n" + '\n'.join('\t'.join(str(v) for v in x) for x in full_table))