
import logging
logger = logging.getLogger(__name__)

def mol_selector(name,ws):
    if name == 'reviewed':
        return set(reviewed_mols_ids(ws))
    if name == 'final_review':
        return set(final_review_mols_ids(ws))
    if name == 'selected':
        return set(selected_mols_ids(ws))
    if name == 'screened':
        return set(screened_mols_ids(ws))
    if name == 'failed_screen':
        return set(failed_screen_mols_ids(ws))
    if name == 'passed_first_scrn':
        return set(first_screen_mols_ids(ws, passed=True))
    if name == 'failed_first_scrn':
        return set(first_screen_mols_ids(ws, passed=False))

def filter_workspaces(allowed_wsid=None):
    from browse.models import Workspace, Vote
    all_ws = Workspace.objects.all()
    if allowed_wsid:
        all_ws = all_ws.filter(pk__in=allowed_wsid)
    workspaces = [ws for ws in all_ws if Vote.objects.filter(election__ws=ws).exists()]

    # By default sort by review date.
    return sorted(workspaces, key=ws_review_date_key)

def prescreened_mols(ws):
    return ws.wsannotation_set.filter(indication__gt=0)

def reviewed_mols_ids(ws):
    """All molecules that have ever had an indication in 'discovery_order'.

    Effectively this is all molecules that either made it through prescreen,
    or were manually assigned to later statuses.

    Note that manually assigned molecules won't have a 'marked' status, so
    that should not be used to filter.
    """
    from browse.models import WsAnnotation
    disco_inds = set(WsAnnotation.discovery_order) - {WsAnnotation.indication_vals.UNCLASSIFIED}
    return inds_to_ids(disco_inds, ws)

def reviewed_mols(ws):
    return ids_to_wsas(reviewed_mols_ids(ws))

def selected_mols_ids(ws):
    from browse.models import WsAnnotation
    inds = WsAnnotation.selected_inds
    return inds_to_ids(inds,ws)

def final_review_mols_ids(ws):
    from browse.models import WsAnnotation
    inds = {WsAnnotation.indication_vals.REVIEWED_PREDICTION}
    return inds_to_ids(inds,ws)

def selected_mols(ws):
    return ids_to_wsas(selected_mols_ids(ws))

def screened_mols_ids(ws):
    from browse.models import WsAnnotation
    inds = WsAnnotation.screened_inds
    return inds_to_ids(inds,ws)

def screened_mols(ws):
    return ids_to_wsas(screened_mols_ids(ws))

def first_screen_mols_ids(ws, passed=True):
    from browse.models import WsAnnotation
    all=set(screened_mols_ids(ws))
    to_ret=set()
    for wsa_id in all:
         wsa=WsAnnotation.objects.get(pk=wsa_id)
         cnt = wsa.count_screening_indications()
         if cnt > 1 and passed:
             to_ret.add(wsa_id)
         elif cnt == 1 and not passed:
             to_ret.add(wsa_id)
    return to_ret

def failed_screen_mols_ids(ws):
    from browse.models import WsAnnotation
    # ever screened
    all=set(screened_mols_ids(ws))
    # current screened
    ind_vals = WsAnnotation.indication_group_members('screened')
    current = set(WsAnnotation.objects.filter(
                            ws=ws,
                            indication__in=ind_vals,
                            ).values_list('id',flat=True)
                    )
    return all-current

def inds_to_ids(inds,ws):
    from browse.models import DispositionAudit, WsAnnotation
    das = DispositionAudit.objects.filter(indication__in=inds, wsa__ws=ws)
    return das.values_list('wsa', flat=True)

def ids_to_wsas(ids):
    from browse.models import WsAnnotation
    return WsAnnotation.objects.filter(pk__in=ids)

def unreplaced_selected_mols(ws):
    """Like selected mols, but 'replacement' molecules are substituted with
    the original molecule that was used to find it.
    Use this when you're interested in the molecules we found via the platform
    rather than the ones we ended up testing.
    """
    selected = selected_mols(ws)
    return unreplaced_mols(selected)

def unreplaced_mols(starting_mols):
    from browse.models import WsAnnotation
    if starting_mols and not isinstance(next(iter(starting_mols)), WsAnnotation):
        starting_mols = WsAnnotation.all_objects.filter(pk__in=starting_mols)
    resolved_wsas = set()
    def resolve_wsa(wsa):
        # Short-circuit for efficiency but also to break any accidental cycles.
        if wsa.id in resolved_wsas:
            return []
        resolved_wsas.add(wsa.id)

        replacements = list(wsa.replacement_for.all())
        if not replacements:
            # Not replaced, return original.
            return [wsa.id]

        # Replaced, return all the replacement wsas.
        out = []
        for rpl in replacements:
            out += resolve_wsa(rpl)
        return out

    ids = set()
    for wsa in starting_mols:
        orig_ids = resolve_wsa(wsa)
        ids.update(orig_ids)
    return WsAnnotation.objects.filter(pk__in=ids)

def mol_progress(wsas_fn, ws):
    from browse.models import WsAnnotation
    reviewed_mols = list(wsas_fn(ws).prefetch_related('dispositionaudit_set'))
    from collections import defaultdict
    progress_cnt = defaultdict(int)
    for mol in reviewed_mols:
        ind = mol.max_discovery_indication()
        if ind == WsAnnotation.indication_vals.UNCLASSIFIED:
            # Ignore these, they are probably just old wsa's without
            # any dispositionaudit entries.
            continue
        progress_cnt[ind] += 1

    return progress_cnt

def demerit_labels_and_cats():
    from browse.models import Demerit
    demerits = Demerit.objects.filter(active=True)
    # We append extra non-demerit categories for reviewed molecules that are
    # still active.
    demerit_cats = [x.id for x in demerits] + [-1, -2, -3, -4]
    demerit_labels = [x.desc for x in demerits] + ['Hits', 'CIT', 'In Review', 'Inactive, no demerits']
    return demerit_labels, demerit_cats

def mol_demerits(wsas_fn, ws):
    demerit_lists_and_inds = wsas_fn(ws).values_list('demerit_list', 'indication', 'id')

    from collections import defaultdict
    from browse.models import WsAnnotation
    ivals = WsAnnotation.indication_vals
    demerit_cnt = defaultdict(int)
    kt_inds = WsAnnotation.ordered_kt_indications() + [ivals.CANDIDATE_CAUSE]
    for demerit_list, ind, id in demerit_lists_and_inds:
        demerits = WsAnnotation.parse_demerits(demerit_list)
        # Multi-counting multiple demerits, could alternatively just pick one.
        for demerit in demerits:
            demerit_cnt[demerit] += 1
        if not demerits:
            # We still need to categorize these; figure out what they are.
            # These -keys correspond to the demerit_labels_and_cats function
            # above.
            if ind in WsAnnotation.selected_inds:
                demerit_cnt[-1] += 1
            elif ind in kt_inds:
                demerit_cnt[-2] += 1
            elif ind in (ivals.REVIEWED_PREDICTION, ivals.INITIAL_PREDICTION):
                demerit_cnt[-3] += 1
            else:
                expected_inds = {ivals.INACTIVE_PREDICTION, ivals.UNCLASSIFIED, ivals.REVIEWED_AS_MOLECULE}
                assert ind in expected_inds, f'Unexpected ind {ind} on {id}'
                demerit_cnt[-4] += 1

    return demerit_cnt

def mol_score_imps(wsas_fn, ws, score_imp_group_fn):
    from browse.models import Prescreen
    from runner.process_info import JobInfo
    wsas = wsas_fn(ws)
    prescreen_ids = [wsa.marked_prescreen_id for wsa in wsas]
    prescreens = Prescreen.objects.filter(pk__in=set(prescreen_ids))
    prescreens_map = {p.id:p for p in prescreens}

    from collections import defaultdict
    psid_to_wsaids = defaultdict(list)
    for pscr_id, wsa in zip(prescreen_ids, wsas):
        psid_to_wsaids[pscr_id].append(wsa.id)

    from dtk.score_importance import ScoreImportance
    raw_scoretype_scores = defaultdict(float)
    for pscr_id, wsaids in psid_to_wsaids.items():
        if pscr_id == None:
            print("Ignoring non-prescreened wsas")
            continue
        ps = prescreens_map[pscr_id]
        if ps.eff_code() != 'wzs':
            print("Ignoring non-wzs prescreen")
            continue

        si = ScoreImportance(ws.id, ps.eff_jid())
        print(f"Loading score importance for {ws.name}, {ps.eff_jid()}, {ps.eff_code()}")
        try:
            weights, scores, weighted_scores = si.get_score_importances(wsaids)
        except (ValueError, FileNotFoundError, KeyError) as e:
            # ValueError seems to be some old feature matrices we can't load
            # because they have duplicate feature names
            #
            # FileNotFoundError is some old WZS jobs that seem to not have
            # a weights.tsv file on LTS.
            #
            # KeyError seems to be old wzs jobs that didn't have any norm_choice parameter.
            print("Failed to load score importances")
            continue
        for scoretypes_to_scores in weighted_scores.values():
            for scoretype, score in scoretypes_to_scores.items():
                scoretype_parts = scoretype.split('_')
                label = score_imp_group_fn(scoretype_parts)

                # SLE had some negative importances, which I can't really plot
                # and also break normalization here... just ignore them.
                raw_scoretype_scores[label] += max(0, score)


    # Normalize so it adds to 1
    denom = sum(raw_scoretype_scores.values())
    if denom > 0:
        for key in raw_scoretype_scores.keys():
            raw_scoretype_scores[key] /= denom
    return raw_scoretype_scores


def make_category_funcs(workspaces, wsas_fn, categorizer, categories=None):
    ws_data = {ws.id: categorizer(wsas_fn, ws) for ws in workspaces}
    if categories is None:
        categories = set()
        for data in ws_data.values():
            categories.update(data.keys())

    return [lambda ws, cat=cat: ws_data[ws.id].get(cat, 0) for cat in categories], categories



def make_rank_func(wsas_fn):
    from browse.models import Prescreen
    from runner.process_info import JobInfo
    def rank_func(ws):
        wsas = list(wsas_fn(ws))
        prescreens = [wsa.marked_prescreen for wsa in wsas]

        from collections import defaultdict
        ps_to_wsas = defaultdict(list)
        for pscr, wsa in zip(prescreens, wsas):
            ps_to_wsas[pscr].append(wsa)

        from dtk.scores import Ranker
        ranks = []
        for ps, wsas in ps_to_wsas.items():
            if ps == None:
                if False:
                    print("Ignoring non-prescreened wsas")
                else:
                    print("Inserting dummy ranks for non-prescreened wsas")
                    ranks += [(100000, wsa.get_name(False)) for wsa in wsas]
                continue
            bji = JobInfo.get_bound(ws, ps.primary_job_id())
            cat = bji.get_data_catalog()
            ranker = Ranker(cat.get_ordering(ps.primary_code(),True))

            ranks.extend([(ranker.get(wsa.id), wsa.get_name(False)) for wsa in wsas])
        return ranks
    return rank_func


def ws_review_date_key(ws):
    from django.db.models import Min, Max
    from browse.models import Election
    out = Election.objects.filter(ws=ws).aggregate(Max('due'))['due__max']
    return out


def cross_ws_box_plot(workspaces, order_key, y_func, title, x_title, y_title):
    workspaces = sorted(workspaces, key=order_key)
    data = []
    for ws in workspaces:
        y_data = list(y_func(ws))
        if y_data and isinstance(y_data[0], tuple):
            ys = [y[0] for y in y_data]
            texts = [f'{ws.name} {y[1]}' for y in y_data]
        else:
            ys = [y for y in y_data]
            texts = ws.name

        data.append({
            'y': ys,
            'name': ws.get_short_name(),
            'text': texts,
            'type': 'box',
            'boxpoints': 'all',
            'jitter': 0.5,
            'pointpos': -2,
                })

    layout = {
            'title': title,
            'xaxis': { 'title': x_title, 'tickangle': 45, 'automargin': True},
            'yaxis': { 'title': y_title, 'automargin': True, 'type': 'log'},
            'barmode': 'stack',
            'width': 1200,
            'height': 800,
            'showlegend': False,
        }
    from dtk.plot import PlotlyPlot
    return PlotlyPlot(data, layout)
def cross_ws_plot(workspaces, order_key, y_keys, y_names, title, x_title, y_title, data_order_key=None, plot_type='bar'):
    workspaces = sorted(workspaces, key=order_key)
    data = [{
        'x': [ws.get_short_name() for ws in workspaces],
        'y': [y_key(ws) for ws in workspaces],
        'name': y_name,
        'text': y_name,
        'type': plot_type,
            } for y_key, y_name in zip(y_keys, y_names)]
    if data_order_key is not None:
        data.sort(key=data_order_key)

    layout = {
            'title': title,
            'xaxis': { 'title': x_title, 'tickangle': 45, 'automargin': True},
            'yaxis': { 'title': y_title, 'automargin': True},
            'barmode': 'stack',
            'width': 1200,
            'hovermode': 'closest',
        }
    from dtk.plot import PlotlyPlot
    return PlotlyPlot(data, layout)

