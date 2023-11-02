
import logging
from dtk.cache import cached_dict_elements, cached
logger = logging.getLogger(__name__)

from rvw.models import PrescreenEntry
from browse.models import WsAnnotation
from concurrent.futures import ThreadPoolExecutor
from dtk.table import Table

class PrescreenCacher:
    """
    There are two (top-level) tasks that take a long time and should be cached:
    1) Computing which molecule to screen next (if the ordering is a slow one like selectability)
    2) Computing the details to display for a particular molecule.

    #1 is straightforward to cache, but does result in a slightly out-of-date order when using liveselectability.
    That should be fine, though we could put in some extra effort to avoid putting overlapping targets in the queue.

    #2 is a bit data-dependent too, in particular per-target WS indication could be impacted by the immediately previous review.
    Rather than caching all the data for the page, we're relying on per-method caching to occur appropriately.

    """
    running = False
    @classmethod
    def launch_precache(cls, prescreen):
        if not cls.running:
            cls.running = True
            from threading import Thread
            Thread(target=lambda: cls.precache_prescreen(prescreen), daemon=True).start()
    
    @classmethod
    def prefetched_count(cls, prescreen):
        iv = WsAnnotation.indication_vals
        return PrescreenEntry.objects.filter(prescreen=prescreen, wsa__indication=iv.UNCLASSIFIED).count()

    @classmethod
    def precache_prescreen(cls, prescreen):
        from dtk.lock import FLock
        from path_helper import PathHelper
        try:
            # Ensure only 1 proc at a time is doing this in prod, so they're not duplicating effort.
            with FLock(PathHelper.timestamps + 'precache_lock'):
                CACHE_SIZE = 4
                iv = WsAnnotation.indication_vals
                qs = PrescreenEntry.objects.filter(prescreen=prescreen, wsa__indication=iv.UNCLASSIFIED)
                num_to_queue = CACHE_SIZE - len(qs)

                logger.info(f"Starting precache with {num_to_queue} to queue")
                if num_to_queue > 0: 
                    queued_wsa_ids = set(qs.values_list('wsa_id', flat=True))
                    for wsa_id in PrescreenOrdering.mols_to_prescreen_ordering(prescreen):
                        if wsa_id in queued_wsa_ids:
                            continue
                        cls.precache_prescreen_mol(prescreen, wsa_id)
                        num_to_queue -= 1
                        logger.info(f"{num_to_queue} left to precache")
                        if num_to_queue <= 0:
                            break
                
                cls.evict_prescreened()
        except:
            logger.error("Failed to precache prescreen")
            import traceback as tb
            tb.print_exc()
        finally:
            cls.running = False
    
    @classmethod
    def precache_prescreen_mol(cls, prescreen, wsa_id):
        logger.info(f"Precaching prescreen for {wsa_id}")
        PrescreenEntry.objects.create(
            prescreen=prescreen,
            wsa_id=wsa_id
        )

        # We rely on the underlying methods to cache results as appropriate, so a subsequent call
        # to this from the view will be fast.
        PrescreenData.data_for_moa(wsa=WsAnnotation.objects.get(pk=wsa_id), prescreen=prescreen)
    
    @classmethod
    def evict_prescreened(cls):
        iv = WsAnnotation.indication_vals
        to_evict = PrescreenEntry.objects.exclude(wsa__indication=iv.UNCLASSIFIED)
        if to_evict:
            logger.info(f"Evicting {len(to_evict)} processed prescreens")
            to_evict.delete()

    
class PrescreenOrdering:
    @classmethod
    def next_mol_to_prescreen(cls, prescreen):
        # First see if we have any in the cache queue.
        iv = WsAnnotation.indication_vals
        qs = PrescreenEntry.objects.filter(prescreen=prescreen, wsa__indication=iv.UNCLASSIFIED).order_by('id')
        if qs:
            return qs[0].wsa_id
            
        # Nothing cached, pull the next one.
        return next(cls.mols_to_prescreen_ordering(prescreen))

    @classmethod
    def mols_to_prescreen_ordering(cls, prescreen):
        """
        Generates an ordering of molecules to review from this prescreen.
        Used for caching (which will pull a few) and sometimes for the view itself (if the cache is empty).
        """
        from runner.process_info import JobInfo
        cat = JobInfo.get_bound(prescreen.ws, prescreen.primary_job_id()).get_data_catalog()
        try:
            ord = cat.get_ordering(prescreen.primary_code(), True)
        except ValueError:
            logger.error(f"Couldn't load {prescreen.primary_code()} from {prescreen.primary_job_id()}")
            # This can happen on dev if there's an old prescreen on a non-LTS job (e.g. ws 29, struct)
            yield 0
            return None
        non_unk = prescreen.ws.get_wsa_id_set('classified')
        invalid = set(WsAnnotation.all_objects.filter(ws=prescreen.ws, invalid=True).values_list('id', flat=True))
        invalid |= set(WsAnnotation.all_objects.filter(ws=prescreen.ws, agent__hide=True).values_list('id', flat=True))
        # Pre-filter invalid, as it doesn't count towards rank/ord.
        for wsa_id,score in ord:
            if wsa_id in invalid:
                continue
            if wsa_id in non_unk:
                continue
            yield wsa_id
        return None
        
def get_agent_prots(agent_ids, dpi):
    from dtk.prot_map import  AgentTargetCache
    atc = AgentTargetCache(mapping=dpi, agent_ids=agent_ids, dpi_thresh=0.5)
    out = {}
    for agent_id in agent_ids:
        prots = {x[1] for x in atc.raw_info_for_agent(agent_id)}
        out[agent_id] = prots
    return out

@cached()
def get_wsas_for_moa(ws_id, dpi_choice, prot_to_dr):
    from dtk.data import assemble_attribute_records
    from dtk.prot_map import DpiMapping
    dpi_map = DpiMapping(dpi_choice)
    prots = prot_to_dr.keys()
    if not prots:
        return []

    # Pick an arbitrary prot to whittle down the initial list.
    bindings = dpi_map.get_drug_bindings_for_prot_list(prot_list=list(prots)[:1])
    prot_agentkeys = set(x[0] for x in bindings)
    # Now get the full set of bindings for that smaller list.
    bindings = dpi_map.get_dpi_info_for_keys(prot_agentkeys, min_evid=0.5)
    agentkeys = set()

    for agentkey, recs in assemble_attribute_records(bindings, one_to_one=True):
        protdrs = {prot: dr for _, prot, _, dr in recs.values()}
        if protdrs == prot_to_dr:
            agentkeys.add(agentkey)
    key_wsa = dict(dpi_map.get_key_wsa_pairs(ws_id, keyset=agentkeys))
    wsas = [key_wsa[key] for key in agentkeys if key in key_wsa]
    return wsas

from collections import defaultdict

class PrescreenData:
    _get_agent_prots_cache = defaultdict(dict)
    @classmethod
    def cached_get_agent_prots(cls, ws, agent_ids, dpi):
        """Caches agent prots on a per-ws basis.

        Typically the list of agent prots we care about grows by one agent each time,
        so it is pretty efficient to just accumulate them.
        """
        cache = cls._get_agent_prots_cache[(ws.id, dpi.choice)]
        rem_keys = set(agent_ids) - cache.keys()
        new_data = get_agent_prots(rem_keys, dpi)
        cache.update(new_data)
        return cache
    
    _interesting_wsas_cache = defaultdict(dict)
    @classmethod
    def cached_interesting_wsas(cls, ws):
        """Caches prefetched interesting wsas on a per-ws basis.

        Typically the list of wsas we care about grows by one each time,
        so it is pretty efficient to just accumulate them.
        """
        cache = cls._interesting_wsas_cache[ws.id]
        interesting_wsa_ids = list(WsAnnotation.objects.filter(ws=ws, indication__gt=0).values_list('id', flat=True))
        missing_ids = {x for x in interesting_wsa_ids if x not in cache}
        missing_wsas = WsAnnotation.prefetch_agent_attributes(
            WsAnnotation.objects.filter(pk__in=missing_ids),
            prop_names=['canonical', 'override_name'],
            )
        
        cache.update({x.id: x for x in missing_wsas})
        return [cache[x] for x in interesting_wsa_ids]

    @classmethod
    def wsinds_for_prots(cls, ws, prots):
        from dtk.prot_map import DpiMapping, AgentTargetCache
        interesting_wsas = cls.cached_interesting_wsas(ws)
        agent_ids = {x.agent_id for x in interesting_wsas}
        prots = set(prots)
        iv = WsAnnotation.indication_vals

        def run_for_dpi(dpi):
            agent_prots = cls.cached_get_agent_prots(ws, agent_ids, dpi)
            from collections import defaultdict
            inds = defaultdict(lambda: defaultdict(set))
            for wsa in list(interesting_wsas):
                wsa_prots = agent_prots[wsa.agent_id]
                tot_prots = len(wsa_prots)
                matched_wsa_prots = wsa_prots & prots
                demerits = wsa.demerits()
                for prot in matched_wsa_prots:
                    if not demerits:
                        if wsa.indication != iv.REVIEWED_AS_MOLECULE:
                            inds[prot][wsa.indication].add((wsa, tot_prots))
                    else:
                        for demerit in demerits:
                            inds[prot][f'd{demerit}'].add((wsa, tot_prots))
            
            return inds
        
        def merge_results_into(out, out2):
            for prot, protdata in out2.items():
                if prot not in out:
                    out[prot] = protdata
                else:
                    out_protdata = out[prot]
                    for ind, wsas in protdata.items():
                        out_protdata[ind].update(wsas)
            return out

        # If you have an moa-dpi file selected, we'll want to look at both molecule and moa dpi
        # overlaps together here.
        dpi1 = cls.get_mol_dpi_map(ws)
        dpi2 = cls.get_default_dpi_map(ws)
        out = run_for_dpi(dpi1)
        if dpi1 != dpi2:
            out2 = run_for_dpi(dpi2)
            out = merge_results_into(out, out2)
        return out
    
    @classmethod
    def make_prot_imp(cls, pscr, wsa_id):
        from dtk.target_importance import get_target_importances
        prot2imp, errs = get_target_importances(pscr.ws.id, pscr.eff_jid(), [wsa_id], cache_only=True)[wsa_id]
        if errs:
            logger.info(f"Had warnings/errors from trgimp {errs}")
        return prot2imp
    
    @classmethod
    @cached(argsfunc=lambda cls, pscr, wsa_id: (pscr.id, wsa_id))
    def make_clin_imp(cls, pscr, wsa_id):
        from dtk.score_importance import ScoreImportance
        si = ScoreImportance(pscr.ws.id, pscr.eff_jid())
        _, _, importances = si.get_score_importance(wsa_id)

        from runner.process_info import JobInfo
        from dtk.target_importance import get_wzs_jids
        from moldata.utils import extract_defus_connections
        wzs_bji = JobInfo.get_bound(pscr.ws, pscr.eff_jid())
        wzs_jids = get_wzs_jids(pscr.ws.id, wzs_bji.job.settings())
        defus_jids = [k for (k,v) in wzs_jids.items() if v.endswith('_defus')]
        if len(defus_jids) == 1:
            defus_bji = JobInfo.get_bound(pscr.ws, defus_jids[0])
            connections = extract_defus_connections(wsa_id, defus_bji)
        else:
            connections = {}

        structural = {k:v for k,v in importances.items() if k.endswith('rdkitscore') or k.endswith('indigoscore')}
        imp = sum(structural.values())

        return imp, connections


    @classmethod
    def data_for_prots(cls, ws, prots, prot_imp, clin_imp):
        from browse.models import Protein, ProtSet, ProteinAttribute, Workspace
        from dtk.table import Table
        from dtk.data import MultiMap

        u2g = {x.uniprot: x.get_html_url(ws.id) for x in Protein.objects.filter(uniprot__in=prots)}
        u2name = dict(Protein.objects.filter(uniprot__in=prots, proteinattribute__attr__name='Protein_Name').values_list('uniprot', 'proteinattribute__val'))

        u2protset = MultiMap(ProtSet.objects.filter(proteins__uniprot__in=prots, ws=ws).values_list('proteins__uniprot', 'name')).fwd_map()

        u2other_ws = MultiMap((prot, (name,id)) for prot, name, id in Workspace.objects.filter(
                targetannotation__uniprot__in=prots,
                targetannotation__targetreview__note__isnull=False,
                ).values_list('targetannotation__uniprot', 'name', 'id').distinct()).fwd_map()
        
        u2inds = cls.wsinds_for_prots(ws, prots)

        from browse.models import Demerit
        demerit_map = dict(Demerit.objects.all().values_list('id', 'desc'))

        def ind_label(ind):
            if isinstance(ind, int):
                return WsAnnotation.indication_vals.get("label", ind)
            elif ind[0] == 'd':
                return demerit_map.get(int(ind[1:]), f'Unknown Ind {ind}')
            assert False, f"Unexpected {ind}"
        
        def wsa_detail_list(wsas):
            from django.utils.safestring import mark_safe
            from dtk.html import glyph_icon, popover
            wsas = sorted(wsas, key=lambda x: x[1])
            append = ''
            num_wsas = len(wsas)
            if num_wsas > 10:
                wsas = wsas[:10]
                append = '<br>...'
            text = '<br>'.join(wsa.html_url() + f" ({cnt} targets)" for wsa, cnt in wsas)
            text += append
            return popover(str(num_wsas), text)

        def inds2str(inds):
            labels = [f'{ind_label(iv)} [{wsa_detail_list(wsas)}]' for iv, wsas in inds.items()]
            return ', '.join(labels)
        
        def imp2str(imp):
            return f'{imp:.2f}' if imp is not None else 'None'
        
        def otherws_str(prot):
            from dtk.html import truncate_hover, link
            ws_nameid = sorted(u2other_ws.get(prot, []))
            links = [link(name, Workspace.ws_reverse('protein', ws_id, prot)) for name, ws_id in ws_nameid]
            return truncate_hover(links, max_len=4, sep=', ')

        out = [{
            'gene': u2g.get(prot, prot),
            'name': u2name.get(prot, ''),
            'protsets': u2protset.get(prot, []),
            'max_phase': 0,
            'other_ws': otherws_str(prot),
            'ws_inds': inds2str(u2inds.get(prot, {})),
            'importance': imp2str(prot_imp.get(prot, 0)),
        } for prot in prots] 
        if clin_imp > 0.01:
            out += [{
                'gene': '',
                'name': 'Clinical',
                'protsets': '',
                'max_phase': '',
                'other_ws': '',
                'ws_inds': '',
                'importance': f'{clin_imp:.2f}',
            }]

        columns = [
            Table.Column('Gene', idx='gene'),
            Table.Column('Name', idx='name'),
            Table.Column('Importance', idx='importance'),
            Table.Column('Protein Sets', idx='protsets'),
            Table.Column('WS Inds', idx='ws_inds'),
            Table.Column('Reviewed In', idx='other_ws'),
        ]
        return Table(out, columns)

    @classmethod
    def get_mol_dpi_map(cls, ws):
        return cls.get_default_dpi_map(ws).get_baseline_dpi()

    @classmethod
    def get_default_dpi_map(cls, ws):
        from dtk.prot_map import DpiMapping
        from browse.default_settings import DpiDataset
        return DpiMapping(DpiDataset.value(ws=ws))

    @classmethod
    def protlu_to_wsas(cls, ws, protlu):
        from dtk.data import assemble_attribute_records
        dpi_map = cls.get_mol_dpi_map(ws)
        prot_to_dr = {prot:dr for prot, dr in protlu}

        return get_wsas_for_moa(ws.id, dpi_map.choice, prot_to_dr)

    @classmethod
    def mols_for_prots(cls, ws, protdirs):
        from browse.models import WsAnnotation
        wsa_ids = cls.protlu_to_wsas(ws, protdirs)
        wsas = WsAnnotation.objects.filter(id__in=wsa_ids)

        import dtk.molecule_table as MT
        columns = [
            MT.Name(),
            MT.MaxPhase(),
            MT.CommAvail(),
        ]
        columns = MT.resolve_cols(columns, wsas)
        return Table(wsas, columns)

    @classmethod
    def data_for_moa(cls, wsa, prescreen):
        # Look up prots.
        from dtk.prot_map import DpiMapping
        dpi = DpiMapping(wsa.ws.get_dpi_default())
        protdirs = [(x[1], x[3]) for x in dpi.get_dpi_info(wsa.agent)]
        prots = [x[0] for x in protdirs]

        # Find mols that share this MoA
        prot_imp = cls.make_prot_imp(prescreen, wsa.id)

        clin_imp, clin_conn = cls.make_clin_imp(prescreen, wsa.id)

        return {
            'prot_data': cls.data_for_prots(wsa.ws, prots, prot_imp, clin_imp),
            'mol_data': cls.mols_for_prots(wsa.ws, protdirs),
            'scores': cls.scores_for_moa(wsa, prescreen),
            'stats': cls.stats_for_prescreen(prescreen),
            'prescreen': prescreen,
            'clin_imp': clin_imp,
            'clin_conn': clin_conn,
        }

    @classmethod
    def scores_for_moa(cls, wsa, prescreen):
        if not prescreen:
            return []
        jids_and_codes = [(prescreen.primary_job_id(), prescreen.primary_code())] + prescreen.extra_score_jids_and_codes()
        from dtk.scores import Ranker
        from runner.process_info import JobInfo
        out = {}
        for jid, code in jids_and_codes:
            cat = JobInfo.get_bound(wsa.ws, jid).get_data_catalog()
            ord = cat.get_ordering(code, True)
            ranker = Ranker(ord)
            rank = ranker.get(wsa.id)
            try:
                score = [x[1] for x in ord if x[0] == wsa.id][0]
            except IndexError:
                # wsa not present in score; skip this one
                # leaving out the 'N/A' because it made the column widths
                # format weirdly for some reason
                #out[f'{jid}_{code} Score'] = 'N/A'
                continue
            out[f'{jid}_{code} Score'] = f'{score:.3g}'
            out[f'{jid}_{code} Rank'] = f'{rank} / {ranker.total}'
        

        from dtk.table import Table
        cols = [Table.Column(key, idx=key) for key in out.keys()]
        return Table([out], cols)
    
    @classmethod
    def stats_for_prescreen(cls, prescreen):
        screened = len(prescreen.ws.get_wsa_id_set('classified'))
        from browse.models import DispositionAudit, WsAnnotation
        iv = WsAnnotation.indication_vals
        ever_for_review = WsAnnotation.objects.filter(ws=prescreen.ws, dispositionaudit__indication=iv.INITIAL_PREDICTION).count()
        cur_for_review = WsAnnotation.objects.filter(ws=prescreen.ws, indication=iv.INITIAL_PREDICTION).count()
        return dict(
            ever_for_review=ever_for_review,
            cur_for_review=cur_for_review,
            screened=screened,
        )
