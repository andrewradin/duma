
import logging
logger = logging.getLogger(__name__)
from dtk.cache import cached
from dtk.lazy_loader import LazyLoader

def pick_canonical_wsa(wsa_ids):
    return min(wsa_ids)

# PLAT-4075 was caused by the proper version of the MOA collection not being
# loaded on a dev machine. This class provides a way to check for this
# condition, which matters to clients of make_wsa_to_moa_{wsa,agent}.
# (Although only one MOA collection version exactly matches a dpimerge file,
# having a later collection version is usually ok, since MOAs are added
# much more often than they're removed.)
class MoaLoadStatus(LazyLoader):
    # Note that normally there will always be an moa collection loaded,
    # but there might not be inside unit tests (ws_defaults_test has this
    # issue). So, at least do something reasonable in this case.
    def _last_moa_version_loader(self):
        from drugs.models import UploadAudit
        try:
            fn = UploadAudit.objects.filter(
                    filename__startswith='moa.',
                    ok=True,
                    ).order_by('-id')[0].filename
        except IndexError:
            return None # no moa collection uploads (happens in ws_defaults_test)
        from dtk.files import VersionedFileName
        vfn = VersionedFileName(file_class='moa',name=fn)
        return vfn.version
    def _last_merge_version_loader(self):
        if self.last_moa_version == None:
            return None
        from dtk.etl import get_versions_namespace
        ns=get_versions_namespace('moa')
        return int(ns['versions'][self.last_moa_version]['MATCHING_VER'])
    def ok(self,dpi_mapping):
        if dpi_mapping.version is None:
            return True
        if self.last_merge_version is None:
            return False
        return dpi_mapping.version <= self.last_merge_version

def make_wsa_to_moa_wsa(wsa_ids, ws=None, pick_canonical=False, version=None, dpi_mapping=None):
    """Returns a mapping from wsa to moa-exemplar-wsa.
    Each mapping should retain the same underlying MoA, but the
    output will correspond to scored entities that map to the moa dpi file.

    In the case of clustermates, there could be multiple WSAs for a given MoA.
    pick_canonical will tell it to pick one arbitrarily, otherwise it will return all.
    """
    assert not wsa_ids or isinstance(next(iter(wsa_ids)), (str, int)), "Expecting ids"
    if ws is None:
        from browse.models import WsAnnotation
        ws = WsAnnotation.objects.get(pk=wsa_ids[0]).ws

    from browse.default_settings import DpiDataset
    from dtk.prot_map import DpiMapping
    dpi_mapping = dpi_mapping or DpiMapping(moa_dpi_variant(DpiDataset.value(ws=ws)))
    if not version:
        version = dpi_mapping.version
    from browse.models import WsAnnotation
    wsa_to_agent = make_wsa_to_moa_agent(wsa_ids, ws=ws, dpi_mapping=dpi_mapping)
    agent_to_moawsa = WsAnnotation.from_agent_ids(ws=ws, agent_ids=wsa_to_agent.values(), version=version).fwd_map()
    wsa2wsas = []
    for wsa, agent in wsa_to_agent.items():
        moa_wsas = agent_to_moawsa.get(agent)
        if not moa_wsas:
            continue
        if pick_canonical:
            wsa2wsas.append((wsa, pick_canonical_wsa(moa_wsas)))
        else:
            for moa_wsa in moa_wsas:
                wsa2wsas.append((wsa, moa_wsa))

    if pick_canonical:
        return dict(wsa2wsas)
    else:
        from dtk.data import MultiMap
        return MultiMap(wsa2wsas)

def moas_from_dpi_file(dpi_file):
    from dtk.files import get_file_records
    from dtk.data import assemble_attribute_records
    from dtk.prot_map import DpiMapping
    rows = list(get_file_records(dpi_file, keep_header=False))
    def canonical(prots):
        return tuple(sorted(prots))

    all_molkeys = {x[0] for x in rows}

    DPI_THRESH = DpiMapping.default_evidence
    keys_moas = []
    for molkey, molrows in assemble_attribute_records(rows, one_to_one=True):
        prots = canonical((prot, dr) for dpikey, prot, ev, dr in molrows.values() if float(ev) >= DPI_THRESH)
        keys_moas.append((molkey, prots))
    
    return keys_moas

@cached()
def make_moa_to_mol_dpikeys(dpi_choice, dpi_thresh):
    from dtk.prot_map import DpiMapping
    dpi_mapping = DpiMapping(moa_dpi_variant(dpi_choice))
    if dpi_mapping.version is None:
        return {}
    moafile_key_moa = moas_from_dpi_file(dpi_mapping.get_path())
    molfile_key_moa = moas_from_dpi_file(dpi_mapping.get_baseline_dpi().get_path())

    from dtk.data import MultiMap
    moa_to_molkeys = MultiMap(molfile_key_moa).rev_map()

    out = {}
    for moakey, moa in moafile_key_moa:
        # It's possible we're missing molecules for some MoAs.
        # This can happen even with corresponding dpi files because we
        # retain MoAs from previous DPI files even if there are no longer molecules
        # that correspond to them.
        molkeys = moa_to_molkeys.get(moa, set())
        out[moakey] = molkeys
    return out


@cached()
def make_moa_to_mol_agents(dpi_choice, dpi_thresh):
    from dtk.prot_map import DpiMapping
    dpi_mapping = DpiMapping(moa_dpi_variant(dpi_choice))
    if dpi_mapping.mapping_type() == 'uniprot':
        return {}
    from dtk.data import MultiMap
    moa_to_mol_dpikeys = make_moa_to_mol_dpikeys(dpi_choice, dpi_thresh)

    from drugs.models import Drug
    all_agent_ids = list(Drug.objects.all().values_list('id', flat=True))
    key_agent_mm = MultiMap(dpi_mapping.get_key_agent_pairs(all_agent_ids))

    out = {}
    for moakey, molkeys in moa_to_mol_dpikeys.items():
        for moa_agent in key_agent_mm.fwd_map().get(moakey, []):
            mol_agents = set()
            for molkey in molkeys:
                mol_agents.update(key_agent_mm.fwd_map().get(molkey, []))
            out[moa_agent] = mol_agents
    return out
    

def make_wsa_to_moa_agent(wsa_ids, ws=None, dpi_mapping=None):
    """Returns {wsa_id:moa_agent_id,...}

    Each WSA will be mapped to an agent ID that has the same MoA
    and is in the moa DPI file.
    """
    wsa_ids = list(wsa_ids)
    from browse.models import WsAnnotation
    if ws is None:
        ws = WsAnnotation.objects.get(pk=wsa_ids[0]).ws

    from browse.default_settings import DpiDataset
    from dtk.prot_map import DpiMapping, AgentTargetCache
    from dtk.files import get_file_records
    from dtk.data import assemble_attribute_records, MultiMap

    dpi = dpi_mapping or DpiMapping(moa_dpi_variant(DpiDataset.value(ws=ws)))
    base_dpi = dpi.get_baseline_dpi()

    try:
        dpi.get_path()
    except OSError as e:
        logger.warn("No MoA file for this dpi, returning empty list: %s", e)
        return {}

    def canonical(prots):
        return tuple(sorted(prots))

    wsas = WsAnnotation.objects.filter(pk__in=wsa_ids)
    atc = AgentTargetCache.atc_for_wsas(wsas, ws=ws, dpi_mapping=base_dpi)
    def make_moa(raw_info):
        return canonical((x[1], str(x[3])) for x in raw_info)

    base_wsa2dpi = {wsa.id: make_moa(atc.raw_info_for_agent(wsa.agent_id)) for wsa in wsas}
    # Remove any empties (no-MoA drugs)
    base_wsa2dpi = {k:v for k,v in base_wsa2dpi.items() if v}

    target_moas = MultiMap((k, canonical(v)) for k,v in base_wsa2dpi.items()).rev_map()

    rows = list(get_file_records(dpi.get_path(), keep_header=False))

    all_molkeys = {x[0] for x in rows}
    # There will be multiple molkey choices per moa - be sure we pick one that is at least a Drug uploaded to the platform.
    from drugs.models import DpiMergeKey, Drug, Prop

    # NOTE: good_molkeys is only needed if we're using exemplars instead of explicit MoAs, which we currently are not.
    # This is fairly slow, so skip it for now unless we change how this works.
    #good_molkeys = set(Drug.objects.filter(tag__prop__name=Prop.NATIVE_ID, tag__value__in=all_molkeys).values_list('tag__value', flat=True))

    wsa_to_moamolkey = {}
    DPI_THRESH = 0.5
    for molkey, molrows in assemble_attribute_records(rows, one_to_one=True):
        #if molkey not in good_molkeys:
            #continue
        prots = canonical((k, v[3]) for k,v in molrows.items() if float(v[2]) >= DPI_THRESH)
        for wsa in target_moas.get(prots, []):
            if wsa not in wsa_to_moamolkey:
                wsa_to_moamolkey[wsa] = molkey

# XXX There are at least two instances of this returning no matches:
# XXX Mycophenolate mofetil & Lidocaine in v38 (WS 46 though that shouldn't matter)
    qs = Drug.objects.filter(tag__prop__name=Prop.NATIVE_ID, tag__value__in=wsa_to_moamolkey.values())
    moa_key2agent = dict(qs.values_list('tag__value', 'id'))
    return {wsa: moa_key2agent.get(key, None) for wsa, key in wsa_to_moamolkey.items()}



def update_moa_indications(ws):
    from browse.models import WsAnnotation, Workspace
    if not isinstance(ws, Workspace):
        ws = Workspace.objects.get(id=ws)
    ind_wsas = WsAnnotation.objects.filter(ws=ws, indication__gt=0)
    id2wsa = {x.id:x for x in ind_wsas}
    print(f"{len(ind_wsas)} wsas with interesting indications")
    indwsa2moawsa = make_wsa_to_moa_wsa(id2wsa.keys(), ws=ws, pick_canonical=False)

    moawsa2indwsa = indwsa2moawsa.rev_map()
    moawsa_ids = moawsa2indwsa.keys()
    print(f"Corresponds to {len(moawsa_ids)} moas, which will be marked as REVIEWED_AS_MOLECULE")

    unclassified_moa_wsas = WsAnnotation.objects.filter(id__in=moawsa_ids, indication=0)
    iv = WsAnnotation.indication_vals

    for unc_moa_wsa in unclassified_moa_wsas:
        unc_moa_wsa.update_indication(iv.REVIEWED_AS_MOLECULE, user='MolToMoA')


def moa_dpi_variant(dpi_choice):
    """Returns the -moa variant of a DPI file choice."""
    parts = dpi_choice.split('.')
    if parts[0].endswith('-moa'):
        return dpi_choice

    parts[0] += '-moa'
    out = '.'.join(parts)
    logger.info(f"Transforming {dpi_choice} to {out} for moa")
    return out

def un_moa_dpi_variant(dpi_choice):
    """Removes and returns the non-moa variant of a DPI file choice."""
    parts = dpi_choice.split('.')
    if not parts[0].endswith('-moa'):
        return dpi_choice

    temp = parts[0].split('-')
    temp.pop()
    parts[0] = "-".join(temp)
    out = '.'.join(parts)
    logger.info(f"Transforming {dpi_choice} to {out} for non-moa")
    return out

def un_moa_drugset_variant(drugset_choice):
    """Returns the non-moa variant of a Drugset choice, if the original is an MoA drugset.
       If not, returns the input drugset
    """
    if drugset_choice.startswith('moa-'):
        out = "-".join(drugset_choice.split('-')[1:])
        logger.info(f"Transforming {drugset_choice} to {out} for non-moa")
        return out
    return drugset_choice

def is_moa_score(wsa_list):
    from browse.models import WsAnnotation
    for i in range(len(wsa_list)):
        try:
            sample_wsa = WsAnnotation.objects.get(pk=wsa_list[i])
            return sample_wsa.is_moa()
        except WsAnnotation.DoesNotExist:
            print('bad wsa id',wsa_list[i],'; trying next')
    print('no valid wsa found')
    return None

