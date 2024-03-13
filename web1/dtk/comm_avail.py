
from dataclasses import dataclass
from typing import List


@dataclass
class CommercialAvailability:
    has_zinc: bool
    zinc_reason: str
    has_cas: bool
    has_commdb: bool
    details: List[str]


def wsa_comm_availability(ws, wsas, ncats_as_comm=True, include_details=True):
    ver = ws.get_dpi_version()

    def get_label2zincs(zinc_ids):
        '''Return {label:set([zinc_id,zinc_id,...]),...}'''
        from browse.default_settings import zinc as ds_zinc
        zinc_ver = ds_zinc.value(ws)
        from dtk.zinc import zinc
        z = zinc()
        label2zincs = {
                label:z.get_zinc_id_set_for_label(label, zinc_ver, ids=zinc_ids)
                for label in z.get_labels()
                }
        return label2zincs
    def get_wsa2zincs():
        '''Return {wsa:set([zinc_id,zinc_id,...]),...}'''
        from dtk.unichem import UniChem
        uc = UniChem()
        wsa2zincs = {}
        vdefaults = ws.get_versioned_file_defaults()
        from drugs.models import Drug
        for keyspace in ('drugbank','chembl','bindingdb'):
            # first, gather all needed external ids, so we can use the
            # key_subset speedup when building the converter dict
            all_external_ids = set()
            agents = [wsa.agent for wsa in wsas]
            agent_external_ids = Drug.bulk_external_ids(keyspace, ver, agents)
            for external_ids in agent_external_ids:
                all_external_ids |= external_ids

            # now build a converter for this collection
            key2zinc = uc.get_converter_dict(
                    keyspace,
                    'zinc',
                    vdefaults['unichem'],
                    key_subset=all_external_ids,
                    )
            # now collect all zinc labels for each wsa
            for wsa, external_ids in zip(wsas, agent_external_ids):
                s = wsa2zincs.setdefault(wsa,set())
                for coll_key in external_ids:
                    s |= set(key2zinc.get(coll_key,[]))
        return wsa2zincs

    from drugs.models import Drug
    linked_agents_by_src = Drug.get_linked_agents_map(set(
            wsa.agent_id
            for wsa in wsas
            ), version=ver)
    # load zinc data
    wsa2zincs = get_wsa2zincs()
    zincs = [zid for wsazincs in wsa2zincs.values() for zid in wsazincs]
    label2zincs = get_label2zincs(zincs)
    def avail_via_zinc(wsa):
        # Not available if explict not-for-sale label, or if id exists
        # but no labels exist. Therefore it's considered available if:
        # - there's no zinc id, or
        # - there's a zinc id with labels, none of which say 'not-for-sale'
        zinc_ids = wsa2zincs[wsa]
        if not zinc_ids:
            return (False,'no ZincID')
        for zid in zinc_ids:
            labels = [
                    label
                    for label in label2zincs
                    if zid in label2zincs[label]
                    ]
            if not labels:
                return (False,'no labels for '+zid)
            if any('not-for-sale' in x for x in labels):
                return (False,'not-for-sale for '+zid)
        return (True,'ZincID(s) indicate for-sale')

    comm_ids = ['med_chem_express','selleckchem','cayman']
    if ncats_as_comm:
        comm_ids += ['ncats']

    agents = [wsa.agent for wsa in wsas]
    all_external_comm_ids = Drug.bulk_external_ids(
            sources=comm_ids,
            version=ver,
            drugs=agents,
            )

    assert len(all_external_comm_ids) == len(agents)

    for wsa, external_comm_ids in zip(wsas, all_external_comm_ids):
        details = []
        cas_vals = set(wsa.agent.cas_set)
        for x in linked_agents_by_src.get(wsa.agent_id,set()):
            cas_vals.update(x.cas_set)
        if cas_vals:
            details.append('cas: ' + ','.join(cas_vals))
        comm = any(external_comm_ids)
        if comm and include_details:
            from dtk.html import link
            details.extend((link(*x, new_tab=True) for x in wsa.agent.commercial_urls(ver)))
        zinc, zinc_reason = avail_via_zinc(wsa)
        if zinc and include_details:
            details.append(zinc_reason)
        yield CommercialAvailability(
                has_zinc=zinc,
                zinc_reason=zinc_reason,
                has_cas=any(cas_vals),
                has_commdb=comm,
                details=details,
                )
