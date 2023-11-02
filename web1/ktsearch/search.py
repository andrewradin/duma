
def make_auto_drugset(ws, name, wsas, username):
    from browse.models import DrugSet
    ds = DrugSet.objects.create(
        name=name,
        ws=ws,
        created_by=username,
        )
    ds.add_mols(wsas, username)
    return ds
    


def drugs_from_results(ktsearch, ind_group):
    from browse.models import WsAnnotation
    good_inds = WsAnnotation.indication_group_members(ind_group)
    wsas = []
    for resultgroup in ktsearch.ktresultgroup_set.all():
        resultgroup.cache_evidence()
        ind = resultgroup.proposed_indication
        if ind in good_inds:
            wsas.append(resultgroup.wsa)
    
    return wsas