import pytest

from dtk.tests.ws_with_attrs import make_ws

def get_enabled_disabled(ws):
    from workflows.refresh_workflow import StandardWorkflow
    std_wf = StandardWorkflow(ws=ws)
    part_lookup = dict(std_wf.get_refresh_part_choices())
    # XXX by default, this list is missing tissue-set-specific parts
    # XXX (Pathsum, GESig, MolGSig)
    print(part_lookup)
    # XXX DisGeNet and AGR are disabled by default
    enabled = set(part_lookup[x] for x in std_wf.get_refresh_part_initial())
    disabled = set(part_lookup.values()) - enabled
    print(enabled)
    print(disabled)
    return enabled,disabled

def test_initial(make_ws):
    ws = make_ws(attrs=[])
    enabled,disabled = get_enabled_disabled(ws)
    # verify no ts-dependent parts exist
    assert 'Case/Control Pathsum' not in disabled|enabled
    assert 'cc GESig' not in disabled|enabled
    assert 'cc GESig MolGSig' not in disabled|enabled
    # test DisGeNet and AGR
    assert 'DisGeNet' in disabled
    assert 'AGR' in disabled
    # Originally, the code used disease name as a proxy for data
    # status; this was put here to verify we didn't break the legacy
    # code by moving the logic to the CMs. Now that we use DataStatus,
    # setting the disease name is no longer sufficient. Verify this.
    ws.set_disease_default('DisGeNet','C0036341','qa')
    enabled,disabled = get_enabled_disabled(ws)
    assert 'DisGeNet' in disabled
    # ditto with AGR
    ws.set_disease_default('AGR','DOID:5419','qa')
    enabled,disabled = get_enabled_disabled(ws)
    assert 'AGR' in disabled
    # the enabled cases depends on data status, which depends on finding
    # background job results; this isn't worth the hassle of setting up
    # a test
    #
    # Now verify that tissue-set-dependent parts appear
    #
    # Note: get_tissue_sets() creates default tissue sets for the workspace
    ts_by_name = {x.name:x for x in ws.get_tissue_sets()}
    from browse.models import Tissue,TissueSet
    assert 'default' in ts_by_name
    assert 'miRNA' in ts_by_name
    # Tissue sets aren't considered valid without a valid tissue
    t1=Tissue(ws=ws,tissue_set=ts_by_name['default'],geoID='GSE62191')
    t1.total_proteins = 1000
    t1.over_proteins = 100
    t1.save()
    enabled,disabled = get_enabled_disabled(ws)
    assert 'Case/Control Pathsum' in disabled|enabled
    assert 'miRNA Pathsum' not in disabled|enabled
    # verify Pathsum and GESig depend on DataStatus
    from dtk.ws_data_status import GE
    ge_ds = GE(ws)
    assert ge_ds.scores()['Case/Control'] < 1
    assert 'Case/Control Pathsum' in disabled
    assert 'Case/Control GESig' in disabled
    for i in range(2):
        t=Tissue(ws=ws,tissue_set=ts_by_name['default'],geoID='GSE6219'+str(i))
        t.total_proteins = 1000
        t.over_proteins = 100
        t.save()
    ge_ds = GE(ws)
    assert ge_ds.scores()['Case/Control'] >= 1
    enabled,disabled = get_enabled_disabled(ws)
    assert 'Case/Control Pathsum' in enabled
    assert 'Case/Control GESig' in enabled
    # verify MolGSig appears, but isn't enabled (since it's experimental)
    assert 'cc GESig MolGSig' in disabled
    # create TR tissue set; verify it appears, but isn't enabled
    assert 'Treatment/Response Pathsum' not in disabled|enabled
    ts = TissueSet(ws=ws,name='Treatment/Response')
    ts.save()
    for i in range(3):
        t=Tissue(ws=ws,tissue_set=ts,geoID='GSE6219'+str(i))
        t.total_proteins = 1000
        t.over_proteins = 100
        t.save()
    enabled,disabled = get_enabled_disabled(ws)
    assert 'Treatment/Response Pathsum' in disabled
    # test GWAS 
    from dtk.ws_data_status import Gwas
    gwas_ds = Gwas(ws)
    assert gwas_ds.scores()['GWAS Datasets'] < 1
    assert 'ESGA' in disabled
    assert 'GWASig' in disabled
    assert 'gPath' in disabled
    from browse.models import GwasDataset
    for i in range(3):
        gwd=GwasDataset(ws=ws)
        gwd.phenotype = str(i)
        gwd.pubmed_id= str(i)
        gwd.save()
    gwas_ds = Gwas(ws)
    assert gwas_ds.scores()['GWAS Datasets'] >= 1
    enabled,disabled = get_enabled_disabled(ws)
    assert 'ESGA' in enabled
    assert 'GWASig' in enabled
    assert 'gPath' in enabled
    # Test OpenTargets
    assert 'OpenTargets' in disabled
    # setting a disease name will allow it to match things
    ws.set_disease_default('openTargets','key:EFO_0000692','qa')
    enabled,disabled = get_enabled_disabled(ws)
    assert 'OpenTargets' in enabled
    # Test FAERS methods
    assert 'FAERS Sig' in disabled
    assert 'faers.v1 CAPP' in disabled
    assert 'faers.v1 DEFUS' in disabled
    # positive case depends on a background job; too hard to test
