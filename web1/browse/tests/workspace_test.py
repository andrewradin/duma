
from dtk.tests import mock_dpi, make_ws

def test_short_name(db):
    from browse.models import Workspace
    def make_ws(name):
        ws = Workspace(name=name)
        ws.save()
        return ws
    assert make_ws(name='Sjogren').get_short_name() == 'Sjog'
    assert make_ws(name='Sjogren Syndrome').get_short_name() == 'Sjo Syn'
    assert make_ws(name='Sjogren Syndrome ').get_short_name() == 'Sjo Syn'
    assert make_ws(name='System Lupus Erythe').get_short_name() == 'SLE'
    ws = make_ws(name='Sjogren')
    from browse.default_settings import DiseaseShortName
    DiseaseShortName.set(ws, 'sjogren', 'test')
    assert ws.get_short_name() == 'sjogren'
    assert make_ws(name=' P D  A  C  ').get_short_name() == 'PDAC'

def test_tissue_set_defaults(db):
    from browse.models import Workspace
    ws = Workspace(name='tissue set default test')
    ws.save()
    tsets = ws.get_tissue_sets()
    assert(set(x.name for x in tsets) == set(['default','miRNA']))
    for ts in tsets:
        assert ts.miRNA == (ts.name == 'miRNA')


def test_uniprot_set(mock_dpi, make_ws):
    dpi = [
        ('dpimerge_id', 'uniprot_id', 'evidence', 'direction'),
        ('DB01', 'P1', '0.5', '0'),
        ('DB02', 'P2', '0.5', '0'),
    ]
    mock_dpi('fake_dpi', dpi)

    ws_attrs = []
    for i in range(1, 6):
        ws_attrs += [('DB0%d' % i, 'canonical', 'Drug%d' % i),
                    ('DB0%d' % i, 'm_dpimerge_id', 'DB0%d' % i),
                    ]

    from browse.models import Workspace, ProtSet, Protein, DrugSet, WsAnnotation
    ws = make_ws(ws_attrs)


    # Empty protset
    ps1 = ProtSet.objects.create(ws=ws, name='testps1')
    assert ws.get_uniprot_set(f'ps{ps1.id}') == set()

    # 1 Protein
    u1 = Protein.objects.create(uniprot='P1', gene='G1')
    ps1.proteins.add(u1)
    assert ws.get_uniprot_set(f'ps{ps1.id}') == set(['P1'])

    # Drugset, empty
    ds1 = DrugSet.objects.create(ws=ws, name='testds1')
    assert ws.get_uniprot_set(f'ds_0.5_ds{ds1.id}') == set()

    # Drugset, 1 drug
    wsa1, wsa2 = WsAnnotation.objects.all()[:2]
    assert wsa1.agent.canonical == 'Drug1'
    assert wsa2.agent.canonical == 'Drug2'
    ds1.drugs.add(wsa1)
    assert ws.get_uniprot_set(f'ds_0.5_ds{ds1.id}') == set(['P1'])

    # Drugset, 2 drug
    ds1.drugs.add(wsa2)
    assert ws.get_uniprot_set(f'ds_0.5_ds{ds1.id}') == set(['P1', 'P2'])
    