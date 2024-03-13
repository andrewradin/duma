
from dtk.tests import make_ws

def test_trials_for_drugs(make_ws):
    ws_attrs = [
            ('DB00051', 'canonical', 'Adalimumab'),
            ]
    ws = make_ws(ws_attrs)
    from browse.models import WsAnnotation

    wsa = WsAnnotation.objects.all()[0]

    from dtk.aact import lookup_trials_by_molecules
    trialmm = lookup_trials_by_molecules([wsa])
    out = trialmm.fwd_map()

    assert wsa.id in out
    trials = out[wsa.id]
    assert len(trials) > 300, "This should have hundreds of trials"
    trial = [x for x in trials if x.study == 'NCT01000441']
    assert trial, "Couldn't find expected trial"
