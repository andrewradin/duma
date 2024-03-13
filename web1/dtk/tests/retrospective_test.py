

import pytest
from dtk.tests import make_ws

ws_attrs = []
for i in range(1, 8):
    ws_attrs += [('DB0%d' % i, 'canonical', 'Drug%d' % i),
                 ('DB0%d' % i, 'm_dpimerge_id', 'DB0%d' % i)]
def test_unreplaced_selected_mols(make_ws):
    ws = make_ws(ws_attrs)
    
    from dtk.retrospective import unreplaced_selected_mols

    assert list(unreplaced_selected_mols(ws)) == [], "No hits, so nothing selected"

    from browse.models import WsAnnotation
    wsa1, wsa2, wsa3 = WsAnnotation.objects.all()[0:3]
    wsa1.update_indication(WsAnnotation.indication_vals.HIT)

    assert list(unreplaced_selected_mols(ws)) == [wsa1], "First hit, it should be in the list now"

    wsa2.replacements.add(wsa1)
    wsa2.save()

    assert list(unreplaced_selected_mols(ws)) == [wsa2], "WSA1 is actually a replacement for WSA2; should give back WSA2"


    wsa3.replacements.add(wsa2)
    wsa3.save()

    assert list(unreplaced_selected_mols(ws)) == [wsa3], "WSA1 replaced WSA2 which replaced WSA2; wsa3 was the original."




def test_retro_groups(make_ws):
    ws = make_ws(ws_attrs)

    from browse.models import WsAnnotation
    wsa1, wsa2, wsa3, wsa4, wsa5, wsa6 = WsAnnotation.objects.all()[0:6]
    wsa1.update_indication(WsAnnotation.indication_vals.REVIEWED_PREDICTION)
    wsa2.update_indication(WsAnnotation.indication_vals.INITIAL_PREDICTION)
    wsa3.update_indication(WsAnnotation.indication_vals.HIT)
    wsa4.update_indication(WsAnnotation.indication_vals.IN_VITRO_1)

    wsa5.update_indication(WsAnnotation.indication_vals.IN_VITRO_1)
    wsa5.update_indication(WsAnnotation.indication_vals.INACTIVE_PREDICTION, demerits=['1'])

    assert ws.get_wsa_id_set('retro_reviewed') == {wsa1.id, wsa2.id, wsa3.id, wsa4.id, wsa5.id}
    assert ws.get_wsa_id_set('retro_final_review') == {wsa1.id}
    assert ws.get_wsa_id_set('retro_selected') == {wsa3.id, wsa4.id, wsa5.id}
    assert ws.get_wsa_id_set('retro_screened') == {wsa4.id, wsa5.id}
    assert ws.get_wsa_id_set('retro_failed_screen') == {wsa5.id}
    assert ws.get_wsa_id_set('retro_passed_first_scrn') == {wsa4.id}
    assert ws.get_wsa_id_set('retro_failed_first_scrn') == {wsa5.id}