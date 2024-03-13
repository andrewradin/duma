import pytest
from mock import patch

from scripts.flag_drugs_for_previous_targets import PreviousTargetsFlagger as Flagger

from dtk.tests import make_ws, mock_dpi

@pytest.mark.django_db(transaction=False)
@patch.object(Flagger, 'get_target_wsa_ids')
@patch.object(Flagger, 'create_flag')
@patch.object(Flagger, 'create_flag_set')
def test_flag_drugs(create_flag_set,create_flag,get_target_wsa_ids,
                    make_ws, mock_dpi):

    from browse.models import WsAnnotation, Demerit
    # Test that:
    # 1) No interesting shared targets, no flags
    # 2) Shared target with ind of interest, get flagged as such
    # 3) Shared target with demerit'd mol, get flagged as such
    # 4) Combo flags

    attrs = [
                ("DB07","canonical","P01 drug"),
                ("DB08","canonical","P02 drug"),
                ("DB09","canonical","P03 drug"),
                ("DB10","canonical","reviewed P02 drug"),
                ("DB11","canonical","demerited P03 drug"),
                ("DB12","canonical","FDA P04 drug"),
                ("DB13","canonical","P01 and P02 and P03 and P04 drug"),
            ]
    attrs += [(id, 'dpimerge_id', id) for id, _, _ in attrs]

    details = (
                (1,'Reviewed Prediction'),
                (2,'Exacerbating'),
                (3,'Reviewed Prediction'),
                (4,'Exacerbating'),
                (5,'FDA Approved Treatment'),
                (6,'Exacerbating, FDA Approved Treatment, Reviewed Prediction'),
                )

    excb_demerit = Demerit.objects.create(desc='Exacerbating')

    iv = WsAnnotation.indication_vals

    ws = make_ws(attrs)

    dpi = [
        ('dpimerge_id', 'uniprot_id', 'evidence', 'direction'),
        ('DB07', 'P01', '0.5', '0'),
        ('DB08', 'P02', '0.9', '0'),
        ('DB09', 'P03', '0.5', '0'),
        ('DB10', 'P02', '0.5', '0'),
        ('DB11', 'P03', '0.5', '1'),
        ('DB12', 'P04', '0.5', '0'),
        ('DB13', 'P01', '0.5', '1'),
        ('DB13', 'P02', '0.5', '1'),
        ('DB13', 'P03', '0.5', '1'),
        ('DB13', 'P04', '0.5', '1'),
        ]
    mock_dpi('fake_dpi', dpi)

    wsas = WsAnnotation.objects.filter(ws=ws)
    get_target_wsa_ids.return_value = sorted([wsa.id for wsa in wsas])

    key2wsa = {wsa.agent.get_key():wsa for wsa in wsas}
    key2wsa['DB10'].update_indication(iv.REVIEWED_PREDICTION)
    key2wsa['DB11'].update_indication(iv.INACTIVE_PREDICTION, demerits=[excb_demerit.id])
    key2wsa['DB12'].update_indication(iv.FDA_TREATMENT, href='http://twoxar.com')

     

    # all the following parameters are arbitrary, other than ws_id;
    # the rest control the base class each_target_wsa(), which is mocked
    flagger=Flagger(ws_id=ws.id,job_id=666,score='wzs',start=0,count=200, dpi='dpimerge.default', dpi_threshold=0.5)
    flagger.flag_drugs()
    print(create_flag.mock_calls)
    from mock import call, ANY
    expect=[
            call(
                wsa_id=wsas[idx].id,
                category='PreviousTargets',
                detail=detail,
                href=ANY
                )
            for idx,detail in details
            ]
    assert create_flag.call_count == len(expect)
    create_flag.assert_has_calls(expect)
