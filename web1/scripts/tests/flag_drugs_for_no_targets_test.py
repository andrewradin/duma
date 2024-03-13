

from scripts.flag_drugs_for_no_targets import main, NoTargetsFlagger

from mock import patch

@patch('browse.models.Workspace')
@patch('dtk.prot_map.DpiMapping')
@patch.object(NoTargetsFlagger, 'each_target_wsa')
@patch.object(NoTargetsFlagger, 'create_flag')
@patch.object(NoTargetsFlagger, 'create_flag_set')
def test_no_targets_flag(create_flag_set, create_flag, each_target_wsa, dpi_map, workspace):

    # -- Setup --
    dpis = [
           [], 
           ['a target'], 
           ['another'], 
            ]
    def get_dpi(*args, **kwargs):
        return dpis.pop(0)

    dpi_map.return_value.get_dpi_info.side_effect = get_dpi

    from browse.models import WsAnnotation
    drugs = [
            WsAnnotation(id=12),
            WsAnnotation(id=23),
            WsAnnotation(id=34),
            ]
    each_target_wsa.return_value = drugs
    
    created = []
    create_flag.side_effect = lambda **kwargs: created.append(kwargs)

    created_set = []
    create_flag_set.side_effect = lambda *args: created_set.append(args)

    # -- Test --

    main(['43', '61215', 'wzs'])

    assert created_set == [('NoTargets',)]
    assert len(created) == 1
    assert created[0]['wsa_id'] == 12

