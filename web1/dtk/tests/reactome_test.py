
import pytest

REASON="Requires neo4j reactome server, no longer persistant, could spin up for these tests though."

@pytest.mark.skip(reason=REASON)
def test_pathways_with_prot():
    from dtk import reactome as rct
    r = rct.Reactome()
    pw = r.get_pathways_with_prot('P29275')
    assert len(pw) > 0, "There should be pathways with this protein"

@pytest.mark.skip(reason=REASON)
def test_find_pathways():
    from dtk import reactome as rct
    r = rct.Reactome()
    ID1 = 'R-HSA-1221632'
    ID2 = 'R-HSA-912505'
    pws = r.get_pathways([ID1, ID2])
    assert pws[0].id == ID1
    assert pws[1].id == ID2

    pws = r.get_pathways([ID2, ID1])
    assert pws[0].id == ID2
    assert pws[1].id == ID1


@pytest.mark.skip(reason=REASON)
def test_score_pathways():
    from dtk import reactome as rct
    r = rct.Reactome()
    pw = r.get_pathway('R-HSA-1221632')
    assert len(pw.get_all_sub_proteins()) > 0, "Should have prots in path"

    score_list = [
            ('P58876', 0.9),
            ('P29275', 0.5),
            ('Q9UJ98', 0.1),
            ]

    scored = rct.score_pathways([pw], score_list)
    assert len(scored) == 1
    assert scored[0][0] == 'R-HSA-1221632'
    assert scored[0][1] == 62

