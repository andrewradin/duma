

import pytest


from dtk.tests import make_ws
from browse.utils import drug_search
from drugs.models import DpiMergeKey, Drug


def test_search_wsa(make_ws):
    attrs = [
                ("DB07","canonical","Drug1"),
                ("DB08","canonical","Drug2"),
            ]
    ws = make_ws(attrs)
    ws2 = make_ws(attrs=None)

    drugs = Drug.objects.all()

    DpiMergeKey.objects.create(
            drug=drugs[0],
            dpimerge_key='Key1',
            version=1
            )
    DpiMergeKey.objects.create(
            drug=drugs[1],
            dpimerge_key='Key2',
            version=1
            )


    r = drug_search('rug', version=None)
    assert len(r) == 0, "Doesn't match anything"

    r = drug_search('rug', version=1)
    assert len(r) == 0, "Doesn't match anything in workspace"


    r = drug_search('drug', version=1, pattern_anywhere=False)
    assert len(r) == 2
    keys = {x[0] for x in r}
    assert keys == {'Key1', 'Key2'}


    r = drug_search('rug', version=1, pattern_anywhere=True)
    assert len(r) == 2
    r = drug_search('rug1', version=1, pattern_anywhere=True)
    assert len(r) == 1
    assert Drug.objects.get(pk=r[0][3]).canonical == 'Drug1'

