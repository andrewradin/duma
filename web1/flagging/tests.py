import pytest

from flagging.utils import FlaggerBase

class MyFlagger(FlaggerBase):
    def __init__(self,**kwargs):
        super(MyFlagger,self).__init__(kwargs)
        self.wsa_count = 0
    def flag_drugs(self):
        for x in self.each_target_wsa():
            self.wsa_count += 1

from dtk.tests.ws_with_attrs import ws_with_attrs

ws_attrs=[
    ('DB00001','canonical','Drug 1'),
    ]

@pytest.mark.parametrize('ws_with_attrs', [ws_attrs], indirect=True)
def test_flagger_base(ws_with_attrs):
    from browse.models import WsAnnotation, Workspace
    ws = Workspace.objects.all()[0]
    # to avoid having to mock the entire data catalog mechanism, we
    # just override get_target_wsa_ids(); the selection parameters
    # are still required, but won't be used
    flagger = MyFlagger(
            ws_id=ws.id,
            job_id=666,
            score='myscore',
            start=0,
            count=100,
            )
    flagger.get_target_wsa_ids = lambda:[
            y.id
            for y in ws.wsannotation_set.all()
            ]
    from django.test.utils import CaptureQueriesContext
    from django.db import connection
    with CaptureQueriesContext(connection) as context:
        flagger.flag_drugs()
    assert flagger.wsa_count == 1
    from django.db import connection
    expected_queries = 3
    if connection.mysql_version[:2] == (5,5):
        expected_queries += 1
    print("Context? ", '\n'.join(str(x) for x in list(context)))
    # There is a "SELECT VERSION()" call that seems to appear/disappear depending on whether this
    # is run standalone or as part of a sequence of tests.  Allow either count.
    assert len(context) == expected_queries or len(context) == expected_queries + 1 # one for lambda above, 2 for prefetch
    # verify a second iteration through the list doesn't re-fetch
    with CaptureQueriesContext(connection) as context:
        for x in flagger.each_target_wsa():
            pass
    assert len(context) == 0
