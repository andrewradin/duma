


def test_filter_and_order(db, django_assert_num_queries):
    from dtk.django_util import filter_in_and_order
    from browse.models import WsAnnotation, Workspace
    ws = Workspace.objects.create(name='ws')

    wsa1 = WsAnnotation.objects.create(ws=ws)
    wsa2 = WsAnnotation.objects.create(ws=ws)
    wsa3 = WsAnnotation.objects.create(ws=ws)

    wsa_ids = [wsa2.id, wsa1.id, wsa3.id]

    with django_assert_num_queries(1) as c:
        out = filter_in_and_order(WsAnnotation.objects, 'pk', [wsa2.id, wsa1.id, wsa3.id])
        assert list(out) == [wsa2, wsa1, wsa3]

    with django_assert_num_queries(1) as c:
        out = filter_in_and_order(WsAnnotation.objects, 'pk', [wsa3.id, wsa2.id, wsa1.id])
        assert list(out) == [wsa3, wsa2, wsa1]

    for q in c.captured_queries:
        print("Check", q['sql'])
