
import pytest

@pytest.mark.django_db(transaction=True)
def test_attr_replace(tmp_path):
    create = [
    ("drugbank_id", "attribute", "value"),
    ("DB00007","canonical","Leuprolide"),
    ]
    matching1 = [
    ("drugbank_id", "attribute", "value"),
    ("DB00007","m_dpimerge_id","merge_id_1"),
    ]
    
    matching2 = [
    ("drugbank_id", "attribute", "value"),
    ("DB00007","m_dpimerge_id","merge_id_2"),
    ]

    matching_fail = [
    ("drugbank_id", "attribute", "value"),
    ("MISSING_ID","m_dpimerge_id","merge_id_broken"),
    ]

    create_fn =  'create.drugbank.full.tsv'
    matching_fn = 'm.drugbank.full.xref.tsv'

    from drugs.models import Collection, Prop, Drug
    names="canonical m_dpimerge_id"
    pt=Prop.prop_types
    for name in names.split():
        Prop.get_or_create(name,pt.TAG,multival=False)

    Collection.load_from_records(create_fn, iter(create))
    Collection.load_from_records(matching_fn, iter(matching1))

    # Check that we have our initial value.
    assert len(Drug.objects.all()) == 1
    drug=Drug.objects.all()[0]
    assert drug.drugbank_id == 'DB00007'
    assert drug.m_dpimerge_id == 'merge_id_1'

    Collection.load_from_records(matching_fn, iter(matching2))

    # Check that we have our updated value from the new matching file.
    drug=Drug.objects.all()[0]
    assert drug.m_dpimerge_id == 'merge_id_2'

    # Load a bad matching file that throws an exception.
    with pytest.raises(Exception):
        Collection.load_from_records(matching_fn, iter(matching_fail))

    # Verify that even though the import above failed, we didn't blow away
    # existing attributes.
    drug=Drug.objects.all()[0]
    assert drug.m_dpimerge_id == 'merge_id_2'

    success = [] 
    def check():
        drug=Drug.objects.all()[0]
        assert drug.m_dpimerge_id == 'merge_id_2'
        success.append(True)


    def dynamic_matching():
        yield ("drugbank_id", "attribute", "value")
        
        # Our thread is within the transaction, we should fail the assertion
        # because we see all our attributes as missing.
        with pytest.raises(AssertionError):
            check()

        # But any other thread will be outside the transaction and see no
        # changes to the state until it has been committed.
        from threading import Thread
        t = Thread(target=check)
        t.start()
        t.join()
        assert success == [True]

        yield ("DB00007","m_dpimerge_id","merge_id_3")

    # Check that using the DB in the middle of the transaction doesn't
    # see all the attributes as deleted.
    Collection.load_from_records(matching_fn, dynamic_matching())


@pytest.mark.django_db(transaction=True)
def test_native_key():
    create = [
    ("drugbank_id", "attribute", "value"),
    ("DB00007","canonical","Leuprolide"),
    ]

    create_fn =  'create.drugbank.full.tsv'

    from drugs.models import Collection, Prop, Drug
    names="canonical"
    pt=Prop.prop_types
    for name in names.split():
        Prop.get_or_create(name,pt.TAG,multival=False)

    Collection.load_from_records(create_fn, iter(create))

    # Check that we have our initial value.
    assert len(Drug.objects.all()) == 1
    drug=Drug.objects.all()[0]
    assert drug.drugbank_id == 'DB00007'
    assert drug.native_id == drug.drugbank_id

    # Reload it, make sure everything is still happy.
    Collection.fast_create_from_records(src_label=create_fn, inp=iter(create), coll_name='drugbank.full', key_name='drugbank_id')

    assert len(Drug.objects.all()) == 1
    drug=Drug.objects.all()[0]
    assert drug.drugbank_id == 'DB00007'
    assert drug.native_id == drug.drugbank_id
