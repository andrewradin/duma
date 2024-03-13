

import pytest

def coll_name(name, attrs):
    '''Create a collection with specified attrs and name, returning the name.
    '''
    from drugs.models import Prop
    for ext_key in (
            'med_chem_express',
            'selleckchem',
            'cayman',
            'ncats',
            'ttd',
            'chembl',
            'bindingdb',
            'stitch',
            'dpimerge',
            'drugbank',
            'moa',
            ):
        Prop.get_or_create(ext_key+'_id',Prop.prop_types.TAG,False)
        Prop.get_or_create('m_'+ext_key+'_id',Prop.prop_types.TAG,True)
    Prop.get_or_create('pubchem_cid',Prop.prop_types.TAG,False)
    Prop.get_or_create('synonym',Prop.prop_types.BLOB,True)
    Prop.get_or_create('std_smiles',Prop.prop_types.BLOB,False)
    Prop.get_or_create('cas',Prop.prop_types.TAG,False)
    Prop.get_or_create('native_id',Prop.prop_types.TAG,False)
    from drugs.tests import create_dummy_collection
    coll_name=name
    create_dummy_collection(coll_name,data=attrs)
    return coll_name

@pytest.fixture
def ws_with_attrs(request, db):
    '''Return a workspace with a collection imported.

    Parametrize with a list of attributes for the drug collection.

    NOTE: If you need a property to be multival, make sure it is marked
    as such in coll_name above - the default for the test is non-multi.
    '''
    attrs = request.param
    from browse.models import Workspace,WsAnnotation
    ws,new = Workspace.objects.get_or_create(name='Test Workspace')
    from drugs.models import Drug
    collection = coll_name('test_col.default', attrs)
    for drug in Drug.objects.filter(collection__name=collection):
        WsAnnotation.objects.get_or_create(ws=ws,agent=drug)
    return ws

@pytest.fixture
def make_ws(db):
    '''Provides a function to create a workspace with a collection imported.'''
    def func(attrs, name='Test Workspace'):
        from browse.models import Workspace,WsAnnotation
        ws = Workspace.objects.create(name=name)
        from drugs.models import Drug, Collection
        if attrs is None:
            collection = Collection.objects.all()[0]
        else:
            collection = coll_name('test_col.default', attrs)
        for drug in Drug.objects.filter(collection__name=collection):
            WsAnnotation.objects.get_or_create(ws=ws,agent=drug)
        return ws
    yield func


@pytest.fixture
def make_score_job(db):
    '''Provides a function to create a job with scores and populate it in the DB

    db is a fixture and will be autopopulated.
    '''
    def func(ws, job_name, job_role, fn_attr, header, rows):
        from runner.models import Process
        from runner.process_info import JobInfo
        proc = Process.objects.create(
            name=job_name,
            role=job_role,
            status=Process.status_vals.SUCCEEDED,
        )

        bji = JobInfo.get_bound(ws, proc)
        full_fn = getattr(bji, fn_attr)

        import os
        os.makedirs(os.path.dirname(full_fn))
        with open(full_fn, 'w') as f:
            f.write("\t".join(header) + '\n')
            for row in rows:
                f.write("\t".join([str(x) for x in row] ) + '\n')
        
        return proc.id
    return func