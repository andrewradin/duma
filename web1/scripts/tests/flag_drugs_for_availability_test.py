from __future__ import print_function
# This may be a general pattern for testing flaggers:
# - mocking out create_flag and create_flag_set lets us capture results
# - mocking out each_target_wsa lets us feed in test cases
# We can either create a real workspace, or mock it out and build dummy
# objects in each_target_wsa.
# Alternatively, could we mock FlaggerBase?
# The idea would be that in almost every flagging case, it's relevant
# to have the DB system, but we can assume that retrieving WSAs from a job,
# and recording flags are out of scope for the specific logic of one flagger.

import pytest
from mock import patch
from drugs.tests import create_dummy_collection

from scripts.flag_drugs_for_availability import Flagger
from dtk.unichem import UniChem

@pytest.mark.django_db(transaction=True)
@patch.object(UniChem, 'get_converter_dict')
@patch.object(Flagger, 'each_target_wsa')
@patch.object(Flagger, 'create_flag')
@patch.object(Flagger, 'create_flag_set')
def test_flag_drugs(create_flag_set,create_flag,each_target_wsa,get_converter_dict):

    create_dummy_collection('drugbank.full',
            data=[
                ("DB00007","canonical","Leuprolide"),
                ("DB00007",'cas','53714-56-0'),
                ("DB00008","canonical","Peginterferon alfa-2a"),
                ("DB00009","canonical","Without local CAS"),
                ("DB00009","m_chembl_id","CHEMBL000001"),
                ("DB00010","canonical","medchem"),
                ("DB00010","m_med_chem_express_id","anything"),
                ("DB00011","canonical","selleck"),
                ("DB00011","m_selleckchem_id","anything"),
                ("DB00012","canonical","cayman"),
                ("DB00012","m_cayman_id","anything"),
                ("DB00013","canonical","has_good_zinc"),
                ("DB00014","canonical","has_bad_zinc"),
                ("DB00015","canonical","has_unlabeled_zinc"),
                ("DB00016","canonical","has_matched_good_zinc"),
                ("DB00016","m_chembl_id","CHEMBL000002"),
            ],
            extra_props=[
                'bindingdb_id',
                'm_bindingdb_id',
            ],
            )
    create_dummy_collection('chembl.full',
            data=[
                ("CHEMBL000001","canonical","matching DB00009"),
                ("CHEMBL000001",'cas','12345-67-0'),
                ("CHEMBL000002","canonical","matching DB00016"),
            ],
            )
    # Stubbing this out means the test runs faster, while still verifying
    # all the data access code within the flagger. It also frees us from
    # depending on magical collection keys that match certain ZINC codes.
    # We still depend on selecting actual ZINC codes below with the desired
    # label characteristics.
    def gcd_stub(ks1,ks2,ver,key_subset=None):
        assert ks2=='zinc'
        if ks1=='drugbank':
            return {
                    "DB00013":set(["ZINC000085433138"]),
                    "DB00014":set(["ZINC000085881957"]),
                    "DB00015":set(["ZINCnot_matching"]),
                    }
        elif ks1=='chembl':
            return {
                    "CHEMBL000002":set(["ZINC000085433138"]),
                    }
        else:
            return {}
    get_converter_dict.side_effect=gcd_stub
    from drugs.models import Collection
    agent_lookup = {}
    for coll in Collection.objects.all():
        for d in coll.drug_set.all():
            native_key = getattr(d,coll.key_name)
            agent_lookup[(coll.name,native_key)] = d

    from browse.models import WsAnnotation,Workspace
    ws,new = Workspace.objects.get_or_create(name='Test')
    assert ws.get_dpi_version() == None, "This depends on unversioned matching functionality"
    drugs = [
            WsAnnotation(ws=ws,id=i,agent=agent_lookup[(coll,key)])
            for i,(coll,key) in enumerate([
                    ('drugbank.full','DB00007'), # source cas, no flag
                    ('drugbank.full','DB00008'), # should be flagged
                    ('drugbank.full','DB00009'), # matched cas, no flag
                    ('drugbank.full','DB00010'), # med_chem_express, no flag
                    ('drugbank.full','DB00011'), # selleck, no flag
                    ('drugbank.full','DB00012'), # cayman, no flag
                    ('drugbank.full','DB00013'), # zinc, ok
                    ('drugbank.full','DB00014'), # zinc, not-for-sale
                    ('drugbank.full','DB00015'), # zinc, but no labels
                    ('drugbank.full','DB00016'), # matched zinc, ok
                    ],start=1)
            ]
    each_target_wsa.return_value = drugs
    # all the following parameters are arbitrary, other than ws_id;
    # the rest control the base class each_target_wsa(), which is mocked
    flagger=Flagger(ws_id=ws.id,job_id=666,score='wzs',start=0,count=200)
    flagger.flag_drugs()
    #print create_flag_set.mock_calls
    print(create_flag.mock_calls)
    from mock import call
    expect=[
            call(
                wsa_id=drugs[idx].id,
                category='Availability',
                detail=detail,
                href=drugs[idx].drug_url(),
                )
            for idx,detail in (
                    (1,'no CAS or CommDB; no ZincID'),
                    (7,'no CAS or CommDB; not-for-sale for ZINC000085881957'),
                    (8,'no CAS or CommDB; no labels for ZINCnot_matching'),
                    )
            ]
    assert create_flag.call_count == len(expect)
    create_flag.assert_has_calls(expect)
