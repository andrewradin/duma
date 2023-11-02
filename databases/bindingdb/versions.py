def match_config(match_list):
    from dtk.s3_cache import attribute_file_path as coll
    return dict(
            OTHER_MATCH_INPUTS=' '.join(coll(*x) for x in match_list),
            **{c.upper()+'_VER':v for c,v in match_list}
            )

versions={
    16:dict(
            description='Dec2022 ETL Refresh',
            BINDINGDB_VER='2022m11',
            UNIPROT_VER='v15',
            MATCHING_VER='v45',
            ),
    15:dict(
            description='Sep2022 ETL Refresh',
            BINDINGDB_VER='2022m7',
            UNIPROT_VER='v14',
            MATCHING_VER='v42',
            ),
    14:dict(
            description='Jun2022 ETL Refresh',
            BINDINGDB_VER='2022m5',
            UNIPROT_VER='v13',
            MATCHING_VER='v38',
            ),
    13:dict(
            description='1Mar2022 ETL Refresh',
            BINDINGDB_VER='2022m2',
            UNIPROT_VER='v12',
            MATCHING_VER='v36',
            ),
    12:dict(
            description='16Aug2021 ETL Refresh',
            BINDINGDB_VER='2021m7',
            UNIPROT_VER='v11',
            MATCHING_VER='v31',
            ),
    11:dict(
            description='1Jun2021 ETL Refresh',
            BINDINGDB_VER='2021m5',
            UNIPROT_VER='v10',
            MATCHING_VER='v28',
            ),
    10:dict(
            description='1Mar2021 ETL Refresh',
            BINDINGDB_VER='2021m2',
            UNIPROT_VER='v8',
            MATCHING_VER='v23',
            ),
    9:dict(
            description='Updated normalization',
            BINDINGDB_VER='2020m11',
            UNIPROT_VER='v7',
            MATCHING_VER='v21',
            ),
    8:dict(
            description='10Dec2020 ETL update',
            BINDINGDB_VER='2020m11',
            UNIPROT_VER='v7',
            MATCHING_VER='v19',
            ),
    7:dict(
            description='4Sep2020 ETL update',
            BINDINGDB_VER='2020m8',
            UNIPROT_VER='v6',
            MATCHING_VER='v14',
            ),
    6:dict(
            description='Standard ETL refresh',
            BINDINGDB_VER='2020m5',
            UNIPROT_VER='v5',
            **match_config([
                    ('chembl','v6'),
                    ('drugbank','v5'),
                    ('duma','v7'),
                    ('ncats','v1'),
                    ('med_chem_express','v4'),
                    ('selleckchem','v4'),
                    ('cayman','v3'),
                    ])
            ),
    5:dict(
            description='V3 Uniprot + match update',
            BINDINGDB_VER='2020m1',
            UNIPROT_VER='v3',
            **match_config([
                    ('chembl','v3'),
                    ('drugbank','v3'),
                    ('duma','v6'),
                    ('ncats','v1'),
                    ('med_chem_express','v3'),
                    ('selleckchem','v3'),
                    ('cayman','v2'),
                    ])
            ),
    4:dict(
            description='strip patent-based names',
            BINDINGDB_VER='2020m1',
            UNIPROT_VER='v2',
            **match_config([
                    ('chembl','v2'),
                    ('drugbank','v2'),
                    ('duma','v2'),
                    ('ncats','v1'),
                    ('med_chem_express','v2'),
                    ('selleckchem','v2'),
                    ('cayman','v2'),
                    ])
            ),
    3:dict(
            description='handle multi-part name field',
            BINDINGDB_VER='2020m1',
            UNIPROT_VER='v2',
            **match_config([
                    ('chembl','v2'),
                    ('drugbank','v2'),
                    ('duma','v1'),
                    ('ncats','v1'),
                    ('med_chem_express','v2'),
                    ('selleckchem','v2'),
                    ('cayman','v2'),
                    ])
            ),
    2:dict(
            description='re-extract',
            BINDINGDB_VER='2020m1',
            UNIPROT_VER='v2',
            **match_config([
                    ('chembl','v2'),
                    ('drugbank','v2'),
                    ('duma','v1'),
                    ('ncats','v1'),
                    ('med_chem_express','v2'),
                    ('selleckchem','v2'),
                    ('cayman','v2'),
                    ])
            ),
    1:dict(
            description='match legacy extraction',
            BINDINGDB_VER='2016m2',
            UNIPROT_VER='v1',
            **match_config([
                    ('chembl','v1'),
                    ('drugbank','v1'),
                    ('duma','v1'),
                    ('ncats','v1'),
                    ('med_chem_express','v1'),
                    ('selleckchem','v1'),
                    ('cayman','v1'),
                    ])
            ),
    }
