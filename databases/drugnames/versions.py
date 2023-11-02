def coll_config(coll_list):
    from dtk.s3_cache import attribute_file_path
    return dict(
            COLL_FILES=' '.join(
                    attribute_file_path(*x)
                    for x in coll_list
                    ),
            **{c.upper()+'_VER':v for c,v in coll_list}
            )
versions={
    15:dict(
            description='Dec2022 ETL update',
            MATCHING_VER='v45',
            ),
    14:dict(
            description='Sep2022 ETL update',
            MATCHING_VER='v42',
            ),
    13:dict(
            description='Jun2022 ETL update',
            MATCHING_VER='v38',
            ),
    12:dict(
            description='1Mar2022 ETL update',
            MATCHING_VER='v36',
            ),
    11:dict(
            description='1Dec2021 ETL update',
            MATCHING_VER='v33',
            ),
    10:dict(
            description='16Aug2021 ETL update',
            MATCHING_VER='v31',
            ),
    9:dict(
            description='1Jun2021 ETL update',
            MATCHING_VER='v28',
            ),
    8:dict(
            description='1Mar2021 ETL update',
            MATCHING_VER='v23',
            ),
    7:dict(
            description='10Dec2020 ETL update',
            MATCHING_VER='v19',
            ),
    6:dict(
            description='4Sep2020 ETL update',
            MATCHING_VER='v14',
            ),
    5:dict(
            description='Adding pubchem',
            **coll_config([
                        ('chembl','v6'),
                        ('drugbank','v5'),
                        ('duma','v7'),
                        ('ncats','v1'),
                        ('bindingdb','v6'),
                        ('med_chem_express','v4'),
                        ('selleckchem','v4'),
                        ('cayman','v3'),
                        ('pubchem','v1'),
                        ])
            ),
    4:dict(
            description='Standard ETL refresh',
            **coll_config([
                        ('chembl','v6'),
                        ('drugbank','v5'),
                        ('duma','v7'),
                        ('ncats','v1'),
                        ('bindingdb','v6'),
                        ('med_chem_express','v4'),
                        ('selleckchem','v4'),
                        ('cayman','v3'),
                        ])
            ),
    3:dict(
            description='extracted from versioned attribute files',
            **coll_config([
                        ('chembl','v3'),
                        ('drugbank','v4'),
                        ('duma','v6'),
                        ('ncats','v1'),
                        ('bindingdb','v5'),
                        ('med_chem_express','v3'),
                        ('selleckchem','v3'),
                        ('cayman','v2'),
                        ])
            ),
    2:dict(
            description='extracted from create files 2019-09-16',
            ),
    1:dict(
            description='reverse-engineered to reconstruct pre-versioned AACT output',
            ),
    }
