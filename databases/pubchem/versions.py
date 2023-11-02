def match_config(match_list):
    from dtk.s3_cache import attribute_file_path as coll
    return dict(
            OTHER_MATCH_INPUTS=' '.join(coll(*x) for x in match_list),
            **{c.upper()+'_VER':v for c,v in match_list}
            )

versions={
    12:dict(
            description='Dec2022 ETL refresh',
            MATCHING_VER='v45',
        ),
    11:dict(
            description='Sep2022 ETL refresh',
            MATCHING_VER='v42',
        ),
    10:dict(
            description='Jun2022 ETL refresh',
            MATCHING_VER='v38',
        ),
    9:dict(
            description='1Mar2022 ETL refresh',
            MATCHING_VER='v36',
        ),
    8:dict(
            description='1Dec2021 ETL refresh',
            MATCHING_VER='v33',
        ),
    7:dict(
            description='16Aug2021 ETL refresh',
            MATCHING_VER='v31',
        ),
    6:dict(
            description='1Jun2021 ETL refresh',
            MATCHING_VER='v28',
        ),
    5:dict(
            description='1Mar2021 ETL refresh',
            MATCHING_VER='v23',
        ),
    4:dict(
            description='add CAS, moll weight, mol formula',
            MATCHING_VER='v21',
        ),
    3:dict(
            description='10Dec2020 ETL update',
            MATCHING_VER='v19',
        ),
    2:dict(
            description='4Sep2020 ETL update',
            MATCHING_VER='v14',
        ),
    1:dict(
            description='Initial extraction, 2020-06-08',
            **match_config([
                    ('drugbank','v8'),
                    ('chembl','v6'),
                    ]),
        ),
    }
