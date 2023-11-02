# NOTE: The roles list in path_helper should be the union of all mappings
# ever published
mappings_1=' '.join([
        'chembl_to_zinc',
        'chembl_to_surechembl',
        'drugbank_to_zinc',
        'drugbank_to_surechembl',
        'zinc_to_bindingdb',
        ])

versions={
    13:dict(
            description='Dec2022 ETL update',
            UNICHEM_VER='2022-12-01',
            MAPPINGS=mappings_1,
            ),
    12:dict(
            description='Sep2022 ETL update',
            UNICHEM_VER='2022-08-29',
            MAPPINGS=mappings_1,
            ),
    11:dict(
            description='Jun2022 ETL update',
            UNICHEM_VER='2022-06-07',
            MAPPINGS=mappings_1,
            ),
    10:dict(
            description='1Mar2022 ETL update',
            UNICHEM_VER='2022-03-01',
            MAPPINGS=mappings_1,
            ),
    9:dict(
            description='1Dec2021 ETL update',
            UNICHEM_VER='2021-12-01',
            MAPPINGS=mappings_1,
            ),
    8:dict(
            description='16Aug2021 ETL update',
            UNICHEM_VER='2021-08-16',
            MAPPINGS=mappings_1,
            ),
    7:dict(
            description='1Jun2021 ETL update',
            UNICHEM_VER='2021-06-01',
            MAPPINGS=mappings_1,
            ),
    6:dict(
            description='1Mar2021 ETL update',
            UNICHEM_VER='2021-03-01',
            MAPPINGS=mappings_1,
            ),
    5:dict(
            description='10Dec2020 ETL update',
            UNICHEM_VER='2020-12-10',
            MAPPINGS=mappings_1,
            ),
    4:dict(
            description='4Sep2020 ETL update',
            UNICHEM_VER='2020-09-04',
            MAPPINGS=mappings_1,
            ),
    3:dict(
            description='version refresh',
            UNICHEM_VER='2020-06-01',
            MAPPINGS=mappings_1,
            ),
    2:dict(
            description='version refresh',
            UNICHEM_VER='2020-02-24',
            MAPPINGS=mappings_1,
            ),
    1:dict(
            description='duplicates last pre-versioning output',
            UNICHEM_VER='2019-06-20',
            MAPPINGS=mappings_1,
            ),
    }
