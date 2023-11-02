def input_collections(inp_list):
    # this is only used for versions prior to PLAT-3484
    from path_helper import PathHelper
    return dict(
            CREATE_FILES=' '.join([
                    PathHelper.storage+f'{c}/{c}.full.{v}.attributes.tsv'
                    for c,v in inp_list
                    ]),
            **{c.upper()+'_VER':v for c,v in inp_list}
            )

versions={
        14:dict(
                description='Dec2022 ETL update',
                MATCHING_VER='v45',
                SOURCE_COLLECTIONS=' '.join([
                        'drugbank',
                        'cayman',
                        'med_chem_express',
                        'selleckchem',
                        'pubchem',
                        'globaldata',
                        ])
                ),
        13:dict(
                description='Sep2022 ETL update',
                MATCHING_VER='v42',
                SOURCE_COLLECTIONS=' '.join([
                        'drugbank',
                        'cayman',
                        'med_chem_express',
                        'selleckchem',
                        'pubchem',
                        'globaldata',
                        ])
                ),
        12:dict(
                description='Jun2022 ETL update',
                MATCHING_VER='v38',
                SOURCE_COLLECTIONS=' '.join([
                        'drugbank',
                        'cayman',
                        'med_chem_express',
                        'selleckchem',
                        'pubchem',
                        'globaldata',
                        ])
                ),
        11:dict(
                description='1Mar2022 ETL update',
                MATCHING_VER='v36',
                SOURCE_COLLECTIONS=' '.join([
                        'drugbank',
                        'cayman',
                        'med_chem_express',
                        'selleckchem',
                        'pubchem',
                        ])
                ),
        10:dict(
                description='1Dec2021 ETL update',
                MATCHING_VER='v33',
                SOURCE_COLLECTIONS=' '.join([
                        'drugbank',
                        'cayman',
                        'med_chem_express',
                        'selleckchem',
                        'pubchem',
                        ])
                ),
        9:dict(
                description='16Aug2021 ETL update',
                MATCHING_VER='v31',
                SOURCE_COLLECTIONS=' '.join([
                        'drugbank',
                        'cayman',
                        'med_chem_express',
                        'selleckchem',
                        'pubchem',
                        ])
                ),
        8:dict(
                description='1Jun2021 ETL update',
                MATCHING_VER='v28',
                SOURCE_COLLECTIONS=' '.join([
                        'drugbank',
                        'cayman',
                        'med_chem_express',
                        'selleckchem',
                        ])
                ),
        7:dict(
                description='1Mar2021 ETL update',
                MATCHING_VER='v23',
                SOURCE_COLLECTIONS=' '.join([
                        'drugbank',
                        'cayman',
                        'med_chem_express',
                        'selleckchem',
                        ])
                ),
        6:dict(
                description='10Dec2020 ETL update',
                MATCHING_VER='v19',
                SOURCE_COLLECTIONS=' '.join([
                        'drugbank',
                        'cayman',
                        'med_chem_express',
                        'selleckchem',
                        ])
                ),
        5:dict(
                description='4Sep2020 ETL update',
                MATCHING_VER='v14',
                SOURCE_COLLECTIONS=' '.join([
                        'drugbank',
                        'cayman',
                        'med_chem_express',
                        'selleckchem',
                        ])
                ),
        4:dict(
                description='Standard ETL Refresh',
                **input_collections([
                        ('drugbank','v5'),
                        ('cayman','v3'),
                        ('med_chem_express','v4'),
                        ('selleckchem','v4'),
                        ])
                ),
        3:dict(
                description='use versioned molecule collections',
                **input_collections([
                        ('drugbank','v2'),
                        ('cayman','v2'),
                        ('med_chem_express','v3'),
                        ('selleckchem','v3'),
                        ])
                ),
        2:dict(
                description='replace TTD with commercial collections',
                INPUT_COLLECTIONS=' '.join([
                        'drugbank',
                        'adrecs',
                        'cayman',
                        'med_chem_express',
                        'selleckchem',
                        ])
                ),
        1:dict(
                description='legacy file from drugbank and ttd',
                ),
        }
