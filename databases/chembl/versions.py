def match_config(match_list):
    from dtk.s3_cache import attribute_file_path as coll
    return dict(
            OTHER_MATCH_INPUTS=' '.join(coll(*x) for x in match_list),
            **{c.upper()+'_VER':v for c,v in match_list}
            )

versions={
    19:dict(
            description='Dec2022 ETL Update',
            CHEMBL_VER='31',
            UNIPROT_VER='v15',
            MATCHING_VER='v45',
            ),
    18:dict(
            description='Sep2022 ETL Update',
            CHEMBL_VER='31',
            UNIPROT_VER='v14',
            MATCHING_VER='v42',
            ),
    17:dict(
            description='Jun2022 ETL Update',
            CHEMBL_VER='30',
            UNIPROT_VER='v13',
            MATCHING_VER='v38',
            ),
    16:dict(
            description='1Mar2022 ETL Update',
            CHEMBL_VER='29',
            UNIPROT_VER='v12',
            MATCHING_VER='v36',
            ),
    15:dict(
            description='1Dec2021 ETL Update',
            CHEMBL_VER='29',
            UNIPROT_VER='v12',
            MATCHING_VER='v33',
            ),
    14:dict(
            description='16Aug2021 ETL Update',
            CHEMBL_VER='29',
            UNIPROT_VER='v11',
            MATCHING_VER='v31',
            ),
    13:dict(
            description='1Jun2021 ETL Refresh',
            CHEMBL_VER='28',
            UNIPROT_VER='v10',
            MATCHING_VER='v28',
            ),
    12:dict(
            description='1Mar2021 ETL Refresh',
            CHEMBL_VER='28',
            UNIPROT_VER='v8',
            MATCHING_VER='v23',
            ),
    11:dict(
            description='Change ADME filtering to include at least one exemplar for each MoA',
            CHEMBL_VER='27',
            UNIPROT_VER='v7',
            MATCHING_VER='v22',
            ),
    10:dict(
            description='v9 w/ molecule_type filtering and mol_formula',
            CHEMBL_VER='27',
            UNIPROT_VER='v7',
            MATCHING_VER='v21',
            ),
    9:dict(
            description='10Dec2020 ETL update',
            CHEMBL_VER='27',
            UNIPROT_VER='v7',
            MATCHING_VER='v19',
            ),
    8:dict(
            description='4Sep2020 ETL update',
            CHEMBL_VER='27',
            UNIPROT_VER='v6',
            MATCHING_VER='v14',
            ),
    7:dict(
            description='Add LogD7.4 data to extraction',
            CHEMBL_VER='25',
            UNIPROT_VER='v5',
            **match_config([
                    ('drugbank','v5'),
                    ('duma','v7'),
                    ('ncats','v1'),
                    ('med_chem_express','v4'),
                    ('selleckchem','v4'),
                    ('cayman','v3'),
                    ])
            ),
    6:dict(
            description='Standard ETL Refresh',
            CHEMBL_VER='25',
            UNIPROT_VER='v5',
            **match_config([
                    ('drugbank','v5'),
                    ('duma','v7'),
                    ('ncats','v1'),
                    ('med_chem_express','v4'),
                    ('selleckchem','v4'),
                    ('cayman','v3'),
                    ])
            ),
    5:dict(
            description='Include all drugs with comm data',
            CHEMBL_VER='25',
            UNIPROT_VER='v4',
            **match_config([
                    ('drugbank','v4'),
                    ('duma','v6'),
                    ('ncats','v1'),
                    ('med_chem_express','v3'),
                    ('selleckchem','v3'),
                    ('cayman','v2'),
                    ])
            ),
    4:dict(
            description='Include all drugs with phases',
            CHEMBL_VER='25',
            UNIPROT_VER='v4',
            ),
    3:dict(
            description='v3 uniprot',
            CHEMBL_VER='25',
            UNIPROT_VER='v3',
            ),
    2:dict(
            description='from 3/25/19',
            CHEMBL_VER='25',
            UNIPROT_VER='v2',
            ),
    1:dict(
            description='match legacy extraction',
            CHEMBL_VER='23',
            UNIPROT_VER='v1',
            ),
    }


def custom_cleanup_data(versions,last_to_delete):
    dnld_cutoff = versions[last_to_delete]['CHEMBL_VER']
    return [
            (r'chembl_([0-9]+)[_A-z]*$',str,dnld_cutoff,True),
            ]
