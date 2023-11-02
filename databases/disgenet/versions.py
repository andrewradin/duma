versions={
# meddra should be set per what the latest UMLS is using
# see here for the MedDRA version being used by UMLS
# Per the below we probably want to update the meddra version every December and the UMLS version December and June
# https://www.nlm.nih.gov/research/umls/sourcereleasedocs/current/MDR/sourcerepresentation.html
    16:dict(
           description='Dec2022 ETL update',
            DISGENET_VER='2022-12-01',
            UNIPROT_VER='v15',
            MEDDRA_VER='v8',
            UMLS_VER='v7',
            ),
    15:dict(
           description='Sep2022 ETL update',
            DISGENET_VER='2022-08-29',
            UNIPROT_VER='v14',
            MEDDRA_VER='v8',
            UMLS_VER='v7',
            ),
    14:dict(
           description='Jun2022 ETL update',
            DISGENET_VER='2022-06-07',
            UNIPROT_VER='v13',
            MEDDRA_VER='v8',
            UMLS_VER='v7',
            ),
    13:dict(
           description='1Mar2022 ETL update',
            DISGENET_VER='2022-03-01',
            UNIPROT_VER='v12',
            MEDDRA_VER='v8',
            UMLS_VER='v6',
            ),
    12:dict(
           description='1Dec2021 ETL update',
            DISGENET_VER='2021-012-01',
            UNIPROT_VER='v12',
            MEDDRA_VER='v8',
            UMLS_VER='v6',
            ),
    11:dict(
           description='16Aug2021 ETL update',
            DISGENET_VER='2021-08-16',
            UNIPROT_VER='v11',
            MEDDRA_VER='v7',
            UMLS_VER='v5',
            ),
    10:dict(
           description='1Jun2021 ETL update',
            DISGENET_VER='2021-06-02',
            UNIPROT_VER='v10',
            MEDDRA_VER='v7',
            UMLS_VER='v5',
            ),
    9:dict(
           description='1Mar2021 ETL update',
            DISGENET_VER='2021-03-01',
            UNIPROT_VER='v8',
            MEDDRA_VER='v7',
            UMLS_VER='v4',
            ),
    8:dict(
           description='10Dec2020 ETL update',
            DISGENET_VER='2020-12-11',
            UNIPROT_VER='v7',
            MEDDRA_VER='v7',
            UMLS_VER='v4',
            ),
    7:dict(
           description='4Sep2020 ETL update',
# it's not clear this is actually an update, but we're updating b/c of uniprot anyhow
            DISGENET_VER='2020-09-04',
            UNIPROT_VER='v6',
# Note the latest version of meddra wasn't used b/c it would be in front of UMLS
# UMLS is using MedDRA version 22.1. Our v6 is 22.0 and v7 is 23.0
# see here for the MedDRA version being used by UMLS
# https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/updated_sources.html
            MEDDRA_VER='v6',
            UMLS_VER='v3',
            ),
    6:dict(
            description='Standard ETL Refresh',
            DISGENET_VER='2020-05-01',
            UNIPROT_VER='v5',
# Note the latest version of meddra wasn't used b/c it would be in front of UMLS
# UMLS is using MedDRA version 22.1. Our v6 is 22.0 and v7 is 23.0
# see here for the MedDRA version being used by UMLS
# https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/updated_sources.html
            MEDDRA_VER='v6',
            UMLS_VER='v3',
            ),
    5:dict(
            description='v3 uniprot',
            DISGENET_VER='2019-10-08',
            UNIPROT_VER='v3',
            MEDDRA_VER='v4',
            UMLS_VER='v2',
            ),
# updated the extraction to use the latest UMLS
    4:dict(
            description='Latest UMLS extraction',
            DISGENET_VER='2019-10-08',
            UNIPROT_VER='v2',
            MEDDRA_VER='v4',
            UMLS_VER='v2',
            ),
# added a new file type (UMLS CUI to disease names)
# in order to make selecting a disGeNet name easier
    3:dict(
            description='added new file type',
            DISGENET_VER='2019-10-08',
            UNIPROT_VER='v2',
            MEDDRA_VER='v4',
            UMLS_VER='v1',
            ),
    2:dict(
            description='update to latest inputs',
            DISGENET_VER='2019-10-08',
            UNIPROT_VER='v2',
            MEDDRA_VER='v4',
            UMLS_VER='v1',
            ),
    1:dict(
            description='rebuilt match for legacy file',
            DISGENET_VER='2018-08-09',
            UNIPROT_VER='v1',
            MEDDRA_VER='v1',
            UMLS_VER='v1',
            ),
    }
