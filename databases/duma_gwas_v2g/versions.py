
def get_hg_v(ucsc_hg_v):
    from dtk.etl import get_versions_namespace
    return get_versions_namespace('ucsc_hg')['versions'][ucsc_hg_v]['HG_VER']


versions = {
        6:dict(
                description='Dec2022 ETL update',
                VEP_RELEASE_VERSION=105, # left unchanged b/c 108 was giving issues on the install
                UCSC_HG_VER=2,
                VEP_UCSC_HG_VER=get_hg_v(2),
                UNIPROT_VER='v15',
                DUMA_GWAS_V2D_VER=8,
                OTARG_GEN_VERSION="22.02.01" # didn't update to 22.09 bc they switched from json to parquet
                ),
        5:dict(
                description='Sep2022 ETL update',
                VEP_RELEASE_VERSION=105,
                UCSC_HG_VER=2,
                VEP_UCSC_HG_VER=get_hg_v(2),
                UNIPROT_VER='v14',
                DUMA_GWAS_V2D_VER=7,
                OTARG_GEN_VERSION="22.02.01"
                ),
        4:dict(
                description='1Mar2022 ETL update',
                VEP_RELEASE_VERSION=105,
                UCSC_HG_VER=2,
                VEP_UCSC_HG_VER=get_hg_v(2),
                UNIPROT_VER='v12',
                DUMA_GWAS_V2D_VER=5,
                OTARG_GEN_VERSION="22.02.01"
                ),
        3:dict(
                description='16Aug2021 ETL update',
                VEP_RELEASE_VERSION=104,
                UCSC_HG_VER=2,
                VEP_UCSC_HG_VER=get_hg_v(2),
                UNIPROT_VER='v11',
                DUMA_GWAS_V2D_VER=3,
                OTARG_GEN_VERSION="210608"
                ),
        2:dict(
                description='Incorporating opentargets v2g data',
                VEP_RELEASE_VERSION=104,
                UCSC_HG_VER=2,
                VEP_UCSC_HG_VER=get_hg_v(2),
                UNIPROT_VER='v10',
                DUMA_GWAS_V2D_VER=2,
                # Find the latest directory name here: http://ftp.ebi.ac.uk/pub/databases/opentargets/genetics/
                OTARG_GEN_VERSION="20022712"
                ),
        1:dict(
                description='First v2g with split gwas',
                VEP_RELEASE_VERSION=104,
                UCSC_HG_VER=2,
                VEP_UCSC_HG_VER=get_hg_v(2),
                UNIPROT_VER='v10',
                DUMA_GWAS_V2D_VER=1,
                ),
        }
