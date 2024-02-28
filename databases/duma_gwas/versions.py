def get_hg_v(ucsc_hg_v):
    from dtk.etl import get_versions_namespace
    return get_versions_namespace('ucsc_hg')['versions'][ucsc_hg_v]['HG_VER']

deprecated=True

versions = {
        10:dict(
                description='Fix misassociated prots',
                VEP_RELEASE_VERSION=104,
                VCF_VERSION='GCF_000001405.38',
                UCSC_HG_VER=2,
                VEP_UCSC_HG_VER=get_hg_v(2),
                UNIPROT_VER='v10',
                GRASP_VER='v1',
                GWASCAT_VER='v7',
                PHEWAS_VER='v2',
                UKBB_VER='v3',
                ),
        9:dict(
                description='1Jun2021 ETL Refresh',
                VEP_RELEASE_VERSION=104,
                VCF_VERSION='GCF_000001405.38',
                UCSC_HG_VER=2,
                VEP_UCSC_HG_VER=get_hg_v(2),
                UNIPROT_VER='v10',
                GRASP_VER='v1',
                GWASCAT_VER='v7',
                PHEWAS_VER='v2',
                UKBB_VER='v3',
                ),
        8:dict(
# ACD: Stil leaving vep at 101 b/c the changes seemed small/irrelevant
                description='1Mar2021 ETL Refresh',
                VEP_RELEASE_VERSION=101,
                VCF_VERSION='GCF_000001405.38',
                UCSC_HG_VER=2,
                VEP_UCSC_HG_VER=get_hg_v(2),
                UNIPROT_VER='v8',
                GRASP_VER='v1',
                GWASCAT_VER='v6',
                PHEWAS_VER='v2',
                UKBB_VER='v3',
                ),
        7:dict(
                description='Rebuild with all lowercase ukbb',
                VEP_RELEASE_VERSION=101,
                VCF_VERSION='GCF_000001405.38',
                UCSC_HG_VER=2,
                VEP_UCSC_HG_VER=get_hg_v(2),
                UNIPROT_VER='v7',
                GRASP_VER='v1',
                GWASCAT_VER='v5',
                PHEWAS_VER='v2',
                UKBB_VER='v3',
                ),
        6:dict(
                description='10Dec2020 ETL update',
# ACD: I left vep at 101 b/c the changes in 102 seemed small/irrelevant
                VEP_RELEASE_VERSION=101,
                VCF_VERSION='GCF_000001405.38',
                UCSC_HG_VER=2,
                VEP_UCSC_HG_VER=get_hg_v(2),
                UNIPROT_VER='v7',
                GRASP_VER='v1',
                GWASCAT_VER='v5',
                PHEWAS_VER='v2',
                UKBB_VER='v2',
                ),
        5:dict(
                description='4Sep2020 ETL update',
                VEP_RELEASE_VERSION=101,
                VCF_VERSION='GCF_000001405.38',
                UCSC_HG_VER=2,
                VEP_UCSC_HG_VER=get_hg_v(2),
                UNIPROT_VER='v6',
                GRASP_VER='v1',
                GWASCAT_VER='v4',
                PHEWAS_VER='v2',
                UKBB_VER='v2',
                ),
        4:dict(
                description='Updated Uniprot, GWASCat, and restricted UKBB',
                VEP_RELEASE_VERSION=95,
# check here for version updates ftp://ftp.ncbi.nih.gov/snp/latest_release/VCF
                VCF_VERSION='GCF_000001405.38',
# Ensure we're using the same numbers here and that it's the latest UCSC version published
                UCSC_HG_VER=2,
                VEP_UCSC_HG_VER=get_hg_v(2),
                UNIPROT_VER='v5',
                GRASP_VER='v1',
                GWASCAT_VER='v3',
                PHEWAS_VER='v1',
                UKBB_VER='v2',
                ),
        3:dict(
                description='Updated Uniprot, GWASCat, and restricted UKBB',
                VEP_RELEASE_VERSION=95,
# check here for version updates ftp://ftp.ncbi.nih.gov/snp/latest_release/VCF
                VCF_VERSION='GCF_000001405.38',
# Ensure we're using the same numbers here and that it's the latest UCSC version published
                UCSC_HG_VER=2,
                VEP_UCSC_HG_VER=get_hg_v(2),
                UNIPROT_VER='v3',
                GRASP_VER='v1',
                GWASCAT_VER='v2',
                PHEWAS_VER='v1',
                UKBB_VER='v2',
                ),
        2:dict(
                description='first refresh with (partial) versioning',
                VEP_RELEASE_VERSION=95,
                VCF_VERSION='GCF_000001405.38',
# Ensure we're using the same numbers here and that it's the latest UCSC version published
                UCSC_HG_VER=1,
                VEP_UCSC_HG_VER=get_hg_v(1),
                UNIPROT_VER='v1',
                GRASP_VER='v1',
                GWASCAT_VER='v1',
                PHEWAS_VER='v1',
                UKBB_VER='v1',
                ),
        1:dict(
                description='pre-versioning',
                UCSC_HG_VER=1,
                UNIPROT_VER='v1',
                ),
        }
