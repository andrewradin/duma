def get_hg_v(ucsc_hg_v):
    from dtk.etl import get_versions_namespace
    return get_versions_namespace('ucsc_hg')['versions'][ucsc_hg_v]['HG_VER']

months_between_updates=6

versions={
    9:dict(
            description='Sep2022 ETL update',
# Ensure we're using the same numbers here and that it's the latest UCSC version published
            UCSC_HG_VER=2,
            HG_V=get_hg_v(2),
            ENSEMBL_VER='107',
            SALMON_PKG_V='1.9.0',
            MM_V='GRCm39',
            RN_V='mRatBN7.2',
            CF_V='ROS_Cfam_1.0',
            DR_V='GRCz11',
            ),
    8:dict(
            description='1Mar2022 ETL update',
# Ensure we're using the same numbers here and that it's the latest UCSC version published
            UCSC_HG_VER=2,
            HG_V=get_hg_v(2),
            ENSEMBL_VER='105',
            SALMON_PKG_V='1.7.0',
            MM_V='GRCm39',
            RN_V='mRatBN7.2',
            CF_V='ROS_Cfam_1.0',
            DR_V='GRCz11',
            ),
    7:dict(
            description='1JunETL update',
# Ensure we're using the same numbers here and that it's the latest UCSC version published
            UCSC_HG_VER=2,
            HG_V=get_hg_v(2),
            ENSEMBL_VER='104',
            SALMON_PKG_V='1.4.0',
            MM_V='GRCm39',
            RN_V='Rnor_6.0',
            CF_V='CanFam3.1',
            DR_V='GRCz11',
            ),
    6:dict(
            description='Adding model organism support',
# Ensure we're using the same numbers here and that it's the latest UCSC version published
            UCSC_HG_VER=2,
            HG_V=get_hg_v(2),
            ENSEMBL_VER='103',
            SALMON_PKG_V='1.4.0',
            MM_V='GRCm39',
            RN_V='Rnor_6.0',
            CF_V='CanFam3.1',
            DR_V='GRCz11',
            ),
    5:dict(
            description='1Mar2021 ETL Refresh',
# Ensure we're using the same numbers here and that it's the latest UCSC version published
            UCSC_HG_VER=2,
            HG_V=get_hg_v(2),
            ENSEMBL_VER='103',
            SALMON_PKG_V='1.4.0',
            ),
    4:dict(
            description='Updated the Salmon package',
# Ensure we're using the same numbers here and that it's the latest UCSC version published
            UCSC_HG_VER=2,
            HG_V=get_hg_v(2),
            ENSEMBL_VER='101',
            SALMON_PKG_V='1.3.0',
            ),
    3:dict(
            description='4Sep2020 ETL update',
# Ensure we're using the same numbers here and that it's the latest UCSC version published
            UCSC_HG_VER=2,
            HG_V=get_hg_v(2),
            ENSEMBL_VER='101',
            ),
    2:dict(
            description='Updated 2Apr2020 - new Ensembl and new Salmon (1.1.0)',
            HG_V='38',
            ENSEMBL_VER='99',
            ),
    1:dict(
            description='Pre-versioning data',
            HG_V='38',
            ENSEMBL_VER='88',
            ),
    }
