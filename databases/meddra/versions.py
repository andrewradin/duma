versions={
    10:dict(
            description='Dec2022 ETL update',
            MEDDRA_VER='v25.1',
            ),
    9:dict(
            description='1Mar2022 ETL update',
            MEDDRA_VER='v25',
            ),
    8:dict(
            description='1Jun2021 ETL update',
            MEDDRA_VER='v24',
            ),
    7:dict(
            description='Standard ETL update',
            MEDDRA_VER='v23',
            ),
    6:dict(
            description='backfilling missed versions',
            MEDDRA_VER='v22',
            ),
    5:dict(
            description='backfilling missed versions',
            MEDDRA_VER='v21',
            ),
    4:dict(
            description='first version built after versioning added',
            MEDDRA_VER='v20',
            ),
    # The versions below are all legacy versions, in ws as
    # meddra.v(19|19_1|20).tsv. Only v2/19_1 actually matches what was built
    # with this code. For v1 there was no download archive available. For v3,
    # the available archive produced different, presumably better output
    # than the S3 version meddra.v20.tsv, so that's labeled below as coming
    # from v20_x, and the new output from the v20 download is labeled v4 above.
    3:dict(
            description='pre-versioning v20; different from current output',
            MEDDRA_VER='v20_x',
            ),
    2:dict(
            description='pre-versioning v19_1; can rebuild from download',
            MEDDRA_VER='v19_1',
            ),
    1:dict(
            description='pre-versioning v19; no input download available',
            MEDDRA_VER='v19',
            ),
    }
