versions = {
        15:dict(
                description='Dec2022 ETL update',
                UNIPROT_VER='2022_04',
                ),
        14:dict(
                description='Sep2022 ETL update',
                UNIPROT_VER='2022_03',
                ),
        13:dict(
                description='Jun2022 ETL update - dont use Ensemble ID issues',
                UNIPROT_VER='2022_02',
                ),
        12:dict(
                description='1Dec2021 ETL update',
                UNIPROT_VER='2021_04',
                ),
        11:dict(
                description='16Aug2021 ETL update',
                UNIPROT_VER='2021_03',
                ),
        10:dict(
                description='1Jun2021 ETL update',
                UNIPROT_VER='2021_02',
                ),
        9:dict(
                description='Prevent merging across unreviewed clusters, and keep all reviewed uniprots',
                UNIPROT_VER='2021_01',
                ),
        8:dict(
                description='1Mar2021 ETL update',
                UNIPROT_VER='2021_01',
                ),
        7:dict(
                description='10Dec2020 ETL update',
                UNIPROT_VER='2020_06',
                ),
        6:dict(
                description='4Sep2020 ETL update',
                UNIPROT_VER='2020_04',
                ),
        5:dict(
                description='Standard ETL refresh',
                UNIPROT_VER='2020_02',
                ),
        4:dict(
                description='Include reactome-only prots and more alt_uniprots',
                UNIPROT_VER='2020_01',
                ),
        3:dict(
                description='New canonicalization',
                UNIPROT_VER='2020_01',
                ),
        2:dict(
                description='first refresh with versioning',
                UNIPROT_VER='2019_07',
                ),
        1:dict(
                description='pre-versioning, built around May 2018',
                UNIPROT_VER='2018_XX',
                ),
                # The source for the legacy tsv files was probably extracted
                # around May of 2018, when some work was being done on the
                # RNAseq pipeline. The legacy names json file was extracted
                # on dev08, around a year later when gene names were added
                # to the platform. At the time I converted the uniprot
                # directory to versioning, there had been some subsequent
                # formatting changes to the json output (in plat2659) but
                # these hadn't been pushed to S3. The v1 output files were
                # recovered from S3, and the original input files recovered
                # from dev07 and dev08.

        }
