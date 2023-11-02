
versions = {
        15:dict(
                description='Dec2022 ETL update',
                UNIPROT_VER='14',
                STRING_VER='11.5',
                ),
        14:dict(
                description='Sep2022 ETL update',
                UNIPROT_VER='14',
                STRING_VER='11.5',
                ),
        13:dict(
                description='A stub that was not fully created and not usable',
                UNIPROT_VER='13',
                STRING_VER='11.5',
                ),
        12:dict(
                description='1Mar2022 ETL update',
                UNIPROT_VER='12',
                STRING_VER='11.5',
                ),
        11:dict(
                description='16Aug2021 ETL update',
                UNIPROT_VER='11',
                STRING_VER='11.0',
                ),
        10:dict(
                description='1Jun2021 ETL update',
                UNIPROT_VER='10',
                STRING_VER='11.0',
                ),
        9:dict(
                description='1Mar2021 ETL update',
                UNIPROT_VER='8',
                STRING_VER='11.0',
                ),
        8:dict(
                description='10Dec2020 ETL update',
                UNIPROT_VER='7',
# in looking at their site they claim there is a ver 11.0b, but in going to the download it's not there
                STRING_VER='11.0',
                ),
        7:dict(
                description='4Sep2020 ETL update',
                UNIPROT_VER='6',
                STRING_VER='11.0',
                ),
        6:dict(
                description='Standard ETL refresh',
                UNIPROT_VER='5',
                STRING_VER='11.0',
                ),
        5:dict(
                description='Using updated uniprot',
                UNIPROT_VER='3',
                STRING_VER='11.0',
                ),
        4:dict(
                description='Known directionality applied',
                UNIPROT_VER='2',
                STRING_VER='11.0',
                ),
        3:dict(
                description='Latest string and uniprot data.',
                UNIPROT_VER='2',
                STRING_VER='11.0',
                ),
        2:dict(
                description='first refresh with versioning',
                UNIPROT_VER='1',
                STRING_VER='10.5',
                ),
        1:dict(
                description='pre-versioning',
                UNIPROT_VER='1',
                STRING_VER='10.5',
                # This is the original archived version.  It is very close to
                # but doesn't quite match version 2.  Possibly differences in
                # the uniprot conversion file?
                ),
        }
