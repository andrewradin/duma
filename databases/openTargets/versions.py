# NOTE that starting with our version 12, OpenTargets totally changed the
# file structure of non-efficacy data. As a workaround, there are now two
# variables for the source version:
# - OPENTARGETS_VER should be advanced with each refresh
# - OPENTARGETS_NONEFF_VER should remain at 21.06 until the code handles
#   the new format
# Once the new format is handled (PLAT-3938), consider whether we should:
# - fold the source version down to one variable again?
# - maintain two variables going forward?
# - split efficacy and non-efficacy ETL into two directories?
versions = {
        15:dict(
                description='Dec2022 ETL update',
                UNIPROT_VER='15',
                OPENTARGETS_VER='22.11',
                OPENTARGETS_NONEFF_VER='21.06',
                TARG_SAFETY_SUFFIX='2021-06-18',
                TRACT_SUFFIX='2021-06-03',
                ),
        14:dict(
                description='Sep2022 ETL update',
                UNIPROT_VER='14',
                OPENTARGETS_VER='22.06',
                OPENTARGETS_NONEFF_VER='21.06',
                TARG_SAFETY_SUFFIX='2021-06-18',
                TRACT_SUFFIX='2021-06-03',
                ),
        13:dict(
                description='Jun2022 ETL update - dont use, problems with uniprot v13',
                UNIPROT_VER='13',
                OPENTARGETS_VER='22.04',
                OPENTARGETS_NONEFF_VER='21.06',
                TARG_SAFETY_SUFFIX='2021-06-18',
                TRACT_SUFFIX='2021-06-03',
                ),
        12:dict(
                description='1Dec2021 ETL update',
                UNIPROT_VER='12',
                OPENTARGETS_VER='21.11',
                OPENTARGETS_NONEFF_VER='21.06',
                TARG_SAFETY_SUFFIX='2021-06-18',
                TRACT_SUFFIX='2021-06-03',
                ),
        11:dict(
                description='16Aug2021 ETL update',
                UNIPROT_VER='11',
                OPENTARGETS_VER='21.06',
                TARG_SAFETY_SUFFIX='2021-06-18',
                TRACT_SUFFIX='2021-06-03',
                ),
        10:dict(
                description='1Jun2021 ETL update',
                UNIPROT_VER='10',
                OPENTARGETS_VER='21.04',
                TARG_SAFETY_SUFFIX='2021-04-16',
                TRACT_SUFFIX='2021-03-08',
                ),
        # Starting ver 9, scores are very different - mostly I believe due to using a new
        # harmonic sum approach for combining data.
        # Also, currently computed scores do not match up with the opentargets web UI.
        # See https://github.com/opentargets/platform/issues/1508.
        9:dict(
                description='Working with new OTarg formats',
                UNIPROT_VER='9',
                OPENTARGETS_VER='21.04',
                TARG_SAFETY_SUFFIX='2021-04-16',
                TRACT_SUFFIX='2021-03-08',
                ),
        8:dict(
                description='1Mar2021 ETL update',
                UNIPROT_VER='8',
                OPENTARGETS_VER='21.02',
                TARG_SAFETY_SUFFIX='2021-02-09',
                TRACT_SUFFIX='2021-01-12',
                ),
        7:dict(
                description='10Dec2020 ETL update',
                UNIPROT_VER='7',
                OPENTARGETS_VER='20.11',
                TARG_SAFETY_SUFFIX='2020-11-10',
                TRACT_SUFFIX='2020-10-23',
                ),
        6:dict(
                description='4Sep2020 ETL update',
                UNIPROT_VER='6',
                OPENTARGETS_VER='20.06',
                ),
        5:dict(
                description='Standard ETL Refresh',
                UNIPROT_VER='5',
                OPENTARGETS_VER='20.04',
                ),
        4:dict(
                description='v3 uniprot',
                UNIPROT_VER='3',
                OPENTARGETS_VER='20.02',
                ),
        3:dict(
                description='Latest update before Q1 disease predictions',
                UNIPROT_VER='2',
                OPENTARGETS_VER='20.02',
                ),
        2:dict(
                description='first refresh with versioning',
                UNIPROT_VER='2',
                OPENTARGETS_VER='19.06',
                ),
        1:dict(
                description='pre-versioning',
                UNIPROT_VER='1',
                OPENTARGETS_VER='18.04',
                # The version archived was rebuilt from the input versions
                # specified above. The .gz file had a different checksum
                # from the production version, but the unzipped content
                # of the two files matched.
                ),
        }
