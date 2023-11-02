
versions={
    16:dict(
            description='Dec2022 ETL update',
            AACT_VER='20221202',
            DRUGNAMES_VER='15',
            ),
    15:dict(
            description='Sep2022 ETL update',
            AACT_VER='20220830',
            DRUGNAMES_VER='14',
            ),
    14:dict(
            description='Jun2022 ETL update',
            AACT_VER='20220104',
            DRUGNAMES_VER='13',
            ),
    13:dict(
            description='intermediate extraction with completion dates',
            AACT_VER='20220401',
            DRUGNAMES_VER='12',
            ),
    12:dict(
            description='1Mar2022 ETL update',
            AACT_VER='20220104',
            DRUGNAMES_VER='12',
            ),
    11:dict(
            description='1Dec2021 ETL update',
            AACT_VER='20211201',
            DRUGNAMES_VER='11',
            ),
    10:dict(
            description='1Jun2021 ETL update',
            AACT_VER='20210601',
            DRUGNAMES_VER='9',
            ),
    9:dict(
            description='1Mar2021 ETL update',
            AACT_VER='20210301',
            DRUGNAMES_VER='8',
            ),
    8:dict(
            description='10Dec2020 ETL update',
            AACT_VER='20201201',
            DRUGNAMES_VER='7',
            ),
    7:dict(
            description='4Sep2020 ETL update',
            AACT_VER='20200901',
            DRUGNAMES_VER='6',
            ),
    6:dict(
            description='Updated drugnames and new format, 5% more matches',
            AACT_VER='20200601',
            DRUGNAMES_VER='5',
            ),
    5:dict(
            description='Standard ETL refresh',
            AACT_VER='20200601',
            DRUGNAMES_VER='4',
            ),
    4:dict(
            description='Latest AACT data',
            AACT_VER='20200501',
            DRUGNAMES_VER='3',
            ),
    3:dict(
            description='same AACT data with new drugnames',
            AACT_VER='20190901',
            DRUGNAMES_VER='3',
            ),
    # 2019 aact data with v2 drugnames, no code mods
    #   338398 interventions; 90006 studies; 3104 diseases; 6285 drugs
    # extracting with v1 drugnames:
    #   335633 interventions; 89202 studies; 3101 diseases; 5676 drugs
    #   (the drop in drugs is because some studies changed drugs, and
    #   since the v1 drugnames file only contains things that already
    #   match, we can only lose drugs, not gain them)
    # extracting with v2 drugnames minus TTD:
    #   331713 interventions; 88197 studies; 3089 diseases; 5627 drug
    #   (this indicates that some trialled drugs were only in TTD, not
    #   drugbank or chembl; this should be followed up)
    2:dict(
            description='refresh with current AACT data',
            AACT_VER='20190901',
            DRUGNAMES_VER='2',
            ),
    # aact v1 baseline
    #   306391 interventions; 76997 studies; 2971 diseases; 5710 drugs
    1:dict(
            description='duplicates last pre-versioning output',
            AACT_VER='20171102',
            DRUGNAMES_VER='1',
            ),
    }

def custom_cleanup_data(versions,last_to_delete):
    dnld_cutoff = versions[last_to_delete]['AACT_VER']
    return [
            (r'([0-9]+)_pipe-delimited-export\.zip$',str,dnld_cutoff,False),
            ]
