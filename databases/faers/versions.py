
versions={
        15:dict(
                description='Dec2022 ETL update',
                FAERS_VER='2022Q3',
                NAME2CAS_VER='v14',
                MEDDRA_VER='v10',
                ),
        14:dict(
                description='Sep2022 ETL update',
                FAERS_VER='2022Q2',
                NAME2CAS_VER='v13',
                MEDDRA_VER='v9',
                ),
        13:dict(
                description='Jun2022 ETL update',
                FAERS_VER='2022Q1',
                NAME2CAS_VER='v12',
                MEDDRA_VER='v9',
                ),
        12:dict(
                description='1Dec2021 ETL update',
                FAERS_VER='2021Q3',
                NAME2CAS_VER='v10',
                MEDDRA_VER='v8',
                ),
        11:dict(
                description='16Aug2021 ETL update',
                FAERS_VER='2021Q2',
                NAME2CAS_VER='v9',
                MEDDRA_VER='v8',
                ),
        10:dict(
                description='1Jun2021 ETL update',
                FAERS_VER='2021Q1',
                NAME2CAS_VER='v8',
                MEDDRA_VER='v8',
                ),
        9:dict(
                description='1Mar2021 ETL update',
                FAERS_VER='2020Q4',
                NAME2CAS_VER='v7',
                MEDDRA_VER='v7',
                ),
        8:dict(
                description='10Dec2020 ETL update',
                FAERS_VER='2020Q3',
                NAME2CAS_VER='v6',
                MEDDRA_VER='v7',
                ),
        7:dict(
                description='Case deduplication, reporter field in demo, indi_drug_dose matrix',
                FAERS_VER='2020Q2',
                NAME2CAS_VER='v5',
                MEDDRA_VER='v6',
                ),
        6:dict(
                description='4Sep2020 ETL update',
                FAERS_VER='2020Q2',
                NAME2CAS_VER='v5',
                MEDDRA_VER='v6',
                ),
        5:dict(
# go here to see the latest version available
# https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
                FAERS_VER='2020Q1',
                NAME2CAS_VER='v4',
                MEDDRA_VER='v6',
                ),
        4:dict(
                FAERS_VER='2019Q2',
                NAME2CAS_VER='v2',
                MEDDRA_VER='v4',
                ),
        3:dict(
                description='versioned+update to latest',
                FAERS_VER='2019Q2',
                NAME2CAS_VER='v1',
                MEDDRA_VER='v1',
                ),
        2:dict(
                # almost identical to v1, but reproduced from versioned
                # inputs; differences are due to the shell script parser
                # passing a few partial input lines which get stripped by
                # the python parser
                description='attempt to reproduce v1; small differences',
                FAERS_VER='2017Q4',
                NAME2CAS_VER='v1',
                MEDDRA_VER='v1',
                ),
        1:dict(
                description='legacy faers files from S3',
                ),
        }

def custom_cleanup_data(versions,last_to_delete):
    cutoff = versions[last_to_delete]['FAERS_VER']
    return [
            (r'raw\.(\d\d\d\dQ\d)\..*\.tsv$',str,cutoff,False),
            ]
