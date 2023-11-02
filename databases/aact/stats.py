#!/usr/bin/env python3

def make_datas(studies_fn, drugs_fn, diseases_fn):
    from dtk.tsv_alt import SqliteSv

    studies = set()
    studies_sqlite = SqliteSv(studies_fn)

    for study in studies_sqlite.get_records(columns=['study']):
        studies.add(*study)

    drugs = set()
    drugs_sqlite = SqliteSv(drugs_fn)

    for drug in drugs_sqlite.get_records(columns=['drug']):
        drugs.add(*drug)

    diseases = set()
    diseases_sqlite = SqliteSv(diseases_fn)

    for disease in diseases_sqlite.get_records(columns=['disease']):
        diseases.add(*disease)

    return dict(
        studies=studies,
        drugs=drugs,
        diseases=diseases,
    )

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=[
        'aact.v{version}.studies.sqlsv',
        'aact.v{version}.drugs.sqlsv',
        'aact.v{version}.diseases.sqlsv',
        ])
