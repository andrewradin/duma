
def test_ae_sample_sdff():
    from dtk.ae_parser import parse_ae_sample_sdrf
    with open('ge/tests/data/ae_samples.srdf.txt', 'rb') as f:
        out = parse_ae_sample_sdrf(f)

    assert len(out) == 2
    expected_0 = {
        'Sex': 'male',
        'age': '39',
        'cell type': 'primary cell',
        'time': 'wk10',
        'Source Name': 'GSM1382215 1',
    }
    for k, v in expected_0.items():
        assert out[0][k] == v

    expected_1 = {
        'Sex': 'female',
        'age': '58',
        'cell type': 'primary cell',
        'time': 'wk10',
        'Source Name': 'GSM1382214 1',
    }
    for k, v in expected_1.items():
        assert out[1][k] == v

def test_ae_sample_xml():
    from dtk.ae_parser import parse_ae_sample_xml
    with open('ge/tests/data/ae_samples.xml') as f:
        data = f.read()
    out = parse_ae_sample_xml(data)

    assert len(out) == 2
    expected_0 = {
        'Sex': 'male',
        'age': '39',
        'cell type': 'primary cell',
        'time': 'wk10',
        '_source': 'GSM1382215 1',
    }
    for k, v in expected_0.items():
        assert out[0][k] == v

    expected_1 = {
        'Sex': 'female',
        'age': '58',
        'cell type': 'primary cell',
        'time': 'wk10',
        '_source': 'GSM1382214 1',
    }
    for k, v in expected_1.items():
        assert out[1][k] == v

def test_geo_sample():
    from dtk.ae_search import parse_gds_samples

    with open('ge/tests/data/geo_samples.soft', 'rb') as f:
        out = parse_gds_samples(f)
    
    assert len(out) == 2

    expected_0 = {
        'title': 'Liver_ALGS3',
        'geo_accession': 'GSM2809205',
        'ch_subject status': 'Alagille syndrome',
        'ch_tissue': 'Liver',
        'data_processing': 'Genome_build: hg19, dp2',
    }
    for k, v in expected_0.items():
        assert out[0][k] == v

    expected_1 = {
        'title': 'Liver_ALGS5',
        'geo_accession': 'GSM2809206',
        'ch_subject status': 'Alagille syndrome',
        'ch_tissue': 'Liver',
        'data_processing': 'Genome_build: hg19, dp2',
    }
    for k, v in expected_1.items():
        assert out[1][k] == v
