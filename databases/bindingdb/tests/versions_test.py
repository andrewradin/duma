
def test_match_config():
    from dtk.etl import get_versions_namespace
    ns = get_versions_namespace('bindingdb')
    match_config = ns['match_config']
    d = match_config([
                    ('duma','v5'),
                    ('drugbank','v2'),
                    ('ncats','v1'),
                    ])
    from path_helper import PathHelper
    assert d['OTHER_MATCH_INPUTS'] == ' '.join([
            PathHelper.storage+'duma/duma.full.v5.attributes.tsv',
            PathHelper.storage+'drugbank/drugbank.full.v2.attributes.tsv',
            PathHelper.storage+'ncats/ncats.full.v1.attributes.tsv',
            ])
    assert d['DUMA_VER'] == 'v5'
    assert d['DRUGBANK_VER'] == 'v2'
    assert d['NCATS_VER'] == 'v1'
