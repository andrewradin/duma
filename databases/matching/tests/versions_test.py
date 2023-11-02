
def test_dpimerge_config():
    from dtk.etl import get_versions_namespace
    ns = get_versions_namespace('matching')
    dpimerge_config = ns['dpimerge_config']
    d = dpimerge_config(
                    match_inputs=[
                            ('duma','v5'),
                            ('drugbank','v2'),
                            ('ncats','v1'),
                            ],
                    my_dpi_merge1=(
                            'duma.default',
                            'drugbank.default',
                            'ncats.default',
                            ),
                    my_dpi_merge2=(
                            'duma.default',
                            ),
                    )
    assert d['DPIMERGE_OUT_STEMS'] == ' '.join([
            'matching.my_dpi_merge1.VERSION.dpimerge.tsv',
            'matching.my_dpi_merge2.VERSION.dpimerge.tsv',
            ])
    from path_helper import PathHelper
    assert d['DPIMERGE_ARG_STEMS'] == ' '.join([
            'matching.my_dpi_merge1.VERSION.dpimerge.tsv',
            PathHelper.storage+'duma/duma.default.v5.evidence.tsv',
            PathHelper.storage+'drugbank/drugbank.default.v2.evidence.tsv',
            PathHelper.storage+'ncats/ncats.default.v1.evidence.tsv',
            '+',
            'matching.my_dpi_merge2.VERSION.dpimerge.tsv',
            PathHelper.storage+'duma/duma.default.v5.evidence.tsv',
            ])
    assert d['DPIMERGE_INPUTS'] == ' '.join([
            PathHelper.storage+'drugbank/drugbank.default.v2.evidence.tsv',
            PathHelper.storage+'duma/duma.default.v5.evidence.tsv',
            PathHelper.storage+'ncats/ncats.default.v1.evidence.tsv',
            ])
    assert d['MATCH_INPUTS'] == ' '.join([
            PathHelper.storage+'duma/duma.full.v5.attributes.tsv',
            PathHelper.storage+'drugbank/drugbank.full.v2.attributes.tsv',
            PathHelper.storage+'ncats/ncats.full.v1.attributes.tsv',
            ])
    assert d['DUMA_VER'] == 'v5'
    assert d['DRUGBANK_VER'] == 'v2'
    assert d['NCATS_VER'] == 'v1'
    d = dpimerge_config(
                    match_inputs=[
                            ('duma','v5'),
                            ('ncats','v1'),
                            ],
                    my_dpi_merge1=(
                            'duma.default',
                            'ncats.default',
                            ),
                    combo_dpi=(
                            'my_dpi_merge1+SomeBaseDrug',
                            'my_dpi_merge1+AnotherBaseDrug',
                            ),
                    )
    assert d['COMBO_OUT_STEMS'] == ' '.join([
            'matching.my_dpi_merge1+AnotherBaseDrug.VERSION.dpimerge.tsv',
            'matching.my_dpi_merge1+SomeBaseDrug.VERSION.dpimerge.tsv',
            ])
    assert d['COMBO_IN_STEMS'] == ' '.join([
            'matching.my_dpi_merge1.VERSION.dpimerge.tsv',
            ])
