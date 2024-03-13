import pytest

def test_filter_parsing():
    from dtk.faers import DemoFilter
    df = DemoFilter()
    assert df.as_string() == ''
    assert df.sex_filter is None
    assert df.min_age is None
    assert df.max_age is None
    df = DemoFilter('m')
    assert df.as_string() == 'm'
    assert df.sex_filter == 'm'
    assert df.min_age is None
    assert df.max_age is None
    df = DemoFilter('a<10')
    assert df.as_string() == 'a<10'
    assert df.sex_filter is None
    assert df.min_age is None
    assert df.max_age == 10
    df = DemoFilter('a>10.5')
    assert df.as_string() == 'a>10.5'
    assert df.sex_filter is None
    assert df.min_age == 10.5
    assert df.max_age is None
    df = DemoFilter('a>0.5fa<5')
    assert df.as_string() == 'fa>0.5a<5'
    assert df.sex_filter == 'f'
    assert df.min_age == 0.5
    assert df.max_age == 5
    with pytest.raises(Exception):
        DemoFilter('a>0.5xa<5') # unknown control character
    with pytest.raises(Exception):
        DemoFilter('a>0.5.4fa<5') # bad numeric format
    with pytest.raises(Exception):
        DemoFilter('a>fa<5') # missing number

def disabled_test_name_parsing():
    # XXX This functionality is parked as the ClinicalDatasetName class,
    # XXX and commented out. It may get tweaked and revived as more
    # XXX versioned clin_ev datasets come on line.
    from dtk.faers import ClinicalEventCounts

    FileName = ClinicalEventCounts.FileName
    for cds,vocab,base,file_class,version,supl,filt in [
            # legacy cds formats
            ('FAERS','FAERS','FAERS',None,None,None,None),
            #('CVAROD','CVAROD','CVAROD',None,None,None,None),
            # simple versioning
            ('faers.v3','FAERS','faers.v3','faers','v3',None,None),
            # special case for ws default version
            ('faers.v','FAERS','faers.v','faers','v',None,None),
            # supplemental
            ('FAERS+EXTRA','FAERS','FAERS',None,None,'EXTRA',None),
            ('faers.v3+EXTRA','FAERS','faers.v3','faers','v3','EXTRA',None),
            # filters
            ('faers.v3?f','FAERS','faers.v3','faers','v3',None,'f'),
            ('faers.v?f','FAERS','faers.v','faers','v',None,'f'),
            ('faers.v3+EXTRA?f','FAERS','faers.v3','faers','v3','EXTRA','f'),
            ]:
        fn = FileName.from_string(cds)
        assert fn.dataset_name == cds
        assert fn.vocab == vocab
        assert fn.file_class == file_class
        assert fn.version == version
        assert fn.base == base
        assert fn.supl == supl
        assert fn.filt == filt


def test_diff_table():
    import pandas as pd
    data1 = [
        ['Name', 'Type', 'Score'],
        ['Drug1', 'KT', 12],
        ['Drug2', 'KT', 4],
        ['Drug3', '', 21],
        ['Drug4', 'KT', -10],
    ]

    data2 = [
        ['Name', 'Type', 'Score'],
        ['Drug1', 'KT', 13],
        ['Drug2', 'KT', -100],
        ['Drug3', '', -200],
        ['Drug4', 'KT', -15],
    ]

    df1 = pd.DataFrame(data1)
    df1, df1.columns = df1[1:] , df1.iloc[0] # Use first line as header
    df2 = pd.DataFrame(data2)
    df2, df2.columns = df2[1:] , df2.iloc[0]

    from dtk.faers import make_diff_table
    diff_df = make_diff_table(df1, df2,
        sort_col='Score',
        id_col='Name',
        best_n=2,
        keep_cols=['Name', 'Type'],
        diff_cols=['Score'],
        abs_order=False,
    )

    print(diff_df)

    # Drug1&2 are top2 in df1.
    # Drug1&3 are top2 in df3.

    assert diff_df.values.tolist() == [
        # Name, Type, Score A, Score B, Rank A, Rank B
        ['Drug1', 'KT', 12, 13, 1, 0],
        ['Drug3', '', 21, -200, 0, 3],
        ['Drug4', 'KT', -10, -15, 3, 1],
    ]