#!/usr/bin/env python


import sys
try:
    from clean_simple import bioChemDPICleaner,dpi_counts
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../")
    from clean_simple import bioChemDPICleaner,dpi_counts

input = [['test_id','uniprot_id', 'C50', 'direction'],
             ['x','1','1.0','1'],
# multiple drugs hitting the same prot
             ['x','2','2.0','0'],
             ['y','2','1.0','-1'],
# the same number of measurments for a DPI, but going opposite directions
             ['y','3','1.0','1'],
             ['y','3','0.5','-1'],
# multiple measurements of the same DPI and direction,
# plus a test of the direction majority winning
             ['z','4','1.0','1'],
             ['z','4','1.0','-1'],
             ['z','4','2.0','-1'],
# a test for outliers
             ['z','5','50000.0','-1'],
             ['z','5','5.0','-1'],
             ['z','6','50.0','-1'],
            ]
bcdp = bioChemDPICleaner(input)

def test_dpi_counts():
    input={('x','1'):[(1.0,1),(2.0,0)],
           ('x','2'):[(1.0,1)],
           ('y','2'):[(1.0,-1)],
           ('y','3'):[(1.0,1)],
           ('z','4'):[(1.0,1), (1.0,-1)]
          }
    dc = dpi_counts(input)
    assert dc.drugs==3
    assert dc.prots==4
    assert dc.uniq_dpi==5
    assert dc.total_dpi==7

def test_init():
# just make sure the last step in init happens
    assert bcdp.counts==[]

def test_load():
    bcdp.load()
    assert bcdp.all_vals==[1.,2., 1., 1., .5, 1., 1., 2.,50000.,5.,50.]

def test_get_counts():
    bcdp.get_counts('test')
    assert bcdp.counts[0][0]=='test'
    assert bcdp.counts[0][1].drugs==3
    assert bcdp.counts[0][1].prots==6
    assert bcdp.counts[0][1].uniq_dpi==7
    assert bcdp.counts[0][1].total_dpi==11

def test_remove_outliers():
    bcdp._get_ceiling()
    assert bcdp.upper_ceil==78.82076346195376
    d=bcdp._remove_outliers()
    expected={('x', '1'): [(1., 1)],
              ('x', '2'): [(2., 0)],
              ('y', '2'): [(1., -1)],
              ('y', '3'): [(1., 1), (.5, -1)],
              ('z', '4'): [(1., 1), (1., -1), (2., -1)],
              ('z', '5'): [(5., -1)],
              ('z', '6'): [(50., -1)]
             }
    assert d==expected

def test_collapse_rptd_measures():
    d = bcdp.collapse_rptd_measures()
    expected={('x', '1'): [(1., 1, 1, 0.)],
              ('x', '2'): [(2., 0, 1, 0.)],
              ('y', '2'): [(1., -1, 1, 0.)],
              ('y', '3'): [(1., 1, 1, 0.), (.5, -1, 1, 0.)],
              ('z', '4'): [(1., 1, 1, 0.), (1.41, -1, 2, .71)],
              ('z', '5'): [(500., -1, 2, 35351.80)],
              ('z', '6'): [(50., -1, 1, 0.)]
             }
    assert d==expected

full_filter_expected={('x', '1'): [(1., 1, 1, 0.)],
              ('x', '2'): [(2., 0, 1, 0.)],
              ('y', '2'): [(1., -1, 1, 0.)],
              ('y', '3'): [(.5, -1, 1, 0.)],
              ('z', '4'): [(1.41, -1, 2, 0.71)],
              ('z', '5'): [(5., -1, 1, 0.)],
              ('z', '6'): [(50., -1, 1, 0.)]
             }

# this will also test the last step, collapse direction
def test_filter():
    bcdp.filter()
    assert bcdp.data==full_filter_expected

def test_full():
# end to end now that we've tested most of the parts
    my_bcdp = bioChemDPICleaner(input)
    my_bcdp.load()
    my_bcdp.filter()
    assert bcdp.data==full_filter_expected
    cnts = my_bcdp._org_count_report()
    assert cnts == ['baseline',
                    '\tDrugs:     3',
                    '\tProteins:  6',
                    '\tUniq. DPI: 7',
                    '\tTotal DPI: 11',
                    '',
                    '\t\toutliers\trepeated measures\tdirections',
                    'Drugs    \t\t100.0\t\t100.0\t\t100.0',
                    'Proteins \t\t100.0\t\t100.0\t\t100.0',
                    'Uniq. DPI\t\t100.0\t\t100.0\t\t100.0',
                    'Total DPI\t\t90.91\t\t81.82\t\t63.64',
                    '',
                    'Final totals',
                    '\tDrugs:     3',
                    '\tProteins:  6',
                    '\tUniq. DPI: 7',
                    '\tTotal DPI: 7',
                   ]
