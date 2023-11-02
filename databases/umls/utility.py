#!/usr/bin/env python

import pwd,os
user=pwd.getpwuid(os.getuid())[0]
root='/home/%s/2xar/' % user

import sys
sys.path.insert(1,root+'twoxar-demo/web1/')

def setup_django():
    if not 'django' in sys.modules:
        print 'loading django'
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
        import django
        django.setup()

################################################################################
# This program is unrelated to the UMLS mapping files produced by
# databases/umls/Makefile. Instead, this file uses dtk.umls to access
# a more complete UMLS extraction (in UMLS's preferred format of multiple
# RRF files) in order to experiment with converting the disease labeling
# of NashBio data so it can be used interchangeably with FAERS. This is
# not actively supported.
################################################################################

def dump_down(cui,chd,c2n,indent,seen):
    print indent,cui,c2n[cui]
    skipped_count = 0
    seen_count = 0
    for child in chd.fwd.get(cui,[]):
        if child in seen:
            seen_count += 1
        elif child in c2n:
            seen.add(child)
            dump_down(child,chd,c2n,'  '+indent,seen)
        else:
            skipped_count += 1
    print '  '+indent,skipped_count,'not meddra;',seen_count,'already seen'

def trace_down(args):
    assert args.cui
    # XXX above could fall back to phrase, as below
    import dtk.umls as umls
    chd = umls.CUIREL('CHD')
    c2n = umls.get_meddra_cui2name()
    dump_down(args.cui,chd,c2n,'',set())

def trace_up(args):
    assert args.cui
    # XXX above could fall back to phrase, as below
    import dtk.umls as umls
    chd = umls.CUIREL('PAR')
    c2n = umls.get_meddra_cui2name()
    dump_down(args.cui,chd,c2n,'',set())

def raw_dump(cui,chd,indent,seen):
    children = chd.fwd.get(cui,[])
    print indent,cui,children
    seen_count = 0
    for child in children:
        if child in seen:
            seen_count += 1
        else:
            seen.add(child)
            raw_dump(child,chd,'  '+indent,seen)
    print '  '+indent,seen_count,'already seen'

def raw_trace(args):
    assert args.cui
    import dtk.umls as umls
    chd = umls.CUIREL('PAR','inverse_isa')
    raw_dump(args.cui,chd,'',set())

def trace(args):
    assert args.phrase
    import dtk.umls as umls
    n2c = umls.Name2CUI()
    cui_set = n2c.fwd[args.phrase]
    tc2n = umls.get_meddra_cui2name()
    targ_set = set(tc2n)
    parents = umls.CUIREL('PAR')
    seen = set()
    while cui_set:
        print cui_set
        # The following displays the names for each parent CUI, and the number
        # of children associated with that CUI. In general, CUIs with fewer
        # children are more specific. In the first generation, we could just
        # order the CUIs by increasing numbers of children, and select the
        # first in the target set. For higher generations, the ordering should
        # be influenced by both the immediate count and the count of
        # intervening generations. Maybe as a first cut, only one generation
        # of search is needed.
        for cui in cui_set:
            print ' ',cui,len(parents.rev[cui])
            for n in n2c.rev[cui]:
                print '   ',n
        overlap = cui_set & targ_set
        if overlap:
            for cui in overlap:
                print ' ',cui,tc2n[cui]
            # XXX commented out to walk up entire hierarchy
            # XXX break
        parent_set = set()
        for cui in cui_set:
            if cui in seen:
                continue
            try:
                parent_set |= parents.fwd[cui]
            except KeyError:
                pass
            seen.add(cui)
        cui_set = parent_set
        cui_set -= seen

# This is the prototype for the Nash indication mapping. 
def strmap4(args):
    from dtk.umls import UMLSIndiMapper,get_meddra_cui2name
    uim = UMLSIndiMapper()
    from collections import Counter
    cui_ctr = Counter()
    stats = dict()
    from dtk.files import get_file_records
    aliases=dict([
        ('type ii diabetes mellitus uncontrolled','type ii diabetes mellitus'),
        ])
    for rec in get_file_records(
            '../nashBio/RAW_condition_occurence.csv',
            keep_header=False,
            parse_type='csv_strict',
            ):
        # Most of this is working around NashBio peculiarities. The RAW
        # file is csv, where sometimes quoted strings span newlines.
        # csv_strict handles this, but they also backslash commas inside
        # the quoted strings, so the backslashes need to be stripped here.
        # Also, force lower case to match how we import UMLS.
        # XXX Still not fixed is the encoding. e.g. Sjogren's disease appears
        # XXX in UMLS both with a latin 'o', and with an umlat, encoded c3 b6.
        # XXX In NashBio, it's encoded c3 83 c2 b6.
        # XXX Similarly, in Meniere's disease, c3 a9 is encoded c3 83 c2 a9
        # XXX and c3 a8 is encoded c3 83 c5 a1.
        s=rec[1].lower()
        s=s.replace('\\,',',')
        s=aliases.get(s,s)
        cui,how = uim.translate(s)
        ctr = stats.setdefault(how,Counter())
        ctr[s] += 1
        if cui:
            cui_ctr[cui] += 1
    for how in stats:
        ctr = stats[how]
        with open(how+'.log','w') as f:
            for s,cnt in ctr.most_common():
                f.write('\t'.join([str(cnt),s])+'\n')
    c2n = get_meddra_cui2name()
    with open('CUI_USAGE.log','w') as f:
        for cui,cnt in cui_ctr.most_common():
            f.write('\t'.join([str(cnt),cui,c2n[cui]])+'\n')

################################################################################
# Stuff below here hasn't been converted for the relocation of core
# utilities to dtk.
################################################################################
dgn_dir='../disgenet/'

# schema info at:
# https://www.ncbi.nlm.nih.gov/books/NBK9685/
#
# highlights:
# - MRFILES and MRCOLS define file formats
# - MRDOC defines enumerated column values
# - MRCONSO maps each external atom to its UMLS CUI
# - MRSTY holds the sematic type for each concept
#   - 69738 CUIs are of type 'Disease or Syndrome'
#   - there are 127 distinct types; see output of:
#     cut -d\| -f4 MRSTY.RRF | sort | uniq -c | sort -nr
# - MRREL, MRMAP, and MRSMAP hold relationships between concepts
# - MRHIER holds source hierarchies; these are looked up by CUI, but the data
#   is in terms of AUIs; it holds a complete path from the root in the source
#   vocabulary; the description of this file says:
#   "NLM editors do not assert concept-level (CUI-to-CUI) hierarchical
#   relationships. Hierarchical relationships are asserted by sources
#   at the atom level (AUI-to-AUI)."
# - there are several language-specific word and phrase index files that may
#   be useful; note that the string version converts the input to a
#   canonicalized form that is not itself useful (splits into words on all
#   spaces and punctuation, lower-cases, and sorts resulting words)

# MedDRA mapping test cases:
# - C2875903 fails to map, but appears in MRHIER; the parent atom maps to
#   C0029668, which has a MedDRA expression
#   - this same relationship is encoded in two rows in MRREL with types of
#     'PAR' and 'CHD'
# - C0700208 is a DGN key that doesn't map directly to MedDRA but seems to
#   be a disease. (scoliosis or some variation on 'Acquired scoliosis')
#   - C0036439 seems to be the preferred MDR term; MRREL contains a row
#     asserting C0700208|PAR|C0036439
#   - two other asserted parents are:
#     - C4023747 (HPO Abnormality of the curvature of the vertebral column)
#     - C0264160 (SNOMEDCT_US Acquired curvature of spine)

class KeyCounter:
    def __init__(self):
        self.keys = 0
        self.recs = 0
    def count(self,v):
        self.keys += 1
        self.recs += v
    def __str__(self):
        return '%d keys, %d recs'%(self.keys,self.recs)

class MRSTY:
    def __init__(self):
        from dtk.files import get_file_records
        self.cui2sty = dict()
        ci = MRFILES.get('MRSTY.RRF')
        from dtk.files import get_file_records
        for fields in get_file_records(
                umls_path('MRSTY.RRF'),
                parse_type='psv',
                ):
            self.cui2sty[fields[ci.CUI]] = fields[ci.STY]

class MRCONSO:
    def __init__(self):
        from collections import Counter
        from dtk.files import get_file_records
        d = dict()
        c = Counter()
        lines = 0
        ci = MRFILES.get('MRCONSO.RRF')
        sty = sty_filter(set(['Disease or Syndrome']))
        for fields in get_file_records(
                umls_path('MRCONSO.RRF'),
                parse_type='psv',
                ):
            cui = fields[ci.CUI]
            #if cui not in sty:
            #    continue
            # Adding the filter above drops the stats from:
            #   missed 210 keys, 2573 recs
            #   matched 8754 keys, 76654 recs
            # to:
            #   missed 4715 keys, 65892 recs
            #   matched 4249 keys, 13335 recs
            # but maybe it better focuses us on the keys that are diseases
            # rather than phenotypes.
            s1,s2=d.setdefault(cui,(set(),set()))
            s1.add(fields[ci.SAB])
            s2.add(fields[ci.STR])
            c[fields[11]] += 1
            lines += 1
        print lines,'lines scanned;',len(d),'UMLS codes loaded'
        print 'Sources:',c
        self.lookup=d
    def sources(self,key):
        return frozenset(self.lookup[key][0])
    def source(self,key):
        d=self.lookup
        s1,s2=d[key]
        assert len(s1) == 1,'key %s has multiple sources: %s'%(key,str(s1))
        return list(s1)[0]

# XXX test alternative nash mapping:
# XXX - extract all name,code pairs from RAW_condition file
# XXX   cut -d, -f2,5 RAW_condition_occurence.csv | sort -u >/tmp/yyy.csv
# XXX   Check cleanup in current extraction code:
# XXX   - some aren't valid ICD codes; try using name?
# XXX   - have leading/trailing punctuation
# XXX   - 29957 unique combinations

def has_mdr_parent(rel,mdr,cui):
    if cui not in rel.fwd:
        return False
    for pcui in rel.fwd[cui]:
        if pcui in mdr.rev:
            return True
        if has_mdr_parent(rel,mdr,pcui):
            return True
    return False

def dump2(rel,cui,skip,indent=''):
    print indent,cui,'(SKIP)' if cui in skip else ''
    if cui in skip:
        return
    skip.add(cui)
    if cui in rel.fwd:
        for pcui in rel.fwd[cui]:
            dump2(rel,pcui,skip,indent+'  ')

def loop(args):
    rel = CUIREL('PAR')
    start = 'C0576471'
    dump2(rel,start,set())

def meddra_triples():
    from path_helper import PathHelper
    fn = PathHelper.storage+'meddra.v19.tsv'
    from dtk.files import get_file_records
    for code,attr,val in get_file_records(fn):
        yield code,attr,val

class IndiUpMapper:
    def __init__(self):
        from path_helper import PathHelper
        fn = PathHelper.storage+'meddra.v19.tsv'
        from dtk.files import get_file_records
        from dtk.data import MultiMap
        mm = MultiMap(
                (val,code)
                for code,attr,val in get_file_records(fn)
                if attr == 'pt_llts' and code != val
                )
        self.fwd = mm.fwd_map()
        self.rev = mm.rev_map()
        print len(self.fwd),'source codes loaded'
        print len(self.rev),'target codes loaded'

class IndiMapper:
    def __init__(self):
        from path_helper import PathHelper
        fn = PathHelper.storage+'meddra.v19.tsv'
        from dtk.files import get_file_records
        from dtk.data import MultiMap
        lookup_attrs = set([
                'adr_term',
                'synonym',
                ])
        mm = MultiMap(
                (val.lower(),code)
                for code,attr,val in get_file_records(fn)
                if attr in lookup_attrs
                )
        self.fwd = mm.fwd_map()
        self.rev = mm.rev_map()
        print len(self.fwd),'names loaded'
        print len(self.rev),'codes loaded'

def cmp_set(l1,s1,l2,s2):
    s='%d in both; %d in %s only; %d in %s only'
    print s%(len(s1&s2),len(s1-s2),l1,len(s2-s1),l2)
    print l2,'only'
    for x in s2-s1:
        print '  ',x

# Existing IndiMap implementations build:
# - a map from names and synonyms to codes
# - a map from codes to prefered codes
# - a map from codes to canonical names
# The first one is used to produce a list of codes for a name, and then the
# last two are used to map that code back to a preferred name. This may not
# be working as expected:
# - if we ignore pt_llts records that map an id back to itself, none of the
#   'from' pt_llts codes have names in the meddra.v19.tsv file; this implies
#   no mapping up of low-level terms is actually taking place
# There are 24014 distinct codes in the meddra.v19.tsv file, with 77912
# distinct strings mapping into them. 21920 of the codes have pt_llts records,
# but some of these just alias a code to itself; only 12971 are aliased from
# other codes.
# All 24014 distinct codes have one adr_term record each.
# XXX Can we extract this same set of MedDRA terms from UMLS?
#
# UMLS holds 80033 distinct MDR strings and 53173 codes.
# - this is in 102243 records, so some names are duplicated
# - the TTY field specifies the record subtype; 22210 are 'PT'
# - 21875 of the MedDRA and PT names overlap; 2138 are unique to MedDRA;
#   335 are unique to UMLS
#   - sampling the 2138 unique to MedDRA, they seem to be mostly higher-level
#     categories ('breast disorders', 'visual colour distortions')
#
# XXX If we assume the above is a good proxy for the final set of target
# XXX MedDRA strings, we can build a mapper class that can take a string
# XXX and produce the closest MedDRA string. This could then be used to
# XXX re-run Nash.

# XXX Alternatives are:
# XXX - re-extract MedDRA with lower-level strings
# XXX - see if we can extract the needed data from UMLS:
# XXX   - the set of MedDRA preferred keys
# XXX   - a broader set of strings that can be mapped to those keys (and back to
# XXX     a preferred term)
# XXX Once the above is done, re-run Nash ETL with an updated IndiMapper, and
# XXX apply a metric to the change.
def v19stats(args):
    im = IndiMapper()
    mdr = Name2CUI(sab='MDR')
    cmp_set('UMLS names',set(mdr.fwd.keys()),'MedDRA names',set(im.fwd.keys()))

def get_nash_indi_strings():
    from dtk.files import get_file_records
    d = dict()
    for fields in get_file_records('/tmp/yyy.csv'):
        if len(fields) < 2:
            print fields
            continue
        s = d.setdefault(fields[0].strip('"\\'),set())
        s.add(fields[1])
    return set(x.lower() for x in d.keys())

def mapping_error(s):
    raise RuntimeError('ambiguous mapping: '+repr(s))

# XXX get the following to work
def extract_meddra2(args):
    from path_helper import PathHelper
    print PathHelper.cfg('UMLS_directory')
    umed = UMLSMedDRA()
    print len(umed.cui_set)
    for cui in umed.cui_set:
        n = umed.canonical_name(cui)

def extract_meddra(args):
    from dtk.files import get_file_records
    ci = MRFILES.get('MRCONSO.RRF')
    from dtk.data import MultiMap
    mm = MultiMap(
        (fields[ci.CUI],(fields[ci.TS],fields[ci.STR].lower()))
        for fields in get_file_records(
                umls_path('MRCONSO.RRF'),
                parse_type='psv',
                )
        if fields[ci.TTY]=='PT' and fields[ci.SAB]=='MDR'
        )
    print len(mm.fwd_map()),'CUIs extracted'
    from collections import Counter
    ctr=Counter()
    for cui,s in mm.fwd_map().iteritems():
        if len(s) == 1:
            ctr['UNIQUE'] += 1
            continue
        s2=set([x for x in s if x[0] == 'P'])
        if len(s2) == 1:
            ctr['UNIQUE_TS'] += 1
            continue
        print cui,s
        ctr['NON_UNIQUE'] += 1
    print ctr

def strmap3(args):
    uim = UMLSIndiMapper()
    from collections import Counter
    if True:
        ctr=Counter()
        cui_set = set()
        source_strings = get_nash_indi_strings()
        for s in source_strings:
            cui,how = uim.translate(s)
            ctr[how] += 1
            cui_set.add(cui)
            if True and how == 'CHILD':
                print s,cui,uim.n2c.rev[cui]
        print len(cui_set),'distinct targets found'
        print ctr

def strmap2(args):
    from dtk.files import get_file_records
    d = dict()
    for fields in get_file_records('/tmp/yyy.csv'):
        if len(fields) < 2:
            print fields
            continue
        s = d.setdefault(fields[0].strip('"\\'),set())
        s.add(fields[1])
    source_strings = set(x.lower() for x in d.keys())
    print len(source_strings),'distinct indi strings present in Nash data'
    n2c = Name2CUI()
    rel = CUIREL('PAR')
    ci = MRFILES.get('MRCONSO.RRF')
    # targets are MedDRA 'preferred terms'
    targets = set(
            fields[ci.STR].lower()
            for fields in get_file_records(
                    umls_path('MRCONSO.RRF'),
                    parse_type='psv',
                    )
            if fields[ci.TTY]=='PT' and fields[ci.SAB]=='MDR'
            )
    from collections import Counter
    ctr=Counter()
    for s in source_strings:
        if s in targets:
            ctr['EXACT'] += 1
            continue
        try:
            cui_set = n2c.fwd[s]
        except KeyError:
            ctr['NO_CUI'] += 1
            continue
        pcui = None
        for cui in cui_set:
            if n2c.rev[cui] & targets:
                ctr['ALIAS'] += 1
                continue
            pcui = get_target_parent(rel,targets,cui,set())
            if pcui:
                break
        if pcui:
            ctr['PARENT'] += 1
        else:
            ctr['MISSED'] += 1
    print ctr

def get_target_parent(rel,targets,cui,seen):
    if cui in seen:
        return None
    if cui not in rel.fwd:
        return None
    seen.add(cui)
    for pcui in rel.fwd[cui]:
        if pcui in targets:
            return pcui
        ancestor = get_target_parent(rel,targets,pcui,seen)
        if ancestor:
            return ancestor
    return None

# for each name, we want to produce a MDR mapping
# - we may want to reduce multiples prior to this mapping? or assume the first
#   successfully matched CUI should be used?
# - some names will map directly to a MDR CUI
def strmap(args):
    from dtk.files import get_file_records
    d = dict()
    for fields in get_file_records('/tmp/yyy.csv'):
        if len(fields) < 2:
            print fields
            continue
        s = d.setdefault(fields[0].strip('"\\'),set())
        s.add(fields[1])
    print len(d)
    n2c = Name2CUI()
    if False:
        # source type breakdown
        sty = MRSTY()
        from collections import Counter
        ctr = Counter()
        for key in d:
            key = key.lower()
            try:
                cui_set = n2c.fwd[key]
            except KeyError:
                ctr['NONE'] += 1
                continue
            if len(cui_set) > 1:
                ctr['MULTIPLE'] += 1
                ctr['MULTIPLE_EXPANDED'] += len(cui_set)
                if False:
                    print '--------'
                    for x in cui_set:
                        print x,sty.cui2sty[x],n2c.rev[x]
                        # for C0036529 and C1689817, there's same_as records in
                        # both directions; there's also a 'was_a' record
                        # indicating that C0036529 has been replaced; maybe
                        # this is true for all multiples?
            for cui in cui_set:
                ctr[sty.cui2sty[cui]] += 1
        print ctr
    if True:
        mdr = Name2CUI(sab='MDR')
        rel = CUIREL('PAR')
        from collections import Counter
        ctr = Counter()
        for key in d:
            key = key.lower()
            try:
                cui_set = n2c.fwd[key]
            except KeyError:
                ctr['NO_CUI'] += 1
                continue
            for cui in cui_set:
                if cui in mdr.rev:
                    ctr['EXACT'] += 1
                    continue
                try:
                    if has_mdr_parent(rel,mdr,cui):
                        ctr['PARENT'] += 1
                        continue
                except RuntimeError:
                    if False:
                        dump2(rel,cui,set())
                    ctr['DEPTH'] += 1
                    continue
                ctr['MISSED'] += 1
        print ctr
    
def test(args):
    # 'oldtest' below is overly simplistic:
    # - a single code may have multiple MRCONSO records
    # - these records may map the code to multiple vocabularies, and to
    #   multiple different texts within each vocabulary
    # Checking the 10 most common missed CUIs in DisGeNet:
    # - the top one is C4020899 (Autosomal recessive predisposition), one of a
    #   small number of terms in HPO that aren't "Phenotypic Abnormalities"
    # - the next 8 refer to some developmental failure (Dull intelligence,
    #   Poor school performance, Delayed cognitive development, etc.)
    # - the 10th is the first one that's a disease (Acquired scoliosis)
    #   - FAERS (and thus MedDRA) contains 'scoliosis'; a good test case might
    #     be if we could map to that as a preferred term via UMLS
    # It may be that we learn more approaching it from the NashBio side, if
    # most of the missed stuff from DisGeNet is phenotypes rather than
    # diseases.
    cons=MRCONSO()
    from collections import Counter
    from dtk.files import get_file_records
    c = Counter()
    missed = KeyCounter()
    matched = KeyCounter()
    show = 10
    for fields in get_file_records(
            dgn_dir+'missed_umls_codes.log',
            parse_type='tsv',
            ):
        recs = int(fields[1])
        try:
            c[cons.sources(fields[0])] += recs
            matched.count(recs)
            if show:
                show -= 1
                print fields[0],recs, cons.lookup[fields[0]]
        except KeyError:
            missed.count(recs)
    print 'missed',missed
    print 'matched',matched
    print 'Match Sources:',c

def oldtest(args):
    from collections import Counter
    from dtk.files import get_file_records
    d = dict()
    c = Counter()
    lines = 0
    for fields in get_file_records(
            umls_path('MRCONSO.RRF'),
            parse_type='psv',
            ):
        d[fields[0]] = fields[11]
        c[fields[11]] += 1
        lines += 1
    print lines,'lines scanned;',len(d),'UMLS codes loaded'
    print 'Sources:',c
    c = Counter()
    missed = KeyCounter()
    matched = KeyCounter()
    for fields in get_file_records(
            dgn_dir+'missed_umls_codes.log',
            parse_type='tsv',
            ):
        recs = int(fields[1])
        try:
            c[d[fields[0]]] += recs
            matched.count(recs)
        except KeyError:
            missed.count(recs)
    print 'missed',missed
    print 'matched',matched
    print 'Match Sources:',c

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='''\
''',
            )
    parser.add_argument('cmd',
            )
    parser.add_argument('--phrase',
            )
    parser.add_argument('--cui',
            )
    args = parser.parse_args()
    locals()[args.cmd](args)

    
