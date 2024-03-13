from dtk.lazy_loader import LazyLoader
from functools import reduce

class TextModel(LazyLoader):
    # should be able to start with just the vocabulary and
    # vectorizer, and build everything else from there
    _kwargs=['vocabulary','prepare','vectorizer','extra_data','similarity']
    def score(self,test_cases):
        self.matrix # will trigger fitting if not already done
        # self.matrix has a row for each target, and a column for each feature,
        # with the cell value indicating how strongly the target reflects the
        # feature.
        #
        # test_matrix below is similar, but with a row per test case.
        #
        # We derive a score for each target against each test case by
        # measuring the similarity of the test case and target vectors.
        # Then, we give each target its best score across all test cases.
        test_matrix = self.vectorizer.transform([
                self.prepare(x)
                for x in test_cases
                ])
        sm = self.similarity(test_matrix,self.matrix)
        from collections import namedtuple
        Match = namedtuple('Match','score text extra')
        result = []
        for i,n in enumerate(test_cases):
            match_scores = [
                    Match(*x)
                    for x in zip(sm[i],self.vocabulary,self.extra_data)
                    ]
            match_scores.sort(key=lambda x:-x.score)
            result.append(match_scores)
        return result
    def _matrix_loader(self):
        # build self.matrix based on vectorizer, prepare, and vocabulary
        return self.vectorizer.fit_transform([
                self.prepare(x)
                for x in self.vocabulary
                ])
    def _extra_data_loader(self):
        return [None]*len(self.vocabulary)
    def _prepare_loader(self):
        # if not overridden in ctor, prepare is just a pass-thru function
        return lambda x:x
    def _similarity_loader(self):
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity

from dtk.subclass_registry import SubclassRegistry
class DiseaseVocab(SubclassRegistry):
    '''Base class for looking up disease names.

    A derived class instance represents a particular ontology. Typically the
    entire ontology is loaded when the class is instanciated, making this
    a relatively heavy-weight object. The items() method, which must be
    defined by the derived class, returns all words in the ontology and
    their corresponding codes. This is all that's required for the ontology
    to work with the disease names page.

    header() and detail() methods may be overridden to display additional
    per-disease information.

    If the ontology has an underlying website with informative per-disease
    web pages, a 'disease_info_link' method may be defined to populate links
    on to the disease summary page.
    '''
    lookup_href=None
    # if a derived class provides a lookup_href, it is assumed that the
    # vocabulary is stored externally, and the provided href is a web
    # interface for searching it; there is no 'ontobrowse' functionality,
    # and the top-level disease name page entry must be updated manually;
    # this provides a minimal UI for storing additional disease name types
    # for the workspace (originally for NDFRT KT lookup)
    def __init__(self,version_defaults=None):
        self.version_defaults = version_defaults
    @classmethod
    def get_choices(cls):
        return [(x,x) for x in cls.get_all_names()]
    @classmethod
    def get_instance(cls,key,**kwargs):
        Type = cls.lookup(key)
        return Type(**kwargs)
    def name(self):
        return self.__class__.__name__
    # Translate between a single string pattern and a list of individual
    # codes. Derived classes supporting multiple codes per pattern can
    # set multi_select = True, and optionally set a special delimiter.
    # More complex transformations can override build_pattern() and
    # pattern_elements().
    multi_delim = '|'
    multi_select = False
    def build_pattern(self,code):
        if self.multi_select:
            assert isinstance(code,list)
            return self.multi_delim.join(str(x) for x in sorted(code))
        return code
    def pattern_elements(self,pat):
        return pat.split(self.multi_delim)
    # All derived classes must supply an items() method that returns a list
    # of (code,disease name) pairs; if there's no natural code for the
    # vocabulary, it can return the disease name twice. In addition, any
    # of the following hooks can be overridden:
    def list_help(self):
        # override to supply html for upper right of ontolist page
        pass
    def header(self):
        # override to supply additional column headers for ontolist page
        return tuple()
    def detail(self,code):
        # override to supply additional column data for ontolist page
        return (
                code,
                )
    filter_idx=None
    default_cutoff=1000

class GwasPhenotype(DiseaseVocab):
    def __init__(self,**kwargs):
        super(GwasPhenotype,self).__init__(**kwargs)
        file_class = 'duma_gwas'
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned(
                file_class,
                self.version_defaults[file_class],
                role='studies',
                )
        s3f.fetch()
        self.data = {}
        from dtk.files import get_file_records
        for rec in get_file_records(s3f.path(),keep_header=False):
            if hasattr(rec[0], 'decode'):
                rec[0] = rec[0].decode('utf8')
            phen,pmid = rec[0].split('|')
            try:
                cnt = int(rec[1])
            except ValueError:
                # one record has the value "2943,1878"
                cnt = 0
            self.data[rec[0]] = (phen.replace('_',' '),pmid,cnt)
    def items(self):
        return [(k,v[0]) for k,v in self.data.items()]
    def header(self):
        return ('Pubmed Id','Sample Count')
    def detail(self,code):
        v = self.data[code]
        from dtk.url import pubmed_url
        from dtk.html import link
        return (
                link(v[1], pubmed_url(v[1]), new_tab=True),
                v[2],
                )
    filter_idx=1

class ClinicalTrials(DiseaseVocab):
    multi_select = True
    def __init__(self,**kwargs):
        super(ClinicalTrials,self).__init__(**kwargs)
        from ctsearch.utils import get_ct_search_file
        fn = get_ct_search_file(self.version_defaults)
        from dtk.files import get_file_records
        from collections import Counter
        self.disease_counts = Counter([
                rec[2] for rec in get_file_records(fn,keep_header=False)
                ])
    def items(self):
        return [
                (k,k)
                for k in self.disease_counts.keys()
                ]
    def header(self):
        return ('Trial count',)
    def detail(self,code):
        return (self.disease_counts[code],)
    filter_idx=0
    default_cutoff=100


class OrangeBook(DiseaseVocab):
    def __init__(self,**kwargs):
        super(OrangeBook,self).__init__(**kwargs)
        import dtk.orange_book as dtkob
        self.ob=dtkob.OrangeBook(self.version_defaults['orange_book'])
    multi_select = True
    def items(self):
        return [('U-%d'%x.code,x.desc) for x in self.ob.get_use_codes()]
    def header(self):
        return ('Use Code','New Drug Applications','','Drug names','')
    def detail(self,code):
        assert code.startswith('U-')
        uc_set = set([int(code[2:])])
        ndas = self.ob.get_ndas_for_uses(uc_set)
        drugs = self.ob.get_names_for_ndas(ndas)
        from dtk.url import ob_nda_url
        ndas = ['%06d'%x for x in ndas]
        from dtk.html import link,join
        return (
                code,
                len(ndas),
                join(*[link(x,ob_nda_url(x)) for x in ndas]),
                len(drugs),
                ', '.join(drugs),
                )
    filter_idx=3
    default_cutoff=1

class OpenTargets(DiseaseVocab):
    multi_select = True
    def __init__(self,**kwargs):
        super(OpenTargets,self).__init__(**kwargs)
        file_class = 'openTargets'
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned(
                file_class,
                self.version_defaults[file_class],
                role='names',
                )
        s3f.fetch()
        self._items = []
        self._counts = {}
        from dtk.files import get_file_records
        for name,code,count in get_file_records(s3f.path()):
            if hasattr(name, 'decode'):
                name = name.decode('utf8')
            self._items.append((code,name))
            self._counts[code] = int(count)
    def items(self):
        return self._items
    def build_pattern(self,codes):
        # The 'key:' prefix is to match the sytax supported by run_otarg,
        # which accepts disease names as well as keys.
        return ','.join(['key:'+code for code in codes])
    def pattern_elements(self,pat):
        import re
        matches = [re.match('key:(.*)',x) for x in pat.split(',')]
        return [x.group(1) for x in matches if x is not None]
    def header(self):
        return ('Reference','Protein count')
    def detail(self,code):
        from dtk.url import open_targets_disease_url
        from dtk.html import link
        return (
                link(code, open_targets_disease_url(code), new_tab=True),
                self._counts[code],
                )
    def disease_info_link(self,code):
        from dtk.html import join,link
        from dtk.url import efo_url,open_targets_disease_url
        disease = '???'
        for k,v in self._items:
            if k == code:
                disease = v
                break
        parts = [
                link(f'{disease} ({code})',
                        open_targets_disease_url(code),
                        new_tab=True,
                        ),
                ]
        if code.startswith('EFO_'):
            parts.append( link(f'@OLS', efo_url(code), new_tab=True) )
        return join(*parts)

    filter_idx=1
    default_cutoff=100

class WebSearch(DiseaseVocab):
    lookup_href="https://google.com"

class DisGeNet(DiseaseVocab):
    multi_select = True
    multi_delim = ','
    def __init__(self,**kwargs):
        super(DisGeNet,self).__init__(**kwargs)
        self._items = []
        file_class = 'disgenet'
        from dtk.s3_cache import S3File
        try:
            s3f = S3File.get_versioned(
                file_class,
                self.version_defaults[file_class],
                role='cui_disease_names',
                )
            s3f.fetch()
# XXX see no_items below
        except RuntimeError:
            self.msg = "Unsupported data version. Please use DisGeNet v3 or later."
        else:
            from dtk.files import get_file_records
            for code,name in get_file_records(s3f.path()):
                if hasattr(name, 'decode'):
                    name = name.decode('utf8')
                self._items.append((code,name))
    def items(self):
        if self._items:
            return self._items
        return None
# XXX This approach is currently only used here in DisGeNet,
# XXX but may be helpful for other vocabs later
    def no_items(self):
# XXX The second object is essentially an empty items object in order for the page to load
        return self.msg, [('',self.msg)]
    def header(self):
        return ('Reference',)
    def detail(self,code):
        from dtk.html import link
        from dtk.url import disgenet_url
        return (
                link(code,
                        disgenet_url(code,'dis_map'),
                        new_tab=True,
                        ),
                )
    def disease_info_link(self,code):
        from dtk.html import link,join
        from dtk.url import medgen_url,disgenet_url
        disease = '???'
        for k,v in self._items:
            if k == code:
                disease = v
                break
        return join(
                link(f'{disease} ({code})',
                        disgenet_url(code,'gda_ev'),
                        new_tab=True,
                        ),
                link(f'@MedGen',
                        medgen_url(code),
                        new_tab=True,
                        ),
                )
    filter_idx=None

class AGR(DiseaseVocab):
    multi_select = True
    def __init__(self,**kwargs):
        super(AGR,self).__init__(**kwargs)
        file_class = 'agr'
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned(
                file_class,
                self.version_defaults[file_class],
                role='human',
                )
        s3f.fetch()
        from dtk.files import get_file_records
        self._code_info = {}
        # Fill in _code_info as follows:
        # {doid:[disease_name,association_count],...}
        for rec in get_file_records(
                s3f.path(),
                keep_header=False,
                ):
            l = self._code_info.setdefault(rec[1],[])
            if l:
                l[1] += 1
            else:
                l.append(rec[0])
                l.append(1)
    def items(self):
        return list(
                (k,l[0])
                for k,l in self._code_info.items()
                )
    def header(self):
        return ('DOID','Record Count')
    def detail(self,code):
        from dtk.html import link
        from dtk.url import agr_url
        return (
                link(code, agr_url(code), new_tab=True),
                self._code_info[code][1],
                )
    def disease_info_link(self,code):
        from dtk.html import link
        from dtk.url import agr_url
        disease = self._code_info[code][0]
        return link(f'{disease} ({code})', agr_url(code), new_tab=True)

class Monarch(DiseaseVocab):
    multi_select = True
    def __init__(self,**kwargs):
        super(Monarch,self).__init__(**kwargs)
        file_class = 'monarch'
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned(
                file_class,
                self.version_defaults[file_class],
                role='disease',
                )
        s3f.fetch()
        from dtk.files import get_file_records
        self._code_info = {}
        # Fill in _code_info as follows:
        # {mondo_id:[disease,pheno_count],...}
        for rec in get_file_records(
                s3f.path(),
                keep_header=False,
                ):
            l = self._code_info.setdefault(rec[0],[])
# if we've already seen this ID, increment the count
            if l:
                l[1] += 1
            else:
# if we haven't seen it, set up a new list/row
                l.append(rec[1])
                l.append(1)
    def items(self):
        return list(
                (k,l[0])
                for k,l in self._code_info.items()
                )
    def header(self):
        return ('Disease','Record Count')
    def detail(self,code):
        from dtk.html import link
        from dtk.url import monarch_disease_url
        return (
                link(code, monarch_disease_url(code), new_tab=True),
                self._code_info[code][1],
                )
    def disease_info_link(self,code):
        from dtk.html import link
        from dtk.url import monarch_disease_url
        disease = self._code_info[code][0]
        return link(f'{disease} ({code})', monarch_disease_url(code), new_tab=True)

class EFO(DiseaseVocab):
    multi_select = True
    multi_delim = ';'
    def __init__(self,**kwargs):
        super(EFO,self).__init__(**kwargs)
        self._items = []
        file_class = 'efo'
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned(
                file_class,
                self.version_defaults[file_class],
                role='terms',
                )
        s3f.fetch()
        from dtk.files import get_file_records
        self.syns={}
        self.children={}
        self.ids={}
        for code,name,synonyms,children in get_file_records(s3f.path(), keep_header=True):
            if hasattr(name, 'decode'):
                name = name.decode('utf8')
            self._items.append((name,name))
            self.ids[name]=code.replace(":","_")
            self.syns[name]=synonyms
            self.children[name]=children
    def items(self):
        if self._items:
            return self._items
        return None
    def header(self):
        return ('Synonyms', 'Children terms',)
    def detail(self,code):
        from dtk.html import link
        from dtk.url import efo_url
        id = self.ids[code]
        link_val = 'No link available: ' + id
        if id.startswith('EFO_'):
            link_val = link(id, efo_url(id), new_tab=True)
        return (
                self.syns[code],
                self.children[code],
                link_val,
                )
    filter_idx=None

class Chembl(DiseaseVocab):
    multi_select = True
    def __init__(self,**kwargs):
        super(Chembl,self).__init__(**kwargs)
        from dtk.indications import ChemblIndications
        self._items,self._counts = ChemblIndications().get_disease_info()
    def items(self):
        return list(self._items.items())
    def header(self):
        return ('MeSH','Reference Counts')
    def detail(self,code):
        from dtk.html import link
        from dtk.url import mesh_url
        return (
                link(code,
                        mesh_url(code),
                        new_tab=True,
                        ),
                self._counts[code],
                )
    def disease_info_link(self,code):
        from dtk.html import link
        from dtk.url import mesh_url
        disease = self._items[code]
        return link(f'{disease} ({code})',
                        mesh_url(code),
                        new_tab=True,
                        )
    filter_idx=1
    default_cutoff=1

class CecMixin(object):
    def __init__(self,**kwargs):
        super(CecMixin,self).__init__(**kwargs)
        from dtk.faers import ClinicalEventCounts
        cds = self.cec_source
        if cds in self.version_defaults:
            cds = cds + '.' + self.version_defaults[cds]
        print('using cds',cds)
        self.cec=ClinicalEventCounts(cds)
        self.counts = dict(self.cec.get_disease_names_and_counts())
    def items(self):
        return [(x,x) for x in self.counts.keys()]
    multi_select = True
    def list_help(self):
        return 'Database contains %d events' % self.cec.total_events()
    def header(self):
        return (
                'Event count',
                )
    def detail(self,code):
        return (
                self.counts[code],
                )
    filter_idx=0

if False: # disable CVAROD
 class CVAROD(CecMixin,DiseaseVocab):
    cec_source = 'CVAROD'
    default_cutoff=10

class FAERS(CecMixin,DiseaseVocab):
    cec_source = 'faers'
    default_cutoff=100

if False:
  # XXX if this gets re-enabled, it should also be converted to versioning
  class MedDRA(DiseaseVocab):
    def items(self):
        from path_helper import PathHelper
        fn = PathHelper.storage+'meddra.v19.tsv'
        from dtk.files import get_file_records
        return [
                (x[0],x[2])
                for x in get_file_records(fn)
                if x[1] == 'adr_term'
                ]

class VocabMatcher(LazyLoader):
    def __init__(self,iterable,**kwargs):
        super(VocabMatcher,self).__init__(**kwargs)
        codes,phrases = zip(*iterable)
        self._codes = codes
        phrases = [self._parse(x) for x in phrases]
        # build phrase-level model
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import linear_kernel
        # Tfidf doesn't work very well for cases like OrangeBook, because
        # it overly-penalizes long phrases (a short phrase containing one
        # term of interest gets chosen over a longer phrase containing all
        # terms of interest). So, instead, we implement a straight count
        # of the number of terms matched:
        # - the CountVectorizer in binary mode just flags whether or not
        #   each term is present
        # - the linear_kernel is a straight dot product (like cosine_similarity,
        #   but without any normalization)
        self.phrase_model = TextModel(
                vocabulary=[' '.join(x) for x in phrases],
                extra_data=codes,
                vectorizer=CountVectorizer(
                        tokenizer=self.tokenizer,
                        binary=True,
                        ),
                similarity=linear_kernel,
                )
        # extract all individual words
        import operator
        words = list(reduce(operator.__or__,[
                set(phrase)
                for phrase in phrases
                ]))
        # build word-level model
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.word_model = TextModel(
                vocabulary=words,
                # by wrapping words in spaces, we'll create digraphs
                # even for single letter words, and also emphasize
                # first and last letters
                prepare=lambda x:" %s "%x,
                vectorizer=TfidfVectorizer(
                        analyzer='char',
                        ngram_range=(2,2),
                        )
                )
    def map_words(self,phrase):
        # returns [[phrase_word,[possible_target_word,...],...]
        source_words = self._parse(phrase)
        word_scores = self.word_model.score(source_words)
        word_scores = [
                self.filter_scores(score_list,
                        high_thresh=0.6,
                        cutoff_fraction=0.6,
                        cutoff_count=5,
                        )
                for score_list in word_scores
                ]
        return [
                (src,[targ.text for targ in targs])
                for src,targs in zip(source_words,word_scores)
                ]
    def score_phrases(self,targ_list):
        # each item in targ_list is a list of alternative words in
        # the target vocabulary that might correspond to a word in
        # the source phrase
        #
        # construct a list of the cartesian product of the targ_list words
        import itertools
        vmapped_phrases = [
                ' '.join(x)
                for x in itertools.product(
                        *[x for x in targ_list if x]
                        )
                ]
        # score each target match to each constructed phrase
        phrase_scores = [
                [score for score in score_list if score.score]
                for score_list in self.phrase_model.score(vmapped_phrases)
                ]
        # condense into a single list with the best score for
        # each target phrase; the 'extra' field is included in
        # the key to keep multiple codes with the same text unique
        d = {}
        for score_list in phrase_scores:
            for item in score_list:
                key = (item.text,item.extra)
                if key in d and item.score < d[key].score:
                    continue
                d[key] = item
        # extract all target words in a single set
        import operator
        all_target_words = set(reduce(operator.add,targ_list))
        # return a list of (best_score,highlighted_phrase,key)
        from dtk.html import tag_wrap,join
        def wrap(x):
            if x in all_target_words:
                return tag_wrap('span',x,attr={
                        'style':'background-color:lightgreen',
                        })
            return x
        return [
                (
                    item.score,
                    join(*[wrap(x) for x in name.split()]),
                    item.extra,
                )
                for (name,extra),item in sorted(
                        d.items(),
                        key = lambda x:x[1].score,
                        reverse = True,
                        )
                ]
    def filter_scores(self,scores,
                high_thresh=0.7,cutoff_count=10,cutoff_fraction=0.8,
                ):
        if not scores or scores[0].score < high_thresh:
            return []
        low_thresh = cutoff_fraction * scores[0].score
        return [
                score
                for score in scores[:cutoff_count]
                if score.score > low_thresh
                ]
    def _tokenizer_loader(self):
        import re
        token_regex=re.compile(r"(?u)\b\w+\b")
        return lambda x:[y.lower() for y in token_regex.findall(x)]
    _prefixes=set(['non'])
    _suffixes=set(['cell'])
    _merge_words=_prefixes|_suffixes
    #noise_words=set(['syndrome','disease','disorder'])
    _noise_words=set()
    def _merge(self,phrase):
        prefix=[]
        infix=[]
        suffix=[]
        for word in phrase:
            if word in self._suffixes:
                suffix.append(word)
                continue
            if infix or suffix:
                yield '_'.join(prefix+infix+suffix)
                prefix=[]
                infix=[]
                suffix=[]
            if word in self._prefixes:
                prefix.append(word)
                continue
            infix.append(word)
        if prefix or infix or suffix:
            yield '_'.join(prefix+infix+suffix)
    def _parse(self,s):
        parts = [x for x in self.tokenizer(s) if x not in self._noise_words]
        if set(parts) & self._merge_words:
            parts = list(self._merge(parts))
        return parts

