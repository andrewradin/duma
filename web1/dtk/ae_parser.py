#!/usr/bin/env python3

from builtins import range
import sys
from path_helper import PathHelper
import os, time
import subprocess as sp
import logging
logger = logging.getLogger(__name__)

# XXX Further cleanup:
# XXX - get the model training stuff working again:
# XXX   - this currently resides in algorithms/run_gesearchmodel.py;
# XXX     this hasn't been tried in a while
# XXX   - there's a related model page linked from the progress page,
# XXX     but that page crashes both in production and in the test
# XXX     environment
# XXX   - there's an older version of this in databases/aesearch_training,
# XXX     which should probably just be removed
# XXX - dtk now holds all the relevant logic in ae_search, ae_parser, and
# XXX   sra_bigquery, but none of it is very well commented, and it could
# XXX   maybe be organized a little better

class Experiment:
    def __init__(self, accession, disease, tr_flag):
        self.accession = accession
        self.disease = disease
        self.tr_flag = tr_flag
        self.title = None
        self.description = None
        self.experiment_type = None # rna-seq or microarray, etc
        self.array_types = None # chip type
        self.experiment_vars = None
        self.score = 0.0
        self.samples = []
        self.alt_ids = []
class disease:
    def __init__(self, **kwargs):
        self.search_term = kwargs.get('term', None)
        self.ws = kwargs.get('ws', None)
        self.original = self.search_term.replace('+', ' ').strip('"').lower()
    def get_highlights(self):
        self._fetch_efo_pickle()
        self._find_disease_efo()
        self.find_highlights()
    def _find_disease_efo(self):
        self.efos = [k for k in self.efo_dict_id.keys()
                     if self.efo_dict_id[k].name
                     and self.original == self.efo_dict_id[k].name
                    ]
        if len(self.efos) == 0:
            self.efos = [k for k in self.efo_dict_id.keys()
                         if self.original in self.efo_dict_id[k].synonyms
                        ]
    def find_highlights(self):
        self.names = list(set([self.efo_dict_id[efo].name
                               for efo in self.efos
                              ]))
        self.syns = list(set([syn
                              for efo in self.efos
                              for syn in self.efo_dict_id[efo].synonyms
                              ]))
        self.children = list(set([child
                                  for efo in self.efos
                                  for child in self.efo_dict_id[efo].children
                                  ]))
    def _fetch_efo_pickle(self):
        import pickle
        from dtk.disease_efo import Disease
        from dtk.s3_cache import S3File
        from browse.default_settings import efo
        s3f = S3File.get_versioned(
                'efo',
                efo.value(self.ws),
                role='obo',
                )
        s3f.fetch()
        self.efo_dict_id = pickle.load(open(s3f.path(), 'rb'))

# TODO this is only called in a GE test that will need to be re-written once we update this to a CM based apprpoach
def parse_ae_sample_xml(ae_sample_xml_str):
    import lxml.etree as ET
    root = ET.fromstring(ae_sample_xml_str)
    samples = []
    for sample in root.iter('sample'):
        sample_dict = {}
        for attr in sample.iter('characteristic'):
            key = attr.find('category').text
            value = attr.find('value').text
            sample_dict[key] = value
        for attr in sample.iter('variable'):
            key = attr.find('name').text
            value = attr.find('value').text
            sample_dict[key] = value
        for attr in sample.iter('source'):
            value = attr.find('name').text
            sample_dict['_source'] = value
        samples.append(sample_dict)

    return samples

def parse_ae_sample_sdrf(src, decode=True):
    samples = []
    header = None
    for line in src:
        if decode:
            line = line.decode('utf8', errors='ignore')
        rec = line.strip().split('\t')
        if header is None:
            header = []
            for field in rec:
                if '[' in field:
                    field = field.split('[')[1].split(']')[0]
                header.append(field)
            continue
        samples.append(dict(zip(header, rec)))
# XXX removed this limit when we started using this for the base search
# XXX I haven't seen any downsides, but it may be that this bogs down some large datasets
#        if len(samples) >= 1000:
#            logger.warn("WARNING: parsed 1000 samples, ending parse")
#            break
    return samples

def fetch_acc_samples(acc_id, ftp_path):
    # There are 3 places  we can get this sample data from - the AE API, the http sdrf file,
    # and the ftp sdrf file.
    # API is pretty slow, http is faster but will throttle for a minute if you make a bunch of
    # requests in a row, ftp seems the best for now.
    logger.info(f"Pulling samples for {acc_id}")
    for retry in range(10):
        import ftplib
        import urllib
        try:
            url = f'{ftp_path}/Files/{acc_id}.sdrf.txt'
            import io
            with urllib.request.urlopen(url, timeout=60) as req:
                parsed_samples = parse_ae_sample_sdrf(req)
                logger.info(f"Done {acc_id}")
                return parsed_samples
        except ftplib.error_temp as e:
            time.sleep(retry + 1)
            logger.info(f"Retrying ftp temporary error for {acc_id}: {e} ({retry})")
# for some reason some files are missing from the FTP source, but seem to be there for HTTPS
        except (urllib.error.URLError, ftplib.error_perm) as e:
            from dtk.url import get_html
            logger.info(f"Retrying via http permanent error for {acc_id}: {e}")
            url = f'https://www.ebi.ac.uk/arrayexpress/files/{acc_id}/{acc_id}.sdrf.txt'
            txt=get_html(url)
            parsed_samples = parse_ae_sample_sdrf(txt.decode().splitlines(), decode=False)
            logger.info(f"Done via HTTPs {acc_id}")
            return parsed_samples
        except:
            import traceback as tb
            tb.print_exc()
            # There are a small number of AE accessions that give a server error when you try
            # to pull them.
            logger.error(f"ERROR: Failed while parsing AE acc {acc_id}")
            return []

def get_biostudies_study_info(accession):
    import json
    from dtk.url import get_html
    url = f'https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}/info'
    txt = get_html(url)
    return json.loads(txt)

class parser(object):
    def __init__(self, **kwargs):
        from browse.models import AeSearch
        self.disease = disease(term = kwargs.get('disease', None),
                               ws = kwargs.get('ws', None)
                              )
        self.species = kwargs.get('species', AeSearch.species_vals.human)
        self.tr_flag = kwargs.get('tr_flag', False)
        self.search_type = 'TR' if self.tr_flag else 'CC'
        self.replacement_terms = [
                                  ('names', 'dumaPrimarySearchTerm'),
                                  ('syns', 'dumaSearchTermSynonym'),
                                  ('children', 'dumaSearchTermChild')
                                 ]
        self.sep = " "
        self.model_prefix = "_".join(['AE', self.search_type, 'search'])
        self.base_url = "https://www.ebi.ac.uk/biostudies/api/v1/arrayexpress/search?query="
# it may be that these, and the other options should be available on the settings of a CM
        exp_types = '&'.join(['facet.study_type='+x for x in ['high-throughput+sequencing',
                                  'human+-+high-throughput+sequencing',
                                  'human+-+single-cell+sequencing',
                                  'human+-+two-color+microarray',
                                  'human+-+one-color+microarray',
                                  'microrna+profiling+by+array',
                                  'microrna+profiling+by+high+throughput+sequencing',
                                  'rna-seq+of+coding+rna',
                                  'rna-seq+of+non+coding+rna',
                                  'rna-seq+of+coding+rna+from+single+cells',
                                  'rna-seq+of+total+rna',
                                  'single-cell+sequencing',
                                  'transcription+profiling+by+array',
                                  'transcription+profiling+by+tiling+array'
                                  ]
                               ])
        self.final_url = f'&{exp_types}&pagesize=100'
        if self.species != AeSearch.species_vals.any:
            self.final_url += '&organism='
            species_string = AeSearch.latin_of_species(self.species)
            self.final_url += species_string.replace(" ", "+")

    def fetch_samples_for_accs(self, acc_ids):
        from dtk.ae_search import find_existing_samples
        from dtk.parallel import pmap
        out, remaining = find_existing_samples(acc_ids)
        results = pmap(self.fetch_acc_samples,remaining,num_cores=10)
        out.update(dict(zip(remaining, results)))
        return out

    def fetch_acc_samples(self, acc_id):
# XXX we've already done this in the original search.
# XXX I haven't tried to make his work, but there's no reason we should need to pull this again
        study_info = get_biostudies_study_info(acc_id)
        return fetch_acc_samples(acc_id, study_info['ftpLink'])

    def run(self, add_new_results=True):
        self.get_json()
        self.disease.get_highlights()
        self.results = {}
        for hit in self.hits:
            self.load_experiment(hit)

        logger.info(f"Fetching samples for {len(self.results)} experiments")
        accs_to_fetch = [k for k, result in self.results.items()]
        if not add_new_results:
            from browse.models import AeAccession
            existing_accs = set(AeAccession.objects.filter(ws=self.disease.ws, geoID__in=accs_to_fetch).values_list('geoID', flat=True))
            accs_to_fetch = [x for x in accs_to_fetch if x in existing_accs]
            logger.info(f"Not adding new results, only fetching for {len(accs_to_fetch)}")
        samples_data = self.fetch_samples_for_accs(accs_to_fetch)
        for acc, samples in samples_data.items():
            self.results[acc].samples = samples
        logger.info(f"Samples fetched")

    def predict(self, wsid):
        from dtk.files import safe_name
        from path_helper import make_directory
        dir = os.path.join(PathHelper.storage,
                                 wsid
                           )
        make_directory(dir)
        self.ofp = os.path.join(dir,
                                "_".join([self.model_prefix,
                                          safe_name(self.disease.search_term)
                                         ]
                                        )
                               )
        self.arff = self.ofp + '_pred.arff'
        self.predStats_file = self.ofp+'_predStats.txt'
        self.load_model_pickle()
        self.setup_pred_arff()
        self.run_predictions()
        self.parse_results()
        self.swap_out_general_search_terms()
    def swap_out_general_search_terms(self):
        for attr, replacement in self.replacement_terms:
            # The easiest way to do this is use the first term, though that will lose some info
            l = getattr(self.disease, attr, None)
            if l is None or len(l) < 1:
                continue
            first_term = l[0]
            for k in self.results:
                self.results[k].title = self.results[k].title.replace(replacement, first_term)
                self.results[k].description = self.results[k].description.replace(replacement, first_term)
    def run_predictions(self):
        try:
            import run_eval_weka as rew
        except ImportError:
            sys.path.insert(0, PathHelper.MLscripts)
            import run_eval_weka as rew
        self.preds = rew.runWeka(self.method,
                            None,
                            self.arff,
                            model_name=self.model_file,
                            cost_list=self.cost_list
                           )
    def parse_results(self):
        for line in self.preds.split("\n"):
            if not line:
                continue
            fields = line.rstrip().split()
            if fields[0] == "===" or fields[0].strip().startswith("inst") :
                continue
            if fields[3] == "+":
                fields[3] = fields[4]
            final = float(fields[3].replace('*','').split(",")[0])
            self.results[self.final_rows[int(fields[0])-1]].score = final
    def load_model_pickle(self):
        import pickle
        pkl_name = self.model_prefix+'.pickle'
        pkl_file = os.path.join(PathHelper.storage, pkl_name)
        if not os.path.isfile(pkl_file):
            sp.check_call(['s3cmd','get','s3://duma-datasets/'+pkl_name, pkl_file])
        pkl = pickle.load(open(pkl_file, 'rb'))
        self.cost_list, self.method, self.model_file, self.used_attrs = pkl
        self.model_file = os.path.join(PathHelper.storage,self.model_file)
    def setup_pred_arff(self):
        self._build_mat()
        self._write_arff()
    def _write_arff(self):
        write_sparse_arff("AE %s dataset predictor" % (self.search_type),
                          self.used_attrs,
                          self.final_mat,
                          self.final_rows,
                          {x:'?' for x in self.final_rows},
                          self.arff
                         )
    def _build_mat(self):
        amb = ae_matrix_builder(self.results)
        amb.build_mat(ddf = 0.0)
        self.final_mat = amb.final_mat
        self.final_cols = amb.final_cols
        self.final_rows = amb.final_rows
        self._filter_mat()
    def _filter_mat(self):
        import scipy.sparse as sparse
        if not set(self.final_cols).issubset(self.used_attrs):
            missing = [x for x in self.used_attrs if x not in self.final_cols]
            empty_csr = sparse.csr_matrix((len(self.final_rows), len(missing)))
            self.final_mat = sparse.hstack([self.final_mat, empty_csr], format='csr')
            self.final_cols += missing
        inds = [self.final_cols.index(x) for x in self.used_attrs]
        self.final_mat = self.final_mat.tocsc()[:,inds]
    def load_experiment(self,accession):
        import json
        from dtk.url import biostudies_studies_url,get_html
        study_info = get_biostudies_study_info(accession)
        # this would be a spot to check the stime stamp in study_info['modified'] to see if we can use a previous result
        study_json = json.loads(get_html(biostudies_studies_url(accession)))
        self.results[accession] = Experiment(accession,
                                             self.disease,
                                             self.tr_flag
                                             )
        study_attrs = study_json['section']['attributes']

        self.results[accession].experiment_type = self._get_experiment_type(study_attrs)

        self.results[accession].orig_title = self._get_title(study_json)
        self.results[accession].title = self._prep_text(self.results[accession].orig_title)

        orig_desc = self._get_description(study_attrs)
        self.results[accession].orig_description = orig_desc if orig_desc else 'None'
        self.results[accession].description = self._prep_text(self.results[accession].orig_description)

# XXX in my examples this is looking very hard to parse, but more importantly isn't delivering anything informative
#        self.results[accession].array_types = self._get_array_design()

        self.results[accession].pub_ref = self._get_doi( study_json['section']['subsections'])

        self.parse_sample_table(accession, study_info['ftpLink'])
    def _get_title(self,study_json):
        for x in study_json['attributes']:
            if x['name'].lower() == 'title':
                return x['value'].title()
    def _get_doi(self, subsecs):
# XXX In this new parser I have not checked to see how well this works compared to the previous parser
# XXX but this shouldn't be vital to getting it to run, so I'll defer for now
        for x in subsecs:
            # without explanation or pattern I can see, sometimes X is a list
            try:
                attr_list = x.get('attributes',[])
            except AttributeError:
                continue
            for y in attr_list:
                if y.get('name', '') == 'DOI':
                    return '' if y['value'] is None else y['value']
        return ''

    def parse_sample_table(self, accession, ftp_link):
        sample_data = fetch_acc_samples(accession, ftp_link)
        self.results[accession].sample_n = float(len(sample_data))
        self.results[accession].table_headers = self.sep.join([self._prep_text(k.lower()) for k in sample_data[0]])
        self.results[accession].table_vals = self.sep.join([self._prep_text(v.lower()) for x in sample_data for _,v in x.items()])

    def _get_array_design(self):
#XXX as noted above this has not been updated to work with the json data b/c this field does not
#XXX seem to have been transferred to the new format. I'm also not sure it's helpful info
        array_design = self.exp.findall('arraydesign')
        to_ret = self.sep.join([array_name.find('name').text.lower()
                                for array_name in array_design
                               ]
                              )
        return self._prep_text(to_ret)
    def _get_description(self, attrs):
        for attr in attrs:
            if attr['name'].lower() != 'description':
                continue
            return attr['value']
    def _get_experiment_type(self, attrs_list):
        exp_types=[]
        for atr in attrs_list:
            if atr['name'].lower() != 'study type':
                continue
            if atr['value']:
                exp_types.append(atr['value'])
        to_ret = self.sep.join(exp_types)
        return self._prep_text(to_ret)
    def _prep_text(self, s):
        if s is None:
            return ''
        for attr, replacement in self.replacement_terms:
            l = getattr(self.disease, attr)
            for term in l:
                s = s.replace(term, replacement)
        return self._clean_string(s)
    def _clean_string(self, s):
        from nltk.corpus import stopwords
        from dtk.num import is_number
        import re
        eng_stopwords = stopwords.words("english")
        return self.sep.join([w for w
                                in re.split(r'\W+', s.replace(':', ' ').replace('-', ' '))
                              if w not in eng_stopwords
# I can't come up with a compelling reason why just digits would be informative
# so I'm removing them
                              and not is_number(w)
                             ])
    def get_json(self):
        import json
        from dtk.url import get_three_part_html
        page=1
        logger.info(f"Pulling search results from Biostudies, page {page}")
        first_json_txt = get_three_part_html(self.base_url,
                       self.disease.search_term,
                       self.final_url
                       )
        first_json = json.loads(first_json_txt)
        if 'hits' not in first_json:
            self.hits=[]
            logger.info(f"No results matched via: {self.disease.search_term}")
        else:
            self.hits=[x['accession'] for x in first_json['hits']]
        logger.info(f"{self.disease.search_term} matched {first_json['totalHits']} results")
        while pages_left(page, first_json['pageSize'], first_json['totalHits']):
            page += 1
            logger.info(f"Pulling search results from Biostudies, page {page}")
            next_json = json.loads(get_three_part_html(self.base_url,
                       self.disease.search_term,
                       f'{self.final_url}&page={page}'
                       ))
            self.hits += [x['accession'] for x in next_json['hits']]

def pages_left(current_page, hits_per_page, totalHits):
    if hits_per_page*current_page > totalHits:
        return False
    return True

def write_sparse_arff(title, attrs, data, rows, labels, ofn):
    with open(ofn,'w') as f:
        f.write('@RELATION "%s"\n' % (title))
        f.write('\n')
        for c in attrs:
            f.write(f'@ATTRIBUTE "{c}" REAL\n')
        f.write('@ATTRIBUTE "DumaReady" {True, False}\n')
        f.write('\n')
        f.write('@DATA\n')
        f.write('\n')
### This can't be the most efficient way to do this: we go to a dense list and then to a sparse ARFF
        for iter, r in enumerate(data.todense().tolist()):
            k = rows[iter]
            lab = str(labels[k])
            f.write(sparse_arff_row(r + [lab]) + '\n')
        f.write('\n')
def sparse_arff_row(l):
    vals = []
    for i,x in enumerate(l):
        if x != 0:
            vals.append(" ".join([str(i), str(x)]))
    return '{' + ", ".join(vals) + '}'

class ae_matrix_builder(object):
    def __init__(self, data):
        self.data = data
        self.fields = ['title',
                       'description',
                       'table_headers',
                       'table_vals',
                       'experiment_type'
                      ]
    def build_mat(self, ddf = 0.04):
        all = {}
        for f in self.fields:
            l = []
            row_names = []
            raw = []
            for k in self.data:
                x = getattr(self.data[k], f)
                l.append(full_lem(x))
                raw.append(x)
                row_names.append(k)
            if not any(l):
                # I believe this is happening in the rare case where we have no
                # AE results, as some of these fields are AE-only.
                logger.warning(f"No useful data from {f} '{raw}'")
                continue
            mat, cols = bag_words(l, ddf = ddf)
            all[f] = (mat, cols, row_names)
        all = self.add_single_cols(all)
        self._combine_mats(all)
    def _combine_mats(self, all):
        import scipy.sparse as sparse
        mats = []
        self.final_cols = []
        self.final_rows = None
        for f,tup in all.items():
            if self.final_rows is None:
                self.final_rows = tup[2]
            else:
                assert self.final_rows == tup[2]
            mats.append(tup[0])
            self.final_cols += [":".join([f,x]) for x in tup[1]]
        self.final_mat = sparse.hstack(mats, format='csr')
    def add_single_cols(self, d):
        import scipy.sparse as sparse
        row_names = []
        doi = []
        n = []
        for k in self.data:
            row_names.append(k)
            flag = getattr(self.data[k], 'pub_ref')
            if flag is None or flag == '':
                val = 0.0
            else:
                val = 1.0
            doi.append(val)
            n.append(getattr(self.data[k], 'sample_n'))
        n_mat = sparse.csr_matrix((n, (list(range(len(n))),
                              [0]*len(n)
                              )
                          ),
                          shape=(len(n),1)
                        )
        mat = sparse.csr_matrix((doi, (list(range(len(doi))),
                              [0]*len(doi)
                              )
                          ),
                          shape=(len(doi),1)
                        )
        d['sample_n'] = (n_mat, ['sample_n'], row_names)
        d['doi'] = (mat, ['doi'], row_names)
        return d
def full_lem(s):
    import nltk
    word_list = [
                 nltk.stem.WordNetLemmatizer().lemmatize(
                     nltk.stem.WordNetLemmatizer().lemmatize(
                         w,'v'
                     )
                 )
                 for w in s.split()
                ]
    # I was finding some specific examples not being condensed.
    # this function does it 'manually'
    lemd = " ".join(my_Lemmatizer(word_list))
    return custom_replacements(lemd)
### There are a few sets of words that should be combined because when
### they are next to each other they have special meaning.
### These are different from my lem'er b/c their pairs of words.
### I do this after the lem'er to avoid having to make multiple matches.
### In generall, the approach is to concatenate the words together
### thereby making them a single word which will work with the bag of words
def custom_replacements(s):
    to_replace = get_replacement_pairs()
    for current, replacement in to_replace:
        s = s.replace(current, replacement)
    return s
def my_Lemmatizer(l):
    to_replace = get_replace_dict()
    for i,w in enumerate(l):
        if w in to_replace:
            l[i] = to_replace[w]
    return l
def bag_words(l, ddf=0.02):
    from sklearn.feature_extraction.text import TfidfVectorizer
    transformed_vectorizer = TfidfVectorizer(analyzer = "word",
                                 tokenizer = None,
                                 preprocessor = None,
                                 stop_words = None,
                                 max_df = 1.0-ddf,
                                 min_df = ddf,
# don't need to smooth b/c we already required x instances above (min_df = x)
                                 smooth_idf=False
                                )
    x = transformed_vectorizer.fit_transform(l)
    columns = transformed_vectorizer.get_feature_names()
    return x, columns

def get_replacement_pairs():
    # A list of tuple is preferable as it means
    # I know the order the words are replaced
    # spaces are added to make sure we don't
    # replace in the middle of a word
    return[
           (' cell line ', ' cellline '),
           (' wild type ', ' wildtype '),
           (' genome wide ', ' genomewide '),
           (' amino acid ', ' aminoacid '),
           (' gene expression ', ' geneexpression '),
           (' transcription factor ', ' transcriptionfactor '),
          ]

def get_replace_dict():
    return {
        'directly' : 'direct',
        'cells' : 'cell',
        'cellular' : 'cell',
        'cases' : 'case',
        'epithelial' : 'epithelia',
        'epithelium' : 'epithelia',
        'esophageal' : 'esophagus',
        'expression' : 'express',
        'fetal' : 'fetus',
        'infection' : 'infect',
        'metastatic' : 'metastasis',
        'monocytic' : 'monocyte',
        'muscular' : 'muscule',
        'mutation' : 'mutant',
        'nodular' : 'node',
        'ovarian' : 'ovary',
        'pancreatic' : 'pancreas',
        'transfected' : 'transfect',
        'transfection' : 'transfect',
        'wt' : 'wildtype',
        'activity' : 'active',
        'activate' : 'active',
        'activation' : 'active',
        'additional' : 'addition',
        'additionally' : 'addition',
        'alteration' : 'alter',
        'amplification' : 'amplify',
        'analyse' : 'analyze',
        'analyses' : 'analyze',
        'analysis' : 'analyze',
        'analyzed' : 'analyze',
        'arrays' : 'array',
        'association' : 'associate',
        'biological' : 'biology',
        'biologic' : 'biology',
        'characterization' : 'characteristic',
        'characterize' : 'characteristic',
        'classification' : 'class',
        'classify' : 'class',
        'clinically' : 'clinic',
        'clinical' : 'clinic',
        'comparative' : 'compare',
        'comparison' : 'compare',
        'changes' : 'change',
        'combination' : 'combine',
        'conditions' : 'condition',
        'conclusions' : 'conclusion',
        'controls' : 'control',
        'correlation' : 'correlate',
        'dataset' : 'data',
        'diagnostic' : 'diagnose',
        'diagnosis' : 'diagnose',
        'diagnoses' : 'diagnose',
        'difference' : 'different',
        'differential' : 'different',
        'differentially' : 'different',
        'differentiation' : 'differentiate',
        'discovery' : 'discover',
        'diseases' : 'disease',
        'donors' : 'donor',
        'enrichment' : 'enrich',
        'experimental' : 'experiment',
        'exposure' : 'expose',
        'extraction' : 'extract',
        'functional' : 'function',
        'genes' : 'gene',
        'genetic' : 'gene',
        'genomic' : 'genome',
        'groups' : 'group',
        'growth' : 'grow',
        'heterogeneity' : 'heterogeneous',
        'higher' : 'high',
        'highly' : 'high',
        'hours' : 'hour',
        'hr' : 'hour',
        'hybridization' : 'hybridize',
        'hypothesize' : 'hypothesis',
        'identification' : 'identify',
        'importantly' : 'important',
        'individuals' : 'individual',
        'induction' : 'induce',
        'infection' : 'infect',
        'inflammatory' : 'inflame',
        'inflammation' : 'inflame',
        'inhibition' : 'inhibit',
        'inhibitor' : 'inhibit',
        'involvement' : 'involve',
        'isolation' : 'isolate',
        'leukemic' : 'leukemia',
        'levels' : 'level',
        'likely' : 'like',
        'lines' : 'line',
        'lower' : 'low',
        'lowly' : 'low',
        'metabolic' : 'metabolism',
        'methods' : 'method',
        'microarrays' : 'microarray',
        'micrornas' : 'microrna',
        'mirna' : 'microrna',
        'mirnas' : 'microrna',
        'mir' : 'microrna',
        'mirs' : 'microrna',
        'molecular' : 'molecule',
        'newly' : 'new',
        'normalization' : 'normalize',
        'overexpression' : 'overexpressed',
        'pathogenesis' : 'pathological',
        'pathologic' : 'pathological',
        'pathways' : 'pathway',
        'patients' : 'patient',
        'performed' : 'perform',
        'poorly' : 'poor',
        'potentially' : 'potential',
        'prediction' : 'predict',
        'presence' : 'present',
        'previously' : 'previous',
        'production' : 'product',
        'produce' : 'product',
        'profiles' : 'profile',
        'prognostic' : 'prognosis',
        'reveals' : 'reveal',
        'results' : 'result',
        'recently' : 'recent',
        'regulation' : 'regulate',
        'regulator' : 'regulate',
        'regulatory' : 'regulate',
        'relationship' : 'relate',
        'resistance' : 'resistant',
        'samples' : 'sample',
        'selection' : 'select',
        'sensitivity' : 'sensitive',
        'seq' : 'sequence',
        'severity' : 'severe',
        'significance' : 'significant',
        'significantly' : 'significant',
        'smoker' : 'smoke',
        'specifically' : 'specific',
        'studies' : 'study',
        'subjects' : 'subject',
        'subtypes' : 'subtype',
        'types' : 'type',
        'systemic' : 'system',
        'targets' : 'target',
        'technologies' : 'technology',
        'therapeutic' : 'therapy',
        'tissues' : 'tissue',
        'transcription' : 'transcript',
        'transcriptional' : 'transcript',
        'transcriptomic' : 'transcriptome',
        'tumors' : 'tumor',
        'understood' : 'understand',
        'useful' : 'use',
        'using' : 'use',
        'validation' : 'validate',
        'vs' : 'versus',
        'treat' : 'treatment',
        'lymphoblastic' : 'lymphoblast',
        'lymphoblastoid' : 'lymphoblast',
        'myelogenous' : 'myeloid',
        'embryonal' : 'embryo',
        'embryonic' : 'embryo',
        'musculus' : 'mouse',
    }

