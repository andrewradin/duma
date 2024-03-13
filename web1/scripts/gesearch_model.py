#!/usr/bin/env python
import django_setup

import logging

from sklearn.feature_extraction.text import TfidfVectorizer
logger = logging.getLogger(__name__)
from dataclasses import dataclass
from typing import List
from browse.models import Workspace, AeSearch
import numpy as np

def safe_groupby(iter, key):
    """Normal groupby has some gotchas that this dodges, though less efficient.

    1) It requires sorted input (otherwise only groups adjacent runs of same key)
    2) It generates iterators that get exhausted, and will appear empty if re-iterated
    """
    from collections import defaultdict
    out = defaultdict(list)
    for x in iter:
        keyval = key(x)
        out[keyval].append(x)
    return dict(out)


@dataclass
class DataEntry:
    key: str
    title: str
    description: str
    samples_vals: List[List[str]]
    samples_header: List[str]
    experiment_type: str
    pub_ref: str
    reject_text: str
    num_samples: int
    search_id: int
    search_term: str
    ws_id: int
    ws_name: str

def score_existing(entrys, ws_id, term):
    """ws_id just used for writing out intermediate arff"""
    from dtk.ae_parser import parser, Experiment

    runner = parser(disease=term, ws=Workspace.objects.get(pk=ws_id))
    runner.disease.get_highlights()

    def ae_format(entry):
        e = Experiment(
            accession=entry.key,
            disease=runner.disease,
            tr_flag=False,
        )

        # _prep_text will replace disease highlights with custom terms and removes stop words.
        # during predict the runner also does things like lemmatization
        e.title = runner._prep_text(entry.title)
        e.description = runner._prep_text(entry.description)
        e.table_headers = runner._prep_text(' '.join(set(entry.samples_header)))
        e.table_vals = runner._prep_text(' '.join(set(entry.samples_vals)))
        e.experiment_type = entry.experiment_type
        e.pub_ref = entry.pub_ref
        e.sample_n = len(entry.samples_vals)
        return e

    runner.results = {entry.key: ae_format(entry) for entry in entrys}
    runner.predict(str(ws_id))

    scores = [runner.results[entry.key].score for entry in entrys]
    return scores

def make_entry(aeaccession, aesearch, mode):
    from collections import defaultdict
    sample_data = defaultdict(set)

    samples = aeaccession.get_sample_attrs()
    do_header = True
    for sample in samples:
        for k, v in sample.items():
            sample_data[k].add(v)
    
    samples_header = []
    samples_vals = []
    MAX_WORDS = 100
    for k, v in sample_data.items():
        if len(v) == 1 or len(v) == len(samples):
            # Skip anything that has only 1 value or has all unique values
            continue
        samples_header.append(k)
        for val in v:
            samples_vals.append(' '.join(val.split(' ')[:MAX_WORDS]))
        
    return DataEntry(
        key=aeaccession.geoID,
        title=aeaccession.title,
        description=aeaccession.desc,
        pub_ref=aeaccession.pub_ref or '',
        experiment_type=aeaccession.experiment_type or '',
        samples_header=samples_header,
        samples_vals=samples_vals,
        reject_text=aeaccession.reject_text(mode),
        num_samples=aeaccession.num_samples or len(samples),
        search_id=aesearch.id,
        search_term=aesearch.term,
        ws_id=aesearch.ws.id,
        ws_name=aesearch.ws.name,
    )


def build_dataset(ws_ids):
    ws_ids = list(ws_ids)
    # Shuffle the workspace ids in a deterministic manner, just in case they came in
    # in a specific order (e.g. biggest to smallest)
    import random
    rng = random.Random(0)
    rng.shuffle(ws_ids)
    data = []
    for ws_id in ws_ids:
        data.extend(build_ws_dataset(ws_id))
    
    return data

def build_ws_dataset(ws_id):
    from browse.models import AeSearch, AeDisposition, AeAccession, AeScore, Tissue
    from dtk.ae_search import refresh_ae_search
    mode = AeSearch.mode_vals.CC
    # Find all searches
    searchs = AeSearch.objects.filter(ws=ws_id, mode=mode, species=AeSearch.species_vals.human)

    # We only want to include tissues that are part of a tissue set (i.e. not "excluded tissues") and
    # where the tissue set is not microRNA.
    # Excluded tissues are just ignored, not treated as negative samples (unless they have been explicitly
    # rejected somewhere).
    #
    # This is going to be a problem for building an explicitly microRNA classifier, because we don't have
    # explicitly searches for those, they've just been cherry picked from CC searches, and so in some cases
    # they will have been rejected simply for being miRNA when we weren't looking for miRNA.
    valid_geos = set(Tissue.objects.filter(tissue_set__miRNA=False).values_list('geoID', flat=True).distinct())
    # Split off any user-supplied suffixes  (e.g. GSE1234:GPL91)
    valid_geos = {x.split(':')[0] for x in valid_geos}

    # Split any combined tissues.
    tissues_to_add = set()
    for geo in valid_geos:
        if ',' in geo:
            tissues_to_add.update(geo.split(','))
    valid_geos.update(Tissue.objects.filter(pk__in=tissues_to_add).values_list('geoID', flat=True))

    out = []
    for search in searchs:
        if search.version != AeSearch.LATEST_VERSION:
            logger.info("Refreshing out-of-date AESearch entries")
            refresh_ae_search(search, add_new_results=False)

        # Pick the ones with both positives and negatives
        accs = AeAccession.objects.filter(aescore__search=search).distinct()
        print(f'{search.term}: ({search.id}) {len(accs)} results')


        entries = [make_entry(accession, search, mode) for accession in accs]
        if not entries:
            continue

        imported = set(search.imported())
        rejected = set(search.rejected())
        # TODO: separate category for failed to process?

        def make_label(acc):
            if acc in imported:
                if acc.geoID in valid_geos:
                    return 1
                if acc.alt_ids:
                    for alt_id in acc.alt_ids.split(','):
                        if alt_id in valid_geos:
                            return 1
            elif acc in rejected:
                return -1

            # Anything unlabeled (or imported-but-excluded) will get
            # dropped from the dataset.
            return 0
        
        labels = [make_label(accession) for accession in accs]

        for entry, label in zip(entries, labels):
            if label != 0:
                out.append((entry, label))
    return out


def split_test_train(dataset, test_por, rng=None):
    data_by_ws = safe_groupby(dataset, key=lambda x: x[0].ws_id)
    if rng is None:
        # Pick a reproducible split for comparison convenience.
        from random import Random
        rng = Random(0)
    test_keys = set(rng.sample(data_by_ws.keys(), round(len(data_by_ws) * test_por)))

    data_by_split = safe_groupby(dataset, key=lambda x: x[0].ws_id in test_keys)
    # Returns test, train
    return data_by_split[True], data_by_split[False]

def split_folds(dataset, folds, rng=None):
    """Splits into K folds, based on workspace."""
    data_by_ws = safe_groupby(dataset, key=lambda x: x[0].ws_id)
    if rng is None:
        # Pick a reproducible split for comparison convenience.
        from random import Random
        rng = Random(0)
    
    wses = list(data_by_ws.keys())
    rng.shuffle(wses)

    # Intentionally floating point.
    ws_per_fold = len(wses) / folds
    ws_to_fold = {}
    for i in range(folds):
        low = int(i * ws_per_fold)
        high = int((i+1) * ws_per_fold)
        for ws in wses[low:high]:
            ws_to_fold[ws] = i

    data_by_fold = safe_groupby(dataset, key=lambda x: ws_to_fold[x[0].ws_id])
    return data_by_fold


def score_existing_dataset(dataset):
    # Group by search.
    all_scores = []
    all_labels = []
    from tqdm import tqdm
    for srch_id, group_data in tqdm(safe_groupby(dataset, key=lambda x: x[0].search_id).items()):
        group_data = list(group_data)
        group_entries = [x[0] for x in group_data]
        scores = score_existing(group_entries, group_entries[0].ws_id, group_entries[0].search_term)
        labels = [x[1] for x in group_data]
        all_scores.extend(scores)
        all_labels.extend(labels)


    print(f"Using {len(all_scores)}  {len([x for x in all_labels if x == 1])} positive")

    return all_scores, all_labels


class GEModel:
    def __init__(self, settings):
        self.settings = settings

    def run_train_test(self, train_ds, test_ds):
        train_fmt = self.preprocess(train_ds)
        test_fmt = self.preprocess(test_ds)
        self.train(train_fmt)
        return self.predict(test_fmt), test_fmt, self.predict(train_fmt)
    
    def run_train(self, train_ds):
        train_fmt = self.preprocess(train_ds)
        self.train(train_fmt)
    
    def run_predict(self, ds):
        if not ds:
            return []
        ds_fmt = self.preprocess(ds)
        return self.predict(ds_fmt)
    
    def run_eval(self, ds):
        ds_fmt = self.preprocess(ds)
        scores = self.predict(ds_fmt)
        labels = ds_fmt[1]
        from sklearn.metrics import roc_auc_score
        rating = roc_auc_score(labels, scores)
        return rating

def summarize_results(ds, pred_probs, fmt_ds, train_ds, train_probs):
    table_rows = [['Key', 'Title', 'Description', 'ExpType', 'Pub Ref', 'Pred Score', 'Label', 'Reject Text', 'Search Term', 'WS']]
    for entry, pred_score, label in zip(ds, pred_probs, fmt_ds[1]):
        entry = entry[0]
        row = [entry.key, entry.title, entry.description, entry.experiment_type, entry.pub_ref, pred_score, label, entry.reject_text, entry.search_term, entry.ws_name]
        table_rows.append(row)

    def plot_roc(scores, labels):
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, thresholds = roc_curve(labels, scores)
        rating = roc_auc_score(labels, scores)
        from tools import sci_fmt
        from dtk.plot import scatter2d, annotations
        return rating, scatter2d(
                'False Positive Rate',
                'True Positive Rate',
                zip(fpr,tpr),
                text=['@'+sci_fmt(t) for t in thresholds],
                title=f'ROC Curve',
                refline=False,
                linestyle='lines',
                annotations=annotations('AUC-ROC: %.3f' % (rating))
                )
        
    test_score, test_roc = plot_roc(pred_probs, fmt_ds[1])
    train_score, train_roc = plot_roc(train_probs, [x[1] == 1 for x in train_ds])
    test_roc._data[0]['name'] = 'test'
    train_roc._data[0]['name'] = 'train'
    test_roc._data.append(train_roc._data[0])
    return {
        'plots': {
            'roc': test_roc,
        },
        'table': table_rows,
        'metrics': {
            'roc': test_score,
        }
    }

from functools import lru_cache
class EfoDisease:
    def __init__(self, version):
        self.version = version
        self._fetch_efo_pickle(version)

        name_to_keys = []
        for k, v in self.efo_dict_id.items():
            name_to_keys.append((v.name, k))
        from dtk.data import MultiMap
        self.name_to_keys_mm = MultiMap(name_to_keys)


    @lru_cache(maxsize=None)
    def get_highlights(self, term):
        d_efos, ind_efos = self._find_disease_efo(term)
        return self._find_highlights(d_efos), self._find_highlights(ind_efos)

    def _find_disease_efo(self, term):
        term = term.replace('+', ' ').strip('"').lower()
        direct_efos = [k for k in self.efo_dict_id.keys()
                     if self.efo_dict_id[k].name
                     and term == self.efo_dict_id[k].name
                    ]
        indirect_efos = [k for k in self.efo_dict_id.keys()
                        if term in self.efo_dict_id[k].synonyms
                        and k not in direct_efos
                    ]
        return direct_efos, indirect_efos

    def _find_highlights(self, efos):
        names = list(set([self.efo_dict_id[efo].name
                               for efo in efos
                              ]))
        syns = list(set([syn
                              for efo in efos
                              for syn in self.efo_dict_id[efo].synonyms
                              ]))
        children = list(set([child
                                  for efo in efos
                                  for child in self.efo_dict_id[efo].children
                                  ]))
        child_syns = list(set([syn
                                  for efo in efos
                                  for child in self.efo_dict_id[efo].children
                                  for child_key in self.name_to_keys_mm.fwd_map()[child]
                                  for syn in self.efo_dict_id[child_key].synonyms
                                  ]))
        return {
            'names': names,
            'syns': syns,
            'children': children,
            'child_syns': child_syns,
        }

    def _fetch_efo_pickle(self, version):
        import pickle
        from dtk.disease_efo import Disease
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned('efo', version, role='obo')
        s3f.fetch()
        self.efo_dict_id = pickle.load(open(s3f.path(), 'rb'))


def docfreq(texts, raw_ds):
    seen_doc_ws = set()
    seen_doc = set()
    from collections import Counter, defaultdict
    word_counts = Counter()
    word_doc_counts = Counter()
    ws_seen = defaultdict(set)

    word_doc_pos_counts = Counter()
    word_doc_neg_counts = Counter()
    pos_doc_count = 0
    neg_doc_count = 0

    # TODO: Pass in label for pvalue'ing?
    label = False
    
    for text, ds_entry in zip(texts, raw_ds):
        doc_ws_key = (ds_entry.key, ds_entry.ws_id)
        if doc_ws_key in seen_doc_ws:
            continue
        seen_doc_ws.add(doc_ws_key)

        words = text.split(' ')
        set_words = set(words)

        if ds_entry.key not in seen_doc:
            word_counts.update(words)
            word_doc_counts.update(set_words)
            seen_doc.add(ds_entry.key)

        if label:
            word_doc_pos_counts.update(set_words)
            pos_doc_count += 1
        else:
            word_doc_neg_counts.update(set_words)
            neg_doc_count += 1
        for word in words:
            ws_seen[word].add(ds_entry.ws_id)
    
    from dtk.plot import bar_histogram_overlay, PlotlyPlot
    
    ws_seen = {k:len(v) for k,v in ws_seen.items()}

    # Pvalues disabled for now.
    if False:
        word_pvs = {}

        from fisher import pvalue
        for word in ws_seen:
            a = word_doc_pos_counts[word]
            c = word_doc_neg_counts[word]
            mat = [a, pos_doc_count - a, c, neg_doc_count - c]
            pv = pvalue(*mat)
            word_pvs[word] = pv.two_tail
    
    return word_counts, word_doc_counts, ws_seen

def replace_terms(efo, search_term, text, filter_stopwords, filter_numbers):
    import re
    dir_terms, indir_terms = efo.get_highlights(search_term)
    user_terms = {'query': [search_term.replace('+', ' ').replace('"','')]}

    names = ['direct', 'indirect', 'user']
    for term_type, terms in zip(names, [dir_terms, indir_terms, user_terms]):
        for type_name, term_values in terms.items():
            for term_value in term_values:
                term_value = term_value.lower()
                replacement = f' dumaSearch_{term_type}_{type_name} '
                # Technically this could replace things in the middle of a word,
                # but the regex version is slower and performs worse.
                text = text.replace(term_value, replacement)

    import re
    from dtk.num import is_number

    # Faster than dtk.num is_number and also filters out anything
    # starting with a number like 10mg.
    def is_numberish(w):
        wlen = len(w)
        return (
            (wlen > 0 and w[0].isdigit()) or
            (wlen > 1 and w[0] == '-' and w[1].isdigit())
        )

    words = [w for w in re.split(r'\W+', text.replace(':', '').replace('-', ''))]
    if filter_numbers:
        words = [w for w in words if not is_numberish(w)]
    if filter_stopwords:
        from nltk.corpus import stopwords
        eng_stopwords = stopwords.words("english")
        words = [w for w in words if w not in eng_stopwords]
    return ' '.join(words)

class NoWordsException(Exception):
    pass

class BagOfWordsModel(GEModel):
    def __init__(self, settings):
        super().__init__(settings)
        from browse.default_settings import efo
        ver = efo.latest_version()
        self.efo = EfoDisease(ver)
    
    def save(self, fn):
        with open(fn) as f:
            import json
            f.write(json.dumps(self.serialize()))

    def serialize(self):
        def save_tfidf(tfidf):
            vocab = sorted(tfidf.vocabulary_.items(), key=lambda x: x[1])
            assert vocab[0][1] == 0
            assert len(vocab) == vocab[-1][1] + 1
            return [x[0] for x in vocab]
        

        data = {
            'settings': self.settings,
            'tfidfs': [save_tfidf(x) for x in self.tfidfs],
            'lr': {
                'coef': self.model.coef_.tolist(),
                'intercept': self.model.intercept_.tolist(),
            },
        }

        from runner.process_info import JobInfo
        word_parts = ['title', 'description', 'exp_type', 'sample_header', 'sample_vals'] 
        counts = {part:len(x) for part, x in zip(word_parts, data['tfidfs'])}
        JobInfo.report_info(f"Vocab words counts: {counts}")

        return data

    @classmethod
    def from_file_data(cls, data):
        # LR Model
        model_data = data['lr']
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        model.coef_ = np.array(model_data['coef'])
        model.intercept_ = np.array(model_data['intercept'])

        out = BagOfWordsModel(data['settings']) 

        # TFIDFs
        tfidfs = []
        for sorted_vocab in data['tfidfs']:
            tfidf = out.make_tfidf()
            tfidf.vocabulary_ = {w:i for i,w in enumerate(sorted_vocab)}
            tfidfs.append(tfidf)
        
        out.model = model
        out.tfidfs = tfidfs
        return out
    
    def _replace_terms(self, search_term, text):
        filter_numbers = self.settings.get('filter_numbers', True)
        filter_stopwords = self.settings.get('filter_stopwords', False)
        return replace_terms(
            self.efo,
            search_term,
            text,
            filter_numbers=filter_numbers,
            filter_stopwords=filter_stopwords,
            )

    def preprocess(self, ds):
        def clean_arr(term, arr):
            return [self._replace_terms(term, text.lower()) for text in arr]
        data = [clean_arr(x[0].search_term, [
            x[0].title,
            x[0].description,
            x[0].experiment_type,
            ' '.join(x[0].samples_header),
            ' '.join(x[0].samples_vals),
            ]) for x in ds]
        labels = [x[1] == 1 for x in ds]
        raw = [x[0] for x in ds]
        return data, labels, raw

    def group_transform(self, parts):
        from scipy import sparse
        inv_parts = zip(*parts)
        vecs = [tfidf.transform(x) for x, tfidf in zip(inv_parts, self.tfidfs)]
        return sparse.hstack(vecs, format='csr')
    
    def fit(self, ds, ds_raw):
        settings = self.settings
        wordfilt_num_ws = settings.get('wordfilt_num_ws', 2)
        wordfilt_num_docs = settings.get('wordfilt_num_docs', 3)
        wordfilt_num_occs = settings.get('wordfilt_num_occs', 6)
        tfidfs = []
        # ds is [[title, descr, ...], [title2, descr2, ...], ...]
        # This inverts that into [title1, title2, ...], [descr1, descr2, ...]
        inv_parts = zip(*ds)
        for parts in inv_parts:
            counts, doc_counts, ws_counts = docfreq(parts, ds_raw)
            to_remove = set(
                [k for k,v in counts.items() if v < wordfilt_num_occs]
                + [k for k,v in doc_counts.items() if v < wordfilt_num_docs]
                + [k for k,v in ws_counts.items() if v < wordfilt_num_ws]
                )
            def do_remove(txt):
                return ' '.join(x for x in txt.split(' ') if x not in to_remove)
            removed = [do_remove(x) for x in parts]
            tfidf = self.make_tfidf()
            try:
                tfidf.fit(removed)
            except ValueError as e:
                logger.info(f"Failed to fit, probably no words left {len(removed)}: {e}")
                raise NoWordsException("All words filtered out")

            logger.info(f"Removed {len(to_remove)} words, left with {len(tfidf.vocabulary_)} words")
            tfidfs.append(tfidf)
        self.tfidfs = tfidfs
    
    def make_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        tf = self.settings.get('tf', 'sublinear')
        sublinear_tf = tf == 'sublinear'
        binary = tf == 'binary'

        tfidf = TfidfVectorizer(
            use_idf=False,
            strip_accents='ascii',
            sublinear_tf=sublinear_tf,
            binary=binary,
            norm=self.settings.get('norm', 'l2'),
        )
        return tfidf

    def ds_to_mat(self, ds, ds_raw):
        tfidf_mat = self.group_transform(ds)
        extra_features = self.build_extra_features_mat(ds_raw)

        from scipy import sparse
        mat = sparse.hstack([tfidf_mat, extra_features], format='csr')
        return mat
    
    def build_extra_features_mat(self, ds_raw):
        out = []
        vocab_size = sum(len(x.vocabulary_) for x in self.tfidfs)
        logger.info(f"Total vocab size: {vocab_size}")
        for entry in ds_raw:
            num_samples = entry.num_samples
            if num_samples is None:
                logger.warn(f"Missing samples count for {entry.key}")
                num_samples = 4
            from dtk.num import sigma
            row = [
                num_samples <= 3,
                sigma((num_samples - 150) / 50),
                num_samples > 500,
                1.0 if entry.pub_ref else 0.0,
                entry.key.startswith('PRJ'),
                entry.key.startswith('GS'),
                entry.key.startswith('E-'),
            ]
            # Rescale our one-off features so that they aren't dominated by the thousands of
            # word features.  This is a bit annoying for the solver, but it allows a smaller
            # coefficient to have the same impact for these terms which makes the regularizer happier.
            scale_factor = self.settings.get('extra_feature_scale', vocab_size / 100)
            row = [float(x) * scale_factor for x in row]
            out.append(row)
        return np.array(out, dtype=float)



    def train(self, train_ds):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        # We have more negative than positive samples, so balance out the weights.)
        # We care more about sensitivity (catching all actual positives), so could justify overweighting
        # the positive cases too.

        self.fit(train_ds[0], train_ds[2])
        mat = self.ds_to_mat(train_ds[0], train_ds[2])
        C = self.settings.get('C', 1)
        lr = LogisticRegression(class_weight='balanced', C=C, tol=1e-3, max_iter=10000)
        lr.fit(mat, train_ds[1])
        self.model = lr

    def predict(self, ds):
        mat = self.ds_to_mat(ds[0], ds[2])
        predicted = self.model.predict_proba(mat)
        return predicted[:, 1]


class PreviousModel(GEModel):
    def preprocess(self, ds):
        labels = [x[1] == 1 for x in ds]
        return ds, labels
    
    def train(self, train_ds):
        return
    
    def predict(self, ds):
        samples, labels = ds
        scores, labels = score_existing_dataset(samples)
        return scores
    
    def save(self, fn):
        pass

    @classmethod
    def from_file_data(cls, data):
        pass

def class_from_settings(settings):
    if settings['model'] == 'bagofwords':
        return BagOfWordsModel
    elif settings['model'] == 'previous':
        return PreviousModel
    raise Exception(f"Unknown model {settings['model']}")


def build_model(settings):
    cls = class_from_settings(settings)
    return cls(settings)

def load_model(parms, fn):
    if parms['model'] == 'previous':
        return build_model(parms)

    with open(fn) as f:
        import json
        data = json.loads(f.read())
    cls = class_from_settings(data['settings'])
    model = cls.from_file_data(data)
    return model

def cross_validate(folds, ws_ids, settings, dataset=None):
    dataset = dataset or build_dataset(ws_ids)
    folds = split_folds(dataset, folds)

    model = build_model(settings)

    def run_fold(data):
        fold, fold_data = data
        logger.info(f"Crossvalidating against fold {fold} with {len(fold_data)} samples")
        test = fold_data
        train = []
        for fold2, fold2_data in folds.items():
            if fold2 != fold:
                train.extend(fold2_data)
        
        eval_probs, test_ds_fmt, train_probs = model.run_train_test(train, test)
        return summarize_results(test, eval_probs, test_ds_fmt, train, train_probs)

    from dtk.parallel import pmap
    results = list(pmap(run_fold, folds.items()))
    return results


def train(ws_ids, settings, dataset=None):
    if dataset is None:
        dataset = build_dataset(ws_ids)
    model = build_model(settings)
    model.run_train(dataset)
    return model, dataset

def hyper_xval(folds, dataset, settings):
    from ray import tune
    folds = split_folds(dataset, folds)
    model = build_model(settings)

    results = []
    for fold, fold_data in folds.items():
        logger.info(f"Crossvalidating against fold {fold} with {len(fold_data)} samples")
        test = fold_data
        train = []
        for fold2, fold2_data in folds.items():
            if fold2 != fold:
                train.extend(fold2_data)
        
        try:
            eval_probs, test_ds_fmt, train_probs = model.run_train_test(train, test)
        except NoWordsException:
            logger.info("Parms ended up with no words, ending")
            tune.report(roc_auc=-1)
            return
        results.append(summarize_results(test, eval_probs, test_ds_fmt, train, train_probs))
        rocs = [x['metrics']['roc'] for x in results]
        orig_roc = np.mean(rocs)
        if len(rocs) < len(folds):
            # Be pessimistic about earlier iterations, assume next one is bad.
            rocs.append(0)
        tune.report(
            roc_auc=np.mean(rocs),
            orig_roc=orig_roc,
        )

def hyperopt_train(folds, dataset, settings):
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.suggest.nevergrad import NevergradSearch
    ws_ids = settings['workspaces']

    def opt(config):
        merged_settings = {**settings, **config}
        hyper_xval(folds, dataset, merged_settings)
    
    max_num_ws = max(2, min(5, len(ws_ids) - len(ws_ids)//folds))
    search_space = {
        'C': tune.loguniform(1e-3, 1e3),
        'wordfilt_num_ws': tune.randint(1, max_num_ws),
        'wordfilt_num_docs': tune.randint(1, 10),
        'wordfilt_num_occs': tune.randint(1, 20),
        'extra_feature_scale': tune.lograndint(1, 10000),
        'filter_numbers': tune.choice([True, False]),
        'filter_stopwords': tune.choice([True, False]),
        'tf': tune.choice(['default', 'sublinear', 'binary']),
        'norm': tune.choice([None, 'l1', 'l2']),
    }

    num_trials = settings.get('hyperopt_iters', 100)
    asha_scheduler = ASHAScheduler(max_t=folds)
    import nevergrad as ng
    search_alg = NevergradSearch(ng.optimizers.PortfolioDiscreteOnePlusOne)
    out = tune.run(
        opt,
        config=search_space,
        num_samples=num_trials,
        scheduler=asha_scheduler,
        metric='roc_auc',
        mode='max',
        search_alg=search_alg,
        )
    from runner.process_info import JobInfo
    JobInfo.report_info(f"Best config: {out.best_config}")
    return out.results_df, out.best_config


def run(in_fn, out_fn):
    logger.info("Loading data")
    import isal.igzip as gzip
    with gzip.open(in_fn, 'rb') as f:
        import pickle
        data = pickle.loads(f.read())

    logger.info("Running hyperoptimization")
    parms = data['settings']
    dataset = data['dataset']
    k = parms['kfold']
    from scripts.gesearch_model import hyperopt_train
    if parms['hyperopt_iters'] > 0:
        results_df, best_config = hyperopt_train(k, dataset, parms)
    else:
        import pandas as pd
        results_df = pd.DataFrame([])
        best_config = {}

    logger.info("Training full model")
    parms.update(best_config)

    model, _ = train(None, parms, dataset)

    logger.info("Xval")
    xval = cross_validate(k, None, parms, dataset)

    out = {
        'hyper': results_df.to_dict(),
        'model': model.serialize(),
        'xval': xval,
    }

    logger.info("Writing results")
    import gzip
    import pickle
    with gzip.open(out_fn, 'wb') as f:
        pickle.dump(out, f)



if __name__ == "__main__":
    import argparse
    arguments = argparse.ArgumentParser(description="")
    arguments.add_argument("-i", "--input", help="")
    arguments.add_argument("-o", "--output", help="")

    args = arguments.parse_args()
    from dtk.log_setup import setupLogging
    setupLogging()

    run(args.input, args.output)
