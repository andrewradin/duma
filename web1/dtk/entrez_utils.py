import logging
import os
logger = logging.getLogger(__name__)



class SraSearch:
    def get_projects(self, term, species):
        from browse.models import AeSearch
        self.max_search = 100000
        filters = ['[All Fields]',
                   ]
        if species != AeSearch.species_vals.any:
            latin = AeSearch.latin_of_species(species)
            filters.append(f'AND {latin}[Organism]')
        query = term + ' '.join(filters)
        ec=EClientWrapper()
        r=ec.esearch('bioproject',query, max_results=self.max_search)
        # save actual count so client can check it
        self.full_count = r.count
        import lxml.etree as etree
        self.results = {}
        uids=','.join([str(x) for x in r.ids])
        docs = {}
        try:
            xml=ec.esummary(dict(db='bioproject',id=uids), max_results=self.max_search)
        except EutilsOKError:
            # this comes from having no hits for that search term
            return docs
        root = etree.fromstring(xml)
        def get(fromel, tag):
            els = fromel.findall(f'.//{tag}')
            assert len(els) == 1, f'Found multiple {tag} in {etree.tostring(fromel, pretty_print=True).decode("utf8")}'
            return els[0]

        for doc in get(root, 'DocumentSummarySet'):
            if doc.tag == 'DocumentSummary':
                d = {
                        'title': get(doc, 'Project_Title').text,
                        'summary': get(doc, 'Project_Description').text,
                        'id': get(doc, 'Project_Acc').text,
                        'data_type': get(doc, 'Project_Data_Type').text,
                }
                docs[d['id']] = d
        return docs

    def __init__(self, term, species):
        # We search bioproject with our terms, as that has the descriptions
        # and titles that could match.
        #
        # Rather searching using eutils for the rest of it, we then typically
        # head off to bigquery for the remaining data.
        # The eutils APIs aren't great for this, aren't properly supported
        # with our library/project combo, and they have corner-case issues
        # with unusual projects (e.g. any PRJEB projects) because the sample
        # uids don't line up with the accession ids.
        self.results = self.get_projects(term, species)


class GeoSearch(object):
    def __init__(self,**kwargs):
        from browse.models import AeSearch
        self.term = kwargs.get('term', None)
        self.species = kwargs.get('species', AeSearch.species_vals.human)
        self.max_search = 3000
        filters = ['[All Fields]',
                   'AND (gse[Filter]',
                     'AND (Expression profiling by array[Filter]',
                       'OR Expression profiling by high throughput sequencing[Filter]))',
                   ]
        if self.species != AeSearch.species_vals.any:
            latin = AeSearch.latin_of_species(self.species)
            filters.append(f'AND {latin}[Organism]')
        query = self.term + ' '.join(filters)
        ec=EClientWrapper()
        r=ec.esearch('gds',query, max_results=self.max_search)
        # save actual count so client can check it
        self.full_count = r.count

        import xml.etree.ElementTree as ET
        self.results = {}

        from dtk.parallel import chunker
        logger.info(f"Search return {len(r.ids)} ids")
        for ids in chunker(r.ids, chunk_size=200):
            logger.info(f"Grabbing summaries for {len(ids)} ids")
            uids=','.join([str(x) for x in ids])
            try:
                xml=ec.esummary(dict(db='gds',id=uids), max_results=self.max_search)
            except EutilsOKError:
                # this comes from having no hits for that search term
                return
            root = ET.fromstring(xml)
            for doc in root:
                assert doc.tag == 'DocSum'
                d={}
                for child in doc:
                    if child.tag == 'Id':
                        d['id'] = int(child.text)
                        continue
                    assert child.tag == 'Item'
                    if child.attrib['Name'] == 'Accession':
                        self.results[child.text] = d
                    elif child.attrib['Name'] == 'title':
                        d['title'] = child.text
                    elif child.attrib['Name'] == 'summary':
                        d['summary'] = child.text
                    elif child.attrib['Name'] == 'n_samples':
                        d['sample_n'] = float(child.text)
                    elif child.attrib['Name'] == 'gdsType':
                        d['experiment_type'] = child.text
                    elif child.attrib['Name'] == 'PubMedIds':
                        for subchild in child:
                            d['pmid'] = subchild.text # there could be multiple, I'll just take the first
                            break

class PubMedSearch:
    def __init__(self, use_cache=True):
        self.client = EClientWrapper(use_cache=use_cache)
        self.db = 'pubmed'
    def count_frequency(self, search_terms, bool_search="AND"):
        q = self._build_search(search_terms, bool_search)
        r = self.esearch(q)
        return r.count
    def size(self):
        r = self.client.einfo(self.db)
        return int(r.count)
    def get_authors(self, search_terms, bool_search="AND"):
        q = self._build_search(search_terms, bool_search)
        r = self.esearch(q)
        if not r:
            return []
        f = self.efetch(r.ids)
        if not f:
            return []
        return [a for p in f for a in p.authors]
    def _build_search(self, search_terms, bool_search):
        return str(" "+bool_search+" ").join(search_terms)
    def efetch(self, ids):
        try:
            return self.client.efetch(self.db,ids)
        except EutilsOKError:
            # this comes from having no hits for that search term
            return
    def esearch(self, q):
        import eutils
        try:
            return self.client.esearch(self.db,q)
        except (UnicodeEncodeError, eutils.EutilsNCBIError):
            # If the query string fails to encode, transliterate the unicode
            # string so each unicode character is replaced by an ascii
            # equivalent (often a character name).
            tmp_q = transliterate(q)
            logger.warning(f'Using transliterated query {tmp_q}')
            # It's not clear we should be catching the Eutils error here,
            # but I don't have any test cases and don't want to break things.
            return self.client.esearch(self.db,tmp_q)

def transliterate(s):
    import unidecode
    return unidecode.unidecode(str(s))

# This is the old pre-eutils version. It's left here for reference in case
# we need to access something at a low level that eutils doesn't handle.
class OldPubMedSearch:
    def __init__(self):
        self.root_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed'
        self.last_req_time = None
        self.min_time_between_requests = 0.5

    def count_frequency(self, search_terms, bool_search="AND"):
        data = self.make_request(term=str(" "+bool_search+" ").join(search_terms), retmode='json', retmax=0)
        return data["esearchresult"]["count"]

    def make_request(self, **kwargs):
        import requests, json, time
        if self.last_req_time:
            while (self.min_time_between_requests > (time.time() - self.last_req_time)):
                time.sleep(0.1)
        resp = requests.get(url=self.root_url, params=kwargs)
        self.last_req_time = time.time()
        try:
            data = json.loads(resp.text)
        except ValueError:
            print('WARNING: request did not return a JSON object from the following search: ', kwargs['term'])
            data = {"esearchresult": {'count':0}}
        return data

def prep_PubMedQueries(s, type):
    if type == 'author':
        return s+'[Author]'
    elif type == 'mesh':
        return s+'[MeSH Major Topic]'
    else:
        print('WARNING: unsupported PubMedQuery type: ', type)


MAX_RETRIES = 10
# Our API limit is 3 per second.
# If we're having to retry, let's back off to 2 per second.
RETRY_BACKOFF_SECS = 0.5

class EutilsOKError(Exception):
    """Eutils threw an exception, but the http response was 200 OK."""
    pass

def retry(func):
    """Function decorator, will auto-retry eutils errors."""
    def func_with_retries(*args, **kwargs):
        import eutils
        import requests
        wrapper_self = args[0]
        from path_helper import PathHelper
        from dtk.lock import FLock
        lock_path = PathHelper.timestamps + 'eutils_lock'
        # Using a file lock here to help with throttling.  If we have multiple
        # processes hitting this at the same time (e.g. multiple run_lbn's)
        # the built-in throttling won't work, but this file lock will make
        # sure no one else starts a new request until we've finished or given
        # up on the previous one.
        with FLock(lock_path):
            # Set max results if provided, otherwise set it back to the
            # eutils default of 250.
            wrapper_self._set_max_results(kwargs.pop('max_results', 250))

            # Use the eutils cache if desired
            if wrapper_self._use_cache:
                wrapper_self._qs._cache = wrapper_self._cache
            else:
                wrapper_self._qs._cache = None

            for i in range(MAX_RETRIES):
                try:
                    out = func(*args, **kwargs)
                    if i > 0:
                        logger.warning("Request succeeded after %d tries", i+1)
                    return out
                except (eutils.EutilsNCBIError, eutils.EutilsRequestError, requests.exceptions.ReadTimeout, requests.exceptions.ChunkedEncodingError) as e:
                    # This is ugly, but the interface here is quite bad.
                    # The only mechanism we have to distinguish error types
                    # is string parsing the exception.
                    # To make it worse, they throw exceptions for non-failure
                    # cases, like no results.
                    # In general, anything that responds OK (200) isn't going
                    # to give anything different on retry, so we will treat
                    # those specially.
                    if 'OK (200)' in str(e):
                        logger.warning("Not retrying because it said OK, but failed: %s", e)
                        raise EutilsOKError(e)

                    # The primary reason we get here is going to be us hitting
                    # rate-limiting issues.
                    # https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/
                    # Version 0.3 (currently on platform) throws this as
                    # eutils.EutilsNCBIError like every other error.
                    # Newer versions of eutils distinguish this as
                    # eutils.EutilsRequestError (which we also catch above).

                    if i + 1 == MAX_RETRIES:
                        logger.warning("Giving up on request, failed too many times: %s", (args, kwargs))
                        raise
                    logger.warning(f"eutils call failed, going to retry in bit ({e})")
                    import time
                    time.sleep(RETRY_BACKOFF_SECS)
    return func_with_retries

class EClientWrapper:
    """Wrapper around eutils.client.Client

    This wraps everything with retries, and also uses a single client
    under the hood which allows it to properly throttle, which reduces
    errors.
    The retry-wrapper also uses a file lock to ensure we have some level of
    cross-process throttling if we start hitting errors.

    This also has some special logic to attempt to distinguish "real"
    retryable errors from things that aren't really errors (e.g. no results).

    NOTE: The version of eutils on platform (0.3.1) changes its throttling
    behavior based on time-of-day.  Seems the service previously allowed
    unthrottled access during off-hours, but that is no longer true.
    This class should still work fine in that case, as we fall back to our
    own throttling when we hit retries.  But it might be nice to upgrade it
    at some point. (v0.5 will be the last version to support py2.7).
    """

    @classmethod
    def classinit(cls):
        # This exists as a separate method rather than inline in the class
        # so that we can easily reset it during tests.
        import eutils
        from path_helper import PathHelper
        cls.cache_path = os.path.join(PathHelper.storage, 'eutils.cache')
        cls.ec=eutils.client.Client(cache=cls.cache_path)
        cls._qs = cls.ec._qs
        cls._cache = cls._qs._cache

    def __init__(self, use_cache=False):
        self._use_cache = use_cache

    @retry
    def esummary(self, *args, **kwargs):
        return self.ec._qs.esummary(*args, **kwargs)

    @retry
    def efetch(self, *args, **kwargs):
        return self.ec.efetch(*args, **kwargs)

    @retry
    def einfo(self, *args, **kwargs):
        return self.ec.einfo(*args, **kwargs)

    @retry
    def esearch(self, *args, **kwargs):
        return self.ec.esearch(*args, **kwargs)

    @classmethod
    def clear_cache(cls):
        # Expires everything.
        cls._cache.expire(-1)

    @classmethod
    def expire_old_cache_data(cls):
        """We set a default expiry here, we don't expect our queries to get out of date quickly, but
        super old data is probably not ideal.
        """
        # By default expire anything older than 90 days
        EXPIRE_OLD_DAYS=90
        EXPIRE_OLD_SECONDS=60*60*24*EXPIRE_OLD_DAYS
        cls._cache.expire(EXPIRE_OLD_SECONDS)

    def _set_max_results(self, max_results):
        """Sets the maximum number of results to return for any query."""
        d = dict(self._qs.default_args)
        d['retmax'] = max_results
        self._qs.default_args = d
EClientWrapper.classinit()
