import json
import requests
import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from patsearch.models import Patent, GooglePatentSearch, GooglePatentSearchResult, PatentFamily, PatentContentInfo, PatentSearchResult, BigQueryPatentSearchResult
from django.db import transaction
from path_helper import PathHelper

import logging
logger = logging.getLogger(__name__)

BQ_TABLES = (
    ("All Patents", "patents-public-data.patents.publications"),
    ("Disease Patents", "steam-talent-242816.patent_data.patent_bio_cluster"),
    ("Mini Test Patents", "steam-talent-242816.patent_data.patent_bio_cluster_mini2"),
    ("Micro Test Patents", "steam-talent-242816.patent_data.patent_bio_micro"),
    )

class ApiInfo(object):
    _cse_key = None

    @classmethod
    def cse_key(cls):
        if not cls._cse_key:
            from dtk.s3_cache import S3Bucket, S3File
            bucket = S3Bucket('keys')
            f = S3File(bucket, 'cse.json')
            f.fetch()
            with open(f.path(), 'r') as f:
                cls._cse_key = json.loads(f.read())
        return cls._cse_key

    @staticmethod
    def bq_client():
        from dtk.s3_cache import S3Bucket, S3File
        bucket = S3Bucket('keys')
        f = S3File(bucket, 'bigquery.json')
        f.fetch()
        from google.cloud import bigquery
        path = os.path.join(PathHelper.keys, "bigquery.json")
        client = bigquery.Client.from_service_account_json(path)
        return client

    @staticmethod
    def epo_client():
        from dtk.s3_cache import S3Bucket, S3File
        bucket = S3Bucket('keys')
        f = S3File(bucket, 'epo.json')
        f.fetch()
        with open(f.path(), 'r') as f:
            epo_data = json.loads(f.read())

        import epo_ops
        return epo_ops.Client(
                key=epo_data['key'],
                secret=epo_data['secret'],
                # We can request json, but it is auto-converted from XML, which
                # makes it quite awful.
                accept_type='xml',
                )

def google_patent_search(terms, max_results=10, start=0):
    if max_results == 100:
        out = None
        for i in range(10):
            # Note that 'start' is actually 1-indexed.
            cur = google_patent_search(terms, 10, start=i*10+1)
            if out is None:
                out = cur
            items = cur.get('items', [])
            if len(items) == 0:
                break
            out['items'].extend(items)
        return out
    assert max_results <= 10, "Only support <=10 or 100"
    max_req = str(max_results)



    from dtk.url import google_term_format
    # We need to use patent=False here because we're generating a single query
    # string for the CSE.  patent=True style queries require multiple q= parms.
    # (Haven't actually tested if that works with this API, undocumented)
    search_text = ' '.join([google_term_format(x, patent=False) for x in terms])
    cse = ApiInfo.cse_key()['cse']
    key = ApiInfo.cse_key()['key']
    params = {
            'key': key, # API Key
            'cx': cse, # Custom Search Engine ID
            'num': max_req, # This cannot be more than 10
            'tbm': 'pts', # Only return patents
            'q': search_text
            }
    assert start <= 91, "Cannot request beyond 100th result"
    if start > 0:
        params['start'] = str(start)

    query_str = '&'.join(k+'='+str(v) for k, v in params.items())
    url = 'https://www.googleapis.com/customsearch/v1/siterestrict?' + query_str
    return requests.get(url).json()

def parse_pub_num(item):
    if 'pagemap' not in item:
        return None
    meta = item['pagemap']['metatags'][0]
    pub_num = meta.get('citation_patent_publication_number')
    if not pub_num:
        # This is sometimes missing the trailing code, unfortunately.
        # We use a STARTS_WITH query to compensate.
        pub_num = meta.get('citation_patent_number')
    if not pub_num:
        # This is pretty rare, but there are cases where we had no metadata,
        # but the link was still good, so parse that.
        import re
        m = re.search(r'google.com/patents?/(\w\w)(\d+)(\w*)', item['link'])
        if m:
            pub_num = m.groups()[0] + '-' + m.groups()[1]
            print(("Parsed using title syntax!", item['link'], pub_num))


    if not pub_num:
        print(("No useful information in", item))
        return None
    # The search results here use ':' as a delimiter, but the bigquery
    # dataset uses '-'.  Picking '-' as our canonical form.
    pub_num = pub_num.replace(':', '-')
    return pub_num

def patent_item_to_record(item):
    pub_num = parse_pub_num(item)
    if not pub_num:
        return None
    meta = item['pagemap']['metatags'][0]
    logger.info("Finding patent object for %s", pub_num)
    patent, new = Patent.objects.get_or_create(pub_id=pub_num)
    if not patent.title:
        patent.title = meta.get('dc.title', None)
    # Some patents don't have a description here.
    # TODO: google patent has the content, is it just not in the
    # structured data for the page for some reason?
    patent.abstract_snippet=meta.get('dc.description', ''),

    # TODO: We could include this data, but we'd need to parse it from
    # a string to really be useful.  Would have to see how consistent it is.
    #patent.date=meta['dc.date'],
    patent.href=item['link']
    patent.save()
    return patent


def apply_patent_resolution(patent_result_id, resolution):
    """Applies the resolution to all applicable search results.

    e.g. if the resolution is IRRELEVANT_ALL, then all results that found this
    patent are marked IRRELEVANT_ALL (if they were UNRESOLVED).

    For related search results, will only update results that are UNRESOLVED.
    """
    from .models import PatentSearchResult
    rv = PatentSearchResult.resolution_vals
    resolution = int(resolution)

    search_result = PatentSearchResult.objects.get(id=patent_result_id)
    search_result.resolution = resolution
    search_result.save()

    patent_results = search_result.patent.patentsearchresult_set
    if resolution == rv.IRRELEVANT_ALL:
        # Find all related results that are unresolved.
        for result in patent_results.filter(resolution=rv.UNRESOLVED):
            result.resolution = resolution
            result.save()
    elif resolution == rv.IRRELEVANT_DRUG:
        # Find all related results with this drug that are unresolved.
        wsa = search_result.search.wsa
        if wsa:
            patent_results = patent_results.filter(search__wsa=wsa)
            for result in patent_results.filter(resolution=rv.UNRESOLVED):
                result.resolution = int(resolution)
                result.save()
    elif resolution == rv.IRRELEVANT_DISEASE:
        # Find all related results with this disease that are unresolved.
        ws = search_result.search.patent_search.ws
        patent_results = patent_results.filter(search__patent_search__ws=ws)
        for result in patent_results.filter(resolution=rv.UNRESOLVED):
            result.resolution = int(resolution)
            result.save()

    return search_result


def get_initial_resolution(patent, ws, wsa):
    """Returns an initial resolution value for a patent.

    If it has ever been marked as irrelevant_all, that is the initial.
    If it is irrelevant_drug for this WSA, then that is the initial.
    If it is irrelevant_disease in this WS, then that is the initial.

    Otherwise, it start as unresolved.
    """

    from patsearch.models import PatentSearchResult
    rv = PatentSearchResult.resolution_vals
    results = patent.patentsearchresult_set

    all_resolutions = results.all().values_list('resolution', flat=True)
    if rv.IRRELEVANT_ALL in all_resolutions:
        return rv.IRRELEVANT_ALL

    if wsa:
        drug_results_qs = results.filter(search__wsa=wsa)
        drug_resolutions = drug_results_qs.values_list('resolution', flat=True)
        if rv.IRRELEVANT_DRUG in drug_resolutions:
            return rv.IRRELEVANT_DRUG

    dis_results_qs = results.filter(search__patent_search__ws=ws)
    dis_resolutions = dis_results_qs.values_list('resolution', flat=True)
    if rv.IRRELEVANT_DISEASE in dis_resolutions:
        return rv.IRRELEVANT_DISEASE

    return rv.UNRESOLVED


def re_escape(s):
    import re
    return re.escape(s)

@transaction.atomic
def patent_results_to_records(query, terms, data):
    from dtk.url import google_patent_search_url
    external_link = google_patent_search_url(terms)
    search = GooglePatentSearch.objects.create(
            query=query,
            total_results=data['searchInformation']['totalResults'],
            href=external_link
            )
    search_results = []
    items = data.get('items', [])
    for item in items:
        patent = patent_item_to_record(item)
        if patent:
            search_result = GooglePatentSearchResult.objects.create(
                    patent=patent,
                    search_snippet=item['htmlSnippet'],
                    search=search
                    )
            search_results.append(search_result)
        else:
            #TODO: Handle this?
            # The one case I've seen it links to a sitemap instead of a patent.
            print(("Warning, couldn't handle search result", item['link']))

    return search, search_results

def patent_search(terms, max_results):
    # TODO: If the 100 limit is problematic, might make sense to split up
    # all of the 'OR'd terms into separate searches.
    assert max_results <= 10 or max_results == 100, "Only really support these"
    terms_json = json.dumps([terms, max_results])
    print("Running patent search for %s" % terms)

    query = terms_json
    search = GooglePatentSearch.objects.filter(query=query)
    if search:
        search = search[0]
        search_results = search.googlepatentsearchresult_set.all()
    else:
        result = google_patent_search(terms, max_results)
        if 'error' in result:
            logger.error("Error in search: %s", result)
            raise Exception("Failed to search: %s" % result)
        search, search_results = patent_results_to_records(query, terms, result)

    return search, list(search_results)


class PatentContentStorage(object):
    def __init__(self, storage_dir, job=None, ws=None):
        self.storage_dir = storage_dir
        self.job = job
        self.ws = ws

    def store_patent_content(self, family_id, patent_family_data):
        assert self.job and self.ws, "Must set job and ws to store new patents"
        has_abstract = False
        has_claims = False
        for patent_id, patent_data in patent_family_data.items():
            has_abstract |= bool(patent_data.get('abstract_localized', None))
            has_claims |= bool(patent_data.get('claims_localized', None))
            # This is stored as a date field, which doesn't JSON'ify.
            if 'family_suffix' in patent_data:
                patent_data['family_suffix'] = str(patent_data['family_suffix'])

        family, new = PatentFamily.objects.get_or_create(family_id=family_id)
        pci = PatentContentInfo.objects.create(
                patent_family=family,
                has_abstract=has_abstract,
                has_claims=has_claims,
                job=self.job,
                ws=self.ws
                )

        print(("Saving patent family", family_id))
        import gzip
        fn = os.path.join(self.storage_dir, family_id+'.json.gz')
        with gzip.open(fn, 'wt') as f:
            f.write(json.dumps(patent_family_data))

    def load_patent_content(self, content_info):
        from runner.process_info import JobInfo
        bji = JobInfo.get_bound(content_info.ws, content_info.job)
        bji.fetch_lts_data()

        storage_dir = bji.lts_abs_root

        fn = os.path.join(storage_dir, content_info.patent_family.family_id+'.json.gz')
        import gzip
        with gzip.open(fn, 'r') as f:
            patents = json.loads(f.read())
            best = None
            best_score = [-1, -1]
            for patent_id, patent_data in patents.items():
                score = [len(patent_data.get('claims_localized', [])),
                        len(patent_data.get('abstract_localized', []))]
                if score > best_score:
                    best = patent_data
                    best_score = score
            return best


    def has_content_for(self, patent_id):
        try:
            patent = Patent.objects.get(pub_id=patent_id)
            if not patent.family:
                return False
            return len(patent.family.patentcontentinfo_set.all()) > 0
        except Patent.DoesNotExist:
            return False

    def find_best_content(self, patent):
        best = None
        cnt = 0
        family = patent.family
        if not family:
            return None

        for content in family.patentcontentinfo_set.all():
            cnt += 1
            if not best:
                best = content
            else:
                best = max([best, content], key=lambda x: int(x.has_abstract)*10 + int(x.has_claims)*100)

        return best


class TermsQuery(object):
    def __init__(self, terms, name):
        self.terms = terms
        self.name = name

    def query_text(self):
        def field_contains_any(field, or_terms):
            regex = '|'.join(re_escape(x.lower()) for x in or_terms)
            return 'REGEXP_CONTAINS(LOWER(%s), r"\\b(%s)\\b")' % (field, regex)

        def or_query(fields, or_terms):
            field_queries = [field_contains_any(field, or_terms) for field in fields]
            return '(' + ' OR '.join(field_queries) + ')'

        fields = ['data.abstract.text', 'data.title.text', 'data.claims.text']
        suffix = " as " + self.query_name()
        return or_query(fields, self.terms) + suffix

    def query_name(self):
        return self.name

class BigQueryPatentSearch(object):
    def __init__(self, client, table_name):
        self.client = client
        self.table_name = table_name

    def _tmp_table_name(self):
        import time
        import random
        return 'patent_data.tmp_results_%s_%s' % (
                int(time.time()*1000), random.randint(0, 10000))

    def search(self, disease_terms, drugs):
        disease_query = TermsQuery(disease_terms, 'disease')

        self._terms_to_id = {}
        drug_queries = []
        for i, drug in enumerate(drugs):
            drug_terms, target_terms  = drug['drug_terms'], drug['target_terms']
            if drug_terms:
                drug_id = 'drug%d' % i
                drug_queries.append(TermsQuery(drug_terms, drug_id))
                self._terms_to_id[tuple(drug_terms)] = drug_id
            if target_terms:
                target_id = 'target%d' % i
                drug_queries.append(TermsQuery(target_terms, target_id))
                self._terms_to_id[tuple(target_terms)] = target_id

        drug_queries_text = ",".join([q.query_text() for q in drug_queries])
        has_any_drug_query = ' OR '.join(['terms.'+d.query_name() for d in drug_queries])
        tmp_table_name = self._tmp_table_name()
        self._tmp_results_table = tmp_table_name

        table_name = self.table_name
        query = u"""
        CREATE TABLE `{tmp_table_name}`
        OPTIONS(
            expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
            )
        AS
        WITH
        data AS (
            SELECT publication_number, family_id, title, abstract, claims
            FROM `{table_name}`
            LEFT JOIN UNNEST(title_localized) as title
            LEFT JOIN UNNEST(abstract_localized) as abstract USING (language)
            LEFT JOIN UNNEST(claims_localized) as claims USING (language)
        ),
        terms AS (
            SELECT publication_number, {disease_query}, {drug_queries}
            FROM data
        )
        SELECT publication_number, family_id, terms
        FROM data
        LEFT JOIN terms USING (publication_number)
        WHERE
            terms.disease AND
            ({has_any_drug_query})
        """.format(
                table_name=table_name,
                tmp_table_name=tmp_table_name,
                disease_query=disease_query.query_text(),
                drug_queries=drug_queries_text,
                has_any_drug_query=has_any_drug_query
                )
        print(("Query is ", query))
        query_job = self.client.query(query)
        # This doesn't get used, but evaluating .result() is the easiest way
        # to block until complete.  There are no actual results.
        results = list(query_job.result())
        bytes = query_job.total_bytes_billed or query_job.total_bytes_processed or 0
        print("Search took {} MB".format(bytes / 1e6))



    def results_for(self, terms, limit=500):
        drug_id = self._terms_to_id[tuple(terms)]
        tmp_table_name = self._tmp_results_table

        # The column is just named the drug id, and it's a boolean.
        condition = "terms." + drug_id

        query = u"""
        SELECT publication_number, family_id
        FROM `{tmp_table_name}`
        WHERE
        {condition}
        LIMIT {limit}
        """.format(
                tmp_table_name=tmp_table_name,
                condition=condition,
                limit=limit)
        print(("Query is ", query))

        query_job = self.client.query(query)

        out = []
        for row in query_job.result():
            patent, created = Patent.objects.get_or_create(pub_id=row['publication_number'])

            if not patent.href:
                # Because bigquery's IDs are often wrong, we can't directly
                # construct the URL.  If we search for the ID, though, we will
                # usually get to the right place.
                from dtk.url import google_patent_search_url
                id = row['publication_number'].replace('-', '')
                patent.href = google_patent_search_url([id])
                patent.save()

            result = BigQueryPatentSearchResult.objects.create(patent=patent)
            out.append(result)

        print("Bigquery search found %d results" % len(out))

        bytes = query_job.total_bytes_billed or query_job.total_bytes_processed or 0
        print("Search took {} MB".format(bytes / 1e6))
        return out

# This is tied into the google patent search above, and doesn't have much
# flexibility due to API restrictions.
MAX_SEARCH_RESULTS = 100

class PatentContentJob(object):
    def __init__(self, storage, query):
        self.query = query
        self.storage = storage
        self.client = ApiInfo.bq_client()
        self.disease_terms = query['diseaseNames']
        self.drug_list = query['drugList']
        self.table_name = BQ_TABLES[int(query['tableNumber'])][1]
        print(("Using table", self.table_name))

    def fix_ids(self, ids):
        """
        Some US patent publication IDs appear to be mangled in the database.
        In particular, it looks like someone parsed out the ID, and dropped
        any leading 0's.  e.g. US-20130184299-A1 is stored as US-2013184299-A1.

        However, when there are 2 leading 0's, only 1 seems to be dropped.
        e.g. US-20060094658-A1 is found under US-2006094658-A1.

        Here we reproduce that process to find additional IDs to search for.

        Also, stripping off the last piece of the ID if present.
        """
        new_ids = []
        mapping = {}
        for id in ids:
            new_ids.append(id)
            mapping[id] = id
            parts = id.split('-')

            # Sometimes we're struggling to find the exact match.  Let's just
            # drop off the trailing code, they should all be same family.
            if len(parts) == 3:
                prefix_id = '-'.join(parts[:2])
                new_ids.append(prefix_id)
                mapping[prefix_id] = id

            if id.startswith("US-"):
                parts = id.split('-')
                code = parts[1]
                if len(code) != 11 and len(code) != 7:
                    # Doesn't follow standard pub format, just use as-is.
                    continue
                if len(code) == 11:
                    # Year-prefixed
                    year = code[:4]
                    code = code[4:]
                else:
                    year = ""
                try:
                    code = int(code)
                except ValueError:
                    print("This ID is unusual, can't parse it - %s" % id)
                    continue

                # Create IDs with any number of leading 0's dropped from code.
                for code_len in range(len(str(code)), 7):
                    num_zeros = code_len - len(str(code))
                    new_code = year + ('0' * num_zeros) + str(code)
                    parts[1] = new_code
                    mangled_id = '-'.join(parts)
                    new_ids.append(mangled_id)
                    mapping[mangled_id] = id
        return new_ids, mapping


    def find_patent_families(self, ids):
        orig_ids = set(ids)
        ids, id_mapping = self.fix_ids(ids)

        table_name = self.table_name
        query = """SELECT publication_number, family_id
        FROM `{table_name}`
        WHERE
        """.format(table_name=table_name)
        query += " OR ".join(['publication_number LIKE "{}%"'.format(id) for id in ids])
        print(("Query is ", query))


        query_job = self.client.query(query)

        from collections import defaultdict
        families = defaultdict(set)

        matched_ids = set()
        for row in query_job.result():
            patent_content = dict(list(row.items()))
            id_col = 'publication_number'
            id = patent_content[id_col]
            family_id = patent_content['family_id']
            families[family_id].add(id)
            if id in id_mapping:
                families[family_id].add(id_mapping[id])
                matched_ids.add(id_mapping[id])
            else:
                # This isn't an ID we directly searched for, either it's a
                # family or we matched the start.  Find out which one we
                # matched from.
                found = None
                for query_id in ids:
                    if id.startswith(query_id):
                        found = query_id
                        break
                if found:
                    families[family_id].add(found)
                    matched_ids.add(found)

        missing_ids = orig_ids - matched_ids
        if missing_ids:
            print("Missed some patents:")
            print(missing_ids)

        bytes = query_job.total_bytes_billed or query_job.total_bytes_processed or 0
        print("Family lookup took {} MB".format(bytes / 1e6))

        return families


    def fetch_patent_content(self, ids):
        if len(ids) == 0:
            print("No ids to fetch, probably no new patents found")
            return 0

        family_mapping = self.find_patent_families(ids)
        # The bigquery data has a family ID of -1 for ~0.1% of patents, so
        # we probably don't want to fetch all of those.
        no_family_pubids = family_mapping.pop('-1', [])
        pub_id_query = ""
        if len(no_family_pubids) > 0:
            queries = [('publication_number = "%s"' % x) for x in no_family_pubids]
            pub_id_query = " OR " + " OR ".join(queries)



        family_id_query = '", "'.join(list(family_mapping.keys()))

        # TODO: parameterized queries.
        table_name = self.table_name
        query = """SELECT *
        FROM `{table_name}`
        WHERE family_id IN ("{family_id_query}") {pub_id_query}
        """.format(
                table_name=table_name,
                family_id_query=family_id_query,
                pub_id_query=pub_id_query,
                )

        client = self.client
        print("Making query: ", query)
        query_job = client.query(query)
        print("Getting results")

        from patsearch.models import PatentContentInfo, PatentFamily

        print("Saving patents to disk")
        from collections import defaultdict
        family_patent_content = defaultdict(dict)
        family_to_title = {}

        def get_english(nested_localized):
            if not nested_localized:
                return None
            for entry in nested_localized:
                if entry['language'] == 'en':
                    return entry['text']
            return nested_localized[0]['text']

        for row in query_job.result():
            if not row:
                # NOTE: I have no idea why this happens.  As far as I can tell
                # all the rows are correctly showing up, we just get the
                # occasional extra row with no data in it.  If you try to call
                # .items() on the row, bigquery will throw.
                # It seems to occur consistently in the same spot for the same
                # queries.
                # Annoying to debug because I can only repro with select* queries
                # on the massive/expensive full patent dataset.
                # Will just skip over these for now, I guess.
                logger.warning("Skipping empty row: %s", row)
                continue
            patent_data = dict(list(row.items()))
            id = patent_data['publication_number']
            logger.info("Processing %s", id)
            family_id = patent_data['family_id']
            if family_id == "-1":
                # For anything we found with no family, put it in its own fam.
                family_id = id
            family_mapping[family_id].add(id)
            family_patent_content[family_id][id] = patent_data
            if not family_id in family_to_title:
                title = get_english(patent_data['title_localized'])
                if title:
                    family_to_title[family_id] = title
                else:
                    print("No title available in data for %s, %s (%s)" % (
                          id, family_id, patent_data))

        with transaction.atomic():
            for family_id, patent_content in family_patent_content.items():
                self.storage.store_patent_content(family_id, patent_content)

            bytes = query_job.total_bytes_billed or query_job.total_bytes_processed or 0
            print("Job processed {} MB".format(bytes / 1e6))

            print("Creating families")
            for family_id, family_members in family_mapping.items():
                family, created = PatentFamily.objects.get_or_create(family_id=family_id)
                print("Family %s has %d members" % (family_id, len(family_members)))
                family_title = family_to_title.get(family_id, None)

                for family_member in family_members:
                    patent, created = Patent.objects.get_or_create(pub_id=family_member)
                    patent.family = family
                    if not patent.title and family_title:
                        # Patents that didn't come in through google search may not
                        # have a title yet, give them one.
                        patent.title = family_title

                    patent.save()
            print("Done creating families")
        return len(family_patent_content)

    def collect_patents(self, search_results):
        patents = {}
        for search_result in search_results:
            cur_patents = {result.patent.pub_id:result.patent
                           for result in search_result}
            patents.update(cur_patents)
        return patents

    def analyze_patent(self, patent, disease_terms, drug, drug_terms):
        content_meta = self.storage.find_best_content(patent)
        if not content_meta:
            return {
                'score': PatentSearchResult.SCORE_UNKNOWN,
                'evidence': {}
            }

        content = self.storage.load_patent_content(content_meta)
        from .analysis import analyze_patent
        return analyze_patent(content, disease_terms, drug_terms)

    def steps(self):
        steps = ["BigQuery search"]
        steps += ['search for ' + drug['name'] for drug in self.drug_list]
        steps += [
                'fetch patent content (BigQuery)',
                'fetch patent content (EPO)',
                'analyze patents'
                ]
        return steps

    def find_and_rank_patents(self, user, ws, job, p_wr):
        search_results, ids, assoc_drugs = self._search_for_patents(p_wr)
        self._download_patents(ids, p_wr)
        self._analyze_patents(search_results, assoc_drugs, user, ws, job, p_wr)

    def _search_for_patents(self, p_wr):
        drugs = self.drug_list
        disease = self.disease_terms
        search_results = []
        assoc_drugs = []

        bq_search = BigQueryPatentSearch(self.client, self.table_name)
        bq_search.search(disease, drugs)
        p_wr.put("BigQuery search", "complete")

        for drug in drugs:
            to_search = [
                    (drug['drug_terms'], ''),
                    (drug['target_terms'], ' (targets)')
                    ]
            log_details = []
            for terms, name_suffix in to_search:
                if len(terms) > 0:
                    search, results = patent_search([disease, terms], MAX_SEARCH_RESULTS)
                    log_details += ['search%s: %d results' % (name_suffix, len(results))]

                    bq_results = bq_search.results_for(terms)
                    results.extend(bq_results)
                    log_details += ['BQ search%s: %d results' % (name_suffix, len(bq_results))]


                    search_results.append(results)
                    assoc_drugs.append((drug, drug['name'] + name_suffix, terms))


            details = ', '.join(log_details)
            p_wr.put("search for " + drug['name'], "complete (%s)" % details)

        all_patents = self.collect_patents(search_results)

        ids = list(all_patents.keys())
        return search_results, ids, assoc_drugs

    def _download_patents(self, ids, p_wr):
        print("Got %d IDs" % len(ids))

        missing_ids = [id for id in ids if not self.storage.has_content_for(id)]
        num_found = self.fetch_patent_content(missing_ids)
        p_wr.put("fetch patent content (BigQuery)", "complete (%d new)" % num_found)

        from .epo import fill_missing_content
        num_found = fill_missing_content(ids, self.storage)
        p_wr.put("fetch patent content (EPO)", "complete (%d new)" % num_found)


    def _analyze_patents(self, search_results, assoc_drugs, user, ws, job, p_wr):
        drugs = self.drug_list
        disease = self.disease_terms

        from patsearch.models import PatentSearch, DrugDiseasePatentSearch, PatentSearchResult
        import json
        query = {
                'drugList': drugs,
                'diseaseNames': disease
        }
        queryStr = json.dumps(query)
        search_model = PatentSearch.objects.create(user=user, ws=ws, job=job, query=queryStr)

        for drug_data, drug_results in zip(assoc_drugs, search_results):
            drug, drug_name, search_terms = drug_data
            query = json.dumps({
                'drug': drug,
                'disease': disease,
                'terms_used': search_terms,
                })
            dd_search = DrugDiseasePatentSearch.objects.create(
                    patent_search=search_model,
                    query=query,
                    drug_name=drug_name,
                    wsa_id=drug.get('wsa', None)
                    )
            dupes = 0
            missing = 0
            seen_families = set()
            seen_patents = set()
            for drug_result in drug_results:
                # We might have modified the patents since we got these results
                # from the search, let's reload.
                drug_result.refresh_from_db()
                drug_result.patent.refresh_from_db()

                family = drug_result.patent.family
                if family is None:
                    print("Patent %s has no family set" % drug_result.patent.pub_id)
                    missing += 1
                    if drug_result.patent.pub_id in seen_patents:
                        print(("Skipping existing known patent", drug_result.patent.pub_id))
                        continue
                    seen_patents.add(drug_result.patent.pub_id)
                if family and family.family_id in seen_families:
                    dupes += 1
                    continue
                if family:
                    print("Patent %s has family %s" % (
                        drug_result.patent.pub_id, family.family_id))
                    seen_families.add(family.family_id)
                analysis = self.analyze_patent(drug_result.patent, disease, drug, search_terms)
                import json

                initial_resolution = get_initial_resolution(
                        drug_result.patent,
                        ws=ws,
                        wsa=drug.get('wsa', None),
                        )

                PatentSearchResult.objects.get_or_create(
                        search=dd_search,
                        patent=drug_result.patent,
                        # We could be associated with either a google or
                        # bigquery search.  We don't need the association now,
                        # so just don't set it for now.
                        #google_patent_search_result=drug_result,
                        score=analysis['score'],
                        evidence=json.dumps(analysis['evidence']),
                        resolution=initial_resolution,
                        )
            print("Search for %s had %d results (%d missing, %d dupes)" % (drug, len(drug_results), missing, dupes))
        p_wr.put("analyze patents", "complete")



def related_searches_for_drug(wsa_id):
    from patsearch.models import DrugDiseasePatentSearch
    return DrugDiseasePatentSearch.objects.filter(wsa=wsa_id)
