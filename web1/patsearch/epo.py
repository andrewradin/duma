#!/usr/bin/env python

"""
Code for pulling patents from the European Patent Office.
"""
from __future__ import print_function

def pub_to_epo(pub_id):
    parts = pub_id.split('-')
    return ''.join(parts[:2])

import epo_ops
class EpoClient(object):
    def __init__(self):
        from .patent_search import ApiInfo
        self.client = ApiInfo.epo_client()

    def fetch_patent(self, pub_id):
        out = {}
        out.update(self.fetch_claims(pub_id))
        # Include title, abstract, family, and misc
        out.update(self.fetch_biblio(pub_id))
        return out


    def fetch_claims(self, pub_id):
        epoid = pub_to_epo(pub_id)
        response = self.client.published_data(
            reference_type = 'publication',
            input = epo_ops.models.Epodoc(epoid),
            endpoint = 'claims',
            )
        response.raise_for_status()
        return self.parse_claims(response)

    def parse_claims(self, response):
        from lxml import objectify
        x = objectify.fromstring(response.content)
        x1 = x["{http://www.epo.org/fulltext}fulltext-documents"]
        doc = x1.getchildren()[0]

        out = []
        for claims in doc.claims:
            lang = claims.get('lang', '??').lower()
            claims_text = '\n'.join([subclaim.text
                                     for claim in claims.getchildren()
                                     for subclaim in claim.getchildren()
                                     ])
            out.append({
                'language': lang,
                'text': claims_text
                })
        return {
            'claims_localized': out
        }

    def fetch_biblio(self, pub_id):
        epoid = pub_to_epo(pub_id)
        response = self.client.published_data(
            reference_type = 'publication',
            input = epo_ops.models.Epodoc(epoid),
            endpoint = 'biblio',
            )
        response.raise_for_status()
        return self.parse_biblio(response)

    def parse_biblio(self, response):
        from lxml import objectify
        x = objectify.fromstring(response.content)
        doc = x.getchildren()[0]['exchange-document']
        family_id = doc.get('family-id')
        pub_id = '%s-%s-%s' % (
                doc.get('country'), doc.get('doc-number'), doc.get('kind'))
        bib = doc['bibliographic-data']
        applicant = bib.parties.applicants.applicant['applicant-name'].name.text

        title_localized = []
        for title in bib['invention-title']:
            lang = title.get('lang', '??').lower()
            title_text = title.text
            title_localized.append({
                'language': lang,
                'text': title_text,
                })

        abstract_localized = []
        for abstract in doc.abstract:
            lang = abstract.get('lang', '??').lower()
            abstracts_text = '\n'.join([subabstract.text
                                     for subabstract in abstract.getchildren()
                                     ])
            abstract_localized.append({
                'language': lang,
                'text': abstracts_text
                })

        out = {
            'title_localized': title_localized,
            'abstract_localized': abstract_localized,
            'publication_number': pub_id,
            'family_id': family_id,
            'assignee': applicant
            }
        return out

# https://www.epo.org/searching-for-patents/data/web-services/ops/faq.html#faq-74
EPO_COUNTRY_CODES = set([
    'EP', 'WO', 'AT', 'BE', 'CA', 'CH', 'CY',
    'ES', 'FR', 'GB', 'HR', 'IE', 'LU', 'MC', 'MD',
    'NO', 'PL', 'PT', 'RO',
    ])

def fill_missing_content(ids, storage):
    from patsearch.models import Patent, PatentFamily
    # Grab all patents
    # Find those without family - we'll have no content for those
    # Check content for missing claims for family.
    # For all collected exemplar/missing IDs, do an EPO search
    patents_to_find = []
    patents = Patent.objects.filter(pub_id__in=ids)
    families_added = set()
    for patent in patents:
        # TODO: We could be less picky here and also include non-EPO countries.
        # This filter is just for things they have full-text coverage of.
        # For data that we couldn't find at all, we could use EPO data to
        # lookup the family and see if there's an EPO-country it is filed to.
        if patent.pub_id[:2] not in EPO_COUNTRY_CODES:
            continue

        if not patent.family_id:
            patents_to_find.append(patent)
        else:
            if patent.family_id in families_added:
                continue
            families_added.add(patent.family_id)
            best_content = storage.find_best_content(patent)
            if not best_content or not best_content.has_claims:
                patents_to_find.append(patent)

        
    epo = EpoClient()
    print("Looking up %d patents via EPO" % len(patents_to_find))
    found = 0
        
    from requests.exceptions import HTTPError
    for patent in patents_to_find:
        print("Looking up %s" % patent.pub_id)
        try:
            content = epo.fetch_patent(patent.pub_id)
            family_id = content['family_id']
            if patent.family_id:
                assert patent.family_id == family_id, "EPO had different family id"
            else:
                family, created = PatentFamily.objects.get_or_create(family_id=family_id)
                patent.family = family
                patent.save()
            storage.store_patent_content(family_id, { patent.pub_id: content })
            found += 1
        except HTTPError as e:
            print("Couldn't find data for patent: %s" % e)
        except Exception as e:
            print("Failed to lookup patent: %s" % e)
            import traceback
            traceback.print_exc()
    
    print("Found content for %d new patents" % found)
    return found


