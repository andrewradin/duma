#!/usr/bin/env python
import os


from tqdm import tqdm
import logging
logger = logging.getLogger("pubchem")

# An example of the eutils output from a pubchem fetch:
# https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pccompound&id=45588096&rettype=docsum
#
# Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25500/

# XXX Ideally, we'd be able to output a biochem True/False here, and maybe
# XXX even sequence data. But although the data appears on the pubchem
# XXX molecule pages, it doesn't seem to be accessible through eutils.
# XXX
# XXX The information we're interested in is the Biologic Description
# XXX section in, e.g.:
# XXX https://pubchem.ncbi.nlm.nih.gov/compound/45588096
# XXX
# XXX Information on the web page indicates this is generated from SMILES
# XXX data using https://www.nextmovesoftware.com/sugarnsplice.html so
# XXX it may not be present in the database at all, or may have special
# XXX licensing issues.
# XXX
# XXX An alternative to eutils described here:
# XXX https://pubchemdocs.ncbi.nlm.nih.gov/pug-view
# XXX supports sequence data, but only supports single-molecule retrieval.
# XXX https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/45588096/JSON

def xget(fromel, tag):
    els = fromel.findall(f'.//{tag}')
    from lxml import etree
    assert len(els) == 1, f'Found multiple {tag} in {etree.tostring(fromel, pretty_print=True).decode("utf8")}'
    return els[0]

# Search for these at https://pubchem.ncbi.nlm.nih.gov/source/#sort=Live-Substance-Count&type=Live-Substances&status=Active&category=Chemical-Vendors
PROVIDERS = [
        ("MedChemexpress MCE", 'medchemexpress'),
        ("Selleckchem", 'selleckchem'),
        ("Tocris Bioscience", 'tocris'),
        ("Sigma-Aldrich", 'sigma'),
        ("VWR, Part of Avantor", 'vwr'),
        ("Fisher Chemical", 'fisher'),
        ("DrugBank", 'drugbank'),
        ]

MAX_RESULTS = 1000000

from dtk.entrez_utils import EClientWrapper
ec = EClientWrapper()


import pickle

class ESummaryWrapper:
    """Handles per-ID caching and chunking/paging"""
    def __init__(self, db, cache_path):
        self.db = db
        self._cache_path = cache_path
        if os.path.exists(cache_path):
            import gzip
            self._data = pickle.load(gzip.open(cache_path, 'rb'))
        else:
            self._data = {}


    def run(self, ids):
        # Canonicalize as strs.
        ids = [str(x) for x in ids]
        cached_results = self._check_cache(ids)
        logger.info(f"Found {len(cached_results)} in cache")
        from lxml import etree
        yield from (etree.fromstring(x) for x in cached_results.values())
        uncached_ids = set(ids) - set(cached_results.keys())

        if uncached_ids:
            yield from self._run_remote(uncached_ids)

            # Only need to save cache if we pulled new ones via run_remote.
            self._save_cache()


    def _check_cache(self, ids):
        return {id:self._data[id] for id in ids if id in self._data}

    def _save_cache(self):
        import gzip
        with gzip.open(self._cache_path, 'wb') as f:
            pickle.dump(self._data, f)

    def _run_remote(self, ids):
        logger.info(f"Fetching {len(ids)} remote results")
        from dtk.parallel import chunker
        # esummary appears to silently truncate if you ask for more than 10000, so we have to do some paging here.
        for id_chunk in chunker(list(ids), chunk_size=10000, progress=False):
            ids_str = ",".join(str(x) for x in id_chunk)
            subdata = ec.esummary(dict(db=self.db, id=ids_str, version=2.0, retmode='xml'), max_results=10000)

            from lxml import etree
            root = etree.fromstring(subdata)

            results = root.findall('.//DocumentSummary')

            for entry in results:
                uid = entry.get('uid')
                self._data[uid] = etree.tostring(entry)
                yield entry



def fetch_provider(esum, provider, short_name, limit):
    limit = limit or MAX_RESULTS
    all_cids = set()
    logger.info(f"Pulling IDs for {provider}")
    query = f'"{provider}"[SourceName] AND hasnohold[filt]'
    res = ec.esearch('pcsubstance', query, max_results=limit)
    assert res.count < MAX_RESULTS, "Hit max results"
    logger.info(f"Found {res.count} results")
    ids = res.ids


    logger.info(f"Pulling substance summary for {provider}")
    prov_data = []

    results = esum.run(ids)

    missing_url = 0
    for data in tqdm(results, total=len(ids)):
        get = lambda val: xget(data, val)
        # There are also synonyms here, but we'll just fetch all the
        # synonyms at once via the compound.
        cids = [x.text for x in get('CompoundIDList')]
        sourceid = get('SourceID').text
        sourceurl = get('SBUrl').text
        if not sourceurl:
            missing_url += 1
            continue

        # CIDs is a list because in the case of mixtures, each component
        # as well as the mixture itself will have a separate compound id.
        # e.g. if you have "<XYZ> Acetate", the compound list will be:
        #  - <XYZ> Acetate
        #  - <XYZ>
        #  - Acetic acid
        #
        # This is probably less-than-useful for Acetic acid; but we do want
        # to both pull in and associate the provider with both <XYZ> and
        # <XYZ> Acetate.
        entry = {
            'cids': cids,
            }

        if short_name == 'drugbank':
            entry['linked_drugbank_id'] = sourceid
        else:
            import urllib
            # Really just need to encode any literal '|' characters.
            sourceurl = urllib.parse.quote(sourceurl, safe=':/')
            entry['commercial_link'] = '|'.join([short_name, sourceid, sourceurl])

        prov_data.append(entry)

        all_cids.update(cids)
    print(f"Missing url for {missing_url}; outputting  {len(prov_data)}, for {len(all_cids)} cids")
    return prov_data, all_cids

def fetch_compounds(cids):
    logger.info(f"Fetching {len(cids)} compounds")
    from dtk.parallel import chunker
    out = []

    # Don't want to overwhelm DB with overly long synonyms.
    MAX_SYN_LEN = 512

    esum = ESummaryWrapper('pccompound', './pccompound.cache.gz')
    from dtk.cas import cas_error
    for data in tqdm(esum.run(cids), total=len(cids)):
        get = lambda val: xget(data, val)
        cid = get("CID").text
        cdata = {
            'cid': cid,
            'synonym': [x.text for x in get('SynonymList') if len(x.text) < MAX_SYN_LEN],
            'smiles_code': get("IsomericSmiles").text,
            'inchi': get("InChI").text,
            'inchi_key': get("InChIKey").text,
            'commercial_link': [],
            'full_mwt': get("MolecularWeight").text,
            'mol_formula': get("MolecularFormula").text,
            }

        # Short synonyms are ones that are short enough that we could use them
        # in the canonical table.  We could make this as high as 256, but
        # realistically that is too long to be user-friendly.
        short_syns = [x for x in cdata['synonym'] if len(x) < 128]
        if short_syns:
            canonical = short_syns[0]
        else:
            canonical = 'PUBCHEM' + cid
        cdata['canonical'] = canonical
        # if synonyms contains a single valid CAS number, label it
        cas_list = [x for x in cdata['synonym'] if not cas_error(x)]
        if len(cas_list) == 1:
            cdata['cas'] = [cas_list[0]]

        out.append(cdata)
    return out


def merge_provider_data(cdata, prov_datas, cas_map):
    logger.info("Merging data")
    cdata_map = {int(x['cid']): x for x in cdata}

    for prov_data in prov_datas:
        for entry in prov_data:
            for cid in entry['cids']:
                centry = cdata_map[int(cid)]
                if 'commercial_link' in entry:
                    centry['commercial_link'].append(entry['commercial_link'])
                if 'linked_drugbank_id' in entry:
                    centry['linked_drugbank_id'] = entry['linked_drugbank_id']
    

    added_cas = 0
    for cid, cas_vals in cas_map.items():
        cid = int(cid)
        if cid not in cdata_map:
            continue
        if 'cas' not in cdata_map[cid]:
            cdata_map[cid]['cas'] = []

        cas_before = set(cdata_map[cid]['cas'])

        from dtk.cas import cas_error
        cas_vals = [x for x in cas_vals if not cas_error(x)]
        cdata_map[cid]['cas'].extend(cas_vals)
        # Filter any dupes
        cdata_map[cid]['cas'] = list(set(cdata_map[cid]['cas']))
        cas_after = set(cdata_map[cid]['cas'])
        added_cas += len(cas_after) - len(cas_before)

    logger.info(f"Added {added_cas} CAS records")
    assert added_cas > 0



def filter_data(cdata):
    logger.info("Filtering data")
    for entry in tqdm(cdata):
        seen_url = set()
        from collections import defaultdict
        provider_count = defaultdict(int)
        keep_links = []
        for link in entry['commercial_link']:
            provider, prov_id, url = link.split('|')
            if url in seen_url:
                # Some providers have separate records for 10mg, 50mg, etc.
                # but usually all go to the same URL, so dedupe that way.
                continue
            seen_url.add(url)

            provider_count[provider] += 1
            if provider_count[provider] > 3:
                # We often see this for things like acetic acid, with hundreds
                # of different mixtures that include it.  Not super useful, so
                # just filter.
                continue

            keep_links.append(link)


        entry['commercial_link'] = keep_links

def write(cdata, out_fn):
    out = [['pubchem_id', 'attribute', 'value']]
    for entry in sorted(cdata, key=lambda x: int(x['cid'])):
        cid = entry.pop('cid')
        eid = 'PUBCHEM' + cid

        entry['pubchem_cid'] = cid
        for attr, attrval in sorted(entry.items(), key=lambda x: (x[0] != 'canonical', x)):
            if isinstance(attrval, (tuple, list)):
                out.extend([(eid, attr, x) for x in sorted(attrval)])
            else:
                out.append((eid, attr, attrval))
    from atomicwrites import atomic_write
    with atomic_write(out_fn, overwrite=True) as f:
        for entry in out:
            f.write('\t'.join(str(x) for x in entry) + '\n')

def run(out_fn, limit, cas_map_fn):
    import json
    with open(cas_map_fn) as f:
        cas_map = json.loads(f.read())

    all_cids = set()

    prov_datas = []

    esum = ESummaryWrapper('pcsubstance', './pcsubstance.cache.gz')
    for provider, prov_short in PROVIDERS:
        prov_data, cids = fetch_provider(esum, provider, prov_short, limit)
        all_cids.update(cids)
        prov_datas.append(prov_data)

    cdata = fetch_compounds(all_cids)

    merge_provider_data(cdata, prov_datas, cas_map)
    logger.info(f"Ended with {len(cdata)} records")

    filter_data(cdata)

    write(cdata, out_fn)



if __name__ == "__main__":
    from dtk.log_setup import setupLogging
    import argparse
    arguments = argparse.ArgumentParser(description="Pulls data from pubchem.")

    arguments.add_argument('-o', '--output', help="Where to write the output")
    arguments.add_argument('--cas-map', help="CAS data to include")
    arguments.add_argument('-l', '--limit', type=int, help="Limit inputs to run quickly for testing changes")
    args = arguments.parse_args()


    setupLogging()
    run(args.output, args.limit, args.cas_map)



