#!/usr/bin/env python

import logging
logger = logging.getLogger(__name__)

def parse(data):
    out = []
    for entry in data['Annotations']['Annotation']:
        cas_list = []
        cid_list = []
        for data_entry in entry['Data']:
            for cas_entry in data_entry['Value']['StringWithMarkup']:
                cas_list.append(cas_entry['String'])

        cid_list = entry.get('LinkedRecords', {}).get('CID', [])

        for cas in cas_list:
            for cid in cid_list:
                out.append((cid, cas))
    
    total_pages = data['Annotations']['TotalPages']
    return out, total_pages

def fetch(page_num):
    import requests
    import random
    import time
    # Server sometimes gets busy and tells us to retry.
    # We retry with backoff to wait out the busy'ness.
    MAX_RETRIES=20
    backoff = 5 + 5 * random.random()
    for retry_num in range(1, MAX_RETRIES + 1):
        req = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/annotations/heading/CAS/JSON?page={page_num}")
        try:
            req.raise_for_status()
            return parse(req.json())
        except requests.exceptions.HTTPError as e:
            if retry_num == MAX_RETRIES:
                logger.error(f"Error fetching page {page_num}")
                raise
            else:
                logger.warning(f"Retrying {retry_num}/{MAX_RETRIES} on error fetching page {page_num}: {e}")
                time.sleep(backoff)
                backoff += 5 + 5 * random.random()
                continue

def run(out_fn, max_pages):
    # Per PUBCHEM docs, this is the max request rate to send.
    REQ_PER_SECOND = 5

    # There are no guidelines around how many concurrent requests we can make, but bumping this too high
    # tends to make it angry and causes a lot of retries.
    MAX_THREADS = REQ_PER_SECOND * 2

    from concurrent.futures import ThreadPoolExecutor
    import time

    results = []
    # Fetch the first one to figure out how many pages there are.
    _, total_pages = fetch(1)
    logger.info(f"Total pages: {total_pages}")
    if max_pages:
        total_pages = max_pages

    from tqdm import trange, tqdm
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as exec:
        for page_num in trange(1, total_pages+1, desc='Pages'):
            results.append(exec.submit(fetch, page_num))

            # Always sleep here to prevent sending more than N per second.
            time.sleep(1 / REQ_PER_SECOND)

            # Extra sleep if we already have MAX requests in flight.
            while len([x for x in results if not x.done()]) > MAX_THREADS:
                time.sleep(1 / REQ_PER_SECOND)

    all_out = set()
    for result in tqdm(results, desc='Processing'): 
        data, _ = result.result()
        all_out.update(data)
    

    from dtk.data import kvpairs_to_dict
    as_lookup = kvpairs_to_dict(all_out)

    import json
    from atomicwrites import atomic_write
    with atomic_write(out_fn, overwrite=True) as out_file:
        out_file.write(json.dumps(as_lookup, indent=2))
    




if __name__ == "__main__":
    from dtk.log_setup import setupLogging
    import argparse
    arguments = argparse.ArgumentParser(description="Pulls CAS data from pubchem.")

    arguments.add_argument('-o', '--output', help="Where to write the output")
    arguments.add_argument('--max-pages', type=int, help="Limit inputs to run quickly for testing changes")
    args = arguments.parse_args()


    setupLogging()
    run(args.output, args.max_pages)



