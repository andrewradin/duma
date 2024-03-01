#!/usr/bin/env python

"""
NOTE:

This is an attempt at pulling MedChemExpress from pubchem instead of from their website, which now requires
human intervention.

There are some advantages and some disadvantages to this approach.
PROS:
- We now get SMILES codes for everything (though I suspect if we pulled the SDF instead of XLS we could get from website)
- We get ~10k extra molecules (though not clear they're all useful, from some weird collections)
- Works automatically

CONS:
- We lose almost all CAS data (not exported to pubchem)
- We lose almost all synonyms (not exported to pubchem)
- Doesn't always seem up-to-date (last mce-pubchem update was late 2020 - https://pubchem.ncbi.nlm.nih.gov/source/959)


Probably for best-of-both-worlds we'd pull both and merge them.  There's a bit of an attempt to do that in the code
below (though only for cas #'s, ideally we'd do it for all data).


Anyhow, for now this is not integrated into the Makefile, to avoid losing existing data.


"""

from path_helper import PathHelper
import sys
import os
sys.path.append(os.path.join(PathHelper.databases, 'pubchem'))
import logging
logger = logging.getLogger(__name__)
from make_pubchem import ESummaryWrapper, MAX_RESULTS, ec, xget
from tqdm import tqdm


def fetch_provider(esum, provider, limit, id_to_cas):
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
        cids = [x.text for x in get('CompoundIDList')]
        sid = get('SID').text
        sourceid = get('SourceID').text
        sourceurl = get('SBUrl').text

        if not sourceid.startswith('HY-'):
            logger.info(f"Skipping non-HY drug {sourceid}")
            continue

        synonyms = []
        cas_list = id_to_cas.get(sourceid, set())
        for syn_el in get('SynonymList'):
            # Sometimes they semi-colon separate synonyms within a single entry
            for syn in syn_el.text.split(';'):
                syn = syn.strip()
                # CAS #'s are explicitly prefixed... sometimes misspelled...
                if syn.lower().startswith('cas:') or syn.lower().startswith('csa:'):
                    cas_list.add(syn[4:])
                elif syn == sourceid:
                    # Don't need primary ID as a synonym.
                    pass
                else:
                    synonyms.append(syn)
        
        if synonyms:
            canonical = synonyms[0]
            synonyms = synonyms[1:]
        else:
            canonical = sourceid
        entry = {
            'cids': cids,
            'sid': sid, # substance id
            'sourceid': sourceid,
            'sourceurl': sourceurl,
            'canonical': canonical,
            'synonym': synonyms,
            'cas': cas_list,
        }

        prov_data.append(entry)
        all_cids.update(cids)
    print(f"Missing url for {missing_url}, skipping those")
    return prov_data, all_cids


def fetch_smiles(sids):
    sid_to_smiles = {}

    from dtk.parallel import chunker
    for id_chunk in chunker(list(sids), chunk_size=200, progress=True):
        id_str = ','.join(id_chunk)
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sid/{id_str}/SDF'
        import requests
        r = requests.get(url, stream=True)
        r.raise_for_status()
        r.raw.decode_content=True
        from rdkit import Chem
        for mol in Chem.ForwardSDMolSupplier(r.raw, sanitize=False):
            if not mol:
                continue
            sm = Chem.MolToSmiles(mol)
            if sm == '*' or not sm:
                continue

            sid = mol.GetProp('PUBCHEM_SUBSTANCE_ID')
            sid_to_smiles[sid] = sm

    return sid_to_smiles

        

def run(output, limit, prev_attrs_fn):
    id_to_cas = load_prev_cas(prev_attrs_fn)

    esum = ESummaryWrapper('pcsubstance', './pcsubstance.cache.gz')
    prov_data, cids = fetch_provider(esum, provider='MedChemexpress MCE', limit=limit, id_to_cas=id_to_cas)
    print("Got ", len(prov_data))

    sid_to_smiles = fetch_smiles([x['sid'] for x in prov_data])

    for entry in prov_data:
        smiles = sid_to_smiles.get(entry['sid'], None)
        if smiles is not None:
            entry['smiles_code'] = smiles

    write_create_file(output, prov_data)

def write_create_file(out_fn, data):
    out = [['med_chem_express_id', 'attribute', 'value']]
    for entry in sorted(data, key=lambda x: x['sourceid']):
        cids = entry.pop('cids')

        eid = entry.pop('sourceid')
        entry.pop('sourceurl')
        entry.pop('sid') # Don't need the substance id anymore

        if cids:
            entry['pubchem_cid'] = cids[0]

        for attr, attrval in sorted(entry.items(), key=lambda x: (x[0] != 'canonical', x)):
            if isinstance(attrval, (tuple, list, set)):
                out.extend([(eid, attr, x) for x in sorted(attrval)])
            else:
                out.append((eid, attr, attrval))
    from atomicwrites import atomic_write
    with atomic_write(out_fn, overwrite=True) as f:
        for entry in out:
            f.write('\t'.join(str(x) for x in entry) + '\n')

def load_prev_cas(fn):
    from collections import defaultdict
    cas_out = defaultdict(set)
    from dtk.files import get_file_records
    for id, attr, val in get_file_records(fn, keep_header=False, progress=True):
        if attr == 'cas':
            cas_out[id].add(val)
    return cas_out


if __name__ == "__main__":
    from dtk.log_setup import setupLogging
    import argparse
    arguments = argparse.ArgumentParser(description="Pulls data from pubchem.")
    arguments.add_argument('-o', '--output', help="Where to write the output")
    arguments.add_argument('-l', '--limit', type=int, help="Limit inputs to run quickly for testing changes")
    arguments.add_argument('--prev-attrs', help="Previous export, to load in cas data from")
    args = arguments.parse_args()


    setupLogging()
    run(args.output, args.limit, args.prev_attrs)