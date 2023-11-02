#!/usr/bin/env python

import logging
logger = logging.getLogger(__name__)

all_id_map = {}
def clean_id(id):
    global all_id_map
    orig_id = id

    # Most of the IDs are what we'd expect (e.g. "EFO:12345")
    # Some, though, are prefixed with a URL namespace and use underscores (e.g.: https://efo.ebi..../EFO_12345)
    # Transform them all to be the first style, but also check that we're not causing 
    id = id.split('/')[-1]
    id = id.replace('_', ':')

    if id in all_id_map:
        assert all_id_map[id] == orig_id, f"Multiple mappings for {id} - {orig_id}, {all_id_map[id]}"
    # added the lower in for tracking only as some of the URLs had .../efo/EFO_... and others had .../EFO/EFO_...
    all_id_map[id.lower()] = orig_id.lower()

    return id



def run(input, output):
    from pronto import Ontology, Term
    import warnings
    with warnings.catch_warnings():
        # Tons of SyntaxWarnings loading this ontology around axioms we don't care about.
        warnings.simplefilter("ignore")
        logger.info("Loading ontology")
        efo = Ontology(input)
        logger.info("Done")

    out_data = []
    for term in efo.values():
        if not isinstance(term, Term):
            # There are 2 or 3 non-term entities in here, ignore them.
            continue

        for child in term.subclasses(distance=1, with_self=False):
            out_data.append((clean_id(term.id), clean_id(child.id)))
    
    from dtk.tsv_alt import SqliteSv
    SqliteSv.write_from_data(output, out_data, [str, str], header=['parent', 'child'])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Parse data')
    parser.add_argument('-i','--input', help = 'EFO OpenTargets OWL file')
    parser.add_argument('-o','--output', help = 'Output sqlsv filename')

    args = parser.parse_args()
    from dtk.log_setup import setupLogging
    setupLogging()

    run(**vars(args))
