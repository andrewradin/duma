#!/usr/bin/env python

# Mondo is an ontology that attempts to also harmonize other ontologies.
# OpenTargets/EFO work closely with them and so should work well together.
# It also seems to connect reasonably well with UMLS (particularly if you include
# non-exact matches).

def extract_mondo_xrefs(mondo_data, other_ontos=None):
    """
    Returns a multimap linking each mondo ID with equivalent terms
    in the requested ontologies.
    """
    other_ontos = other_ontos or ['UMLS', 'Orphanet', 'EFO', 'HP']
    link_pairs = []
    labels = {}
    items = mondo_data['graphs'][0]['nodes']
    from collections import defaultdict
    for item in items:
        if 'meta' not in item or 'xrefs' not in item['meta']:
            continue

        id = item['id'].split('/')[-1]
        if not id.startswith('MONDO_'):
            continue
        mondo_id = id.split('_')[1]
        labels[mondo_id] = item['lbl']

        # Anything deemed an exact match will show up in the xrefs.
        xrefs = item['meta']['xrefs']
        equiv = defaultdict(list)
        
        for xref in xrefs:
            onto, val, *rest = xref['val'].split(':')
            equiv[onto].append(val)
        
        # We also parse out the properties to pull out close matches. 
        for prop in item['meta'].get('basicPropertyValues', []):
            pred, val = prop['pred'], prop['val']
            if pred.endswith('closeMatch') and '/umls/' in val:
                umls_id = val.split('/')[-1]
                equiv['UMLS'].append(umls_id)
                
        for onto_name in other_ontos:
            for onto_id in equiv[onto_name]:
                link_pairs.append((mondo_id, onto_name + '_' + onto_id))
        
    from dtk.data import MultiMap
    return MultiMap(link_pairs), labels


def extract_mondo(mondo_json_fn):
    import json
    with open(mondo_json_fn) as f:
        mondo_data = json.loads(f.read())
    mondo2other, mondo2name = extract_mondo_xrefs(mondo_data)

    return mondo2other, mondo2name

class Mondo:
    def __init__(self, mondo_ver):
        from dtk.s3_cache import S3File
        from dtk.files import get_file_records 
        from dtk.data import MultiMap
        mappings_f = S3File.get_versioned('mondo', mondo_ver, role='mappings')
        labels_f = S3File.get_versioned('mondo', mondo_ver, role='labels')
        mappings_f.fetch()
        labels_f.fetch()

        self.mondo2name = {k:v for k, v in get_file_records(labels_f.path(), keep_header=None)}
        self.mondo2other = MultiMap((k,v) for k, v in get_file_records(mappings_f.path(), keep_header=None))

    def get_related(self, onto_id):
        mondo_ids = self.mondo2other.rev_map().get(onto_id, [])
        out = set()
        for mondo_id in mondo_ids:
            out.update(self.mondo2other.fwd_map().get(mondo_id, []))
        return mondo_ids, out 

    def map_meddra_to_mondo(self, meddra_names, ws):
        """
        Meddra maps best to UMLS, which maps reasonably well to MONDO.
        Most of the terms that don't map are either phenotypes or procedure-related.
        (MONDO defers to HPO for phenotypes, which we could pull in separately).
        """
        def flatten(list_of_sets):
            out = []
            for sets in list_of_sets:
                cur_set = set()
                for entry in sets:
                    cur_set.update(entry)
                cur_set -= {None}
                out.append(cur_set)
            return out

        from dtk.meddra import IndiMapper
        from browse.default_settings import meddra
        meddra_version = meddra.value(ws=ws)
        med_im = IndiMapper(meddra_version)
        # For each term in meddra_names, this contains a set of meddra IDs.
        meddra_mapped = ([{x[0] for x in med_im.translate(name)} for name in meddra_names])

        meddra2umls = med_im.make_meddra_umls_converter(ws=None)
        # Each term now mapped to a set of UMLS IDs.
        umls_lists = flatten([[meddra2umls.get(x, []) for x in meddras] for meddras in meddra_mapped])
        #... and now MONDO IDs.
        mondos = flatten([[self.mondo2other.rev_map().get(f'UMLS_{x}', []) for x in umls] for umls in umls_lists])

        return mondos
        