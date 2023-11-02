#!/usr/bin/env python

import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm

# XXX As currently written, this code can report a single uniprot as an
# XXX alternate in more than one cluster. Protein.get_canonical_of_uniprot()
# XXX contains code to deal with this by making an arbitrary choice, but
# XXX also contains a comment saying that would be better handled here.
# XXX
# XXX canonicalize_alt_uniprots() is an attempt to do this, but after checking
# XXX this in near the end of his tenure here, Darin decided it was maybe
# XXX a bad idea to just toss data here, and instead we should mark a
# XXX preferred alternate. This is on hold pending defining an additional
# XXX attribute in the output file, and querying that attribute in the
# XXX Protein model.


def canonicalize_alt_uniprots(clusters, acc_clusters, cluster_order):
    """
    There are a few hundred cases where a uniprot is listed as an alt_uniprot
    for multiple other uniprots.
    e.g. https://www.uniprot.org/uniprot/Q5T2V8
    
    There is code in Protein.get_canonical_of_uniprot that kind of doesn't
    like this, it would prefer a single definitive canonical uniprot for
    each uniprot.

    The code in this function attempts to do that, but it does so by removing
    all-but-one alt_uniprot.  Instead, there should probably be a new property
    (e.g. canon_alt_uniprot) that we create and gets used.  So for now this
    function is not actually invoked, but the logic should be most of the
    way to what you need.
    """
    alt_to_cluster = []

    logger.info("Canonicalizing alt uniprots")

    for cluster in clusters:
        # Write to the standard output.
        # Pick a representative prot.
        prot = max(cluster['uniprot'], key=cluster_order)
        alt_prot = {x for x in cluster['uniprot'] if x != prot}
        if prot in acc_clusters:
            alt_prot |= acc_clusters[prot]

        cluster['prot'] = prot
        cluster['Alt_uniprot'] = alt_prot

        for prot in alt_prot:
            alt_to_cluster.append((prot, cluster))

    def cluster_summary(clust):
        # Can use this for more details.
        #return f"{clust['prot']} ({clust['Alt_uniprot']})"
        return f"{clust['prot']}"

    from dtk.data import kvpairs_to_dict
    num_fixed = 0
    for alt, clusts in kvpairs_to_dict(alt_to_cluster).items():
        if len(clusts) == 1:
            continue

        # This prot is the alt_uniprot for multiple clusters.
        # Have to pick one to canonicalize on... arbitrarily picking
        # the cluster with the most
        # properties and then lexicographically smallest amongst ties.
        best_clust = min(clusts, key=lambda x: (-len(x), x['prot']))

        remove_clusts = [x for x in clusts if x is not best_clust]

        logger.info(f"Removing {alt} from {[cluster_summary(x) for x in remove_clusts]}, leaving in {cluster_summary(best_clust)}")

        for clust in remove_clusts:
            logger.info(f"Removing from {cluster_summary(clust)}")
            try:
                clust['Alt_uniprot'].remove(alt)
            except KeyError:
                logger.error(f"Missing {alt}")

        num_fixed += 1

    logger.info(f"Canonicalized {num_fixed} ambiguous alt_uniprots")



def group_uniprot_data(src):
    """Takes records from src and groups them by key.

    Outputs a forward multimap of data for each uniprot.
    """

    from collections import defaultdict
    prot_pairs = defaultdict(list)
    from dtk.data import MultiMap
    # Note that it would be more efficient to process these
    # in-order if they were all grouped, but there are weird
    # one-off entries that are out-of-order.

    for uniprot, key, value in src:
        # Isoforms are listed uniprot-#
        if '-' in uniprot:
            uniprot, isoid = uniprot.split('-')
        prot_pairs[uniprot].append((key, value))

    from tqdm import tqdm
    for prot, pairs in tqdm(prot_pairs.items(), desc='Grouping keys'):
        yield prot, MultiMap(pairs).fwd_map()

def map_to_pairs(fwd_map):
    for key, vals in fwd_map.items():
        for val in vals:
            yield key, val

attrs_to_keep = [
           'UniProtKB-ID',
           'Alt_UniProtKB-ID',
           'Gene_Name',
           'Gene_Synonym',
           'STRING',
           'GeneCards',
           'KEGG',
           'Reactome',
           'MIM',
           'GeneID', # Entrez
           'Ensembl',
           'Ensembl_TRS',
           'Ensembl_PRO',
           'Alt_uniprot',
           'hgnc',
           ]

# A cluster has to report at least one of these identifiers, or it isn't
# actually useful to us and we can just toss it.
useful_attrs = [
           'GeneID', # Entrez
           'Ensembl',
           'Ensembl_TRS',
           'Ensembl_PRO',
           'STRING',
           'hgnc',
           'Reactome',
        ]

def is_useful_cluster(cluster):
    for attr in useful_attrs:
        if attr in cluster:
            return True
    return False

def write_cluster(cluster, prot, out):
    for prop_name, prop_vals in sorted(cluster.items()):
        if prop_name not in attrs_to_keep:
            continue
        for val in prop_vals:
            out.write(f'{prot}\t{prop_name}\t{val}\n')


#CAREFUL
# ProtA:  HGNC: 001
# ProtB: HGNC:  002
# ProtC (unreviewed): HGNC: 001, 002

def primary_cluster_keys(prot_data):
    """Primary cluster keys always combine to the same cluster.

    We compute the transitive closure of these and put all together.
    We also only compute these for reviewed uniprots, as unreviewed ones
    can span multiple genes.
    """
    cluster_on = ['hgnc', 'uniprot']
    out = []
    if 'reviewed' not in prot_data['reviewed']:
        return out
    for key in cluster_on:
        vals = prot_data.get(key, [])
        for val in vals:
            if val:
                out.append((key, val))

    return out

def secondary_cluster_keys(prot_data):
    """These are more ambiguous keys that suggest clustering but could overcluster.

    These are used to cluster unassigned prots, but two primary clusters
    will not be merged across only secondary keys.
    """
    cluster_on = ['hgnc', 'Ensembl', 'Ensembl_PRO', 'Ensembl_TRS', 'UniRef100', 'GeneID']
    out = []
    for key in cluster_on:
        vals = prot_data.get(key, [])
        for val in vals:
            if val:
                out.append((key, val))

    # Merge both Gene_Name and Gene_Synonym
    gene_keys = ['Gene_Name', 'Gene_Synonym']
    for key in gene_keys:
        vals = prot_data.get(key, [])
        for val in vals:
            if val:
                out.append(('Gene', val))
    return out


def load_accession_clusters(xml_fn):
    from lxml import etree as ET

    ns = '{http://uniprot.org/uniprot}'
    import gzip
    it = ET.iterparse(gzip.open(xml_fn), events=('end',), tag=ns+'entry')
    output = {}
    logger.info("Starting to parse XML")
    for action, elem in tqdm(it, total=200000):
        if elem.tag != ns+'entry':
            elem.clear()
            continue

        uniprots = set(x.text for x in elem.findall(ns+'accession'))
        for prot in uniprots:
            # Apparently this happens a lot.  e.g. P62158 is an obsolete
            # entry for 3 genes.
            # assert prot not in output, f'{prot} showed up multiple times'
            output[prot] = uniprots

        elem.clear()
    logger.info("Done parsing XML")
    return output


def run(idmap_fn, canonical_fn, xml_fn, output_fn, debug_fn):
    from dtk.files import get_file_records
    from collections import defaultdict

    # We're just using this to pull out obsolete accessions used by other
    # data sources that aren't listed in any of our other files.
    acc_clusters = load_accession_clusters(xml_fn)

    # {uniprot:{attr:val,...},...}
    uniprot_data = defaultdict(dict)

    # [(hgnc,uniprot),...]
    # XXX this is not actually used; superceded by prot_cluster_* below?
    hgnc_to_prot = []

    # Open the canonical file, generate an initial set of attributes
    # in the uniprot_data dict, and build hgnc_to_prot list.
    # This file has one record per uniprot with attributes in columns.
    # Multi-value attributes are converted to sets. Attributes are:
    # 'uniprot' - same as key in uniprot_data, e.g. 'P31946'
    # 'reviewed' - Status column; either 'reviewed' or 'unreviewed'
    # 'hgnc' - set of HGNC ids, e.g. {'26422'}
    # 'primary_gene_name' - set of gene names, e.g. {'ZNF558'}

    for rec in get_file_records(canonical_fn, parse_type='tsv', keep_header=False, progress=True):
        uniprot, uni_entry_name, prot_names, gene_names, \
                organism, gene_names_prim, gene_names_second, \
                reviewed, hgnc_str, ensembl = rec
        if reviewed:
            uniprot_data[uniprot]['reviewed'] = {reviewed}
        if hgnc_str:
# starting with our version 14, ensembl data was getting pulled in here as well, so now actively filtering those out
            hgncs = set(h for h in [x.strip() for x in hgnc_str.split(';') if x.strip()] if not h.startswith('ENS'))
            uniprot_data[uniprot]['hgnc'] = hgncs
            for hgnc in hgncs:
                hgnc_to_prot.append((hgnc, uniprot))
        uniprot_data[uniprot]['uniprot'] = {uniprot}
        # A few of them have multiple primary names, ;-separated.
        uniprot_data[uniprot]['primary_gene_name'] = [x.strip() for x in gene_names_prim.split(';') if x.strip()]



    # [(prot,(key_name,key_value)),...]
    prot_cluster_keys = []
    prot_cluster_keys2 = []

    # Open the ID file, get more attributes
    # NOTE: IDMap file has no header.
    src = get_file_records(idmap_fn, parse_type='tsv', keep_header=None, progress=True)
    # {uniprot:{attr:{val,...},...},...}
    prot_to_data = dict(group_uniprot_data(src))


    # If you're trying to figure out why some prots are or aren't clustering
    # with each other you can stick them in here and get some more debug
    # output about them.
    # Also take a look at debug.tsv which has the unmerged props for each.
    DEBUG_PROTS=set( [
        #'P0DJD8',
        #'P0DJD7',
        ])
    DEBUG_KEYS=set([
        #('hgnc', '8885'),
        #('hgnc', '8886'),
        ])

    # for each uniprot
    for prot, data in prot_to_data.items():
        # augment prot_to_data entry with uniprot_data entry
        data.update(uniprot_data.get(prot, {'uniprot': {prot}}))

        # Starting in our version 13 Uniprot included Ensembl version info
        # we don't need to use that info, so strip it
        for x in data:
            if x.startswith('Ensembl'):
                if isinstance(data[x], str): # if there's just one entry
                    new_val = data[x].split('.')[0]
                    data[x] = new_val
                else:
                    new_val = set()
                    for v in data[x]:
                        new_val.add(v.split('.')[0])
                    data[x] = new_val

        # merge any 'HGNC' data from prot_to_data into the 'hgnc' attribute
        # from uniprot_data (after format conversion)
        if 'HGNC' in data:
            # Convert any HGNC references into the same format as the
            # canonical.
            for hgnc_str in data['HGNC']:
                prefix, hgnc = hgnc_str.split(':')
                assert prefix == 'HGNC'
                if 'hgnc' not in data:
                   data['hgnc']=set()
                data['hgnc'].add(hgnc)

        # collect keys for potential clustering of duplicate uniprot ids;
        # primary keys are trusted; secondary keys are hints; log any
        # keys of interest for debugging
        keys = primary_cluster_keys(data)
        if prot in DEBUG_PROTS or DEBUG_KEYS & set(keys):
            logger.info(f"{prot} has primary keys {keys}")
        for key in keys:
            prot_cluster_keys.append((prot, key))


        keys = secondary_cluster_keys(data)
        if prot in DEBUG_PROTS or DEBUG_KEYS & set(keys):
            logger.info(f"{prot} has secondary keys {keys}")
        for key in keys:
            prot_cluster_keys2.append((prot, key))


    from tqdm import tqdm
    from dtk.data import MultiMap
    prot_keys_mm = MultiMap(prot_cluster_keys)
    prot_keys2_mm = MultiMap(prot_cluster_keys2)
    done_keys = set()
    done_keys2 = set()

    # [{attr:{val,...},...},...]
    # a list of attribute collections, one per canonical protein
    clusters = []
    # {uniprot,...}
    # all proteins found via secondary key
    used_prots = set()


    # Primary prots with multi HGNC should cluster.
    #   Do I need to actually cluster for that, or all listed correctly?
    #   Assume correct, just pull in, merge attrs, suck in any secondaries.
    #   See what is left.
    #   --> Run primary cluster code, do secondary non-transitively, see
    #       what is left.

    for key in tqdm(sorted(prot_keys_mm.rev_map()), desc='Clustering'):
        if key in done_keys:
            continue

        keys_to_visit = set([key])
        merged_data = set()
        while keys_to_visit:
            cur_key = keys_to_visit.pop()
            if cur_key in done_keys:
                continue
            done_keys.add(cur_key)
            prots = prot_keys_mm.rev_map()[cur_key]

            if cur_key in DEBUG_KEYS:
                logger.info(f"Processing key {cur_key} for {key} with prots {prots}")
            for prot in prots:
                if prot in DEBUG_PROTS:
                    logger.info(f"Merging {prot} into prim cluster for {key}")
                    logger.info(prot_to_data[prot])
                # Merge in this prot's data.
                merged_data.update(map_to_pairs(prot_to_data[prot]))

                # Find this prot's secondary keys,
                for key2 in prot_keys2_mm.fwd_map()[prot]:
                    if key2 in done_keys2:
                        continue
                    done_keys2.add(key2)
                    for prot2 in prot_keys2_mm.rev_map()[key2]:
                        # Make sure it's not a primary prot.
                        if prot2 not in prot_keys_mm.fwd_map() and prot2 not in used_prots:
                            used_prots.add(prot2)
                            merged_data.update(map_to_pairs(prot_to_data[prot2]))

                # Add any new keys attached to this prot that we haven't seen yet.
                keys_to_visit.update(prot_keys_mm.fwd_map()[prot] - done_keys)
        clusters.append(MultiMap(merged_data).fwd_map())


    num_secondary_clusters = 0
    for key2 in tqdm(prot_keys2_mm.rev_map(), desc='2nd Clustering'):
        if key2 in done_keys2:
            continue
        keys_to_visit = set([key2])
        merged_data = set()
        while keys_to_visit:
            cur_key = keys_to_visit.pop()
            if cur_key in done_keys:
                continue
            done_keys2.add(cur_key)
            prots = prot_keys2_mm.rev_map()[cur_key] - used_prots
            for prot in prots:
                if prot in DEBUG_PROTS:
                    logger.info(f"Merging {prot} into 2ndary cluster for {key}")
                # Merge in this prot's data.
                merged_data.update(map_to_pairs(prot_to_data[prot]))
                # Add any new keys attached to this prot that we haven't seen yet.
                keys_to_visit.update(prot_keys2_mm.fwd_map()[prot] - done_keys2)
        cluster = MultiMap(merged_data).fwd_map()
        if is_useful_cluster(cluster):
            num_secondary_clusters += 1
            clusters.append(cluster)


    def cluster_order(prot):
        # Prefer 'reviewed' uniprots; otherwise, use # of keys as a proxy
        # for how well-studied/referenced it is.
        d = prot_to_data[prot]
        if d['reviewed'] == {'reviewed'}:
            rvw = 2
        elif 'reviewed' in d:
            rvw = 1
        else:
            # Not even unreviewed, probably obsolete prot.
            # This would happen if everything in the cluster were only in
            # the idmapping file, not in the canonical file.
            rvw = 0
        return (rvw,
                len(prot_keys_mm.fwd_map().get(prot, [])),
                len(prot_keys2_mm.fwd_map()[prot]),
                len(d),
                )

    from atomicwrites import atomic_write


    # go through all clusters, and report anything suspicious
    with atomic_write(debug_fn, overwrite=True) as debug:
        cluster_warnings = []
        # The debug file is like a merged input file, with attributes associated
        # to the underlying uniprots rather than the merged cluster.
        debug.write('Uniprot\tattribute\tvalue\n')
        for i, cluster in enumerate(tqdm(clusters, desc='Debug')):
            all_reviewed = {uni for uni in cluster['uniprot'] if 'reviewed' in prot_to_data[uni]['reviewed']}
            all_rev_hgnc = {frozenset(prot_to_data[uni].get('hgnc', [])) for uni in cluster['uniprot'] if 'reviewed' in prot_to_data[uni]['reviewed']}

            if len(all_reviewed) > 1:
                cluster_warnings.append(f"Clustered reviewed uniprots: {sorted(all_reviewed)} [rev hgnc: {all_rev_hgnc}]")
            for prot in sorted(cluster['uniprot'], key=cluster_order, reverse=True):
                debug.write(f'{prot}\tCluster\t{i}\n')
                for attr, val in prot_keys2_mm.fwd_map()[prot]:
                    debug.write(f'{prot}\t{attr}\t{val}\n')
                debug.write(f'{prot}\treviewed\t{prot_to_data[prot].get("reviewed")}\n')

        # These aren't necessarily problematic - there are a bunch of reviewed
        # uniprots that HGNC considers the same gene - but when debugging
        # these are usually the ones that you care about changing.
        # Note that these text warnings are written to the end of what is
        # otherwise a tsv file.
        for msg in sorted(cluster_warnings):
            debug.write(msg + '\n')

    # XXX Disabled for now, see comment at top of function.
    # XXX If/when this gets renabled the "prot =" line noted below should be
    # XXX re-enabled as well
    # canonicalize_alt_uniprots(clusters, acc_clusters, cluster_order)

    # Write out clusters. Some attributes are split into a single-valued
    # primary attribute, and multi-values alternate attributes.
    with atomic_write(output_fn, overwrite=True) as out:
        # The full output file has all attributes from any uniprot in the cluster.
        out.write('Uniprot\tattribute\tvalue\n')
        for cluster in tqdm(clusters, desc='Output'):
            # Write to the standard output.
# XXX see "XXX" note above
# XXX When this line is reneabled the lines from "prot=..." to "cluster['Alt..." should be taken out
            # prot = cluster.pop('prot')
            # Pick a representative prot.
            prot = max(cluster['uniprot'], key=cluster_order)
            alt_prot = {x for x in cluster['uniprot'] if x != prot}
            if prot in acc_clusters:
                alt_prot |= acc_clusters[prot]
            del cluster['uniprot']
            cluster['Alt_uniprot'] = alt_prot

            # Pick a representative primary gene name, the rest become synonyms.
            canon_names = list(prot_to_data[prot]['primary_gene_name'])
            if canon_names:
                canon_name = canon_names[0]
            else:
                canon_name = list(cluster.get('Gene_Name', [None]))[0]

            secondary_names = (cluster.get('Gene_Name', set()) | cluster.get('Gene_Synonym', set())) - {canon_name}
            if canon_name:
                cluster['Gene_Name'] = {canon_name}
            cluster['Gene_Synonym'] = secondary_names

            # Pick a primary uniprotkb - not clear we really use this anywhere,
            # but platform asserts there is only 1.
            cluster['Alt_UniProtKB-ID'] = cluster['UniProtKB-ID']
            if len(cluster['UniProtKB-ID']) > 1:
                # Grab the value from the primary protein.
                prim_data = prot_to_data[prot]
                assert len(prim_data['UniProtKB-ID']) == 1, f"Unexpected UniProtKB for {prot}: {prim_data[prot]}"
                cluster['UniProtKB-ID'] = {list(prim_data['UniProtKB-ID'])[0]}
            cluster['Alt_UniProtKB-ID'] -= cluster['UniProtKB-ID']

            write_cluster(cluster, prot, out)


    print(f"Generated {len(clusters)} clusters; {num_secondary_clusters} secondary")
if __name__ == "__main__":
    from dtk.log_setup import setupLogging
    import argparse
    arguments = argparse.ArgumentParser(description="Parses and generates uniprot")
    arguments.add_argument("-i", '--id-input', required=True, help="ID Mapping input file")
    arguments.add_argument("-c", '--canonical-input', required=True, help="Canonical input file")
    arguments.add_argument("-x", '--xml-input', required=True, help="XML input file")
    arguments.add_argument("-o", '--output',  help="output tsv file")
    arguments.add_argument("-d", '--debug',  help="debug output tsv file")

    args = arguments.parse_args()
    setupLogging()

    run(args.id_input, args.canonical_input, args.xml_input, args.output, args.debug)
