#!/usr/bin/env python3

import logging
from tqdm import tqdm
logger = logging.getLogger('parse')

def make_alt2uni(unifile):
    """Maps uniprots to our canonical uniprot identifier"""
    from dtk.files import get_file_records
    alt2uni = {}
    for rec in get_file_records(unifile, keep_header=False):
        if rec[1] == 'Alt_uniprot':
            alt2uni[rec[2]] = rec[0]
        alt2uni[rec[0]] = rec[0]
    return alt2uni

def bad_edge(parent, child, G):
    # This is actually the inverse of the relationship, ignore it.
    return 'has_part' in G.adj[child][parent]

def make_full_genesets(direct_genesets, G, cur_nodes, out, stack=[]):
    for node in cur_nodes:
        if node in out:
            # Already computed via another path in the graph.
            assert out[node] is not None, f'Cycle found at {node}, {stack}'
            continue

        stack.append(node)
        # Pre-assign to None so that we can check for cycles.
        out[node] = None
        children = [x[0] for x in G.in_edges(node) if not bad_edge(node, x[0], G)]
        make_full_genesets(direct_genesets, G, children, out, stack)
        cur_set = set(direct_genesets[node])
        for child in children:
            cur_set.update(out[child])
        out[node] = cur_set 
        stack.pop()
    return out

def run(go, goa, uniprot, out_genesets, out_hier):
    alt2uni = make_alt2uni(uniprot)
    import obonet
    import networkx as nx
    from tqdm import tqdm
    from dtk.files import get_file_records

    logger.info(f"Reading in ontology from {go}")
    G = obonet.read_obo(go)

    header = ['id', 'name', 'type', 'description', 'children', 'has_diagram']
    rows = []
    # GO has 3 roots - biological_process, molecular_function and cellular_component
    # Just find them dynamically rather than hardcoding.
    roots = []
    for node, attrs in tqdm(G.nodes(data=True)):
        in_edges = [x[0] for x in G.in_edges(node) if not bad_edge(node, x[0], G)]
        num_out_edges = len(G.out_edges(node))
        if num_out_edges == 0:
            roots.append(node)
            logger.info(f"Found root {attrs['name']} {node}")
        children = '|'.join(sorted(in_edges))
        row = [
            node,
            attrs['name'],
            attrs['namespace'],
            attrs['def'],
            children,
            '0'
        ]
        rows.append(row)

    assert  len(roots) == 3, f'Expected to find 3 roots, instead found {roots}'

    from collections import defaultdict
    direct_genesets = defaultdict(set)
    
    found_good_ver = False
    missing_uniprot = set()
    for rec in get_file_records(goa, progress=True, parse_type='tsv'):
        if rec[0][0] == '!':
            # Comments / metadata
            if rec[0].startswith('!gaf-version:'):
                ver = float(rec[0].split(' ')[1])
                assert ver in [2.1, 2.2], f"""
                    Unrecognized GAF version {ver}.  Check the latest info here:
                      http://geneontology.org/docs/download-go-annotations/
                    make any necessary changes and add the version to supported list."""

                found_good_ver = True
            continue
            
        assert found_good_ver, f"Never found a version header, please check the input file {goa}"

        # Details at http://geneontology.org/docs/go-annotation-file-gaf-format-2.2/
        db, db_id, db_sym, qual, go_id, db_ref, ev_code, with_or_from, aspect, db_name, db_syn, db_type, taxon, date, assiged_by, anno_ext, gene_product_form = rec

        assert db == 'UniProtKB', f'Unknown db {db}'

        if db_id not in alt2uni:
            missing_uniprot.add(db_id)
            continue
        uniprot = alt2uni[db_id]

        direct_genesets[go_id].add(uniprot)
    
    if missing_uniprot:
        logger.error(f"Couldn't convert {len(missing_uniprot)} uniprots: {missing_uniprot}")
        logger.error("Maybe a uniprot version mismatch?  This is no longer expected after uniprot v9.")


    from atomicwrites import atomic_write
    with atomic_write(out_hier, overwrite=True) as f:
        for row in [header] + rows:
            f.write('\t'.join(row) + '\n')

    full_genesets = make_full_genesets(direct_genesets, G, roots, {})
    with atomic_write(out_genesets, overwrite=True) as f:
        for id, prots in tqdm(full_genesets.items()):
            # Only write out protsets with actual proteins.
            if prots:
                protstr = ','.join(prots)
                f.write('\t'.join([id, protstr]) + '\n')

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse data')
    parser.add_argument('-g','--go', help = 'Gene ontology .OWL file')
    parser.add_argument('-a', '--goa', help = 'Annotation .gaf file')
    parser.add_argument('--out-genesets', help = 'Where to write the genesets')
    parser.add_argument('--out-hier', help = 'Where to write the hierarchy')
    parser.add_argument('-u', '--uniprot', help = 'Uniprot converter file')
    args = parser.parse_args()
    from dtk.log_setup import setupLogging
    setupLogging()

    run(**vars(args))