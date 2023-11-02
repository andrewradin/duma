#!/usr/bin/env python

import sys
try:
    import path_helper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")

import os
#-------------------------------------------------------------------------------
# Driver
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                description='count the number of Uniprots in each HPA tissue',
                )
    parser.add_argument("i", help="humanProtAtlas.expression.tsv.gz")
    parser.add_argument("ppi", help="one of our PPI files")
    parser.add_argument("--min_ppi_t", default = 0.5, help="Minimum PPI threshold. Default: %d(default)")
    args = parser.parse_args()

    from dtk.files import get_file_records
    ppi={}
    for frs in get_file_records(args.ppi, keep_header=False, parse_type='tsv'):
        if float(frs[2]) < args.min_ppi_t:
            continue
        if frs[0] not in ppi:
            ppi[frs[0]] = {}
        ppi[frs[0]][frs[1]] = frs[2]
    gene2uni = {}
    tiss = {}
    cell = {}
    for frs in get_file_records(args.i, keep_header=True, parse_type='tsv'):
        if frs[1] == 'Uniprot':
            gene2uni[frs[0]] = frs[2]
        elif frs[3] == 'tisExp':
            if frs[0] not in gene2uni or frs[2] == '0':
                continue
            t = frs[1].split(";")[0]
            if ';' in frs[1]:
                if frs[1] not in cell:
                    cell[frs[1]] = set()
                cell[frs[1]].add(gene2uni[frs[0]])
            if t not in tiss:
                tiss[t] = set()
            tiss[t].add(gene2uni[frs[0]])
    with open('tissue_uniprot_counts.tsv', 'w') as f:
        f.write("\t".join(['tissue',
                           'expressed prots',
                           'present PPI prots',
                           'missing PPI prots',
                           'present PPI edges',
                           'missing PPI edges'
                          ])
                + "\n")
        for t,s in tiss.iteritems():
            present_nodes = set(ppi.keys()) & s
            present_edge_cnts = str(sum([len(d)
                                    for n in present_nodes
                                    for d in ppi[n]
                                   ]))
            missing_nodes = set(ppi.keys()) - s
            missing_edge_cnts = str(sum([len(d)
                                    for n in missing_nodes
                                    for d in ppi[n]
                                   ]))
            f.write("\t".join([t]+
                              [str(len(x)) for x in [s,present_nodes,missing_nodes]]+
                              [present_edge_cnts, missing_edge_cnts]
                             )
                    + "\n")
    with open('cell_uniprot_counts.tsv', 'w') as f:
        f.write("\t".join(['tissue',
                           'expressed prots',
                           'present PPI prots',
                           'missing PPI prots',
                           'present PPI edges',
                           'missing PPI edges'
                          ])
                + "\n")
        for t,s in cell.iteritems():
            present_nodes = set(ppi.keys()) & s
            present_edge_cnts = str(sum([len(d)
                                    for n in present_nodes
                                    for d in ppi[n]
                                   ]))
            missing_nodes = set(ppi.keys()) - s
            missing_edge_cnts = str(sum([len(d)
                                    for n in missing_nodes
                                    for d in ppi[n]
                                   ]))
            f.write("\t".join([t]+
                              [str(len(x)) for x in [s,present_nodes,missing_nodes]]+
                              [present_edge_cnts, missing_edge_cnts]
                             )
                    + "\n")
