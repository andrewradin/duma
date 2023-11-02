

# For each primary uniprot in f1
# If it's now an alt uniprot, output

# For each cluster in f1
# If its prots are now part of separate clusters, output


# For each cluster in f2
# If its prots used to be separate clusters, output


def compare(unidata1, unidata2):
    from collections import defaultdict
    stats = defaultdict(int)

    altunis1 = {}
    for uni1, uni1_data in unidata1.items():
        for alt in uni1_data.get('Alt_uniprot', set()):
            altunis1[alt] = uni1

    altunis2 = {}
    for uni2, uni2_data in unidata2.items():
        for alt in uni2_data.get('Alt_uniprot', set()):
            altunis2[alt] = uni2

    stats['uni1'] = len(unidata1)
    stats['uni2'] = len(unidata2)
    stats['altuni1'] = len(altunis1)
    stats['altuni2'] = len(altunis2)

    # Each is {uniprot: {attr_name: val}}
    for uni1, uni1_data in unidata1.items():
        if uni1 in altunis2:
            stats['unichg'] += 1
            prim2 = altunis2[uni1]
        else:
            if uni1 not in unidata2:
                stats['dropped'] += 1
                print(f'Dropped {uni1}')
                continue
            else:
                # This is still a primary uniprot
                prim2 = uni1

        uni2_data = unidata2[prim2]

        gene1 = uni1_data.get('Gene_Name', set())
        gene2 = uni2_data.get('Gene_Name', set())
        change = 'same' if gene1 == gene2 else 'diff'
        stats['gene_' + change] += 1

        if uni1 != prim2 or gene1 != gene2:
            print(f'{uni1} {gene1} converted to {prim2} {gene2} ({change})')


        all_gene_1 = uni1_data.get('Gene_Synonym', set()) | gene1

        cluster1_genes2 = defaultdict(set)
        #cluster1_genes2[frozenset(all_gene_1)].add(uni1)

        for alt1 in uni1_data.get('Alt_uniprot', []):
            # Find the gene name of its new cluster
            # If we have a mixed set of gene names, output and track stats.
            if alt1 in altunis2:
                prim2 = altunis2[alt1]
            elif alt1 not in unidata2:
                continue
            else:
                prim2 = alt1
            uni2_data = unidata2[prim2]
            gene_name2 = uni2_data.get('Gene_Name', set()) | uni2_data.get('Gene_Synonym', set())
            #cluster1_genes2.add(frozenset(gene_name2))
            cluster1_genes2[frozenset(gene_name2)].add((alt1, prim2))

        if len(cluster1_genes2) > 1:
            stats['split_cluster'] += 1
            print(f'Split {uni1} {gene1} ({all_gene_1}) into:')
            for genes, prots in cluster1_genes2.items():
                print(f'{list(genes)} - {list(prots)}')


    from pprint import pprint
    pprint(dict(stats))

if __name__ == "__main__":
    import sys

    from dtk.data import MultiMap
    from dtk.files import get_file_records
    from uniprot_parser2 import group_uniprot_data
    unidata1 = dict(group_uniprot_data(get_file_records(sys.argv[1], progress=True)))
    unidata2 = dict(group_uniprot_data(get_file_records(sys.argv[2], progress=True)))
    compare(unidata1, unidata2)
