#!/usr/bin/env python
from atomicwrites import atomic_write
import logging
logger = logging.getLogger(__name__)

def run(genesets, hierarchies, output_hierarchy, output_geneset, output_gene_to_pathway):
    prot2pathway = []

    from tqdm import tqdm

    from dtk.files import get_file_lines
    with atomic_write(output_geneset, overwrite=True) as f:
        for gs in genesets:
            for line in get_file_lines(gs, progress=True):
                f.write(line)

                pw_id, prots = line.strip().split('\t')
                for prot in prots.split(','):
                    prot2pathway.append((prot, pw_id))
    
    with atomic_write(output_hierarchy, overwrite=True) as f:
        wrote_header = False
        for hier in hierarchies:
            header = None
            for line in get_file_lines(hier, progress=True):
                if not header:
                    header = line
                    if not wrote_header:
                        wrote_header = True
                        f.write(line)
                    continue
                f.write(line)

    from dtk.tsv_alt import SqliteSv 
    SqliteSv.write_from_data(output_gene_to_pathway, tqdm(prot2pathway), types=[str, str], header=['prot', 'pw'])


if __name__ == "__main__":
    from dtk.log_setup import setupLogging
    setupLogging()
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--genesets", nargs='+', help="Input for all pathways and reactions")
    parser.add_argument("--hierarchies", nargs='+', help="Input for hierarchy data")
    parser.add_argument("--output-hierarchy", help="Output for hierarchy data")
    parser.add_argument("--output-geneset", help="Output for hierarchy data")
    parser.add_argument("--output-gene-to-pathway", help="Output for gene2pathway")
    args = parser.parse_args()
    run(**vars(args))
