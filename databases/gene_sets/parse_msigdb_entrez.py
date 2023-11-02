#!/usr/bin/env python3
import sys, argparse, os
sys.path.insert(1,"../lincs")
sys.path.insert(1,"../disgenet")
sys.path.insert(1,"../../web1")
from parse_disgenet import make_entrez_2_uniprot_map
from path_helper import make_directory

def get_uniprot_csv(entrzs, entrez2prot_map):
    misses = set()
    unis = set()
    for es in entrzs:
        try:
            unis.update(entrez2prot_map[es])
        except KeyError:
            misses.add(str(es))
    return ",".join(list(unis))

def get_out_file(ifile):
    name_parts = ifile.split(".")
    if name_parts[0] == 'h':
        first = 'hallmark'
        second = 'all'
    elif name_parts[0] == 'c1':
        first = 'genomic_position'
        second = 'all'
    elif name_parts[0] == 'c2':
        first = 'annotated'
        if name_parts[1] == 'all':
            second = 'all'
        elif name_parts[1] == 'cgp':
            second = 'chem_gene_perturbations'
        elif name_parts[1] == 'cp' and name_parts[2] == 'reactome':
            second = 'pathways_reactome'
        elif name_parts[1] == 'cp' and name_parts[2] == 'biocarta':
            second = 'pathways_biocarta'
        elif name_parts[1] == 'cp' and name_parts[2] == 'kegg':
            return None
        elif name_parts[1] == 'cp':
            second = 'pathways_all'
    elif name_parts[0] == 'c3':
        first = 'motifs'
        if name_parts[1] == 'all':
            second = 'all'
        elif name_parts[1] == 'tft':
            second = 'TF_targets'
        elif name_parts[1] == 'mir':
            second = 'miR_targets'
    elif name_parts[0] == 'c4':
        first = 'computational'
        if name_parts[1] == 'all':
            second = 'all'
        elif name_parts[1] == 'cgn':
            second = 'cancer_neighborhoods'
        elif name_parts[1] == 'cm':
            second = 'cancer_modules'
    elif name_parts[0] == 'c5':
        first = 'gene_ontology'
        if name_parts[1] == 'all':
            second = 'all'
        elif name_parts[1] == 'bp':
            second = 'bio_process'
        elif name_parts[1] == 'mf':
            second = 'molec_functions'
        elif name_parts[1] == 'cc':
            second = 'cell_component'
    elif name_parts[0] == 'c6':
        first = 'onco'
        second = 'all'
    elif name_parts[0] == 'c7':
        first = 'immun'
        second = 'all'
    name = ".".join([first, second, name_parts[2], 'uniprot', name_parts[-1]])
    return name

if __name__=='__main__':
    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="msigdb gmt file")
    arguments.add_argument("dir", help="e.g. msigdb_v5.1_entrez/")
    arguments.add_argument("uniprot_data", help="HUMAN_9606_Uniprot_data.tsv, processed from uniprot")
    arguments.add_argument("odir", help="outputdirectory")
    args = arguments.parse_args()

    entrez2uniprot_map = make_entrez_2_uniprot_map(args.uniprot_data)
    from dtk.files import get_dir_file_names
    all_files = get_dir_file_names(args.dir)
    make_directory(args.odir)
    bad_prefixes=['KEGG_','ST_']
    for file in all_files:
        if not file.endswith('entrez.gmt') or file.startswith('msigdb'):
            continue
        outfile = get_out_file(file)
        if not outfile:
            continue
        with open(os.path.join(args.dir, file), 'r') as f:
            with open(os.path.join(args.odir, outfile), 'w') as o:
                for l in f:
                    if any([l.startswith(x) for x in bad_prefixes]):
                        continue
                    fields = l.rstrip("\n").split("\t")
                    csv = get_uniprot_csv(fields[2:], entrez2uniprot_map)
                    o.write("\t".join([fields[0], csv]) + "\n")

