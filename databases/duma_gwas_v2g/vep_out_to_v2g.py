#!/usr/bin/env python

def make_uniprot_map(file, search_term):
    from dtk.files import get_file_records
    '''Makes a uniprot converter map '''
    conv_d = {}
    for fields in get_file_records(file,
                                   keep_header = True,
                                   parse_type = 'tsv'):
        if fields[1] == search_term:
            try:
                conv_d[fields[2]].add(fields[0])
            except KeyError:
                conv_d[fields[2]] = set()
                conv_d[fields[2]].add(fields[0])
    return conv_d

def create_v2g_output(uniprot_file, input_vep_file, out_fn):
    from dtk.files import get_file_records
    uni_conv = make_uniprot_map(uniprot_file, 'Ensembl')
    fn = input_vep_file
    vep_gen = (x for x in get_file_records(fn, parse_type='tsv', progress=True) if x[0][0:2] != '##')
    header = next(vep_gen)
    header_inds = [header.index(x) for x in header]
    col2ind = dict(zip(header, header_inds))
    iloc = col2ind['Location']
    iallele = col2ind['Allele']
    iconsequence = col2ind['Consequence']
    igene = col2ind['Gene']


    from collections import defaultdict
    loc2prot = defaultdict(set)
    loc2cons = defaultdict(set)
    loc2rsid = defaultdict(set)

    missing_gene = set()
    for item in vep_gen:
        # VEP adds a range sometimes, we just want the start.
        base_loc = item[iloc].split('-')[0]
        snp_loc = base_loc + ':' + item[iallele]
        
        gene = item[igene]
        if gene in uni_conv:
            loc2prot[snp_loc].update(uni_conv[gene])
        else:
            missing_gene.add(gene)
        
        exist_vars = [x.strip() for x in item[col2ind['Existing_variation']].split(',') if x.startswith('rs')]

        if item[0].startswith('rs'):
            # Also add in any rs id that we passed in as input.
            # (If we didn't supply one, it'll get autonamed based on position)
            exist_vars.append(item[0])


        loc2rsid[snp_loc].update(exist_vars)
        loc2cons[snp_loc].update(item[iconsequence].split(','))

    import random
    missing_gene = list(missing_gene)
    random.shuffle(missing_gene)
    print(f"Missing conversions for {len(missing_gene)} genes: (samples): {list(missing_gene)[:10]}")
    print("(As of this writing we expect ~38000 of these, mostly pseudogenes, lncRNA, etc.)")

    out_header = ['chrm_and_pos', 'allele', 'rs_ids', 'consequences', 'uniprot']
    out_data = []

    from atomicwrites import atomic_write
    from tqdm import tqdm
    with atomic_write(out_fn, overwrite=True) as out:
        out.write('\t'.join(out_header) + '\n')

        all_locs = loc2prot.keys() | loc2cons.keys() | loc2rsid.keys()
        for loc in tqdm(all_locs):
            genes = loc2prot[loc]
            cons = loc2cons[loc]
            rsid = loc2rsid[loc]

            cons_val = ','.join(cons)
            rsid_val = ','.join(rsid)

            if not genes:
                # We want to keep track of the fact that we processed this variant, but couldn't find a gene for it.
                genes = {'-'}
            
            for gene in genes:
                chrm, pos, allele = loc.split(':')
                row = [f'{chrm}:{pos}', allele, rsid_val, cons_val, gene]
                out_data.append(row)
    
        for row in sorted(out_data):
            out.write('\t'.join(row) + '\n')



if __name__ == '__main__':
    import argparse
    import sys
    arguments = argparse.ArgumentParser(description='''Converts VEP output to our expected output format''')
    arguments.add_argument('-u', "--uniprot", help = 'Input Uniprot file')
    arguments.add_argument('-i', "--input-vep", help = 'Input VEP file')
    arguments.add_argument('-o', "--output", help = 'Output filename')
    args = arguments.parse_args()
    create_v2g_output(args.uniprot, args.input_vep, args.output)