#!/usr/bin/env python

import sys

def make_sting_2_uniprot_map(file):
    conv_d = {}
    with open(file, 'r') as f:
        for l in f:
            fields = l.rstrip("\n").split("\t")
            if fields[1] == 'STRING':
                try:
                    conv_d[fields[2]].add(fields[0])
                except KeyError:
                    conv_d[fields[2]] = set()
                    conv_d[fields[2]].add(fields[0])
        return conv_d
def update_converter(k, v, d):
    try:
        d[k].add(v)
    except KeyError:
        d[k] = set([v])
    return d
def update_create_data(d,f):
    if f[1] in d:
        ### We use the first one unless the later one has a PCID corresponding to the stitch ID
        if d[f[1]]['pubchem_cid'] != f[2] and f[2] == f[1][4:].lstrip('0'):
            d[f[1]]['pubchem_cid'] = f[2]
            d[f[1]]['inchi_key'] = f[3]
    else:
        d[f[1]]['inchi_key'] = f[3]
        d[f[1]]['pubchem_cid'] = f[2]
    return d

# XXX I went in and by hand identified which IDs overlapped with ours.
# XXX They don't do a good job at all with a canonical source term, hence the startswith.
# XXX They have bindingDb listed, but the IDs aren't right, so I didn't use it
def check_source(source, val):
    # XXX For now, don't return keys that would use the same property name as
    # XXX the native key of another collection. Eventually, we might define
    # XXX a new prefix like m_, but for externally matched collection keys.
    #if "DrugBank" in source.split() and val.startswith('DB'):
    #    return "DrugBank"
    if 'KEGG' in source.split() and val.startswith('C'):
        return 'KEGG'
    #if "ChEMBL" in source.split() and val.startswith('CHEMBL'):
    #    return "ChEMBL"
    return False

if __name__=='__main__':
    import argparse
    from collections import defaultdict
    try:
        from dtk.mol_struct import getCanonSmilesFromSmiles
    except ImportError:
        sys.path.insert(1, (sys.path[0] or '.')+'/../../web1')
        from dtk.mol_struct import getCanonSmilesFromSmiles
    import time
    import subprocess

    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="Parse Stitch data")
    arguments.add_argument("dpi", help="9606.protein_chemical.links.v5.0.tsv.gz")
    arguments.add_argument("chem", help="chemicals.v5.0.tsv.gz")
    arguments.add_argument("inchi", help="chemicals.inchikeys.v5.0.tsv.gz")
    arguments.add_argument("match", help="chemical.aliases.v5.0.tsv.gz")
    arguments.add_argument("uni_converter", help="HUMAN_9606_Uniprot_data.tsv")
    args = arguments.parse_args()
    ##### INPUTS AND OUTPUTS AND SETTINGS #####
    attr_out_file = "stitch.attr.tsv"
    dpi_out_file = "stitch.dpi.tsv"

    proper_source_name = {'DrugBank': 'drugbank_id'
            , 'KEGG': 'kegg'
            , 'ChEMBL': 'chembl_id'
            }

    # start with DPI
    ts_all = time.time()
    conv_d = make_sting_2_uniprot_map(args.uni_converter)
    print ("converter loaded: ", time.time()-ts_all)

    ts = time.time()
    dpi_data = defaultdict(dict)
    conv_file = 'conv.tmp'
    with open(conv_file, 'w') as f:
        f.write("\n".join(conv_d.keys()) + "\n")
    p = subprocess.Popen(['zgrep', '-F', '-f', conv_file, args.dpi], stdout = subprocess.PIPE)
    for line in p.stdout:
        fields = line.rstrip("\n").split("\t")
        for x in conv_d[fields[1]]:
            try:
                dpi_data[fields[0]][x].append("0."+fields[2])
            except KeyError:
                dpi_data[fields[0]][x] = ["0."+fields[2]]

    print ("DPI loaded: ", time.time()-ts)

    ts = time.time()
    dpi_drugs_file = 'dpi.tmp'
    with open(dpi_drugs_file, 'w') as f:
        f.write("\n".join(dpi_data.keys()) + "\n")
    p = subprocess.Popen(['zgrep', '-F', '-f', dpi_drugs_file, args.match], stdout = subprocess.PIPE)
    # The problem is that some DPI are keyed with the flat ID, and we want to use the stereo
    flat_2_stero = {}
    stereo_2_flat = {}
    create_data = defaultdict(dict)
    other_ids = defaultdict(dict)
    for line in p.stdout:
        fields = line.rstrip("\n").split("\t")
        flat_2_stero = update_converter(fields[0], fields[1],  flat_2_stero)
        stereo_2_flat = update_converter(fields[1], fields[0],  stereo_2_flat)
        source_k = check_source(fields[3], fields[2])
        if source_k:
            other_ids[fields[1]][source_k] = fields[2]
    print ("aliases loaded: ", time.time()-ts)

    # I'll repeat the process with the Inchi file just in case there are unique drugs in each
    # thereby keeping more for DPI
    ts = time.time()
    p = subprocess.Popen(['zgrep', '-F', '-f', dpi_drugs_file, args.inchi], stdout = subprocess.PIPE)
    for line in p.stdout:
        fields = line.rstrip("\n").split("\t")
        flat_2_stero = update_converter(fields[0], fields[1],  flat_2_stero)
        stereo_2_flat = update_converter(fields[1], fields[0],  stereo_2_flat)
        create_data = update_create_data(create_data, fields)
    print ("inchi loaded: ", time.time()-ts)

    # Report the DPI files and clear up that memory
    ts = time.time()
    ids_seen = set()
    with open(dpi_out_file, 'w') as f:
        f.write("\t".join(['stitch_id', 'uniprot_id', 'evidence', 'direction']) + "\n")
        for k in dpi_data.keys():
            if k.startswith('CIDm'):
                try:
                    ids = flat_2_stero[k]
                except KeyError:
                    continue
            else:
                ids = set([k])
            for u in dpi_data[k].keys():
                for e in dpi_data[k][u]:
                    for id in ids - ids_seen:
                        f.write("\t".join([id,u,e,'0']) + "\n")
            ids_seen.update(ids)
    print ("DPI reported: ", time.time()-ts)
    # now clean up
    del dpi_data

    # Now finish up with the attributes
    ts = time.time()
    # reset this
    ids_seen = set()
    with open(attr_out_file, 'w') as f:
        f.write("\t".join(["stitch_id", "attribute", "value"]) + "\n")
        p = subprocess.Popen(['zgrep', '-F', '-f', dpi_drugs_file, args.chem], stdout = subprocess.PIPE)
        for line in p.stdout:
            fields = line.rstrip("\n").split("\t")
            if k.startswith('CIDm') and fields[0] in flat_2_stero:
                ids = flat_2_stero[fields[0]]
                stitch_flat_id = [k]
            else:
                ids = set([fields[0]])
                try:
                    stitch_flat_id = stereo_2_flat[fields[0]]
                except KeyError:
                    stitch_flat_id = []
            for id in ids - ids_seen:
                if len(fields[1]) > 256:
                    print('truncating canonical name',fields[1])
                    fields[1] = fields[1][:256]
                f.write("\t".join([id, 'canonical', fields[1]]) + "\n")
                for x in stitch_flat_id:
                    f.write("\t".join([id, 'stitch_flat_id', x]) + "\n")
                for a,v in create_data[id].items():
                    f.write("\t".join([id, a, v]) + "\n")
                smiles = getCanonSmilesFromSmiles(fields[3])
                if smiles:
                    f.write("\t".join([id, 'smiles_code', smiles]) + "\n")
                f.write("\t".join([id, 'full_mwt', fields[2]]) + "\n")
                for s in other_ids[id].keys():
                    f.write("\t".join([id, proper_source_name[s], other_ids[id][s]]) + "\n")
            ids_seen.update(ids)
            # we also don't want to double report data just b/c the flat and stereo ID are listed in the same file
            ids_seen.update(stitch_flat_id)
    print ("Attr reported: ", time.time()-ts)
    print ("Total run time: ", time.time()-ts_all)
