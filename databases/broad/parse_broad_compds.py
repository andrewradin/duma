#!/usr/bin/env python
import sys
try:
    from dtk.files import get_file_records
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
    from dtk.files import get_file_records
from dtk.mol_struct import getCanonSmilesFromSmiles

try:
    from parse_grasp import make_uniprot_map
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../grasp")
    from parse_grasp import make_uniprot_map

def parse_ids(raw):
    return "-".join(raw.split('-')[0:2])


if __name__=='__main__':
    import argparse
    #=================================================
    # Read in the arguments/define options
    #=================================================

    arguments = argparse.ArgumentParser(description="Parse Broad repurposing hub compound info")

    arguments.add_argument("c", help="repurposing_samples_20170327.txt")
    arguments.add_argument("s", help="repurposing_drugs_20170327.txt")
    arguments.add_argument("u", help="HUMAN_9606_Uniprot_data.tsv")

    args = arguments.parse_args()
    
    conv_d = make_uniprot_map(args.u, 'Gene_Name')
    cmpd_data = {}
    can_to_id = {}
    header = None
    dpi={}
    for frs in get_file_records(args.s, parse_type = 'tsv'):
        if frs[0].startswith('!'):
            continue
        if not header:
            header = frs
            continue
        can = frs[header.index('pert_iname')].strip('"')
        targs = frs[header.index('target')].split('|')
        if targs and targs[0]:
            dpi[can] = set([u for gn in targs for u in conv_d.get(gn,[])])
    header = None
    for frs in get_file_records(args.c, parse_type = 'tsv'):
        if frs[0].startswith('!'):
            continue
        if not header:
            header = frs
            continue
        id = parse_ids(frs[header.index('broad_id')])
        can = frs[header.index('pert_iname')].strip('"')
        # handle a special case encoding error:
        # The canonical string for this drug "ibuproxam-X-cyclodextrin"
        # has a hex A4 in the X position in the download file. This isn't
        # valid utf8. Poking around on the internet indicates this should
        # probably be a Greek Beta (e.g. patent EP0268215A1) but there's
        # no encoding I could find that makes that work. Since this is the
        # only non-ascii character in the file, assume it was mangled by
        # Broad somehow, and patch in a Beta here.
        if id == 'BRD-A40940854':
            can = can.replace('\xa4',u'\u03B2'.encode('utf8'))
        if id in cmpd_data:
            if can != cmpd_data[id]['canonical']:
                current = cmpd_data[id]['canonical']
                if can in dpi and current not in dpi:
                    # take the one that has DPI data
                    # otherwise leave the current term
                    other = cmpd_data[id]['canonical']
                    cmpd_data[id]['canonical'] = can
                else:
                    other = can
                    can = cmpd_data[id]['canonical']
                cmpd_data[id]['synonym'].add(other)
### I removed this b/c I was modifying the mwt (removing the commas)
### and had shown that this was never an issue
#            for x,y in [
#                         ('InChIKey', 'inchi_key'),
#                         ('pubchem_cid', 'pubchem_cid'),
#                         ('expected_mass', 'full_mwt')
#                       ]:
#                if frs[header.index(x)].strip('"') != cmpd_data[id][y]:
#                    sys.stderr.write(" ".join(['mismatch!',
#                                           id,
#                                           x,y,
#                                           frs[header.index(x)].strip('"'),
#                                           cmpd_data[id][y]
#                                         ]) + "\n")
            cmpd_data[id]['synonym'].add(frs[header.index('vendor_name')].strip('"'))
        else:
            cmpd_data[id] = {
                'canonical':can,
                'inchi_key':frs[header.index('InChIKey')],
                'pubchem_cid':frs[header.index('pubchem_cid')],
                'synonym':set([frs[header.index('vendor_name')].strip('"')]),
                'full_mwt':frs[header.index('expected_mass')].strip('"').replace(',','')
            }
            smiles = getCanonSmilesFromSmiles(frs[header.index('smiles')])
            if smiles:
                cmpd_data[id]['smiles_code'] = smiles
        if can not in can_to_id:
            can_to_id[can] = set()
        can_to_id[can].add(id)

    with open('dpi', 'w') as f:
        f.write('\t'.join(['broad_id', 'uniprot_id', 'evidence', 'direction']) + "\n")
        for can, unis in dpi.iteritems():
# there are a few DPI with what I think is the wrong name and thus don't have a 1:1 ID mapping
# skipping those for now
            for id in can_to_id.get(can,[]):
                for u in unis:
                    f.write("\t".join([id, u,  '1.0', '0']) + "\n")
    with open('broad', 'w') as f:
        f.write('\t'.join(['broad_id', 'attribute', 'value']) + "\n")
        for id, d in cmpd_data.iteritems():
            for k in 'canonical inchi_key pubchem_cid full_mwt'.split():
                if d[k] or k == 'canonical':
                    f.write('\t'.join([id, k, d[k]]) + "\n")
            for s in d['synonym']:
                f.write('\t'.join([id, 'synonym', s]) + "\n")
            if 'smiles_code' in d:
                f.write('\t'.join([id, 'smiles_code', d['smiles_code']]) + "\n")
