#!/usr/bin/env python
from atomicwrites import atomic_write
import logging
logger = logging.getLogger(__name__)

def make_id_to_canon_map(uniprot_fn):
    id_to_canon = {}
    attrs = {'Alt_uniprot', 'Ensembl', 'Ensembl_TRS'}
    from dtk.files import get_file_records

    for uni, attrname, val in get_file_records(uniprot_fn, keep_header=False, progress=True):
        id_to_canon[uni] = uni
        if attrname in attrs:
            id_to_canon[val] = uni

    return id_to_canon


debug_missing = set()
debug_remapped = set()

def write_protsets(protset, id_to_canon, all_f):
    pw = protset.pw

    prots_to_use = set()
    for prot in protset:
        if 'MI' in prot:
            # micro RNA, ignore
            continue

        if prot not in id_to_canon:
            debug_missing.add(prot)
            continue
        elif id_to_canon[prot] != prot:
            if not 'ENS' in prot:
                debug_remapped.add((prot, id_to_canon[prot]))
            prots_to_use.add(id_to_canon[prot])
        else:
            prots_to_use.add(prot)


    prot_str = ','.join(sorted(prots_to_use))
    if prot_str == '':
        # We have no prots and so no children will either; end here.
        return

    full_str =  f'{pw.id}\t{prot_str}\n'
    all_f.write(full_str)
    if pw.type == 'pathway':
        for child in sorted(protset.children, key=lambda x:x.pw.id):
            write_protsets(child, id_to_canon, all_f)


def write_hierarchy(pw, out_f):
    children = pw.get_sub_pathways()
    children_ids = [x.id for x in children]
    row = [pw.id, pw.name, pw.type, '', '|'.join(children_ids), str(int(pw.hasDiagram))]
    out_f.write('\t'.join(row) + '\n')
    for child in children:
        write_hierarchy(child, out_f)

def run(args):
    from dtk.reactome import Reactome
    r = Reactome()

    logger.info("Loading in uniprot conversions")
    id_to_canon = make_id_to_canon_map(args.uniprot)

    with atomic_write(args.hier, overwrite=True) as f:
        logger.info("Generating hierarchy")
        header = ['id', 'name', 'type', 'description', 'children', 'has_diagram']
        f.write('\t'.join(header) + '\n')
        top_pws = Reactome.get_toplevel_pathways(r.db)
        for pw in sorted(top_pws, key=lambda x: x.id):
            logger.info("Making hierarchy for %s", pw)
            write_hierarchy(pw, f)

    logger.info("Generating reaction to prot map")
    reactions_to_prots = r.get_reactions_to_prots().fwd_map()



    with atomic_write(args.all, overwrite=True) as all_f:
        logger.info("Generating protsets")
        top_pws = Reactome.get_toplevel_pathways(r.db)
        for pw in sorted(top_pws, key=lambda x: x.id):
            logger.info("Making protset for %s", pw)
            from dtk.reactome import generate_protset
            ps = generate_protset(r, pw, reactions_to_prots)
            write_protsets(ps, id_to_canon, all_f)

    # Most of the entirely missing ones are prots that we tossed because they
    # don't have an associated gene or any useful identifiers.
    # Some of them are also non-human uniprots
    print(f'Missing {len(debug_missing)} entirely: {sorted(debug_missing)}')

    # Just for tracking, list which uniprots we remapped.
    print(f'Noncanonical remapped {len(debug_remapped)} : {sorted(debug_remapped)}')

    print("Many remappings and missings are expected, just outputting for debug purposes")

if __name__ == "__main__":
    from dtk.log_setup import setupLogging
    setupLogging()
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-u", "--uniprot", required=True, help="Uniprot data conversion file")
    parser.add_argument("--all", help="Output for all pathways and reactions")
    parser.add_argument("--hier", help="Output for hierarchy data")
    args = parser.parse_args()
    run(args)
