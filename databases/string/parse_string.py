#!/usr/bin/env python3

import sys, argparse
from dtk.files import get_file_records
import logging
logger = logging.getLogger(__name__)

from dtk.log_setup import setupLogging
from atomicwrites import atomic_write

# created 02.Apr.2015 - Aaron C Daugherty
# A program to process the protein-protein interaction data from ConsensusPathDB; the final product being uniprot IDs and evidence, as below
# protID, protID, evidence, direction
# P04217    P04217    1    0

if __name__=='__main__':
    #=================================================
    # Read in the arguments/define options
    #=================================================

    arguments = argparse.ArgumentParser(description="This will parse a file downloaded and gunzipped from STRING")

    arguments.add_argument("-u", help="ENSP to Uniprot conversion file")

    arguments.add_argument("-i", help="9606.protein.links.v10.txt")
    arguments.add_argument("-a", "--actions", help="9606.protein.actions.v10.txt")
    arguments.add_argument("-o", '--output',  help="output tsv file")

    args = arguments.parse_args()
    setupLogging()

    # return usage information if no argvs given

    if not args.i or not args.u or not args.output:
        arguments.print_help()
        sys.exit(10)

    # Initialize a dictionary that has as a key a gene a name, and as a value a list of UniProt IDs
    uniProtConverter = {}
    missingConverter = set()
    # use a loop to read each line and pull out the info I need and store the pairs in a hash of arrays
    for info in get_file_records(args.u, keep_header=True):
        uniProtConverter.setdefault(info[1].strip(), []).append(info[0].strip())

    def uniconvert(ens):
        try:
            return uniProtConverter[ens]
        except KeyError:
            #logger.warning("Couldn't find any uniprot conversion for %s", ens)
            missingConverter.add(ens)
            return []

    directed = set()
    from tqdm import tqdm
    if args.actions:
        recs = get_file_records(args.actions, keep_header=False, parse_type='tsv', progress=True)
        for (ens_a, ens_b, mode, action, is_directional, a_is_acting, score) in recs:
            if int(score) < 900:
                continue
            lefts = uniconvert(ens_a.lstrip('9606.'))
            rights = uniconvert(ens_b.lstrip('9606.'))
            for left in lefts:
                for right in rights:
                    if is_directional != 't':
                        directed.add((left, right))
                        directed.add((right, left))
                    elif a_is_acting == 't':
                        directed.add((left, right))
    logger.info("Have %d directed interaction datas", len(directed))

    discarded = 0
    gt900 = 0

    with atomic_write(args.output, overwrite=True) as f:
        f.write("\t".join(['prot1', 'prot2', 'evidence', 'direction']) + '\n')
        for info in get_file_records(args.i, keep_header=False, progress=True):
            # pull out the relevant bits
            lefts = uniconvert(info[0].lstrip('9606.'))
            rights = uniconvert(info[1].lstrip('9606.'))
            evidence = str(float(info[2])/1000.0)
            for left in lefts:
                for right in rights:
                    if int(info[2]) >= 900:
                        gt900 += 1

                    # If we have directional information about this pair and it's only
                    # in the reverse, direction, don't include this link.
                    if (left,right) not in directed and (right,left) in directed:
                        # logger.info("Discarding %s %s %s", left, right, evidence)
                        discarded += 1
                        continue

                    f.write("\t".join([left, right, evidence, '0']) + '\n')

    if missingConverter:
        logger.warning("Missing conversion for %d/%d prots", len(missingConverter), len(uniProtConverter))
        for missing in missingConverter:
            print(missing)
    logger.info("Discarded %d / %d >=0.9 directed interactions", discarded, gt900)
