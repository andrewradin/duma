#!/usr/bin/env python

from dtk.files import get_file_records, get_file_lines
import logging
logger = logging.getLogger(__name__)

def load_target_variants(fn):
    out = []
    for rec in get_file_records(fn, keep_header=None, progress=True):
        chrnum = rec[2]
        chrpos = rec[3]
        out.append((chrnum + ':' + chrpos))
    return set(out)


def find_file_variants(file, to_find):
    import json
    out = []
    first = True
    for line in get_file_lines(file, keep_header=None, progress=False):
        first_comma = line.find(',')
        second_comma = line.find(',', first_comma+1)
        mini_line = line[:second_comma] + '}'
        first_colon = line.find(':')
        second_colon = line.find(':', first_comma+1)

        chrm = line[first_colon+2:first_comma-1]
        pos = line[second_colon+1:second_comma]

        if first:
            # This is the safer way to parse, but it's also 10x slower.
            # As a sanity check, we'll check the first line of each file to make sure they match.
            data = json.loads(line)
            safe_chrm = data['chr_id']
            safe_pos = str(data['position'])
            assert chrm == safe_chrm, f'Compare {safe_chrm} {chrm}'
            assert pos == safe_pos, f'Compare {safe_pos} {pos}'
            first = False
        
        id = chrm + ':' + pos
        if id in to_find:
            out.append((id, chrm, pos, line))
    return out

class Writer:
    """Manages a bunch of filehandles for writing things out.

    Writing SNPs such that all with same position end up in the same file.
    """
    def __init__(self, out_prefix):
        self.out_prefix = out_prefix
        self.fhs = {}

    def write(self, chrm, pos, line):
        # Could just hash the position for the grouping key, but this makes it easier to manually search, if desired.
        key = chrm + pos[-1]
        fh = self.get_fh(key)
        fh.write(line)

    def get_fh(self, key):
        if key in self.fhs:
            return self.fhs[key]
        
        import isal.igzip as gzip
        fh = gzip.open(f'{self.out_prefix}.{key}.tsv.gz', 'wt')
        self.fhs[key] = fh
        return fh
        

def find_variants(to_find, files, out_prefix, fail_file):
    from tqdm import tqdm
    from dtk.parallel import pmap
    from atomicwrites import atomic_write

    to_write = set()

    writer = Writer(out_prefix)
    from collections import defaultdict
    found_ids = defaultdict(int)
    import random
    # Shuffle files to get a better time estimate, some groups of files are much larger than others
    random.shuffle(files)
    for data in pmap(find_file_variants, files, static_args={'to_find': to_find}, progress=True, fake_mp=True):
        for id, chrm, pos, line in data:
            found_ids[id] += 1
            writer.write(chrm, pos, line)

    logger.info(f"Found {len(found_ids)} / {len(to_find)}  ({len(found_ids)*100/len(to_find):.2f}%)")
    assert len(found_ids.keys() - to_find) == 0, "Found things we weren't looking for??"

    mean_finds = sum(found_ids.values()) / len(found_ids)
    logger.info(f"Avg finds: {mean_finds}")

        
    with atomic_write(fail_file, overwrite=True) as fail:
        for missing in (to_find - found_ids.keys()):
            fail.write(missing + '\n')



def run(input_variants, otarg_parts_files, output, fail_file):
    logger.info("Loading variants")
    ordered_variants = load_target_variants(input_variants)
    logger.info("Searching")
    find_variants(ordered_variants, otarg_parts_files, output, fail_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=':inds our variants of interest in the opentargets data and writes them out')
    parser.add_argument("--input-variants", help="Input variants file with chr# and positions.")
    parser.add_argument("otarg_parts_files", nargs='+', help="Downloaded opentargets parts files")
    parser.add_argument('-o', "--output", help="Output prefix")
    parser.add_argument('-f', "--fail-file", help="Write missing variants")
    from dtk.log_setup import addLoggingArgs, setupLogging
    args = parser.parse_args()
    setupLogging()
    run(**vars(args))
