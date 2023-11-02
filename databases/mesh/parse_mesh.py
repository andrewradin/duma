#!/usr/bin/env python3

# The MeSH data is freely available for download from
# https://www.nlm.nih.gov/databases/download/mesh.html
# It consists of 3 ascii text files [cdq]YYYY.bin. The 'q' file seems to
# hold metadata terms. The 'c' file holds mostly compound records, which
# might be useful for drugname matching.
#
# The useful information seems to be in the 'd' file, which holds records
# that appear as nodes in the MeSH hierarchy (and have MN fields that hold
# their tree number).  Disease ('condition') records have a tree number
# starting with 'C'.
#
# Field descriptions:
#   https://www.nlm.nih.gov/mesh/xmlconvert_ascii.html
#   https://www.nlm.nih.gov/mesh/dtype.html
# Categories & browser: https://meshb.nlm.nih.gov/treeView

def get_consecutive_mesh_pairs(fh):
    '''Extract one MeSH entry as name-value pairs'''
    import re
    for line in fh:
        if line == '\n':
            break
        m = re.match(r'([A-Z1-9_ ]+) = (.*)\n',line)
        if not m:
            print(repr(line))
        assert m
        yield (m.group(1),m.group(2))

def get_mesh_mms(fh):
    '''Extract entire MeSH file as MultiMaps, one per entry'''
    from dtk.data import MultiMap
    for line in fh:
        assert line == "*NEWRECORD\n"
        yield MultiMap(get_consecutive_mesh_pairs(fh))

class MeshEntry:
    '''Object wrapper for raw MeSH multimap'''
    def __init__(self,mm):
        fwd = mm.fwd_map()
        s = fwd['MH']
        if len(s) != 1:
            print(fwd)
        assert len(s) == 1
        # record preferred name and synonyms
        self.MH = next(iter(s))
        self.ENTRIES = set()
        for entry in fwd.get('ENTRY',set()) | fwd.get('PRINT ENTRY',set()):
            # As per https://www.nlm.nih.gov/mesh/dtype.html, these fields
            # may have a pipe-delimited multi-part format. If so, the last
            # part is a string of single-letter codes identifying the other
            # parts. For now we're just interested in the term itself, coded
            # by 'a'.
            parts = entry.split('|')
            if len(parts) == 1:
                self.ENTRIES.add(parts[0])
            else:
                idx = parts[-1].index('a')
                self.ENTRIES.add(parts[idx])
        self.TREE_CODES = fwd.get('MN',set())
        self.SEMANTIC_TYPES = fwd.get('ST',set())
    def categorize(self,categories):
        matches = set()
        for code in self.TREE_CODES:
            for k,s in categories.items():
                if code.startswith(k+'.'):
                    matches |= s
        return ','.join(matches)

def get_mesh_entries(fh):
    '''Extract interesting MeSH entries as objects'''
    for mm in get_mesh_mms(fh):
        if any(
                any(x.startswith(y) for y in ('C','F03','F01'))
                for x in mm.fwd_map().get('MN',[])
                ):
            yield MeshEntry(mm)
        
def suggest_categories():
    '''Output suggested category names.'''
    top_level_terms = set()
    with open('d2020.bin') as fh:
        for me in get_mesh_entries(fh):
            for code in me.TREE_CODES:
                if '.' not in code:
                    top_level_terms.add((code,me.MH))
    for code,term in sorted(top_level_terms):
        print(code,term)

def parse_mesh():
    '''Print all disease names and synonyms in MeSH'''
    with open('d2020.bin') as fh:
        with open('mesh_disease_names.tsv','w') as out:
            for me in get_mesh_entries(fh):
                out.write('\t'.join(
                        [me.MH]+sorted(me.ENTRIES)
                        )+'\n')

def parse_categories(fn):
    result = {}
    from dtk.files import get_file_records
    for codes,label in get_file_records(fn,keep_header=None):
        for code in codes.split(','):
            if code.startswith('?'):
                code = code[1:]
            if not code:
                continue
            result.setdefault(code,set()).add(label)
    return result

def categorize_mesh(categories, in_fn, out_fn):
    '''Output types, terms and synonyms in MeSH.

    The first output field is a classification code, followed by the
    primary term, followed by any number of synonyms.
    '''
    # For now, we're interested in separating terms into disease leaf
    # names and other terms. The other terms are included because they
    # allow better assessment of how well we match other sources, which
    # may themselves have non-leaf terms.
    #
    # We want to group disease leaf terms into one or more categories
    # to roll up counts for different disease areas.
    #
    # Ideally, we'd be able to easily extract all this from MeSH tree
    # codes:
    # - codes with no children would be leaf nodes
    # - the first leading prefix of the codes would be the disease category
    # Unfortunately, it's not that simple.
    #
    # One problem is that some diseases we'd like to consider as leaves
    # have children in the MeSH tree. An example is C05.550.114.154
    # (Rheumatoid Arthritis) which has several children, but most things
    # aren't coded down to that level.
    # XXX For now, treat anything as a leaf that is an actual leaf, or
    # XXX one above an actual leaf.
    #
    # Another problem is that diseases are cross-categorized in multiple
    # ways in MeSH, and some are more appropriate than others as our
    # disease category terms, but there's no structural indication within
    # MeSH itself as to which is which.
    # XXX For now, have a manually-curated and coded list of disease
    # XXX categories. We want on the order of 30 categories, and manually
    # XXX coding a possible list from a BD spreadsheet revealed lots of
    # XXX barriers to automation, so a manual approach seems best for now.
    #
    # The classification code for each term will be one of:
    # - a list of category names the code falls under
    # - a marker for a disease leaf code that isn't categorized
    # - a marker for a non-leaf term
    with open(in_fn) as fh:
        all_entries = list(get_mesh_entries(fh))
    if True:
        # for every terminal code encountered, remember all codes
        # 'back_gen' generations above it as non-leaves.
        back_gen = 3
        ancestor_codes = set()
        for me in all_entries:
            for code in me.TREE_CODES:
                parts = code.split('.')
                for i in range(len(parts)-back_gen):
                    ancestor_codes.add('.'.join(parts[:i+1]))
    else:
        # an alternative approach based on excluding nodes based
        # on the number of children
        from collections import Counter
        direct_child_count = Counter()
        for me in all_entries:
            for code in me.TREE_CODES:
                parts = code.split('.')
                if len(parts) > 1:
                    parent = '.'.join(parts[:-1])
                    direct_child_count[parent] += 1
        ancestor_codes = set(
                k
                for k,v in direct_child_count.items()
                if v > 1
                )
    from atomicwrites import atomic_write
    with atomic_write(out_fn,overwrite=True) as out:
        for me in all_entries:
            leaf = not all(x in ancestor_codes for x in me.TREE_CODES)
            if leaf:
                category = me.categorize(categories)
                if not category:
                    category = 'Uncategorized'
            else:
                category = 'Non-disease'
            out.write('\t'.join(
                    [category,me.MH]+sorted(me.ENTRIES)
                    )+'\n')

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='''\
''',
            )
    parser.add_argument('--suggest-categories',action='store_true')
    parser.add_argument('--categories')
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    if args.suggest_categories:
        suggest_categories()
    elif args.categories:
        categorize_mesh(
                parse_categories(args.categories),
                args.input,
                args.output,
                )
    else:
        parse_mesh()

