#!/usr/bin/env python

# Parses the full MSIGDB XML dump to grab a name to ID mapping.
# Some of the IDs are legacy reactome IDs, though, so they need to be
# converted via reactome.


def parse(fn, out_fn):
    with open(out_fn, 'w') as out:
        from dtk.reactome import Reactome

        rct = Reactome()
        old2new = rct.get_old_to_new_ids()
        import lxml.etree as ET
        doc = ET.parse(fn)
        for gs in doc.iter('GENESET'):
            name = gs.get('STANDARD_NAME')
            if not name.startswith('REACTOME_'):
                continue
            src = gs.get('EXACT_SOURCE')
            if src.startswith('REACT_'):
                new_src = old2new.get(src, None)
                if new_src is None:
                    print("Couldn't find ", src)
                    continue
                else:
                    src = new_src
            out.write(f'{name}\t{src}\n')


if __name__ == "__main__":
    import sys
    parse(sys.argv[1], sys.argv[2])
