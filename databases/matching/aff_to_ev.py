#!/usr/bin/env python3

import sys
import os
try:
    from path_helper import PathHelper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
    from path_helper import PathHelper

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='convert affinity values to evidence',
        )
    parser.add_argument('aff_file')
    parser.add_argument('ev_file')
    args = parser.parse_args()

    aff_type = os.path.basename(args.aff_file).split('.')[1]
    import dtk.affinity as converters
    if aff_type == 'c50':
        cvt = converters.c50_to_ev
    elif aff_type == 'ki':
        cvt = converters.ki_to_ev
    else:
        raise NotImplementedError("Unknown affinity type '%s'"%aff_type)
    from atomicwrites import atomic_write
    with atomic_write(args.ev_file, overwrite=True) as outf:
        header = None
        from dtk.files import get_file_records
        for rec in get_file_records(args.aff_file):
            if not header:
                header = rec[:2]+['evidence','direction']
                outf.write('\t'.join(header)+'\n')
                continue
            direction = rec[2]
            ev = cvt(float(rec[3]))
            if ev:
                outf.write('\t'.join(rec[:2]+[str(ev),direction])+'\n')
