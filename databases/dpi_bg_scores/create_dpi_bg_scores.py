#!/usr/bin/env python3

import sys
sys.path.insert(1,"../../web1")

import os

if __name__ == '__main__':
    import argparse
    from dtk.prot_map import DpiMapping
    parser = argparse.ArgumentParser(
                description='test score access',
                )
    parser.add_argument("--dpi_name", default=DpiMapping.preferred)
    parser.add_argument("--dpi_min_evid", default=DpiMapping.default_evidence)

    args = parser.parse_args()

    dpi = DpiMapping(args.dpi_name)
    from dtk.files import get_file_records
    scores = {}
    header = None
    for frs in get_file_records(dpi.get_path()):
        if not header:
            header = frs
            continue
        if frs[0] not in scores:
            scores[frs[0]] = 0.0
        scores[frs[0]] += float(frs[header.index('evidence')])
    for k,v in scores.items():
        print("\t".join([k,str(v)]))
