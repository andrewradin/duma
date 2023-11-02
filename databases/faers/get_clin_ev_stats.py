#!/usr/bin/env python3

import sys
try:
    import path_helper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")

# This was pulled together to be able to do some basic cross-version
# comparisons of the FAERS-style clinical data. The 'stats' operation
# shows some basic record counts. The other two operations provide
# histogram-like data that can be used for more detailed comparisons.
# XXX Some next steps might be:
# XXX - compare histogram data between two versions to detect regressions
# XXX   in counts for individual drugs or indications (This can be done
# XXX   manually by sorting drug or indi histogram data by key for two
# XXX   versions, and diffing them. A smarter diff could avoid confusion
# XXX   from partial label matches.)
# XXX - Given a regression in a particular CAS or Meddra term, have a
# XXX   tool to extract specific event numbers where the regression took
# XXX   place, and then re-map some or all of those events under both
# XXX   old and new mappings to determine:
# XXX   - the particular problematic source strings
# XXX   - the old and new mappings for those strings
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                description='FAERS data stats',
                )
    parser.add_argument('--path')
    parser.add_argument('version')
    parser.add_argument('op',nargs='+')
    args = parser.parse_args()

    from dtk.faers import ClinicalEventCounts
    cec = ClinicalEventCounts(args.version,args.path)
    ops = args.op or ['stats']
    for op in ops:
        if op == 'stats':
            print(cec.total_events(),'events')
            print(cec.total_indis(),'distinct indications')
            print(cec.total_drugs(),'distinct drugs')
        elif op == 'drugcnts':
            l = list(cec.get_drug_names_and_counts())
            l.sort(key=lambda x:-x[1])
            for label,cnt in l:
                print('%s\t%d'%(label,cnt))
        elif op == 'indicnts':
            l = list(cec.get_disease_names_and_counts())
            l.sort(key=lambda x:-x[1])
            for label,cnt in l:
                print('%s\t%d'%(label,cnt))
        else:
            raise NotImplementedError("unknown operation: '%s'"%op)
