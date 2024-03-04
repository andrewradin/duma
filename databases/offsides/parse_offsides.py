#!/usr/bin/python
from __future__ import print_function
import sys
sys.path.insert(1,"../../web1")
from path_helper import PathHelper

def message(*objs):
    print(*objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def verboseOut(*objs):
    if verbose:
        print(*objs, file=sys.stderr)

def printOut(*objs):
    print(*objs, file=sys.stdout)

if __name__=='__main__':
    #=================================================
    # Read in the arguments/define options
    #=================================================
    import argparse
    arguments = argparse.ArgumentParser(description="Parse Offsides tsv file")
    
    arguments.add_argument("--meddra-map", help="umls_to_meddra.tsv")
    arguments.add_argument("infile", help="3003377s-offsides.tsv")
    
    args = arguments.parse_args()

    # read in mappings for UMLS ids
    umls2meddra = {}
    for line in open(args.meddra_map):
        rec = line.strip('\n').split('\t')
        if not rec:
            continue
        umls2meddra[rec[0]] = rec[1]
    # for recording bad mappings
    bad_umls = set()
    used_meddra = {}
    used_meddra_shown = set()
    def get_meddra(umls):
        if umls not in umls2meddra:
            bad_umls.add(umls)
            return None
        else:
            result = umls2meddra[umls]
            if result in used_meddra:
                if used_meddra[result] != umls:
                    err = '%s needed for %s but already used for %s' % (
                                    result,umls,used_meddra[result],
                                    )
                    if err not in used_meddra_shown:
                        warning(err)
                        used_meddra_shown.add(err)
            else:
                used_meddra[result] = umls
            return result

    # scan tsv file and output ADR records
    inp = open(args.infile)
    or_fn='adr.offsides.odds_ratio.tsv'
    outp = open(or_fn,'w')
    header = None
    # From Aaron:
    # they didn't really do any sort of multiple hypothesis testing correction,
    # and with the info they gave in the paper, I can't properly either;
    # in place of that, I'll just correct for the number they reported, which
    # means this will still be too liberal, but I think it's better than what
    # they had
    thresh = 0.05/438801
    skipped = 0
    wrote = 0
    for line in inp:
        rec = line.strip('\n').split('\t')
        rec = [x.strip('"') for x in rec]
        if not header:
            header = rec
            stitch_idx = header.index('stitch_id')
            umls_idx = header.index('umls_id')
            pvalue_idx = header.index('pvalue')
            tstat_idx = header.index('t_statistic')
            rr_idx = header.index('rr')
            ev_idx = header.index('event')
            outp.write("stitch_id\tmeddra_id\todds_ratio\tpvalue\n")
            continue
        if float(rec[pvalue_idx]) < thresh:
            umls = rec[umls_idx]
            meddra = get_meddra(umls)
            if not meddra:
                skipped += 1
                warning(umls
                        ,'not in meddra map; skipping'
                        ,rec[stitch_idx]
                        ,rec[ev_idx]
                        )
            else:
                wrote += 1
                outp.write('\t'.join([
                        rec[stitch_idx],
                        meddra,
                        rec[rr_idx],
                        rec[pvalue_idx],
                        ])+'\n')
    print('wrote',wrote,'skipped',skipped,'('+str(len(bad_umls))+' unique)')
    print("found",len(used_meddra_shown),"distinct key clashes")
