#!/usr/bin/env python

# Sample output:
#P04217  A1BG_HUMAN  P04217  A1BG_HUMAN  1

import gzip, argparse

# This code matches the legacy ppi file except:
# - ordering is different
# - the existing file contains dups that this one eliminates
# ./reconstruct_drpias.py | sort >/tmp/xxx && sort -u ../../../ws/ppi/drpias_ppi.tsv | diff - /tmp/xxx

def get_uniprot_map(filename):
    result = {}
    f = open(filename)
    header=None
    for line in f:
        fields = line.rstrip().split('\t')
        if not header:
            header=fields
            continue
        l = result.setdefault(fields[1],[])
        l.append( (fields[0]) )
    return result

def get_entrez_pairs(filename):
    result = []
    f = gzip.open(filename,'rb')
    header=None
    for line in f:
        fields = line.rstrip('\n').split('\t')
        if not header:
            header=fields
            continue
        if fields[3] == 'Entrez Gene':
            result.append( (fields[1],fields[2]) )
    return result

if __name__=='__main__':
    arguments = argparse.ArgumentParser(description="Reconstruct DRPIAS PPI data.")
    
    arguments.add_argument("-u", help="uniprot_id.txt.gz")
    
    arguments.add_argument("-i", help="interaction.txt.gz")
    
    args = arguments.parse_args()
    
    # return usage information if no argvs given

    if not args.i or not args.u:
        arguments.print_help()
        sys.exit(10)
    
    unimap = get_uniprot_map(args.u)
    pairs = get_entrez_pairs(args.i)
    total = list(pairs)
    total += [(x[1],x[0]) for x in pairs]
    total += [(x[0],x[0]) for x in pairs]
    total += [(x[1],x[1]) for x in pairs]
    total.sort(key=lambda x:(int(x[0]),int(x[1])))
    already = set()
    print "\t".join(['prot1', 'prot2', 'evidence', 'direction'])
    for p in total:
        try:
            left = unimap[p[0]]
            right = unimap[p[1]]
        except KeyError:
            continue
        for l in left:
            for r in right:
                t = [l, r, '1','0']
                out = "\t".join(t)
                if out not in already:
                    already.add(out)
                    print out
