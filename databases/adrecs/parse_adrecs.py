#!/usr/bin/python
from __future__ import print_function
import os, django, sys, xmltodict, re, argparse
import collections
sys.path.insert(1,"../../web1")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from path_helper import PathHelper
from algorithms.exit_codes import ExitCoder

# created 7.Mar.2016 - Aaron C Daugherty - twoXAR

# Parse ADReCS

# TO DO:
# A few subroutines to print out as I want

def message(*objs):
    print(*objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def verboseOut(*objs):
    if verbose:
        print(*objs, file=sys.stderr)

def useage(*objs):
    print("USEAGE: ", *objs, file=sys.stderr)

def printOut(*objs):
    print(*objs, file=sys.stdout)

def contains_digits(s):
    digits = re.compile('\d')
    return bool(digits.search(s))

def split_on_nondecimal_or_perc(s):
    return filter(None, re.split(r'[^\d.%]+', s))

def clean_freq(freq):
    # I'm having some issues with unexpected characters (specifically where I expect hyphens)
    # so I'm trying a new approach of splitting on anything that isn't a number, decimal or %
    # To make this work, I first remove all spaces by splitting (defaults to any white space)
    # and joing back together
    freqs = split_on_nondecimal_or_perc("".join(freq.split()))
    if len(freqs) == 2 and contains_digits(freqs[0]) and contains_digits(freqs[1]):
        por1 = str_to_portion(freqs[0])
        por2 = str_to_portion(freqs[1])
        portion = (por1 + por2) / 2.0
    elif len(freqs) == 1:
        portion = str_to_portion(freqs[0])
    else:
        warning("split failed. Len = ", len(freqs), "__".join(freqs), ". original:", freq)
    return portion

def str_to_portion(freq):
    if freq.endswith('%'):
        return strip_unwanted_chars(freq)/100.0
    else:
        return strip_unwanted_chars(freq)

def strip_unwanted_chars(decimal_str):
    non_decimal = re.compile(r'[^\d.]+')
    try:
        return float(non_decimal.sub('', decimal_str).strip())
    except:
        warning(decimal_str, " ____ ", non_decimal.sub('', decimal_str))

def write_attributes_dd(outData, f):
    for k in outData.keys():
        for k2, v in outData[k].items():
            if type(v) == set:
                to_write = [ i for i in list(v) if ok_to_write(i)]
                if len(to_write) > 0:
                    for value in to_write:
                        f.write("\t".join([k, k2, value]) + "\n")
            elif ok_to_write(v):
                f.write("\t".join([k, k2, v]) + "\n")

def ok_to_write(s):
    if s is not None and s != 'None' and s != 'Not Available':
        return True
    return False

if __name__=='__main__':
    #=================================================
    # Read in the arguments/define options
    #=================================================
    exitCoder = ExitCoder()
    
    arguments = argparse.ArgumentParser(description="Parse ADReCS XML files")
    
    arguments.add_argument("--drugs", help="ADReCS_Drug_info.xml")
    
    arguments.add_argument("--adrs", help="ADReCS_ADR_info.xml")
    
    args = arguments.parse_args()

    # return usage information if no argvs given

    if not args.drugs or not args.adrs:
        arguments.print_help()
        sys.exit(exitCoder.encode('usageError'))
    
    adrecs_adr_atr_file = 'tox.adrecs.full.tsv'
    adrecs_drug_atr_file = 'create.adrecs.full.tsv'
    adrecs_freq_file = 'adr.adrecs.portion.tsv'
    
    with open(args.drugs) as f:
        # this yields a list of orderedDicts
        drugs = xmltodict.parse(f.read())[u'ADReCS_BADD'][u'Drug_BADD']
    
    # this is all small enough I can just read it into memory for ease
    adr_freqs = collections.defaultdict(dict)
    # but I'll write out the attributes as we go
    with open(adrecs_drug_atr_file, 'w') as f:
        # add a header
        f.write("\t".join(['adrecs_id', 'attribute', 'value']) + "\n")
        for od in drugs:
            outData = collections.defaultdict(dict)
            drug_id = od[u'DRUG_ID'].encode('utf-8')
            name = od[u'DRUG_NAME'].encode('utf-8')
            outData[drug_id]['atc'] = set([atc.strip() for atc in od[u'ATC'].encode('utf-8').split(";")])
            # some drugs apaprently have multiple cas? I don't trust those
            try:
                outData[drug_id]['cas'] = od[u'CAS'].encode('utf-8')
            except AttributeError:
                pass
            # XXX Synonyms need to be output one per line; since these seem
            # XXX to skew towards generic anyway, just bypass output for now
            #if isinstance(od[u'DRUG_SYNONYMS'], basestring):
            #    synonyms = [od[u'DRUG_SYNONYMS'].encode('utf-8')]
            #else:
            #    synonyms = [s.encode('utf-8') for s in od[u'DRUG_SYNONYMS'][u'SYNONYM'] if s is not None]
            #outData[drug_id]['synonym'] = set(synonyms)
            
            # now write out the attributes
            f.write("\t".join([drug_id, 'canonical', name]) + "\n")
            write_attributes_dd(outData, f)
            
            # now get the actual side effects, but we'll store them in memory so we're not writing to 2 files at once
            # need to take care of the situation where there is 1 instance here
            # just load it in a list like the others are, for ease of handling
            if isinstance(od[u'ADRs'][u'ADR'], collections.OrderedDict):
                od[u'ADRs'][u'ADR'] = [od[u'ADRs'][u'ADR']]
            for sub_od in od[u'ADRs'][u'ADR']:
                #adr_term = sub_od[u'ADR_TERM'].encode('utf-8')
                adrecs_id = sub_od[u'ADRECS_ID'].encode('utf-8')
                raw_freq = sub_od[u'FREQUENCY'].encode('utf-8')
                if contains_digits(raw_freq):
                    adr_freqs[drug_id][adrecs_id] = clean_freq(raw_freq)
    
    # clear out the unecessary memory
    drugs = None
    
    with open(args.adrs) as f:
        all_adrs = xmltodict.parse(f.read())[u'ADReCS_BADD'][u'ADR_BADD']
        
    adr2meddra={}
    with open(adrecs_adr_atr_file, 'w') as f:
        # add a header
        f.write("\t".join(['adrecs_adr_id', 'attribute', 'value']) + "\n")
        for adr in all_adrs:
            outData = collections.defaultdict(dict)
            adrecs_id = adr[u'ADReCS_ID'].encode('utf-8')
            outData[adrecs_id]['adr_term'] = adr[u'ADR_TERM'].encode('utf-8')
            outData[adrecs_id]['who_art'] = adr[u'WHO_ART_CODE'].encode('utf-8')
            if u'MEDDRA_CODE' in adr.keys():
                meddra = adr[u'MEDDRA_CODE'].encode('utf-8')
                adr2meddra[adrecs_id] = meddra
                outData[adrecs_id]['meddra_code'] = meddra
            else:
                warning(adr.keys())
                sys.exit()
            if isinstance(adr[u'ADR_SYNONYMS'], basestring):
                synonyms = [adr[u'ADR_SYNONYMS'].encode('utf-8')]
            else:
                synonyms = [s.encode('utf-8') for s in adr[u'ADR_SYNONYMS'][u'SYNONYM'] if s is not None]
            outData[adrecs_id]['synonym'] = set(synonyms)
            # now write out the attributes
            write_attributes_dd(outData, f)

    # write out adrs drug pair info
    with open(adrecs_freq_file, 'w') as f:
        f.write("\t".join(['adrecs_id', 'meddra_id', 'frequency']) + "\n")
        for did in adr_freqs.keys():
            for aid,freq in adr_freqs[did].items():
                f.write("\t".join([did, adr2meddra[aid], str(freq)]) + "\n")

