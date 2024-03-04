#!/usr/bin/python
from __future__ import print_function
import os, django, sys, argparse, re, urllib, time
from collections import defaultdict
sys.path.insert(1,"../../web1")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from path_helper import PathHelper
from algorithms.exit_codes import ExitCoder

# created 7.Mar.2016 - Aaron C Daugherty
# A program that accepts a TTD id and
# written to work with http://bidd.nus.edu.sg/group/cjttd/ZFTVD2011Detail.asp?ID=TTDS00001
# and verified to work with http://database.idrb.cqu.edu.cn/TTD/ZFTVD2011Detail.asp?ID=TTDS00001
  # the newer TTD website

# A few subroutines to print out as I want

def useage(*objs):
    print("USEAGE: ", *objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)
    
def printOut(*objs):
    print(*objs, file=sys.stdout)

def download_file(file_url, file_to_save_to='file.tmp'):
    tFile = urllib.URLopener()
    tFile.retrieve(file_url, file_to_save_to)
    return file_to_save_to

def get_targets_with_potency(base_url, targetfile):
    file = download_file(base_url + targetfile, file_to_save_to = 'TTD.tmp')
    targets = []
    with open(file, 'r') as f:
        # the 2nd line makes my startswith below not work, so I'll just read it in and ignore it
        ignore = f.readline()
        ignore = f.readline()
        for l in f:
            if not l.startswith('TTD'):
                continue
            fields = l.rstrip().split("\t")
            if fields[1] == 'Target Validation':
                targets.append(fields[0].upper())
    return targets

def get_ttd_to_uniprot(base_url, unip_file):
    file = download_file(base_url + unip_file)
    return ttd_tsv_to_dict(file, 'TTD', value_ind = 3)

def get_drugnames_to_ttdIDs(base_url, syns_file):
    file = download_file(base_url + syns_file)
    return ttd_tsv_to_dict(file, 'D', value_ind = 0, key_ind = 3)

def ttd_tsv_to_dict(file, prefix, key_ind = 0, value_ind = 1):
    d = {}
    with open(file, 'r') as f:
        # the 2nd line makes my startswith below not work, so I'll just read it in and ignore it
        ignore = f.readline()
        ignore = f.readline()
        for l in f:
            if not l.startswith(prefix):
                continue
            fields = l.rstrip().split("\t")
            d[fields[key_ind].upper()] = fields[value_ind].upper()
    return d

def get_html(base_url, target_prefix, target):
    sock = urllib.urlopen(base_url + target_prefix + target)
    htmlSource = sock.read()
    sock.close()
    return htmlSource

def parse_validation_from_html(html, ttd_drug_names):
    # split on Drug Potency against Target", take 2nd entry (should only be 2)
    if "Drug Potency against Target" in html:
        splitToAllData = html.split("Drug Potency against Target")
        # split on Action against Disease Model, take first entry (of 2) # at some point we may want to use this information because it should be even better. It's just harder to interpret
        splitToEasyData = splitToAllData[1].split("Action against Disease Model")[0]
        # split on "size='2'>"
        rawLines = splitToEasyData.split("size='2'>")
        lastLine=""
        drug_dets = defaultdict(dict)
        for line in rawLines:
            # and then "</TD><TD"
            realLine = line.split("</TD><TD")[0]
            #end once you get to a line that starts with <a name=
            if realLine.startswith("<a name="):
                break
            #after that go through and ignore the lines that start with any of the following
            if not realLine.startswith("<") or not realLine.startswith(" ") :
                # deal with and print out to ic50File
                if realLine.startswith("EC50") or realLine.startswith("IC50"):
                    value, direction, relation = clean_potency_string(realLine)
                    # happens when the units are wrong
                    if value is None:
                        continue
                    if lastLine.upper() not in ttd_drug_names.keys():
                        warning("No TTD ID available for ", lastLine)
                        continue
                    ttd_id = ttd_drug_names[lastLine.upper()]
                    if ttd_id in drug_dets.keys():
                        warning("Multiple values found for ", ttd_id) 
                    drug_dets[ttd_id]['value'] = value
                    drug_dets[ttd_id]['direction'] = direction
                    drug_dets[ttd_id]['relation'] = relation
            lastLine = realLine
        return drug_dets
    else:
        return None

def clean_potency_string(stringValue):
    # look for an "=", if so split on it, otherwise look for a > and then a <, split on them.
    if "<=" in stringValue:
        flds = stringValue.split("<=")
        relation = '<='
    elif ">=" in stringValue:
        flds = stringValue.split(">=")
        relation = '>='
    elif "<" in stringValue:
        flds = stringValue.split("<")
        relation = '<'
    elif ">" in stringValue:
        flds = stringValue.split(">")
        relation = '>'
    elif "=" in stringValue:
        flds = stringValue.split("=")
        relation = '='
    else:
        warning("No equality sign found in concentration data: " + stringValue)
        return None, None, None
    
    flds[0] = flds[0].strip()
    if re.match('EC50', flds[0]):
        direction = 1
    elif re.match('IC50', flds[0]):
        direction = -1
    else:
        warning("Unexpected measure:", flds[0])
        direction = 0
    splitValue = flds[1]
    
    # Will need to sub out the nM. Should make sure it's there too
    if "uM" in splitValue:
        try:
            value = 1000 * float(re.sub('[<=> uM]', '', splitValue)) # parse and convert to nM values
        except ValueError:
            return None, None, None
    elif "nM" not in splitValue:
        warning("Unsure of units for " + stringValue)
        return None, None, None
    else:
        try:
            value = float(re.sub('[<=> nM]', '', splitValue))
        except ValueError:
            return None, None, None
    return value, direction, relation


if __name__=='__main__':
    #=================================================
    # Read in the arguments/define options
    #=================================================
    # get exit codes
    exitCoder = ExitCoder()

    arguments = argparse.ArgumentParser(description="Scrape the TRD website for the C50 data (not available for download)")

    arguments.add_argument("--url", default='http://database.idrb.cqu.edu.cn/TTD/', help="Base URL to download everything from. DEFAULT: %(default)s")
    arguments.add_argument("--target_info", default='download/TTD_download.txt', help="TTD target info file. DEFAULT: %(default)s")
    arguments.add_argument("--uniprot", default='download/TTD_uniprot_all.txt', help="TTD target to uniprot file. DEFAULT: %(default)s")
    arguments.add_argument("--syns", default='download/TTD_crossmatching.txt', help="TTD IDs to drugnames. DEFAULT: %(default)s")
    arguments.add_argument("--targets_url_prefix", default='ZFTVD2011Detail.asp?ID=', help="Prefix between the base URL and the target name. DEFAULT: %(default)s")
    arguments.add_argument("-o", help="file to print IC50 and EC50 data to")
    
    args = arguments.parse_args()
    
    # return usage information if no argvs given
    
    if not args.o:
        arguments.print_help()
        sys.exit(exitCoder.encode('usageError'))
    
    ##### INPUTS AND OUTPUTS #####
    
    targets = get_targets_with_potency(args.url, args.target_info)
    uniprot_converter = get_ttd_to_uniprot(args.url, args.uniprot)
    ttdid_drugnames = get_drugnames_to_ttdIDs(args.url, args.syns)
    
    with open(args.o, 'w') as f:
        # write header
        f.write("\t".join(['ttd_id', 'uniprot_id', 'evidence', 'direction', 'relation']) + "\n")
        for target in targets:
            if target not in uniprot_converter.keys():
                warning("No uniprot name for ", target, ". Skipping.")
                continue
            html = get_html(args.url, args.targets_url_prefix, target)
            dd = parse_validation_from_html(html, ttdid_drugnames)
            if dd is None:
                warning("No validation data for: ", target)
            else:
                f.write("\n".join(["\t".join([ttdid, uniprot_converter[target], str(dd[ttdid]['value']), str(dd[ttdid]['direction']), dd[ttdid]['relation']]) for ttdid in dd.keys()]) + "\n")
            # we dont want to crash/piss off the site
            time.sleep(2)

# these are notes for parsing other remaining data that is in a less constrained format and would therefore take more careful parsing

#  Usually in the form "EC50 = 845 nM" # where EC can be IC
# though the spaces aren't always consistent, and sometimes the = is a < or >
# I might search for IC50, EC50 or Ki and then down stream of the nM (if we don't find nM it means it's likely uM and we don't care any way). For the time being we'll ignore the > or < and consider everything just as =, just for ease (also helps because there are some that are :)
#   For each odd line, record it as the drug name
#   for each even line, split on space (maybe better to remove all symbols first)
#       search for IC50, Ki, and EC50; stand alone; hopefully only one will be found; this is now the type of measurement
#           next search for "nM"; stand alone; downstream of the measurement type##
#               then take the string immediately upstream of that
#                   make sure it's only digits; this is the value
#           if that fails, look for nM preceded by a digit, and do the above, but after removing the nM

