#!/usr/bin/env python3

from dtk.files import get_file_records

def standardize_kol_names(s):
    import unidecode
    s = unidecode.unidecode(s)
    try:
        while not s[0].isalpha(): s = s[1:]
    except IndexError:
        return ''
    x = s.split(",")[0].split(":")[0]
    x = x.title()
    x = x.replace('.', '')
    if x.startswith('Dr '):
        x = " ".join(x.split()[1:])
    return x

if __name__=='__main__':
    import argparse
    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="Combine all possible CT contact files into one, keyed by NCT")
    arguments.add_argument("fc", help="facility_contacts.txt")
    arguments.add_argument("fi", help="facility_investigators.txt")
    arguments.add_argument("oo", help="overall_officials.txt")
    arguments.add_argument("rp", help="responsible_parties.txt")
    arguments.add_argument("f", help="facilities.txt")
    args = arguments.parse_args()

    print("\t".join(['nct_id', 'name', 'affiliation', 'role', 'contact', 'source_file']))

    facilitiy_names = {}
    # id|nct_id|status|name|city|state|zip|country
    for frs in get_file_records(args.f, parse_type='psv', keep_header=False):
        facilitiy_names[frs[0]] = frs[3]

    # id|nct_id|facility_id|contact_type|name|email|phone
    for frs in get_file_records(args.fc, parse_type='psv', keep_header=False):
        name = standardize_kol_names(frs[4])
        if name:
            print("\t".join([frs[1],
                         name,
                         facilitiy_names.get(frs[2], 'None'),
                         frs[3],
                         "|".join([frs[5], frs[6]]),
                         'facility_contacts'
                        ]))
    # id|nct_id|facility_id|role|name
    for frs in get_file_records(args.fi, parse_type='psv', keep_header=False):
        name = standardize_kol_names(frs[4])
        if name:
            print("\t".join([frs[1],
                         name,
                         facilitiy_names.get(frs[2], 'None'),
                         frs[3],
                         "None",
                         'facility_investigators'
                        ]))
    # id|nct_id|role|name|affiliation
    for frs in get_file_records(args.oo, parse_type='psv', keep_header=False):
        name = standardize_kol_names(frs[3])
        if name:
            print("\t".join([frs[1],
                         name,
                         frs[4],
                         frs[2],
                         "None",
                         'overall_officials'
                        ]))
    # id|nct_id|responsible_party_type|name|title|organization|affiliation
    for frs in get_file_records(args.rp, parse_type='psv', keep_header=False):
        name = standardize_kol_names(frs[3])
        if name:
            print("\t".join([frs[1],
                         name,
                         frs[6],
                         frs[2],
                         "None",
                         'responsible_parties'
                        ]))
