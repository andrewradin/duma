#!/usr/bin/env python

# This script is just a testbed for flag_drugs_for_orange_book_patents.py
# and other potential uses of the FDA Orange Book

import os, django, sys
sys.path.insert(1,"../../web1")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()

from dtk.orange_book import OrangeBook

# XXX Is this worth unbundling any further?
# XXX - multiple input files in different formats can conceivably be
# XXX   used to generate compatible namedtuples (but it's hard to
# XXX   imagine much savings in specification over the col_map)
# XXX - if a file didn't have a header, the label->idx mapping would
# XXX   be irrelevant; in this case the plan could be supplied
# XXX   directly (and a separate field name list used to generate
# XXX   the type)
# XXX - if a record class already existed, the Type generation part
# XXX   would be irrelevant
# XXX - layers are:
# XXX   - minimal version takes src, type, plan
# XXX   - plan can be built from col_map and header
# XXX   - Type can be built from typename and col_map or fieldname list
# XXX - so all of the inputs above could be kwargs to the function
# XXX   (w/ src as the only positional arg), and the appropriate build
# XXX   sequence invoked automatically
def possible_alternative_convert_records(
                src,
                Type=None,
                plan=None,
                type_name=None,
                field_list=None,
                col_map=None,
                ):
    if Type is None:
        if field_list is None:
            field_list = [x[0] for x in col_map]
        from collections import namedtuple
        Type=namedtuple(type_name,field_list)
    if plan is None:
        header = src.next()
        plan = [
                (header.index(label),cast)
                for field,label,cast in col_map
                ]
    for rec in src:
        yield Type(*[
                cast(rec[idx])
                for idx,cast in plan
                ])

def name_split_demo():
    ob = OrangeBook()
    raw_names = [
            p.name
            for p in ob.get_products()
            ]
    print len(raw_names)
    split_names = []
    div = '; '
    for name in raw_names:
        if div in name:
            split_names += name.split(div)
        else:
            split_names.append(name)
    print len(split_names)
    print len(set(split_names))
    for x in set([
            x
            for x in split_names
            if ';' in x
            ]):
        print x
    from collections import Counter
    suffixes = Counter()
    for name in split_names:
        parts = name.split(' ')
        if len(parts) == 2:
            suffixes.update([parts[1]])
    print suffixes
    s = set([x
        for x in split_names
        if 'potassium' in x
        ])
    print len(s),s

def kt_fetch_demo():
    # XXX this is an example of how we could use the orange book to look up
    # XXX known treatments; we still need to go from the returned name to
    # XXX actual drug ids, but that could be done similarly to the current
    # XXX KT web scrapers (that mechanism should be packaged if it isn't
    # XXX already).
    ob = OrangeBook()
    uc_set = ob.get_use_codes_for_pattern('schizophrenia')
    nda_set = ob.get_ndas_for_uses(uc_set)
    name_set = ob.get_names_for_ndas(nda_set)
    for name in sorted(name_set):
        print name

def get_target_wsa_ids(ws_id,job_id,score,start=0,count=200):
    from browse.models import Workspace
    ws = Workspace.objects.get(pk=ws_id)
    from runner.process_info import JobInfo
    bji=JobInfo.get_bound(ws,job_id)
    cat = bji.get_data_catalog()
    ordering = cat.get_ordering(score,True)
    return [x[0] for x  in ordering[start:count]]

def patent_flag_demo():
    # load a previous selection round
    wsa_ids = get_target_wsa_ids(29,14588,'wzs')
    # count number of name matches
    ob = OrangeBook()
    from browse.models import WsAnnotation
    unmatched = 0
    no_patents = 0
    found = 0
    for wsa in WsAnnotation.objects.filter(id__in=wsa_ids):
        result = ob.get_ndas_for_names([wsa.agent.canonical])
        if not result:
            #print wsa
            unmatched += 1
            continue
        pats = ob.get_patents_for_ndas(result)
        if not pats:
            no_patents += 1
            continue
        found += 1
        if True:
            print wsa
            for pat in pats:
                print '   %s (%s)' % (pat.text,','.join(sorted(pat.pat_list)))
    print unmatched,'unmatched'
    print no_patents,'without patents'
    print found,'with patents'
    # XXX - for names that don't match exactly, we could try synonym
    # XXX   matches, or stripping both sides to a canonical form (e.g.
    # XXX   the first word), or find some linking database (rxnorm? UNII?)
    # XXX   - if this gets elaborate, the function could be moved to
    # XXX     databases/matching, where a new field is provided in the
    # XXX     upload file to hold the relevant orange book NDA# or
    # XXX     Ingredient name
    # XXX - to fit the current Flagging schema, each patent would be a
    # XXX   flag, with an href pointing to:
    # XXX   https://www.google.com/patents/US7608616
    # XXX   (or directly to USPTO, but that seems difficult for some reason)
    # XXX   - this potentially involved duplicating the text description,
    # XXX     but there's no way to store multiple links per flag

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='orange book test')
    args = parser.parse_args()

    #kt_fetch_demo()
    patent_flag_demo()
    #name_split_demo()

