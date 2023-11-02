#!/usr/bin/env python

# This script is just a testbed for flag_drugs_for_orange_book_patents.py
# and other potential uses of the FDA Orange Book

import os, sys
sys.path.insert(1,"../../web1")

def extract_ingredients():
    ingredients = set()
    header=None
    for line in open('products.txt'):
        rec = line.strip('\n').split('~')
        if not header:
            header=rec
            continue
        for name in rec[0].split(';'):
            ingredients.add(name.strip())
    return ingredients

def show_raw(args):
    ingredients = extract_ingredients()
    if args.verbose:
        for name in sorted(ingredients):
            print name
    else:
        print len(ingredients)
        print len(set([x.split()[0] for x in ingredients]))
    # Length of second list is probably closer to the truth due to salt
    # suffixes, but a lot of fiddling might be needed to get a totally
    # accurate number. The OrangeBook object below keeps both the salt
    # and raw forms, without attempting to determine which is correct,
    # so this isn't a solved problem.

def show_ob(args):
    from dtk.orange_book import OrangeBook
    ob=OrangeBook()
    all_ingredients=set()
    for rec in ob.get_products():
        all_ingredients |= rec.parsed_name
    if args.verbose:
        for name in sorted(all_ingredients):
            print name
    else:
        print len(all_ingredients)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='orange book test')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('cmd')
    args = parser.parse_args()

    locals()[args.cmd](args)
