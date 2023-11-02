#!/usr/bin/env python

import sys
sys.path.insert(1,"..")

#import os
#os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")

#import django
#django.setup()

# This program scans all the license.py files and builds a single json
# file used by the website to display credit information.  This works
# around the platform not having direct access to the databases directory.
#
# Although this program makes no assumptions about what's in the license.py
# file, this is as good a place as any to document conventions:
# - credit_list is a list of dicts, one per credit entry.  Each dict may
#   contain 'author', 'title', 'ref', and 'link' keys.  For an article
#   reference, 'ref' describes the publication.  For just a license entry,
#   'title' should be the name of the database and 'link' should link to
#   the license; optionally, 'author' can be the database publisher.
# - a 'license' global can also link to the overall license; if this link
#   is not clearly associated with the database, 'license_info' can be
#   a page tying the db and license together.
# Additional info can be added as comments.  All this is subkect to change.

def build_credits_file(fn):
    from path_helper import PathHelper
    db_root = PathHelper.repos_root+'databases/'
    from dtk.files import scan_dir,is_dir
    import os
    db_credits = []
    for subdir in scan_dir(db_root,filters=[is_dir]):
        target = os.path.join(subdir,'license.py')
        data={}
        if os.path.exists(target):
            exec(open(target).read(),globals(),data)
        for credit in data.get('credit_list',[]):
            db_credits.append(credit)
    result = {
            'db_credits':db_credits,
            }
    with open(fn,'w') as f:
        import json
        # pretty-print for more informative git diffs
        json.dump(result,f,sort_keys=True,indent=4)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description="build credits json file",
            )
    args = parser.parse_args()

    build_credits_file('credits.json')
