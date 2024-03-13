#!/usr/bin/env python3

# on platform, this should be run like:
# sudo -u www-data migrate_ML_output.py

from __future__ import print_function
import sys
import os
import re
import shutil
import six

# Make sure we can find PathHelper
try:
    import path_helper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+'/..')

import os
import django

if not "DJANGO_SETTINGS_MODULE" in os.environ:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    django.setup()

from path_helper import PathHelper,make_directory

class Migrator:
    def __init__(self):
        pass
    def run(self):
        # scan properties table; find the first property with each uni
        spelling2ids = {}
        from drugs.models import Prop
        for prop in Prop.objects.all().order_by('id'):
            lname = prop.name.lower()
            spelling2ids.setdefault(lname,[]).append(prop.id)
        # for each non-unique id,
        # - change all value records to the first id encountered
        # - delete all duplicate prop records
        # for every id, force name to lower case
        for k,v in six.iteritems(spelling2ids):
            p = Prop.objects.get(pk=v[0])
            if len(v) > 1:
                print('mapping',v[1:],'to',v[0])
                p.cls().objects.fliter(prop_id__in=v[1:]).update(prop_id=v[0])
                Prop.objects.delete(pk__in=v[1:])
            if p.name != k:
                print('lcasing',p.name)
                p.name = k
                p.save()

if __name__ == '__main__':
    import argparse
    arguments = argparse.ArgumentParser(
            description="Fold together drug props that differ in case only",
            )
    args = arguments.parse_args()

    m = Migrator()
    m.run()
