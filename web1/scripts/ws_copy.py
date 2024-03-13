#!/usr/bin/env python3

# on platform, this should be run like:
# sudo -u www-data migrate_ML_output.py

from __future__ import print_function
import sys
import os
import re
import shutil

# Make sure we can find PathHelper
try:
    import path_helper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+'/..')

import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()

from path_helper import PathHelper,make_directory

def copy_indications(from_ws,to_ws):
    from browse.models import WsAnnotation
    s = from_ws.get_wsa_id_set('related')
    updated = 0
    for from_wsa in WsAnnotation.objects.filter(pk__in=s):
        src_info = 'source wsa_id %d (%s - %s)' % (
                        from_wsa.id,
                        from_wsa.agent.canonical,
                        from_wsa.indication_label(),
                        )
        try:
            to_wsa = to_ws.wsannotation_set.get(agent=from_wsa.agent)
        except WsAnnotation.DoesNotExist:
            print('no match for',src_info)
            continue
        try:
            to_wsa.update_indication(
                        from_wsa.indication,
                        from_wsa.demerits(),
                        from_wsa.marked_by,
                        from_wsa.marked_because,
                        from_wsa.indication_href,
                        )
        except ValueError as ex:
            print('error from',src_info,'-',ex)
            continue
        updated += 1
    print('updated',updated,'of',len(s),'records')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description="copy things between workspaces",
            )
    parser.add_argument('from_ws')
    parser.add_argument('to_ws')
    parser.add_argument('item',
            nargs='+',
            # XXX build from locals()? add help?
            #choices=[
            #        'indications',
            #        ]
            )
    args = parser.parse_args()
    from browse.models import Workspace
    from_ws = Workspace.objects.get(pk=args.from_ws)
    to_ws = Workspace.objects.get(pk=args.to_ws)
    for item in args.item:
        # XXX maybe pass 'args', so we can have item-specific options?
        locals()['copy_'+item](from_ws,to_ws)

