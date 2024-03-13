#!/usr/bin/env python3

from __future__ import print_function
import sys
try:
    import path_helper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")
    import path_helper

import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")

import django
django.setup()

class ScoreboxCacheReaper:
    def __init__(self,ws_id):
        self.ws_id = ws_id
        from dtk.data import MultiMap
        self.mm = MultiMap([])
    def run(self):
        self.scan_latest_logs()
        self.classify_wsa_ids()
        self.remove_cache_files()
    def remove_cache_files(self):
        prefix='browse.utils.Scorebox'
        from django.core.cache import caches
        cache = caches[prefix]
        removed=0
        failed=0
        import os
        for wsa in self.to_remove:
            for key in self.mm.fwd_map().get(wsa,[]):
                fn=cache._key_to_file(prefix+'.'+key)
                try:
                    os.remove(fn)
                    removed += 1
                except OSError:
                    failed += 1
        print(removed,'keys removed;',failed,'failed')
    def classify_wsa_ids(self):
        print(len(self.mm.fwd_map()),'total wsa ids')
        from browse.models import WsAnnotation
        base_qs=WsAnnotation.objects.filter(
                pk__in=list(self.mm.fwd_map().keys()),
                ws_id=self.ws_id,
                )
        print(base_qs.count(),'wsa ids in workspace',self.ws_id)
        enum=WsAnnotation.indication_vals
        unwanted=(
                enum.UNCLASSIFIED,
                enum.INACTIVE_PREDICTION,
                )
        self.to_remove=base_qs.filter(
                indication__in=unwanted
                ).values_list('id',flat=True)
        print(len(self.to_remove),'wsa ids subject to removal')
    def add_log_file(self,fn):
        from dtk.data import MultiMap
        self.mm.union(MultiMap(self._parse_log_file(fn)))
    def _parse_log_file(self,fn):
        print('parsing',fn,'...')
        from dtk.files import get_file_lines
        for line in get_file_lines(fn,grep=['scorebox cache'],keep_header=None):
            parts = line.split()
            if len(parts) < 4 or parts[-2] != 'key' or parts[-4] != 'wsa':
                continue
            yield (int(parts[-3]),parts[-1])
    def scan_latest_logs(self,days_back=30):
        # scan log files and build index
        self.add_log_file('/var/log/django.log')
        import datetime
        start = datetime.date.today()-datetime.timedelta(days=days_back)
        start = start.strftime('%Y%m%d')
        import glob
        for fn in sorted(glob.glob('/var/log/django.log-*'),reverse=True):
            if fn.split('-')[-1] < start:
                break
            self.add_log_file(fn)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description="remove old scoreboard cache entries",
            )
    parser.add_argument('ws_id',type=int)
    args = parser.parse_args()

    scr = ScoreboxCacheReaper(args.ws_id)
    scr.run()
