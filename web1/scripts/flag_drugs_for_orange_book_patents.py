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

from flagging.utils import FlaggerBase

class PatentFlagger(FlaggerBase):
    def __init__(self,**kwargs):
        super(PatentFlagger,self).__init__(kwargs)
        assert not kwargs
    def flag_drugs(self):
        wsa_ids = self.get_target_wsa_ids()
        self.create_flag_set('OB Patents')
        from dtk.orange_book import OrangeBook
        ob = OrangeBook()
        unmatched = 0
        no_patents = 0
        found = 0
        from dtk.url import google_patent_url
        from browse.models import WsAnnotation
        for wsa in WsAnnotation.objects.filter(id__in=wsa_ids):
            result = ob.get_ndas_for_names([wsa.agent.canonical])
            if not result:
                unmatched += 1
                continue
            pats = ob.get_patents_for_ndas(result)
            if not pats:
                no_patents += 1
                continue
            found += 1
            for pat in pats:
                for pat_no in pat.pat_list:
                    self.create_flag(
                            wsa_id=wsa.id,
                            category='OB Patent',
                            detail=pat.text,
                            href=google_patent_url('US'+pat_no)
                            )
        print(unmatched,'unmatched')
        print(no_patents,'without patents')
        print(found,'with patents')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description="flag drugs for Patents via Orange Book",
            )
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--count',type=int,default=200)
    parser.add_argument('ws_id',type=int)
    parser.add_argument('job_id',type=int)
    parser.add_argument('score')
    args = parser.parse_args()

    from runner.models import Process
    err = Process.prod_user_error()
    if err:
        raise RuntimeError(err)

    flagger = PatentFlagger(
                ws_id=args.ws_id,
                job_id=args.job_id,
                score=args.score,
                start=args.start,
                count=args.count,
                )
    flagger.flag_drugs()
