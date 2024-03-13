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


class NoTargetsFlagger(FlaggerBase):
    def __init__(self,**kwargs):
        super(NoTargetsFlagger,self).__init__(kwargs)
        self.dpi = kwargs.pop('dpi')
        self.threshold = kwargs.pop('threshold')
        assert not kwargs

    def flag_drugs(self):
        from dtk.prot_map import DpiMapping
        dm = DpiMapping(self.dpi)

        to_flag = []

        for wsa in self.each_target_wsa():
            bindings = dm.get_dpi_info(wsa.agent,min_evid=self.threshold)
            if len(bindings) == 0:
                to_flag.append(wsa)

        print("Creating flags")
        self.create_flag_set('NoTargets')

        from django.urls import reverse
        from django.utils.http import urlencode
        for wsa in to_flag:
            print("Creating flag for %s" % wsa.id)
            self.create_flag(
                    wsa_id=wsa.id,
                    category='No Targets',
                    detail=self.dpi,
                    href='',
                    )


def main(argv=None):
    import argparse
    from dtk.prot_map import DpiMapping
    parser = argparse.ArgumentParser(
            description="flag drugs for DPI",
            )
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--count',type=int,default=200)
    parser.add_argument('--dpi',)
    parser.add_argument('--threshold',type=float,
            default=DpiMapping.default_evidence,
            )
    parser.add_argument('ws_id',type=int)
    parser.add_argument('job_id',type=int)
    parser.add_argument('score')
    args = parser.parse_args(argv)

    from runner.models import Process
    err = Process.prod_user_error()
    if err:
        raise RuntimeError(err)

    flagger = NoTargetsFlagger(
                ws_id=args.ws_id,
                job_id=args.job_id,
                score=args.score,
                start=args.start,
                count=args.count,
                dpi=args.dpi,
                threshold=args.threshold,
                )
    flagger.flag_drugs()

if __name__ == '__main__':
    main()
