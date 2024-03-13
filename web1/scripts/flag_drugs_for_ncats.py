#!/usr/bin/env python3

import sys
import six
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

class NcatsFlagger(FlaggerBase):
    def __init__(self,**kwargs):
        super(NcatsFlagger,self).__init__(kwargs)
        assert not kwargs
    def flag_drugs(self):
        self.build_maps()
        self.create_flag_set('NCATS')
        # write results
        for ncats,wsa_l in six.iteritems(self.native2wsa):
            hrf = "http://ncats.nih.gov/files/%s.pdf" % ncats
            for wsa_id in wsa_l:
                self.create_flag(
                        wsa_id=wsa_id,
                        category='NCATS repurposing drug',
                        detail=ncats,
                        href=hrf,
                        )
    def build_maps(self):
        # native2wsa maps from the ncats collection key to a set of wsa_ids,
        # so any wsa_ids in the values portion either come from NCATS or
        # match an NCATS drug
        ks = 'ncats'
        wsa_ids = self.get_target_wsa_ids()
        from dtk.data import MultiMap
        mm=MultiMap([])
        for prop in (ks+'_id', 'm_'+ks+'_id'):
            mm.union(MultiMap([x
                    for x in self.ws.wsa_prop_pairs(prop)
                    if x[0] in wsa_ids
                    ]))
        self.native2wsa = mm.rev_map()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description="flag drugs for being in NCATS",
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

    flagger = NcatsFlagger(
                ws_id=args.ws_id,
                job_id=args.job_id,
                score=args.score,
                start=args.start,
                count=args.count,
                )
    flagger.flag_drugs()
