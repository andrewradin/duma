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

class ShadowFlagger(FlaggerBase):
    def __init__(self,**kwargs):
        super(ShadowFlagger,self).__init__(kwargs)
        assert not kwargs

    def flag_drugs(self):
        self.build_maps()
        self.create_flag_set('Shadow')
        for wsa_id, url in six.iteritems(self.flags):
            self.create_flag(wsa_id=wsa_id, category='Shadow',
                         detail='Has shadow drugs', href=url)

    def build_maps(self):
        from browse.models import WsAnnotation

        self.flags = {}
        for wsa in WsAnnotation.objects.filter(id__in=self.get_target_wsa_ids()):
            shadowed_set = wsa.agent.shadowed_chembl_id_set | wsa.agent.shadowed_bindingdb_id_set
            if shadowed_set:
                self.flags[wsa.id] = wsa.drug_url()

    def _get_native2wsa_map(self,wsa_ids, ks):
        from dtk.data import MultiMap
        mm=MultiMap([])
        for prop in (ks+'_id', 'm_'+ks+'_id'):
            mm.union(MultiMap([x
                    for x in self.ws.wsa_prop_pairs(prop)
                    if x[0] in wsa_ids
                    ]))
        return mm.rev_map()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description="flag drugs for having ChEMBL/BindingDB shadows",
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

    flagger = ShadowFlagger(
                ws_id=args.ws_id,
                job_id=args.job_id,
                score=args.score,
                start=args.start,
                count=args.count,
                )

