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

class DemeritFlagger(FlaggerBase):
    def __init__(self,**kwargs):
        super(DemeritFlagger,self).__init__(kwargs)
        assert not kwargs
    def flag_drugs(self):
        self.build_maps()
        self.create_flag_set('Demerit')
        # write results
        for agent,flags in six.iteritems(self.flags):
            wsa = self.agent2wsa[agent]
            for flag in flags:
                self.create_flag(
                        wsa_id=wsa.id,
                        category='Demerit',
                        detail=flag,
                        href=wsa.drug_url(),
                        )
    def build_maps(self):
        from browse.models import WsAnnotation
        wsa_ids = self.get_target_wsa_ids()
        self.agent2wsa = {
                wsa.agent_id:wsa
                for wsa in WsAnnotation.objects.filter(id__in=wsa_ids)
                }
        self.flags = find_demeritted_drugs(WsAnnotation.objects.filter(
                                             agent_id__in=list(self.agent2wsa.keys()),
                                           ).exclude(
                                             demerit_list='',
                                           ))

def find_demeritted_drugs(wsas, demerits_OI = ['Ubiquitous','Data Quality', 'Unavailable', 'Tox']):
    from browse.models import Demerit
    wanted = {
            d.id:d
            for d in Demerit.objects.filter(
                    desc__in=demerits_OI,
                    )
            }
    wanted_ids = set(wanted.keys())
    flags = {}
    for other_wsa in wsas:
        got = other_wsa.demerits() & wanted_ids
        if not got:
            continue
        s = flags.setdefault(other_wsa.agent_id,set())
        s.update(set([wanted[x].desc for x in got]))
    return flags


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description="flag drugs for previous demerits",
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

    flagger = DemeritFlagger(
                ws_id=args.ws_id,
                job_id=args.job_id,
                score=args.score,
                start=args.start,
                count=args.count,
                )
    flagger.flag_drugs()
