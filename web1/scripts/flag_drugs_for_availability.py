#!/usr/bin/env python3

import django_setup

from flagging.utils import FlaggerBase

class Flagger(FlaggerBase):
    def __init__(self, **kwargs):
        super(Flagger,self).__init__(kwargs)
        self._ncats_as_comm = kwargs.pop('ncats_as_comm', False)
        assert not kwargs
        self.flagset_label = 'Availability'
    def flag_drugs(self):
        self.create_flag_set(self.flagset_label)
        from dtk.comm_avail import wsa_comm_availability
        wsas = self.each_target_wsa(show_progress=False)
        avails = list(wsa_comm_availability(ws=self.ws, wsas=wsas, ncats_as_comm=self._ncats_as_comm, include_details=False))
        for wsa, avail in zip(wsas, avails):
            if avail.has_zinc or avail.has_cas or avail.has_commdb:
                continue
            flag = 'no CAS or CommDB; '+ avail.zinc_reason
            self.create_flag(
                    wsa_id=wsa.id,
                    category=self.flagset_label,
                    detail=flag,
                    href=wsa.drug_url(),
                    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
             description = "flag drugs for availability")
    parser.add_argument('--start', type=int, default=0, help="starting rank on scoreboard to begin testing for novelty")
    parser.add_argument('--count', type=int, default=200, help="total number of drugs to test in scoreboard")
    parser.add_argument('ws_id', type=int, help="workspace id")
    parser.add_argument('job_id', type=int, help="job id")
    parser.add_argument('score', help="score within job_id to rank drugs by")
    args = parser.parse_args()

    from runner.models import Process
    err = Process.prod_user_error()
    if err:
        raise RuntimeError(err)

    flagger = Flagger(
                ws_id=args.ws_id,
                job_id=args.job_id,
                start=args.start,
                count=args.count,
                score=args.score,
                )
    flagger.flag_drugs()
