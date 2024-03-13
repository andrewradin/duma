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

class DpiFlagger(FlaggerBase):
    def __init__(self,**kwargs):
        super(DpiFlagger,self).__init__(kwargs)
        self.uniprots = kwargs.pop('uniprots')
        self.dpi = kwargs.pop('dpi')
        self.threshold = kwargs.pop('threshold')
        assert not kwargs
    def flag_drugs(self):
        from dtk.prot_map import DpiMapping
        self.dm = DpiMapping(self.dpi)
        self.build_maps()
        self.create_flag_set('UnwantedTarget')
        # reorganize uniprots for faster membership checking
        uniprots = set(self.uniprots)
        # scan DPI file
        flagged = {}
        from dtk.files import get_file_records
        for row in get_file_records(self.dm.get_path(),
                            select=(uniprots,1),
                            keep_header=False,
                            ):
            native,uniprot,ev = row[:3]
            if float(ev) < self.threshold:
                continue
            for wsa_id in self.native2wsa.get(native,[]):
                flagged.setdefault(wsa_id,set()).add(uniprot)
        # write results
        from django.urls import reverse
        for wsa_id,uniprots in six.iteritems(flagged):
            ug_list = [
                    (uniprot,self.uniprot2gene[uniprot])
                    for uniprot in uniprots
                    ]
            ug_list.sort(key=lambda x:x[1])
            for uniprot,gene in ug_list:
                self.create_flag(
                        wsa_id=wsa_id,
                        category='Unwanted DPI',
                        detail=gene,
                        href=reverse('protein',args=(self.ws_id,uniprot)),
                        )
    def build_maps(self):
        # native2wsa maps from the dpi native key to a set of wsa_ids
        wsa_ids = self.get_target_wsa_ids()
        self.native2wsa = {
                k:set([x for x in v if x in wsa_ids])
                for k,v in self.dm.get_wsa_id_map(self.ws).items()
                }
        # build uniprot2gene map
        from browse.models import Protein
        self.uniprot2gene = {
                uniprot:Protein.get_gene_of_uniprot(uniprot) or '(%s)'%uniprot
                for uniprot in self.uniprots
                }

if __name__ == '__main__':
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
    parser.add_argument('uniprot',nargs='+')
    args = parser.parse_args()

    from runner.models import Process
    err = Process.prod_user_error()
    if err:
        raise RuntimeError(err)

    flagger = DpiFlagger(
                ws_id=args.ws_id,
                job_id=args.job_id,
                score=args.score,
                start=args.start,
                count=args.count,
                uniprots=args.uniprot,
                dpi=args.dpi,
                threshold=args.threshold,
                )
    flagger.flag_drugs()
