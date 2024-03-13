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


class CommercialDbFlagger(FlaggerBase):
    def __init__(self,**kwargs):
        super(CommercialDbFlagger,self).__init__(kwargs)
        assert not kwargs

    def flag_drugs(self):
        from dtk.prot_map import DpiMapping


        print("Creating flags")
        self.create_flag_set('CommercialDb')

        from django.urls import reverse
        from django.utils.http import urlencode
        ver = self.ws.get_dpi_version()
        for wsa in self.each_target_wsa():
            comm_urls = wsa.agent.commercial_urls(version=ver)
            if len(comm_urls) == 0:
                continue

            names = [x[0] for x in comm_urls]
            urls = [x[1] for x in comm_urls]

            self.create_flag(
                    wsa_id=wsa.id,
                    category='CommercialDb',
                    detail=','.join(names),
                    href=urls[0], # Pick an arbitrary URL
                    )
