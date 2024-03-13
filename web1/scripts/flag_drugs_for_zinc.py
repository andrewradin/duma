#!/usr/bin/env python3

from __future__ import print_function
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

class ZincFlagger(FlaggerBase):
    def __init__(self,**kwargs):
        super(ZincFlagger,self).__init__(kwargs)
        assert not kwargs
    def flag_drugs(self):
        self.build_maps()
        self.create_flag_set('ZINC')
        # write results
        for wsa_id,l in six.iteritems(self.flags):
            for zinc_l in l:
# this should be the zinc page for this drug, where the drug ID is zinc_tuple[0]
                hrf = "https://zinc15.docking.org/substances/%s" % zinc_l[0]
                self.create_flag(
                        wsa_id=wsa_id,
                        category='ZINC labels',
                        detail=zinc_l[1],
                        href=hrf,
                        )
    def build_maps(self):
        temp = get_zinc_labels(self.get_target_wsa_ids())
        self.flags = {}
        for k,l in six.iteritems(temp):
            self.flags[k] = [[x[0], ', '.join(x[1])] for x in l]

def get_zinc_labels(wsas, zinc_labels_oi = None, catch_no_label = True):
    from browse.models import WsAnnotation
    if len(wsas) == 0:
        return {} # trick below for getting ws won't work in this case
    example = WsAnnotation.objects.get(pk=wsas[0])
    ws = example.ws
    from dtk.s3_cache import S3File
    from dtk.files import get_file_records
    # get list of labels of interest
    from dtk.zinc import zinc
    z = zinc()
    if not zinc_labels_oi:
        zinc_labels_oi = z.get_labels()
    print(zinc_labels_oi)
    # build a dict holding the set of zinc ids that go with each
    # label of interest
    zinc_id_dict = {}
    vdefaults = ws.get_versioned_file_defaults()
    for zinc_label in zinc_labels_oi:
        zinc_id_dict[zinc_label] = z.get_zinc_id_set_for_label(
                zinc_label,
                vdefaults['zinc'],
                )
        print(zinc_label,len(zinc_id_dict[zinc_label]))
    # Build maps to allow a 2-step conversion from wsa_id to zinc_id.
    # This is equivalent to wsa.get_zinc_id(), but is more efficient
    # when doing multiple conversions.
    wsa2native = {}
    from dtk.unichem import UniChem
    uc = UniChem()
    native2zinc = {}
    vdefaults = ws.get_versioned_file_defaults()
    for native in ('chembl', 'drugbank', 'bindingdb'):
        wsa2native[native] = dict(ws.wsa_prop_pairs(native+'_id'))
        native2zinc[native] = uc.get_converter_dict(
                native,
                'zinc',
                vdefaults['unichem'],
                )
        print(native,len(wsa2native[native]),len(native2zinc[native]))
    # Scan all WSAs of interest, building flags; output is:
    # {wsa_0001: [[zinc_0001, "fda, in-man"],...]}
    flags = {}
    for wsa_id in wsas:
        for native in wsa2native: # chembl, drugbank, etc.
            native_id = wsa2native[native].get(wsa_id)
            if not native_id:
                continue
            try:
                zinc_id_list = native2zinc[native][native_id]
            except KeyError:
                continue
            # there are one or more zinc ids corresponding to the native
            # drug id that matches this wsa; now see which labels each
            # zinc id has
            for zinc_id in zinc_id_list:
                l = []
                for label in zinc_labels_oi:
                    if zinc_id in zinc_id_dict[label]:
                        l.append(label)
                if not l:
                    # There are no labels; in the normal case don't add this
                    # to the output. But for more inclusive label sets, an
                    # id that doesn't match anything is probably too
                    # experimental to be useful, so flag that.
                    if catch_no_label:
                        l = [z.no_label_description()]
                    else:
                        continue
                flags.setdefault(wsa_id,[]).append([zinc_id,l])
    return flags

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description="flag drugs for being in ZINC",
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

    flagger = ZincFlagger(
                ws_id=args.ws_id,
                job_id=args.job_id,
                score=args.score,
                start=args.start,
                count=args.count,
                )
    flagger.flag_drugs()
