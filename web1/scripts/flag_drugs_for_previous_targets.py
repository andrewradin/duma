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

class PreviousTargetsFlagger(FlaggerBase):
    def __init__(self,**kwargs):
        super(PreviousTargetsFlagger,self).__init__(kwargs)
        self.dpi = kwargs.pop('dpi')
        self.dpi_threshold = kwargs.pop('dpi_threshold')
        assert not kwargs
    def flag_drugs(self):
        from dtk.duma_view import qstr
        from django.urls import reverse
        self.create_flag_set('PreviousTargets')
        self.setup()
        # write results
        for wsa_id,l in six.iteritems(self.flags):
            self.create_flag(
                        wsa_id=wsa_id,
                        category='PreviousTargets',
                        detail=l[0],
                        href=reverse('clust_screen',args=(self.ws_id,))+qstr({},
                            ids=','.join([str(s) for s in l[1]]),
                            dpi=self.dpi,
                            dpi_t=self.dpi_threshold,
                            ),
                        )
    def setup(self):
        from dtk.prot_map import DpiMapping
        self.dm = DpiMapping(self.dpi)
        self.build_native2wsa_map()
        self.build_wsa2dpi_map(self.get_target_wsa_ids())
        self.build_maps()
    def build_maps(self):
        from browse.models import WsAnnotation, Demerit
        iv = WsAnnotation.indication_vals
        flags_OI_raw = [
                iv.FDA_TREATMENT,
                iv.KNOWN_TREATMENT,
                iv.TRIALED_TREATMENT,
                iv.INITIAL_PREDICTION,
                iv.REVIEWED_PREDICTION,
                iv.HIT,
                iv.KNOWN_CAUSE,
                iv.FDA_CAUSE,
                iv.EXP_TREATMENT,
                iv.EXP_CAUSE,
                iv.HYPOTH_TREATMENT,
                    ]
        flags_OI = [iv.get('label', x) for x in flags_OI_raw]
        demerits_OI= [
                     'Patented',
                     'Exacerbating',
                     'No MOA',
   #                  'Non-novel class',
                    ]
        all_OI = flags_OI + demerits_OI

        all_prots = list(set.union(*list(self.wsa2dpi.values())))
        bindings = self.dm.get_drug_bindings_for_prot_list(all_prots,
                                                          self.dpi_threshold
                                                          )
        prot2wsa = {}
        wsa2prot = {}
        for tup in bindings:
            if tup[1] not in prot2wsa:
                prot2wsa[tup[1]] = set()
            for wsa_id in self.native2wsa.get(tup[0],[]):
                if wsa_id not in wsa2prot:
                    wsa2prot[wsa_id] = set()
                wsa2prot[wsa_id].add(tup[1])
                prot2wsa[tup[1]].add(wsa_id)
        prot2ind = {}
        wsa2ind = {}
        relevant_demerits = {
                d.id:d.desc
                for d in Demerit.objects.filter(
                        desc__in=demerits_OI,
                        )
                }
        inactive_label = iv.get('label', iv.INACTIVE_PREDICTION)
        for wsa in WsAnnotation.objects.filter(id__in=list(wsa2prot.keys())):
            ind = wsa.indication_label()
            if ind not in [inactive_label] + flags_OI:
                continue
            if ind == inactive_label:
                ind = []
                for d_id in wsa.demerits():
                    if int(d_id) in relevant_demerits:
                        ind.append(relevant_demerits[d_id])
                if not ind:
                    continue
            else:
                ind = [ind]
            wsa2ind[wsa.id] = ind
            for p in wsa2prot[wsa.id]:
                if p not in prot2ind:
                    prot2ind[p] = set()
                prot2ind[p].update(set(ind))
        self.flags={}
        for wsa_id,prot_set in six.iteritems(self.wsa2dpi):
            inds = set()
            other_wsas = set()
            gen = (p for p in prot_set if p in prot2ind)
            for p in gen:
                inds.update(prot2ind[p])
                other_wsas.update(prot2wsa[p])
            if inds:
                binned_wsas = [[]]*len(all_OI)
            # filter other_wsas, putting this wsa first
                for ow in other_wsas:
                    indication_index = None
                    for o_inds in wsa2ind.get(ow,[]):
                        if o_inds in inds:
                            if indication_index is None:
                                indication_index = all_OI.index(o_inds)
                            else:
                                indication_index = min([all_OI.index(o_inds), indication_index])
                    if indication_index is not None:
                        binned_wsas[indication_index].append(ow)
                final_wsas = [wsa_id]
                for l in binned_wsas:
                    for w in l:
                        if w not in final_wsas:
                            final_wsas.append(w)
                inds = sorted(list(inds))
                self.flags[wsa_id] = [', '.join(inds),
                                      final_wsas
                                     ]
    def build_native2wsa_map(self):
        self.native2wsa = self.dm.get_wsa_id_map(self.ws)
    def build_wsa2dpi_map(self, wsa_id_list):
        self.wsa2dpi = self.dm.get_wsa2dpi_map(self.ws,
                                               wsa_id_list,
                                               min_evid=self.dpi_threshold
                                              )

