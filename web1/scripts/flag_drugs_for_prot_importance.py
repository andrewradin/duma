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


def compute_target_scores(uniprots, piece_data, wsa_weighted_scores):
    from collections import defaultdict
    wsa_to_prot_imp = defaultdict(lambda: defaultdict(float))
    print("Summing importance for each target drug and protein")
    for wsa, jid_name, importance in piece_data:
        for score_name, prot_score_map in six.iteritems(importance):
            for prot, score in six.iteritems(prot_score_map):
                if prot in uniprots:
                    full_score_name = jid_name + "_" + score_name
                    full_score_name = full_score_name.lower()
                    if full_score_name not in wsa_weighted_scores[wsa]:
                        continue
                    score_weight = wsa_weighted_scores[wsa][full_score_name.lower()]
                    wsa_to_prot_imp[wsa][prot] += score * score_weight

    return wsa_to_prot_imp

def find_drugs_to_flag(wsa_target_scores, threshold):
    to_flag = set()
    for wsa, prot_imp in six.iteritems(wsa_target_scores):
        imp_sum = sum(prot_imp.values())
        if imp_sum >= threshold:
            to_flag.add(wsa)
    return to_flag

def make_flag_str(wsa, wsa_target_scores, uniprot2gene):
    prot_scores = wsa_target_scores[wsa]
    parts = []
    for prot, score in sorted(six.iteritems(prot_scores), key=lambda x:-x[1]):
        gene = uniprot2gene[prot]
        parts.append("%s=%.2f" % (gene, score))
    return ' '.join(parts)


def mark_unclassified_wsas(wsas, indication, demerit, detail):
    from browse.models import WsAnnotation
    inds = WsAnnotation.indication_vals
    for wsa in WsAnnotation.objects.filter(pk__in=wsas):
        if wsa.indication == inds.UNCLASSIFIED:
            wsa.update_indication(
                    indication=indication,
                    demerits=wsa.demerits() | set([demerit.id]),
                    detail=detail
                    )


class ProtImportanceFlagger(FlaggerBase):
    def __init__(self,**kwargs):
        super(ProtImportanceFlagger,self).__init__(kwargs)
        self.uniprots = kwargs.pop('uniprots')
        self.threshold = kwargs.pop('threshold')
        self.dry_run = kwargs.pop('dry_run', False)
        self.set_indication = kwargs.pop('set_indication', False)
        self._build_maps()
        assert not kwargs

    def flag_drugs(self):
        uniprots = set(self.uniprots)

        from dtk.target_importance import TargetScoreImportance
        method = "peel_cumulative"
        tsi = TargetScoreImportance(
                ws_id=self.ws_id,
                wzs_job_id=self.job_id,
                wsa_start=self.start,
                wsa_count=self.count,
                imp_method=method
                )
        wsa_ids = tsi.get_wsa_ids()

        from dtk.score_importance import ScoreImportance
        si = ScoreImportance(self.ws_id, self.job_id)
        print("Fetching score weights")
        _, _, wsa_weighted_scores = si.get_score_importances(wsa_ids)

        pieces = tsi.make_pieces()
        from collections import defaultdict
        wsa_to_prot_imp = defaultdict(lambda: defaultdict(float))
        print("Collecting importance data")
        to_flag = set()
        piece_data = []
        for piece in pieces:
            wsa = piece.wsa_id
            importance = piece.get_importance(method=method, cache_only=True)
            if importance is None:
                print("Couldn't get cached target importance data for %s, %s, skipping" % (wsa, piece.name.encode('utf8')))
                continue
            piece_data.append((wsa, tsi.jids[piece.job_id], importance))

        if len(piece_data) == 0:
            raise IOError("Unable to find any target importance data for these drugs, probably was not run, bailing out")

        wsa_target_scores = compute_target_scores(uniprots, piece_data, wsa_weighted_scores)
        to_flag = find_drugs_to_flag(wsa_target_scores, self.threshold)

        print("Creating flags")
        self.create_flag_set('UnwantedImportantTarget')
        from django.urls import reverse
        from django.utils.http import urlencode
        for wsa_id in to_flag:
            flag_str = make_flag_str(wsa_id, wsa_target_scores, self.uniprot2gene)
            print("Creating flag for %s: %s" % (wsa_id, flag_str))
            opts = {
                    'wzs_jid': self.job_id,
                    'method': method
                    }
            url = reverse('moldata:trg_scr_imp',args=(self.ws_id,wsa_id)) + "?" + urlencode(opts)
            if self.dry_run:
                print("Skipping for dry run; url=%s" % url)
            else:
                self.create_flag(
                        wsa_id=wsa_id,
                        category='Unwanted Important Protein',
                        detail=flag_str,
                        href=url,
                        )
        if self.set_indication:
            from browse.models import Demerit, WsAnnotation
            demerit, new = Demerit.objects.get_or_create(desc='Non-novel class')
            ind = WsAnnotation.indication_vals.INACTIVE_PREDICTION
            mark_unclassified_wsas(to_flag, ind, demerit, detail="Flag: Unwanted important targets")

    def _build_maps(self):
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
    parser.add_argument('--threshold',type=float,
            default=DpiMapping.default_evidence,
            )
    parser.add_argument('--dry-run',action='store_true',help='Just list which flags it would create instead of creating.')
    parser.add_argument('ws_id',type=int)
    parser.add_argument('job_id',type=int)
    parser.add_argument('score')
    parser.add_argument('uniprot',nargs='+')
    args = parser.parse_args()

    from runner.models import Process
    err = Process.prod_user_error()
    if err:
        raise RuntimeError(err)

    flagger = ProtImportanceFlagger(
                ws_id=args.ws_id,
                job_id=args.job_id,
                score=args.score,
                start=args.start,
                count=args.count,
                uniprots=args.uniprot,
                threshold=args.threshold,
                dry_run=args.dry_run
                )
    flagger.flag_drugs()
