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
from collections import defaultdict

from dtk.score_source import ScoreSource


def compute_target_scores(piece_data, wsa_weighted_scores, cds_choices):
    from collections import defaultdict
    wsa_scoresrc_prot_score = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    wsa_scoresrc_score = defaultdict(lambda: defaultdict(float))

    # Construct weights for each scoretype.
    for wsa, score_weights in six.iteritems(wsa_weighted_scores):
        for full_score_name, score_weight in six.iteritems(score_weights):
            scoresrc = ScoreSource(full_score_name, cds_choices)
            wsa_scoresrc_score[wsa][scoresrc] = score_weight

    # Go through trgimp data and construct scoretype->prot weights.
    for wsa, jid_name, importance in piece_data:
        for score_name, prot_score_map in six.iteritems(importance):
            full_score_name = jid_name + "_" + score_name
            full_score_name = full_score_name.lower()
            if full_score_name not in wsa_weighted_scores[wsa]:
                continue
            scoresrc = ScoreSource(full_score_name, cds_choices)
            score_weight = wsa_weighted_scores[wsa][full_score_name]
            for prot, score in six.iteritems(prot_score_map):
                x = score * score_weight
                wsa_scoresrc_prot_score[wsa][scoresrc][prot] += x

    # Fill in the missings.
    for wsa, scoresrc_score in six.iteritems(wsa_scoresrc_score):
        for scoresrc, score in six.iteritems(scoresrc_score):
            sum_targ = sum(wsa_scoresrc_prot_score[wsa][scoresrc].values())
            missing = score - sum_targ
            if missing > 1e-6:
                wsa_scoresrc_prot_score[wsa][scoresrc]['missing'] += missing

    return wsa_scoresrc_score, wsa_scoresrc_prot_score

class FlagMaker:
    def __init__(self, parms, wsa_scoresrc_score, wsa_scoresrc_prot_score):
        self.parms = parms
        self.wsa_scoresrc_score = wsa_scoresrc_score
        self.wsa_scoresrc_prot_score = wsa_scoresrc_prot_score

    def flag_pathways_only(self, wsa):
        score_from_pathways = 0

        for scoresrc, score in six.iteritems(self.wsa_scoresrc_score[wsa]):
            if scoresrc.is_pathway():
                score_from_pathways += score

        if score_from_pathways >= self.parms.pathway_threshold:
            return "%.2f" % score_from_pathways
        else:
            return False

    def flag_single_source(self, wsa):
        scores_by_type = defaultdict(float)
        for scoresrc, score in six.iteritems(self.wsa_scoresrc_score[wsa]):
            score_type = scoresrc.source_type()[1]
            scores_by_type[score_type] += score

        if len(scores_by_type) == 0:
            return False

        biggest_type, biggest_val = max(list(scores_by_type.items()), key=lambda x: x[1])

        if biggest_val >= self.parms.single_source_threshold:
            return "%s=%.2f" % (biggest_type, biggest_val)
        else:
            return False

    def flag_otarg_knowndrug(self, wsa):
        score_from_knowndrugs = 0
        for scoresrc, score in six.iteritems(self.wsa_scoresrc_score[wsa]):
            if scoresrc.is_otarg() and scoresrc.otarg_source() == 'knowndrug':
                score_from_knowndrugs += score


        if score_from_knowndrugs >= self.parms.known_drug_threshold:
            return "%.2f" % score_from_knowndrugs
        else:
            return False

    def flag_weak_targets(self, wsa):
        target_score = defaultdict(float)
        for scoresrc, prot_score in six.iteritems(self.wsa_scoresrc_prot_score[wsa]):
            for prot, score in six.iteritems(prot_score):
                target_score[prot] += score

        if len(target_score) == 0:
            # We had no information for this wsa.
            return False

        best_score = max(target_score.values())

        if best_score <= self.parms.weak_targets_threshold:
            return "Max: %.2f" % best_score
        else:
            # We have a strong score, nothing to flag.
            return False




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
    for prot, score in sorted(six.iteritems(prot_scores)):
        gene = uniprot2gene[prot]
        parts.append("%s=%.2f" % (gene, score))
    return ' '.join(parts)


def make_tsi_piece_data(parms):
    from dtk.target_importance import TargetScoreImportance
    method = "peel_cumulative"
    tsi = TargetScoreImportance(
            ws_id=parms.ws_id,
            wzs_job_id=parms.job_id,
            wsa_start=parms.start,
            wsa_count=parms.count,
            imp_method=method
            )
    wsa_ids = tsi.get_wsa_ids()

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

    return piece_data, wsa_ids

def make_wsa_weighted_scores(parms, wsa_ids):
    from dtk.score_importance import ScoreImportance
    si = ScoreImportance(parms.ws_id, parms.job_id)
    print("Fetching score weights")
    _, _, wsa_weighted_scores = si.get_score_importances(wsa_ids)
    return wsa_weighted_scores

class ScoreQualityFlagger(FlaggerBase):
    def __init__(self,**kwargs):
        super(ScoreQualityFlagger,self).__init__(kwargs)
        parm_names = (
                'pathway_threshold',
                'single_source_threshold',
                'known_drug_threshold',
                'weak_targets_threshold'
                )
        from collections import namedtuple
        Parms = namedtuple("Parms", parm_names)
        self.parms = Parms(*[kwargs.pop(x) for x in parm_names])


        assert not kwargs

    def flag_drugs(self):
        method = "peel_cumulative"
        piece_data, wsa_ids = make_tsi_piece_data(self)
        wsa_weighted_scores = make_wsa_weighted_scores(self, wsa_ids)

        wsa_scoresrc_score, wsa_scoresrc_prot_score = compute_target_scores(piece_data, wsa_weighted_scores, self.ws.get_cds_choices())

        fm = FlagMaker(self.parms, wsa_scoresrc_score, wsa_scoresrc_prot_score)

        flagsets = {
            'ScrQual: Only Pathways': FlagMaker.flag_pathways_only,
            'ScrQual: Single Source': FlagMaker.flag_single_source,
            'ScrQual: Only Known Drug': FlagMaker.flag_otarg_knowndrug,
            'ScrQual: Weak Targets': FlagMaker.flag_weak_targets,
        }
        flagset_objs = {flagset_name:self.create_flag_set(flagset_name)
                        for flagset_name in flagsets}

        from django.urls import reverse
        from django.utils.http import urlencode
        for wsa in wsa_scoresrc_score:
            for flagset_name, func in six.iteritems(flagsets):
                flag_str = func(fm, wsa)
                if flag_str:
                    opts = {
                            'wzs_jid': self.job_id,
                            'method': method
                            }
                    url = reverse('moldata:trg_scr_imp',args=(self.ws_id,wsa)) + "?" + urlencode(opts)
                    fs=flagset_objs[flagset_name]
                    self.create_flag(
                            wsa_id=wsa,
                            category=flagset_name,
                            detail=flag_str,
                            href=url,
                            fs=fs,
                            )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description="flag drugs for score quality heuristics",
            )
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--count',type=int,default=200)
    parser.add_argument('ws_id',type=int)
    parser.add_argument('job_id',type=int)
    parser.add_argument('score')
    args = parser.parse_args()

    flagger = ScoreQualityFlagger(
                ws_id=args.ws_id,
                job_id=args.job_id,
                score=args.score,
                start=args.start,
                count=args.count,
                pathway_threshold=0.95,
                single_source_threshold=0.95,
                known_drug_threshold=0.75,
                weak_targets_threshold=0.2
                )
    flagger.flag_drugs()
