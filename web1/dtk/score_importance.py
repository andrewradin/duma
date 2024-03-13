
import os
import json
import six

import logging
logger = logging.getLogger(__name__)

def get_score_importances(wsas_or_ids):
    """Fetches score importances across multiple ws/wzs's."""
    if not wsas_or_ids:
        return []
    from browse.models import WsAnnotation
    if not isinstance(wsas_or_ids[0], WsAnnotation):
        wsas = WsAnnotation.objects.filter(pk__in=wsas_or_ids)
    else:
        wsas = wsas_or_ids

    from dtk.data import MultiMap
    wzs_and_wsas = ((wsa.get_marked_eff_jid(), wsa) for wsa in wsas)
    wzs_to_wsas = MultiMap(wzs_and_wsas).fwd_map()
    datas = {}
    for wzs_jid, wsas in wzs_to_wsas.items():
        if wzs_jid is None:
            continue
        from runner.process_info import JobInfo
        ws = next(iter(wsas)).ws
        logger.info("Fetching wzs jid %s for ws %s, %d wsas", wzs_jid, ws, len(wsas))
        JobInfo.get_bound(ws, wzs_jid).fetch_lts_data() # make sure we have this local
        si = ScoreImportance(ws.id, wzs_jid)
        wsa_ids = [wsa.id for wsa in wsas]
        _, _, weighted_scores = si.get_score_importances(wsa_ids)
        datas.update(weighted_scores)

    return datas


class ScoreImportance:
    def __init__(self, ws_id, wzs_job_id):
        self.ws_id = ws_id
        self.wzs_job_id = wzs_job_id
        self._weights_and_srcs = None

    def get_score_importances(self, wsa_ids):
        cache = ScoreImportanceCache(self.ws_id, self.wzs_job_id)
        pieces = [WsaScoreImportance(wsa_id, self, cache) for wsa_id in wsa_ids]
        importances = [piece.get_score_importance() for piece in pieces]

        weights = dict(zip(wsa_ids, [x[0] for x in importances])) 
        scores = dict(zip(wsa_ids, [x[1] for x in importances])) 
        weighted_scores = dict(zip(wsa_ids, [x[2] for x in importances])) 

        return weights, scores, weighted_scores

    def get_score_importance(self, wsa_id):
        return [x[wsa_id] for x in self.get_score_importances([wsa_id])]

    def get_score_weights_and_sources(self):
        if not self._weights_and_srcs:
            from runner.process_info import JobInfo
            bji = JobInfo.get_bound(self.ws_id, self.wzs_job_id)
            logger.info(f"Fetching score weights and sources for {self.ws_id} {self.wzs_job_id}")
            weights, srcs = bji.get_score_weights_and_sources()
            self._weights_and_srcs = (weights, srcs)
        return self._weights_and_srcs


class ScoreImportanceCache:
    def __init__(self, ws_id, wzs_job_id):
        from path_helper import PathHelper
        self.cache_dir = os.path.join(
                PathHelper.storage,
                'scrimp_cache',
                str(ws_id),
                str(wzs_job_id),
                )

    def get(self, wsa_id):
        cache_file = self._get_cache_file(wsa_id)
        if not os.path.exists(cache_file):
            return None
        with open(cache_file) as f:
            return json.loads(f.read())
            

    def put(self, wsa_id, value):
        from path_helper import make_directory
        make_directory(self.cache_dir)
        cache_file = self._get_cache_file(wsa_id)
        with open(cache_file, 'w') as f:
            f.write(json.dumps(value))


    def _get_cache_file(self, wsa_id):
        return os.path.join(self.cache_dir, '%s.json' % wsa_id)



class WsaScoreImportance:
    def __init__(self, wsa_id, weights_src, cache):
        self.wsa_id = wsa_id
        self.weights_src = weights_src
        self.cache = cache

    def get_score_importance(self):
        result = self.cache.get(self.wsa_id)

        if result is not None:
            return result

        result = self._compute()
        self.cache.put(self.wsa_id, result)
        return result

    def _compute(self):
        weights, srcs = self.weights_src.get_score_weights_and_sources()

        wsa_weights = {}
        wsa_scores = {}
        wsa_weighted_scores = {}

        for (f_key,weight),src in zip(weights,srcs):
            # src may be a dict or a Norm object, so supply the default
            # in a way that's compatible with both
            val = src.get(self.wsa_id) or  0
            key = f_key.lower()

            wsa_weights[key] = weight
            wsa_scores[key] = val
            wsa_weighted_scores[key] = val * weight

        def normalize(weight_map):
            weight_sum = sum(six.itervalues(weight_map)) + 1e-9
            for weight_key in six.iterkeys(weight_map):
                weight_map[weight_key] /= weight_sum

        normalize(wsa_weights)
        normalize(wsa_scores)
        normalize(wsa_weighted_scores)

        return wsa_weights, wsa_scores, wsa_weighted_scores

