#!/usr/bin/env python

# Tools for calculating target importance
#
# The first cut of this is just a small wrapper for the code in
# scripts/extract_importance that implements caching and insulates
# the views from some ideosyncacies of the current implementation.
# This should fairly quickly evolve into a re-implementation that
# allows bulk pre-calculation of importance data on the worker
# machine for a subset of drugs under review.

# XXX - clean up discrepancy as to where loop through scores goes (not one
# XXX   way for LOO and another for peel)
# XXX - clean up interface wrinkle for remapping paths to prot names; this
# XXX   should just happen if necessary in score_importance
# XXX - TargetImportance subclass ctor params shouldn't need to be wrapped
# XXX   in str()

from __future__ import print_function
import os
import sys
from path_helper import PathHelper,make_directory

import django
import django_setup
from runner.process_info import JobInfo

from browse.models import WsAnnotation
from six.moves import cPickle as pickle

import logging
verbose=False
logger = logging.getLogger(__name__)

METHOD_CHOICES = [
   ('peel_cumulative', 'Cumulative'),
   ('peel_previous', 'Previous'),
   ('LOO', 'Leave one out'),
   ('LOI', 'Leave one in'),
   ]

DEFAULT_METHOD = 'peel_cumulative'


class Cache(object):
    """A simple cache for storing repeated computation results.

    This cache is never expired, so it should only be used transiently
    (e.g. for a batch of computations), rather than stored long-term.
    """
    def __init__(self):
        self._cache = {}
    def get_or_compute(self, key, compute_fn):
        """Returns cached value for key, or computes & caches it using compute_fn."""
        try:
            return self._cache[key]
        except KeyError:
            value = compute_fn()
            self._cache[key] = value
            return value

def fm_bji_from_settings(ws_id, settings):
    from runner.process_info import JobInfo
    assert 'fm_code' in settings
    return JobInfo.get_bound(ws_id,
               JobInfo.extract_feature_matrix_job_id(settings['fm_code'])
           )

def get_wzs_jids(ws_id, wzs_settings):
    """Returns the source job ids for a given WZS job settings."""
    jids = {}
    # XXX A possibly more robust way to do this is to search backwards
    # XXX through the job dependency list until a job is found for
    # XXX which a SourceRoleMapper can return job ids. There should
    # XXX only be a single dependency until this happens. But, this is
    # XXX currently tied to WZS anyway, so there's no urgency.
    # starting sprint172 WZS uses FVS as its starting point
    # that means we need to get the initial JIDs from that run
    source_job_settings = wzs_settings

    # If we have an fm_code, we actually want to look at the sources
    # from that job.
    while 'fm_code' in source_job_settings:
        bji = fm_bji_from_settings(ws_id, source_job_settings)
        source_job_settings = bji.job.settings()
    for k,v in  source_job_settings.items():
        if not k.endswith('srcjob'):
            continue
        x = k.split('srm_')[1].split('_srcjob')[0]
        jids[v] = x
    return jids

def filter_accepted_jobs(jids):
    """Returns a filtered set of jids, removing jobs that don't run target importance.
    jids should be {jid: job_role}
    """
    allowed_job_suffixes = ['codes', 'esga', 'depend', 'defus', 'capp', 'gpath', 'path']

    good_jids = {}
    for jid, job_name in jids.items():
        matches = [True for suff in allowed_job_suffixes if job_name.endswith(suff)]
        if len(matches) ==  0:
            logger.debug("Skipping job %s because it doesn't match" % job_name)
        else:
            logger.debug("Using job %s" % job_name)
            good_jids[jid] = job_name
    return good_jids

def find_indirect_jobs(ws, jids):
    supported_types = set(['sigdif'])
    jids_visited = set()
    jids_left = set(jids.keys())
    found_jobs = set()
    while len(jids_left) > 0:
        jid = jids_left.pop()
        if jid in jids_visited:
            continue
        jids_visited.add(jid)

        bji = JobInfo.get_bound(ws, jid)
        if bji.job_type in supported_types:
            found_jobs.add((jid, bji.role_label()))

        jids_left.update(bji.get_input_job_ids())

    return dict(found_jobs)


def get_drug_names(wsa_ids):
    """Returns the name for the given WSA ids, mostly useful for logging."""
    wsas = WsAnnotation.all_objects.select_related('agent').in_bulk(wsa_ids)
    return [wsas[int(wsa_id)].agent.canonical for wsa_id in wsa_ids]

def get_all_targets(ws_id, wsa_ids):
    from browse.models import Workspace
    from dtk.prot_map import DpiMapping, AgentTargetCache
    ws = Workspace.objects.get(pk = ws_id)
    dpi_fn = ws.get_dpi_default()
    dpi_th = ws.get_dpi_thresh_default()
    print("Loading targets for our wsas from %s" % dpi_fn)
    dpi = DpiMapping(dpi_fn)

    # Convert wsa to agents
    agent_ids = WsAnnotation.objects.filter(id__in=wsa_ids).values_list('agent_id', flat=True)
    agent_ids = list(agent_ids)


    targ_cache = AgentTargetCache(
            mapping=dpi,
            agent_ids=agent_ids,
            dpi_thresh=dpi_th,
            )
    uniprots = set()
    for agent_id in agent_ids:
        for agent_info in targ_cache.info_for_agent(agent_id):
            uniprots.add(agent_info[0])
    return uniprots


def trgimp_jids_from_wzs(ws_id, wzs_id, indirect_scores):
    wzs_bji = JobInfo.get_bound(ws_id, wzs_id)
    wzs_settings = wzs_bji.job.settings()
    all_jids = get_wzs_jids(ws_id, wzs_settings)

    if not indirect_scores:
        good_jids = filter_accepted_jobs(all_jids)
    else:
        good_jids = find_indirect_jobs(ws_id, all_jids)
    return good_jids

class TargetScoreImportance:
    """Computes target importance for a set of drugs.

    See run() for the typical workflow.  run_piece, which is the bulk of the
    compute time, can be executed locally or remotely.
    """

    def __init__(self, ws_id, wzs_job_id, wsa_start, wsa_count,
                 imp_method=DEFAULT_METHOD, skip_cache=False,
                 indirect_scores=False, extra_wsas=None, max_input_prots=None,
                 condensed=False):
        self.ws_id = ws_id
        self.wzs_job_id = wzs_job_id
        self.wsa_start = wsa_start
        self.wsa_count = wsa_count
        self.imp_method = imp_method
        self.extra_wsas = extra_wsas or []
        self.condensed = condensed

        # Indicates we should generate indirect prot scores instead of drug scores.
        self._indirect_scores = indirect_scores

        self.jids = trgimp_jids_from_wzs(ws_id=ws_id, wzs_id=wzs_job_id, indirect_scores=indirect_scores)
        self.bound_jids = JobInfo.get_all_bound(self.ws_id, self.jids)
        self.skip_cache = skip_cache

        self._max_input_prots = max_input_prots


    def run(self):
        # Construct the sub jobs, each is a (drug, CM) to be analyzed.
        pieces = self.make_pieces()
        # Prepare the input data for each piece.
        data = self.gen_piece_inputs(pieces)
        # Compute the importance scores for each piece - this is the bulk
        # of the runtime and can be run remotely on serialized data.
        importance = TargetScoreImportance.run_pieces(data)
        # Loads back in the importance scores, storing them into the cache.
        self.load_scores(pieces, importance)

    def get_wsa_ids(self):
        from flagging.utils import get_target_wsa_ids
        out = get_target_wsa_ids(self.ws_id, self.wzs_job_id, "wzs", self.wsa_start, self.wsa_count, condensed=self.condensed)
        return out + self.extra_wsas

    def make_pieces(self):
        jids = self.jids
        wsa_ids = self.get_wsa_ids()
        pieces = []
        if not self._indirect_scores:
            drug_names = get_drug_names(wsa_ids)


            for (jid, job_name), bji in zip(jids.items(), self.bound_jids):
                for wsa_id, drug_name in zip(wsa_ids, drug_names):
                    name = '%s %s' % (drug_name, job_name)
                    imp_job = TrgImpJobResults(self.ws_id, bji, wsa_id=wsa_id, name=name, skip_cache=self.skip_cache)
                    pieces.append(imp_job)
        else:
            prot_ids = get_all_targets(self.ws_id, wsa_ids)
            print("%d targets for our %d wsas" % (len(prot_ids), len(wsa_ids)))
            # Now generate prot-based pieces.

            # 0) Grab the set of sigdiff jobs (animal, kd, etc.)
            # 1) Generate the set of target proteins that we care about
            #       I guess all the targets of drugs here?
            # 2) Generate a piece for each one, keyed by target prot instead of wsa

            for (jid, job_name), bji in zip(jids.items(), self.bound_jids):
                for prot_id in prot_ids:
                    name = '%s %s' % (prot_id, job_name)
                    imp_job = TrgImpJobResults(self.ws_id, bji, target_id=prot_id,
                                            name=name, skip_cache=self.skip_cache)
                    pieces.append(imp_job)


        return pieces

    def gen_piece_inputs(self, pieces):
        # We provide a cache for shared inputs amongst the job pieces.
        # We don't need to explicitly save this afterwards, it just ensures
        # that pieces can share references to the same data.
        # (Pickle will only serialize one copy if running remotely)
        cache = Cache()

        # Wrap gen_input_data so that even if one piece fails, we keep going.
        def safe_gen_input(piece):
            try:
                return piece.gen_input_data(cache, self._max_input_prots)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.info("Failed to generate input data: %s", e)
                return None

        pieces_data = [safe_gen_input(piece) for piece in pieces]


        # Can turn this on if you need to debug what is using up memory/space
        # in the input data.
        if False:
            from collections import defaultdict
            by_job_type = defaultdict(list)
            for d in pieces_data:
                if d:
                    by_job_type[d['job_type']].append(d)
            logger.info("Total size: %d" % len(pickle.dumps(pieces_data)))
            for j, d in by_job_type.items():
                logger.info("%s: %d" % (j, len(pickle.dumps(d))))

        return {
            'pieces': pieces_data,
            'method': self.imp_method
        }

    @classmethod
    def run_pieces(cls, input_data):
        """This can run on the worker or locally."""
        output = []
        method = input_data['method']
        cache = Cache()
        for piece_data in input_data['pieces']:
            try:
                piece_output = cls.run_piece(piece_data, method, cache)
                output.append(piece_output)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.info("Failed to run %s, continuing" % (piece_data['description'], ))
                output.append(None)
        return output

    @classmethod
    def run_piece(cls, piece_data, method, cache):
        # piece_data is None if something failed while generating input for this piece.
        if piece_data is None:
            logger.info("Skipping run, no input data")
            return None

        if verbose:
            logger.info("Running target importance for CM %s (%s)" % (piece_data['description'], method))
        if piece_data['job_type'] == 'depend' and method.startswith('peel'):
            # Depend has too many pathways to do peel-style scoring, it bails out.
            # LOI gives identical scores anyway for psScoreMax.
            logger.info('Overriding peel method to LOI for depend job')
            method = 'LOI'

        import time
        start = time.time()
        from scripts.extract_importance import TargetImportance
        CM_Class = TargetImportance.lookup(piece_data['job_type'])
        cm = CM_Class(
                ws_id=piece_data['ws_id'],
                wsa_id=piece_data['wsa_id'],
                job_id=piece_data['job_id'],
                cache=cache,
                key_exclude=piece_data['key_exclude'],
                data=piece_data['data'],
                )

        peel_prefix='peel_'
        if method == 'LOO':
            cm.leave_one_out()
            cm.score_importance()
        elif method == 'LOI':
            cm.leave_one_in()
            cm.score_importance(scoring_method='LOI')
        elif method.startswith(peel_prefix):
            sub_method = method[len(peel_prefix):]
            for sn in cm.score_names:
                try:
                    cm.peel_one_off(sn)
                except NotImplementedError:
                    # some TargetImportance classes throw this for scores
                    # they don't support; by catching it here we allow
                    # other scores for the job to be processed
                    # XXX some alternative to ignoring silently??
                    logger.info("Not implemented: %s, %s" % (sn, piece_data['description']))
                    continue
                cm.score_importance(sn, scoring_method = sub_method)
        else:
            raise NotImplementedError("unknown method '%s'"%method)

        scores = cm.importance_scores
        score_names = cm.scores_to_report
        cm_data = piece_data['data']
        out = {
            'version': 1,
            'prots': scores,
            'score_names': score_names
        }
        if cm_data.get('needs_remap', False) > 0:
            logger.info("Remapping scores for %s" % piece_data['description'])
            from scripts.extract_importance import PathsToProts
            ptp = PathsToProts(
                    cm_data['dpi'],
                    cm_data['ppi'],
                    cm_data['gmt']
                    )
            path_scores = scores
            prot_scores = ptp.convert_paths_to_prots(path_scores.copy(), score_names, cache)

            out['prots'] = prot_scores
            out['pathways'] = path_scores

        end = time.time()
        logger.info("%s took %.1f seconds, generated scores for %d targets (predicted relative duration %d)" % (piece_data['description'], end - start, len(scores), piece_data['predicted_duration']))

        return out

    def load_scores(self, pieces, importance_data):
        for piece, score_info in zip(pieces, importance_data):
            # Score info is unset if we failed during either input generation
            # or during execution of a piece.  In that case, we don't want
            # to cache the failure.
            if not score_info:
                continue
            score_result = piece.load_score(score_info)
            piece.cache_result(score_result, self.imp_method)



class TrgImpJobResults:
    def __init__(self,ws_id,job_or_id,target_id=None,wsa_id=None,key_exclude=[],skip_cache=False,name=None):
        self.ws_id = ws_id
        self.wsa_id = wsa_id
        self.target_id = target_id
        self.id = wsa_id if wsa_id else target_id
        assert self.id, "Must provide target_id or wsa_id"
        from runner.process_info import JobInfo
        if isinstance(job_or_id, JobInfo):
            self.bji = job_or_id
            self.job_id = self.bji.job.id
        else:
            self.job_id = job_or_id
            from runner.process_info import JobInfo
            self.bji = JobInfo.get_bound(self.ws_id,self.job_id)

        self.name = name
        self.key_exclude = key_exclude
        self.skip_cache = skip_cache
        self.cacheable = not self.key_exclude and not self.skip_cache



    def _get_cache_dirname(self):
        import os
        from path_helper import PathHelper
        return os.path.join(
                PathHelper.storage,
                'trgimp_cache',
                str(self.ws_id),
                str(self.job_id),
                str(self.id),
                )

    def _get_cache_file(self, method):
        dirname = self._get_cache_dirname()
        import os
        fn = os.path.join(dirname,method+'.json')
        return fn

    def _extract_data(self, stored_data, data_type):
        if 'version' not in stored_data:
            # The original version just stored prot importances.
            stored_data = {'prots': stored_data}
        return stored_data.get(data_type, {})


    def get_importance(self,method, cache_only=False, data_type='prots', compute_cache=None):
        # return {score_code:{uniprot:importance,...},...}
        if self.cacheable:
            import json
            fn = self._get_cache_file(method)
            if os.path.exists(fn):
                # if result is cached, just return it
                with open(fn) as f:
                    data = json.loads(f.read())
                    return self._extract_data(data, data_type)

        if cache_only:
            return None

        compute_cache = compute_cache or Cache()
        input_data = self.gen_input_data(compute_cache)
        score_data = TargetScoreImportance.run_piece(input_data, method, compute_cache)
        result = self.load_score(score_data)
        self.cache_result(result, method)

        return self._extract_data(result, data_type)

    def cache_result(self, result, method):
        if self.cacheable:
            import json
            fn = self._get_cache_file(method)
            from path_helper import make_directory
            dirname = self._get_cache_dirname()
            make_directory(dirname)
            with open(fn, 'w') as f:
                json.dump(result,f)

    def gen_input_data(self, cache, max_input_prots=None):
        """Generates the inputs required to run this target importance job.

        cache should be a Cache() object, and can be shared
        across multiple TrgImpJobResults.
        """
        bji = self.bji

        description = "id=%s: job: %s (%s, %s)" % (self.id, self.name, bji.job_type, self.job_id)

        if verbose:
            logger.info("Preparing input data for %s" % description)

        from scripts.extract_importance import TargetImportance

        CM_Class = TargetImportance.lookup(bji.job_type)
        LoaderClass = CM_Class.loader
        cm_loader = LoaderClass(
                ws_id=str(self.ws_id),
                wsa_id=self.wsa_id,
                target_id=self.target_id,
                bji_or_id=bji,
                key_exclude=self.key_exclude,
                cache=cache,
                max_input_prots=max_input_prots
                )

        input_data = cm_loader.fetch()
        predicted_duration = cm_loader.predict_duration()

        return {
            'job_type': bji.job_type,
            'ws_id': self.ws_id,
            'wsa_id': self.wsa_id,
            'job_id': self.job_id,
            'key_exclude': self.key_exclude,
            'data': input_data,
            'description': description,
            'predicted_duration': predicted_duration
        }


    def load_score(self, score_data):
        from scripts.extract_importance import TargetImportance
        score_names = score_data['score_names']
        for score_type in ['prots', 'pathways']:
            score = score_data.get(score_type, {})
            score_data[score_type] = TargetImportance.prep_scores(
                    score, score_names)
        return score_data


def _get_target_importances_loader(ws_id, wzs_jid, wsa_ids, cache_only, data_type='prots', method=DEFAULT_METHOD):
    return True, (ws_id, wzs_jid, cache_only, data_type, method), wsa_ids, 'wsa_ids'

from dtk.cache import cached_dict_elements
@cached_dict_elements(_get_target_importances_loader)
def get_target_importances(ws_id, wzs_jid, wsa_ids, cache_only, data_type='prots', method=DEFAULT_METHOD):
    logger.info(f"Computing target importances for {len(wsa_ids)} wsas in ws {ws_id}, {wzs_jid}")
    tir = TrgImpResults(ws_id, wzs_jid)

    out = {}
    for wsa_id in wsa_ids:
        df, errs = tir.get_results_df(wsa_id, cache_only, data_type, method)
        if df.empty:
            prot2imp = {}
        else:
            prot_imps = df.groupby('prot')[['prot', 'final_score']].sum()
            prot2imp = dict(prot_imps.reset_index().values)
        out[wsa_id] = (prot2imp, errs)
    return out

class TrgImpResults:
    def __init__(self, ws_id, wzs_jid):
        self.jid2label = trgimp_jids_from_wzs(ws_id=ws_id, wzs_id=wzs_jid, indirect_scores=False)
        self.bjis = JobInfo.get_all_bound(ws_id, self.jid2label.keys())
        self.ws_id = ws_id

        from dtk.score_importance import ScoreImportance
        self.si = ScoreImportance(ws_id, wzs_jid)

    def get_results_df(self, wsa_id, cache_only, data_type='prots', method=DEFAULT_METHOD, compute_cache=None):
        compute_cache = compute_cache or Cache()
        errors = []

        w,n,f = self.si.get_score_importance(wsa_id)
        wsa_score_weights = f

        from collections import namedtuple
        Entry = namedtuple('Entry', 'wsa_id jid joblabel scorename prot score score_weight final_score')
        rows = []
        for bji in self.bjis:
            jid = bji.job.id
            try:
                tijr = TrgImpJobResults(
                        ws_id=self.ws_id,
                        wsa_id=wsa_id,
                        job_or_id=bji,
                        )
                d = tijr.get_importance(method, cache_only=cache_only, data_type=data_type, compute_cache=compute_cache)
                # Will be None if not in cache and cache_only.
                if d:
                    for score_name, protdata in d.items():
                        for prot, score in protdata.items():
                            if score == 0:
                                continue
                            jid_name = self.jid2label[jid]
                            full_score_name = jid_name + "_" + score_name
                            full_score_name = full_score_name.lower()
                            score_weight = wsa_score_weights.get(full_score_name, 0)
                            final_score = score * score_weight
                            rows.append(Entry(wsa_id, jid, jid_name, score_name, prot, score, score_weight, final_score))


            except (
                # TODO: This is a bit much, we shouldn't be expecting
                # all of these...
                    JobInfo.AmbiguousKeyError,
                    AssertionError,
                    IOError,
                    ValueError,
                    KeyError,
                ) as e:
                errors.append(e)
        
        import pandas as pd
        df = pd.DataFrame(rows, columns=Entry._fields)
        return df, errors

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Target Score Importance')
    parser.add_argument('-z', '--wzs-id', type=int, required=True, help="WZS job id")
    parser.add_argument('-w', '--ws-id', type=int, required=True, help="Workspace id")
    parser.add_argument('-c', '--wsa-count', type=int, required=True, help="Number of (top) drugs to compute.")
    parser.add_argument('-s', '--wsa-start', type=int, default=0, help="Number of (top) drugs to skip.")
    parser.add_argument('--skip-cache', action='store_true', help="Ignore cached data and recompute all results")
    parser.add_argument('--method', default=DEFAULT_METHOD, help="Use a non-default method (LOO, peel_previous)")
    parser.add_argument('--mode', default='full', choices=['full', 'output-intermediate', 'run-intermediate'], help="Switch between running the full workflow and outputting and/or running the intermediate data.")
    parser.add_argument('--intermediate-file', help="Intermediate file to read/write data to, if running in intermediate mode.")
    parser.add_argument('--indirect', action='store_true', help="Generate indirect results.")
    args = parser.parse_args()

    t = TargetScoreImportance(args.ws_id, args.wzs_id,
                              wsa_start=args.wsa_start,
                              wsa_count=args.wsa_count,
                              skip_cache=args.skip_cache,
                              imp_method=args.method,
                              indirect_scores=args.indirect,
                              )
    if args.mode == 'full':
        t.run()
    elif args.mode == 'output-intermediate':
        pieces = t.make_pieces()
        data = t.gen_piece_inputs(pieces)
        with open(args.intermediate_file, 'wb') as f:
            f.write(pickle.dumps(data))
    elif args.mode == 'run-intermediate':
        with open(args.intermediate_file, 'rb') as f:
            data = pickle.loads(f.read())
        output = TargetScoreImportance.run_pieces(data)
        pieces = t.make_pieces()
        t.load_scores(pieces, output)

