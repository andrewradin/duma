from dtk.lazy_loader import LazyLoader

class BulkFetcher(LazyLoader):
    _kwargs=['scoreset_id']
    phases = (2,3)
    outcomes = ('pass','fail','ongo')
    @classmethod
    def make_wsa_map(cls,wsa_ids):
        from browse.models import WsAnnotation
        return {
                wsa.id:wsa
                for wsa in WsAnnotation.objects.filter(id__in=wsa_ids)
                }
    def _scoreset_loader(self):
        from browse.models import ScoreSet
        return ScoreSet.objects.get(pk=self.scoreset_id)
    def _ws_loader(self):
        return self.scoreset.ws
    def _ct_drugsets_loader(self):
        '''Returns {(phase,outcome):{wsa_ids},...}.

        i.e. all CT status drugsets with structured keys
        These allow assigning outcomes to individual WSAs based on
        which set they're in.
        '''
        ct_drugsets = {
                (phase,outcome):self.ws.get_wsa_id_set(f'ct-ph{phase}{outcome}')
                for phase in self.phases
                for outcome in self.outcomes
                }
        # before returning, verify that within a given phase, the CT
        # outcome sets are disjoint
        for phase in self.phases:
            src_drugsets = [k for k in ct_drugsets if k[0] == phase]
            all_ids = set.union(*[
                    ct_drugsets[k]
                    for k in src_drugsets
                    ])
            assert len(all_ids) == sum(
                    len(ct_drugsets[k])
                    for k in src_drugsets
                    ),f'Phase {phase} CT outcome drugsets overlap'
        return ct_drugsets
    def _all_wsa_ids_loader(self):
        return set.union(*self.ct_drugsets.values())
    def _wsa_map_loader(self):
        return self.make_wsa_map(self.all_wsa_ids)
    def _moa_wsa_map_loader(self):
        from dtk.moa import make_wsa_to_moa_wsa
        id2moa_id = make_wsa_to_moa_wsa(
                list(self.all_wsa_ids),
                pick_canonical=True,
                )
        moa_id2wsa = self.make_wsa_map(id2moa_id.values())
        return {
                base_id:moa_id2wsa[moa_id]
                for base_id,moa_id in id2moa_id.items()
                }
    job_type_map = {
                'wzs':'full',
                'wzs-test':'fold1',
                'wzs-train':'fold2',
                }
    def _wzs_map_loader(self):
        from runner.process_info import JobInfo
        return {
                self.job_type_map[ssj.job_type]
                        : JobInfo.get_bound(self.ws,ssj.job_id)
                for ssj in self.scoreset.scoresetjob_set.filter(
                        job_type__startswith='wzs',
                        )
                }
    def _train_map_loader(self):
        result = {}
        for name in self.job_type_map.values():
            bji = self.wzs_map[name]
            result[name] = bji.get_train_ids()
        return result
    def _moa_ranker_map_loader(self):
        result = {}
        from dtk.scores import Ranker
        for name in self.job_type_map.values():
            bji = self.wzs_map[name]
            cat = bji.get_data_catalog()
            result[name] = Ranker(cat.get_ordering('wzs', True))
        return result
    def _ctas_loader(self):
        '''{wsa_id:latest_cta,...} for any specified wsa_ids with CTAs.'''
        from moldata.models import ClinicalTrialAudit
        return {
                x.wsa_id:x
                for x in ClinicalTrialAudit.get_latest_ws_records(self.ws.id)
                }

class RetroMolData(LazyLoader):
    _kwargs=['wsa_id','bulk_fetch']
    def _wsa_loader(self):
        return self.bulk_fetch.wsa_map[self.wsa_id]
    def _moa_wsa_loader(self):
        try:
            return self.bulk_fetch.moa_wsa_map[self.wsa_id]
        except KeyError:
            # this could happen if there's a versioning issue,
            # or if there's an MOA agent, but it's not imported
            # into the workspace
            return None
    def _moa_id_loader(self):
        # this is for TSV file construction; note that we will load
        # the id if we have an MOA, but not vice-versa
        if self.moa_wsa:
            return self.moa_wsa.id
        return None
    def get_rank(self,run):
        ranker = self.bulk_fetch.moa_ranker_map[run]
        return ranker.get(self.moa_id)
    def _full_rank_loader(self):
        return self.get_rank('full')
    def _fold1_rank_loader(self):
        return self.get_rank('fold1')
    def _fold2_rank_loader(self):
        return self.get_rank('fold2')
    def get_blind(self,run):
        return self.moa_id not in self.bulk_fetch.train_map[run]
    def _full_blind_loader(self):
        return self.get_blind('full')
    def _fold1_blind_loader(self):
        return self.get_blind('fold1')
    def _fold2_blind_loader(self):
        return self.get_blind('fold2')
    def get_combined_rank(self,blinded):
        ranks_and_folds = [
                (getattr(self,f'fold{fold}_rank'),fold)
                for fold in (1,2)
                if getattr(self,f'fold{fold}_blind') == blinded
                ]
        # If a molecule is part of the overall WZS training set, it should
        # be blinded in one fold. If not, it will be blinded in 2 folds.
        # So, if blinded is False, it's possible this list will be empty.
        if not ranks_and_folds:
            return None,None
        rank,src_fold = ranks_and_folds[0]
        # In case of multiple source folds, take best (lowest) rank;
        # report the fold the rank actually came from.
        for tmp_rank,tmp_fold in ranks_and_folds[1:]:
            if tmp_rank < rank:
                rank = tmp_rank
                src_fold = tmp_fold
        return rank,src_fold
    def get_outcome(self,want_phase):
        for (phase,outcome),ids in self.bulk_fetch.ct_drugsets.items():
            if phase != want_phase:
                continue
            if self.wsa_id in ids:
                return outcome
        return ''
    def _ph2_outcome_loader(self):
        return self.get_outcome(2)
    def _ph3_outcome_loader(self):
        return self.get_outcome(3)
    def get_link(self,want_phase):
        try:
            cta = self.bulk_fetch.ctas[self.wsa_id]
        except KeyError:
            return ''
        if want_phase == 2:
            return cta.ph2_url
        elif want_phase == 3:
            return cta.ph3_url
        else:
            raise RuntimeError(f'invalid phase: {want_phase}')
    def _ph2_link_loader(self):
        return self.get_link(2)
    def _ph3_link_loader(self):
        return self.get_link(3)
    def _indication_loader(self):
        return self.wsa.indication
    def get_ct_status(self,want_phase):
        try:
            cta = self.bulk_fetch.ctas[self.wsa_id]
        except KeyError:
            return ''
        if want_phase == 2:
            return cta.ph2_status
        elif want_phase == 3:
            return cta.ph3_status
        else:
            raise RuntimeError(f'invalid phase: {want_phase}')
    def _ph2_status_loader(self):
        return self.get_ct_status(2)
    def _ph3_status_loader(self):
        return self.get_ct_status(3)
    tsv_columns = [
            'wsa_id',
            'moa_id',
            'fold1_rank',
            'fold2_rank',
            'full_rank',
            'fold1_blind',
            'fold2_blind',
            'full_blind',
            'ph2_outcome',
            'ph2_link',
            'ph3_outcome',
            'ph3_link',
            'indication',
            'ph2_status',
            'ph3_status',
            ]
    @classmethod
    def write_tsv(cls,fn,items):
        with open(fn,'w') as fd:
            for row in [cls.tsv_columns]+[
                    [getattr(mol,attr) for attr in cls.tsv_columns]
                    for mol in items
                    ]:
                fd.write('\t'.join(str(x) for x in row)+'\n')
    @classmethod
    def from_tsv(cls,fn):
        result = []
        import re
        int_re=r'(.*_id|.*_rank|.*_status|indication)$'
        bool_re=r'.*_blind$'
        from dtk.files import get_file_records
        header = None
        for rec in get_file_records(fn,keep_header=True):
            if not header:
                header = rec
                continue
            mol = cls()
            for k,v in zip(header,rec):
                if re.match(int_re,k):
                    v = '' if v == '' else int(v)
                elif re.match(bool_re,k):
                    v = (v == 'True')
                setattr(mol,k,v)
            result.append(mol)
        # XXX The mols in result don't have the wsa or moa_wsa fields
        # XXX populated. For now that's ok because they're not used
        # XXX in analysis. If they're needed later, we can populate
        # XXX them efficiently here by gathering the wsa_ids, doing a
        # XXX single query, and then distributing the results.
        return result

support_case_labels = ['supported','unsupported','unclear']
SUPPORTED_THRESHOLD = 1000
UNCLEAR_THRESHOLD = 3000
def rank2support(rank):
    if rank <= SUPPORTED_THRESHOLD:
        return 'supported'
    if rank > UNCLEAR_THRESHOLD:
        return 'unsupported'
    return 'unclear'

def label_molecules(molecules,phase,blinded):
    '''Returns a list of LabeledMol namedtuples.

    One object is returned for each input molecule with a pass/fail
    outcome in the input phase.
    src_fold is set per the input blinded flag.
    support is the platform support level in the specified fold.
    '''
    from collections import namedtuple
    LabeledMol = namedtuple('LabeledMol','outcome src_fold support mol')
    result = []
    for mol in molecules:
        # XXX Eventually, support phase 3 transition as an option here.
        # XXX This involved only keeping rows for which we can extract
        # XXX a phase 2 end date, and determine if it's sufficiently far
        # XXX in the past that a phase 3 should exist. Then, set the outcome
        # XXX to 'pass' if a phase 3 trial exists, and 'fail' if it doesn't.
        attr = f'ph{phase}_outcome'
        outcome = getattr(mol,attr)
        if outcome not in ('pass','fail'):
            # this includes ongoing, and unlabeled CT statuses
            # (where the outcome is blank)
            continue
        rank,src_fold = mol.get_combined_rank(blinded)
        if rank is None:
            # This usually means we're looking for non-blinded data, but the
            # molecule wasn't in the overall WZS training set; just drop this
            # molecule (we barely use the non-blinded data anyway).
            continue
        support = rank2support(rank)
        result.append(LabeledMol(outcome,src_fold,support,mol))
    return result

def count_by_support(labeled):
    '''Return a Counter counting labeled molecules.

    Input is like the output of label_molecules().
    Each molecule is counted twice, once grouped by (support,src_fold)
    and once grouped by (support,'sum').
    '''
    cases = [
            (mol.support,mol.src_fold)
            for mol in labeled
            ]
    # now duplicate the list, without the source side tagging
    cases += [(x[0],'sum') for x in cases]
    # now count the keys; this produces the counts of how many molecules
    # had what level of support, both overall and by side
    from collections import Counter
    ctr = Counter(cases)
    return ctr

def calc_fractions(pass_cnts,fail_cnts):
    '''Returns {support:fraction_of_passing_cts,...} for each support level.'''
    result = {}
    for key in support_case_labels:
        n_passed = pass_cnts.get((key,'sum'),0)
        n_failed = fail_cnts.get((key,'sum'),0)
        try:
            result[key] = n_passed/(n_passed+n_failed)
        except ZeroDivisionError:
            result[key] = 0
    return result

def calc_subset_stats(labeled):
    '''Returns (pass_cnts,fail_cnts,fracs).

    pass_cnts and fail_cnts are the results of count_by_support on passing
    and failing CTs respectively.
    fracs is the result of calculate_fractions
    '''
    passed = [x for x in labeled if x.outcome == 'pass']
    failed = [x for x in labeled if x.outcome == 'fail']
    pass_cnts = count_by_support(passed)
    fail_cnts = count_by_support(failed)
    fracs = calc_fractions(pass_cnts,fail_cnts)
    return pass_cnts,fail_cnts,fracs

def format_stats_for_display(pass_cnts,fail_cnts,fracs):
    '''Input calc_subset_stats results.
       Returns a dict keyed as fracs, but with values as (frac, total count)
    '''
    summed = pass_cnts+fail_cnts
    sums = pull_sums_from_cnts(summed)
    fracs_w_cnts = {}
    for k in sums:
        assert k in fracs
        fracs_w_cnts[k] = (fracs[k], sums[k])
    return fracs_w_cnts

def pull_sums_from_cnts(cnts):
    '''
    Input expected to be output from count_by_support
    Returns dict keyed by support status and values of sum counts
    Convience function to pull out a subset of data from the counts dicts
    '''
    return {k[0]:v for k,v in cnts.items() if k[1]=='sum'}

def calc_enrich(full_pass_cnts, full_fail_cnts, unclear=None):
    '''
    Input pass and fail counts from count_by_support (called in calc_subset_stats
    Optional param dictating what to do with the unclear support counts
    '''
    from dtk.stats import fisher_exact
    # create a confusion matrix
    # [[supported & passed, supported & failed],
    #  [unsupported & passed, unsupported & failed]]
    pass_cnts = pull_sums_from_cnts(full_pass_cnts)
    fail_cnts = pull_sums_from_cnts(full_fail_cnts)
    a = pass_cnts.get('supported',0)
    b = fail_cnts.get('supported',0)
    c = pass_cnts.get('unsupported',0)
    d = fail_cnts.get('unsupported',0)
    if unclear is None:
        pass
    elif unclear == 'supported':
        a += pass_cnts.get('unclear',0)
        b += fail_cnts.get('unclear',0)
    elif unclear == 'unsupported':
        c += pass_cnts.get('unclear',0)
        d += fail_cnts.get('unclear',0)
    else:
        assert False, f'Unexpected param passed for unclear: {unclear}'
    mat = [[a,b],[c,d]]
# use fishers exact bc numbers will be smallish
    return fisher_exact(mat)

