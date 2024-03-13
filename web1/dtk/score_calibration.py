
class Converter:
    def __init__(self,split_points):
        self.split_points = split_points
    def min_p(self):
        return self.split_points[-1][0]
    def raw_to_p(self,v):
        from math import isnan
        if isnan(v):
            v = 0
        for i,(prob,score) in enumerate(self.split_points):
            if score >= v:
                if i == 0:
                    # Instead, we could add another point at the start of the
                    # json file with the minimum score value ever seen, and
                    # the equivalent probability (which would be very slightly
                    # less than 1), and return that probability in this case.
                    # Since we're more interested in points at the other end
                    # of the distribution, this probably doesn't matter.
                    return 0.99
                last_p,last_s = self.split_points[i-1]
                slope = (prob-last_p)/(score-last_s)
                delta_v = v - last_s
                return last_p + slope * delta_v
        return self.min_p()

from dtk.lazy_loader import LazyLoader
class ScoreCalibrator(LazyLoader):
    def calibrate(self,role,code,is_moa,vec):
        cvt = self.lookup[(role,'moa' if is_moa else 'mol',code)]
        import math
        return [1-cvt.raw_to_p(v) for v in vec]
    def all_roles(self):
        return set(k[0] for k in self.lookup.keys())
    def _json_file_loader(self):
        # XXX Eventually, this should be a versioned file in the
        # XXX workspace, in which case the version number would
        # XXX become a keyword parameter to this class, or something.
        # XXX For now, use the checked-in test version.
        import os
        from path_helper import PathHelper
        return os.path.join(
                PathHelper.repos_root,
                'experiments/score_calibration',
                'qa01.score_calibration.json',
                )
    def _lookup_loader(self):
        import json
        curves = json.load(open(self.json_file))
        try:
            self.meta = curves.pop('__meta__')
        except KeyError:
            self.meta = {}
        return {
                (role,stype,code):Converter(curves[role][stype][code])
                for role in curves
                for stype in curves[role]
                for code in curves[role][stype]
                }
    def _max_val_loader(self):
        return 1-min(cvt.min_p() for cvt in self.lookup.values())

class FMCalibrator(LazyLoader):
    _kwargs=['fm']
    warn_input_threshold=1
    def calibrate(self,logscale=False):
        assert self.fm.sample_key == 'wsa'
        key_vec = self.fm.sample_keys
        from dtk.moa import is_moa_score
        is_moa = is_moa_score(key_vec)
        stype = 'moa' if is_moa else 'mol'
        if True:
            # a missing key will fail eventually in the loop below,
            # but this exposes errors faster
            # XXX have some fallback eventually?
            missing = set(
                    (role,stype,code)
                    for col,role,code in self.col_data
                    if (role,stype,code) not in self.sc.lookup
                    )
            if missing:
                raise RuntimeError(f'no cal data for {missing}')
        # XXX This is a bit of a hack, because although data_as_array hides
        # XXX the underlying representation from us, at the bottom we
        # XXX convert (back) to csr since that's what the underlying Repr
        # XXX store code expects. Also, the array isn't really very sparse
        # XXX at that point -- maybe we want to apply a floor as well (say,
        # XXX to every probability under 0.8?) to get some sparseness back
        # XXX (as well as converting 0's to NaN's which I think is what
        # XXX doesn't get saved in sparse arrays).
        # XXX The Repr should provide better encapsulation for this.
        nda = self.fm.data_as_array()
        import math
        jobs_per_role = {
                k:len(v)
                for k,v in self.sc.meta.get('per_role_job_ids',{}).items()
                }
        for col,role,code in self.col_data:
            if jobs_per_role:
                input_cnt = jobs_per_role[role]
                if input_cnt <= self.warn_input_threshold:
                    print('Warning: only',input_cnt,'job(s) in',
                        role,code,'calibration curve',
                        )
            ordering = list(zip(key_vec,nda[:,col]))
            src_bji = self.bji_cache[role]
            ordering = src_bji.remove_workspace_scaling(code,ordering)
            data_vec = [x[1] for x in ordering]
            data_vec = self.sc.calibrate(role,code,is_moa,data_vec)
            if logscale:
                # Put this in logscale, but then compress back to 0-1
                # range (which makes it more convenient for WZS plots).
                # Logscale provides more separation between meaningful
                # scores and noise. Note that 'top' represents the max
                # possible output value across all data in the calibrator,
                # not the max for this vector. This preserves relative
                # scaling between scores, which is the whole point.
                top = -math.log10(1-self.sc.max_val)
                data_vec = [-math.log10(1-x)/top for x in data_vec]
            nda[:,col] = data_vec
        from scipy import sparse
        self.fm.data = sparse.csr_matrix(nda)
    def _sc_loader(self):
        return ScoreCalibrator()
    def _col_data_loader(self):
        self.bji_cache = {}
        result = []
        errors = []
        codes = self.fm.spec.get_codes()
        features = self.fm.feature_names
        if len(codes) != len(features):
            errors.append((
                    'length mismatch',
                    f'{len(codes)} codes for {len(features)} features',
                    ))
        from runner.process_info import JobInfo
        for i,(code,name) in enumerate(zip(codes,features)):
            job_id,dc_code = code.split('_')
            role = name[:name.rindex('_')]
            if role not in self.bji_cache:
                self.bji_cache[role] = JobInfo.get_bound(None,job_id)
            if role.startswith('_'):
                errors.append(('bad role name',f"{role} for job {job_id}"))
            if not name.endswith('_'+dc_code.upper()):
                errors.append((
                        'alignment error',
                        f"{code} doesn't match {name}",
                        ))
            result.append((i,role,dc_code))
        if errors:
            raise RuntimeError("Score reconstruction failed:\n"
                    +"\n".join(f"{err}: {detail}" for err,detail in errors)
                    )
        return result

def get_recent_active_scoresets(dpi_dataset):
    result = []
    from browse.models import ScoreSet,VersionDefault
    moa_ws = set(VersionDefault.objects.filter(
            file_class='DpiDataset',
            choice=dpi_dataset,
            ws_id__isnull=False,
            ).values_list('ws_id',flat=True))
    seen_ws = set()
    for ss in ScoreSet.objects.filter(
            ws__active=True,
            desc='RefreshFlow',
            ).order_by('-id'):
        if ss.ws_id not in moa_ws:
            continue
        if ss.ws_id in seen_ws:
            continue
        seen_ws.add(ss.ws_id)
        result.append(ss)
    result.sort(key=lambda ss:ss.id)
    return result

class ScoreFetcher(LazyLoader):
    '''Class for fetching cal'd or uncal'd scores by code.
    '''
    @classmethod
    def from_global_code(cls,global_code,ws):
        sf = cls()
        job_id,sf.code = global_code.split('_')
        from runner.process_info import JobInfo
        sf.bji = JobInfo.get_bound(ws,int(job_id))
        return sf
    def _sc_loader(self):
        return ScoreCalibrator()
    def _label_loader(self):
        return f'{self.bji.job.role} ({self.bji.job.id}_{self.code})'
    def _ordering_loader(self):
        cat = self.bji.get_data_catalog()
        return cat.get_ordering(self.code,True)
    def _calibrated_ordering_loader(self):
        ordering = self.bji.remove_workspace_scaling(self.code,self.ordering)
        if not ordering:
            return []
        key_vec,data_vec = zip(*ordering)
        from dtk.moa import is_moa_score
        is_moa = is_moa_score(key_vec)
        role = self.bji.job.role
        data_vec = self.sc.calibrate(role,self.code,is_moa,data_vec)
        return list(zip(key_vec,data_vec))

