#!/usr/bin/env python3

import sys
from path_helper import PathHelper,make_directory
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import pickle
import os
from dataclasses import dataclass
import logging
from runner.process_info import JobInfo
logger = logging.getLogger(__name__)



@dataclass
class LRParams:
    test_train_split: bool = False
    C: float = 1.0
    method: str = 'normal'
    test_frac: float = 0.2
    penalty: str = 'l2'
    class_weight: str = None
    stratify_idxs: list = None


def stratify_weights(mat, target, strat_idxs):
    """
    Generates a set of sample weights that class-balance across stratification groups.

    Standard class balancing would result in each class having different demographic distributions,
    based on prevalence in those demos. Each demographic would still be class-imbalanced.

    This method instead class-balances each stratified subgroup separately.
    After applying weights, if you pick any stratified subgroup, it will have equal weight for
    cases and non-cases.  The overall demographic weight distribution will match the demographic
    distribution of cases.

    Another way of thinking about it, imagine matching each case with a control that had identical
    demographics, and throwing out all the others.  This is the same idea, but instead of picking a
    single control and throwing the others out, it takes them all but reweights them such that their
    contribution sums to that of the corresponding case.
    """
    import numpy as np
    from scipy import sparse
    from dtk.numba import unique_bool_rows, bools_to_int64
    target = np.reshape(target.astype(np.bool), (-1, 1))
    strat_mat = sparse.hstack([mat[:, strat_idxs], target], format='csr').toarray()
    vals, counts = unique_bool_rows(strat_mat, return_counts=True)
    

    vals = [tuple(x) for x in vals]
    val2counts = dict(zip(vals, counts))

    total_case = target.sum()

    val2weights = {}

    for val in vals:
        if val[-1] == 0:
            # Handled symmetrically, ignore these.
            continue

        case_count = val2counts[val]
        ctrl_val = (*val[:-1], 0)
        ctrl_count = val2counts.get(ctrl_val, 0)

        # Each group is weighted by # of cases in it.
        # Within each group, half the weight goes to cases, half to controls.
        stratum_weight = case_count / total_case
        class_weight = stratum_weight / 2

        # ctrl_count can be 0, but in that case this weight
        # won't get assigned to anything.
        per_ctrl_weight = class_weight / ctrl_count
        per_case_weight = class_weight / case_count

        if ctrl_count == 0:
            # This is no good, we have nothing to balance the cases against.
            # Something is going to get imbalanced here, choices include:
            # 1) Let the ctrl weight get dropped, which means this stratum only retains half its weight.
            #    This also leaves cases vs ctrls imbalanced.
            # 2) Reassign ctrl weight to cases, which keeps stata balanced but further imbalances cases vs controls.
            # 3) Set both weights to 0, which effectively drops all these cases.
            # 4) "Move" these cases into a different nearby stratum that has controls
            #
            # Option 3 is the least biased choice, but drops data.
            # The places this choice could impact results, such as rare indications where all demos are sparsely
            # populated, are unfortunately the situations where dropping data is the most hurtful.
            # Option 4 is hard to implement and ideally incorporates an understanding of similar features (e.g. age vars)

            # Doing nothing implements option #1.
            pass

            # This implements option #2
            # per_case_weight *= 2

        val2weights[bools_to_int64(np.array(val, dtype=bool))] = per_case_weight * total_case * 2
        val2weights[bools_to_int64(np.array(ctrl_val, dtype=bool))] = per_ctrl_weight * total_case * 2

    # There can be missing values if there are only controls and no cases for
    # a subgroup - in which case, the weights of those controls are all 0.
    return [val2weights.get(bools_to_int64(x), 0) for x in strat_mat]


def stratified_cont_tables(mat, target, stratify_idxs):
    """Constructs contingency tables for each feature vs target, for each strata.

    mat is a sparse input array, usually from faers.

    Returns [num_strata, num_features, 2, 2], where entry[stratum, feature, :, :] is a 2x2 contingency
    table in that stratum for that feature against the target.

    Each stratum's contingency table is separately class-balanced to have an equal number of target and non-target
    samples.  This does not affect the odds ratio, but generally reduces the reported significance of results.
    This follows a 'matched experiment' interpretation, where for each case we found a single control with identical demos.
    """
    import numpy as np
    from dtk.numba import sparse_unique_bool_rows
    vals, counts, idxs = sparse_unique_bool_rows(mat, stratify_idxs, return_counts=True, return_idx_groups=True)
    num_strat = len(vals)
    num_feats = mat.shape[1]
    cont_tables = np.zeros((num_strat, num_feats, 2, 2))

    strata_prevalence = np.zeros(len(vals))

    for strat_idx, cur_idxs in enumerate(idxs):
        targ_idxs = [x for x in cur_idxs if target[x]]
        not_targ_idxs = [x for x in cur_idxs if not target[x]]
        strat_total = len(cur_idxs)
        feat_totals = mat[cur_idxs].sum(axis=0)
        feat_and_targ = mat[targ_idxs].sum(axis=0)
        feat_and_not_targ = mat[not_targ_idxs].sum(axis=0)

        # Convert to float, they are ints before this, so any rescaling
        # would cause truncation.
        a = np.asarray(feat_and_targ, dtype=float)
        b = np.asarray(len(targ_idxs) - a, dtype=float)
        c = np.asarray(feat_and_not_targ, dtype=float)
        d = np.asarray(strat_total - a - b - c, dtype=float)

        # Rescale to class-balance each stratum.
        if len(not_targ_idxs) > 0:
            scale = len(targ_idxs)/len(not_targ_idxs)
            c *= scale
            d *= scale

        entries = [np.asarray(x).reshape(-1, 1) for x in [a,b,c,d]]

        cur_table = np.concatenate(entries, axis=-1).reshape(-1, 2, 2)

        cont_tables[strat_idx] += cur_table

        strata_prevalence[strat_idx] = len(targ_idxs)
    return cont_tables, vals, counts, strata_prevalence
    


class LR:
    def __init__(self, feature_mat, target, lr_params):
        self.feature_mat = feature_mat
        self.target = target
        self.lr_params = lr_params

    def run(self):
        if self.lr_params.test_train_split:
            from sklearn.model_selection import train_test_split
            # TODO: Control the randomness here.
            X_train, X_test, y_train, y_test = train_test_split(
                self.feature_mat,
                self.target,
                stratify=self.target,
                test_size=self.lr_params.test_frac,
            )
            lr = self.fit_lr(
                X_train,
                y_train,
            )
            accuracy, wtd_accuracy = self.eval(lr, X_test, y_test)
            print(f"Validation Acc: Full {accuracy:.4f} vs Wtd: {wtd_accuracy:.4f}")
        
        lr = self.fit_lr(
            self.feature_mat,
            self.target,
        )
        accuracy, wtd_accuracy = self.eval(lr, self.feature_mat, self.target)
        print(f"Train Acc: Full {accuracy:.4f} vs Wtd: {wtd_accuracy:.4f}")
        return lr, accuracy, wtd_accuracy
    
    def compute_pvalues(self, lr):
        from sklearn.feature_selection import chi2, SelectFdr
        from collections import defaultdict
        import scipy.stats as st
        import scipy.sparse as sparse
        import scipy.special as special
        import numpy as np
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.tools.numdiff import approx_hess
        from os import stat

        #Using the orignal full matrix map to get names on clean feature matrix

        # B-H correction on chi-2 p-values
        # fdr = SelectFdr(chi2)
        # fdr.fit(self.feature_mat, self.target)
        # df['chi2 - pvalue'] = fdr.pvalues_

        # Calculate confidence intervals, pvalues using standard error of regression for coefficients.
        # Mimic a statsmodels output for Logit model

        params = np.append(lr.intercept_,lr.coef_)
        predictions = lr.predict_proba(self.feature_mat)[:,1]

        newX = sparse.hstack((sparse.csr_matrix(np.ones((len(self.target),1))), self.feature_mat))
        target_sums = np.sum(self.target)
        print('number of positive cases:', target_sums, 'out of total:', len(self.target))
        JobInfo.report_info(f"Indication Count (w/ demos): {target_sums} / {len(self.target)}")

	    ###### KEY EQUATION - Standard Error Calculation for betas ##########
	    # np.sqrt(-np.linalg.inv(approx_hess(params, est2.model.loglike)).diagonal())
	    ################
        print('assembling hessian matrix')
        # lambdas based on statsmodels functions but adapted for sparse matrices
        cdf = lambda X: [np.float16(special.expit(x)) for x in np.array(X)]
        L = cdf(newX.dot(params))
        L = np.asarray(L)
        temp_mat = sparse.csr_matrix(newX.T.multiply(L*(1-L)), dtype=np.float32)
        del L
        hess = -temp_mat.dot(newX).toarray()
        del newX

        print('attempting to calculate variance of coefficients')

        print(np.linalg.cond(hess))


        try:
            diag = np.linalg.inv(hess).diagonal()
        except np.linalg.linalg.LinAlgError:
            print('Failed to invert hessian, using pseudoinverse')
            diag = np.linalg.pinv(hess).diagonal()
        # In at least some cases, the diag matrix contains positive values,
        # which will cause a warning and return nan when passed through sqrt.
        # This propagates through to a nan in the corresponding p-value.
        # XXX We can make the warning go away by replacing positive values with
        # XXX nan. We might need to do a similar thing with zeros in sd_b, as
        # XXX the zs_b calculation line sometimes reports a divide by zero
        # XXX warning.
        print('%d of %d values on diagonal are positive'%(
                sum(x > 0 for x in diag),
                len(diag),
                ))
        # print(diag)
        sd_b = np.sqrt(-diag)
        # print(sd_b)
        zs_b = params/ sd_b
        # print(zs_b)
        p_values =[st.norm.sf(abs(i))*2 for i in zs_b]
        # print(p_values)

        #### TODO Figure out why male and female columns have a negative variance sometimes####
        #  m_ix = self.feature_names.tolist().index('m')
        #  f_ix = self.feature_names.tolist().index('f')

        # Not going to report intercept...

        sd_b = sd_b[1:]
        zs_b = zs_b[1:]
        p_values = p_values[1:]
        params = params[1:]

        return sd_b,zs_b,p_values
    
    def assemble_df(self, lr, feature_names):
        df = pd.DataFrame()
        col_sums  = np.squeeze(np.asarray(self.feature_mat.sum(axis=0)))

        num_nonzero = len([x for x in lr.coef_[0] if x != 0])
        JobInfo.report_info(f"LR NonZero Coefs: {num_nonzero}")
        df['Total Counts'] = col_sums

        df['Feature'] = feature_names
        df['Coefficients'] = lr.coef_[0]
        df['Odds Ratio'] = df['Coefficients'].apply(lambda x: np.exp(x))
        df["Standard Error"], df["Z"], df["P-Value"] = self.compute_pvalues(lr)
        df.sort_values("P-Value", inplace=True)

        return df


    def fit_lr(self, feature_mat, target):
        params = self.lr_params
        logger.info('Fitting logistic regression model')

        if params.class_weight == 'stratified':
            class_weight = None
            sample_weight = stratify_weights(feature_mat, target, params.stratify_idxs)
            # Mask out the stratified features.
            logger.info("Masking out stratified features")
            feature_mat[:, params.stratify_idxs] = 0
            logger.info("Masked")
        else:
            class_weight = params.class_weight
            sample_weight = None

        if params.method == 'normal':

            from sklearn.linear_model import LogisticRegression
            penalty = params.penalty
            # XXX liblinear is the old default, which is now coded
            # XXX explicitly to get rid of a deprecation warning, but
            # XXX there's no reason to believe it's the best option
            #
            # XXX saga is the only supported solver for elasticnet constraints.
            # XXX it seems to be a fair bit slower than liblinear (though, elasticnet
            # XXX is a harder problem, to be fair) and isn't parallelized (there is an
            # XXX n_jobs parameter, but it is only used for cross validation).
            solver = 'saga' if penalty == 'elasticnet' else 'liblinear'
            if penalty == 'elasticnet':
                kwargs = {'l1_ratio': 0.9}
            else:
                kwargs = {}
            lr = LogisticRegression(
                    penalty=params.penalty,
                    class_weight=class_weight,
                    C=params.C,
                    solver=solver,
                    **kwargs,
                    )
        elif params.method == 'EB':
            from skbayes.linear_models import EBLogisticRegression
            lr = EBLogisticRegression(verbose=True, alpha=params.C)
        elif params.method == 'VB':
            from skbayes.linear_models import VBLogisticRegression
            lr = VBLogisticRegression(verbose=True, a=params.C, b=params.C)
        else:
            raise NotImplementedError("This logistic regression method is not implemented")
        
        
        lr.fit(feature_mat, target, sample_weight=sample_weight)
        return lr

    def eval(self, lr, X, y):
        accuracy = lr.score(X, y)
        if self.lr_params.class_weight:
            tot = len(y)
            pos = sum(y)
            wt = float(pos)/float(tot)
            wts = np.array([wt]*tot)
            wts[y] = 1.0-wt
            wtd_accuracy = lr.score(X, y, wts)
        else:
            wtd_accuracy = accuracy
        
        y_pred = lr.predict_proba(X)[:, 1]
        from sklearn.metrics import roc_auc_score
        roc = roc_auc_score(y, y_pred)

        print("".join(['#']*40))
        print('Accuracy: ',accuracy)
        print('Weighted accuracy: ',wtd_accuracy)
        print('ROC: ', roc)
        print("".join(['#']*40))
        return accuracy, wtd_accuracy



class FAERS_LR():
    def  __init__(self, cds, indir, outdir, C, penalty, class_weight, demo_covariates, method, prefiltering, split_drug_disease, autoscale_C=False, ignore_features=None, ignore_single_indi=False):
        self.cds = cds
        self.indir = indir
        self.outdir = outdir
        self.demo_covariates = demo_covariates
        self.C = float(C)
        self.penalty = penalty
        self.method = method
        if class_weight == 'None':
            class_weight = None
        self.class_weight = class_weight
        self.prefiltering = prefiltering
        self.split_drug_disease = split_drug_disease
        self.autoscale_C = autoscale_C
        self.ignore_features = ignore_features
        self.ignore_single_indi = ignore_single_indi
        self._save_intermediates = False
        from dtk.faers import ClinicalEventCounts
        self.cec = ClinicalEventCounts(self.cds)
        self._out_results_fn = os.path.join(self.outdir,'all_model_results.tsv')

    def compute_background(self):
        """
        Not part of the normal FAERS run, this instead computes the model for
        every indication and outputs the results, which can be used to analyze
        the 'background' coefficient / p-value for each indication.
        """
        import shutil
        def run_for_indi(indi):
            indi_idx = self.cec._indi_cols.index(indi)
            cnt = self.cec._indi_fm[:, indi_idx].sum()
            logger.info(f"{indi} has {cnt}")
            if cnt < 100:
                logger.info(f"Skipping {indi}")
                return

            try:
                out_fn = os.path.join(self.outdir, f'bg_results.{indi}.tsv')
                self.build_matrix([indi])
                self.fit_and_summarize(out_fn)
                logger.info(f"Succeeded for {indi}")
            except:
                logger.info(f"Failed for {indi}")

        indis = list(self.cec._indi_cols)
        import random
        random.shuffle(indis)
        logger.info(f"Computing background across {len(indis)} indications")

        from dtk.parallel import pmap
        list(pmap(run_for_indi, indis, progress=True))


    def build_matrix(self, indi_set):
        assert not isinstance(indi_set, str), "Should be a list of indi names"
        self.feature_mat,self.target,self.feature_names,self.feature_types = self.cec.get_matrix(
                                                        indi_set,
                                                        self.demo_covariates,
                                                        output_types=True
                                                        )
        print('Feature Matrix Assembled. Dimensions: ', self.feature_mat.shape)
        self._clean_matrix()

        if self.autoscale_C:
            # 1000 cases gives C=1.0 with elasticnet a reasonable number of features in general,
            # so scale around that.
            num_cases = np.sum(self.target)
            c_scale = 1000 / num_cases
            self.C *= c_scale
            logger.info(f"Rescaled C to {self.C:.4f}")

        if self._save_intermediates:
            sparse.save_npz(os.path.join(self.indir,'feature_mat'), self.feature_mat)
            np.save(os.path.join(self.indir,'target.npy'), self.target)
            np.save(os.path.join(self.indir,'feature_names.npy'), self.feature_names)
            from os import stat
            print('compressed feature matrix size:', stat(os.path.join(self.indir,'feature_mat.npz')).st_size)
            print('compressed target array size:', stat(os.path.join(self.indir,'target.npy')).st_size)
    
    def _clean_matrix(self, thresh=0.999):
        orig_shape = self.feature_mat.shape

        self.feature_names = np.array(self.feature_names)
        self.feature_types = np.array(self.feature_types)

        if self.ignore_features:
            with open(self.ignore_features) as f:
                to_ignore = {x.strip() for x in f}
            keep_idxs = [i for i, x in enumerate(self.feature_names) if x not in to_ignore]
            logger.info(f"Ignoring {len(self.feature_names) - len(keep_idxs)} found of {len(to_ignore)} to ignore")
            self.feature_names = self.feature_names[keep_idxs]
            self.feature_types = self.feature_types[keep_idxs]
            self.feature_mat = self.feature_mat[:, keep_idxs]

        if self.prefiltering != 'variance':
            # We still run this, but only use it to remove empties.
            # Using a threshold of 1.0 removes only columns that are 100% empty.
            # Some of the plotting gets unhappy if you leave those in, and they're just
            # wasting compute.
            thresh = 1.0

        # Threshold parameter indicates the maximum percentage of identical values allowed within a feature.
        print('reducing feature matrix...')
        var = thresh*(1-thresh)
        from sklearn.feature_selection import VarianceThreshold

        vt = VarianceThreshold(var)
        self.feature_mat = vt.fit_transform(self.feature_mat)
        support = vt.get_support()
        self.feature_names = np.array([x for sup, x in zip(support, self.feature_names) if sup])
        self.feature_types = np.array([x for sup, x in zip(support, self.feature_types) if sup])
        logger.info(f'Feature Matrix Reduced via variance. Dimensions: {self.feature_mat.shape}')


        if self.prefiltering == 'pvalue':
            keep_idxs = set()
            from dtk.faers import make_faers_hypot_tests, make_faers_cmh_hypot_tests
            if self.class_weight == 'stratified':
                stratify_idxs = [i for i, x in enumerate(self.feature_types) if x == 'demo']
                ors, pvs = make_faers_cmh_hypot_tests(self.feature_mat, self.target, stratify_idxs)[:2]
            else:
                ors, pvs, _ = make_faers_hypot_tests(self.feature_mat, self.target)

            # Relatively conservative, just filter out the most spurious of associations.
            # Things get filtered more downstream, this just picks features to not bother including
            # in the logistic regression.
            pv_thresh = 0.1

            # The pvalues are going to depends greatly on the number of events for our target indi.
            # Let's make sure we keep a reasonable number of features (at least ~10%) even for rarer ones.
            pv_thresh = max(pv_thresh, sorted(pvs)[len(pvs) // 10])
            print(f"P value threshold set at {pv_thresh:.2f}")


            for i, p in enumerate(pvs):
                if self.feature_types[i] == 'demo':
                    # Don't drop demographics in case we're stratifying.
                    keep_idxs.add(i)
                    continue
                if p < pv_thresh:
                    keep_idxs.add(i)
            
            keep_idxs = list(sorted(keep_idxs))
            self.feature_names = self.feature_names[keep_idxs]
            self.feature_types = self.feature_types[keep_idxs]
            self.feature_mat = self.feature_mat[:, keep_idxs]

            logger.info(f'Feature Matrix Reduced via P-Value. Dimensions: {self.feature_mat.shape}')
                
                
        JobInfo.report_info(f"Feature Matrix Filtering: {orig_shape[1]} -> {self.feature_mat.shape[1]} features")

    def fit_and_summarize(self, out_fn=None):
        out_fn = out_fn or self._out_results_fn
        parms = LRParams(
            test_train_split = False,
            C = self.C,
            method = self.method,
            penalty = self.penalty,
            class_weight = self.class_weight,
        )

        if self.class_weight == 'stratified':
            # Stratify on demographic indices
            parms.stratify_idxs = [i for i, x in enumerate(self.feature_types) if x == 'demo']

        combined_drug_disease = not self.split_drug_disease

        fm = self.feature_mat
        target = self.target

        if combined_drug_disease:
            logreg = LR(fm, target, parms)
            lr, accuracy, wtd_accuracy = logreg.run()
            df = logreg.assemble_df(lr, self.feature_names)
        else:
            non_drug_idxs = [i for i, x in enumerate(self.feature_types) if x != 'drug']
            non_indi_idxs = [i for i, x in enumerate(self.feature_types) if x != 'indi']
            if self.class_weight == 'stratified':
                parms.stratify_idxs = [i for i, x in enumerate(self.feature_types[non_drug_idxs]) if x == 'demo']

            if self.ignore_single_indi:
                indi_idxs = [i for i, x in enumerate(self.feature_types) if x == 'indi']
                indi_mat = fm[:, indi_idxs]
                multi_indi_rows = np.asarray(indi_mat.sum(axis=1) > 1).reshape(-1)
                fm = fm[multi_indi_rows, :]
                target = target[multi_indi_rows]
                logger.info(f"Resulting matrix has {fm.shape[0]} rows, {target.sum()} cases")

            logreg = LR(fm[:, non_drug_idxs], target, parms)
            lr, accuracy, wtd_accuracy = logreg.run()
            df1 = logreg.assemble_df(lr, self.feature_names[non_drug_idxs])

            # Reset FM and target; we ignore single indi only for disease model, not for drug.
            fm = self.feature_mat
            target = self.target
            if self.class_weight == 'stratified':
                parms.stratify_idxs = [i for i, x in enumerate(self.feature_types[non_indi_idxs]) if x == 'demo']
            logreg = LR(fm[:, non_indi_idxs], target, parms)
            lr, accuracy, wtd_accuracy = logreg.run()
            df2 = logreg.assemble_df(lr, self.feature_names[non_indi_idxs])
            df = pd.concat([df1, df2], ignore_index=True, sort=False)


        target_sums = np.sum(self.target)

        with open(os.path.join(self.outdir,'important_stats.pkl'), 'wb') as handle:
            self.g = pickle.dump({'accuracy' : accuracy,
                                  'wtd_accuracy': wtd_accuracy,
                                  'target_sum' : target_sums},handle,
                )
        df.to_csv(out_fn, sep='\t')

    def _file_to_sparse(self, path, selected_keys, cols=None):
        import gzip
        from collections import defaultdict
        file_dict = defaultdict(dict)
        from dtk.files import get_file_lines
        for line in get_file_lines(path):
            event = line.split('\t')[0]
            if int(event) in selected_keys:
                if cols:
                    # specific columns are specified, so we must "featurize" in different ways
                    for i,v in enumerate(line.split('\t')[1:]):
                        if not any(c.isdigit() for c in v):
                            # No numeric component in value, this is a categorical variable
                            if cols[i] == 'sex':
                                # Handle special case for sex/gender
                                file_dict[int(event)][cols[i]] = {'f':1, 'm':0}[v.strip()]
                        else:
                            if cols[i] == 'qtr':
                                # Special case, we want to expand this as a categorical variable
                                file_dict[int(event)]['Q'+str(v.strip())] = 1
                            else:
                                # this is a random continuous variable
                                file_dict[int(event)][cols[i]] = float(v)
                else:
                    # specific columns are not specified
                    for v in line.split('\t')[1:]:
                        file_dict[int(event)][v.strip()] = 1
            else:
                continue

        sparse_mat = self._dict_to_sparse(file_dict)
        return sparse_mat
    def _dict_to_sparse(self, dict_of_dicts):
        from scipy import sparse
        import numpy as np
        keys = list(dict_of_dicts.keys())
        ix_map_dict = dict(zip(list(keys), list(range(len(keys)))))

        col_names = set()
        for v in dict_of_dicts.values():
            for col in v.keys():
                col_names.add(col)
        col_names = list(col_names)
        col_map_dict = dict(zip(list(col_names), list(range(len(col_names)))))

        row_coord, col_coord, vals = [], [], []
        for key, value in dict_of_dicts.items():
            for inner_key, inner_val in value.items():
                row_coord.append(ix_map_dict[key])
                col_coord.append(col_map_dict[inner_key])
                vals.append(inner_val)

        return sparse.csr_matrix((vals, (row_coord, col_coord))), col_names

    def plot_atc(self):
        '''
        Fetch frequency of each ATC category present in each drug set ATC frequency is stored in
        dictionary of dictionary, first keyed by ATC level, then by ATC code, with an inner value that
        signifies the proportion of drugs with the code in drug set.
        '''
        from dtk.plot import PlotlyPlot

        pos_mat = self.feature_mat[self.target, :]
        neg_mat = self.feature_mat[np.invert(self.target), :]

        pos_denom = float(pos_mat.shape[0])
        neg_denom = float(neg_mat.shape[0])

        pos_count = pos_mat.tocsc().sum(0)
        neg_count = neg_mat.tocsc().sum(0)

        pos_dict = dict(zip(self.feature_names, pos_count.tolist()[0]))
        neg_dict = dict(zip(self.feature_names, neg_count.tolist()[0]))

        import re
        pos_dict = {k:v for k,v in pos_dict.items() if re.match(r'[1-9]{1}[0-9]{1,5}-\d{2}-\d{1,5}', k)}
        neg_dict = {k:v for k,v in neg_dict.items() if re.match(r'[1-9]{1}[0-9]{1,5}-\d{2}-\d{1,5}', k)}


        with open(os.path.join(self.outdir,'neg_atc_dict.pkl'), 'wb') as f:
            pickle.dump(neg_dict, f, protocol=2)
        with open(os.path.join(self.outdir,'pos_atc_dict.pkl'), 'wb') as f:
            pickle.dump(pos_dict, f, protocol=2)



if __name__ == "__main__":
    import sys
    sys.stdout = sys.stderr

    import argparse, time
    parser = argparse.ArgumentParser(description='run Logistic Regression Model on FAERS')
    parser.add_argument("dataset", help="Clinical Dataset")
    parser.add_argument("search_string", help="Indication match pattern")
    parser.add_argument("input", help="Input directory path")
    parser.add_argument("output", help="Output directory path")
    parser.add_argument("method", help="Logistic Regression Method", default='normal')
    parser.add_argument("--C", help="Regularization Strength", default=1, required=False)
    parser.add_argument("--penalty", help="Regularization Penalty", default='l1', required=False)
    parser.add_argument("--class_weight", help="Class Weight", default=None, required=False)
    parser.add_argument("--demos", help="Demographics", nargs='*', default=None, required=False)
    parser.add_argument("--prefiltering", help="PreFiltering", default=None, required=False)
    parser.add_argument("--split-drug-disease", help="Split drug and disease solves", action='store_true', required=False)
    parser.add_argument("--autoscale-C", help="Autoscale regularization strength", action='store_true', default=False, required=False)
    parser.add_argument("--ignore-features", help="File with a list of features to ignore", required=False)
    parser.add_argument("--ignore-single-indi", help="Ignore single indication sampels", action='store_true', default=False, required=False)
    parser.add_argument("--compute-background", help="Compute background instead of normal", action='store_true', default=False, required=False)
    args = parser.parse_args()

    from dtk.log_setup import setupLogging
    setupLogging()

    indi_set = args.search_string.split('|')
    model = FAERS_LR(
        args.dataset,
        args.input,
        args.output,
        args.C,
        args.penalty,
        args.class_weight,
        args.demos,
        args.method,
        args.prefiltering,
        args.split_drug_disease,
        args.autoscale_C,
        args.ignore_features,
        args.ignore_single_indi,
        )
    
    if args.compute_background:
        model.compute_background()
    else:
        model.build_matrix(indi_set)
        model.fit_and_summarize()
        model.plot_atc()
