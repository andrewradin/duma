import numpy as np
import os
import logging
logger = logging.getLogger(__name__)

# This file used to contain lots of experiments with various ways to
# access FAERS data. These have been deleted, but can be retrieved from
# git if necessary. Of possible interest:
# - an example of using peewee in a flow-thru mode to process database
#   queries too large to fit in memory
# - an example of using dtk.cache
# Also deleted was StatsCollector, a tool for pulling counts from the
# FAERS db.  If any old scripts that refer to it need to be revived,
# it should be relatively easy to replace using the ClinicalEventCounts
# class below.

################################################################################
# a function to append one sparse array to another
################################################################################
from builtins import range
def csr_vstack(a,b):
    from scipy import sparse
    # each part of the matrix gets trimmed on restore to the max number of
    # columns that were actually used; here, figure out the total number of
    # columns and restore any missing piece
    need_cols = max(a.shape[1],b.shape[1])
    if a.shape[1] < need_cols:
        pad = sparse.csr_matrix((a.shape[0],need_cols-a.shape[1]),dtype=a.dtype)
        a = sparse.hstack((a,pad))
    if b.shape[1] < need_cols:
        pad = sparse.csr_matrix((b.shape[0],need_cols-b.shape[1]),dtype=b.dtype)
        b = sparse.hstack((b,pad))
    concat_mat = sparse.vstack((a,b), format='csr')
    return concat_mat

################################################################################
# Parsing and nomeclature functions
################################################################################
class DemoFilter:
    # Although demographic filters are split out in the run_faers UI,
    # it's convenient to carry them around as part of the cds name
    # string (it simplifies passing them to the worker, for example,
    # and limits the amount of code affected by adding more filters).
    # This class provides a translation between the string and field
    # representations.
    #
    # Because the filter isn't part of the workspace default cds string,
    # it doesn't affect things like the faers_indi page, but this can
    # be accessed by manually adding a cds qparm to the URL with the
    # desired filtering.
    filter_parts = ['sex_filter','min_age','max_age']
    sex_filter=None
    min_age=None
    max_age=None
    @classmethod
    def split_full_cds(cls,cds):
        '''Return (cds_stem,filter_string) for a full cds name.
        '''
        parts = cds.split('?',maxsplit=1)
        if len(parts) == 1:
            parts.append('')
        return parts
    def __init__(self,s=''):
        import re
        while s:
            if s[0] in 'mf':
                self.sex_filter = s[0]
                s=s[1:]
            elif s[0] == 'a' and s[1] in '<>':
                m = re.match(r'[0-9.]+',s[2:])
                assert m
                num_part = m.group(0)
                num = float(num_part)
                if s[1] == '<':
                    self.max_age = num
                else:
                    self.min_age = num
                s=s[2+len(num_part):]
            else:
                raise NotImplementedError("unknown filter: '%s'"%s)
    def get_form_field(self,filter_part):
        from django import forms
        if filter_part == 'sex_filter':
            return forms.ChoiceField(
                        label='Filter by sex',
                        choices=(
                                ('','None'),
                                ('f','Females only'),
                                ('m','Males only'),
                                ),
                        initial=self.sex_filter,
                        required=False,
                        )
        if filter_part == 'min_age':
            return forms.DecimalField(
                        label='Minimum age',
                        required=False,
                        initial=self.min_age,
                        )
        if filter_part == 'max_age':
            return forms.DecimalField(
                        label='Maximum age',
                        required=False,
                        initial=self.max_age,
                        )
        raise NotImplementedError(f"unknown filter part '{filter_part}'")
    def as_dict(self):
        d = {}
        for filter_part in self.filter_parts:
            d[filter_part] = getattr(self,filter_part)
        return d
    def load_from_dict(self,d,del_from_src=False):
        for filter_part in self.filter_parts:
            setattr(self,filter_part,d[filter_part])
            if del_from_src:
                del d[filter_part]
    def as_string(self):
        result = ''
        if self.sex_filter:
            result += self.sex_filter
        if self.min_age is not None:
            result += 'a>%g' % self.min_age
        if self.max_age is not None:
            result += 'a<%g' % self.max_age
        return result

def get_vocab_for_cds_name(cds):
    # This used to distinguish CVAROD and disease-specific add-ons,
    # but now everything is forced to FAERS (until we bring any other
    # alternatives under versioning).
    return 'FAERS'

if False:
    # XXX This was a partially completed experiment on centralizing the
    # XXX cds name parsing. It turned out what I really wanted was a
    # XXX common parser for the filter part only (above). This might
    # XXX get revived at some point to centralize the parsing functionality,
    # XXX replace get_vocab_for_cds_name, etc.
    # XXX
    # XXX A good time for this might be if we version disease-specific
    # XXX clinical datasets. This might look like:
    # XXX - disease-specific datasets would be separately-versioned
    # XXX   file_classes, with the flavor part designating the base
    # XXX   cds dataset, and maybe the disease. So, the file names
    # XXX   for these would look like:
    # XXX     MDF.faers.v3.v1.drug_mat.npz
    # XXX   (i.e. this is v1 of an MDF file meant to be used with faers.v3)
    # XXX - the corresponding cds would be faers.v3+MDF.v1
    # XXX - if a source provided multiple diseases, this could look like:
    # XXX     some_source.disease.faers.v3.v1.drug_mat.npz
    # XXX   with a cds of faers.v3+some_source.disease.v1
    class ClinicalDatasetName:
        @classmethod
        def from_string(cls,s):
            result = cls()
            def bisect(delim,s):
                parts = s.split(delim)
                if len(parts) == 1:
                    parts.append(None)
                return parts
            result.dataset_name = s
            dataset,result.filt = bisect('?',s)
            result.base,result.supl = bisect('+',dataset)
            result.vocab = get_vocab_for_cds_name(result.base)
            result.file_class,result.version = bisect('.',result.base)
            if result.version is None:
                result.file_class = None
            return result

from dtk.lazy_loader import LazyLoader
from scipy.sparse import load_npz
class DrugIndiDoseCount(LazyLoader):
    """
    NOTE: This data is currently stored as a whole bunch of .npz files and .json files.
    Consider using dtk.arraystore or something similar in the future, to make things more
    ergonomic and probably faster/smaller as well.
    """
    _kwargs=['cds']
    def _get_path(self,dataset,function):
        from dtk.s3_cache import S3File
        file_class,version = dataset.split('.')
        s3f=S3File(file_class,'%s.%s.%s'%(file_class,version,function))
        s3f.fetch()
        return s3f.path()
    def _indi_drug_fm_loader(self):
        return load_npz(self._get_path(self.cds,'indi_drug_mat.npz'))
    def _dose_fm_loader(self):
        return load_npz(self._get_path(self.cds,'dose_mat.npz'))
    def _meta_loader(self):
        fn = self._get_path(self.cds,'indi_drug_dose_meta.json')
        import json
        with open(fn) as f:
            data = json.loads(f.read())
        return data
    def _col_name_to_idx_loader(self):
        meta = self.meta
        out = {}
        for i, col in enumerate(meta['indi_drug_cols']):
            out[col] = i
        return out

    def matching_rows(self, name, fm=None):
        if fm is None:
            fm = self.indi_drug_fm
        """Returns a bool array for selecting rows matching any of the names provided."""
        if isinstance(name, str):
            name2idx = self.col_name_to_idx
            idx = name2idx[name]
            return (fm[:, idx] == 1).toarray().reshape(-1)
        else:
            out = np.zeros(fm.shape[0], dtype=bool)
            for n in name:
                if n in self.col_name_to_idx:
                    out |= self.matching_rows(n)
            return out


    


################################################################################
# An interface to collections of data on clinical events, where each event
# involves a person with a set of diagnoses, taking a set of drugs, at a
# specific time, with optional demographic data.
#
# This implementation stores the database using scipy sparse arrays, which
# have both space and speed advantages over previous implementations based
# on relational databases.
#
# This code can retrieve data from either:
# - new versioned ETL files, where each source is a separate FILE_CLASS,
#   and the dataset string passed to the ctor is of the form file_class.v#
# - the older format where each data source produced a series of clin_ev
#   files in the 2xar/ws directory, and the dataset string is just the
#   source name
################################################################################
class ClinicalEventCounts:
    demo_cols = ['age_yr','wt_kg','sex']
    date_col = 'date'
    quarter_base_year = 1970
    def _get_path(self,dataset,function):
        from dtk.s3_cache import S3MiscBucket,S3File
        if '.v' in dataset:
            # versioned file case
            file_class,version = dataset.split('.')
            s3f=S3File(file_class,'%s.%s.%s'%(file_class,version,function))
        else:
            # old-style clin_ev files
            s3f=S3File(S3MiscBucket(),'clin_ev.%s.%s'%(dataset,function))
        if self.cache_path_override:
            import os
            return os.path.join(
                    self.cache_path_override,
                    os.path.basename(s3f.path()),
                    )
        else:
            s3f.fetch()
            return s3f.path()
    @staticmethod
    def _load_col_list(fn):
        return open(fn).read().split('\n')[:-1]
    @staticmethod
    def _compile(pattern):
        import re
        return re.compile(pattern.lower().replace('%','.*')+'$')
    @staticmethod
    def _one_hot_binning(values, bin_size, feature_name_template):
        import numpy as np
        num_bins = 1+int(max(values)-1)//bin_size
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(categories=[list(range(num_bins))], sparse=True, dtype=bool)
        if isinstance(values, np.ndarray):
            data = ((values-1) // bin_size).reshape(-1, 1)
        else:
            data = np.array([int(val-1)//bin_size for val in values]).reshape(-1, 1)
        m = enc.fit_transform(data)

        if not feature_name_template:
            return m

        feature_names = [
                feature_name_template%(x*bin_size,(x+1)*bin_size-1)
                for x in range(num_bins)
                ]
        return m,feature_names

    def __init__(self,dataset,cache_path=None):
        self.cache_path_override = cache_path # for testing
        path = self._get_path # typographic shortcut
        from scipy.sparse import load_npz
        # parse special suffixes from name:
        # base{+supl}{?row_filt}
        # where supl defines additional data to be appended to the base
        # set, and row_filt defines a demographic subset to use
        dataset,row_filt = DemoFilter.split_full_cds(dataset)
        supl_delim = '+'
        if supl_delim in dataset:
            base,supl = dataset.split(supl_delim)
        else:
            base = dataset
            supl = None
        # load the base (or the only) set of matrices
        self._indi_fm = load_npz(path(base,'indi_mat.npz'))
        self._drug_fm = load_npz(path(base,'drug_mat.npz'))
        self._demo_fm = load_npz(path(base,'demo_mat.npz'))
        self._date_fm = load_npz(path(base,'date_mat.npz'))
        if supl:
            # append custom supplement data to the base set; note that the
            # supplemental arrays need to be built with the same
            # 'cols' numbering scheme used by the base dataset (although they
            # may append additional columns to the end of the list).
            # Note that the files to be appended are named with the
            # full dataset string, not just the supl part (since
            # they're customized to the base dataset).
            self._indi_fm=csr_vstack(
                    self._indi_fm,
                    load_npz(path(dataset,'indi_mat.npz')),
                    )
            self._drug_fm=csr_vstack(
                    self._drug_fm,
                    load_npz(path(dataset,'drug_mat.npz')),
                    )
            self._demo_fm=csr_vstack(
                    self._demo_fm,
                    load_npz(path(dataset,'demo_mat.npz')),
                    )
            # For MDF, date information is not available (and this may be
            # true of future supplemental datasets as well). For now,
            # allow the code to work in the absence of this file, at least
            # to the point where date info is actually needed. So, if the
            # user un-checks 'Include event date as feature', it will work;
            # otherwise it will fail.
            try:
                self._date_fm=csr_vstack(
                        self._date_fm,
                        load_npz(path(dataset,'date_mat.npz')),
                        )
            except IOError:
                self._date_fm = None
        if row_filt:
            # assemble row mask for all filters
            df = DemoFilter(row_filt)
            row_masks = []
            if df.sex_filter:
                code = 1 if df.sex_filter == 'm' else 2
                row_masks.append(self._demo_fm[:,2] == code)
            # an age of 0 means not supplied, so always exclude those
            # on any age filter
            # XXX note that there are a few thousand records with an age
            # XXX explicitly set to zero in the source; we might want to
            # XXX change this to some small decimal
            if df.min_age is not None:
                # a min age of 0 is a convenient way to specify only
                # using records with age supplied, but numpy doesn't implement
                # a >=0 comparison on sparse arrays. Since the min comparison
                # should always pass, just rely on the != 0 comparison in
                # this case
                if df.min_age:
                    row_masks.append(self._demo_fm[:,0] >= df.min_age)
                row_masks.append(self._demo_fm[:,0] != 0)
            if df.max_age is not None:
                row_masks.append(self._demo_fm[:,0] <= df.max_age)
                row_masks.append(self._demo_fm[:,0] != 0)
            row_mask = row_masks[0].toarray()[:,0]
            for mask in row_masks[1:]:
                row_mask = row_mask & mask.toarray()[:,0]
            # now apply composite mask to data
            self._indi_fm = self._indi_fm[row_mask,:]
            self._drug_fm = self._drug_fm[row_mask,:]
            self._demo_fm = self._demo_fm[row_mask,:]
            if self._date_fm is not None:
                self._date_fm = self._date_fm[row_mask,:]
        load_cols = self._load_col_list # typographic shortcut
        self._indi_cols = load_cols(path(dataset,'indi_cols.txt'))
        self._drug_cols = load_cols(path(dataset,'drug_cols.txt'))
    def total_events(self):
        # should be the same for indi, drug, and date
        return self._indi_fm.shape[0]
    def total_indis(self):
        return self._indi_fm.shape[1]
    def total_drugs(self):
        return self._drug_fm.shape[1]
    def _get_indi_target_col_list(self,indi_set):
        # figure out which columns in the indi feature matrix
        # correspond to selected indications
        indi_set = [self._compile(x) for x in indi_set]
        return [
                i
                for i,x in enumerate(self._indi_cols)
                if any([y.match(x) for y in indi_set])
                ]
    def _get_cas_target_col_list(self,cas_set):
        # figure out which columns in the drug feature matrix
        # correspond to selected cas numbers
        return [
                i
                for i,x in enumerate(self._drug_cols)
                if any([y==x for y in cas_set])
                ]
    @staticmethod
    def _get_target_mask(target_cols,fm):
        if not target_cols:
            raise ValueError('no target columns specified')
        # extract a row mask that's the logical OR of the supplied target
        # columns (the mask must be a numpy array of type bool)
        import numpy as np
        return np.array(
                fm[:,target_cols].T.sum(axis=0),
                dtype=bool,
                )[0]
    def _get_portions(self,targ_mask,fm):
        # XXX Maybe throw if targ_mask is all False or all True?
        # get additional info for target table
        if fm is self._drug_fm:
            fm_cols = self._drug_cols
        elif fm is self._indi_fm:
            fm_cols = self._indi_cols
        else:
            raise RuntimeError('Invalid feature matrix')
        bg_counts = fm.sum(axis=0).tolist()[0]
        targ_counts = fm[targ_mask,:].sum(axis=0).tolist()[0]
        import numpy as np
        result = [
                zip(fm_cols,bg_counts), # bg_ctr
                fm.shape[0], # bg_total
                zip(fm_cols,targ_counts), # match_ctr
                np.sum(targ_mask), # match_total
                ]
        return result
    def get_indi_contingency(self, indi_set_a, indi_set_b):
        fm_cols = self._indi_cols
        cols_a = self._get_indi_target_col_list(indi_set_a)
        rows_a = self._get_target_mask(cols_a, self._indi_fm)

        cols_b = self._get_indi_target_col_list(indi_set_b)
        rows_b = self._get_target_mask(cols_b, self._indi_fm)

        rows_sum = rows_a + rows_b

        a_and_b = rows_a == rows_b
        table = [
                rows_a & rows_b,
                rows_a & (~rows_b),
                (~rows_a) & rows_b,
                (~rows_a) & (~rows_b),
                ]
        return [x.sum() for x in table]

    def get_drug_portions(self,indi_set):
        target_cols = self._get_indi_target_col_list(indi_set)
        targ_mask = self._get_target_mask(target_cols,self._indi_fm)
        return self._get_portions(targ_mask,self._drug_fm)
    def get_disease_co_portions(self,indi_set):
        target_cols = self._get_indi_target_col_list(indi_set)
        targ_mask = self._get_target_mask(target_cols,self._indi_fm)
        return self._get_portions(targ_mask,self._indi_fm)
    def get_disease_portions(self,cas_set):
        target_cols = self._get_cas_target_col_list(cas_set)
        targ_mask = self._get_target_mask(target_cols,self._drug_fm)
        return self._get_portions(targ_mask,self._indi_fm)
    def get_drug_co_portions(self,cas_set):
        target_cols = self._get_cas_target_col_list(cas_set)
        targ_mask = self._get_target_mask(target_cols,self._drug_fm)
        return self._get_portions(targ_mask,self._drug_fm)
    def get_disease_names_and_counts(self):
        '''Return an iterable of (disease_name,event_count).'''
        counts = self._indi_fm.sum(axis=0).tolist()[0]
        return zip(self._indi_cols,counts)
    def get_drug_names_and_counts(self):
        '''Return an iterable of (drug_name,event_count).'''
        counts = self._drug_fm.sum(axis=0).tolist()[0]
        return zip(self._drug_cols,counts)
    def get_matrix(self,indi_set,demo_covariates,output_types=False):
        target_cols = self._get_indi_target_col_list(indi_set)
        if True:
            print('search term',indi_set,'matches:')
            counts = self._indi_fm[:,target_cols].sum(axis=0).tolist()[0]
            labels = [self._indi_cols[i] for i in target_cols]
            l = list(zip(labels,counts))
            l.sort(key=lambda x:x[1],reverse=True)
            for label,count in l:
                print("   %d indications labeled '%s'" % (count,label))
        feature_names = []
        matrices = []
        feature_types = []
        indi_fm = self._indi_fm
        drug_fm = self._drug_fm
        date_fm = self._date_fm
        demo_fm = self._demo_fm

        if False:
            # This code filters out rows where the only indication is the target indication.
            # It should eventually be enabled via a setting.
            non_targ_indi = indi_fm[:,list(set(range(indi_fm.shape[1])) - set(target_cols))]
            indi_cnt = non_targ_indi.getnnz(axis=1)
            row_mask = (indi_cnt > 0)
            #print("Used to have ", indi_fm.shape)
            indi_fm = indi_fm[row_mask,:]
            drug_fm = drug_fm[row_mask,:]
            date_fm = date_fm[row_mask,:]
            demo_fm = demo_fm[row_mask,:]
            #print("Now have ", indi_fm.shape)

            
        if demo_covariates:
            # extract selected demo columns
            demo_idx_map = {
                    x:i
                    for i,x in enumerate(self.demo_cols)
                    if x in demo_covariates
                    }
            demo_idxs = list(demo_idx_map.values())
            # the following is like get_target_mask, but using an
            # AND instead of an OR; the mask selects rows where
            # all demo values are present
            import numpy as np
            demo_mask = np.all(demo_fm[:,demo_idxs].toarray(), axis=1)
            print('covariates requested:',demo_covariates)
            print(demo_mask.sum(),'records with all requested demographics')
            demo_fm = demo_fm[demo_mask,:]
            # replace indi and drug fm with subsets matching demo
            indi_fm = indi_fm[demo_mask,:]
            drug_fm = drug_fm[demo_mask,:]
            if date_fm is not None:
                date_fm = date_fm[demo_mask,:]
        # include indications
        matrices.append(indi_fm)
        feature_names.extend(self._indi_cols)
        feature_types.extend(['indi'] * len(self._indi_cols))
        # include drugs
        matrices.append(drug_fm)
        feature_names.extend(self._drug_cols)
        feature_types.extend(['drug'] * len(self._drug_cols))
        if demo_covariates:
            # include demographics/dates
            if 'date' in demo_covariates:
                # include one-hot-encoded decade
                values = date_fm.T.toarray()[0]/4
                my_fm = self._one_hot_binning(values, 10, None)
                my_names = [
                        'the%02ds' % ((self.quarter_base_year+10*i) % 100)
                        for i in range(my_fm.shape[1])
                        ]
                print(my_names)
                matrices.append(my_fm)
                feature_names.extend(my_names)
                feature_types.extend(['demo'] * len(my_names))
                # include one-hot-encoded quarter (encoder assumes 0
                # is a null value, so offset to 1-4)
                values = (date_fm.T.toarray()[0]%4) + 1
                my_fm = self._one_hot_binning(values, 1, None)
                my_names = [
                        'Q%d' % (i+1)
                        for i in range(my_fm.shape[1])
                        ]
                print(my_names)
                matrices.append(my_fm)
                feature_names.extend(my_names)
                feature_types.extend(['demo'] * len(my_names))
            if 'sex' in demo_covariates:
                col = demo_idx_map['sex']
                from sklearn.preprocessing import OneHotEncoder
                enc = OneHotEncoder(dtype=bool,categories=[[1,2]])
                oh = enc.fit_transform(demo_fm[:,col].toarray())
                # it's not clear from the documentation, but our input is
                # an array with the values 1 and 2, and the output has
                # 2 columns per row (not 3, for the unused 0); make sure
                # that remains true
                assert oh.shape[1] == 2
                matrices.append(oh)
                feature_names.extend(['male','female'])
                feature_types.extend(['demo'] * 2)
            if 'age_yr' in demo_covariates:
                col = demo_idx_map['age_yr']
                values = demo_fm[:,col].T.toarray()[0]
                my_fm,my_names = self._one_hot_binning(
                        values,
                        10,
                        'age %d-%d yrs',
                        )
                matrices.append(my_fm)
                feature_names.extend(my_names)
                feature_types.extend(['demo'] * len(my_names))
            if 'wt_kg' in demo_covariates:
                col = demo_idx_map['wt_kg']
                values = demo_fm[:,col].T.toarray()[0]
                my_fm,my_names = self._one_hot_binning(
                        values,
                        20,
                        'weight %d-%d kg',
                        )
                matrices.append(my_fm)
                feature_names.extend(my_names)
                feature_types.extend(['demo'] * len(my_names))
        from scipy import sparse
        for m in matrices:
            print(type(m).__name__,m.dtype,m.shape)
        concat_mat = sparse.hstack(matrices, format='csr')
        targ_mask = self._get_target_mask(target_cols,concat_mat)
        other_idxs = [
                i
                for i in range(len(feature_names))
                if i not in target_cols
                ]
        feature_mat = concat_mat[:, other_idxs]
        feature_names = [
                n
                for i,n in enumerate(feature_names)
                if i not in target_cols
                ]
        feature_types = [
                n
                for i,n in enumerate(feature_types)
                if i not in target_cols
                ]
        if output_types:
            return feature_mat,targ_mask,feature_names,feature_types
        else:
            return feature_mat,targ_mask,feature_names

################################################################################
# A tool for making click-through drug scatterplots from FAERS data
################################################################################
class CASLookup:
    def get_name_and_wsa(self,cas):
        # try lookup in current workspace
        try:
            wsa_id=self.cas2wsa[cas]
            return (self.wsa2name[wsa_id],self.wsa_cache[wsa_id])
        except KeyError:
            pass
        # fall back to global name lookup
        try:
            return (self.cas2name[cas],None)
        except KeyError:
            return None
    def __init__(self,ws_id):
        from dtk.data import MultiMap
        if ws_id:
            from browse.models import Workspace
            ws=Workspace.objects.get(pk=ws_id)
            # map from wsa_id to drug name
            self.wsa2name = ws.get_wsa2name_map()
            # map from wsa_id to wsa object
            self.wsa_cache = {
                    x.id:x
                    for x in ws.wsannotation_set.all()
                    }
            # map from CAS to wsa_id; if a CAS is ambiguous, keep one wsa at random
            mm=MultiMap(ws.wsa_prop_pairs('cas'))
            self.cas2wsa=dict(MultiMap.flatten(
                    mm.rev_map(),
                    selector=lambda x:next(iter(x)),
                    ))
        else:
            self.cas2wsa = {}
        # backup map from CAS to name; if a CAS is ambiguous, keep one name
        # at random
        from drugs.models import Drug,Prop
        mm=MultiMap(Drug.prop_prop_pairs('cas',Prop.NAME))
        mm.update(MultiMap(Drug.prop_prop_pairs('cas',Prop.OVERRIDE_NAME)))
        self.cas2name=dict(MultiMap.flatten(
                mm.fwd_map(),
                selector=lambda x:next(iter(x)),
                ))


from dtk.cache import cached

def make_faers_strat_hypot_tests(fm, target, stratify_idxs):
    from scripts.faers_lr_model import stratified_cont_tables, stratify_weights
    from statsmodels.stats import contingency_tables
    import numpy as np
    from dtk.stats import fisher_exact
    fm_and_targ = fm.multiply(target.reshape(-1, 1)).tocsc()
    strat_weights = np.array(stratify_weights(fm, target, stratify_idxs))
    stratw_fm = fm.multiply(np.reshape(strat_weights, (-1, 1))).tocsc()
    stratw_fm_and_targ = stratw_fm[target == 1]
    stratw_feat_totals = np.asarray(stratw_fm.sum(axis=0)).reshape(-1)
    stratw_feat_and_targ_totals = np.asarray(stratw_fm_and_targ.sum(axis=0)).reshape(-1)

    targw_sum = (target * strat_weights).sum()
    w_sum = strat_weights.sum()
    
    strat_fe_ors = []
    strat_fe_ps = []

    for i in range(fm.shape[1]):
        a = stratw_feat_and_targ_totals[i]
        b = targw_sum - a
        c = stratw_feat_totals[i] - a
        d = w_sum - a - b - c

        fe_or, fe_p = fisher_exact([[a,b],[c,d]])
        strat_fe_ors.append(fe_or)
        strat_fe_ps.append(fe_p)
    
    return strat_fe_ors, strat_fe_ps

def make_faers_cmh_hypot_tests(fm, target, stratify_idxs):
    from scripts.faers_lr_model import stratified_cont_tables, stratify_weights
    from statsmodels.stats import contingency_tables
    cont_tables, strata, strata_counts, strata_prevalence = stratified_cont_tables(fm, target, stratify_idxs)

    # Only analyze contingency tables that have at least 1 case of our ind, and 1 without our ind.
    # Otherwise you get NaNs that propagate everywhere.
    relevant_strata = [i for i, (indi_cnt, cnt) in enumerate(zip(strata_prevalence, strata_counts)) if indi_cnt > 0 and cnt > indi_cnt]
    cont_tables = cont_tables[relevant_strata]
    cont_tables = cont_tables.transpose(1, 2, 3, 0)

    cmh_or = []
    cmh_rr = []
    cmh_p = []

    logger.info(f"Using  {len(relevant_strata)} strata with cases")

    for feat_cont_table in cont_tables:
        with np.errstate(divide='ignore',invalid='ignore'):
            st = contingency_tables.StratifiedTable(feat_cont_table)
            oddsr = st.oddsratio_pooled
            rr = st.riskratio_pooled
            # Currently not using the correction, gone back and forth on this, but it seems to
            # lead to a lot of spurious NaNs.  We do have a lot of 0's in the contingency tables, though...
            p = st.test_null_odds(correction=False).pvalue
            # Sometimes we can get a NaN OR without a NaN pvalue, which makes no sense.
            if np.isnan(oddsr):
                p = float('nan')


            #pequal = st.test_equal_odds().pvalue
            cmh_rr.append(np.nan_to_num(rr))
            cmh_or.append(np.nan_to_num(oddsr))
            # A NaN p value should transform to p=1 rather than p=0.
            # Typically arises from features with 0 incidence in entire table.
            cmh_p.append(np.nan_to_num(p, nan=1))

    return cmh_or, cmh_p, cmh_rr, strata, strata_counts, strata_prevalence
def make_faers_hypot_tests(fm, target):
    import numpy as np
    from dtk.stats import fisher_exact
    pos_only_fm = fm[target == 1]

    raw_counts = np.asarray(fm.sum(axis=0)).reshape(-1)
    pos_counts = np.asarray(pos_only_fm.sum(axis=0)).reshape(-1)

    all_a = pos_counts
    all_b = pos_only_fm.shape[0] - all_a
    all_c = raw_counts - all_a
    all_d = fm.shape[0] - all_a - all_b - all_c
    
    fe_ors = []
    fe_ps = []
    for a,b,c,d in zip(all_a, all_b, all_c, all_d):
        fe_or, fe_p = fisher_exact([[a,b],[c,d]])
        fe_ors.append(fe_or)
        fe_ps.append(fe_p)

    # There are going to be some NaNs and some divide by 0's in this, that is fine, don't warn.
    with np.errstate(divide='ignore',invalid='ignore'):
        relative_risk = (all_a / (all_a + all_b)) / (all_c / (all_c + all_d))
    
    return fe_ors, fe_ps, relative_risk


def lg2(x):
    return np.nan_to_num(np.log2(x), posinf=1024, neginf=-1024)

def lg10(x):
    # Roughly the value that nan_to_num converts to pre-log
    return np.nan_to_num(np.log10(x), posinf=308, neginf=-308)

def _make_faers_internal(search_term, cds, demos, ws_id):
    """Wraps internal_raw with pandas dataframes"""
    import pandas as pd
    data1, cols1, data2, cols2 = _make_faers_internal_raw(search_term, cds, demos, ws_id)
    df1 = pd.DataFrame(data1, columns=cols1)
    df2 = pd.DataFrame(data2, columns=cols2)
    return df1, df2


@cached(version=1)
def _make_faers_internal_raw(search_term, cds, demos, ws_id):
    """Generates and caches the data used for this table.

    You probably want to use _make_faers_internal, which then wraps it in a dataframe.
    This exists separately so that we're caching the dataframe data, not the df itself.
    """
    import numpy as np

    cec = ClinicalEventCounts(cds)
    indi_set = search_term.split('|')
    try:
        fm, target, feature_names, feature_types = cec.get_matrix(indi_set, demos, output_types=True)
    except ValueError:
        # This happens when your query doesn't actually match anything.
        return [], [], [], []


    pos_only_fm = fm[target == 1]
    cas = CASLookup(ws_id)

    resolved_names = [cas.cas2name.get(x, x) for x in feature_names]

    wsa_ids = [cas.cas2wsa.get(x, None) for x in feature_names]
    from browse.models import WsAnnotation
    wsas = WsAnnotation.objects.filter(pk__in=wsa_ids)
    wsas = WsAnnotation.prefetch_agent_attributes(wsas)
    id2wsa = {wsa.id:wsa for wsa in wsas}

    def linkify(name, wsa_id):
        wsa = id2wsa.get(wsa_id, None)
        if not wsa:
            return name
        return wsa.html_url()
    
    def make_ind_label(wsa_id):
        wsa = id2wsa.get(wsa_id, None)
        if not wsa:
            return'' 
        return wsa.indication_label()

    resolved_names = [linkify(name, wsa_id) for name, wsa_id in zip(resolved_names, wsa_ids)]
    ind_labels = [make_ind_label(wsa_id) for wsa_id in wsa_ids]

    raw_counts = np.asarray(fm.sum(axis=0)).reshape(-1)
    pos_counts = np.asarray(pos_only_fm.sum(axis=0)).reshape(-1)
    portions = [f'{x:.3g}%' for x in pos_counts * 100 / pos_only_fm.shape[0]]


    fe_ors, fe_ps, relative_risk = make_faers_hypot_tests(fm, target)
    stratify_idxs = [i for i, x in enumerate(feature_types) if x == 'demo']
    strat_fe_ors, strat_fe_ps = make_faers_strat_hypot_tests(fm, target, stratify_idxs)
    cmh_or, cmh_p, cmh_rr, strata, strata_counts, strata_prevalence = make_faers_cmh_hypot_tests(fm, target, stratify_idxs)

    from dtk.enrichment import mult_hyp_correct
    cmh_q = mult_hyp_correct(cmh_p)
    # TODO: Add column for drug primary ind(s) via faers?  Or link out?

    # Some notes on these:
    # - Ratios are all represented log2, p values as log10
    # - Generally lined up to make it easier to compare between types ORs or between P values.
    # - The CMH and Stratified statistics are two ways of computing what should be roughly the
    #   same thing.  CMH explicitly constructs contingency tables for each strata and passes it 
    #   through to statsmodel's CMH test/odds functionality, whereas the "strat" variant
    #   uses the stratified sample weights used in LR.  The resulting ORs are very similar, but
    #   have different degeneracies leading to nan/infs.  The p-values are unsurprisingly different,
    #   the CMH is using a different test statistic that is probably more valid, and tends to rate things
    #   as more significant.

    with np.errstate(divide='ignore',invalid='ignore'):
        cols = [
            (feature_names, 'Raw Names'),
            (resolved_names, 'Resolved Names'),
            (feature_types, 'Type'),
            (ind_labels, 'Indication'),
            (raw_counts, 'Raw Counts'),
            (pos_counts, 'Co-Counts'),
            (portions, 'Dis Portion'),
            (lg2(fe_ors), 'OR (lg2)'),
            (lg2(cmh_or), 'CMH OR (lg2)'),
            (lg2(strat_fe_ors), 'Strat OR (lg2)'),
            (lg2(relative_risk), 'RR (lg2)'),
            (lg2(cmh_rr), 'CMH RR (lg2)'),

            (lg10(fe_ps), 'FE P (lg10)'),
            (lg10(strat_fe_ps), 'Strat P (lg10)'),
            (lg10(cmh_p), 'CMH P (lg10)'),
            (lg10(cmh_q), 'CMH Q (lg10)'),
        ]
    data = zip(*(x[0] for x in cols))
    col_names = [x[1] for x in cols]

    total_strata_counts = np.sum(strata_counts)
    total_strata_prev = np.sum(strata_prevalence)
    strata_rows = []
    for stratum, count, co_count in zip(strata, strata_counts, strata_prevalence):
        row = [feature_names[stratify_idxs[i]] for i, v in enumerate(stratum) if v]
        row += [count, co_count, count * 100 / total_strata_counts, co_count * 100 / total_strata_prev]
        strata_rows.append(row)
    
    strata_cols = ["" for _ in strata_rows[0]]
    strata_cols[-4:] = ['Total Count', 'Indi Count', '% of All', '% of Indi']
    return data, col_names, strata_rows, strata_cols

def make_faers_general_table(ws):
    cds_name = ws.get_cds_default()
    vocab = get_vocab_for_cds_name(cds_name)
    search_term = ws.get_disease_default(vocab)
    demos = ['sex']
    df, strat_df = _make_faers_internal(search_term, cds_name, demos, ws.id)
    try:
        df.drop(columns=['Raw Names'], inplace=True)
    except KeyError:
        pass # Usually because we got an empty table.
    return df, strat_df

def add_capp_cols(ws_id, capp_jid, base_table):
    from runner.process_info import JobInfo
    bji = JobInfo.get_bound(ws_id, capp_jid)
    dis2prot2score, _, _ = bji.make_cmap_data()

    dis2numprots = {key:len(val) for key, val in dis2prot2score.items()}
    dis2sumprots = {key:sum(val.values()) for key, val in dis2prot2score.items()}

    # CAPP safe_name-encodes its names.
    from dtk.files import safe_name
    capp_cnt = [dis2numprots.get(safe_name(name), 0) for name in base_table['Raw Names']]
    capp_sum = [dis2sumprots.get(safe_name(name), 0) for name in base_table['Raw Names']]

    base_table['# CAPP Prots'] = capp_cnt
    base_table['CAPP Sum'] = capp_sum

def make_faers_run_table(ws_id, faers_jid, capp_jid=None):
    import numpy as np
    import pandas as pd
    from runner.process_info import JobInfo

    bji = JobInfo.get_bound(ws_id, faers_jid)
    settings = bji.job.settings()

    if 'cds' not in settings or 'search_term' not in settings:
        df = pd.DataFrame([])
        return df, df

    cds = settings['cds']
    from algorithms.run_faers import demos_from_settings
    demos = demos_from_settings(settings)

    base_table, strata_table = _make_faers_internal(settings['search_term'], cds, demos, ws_id)

    from dtk.files import get_file_records
    bji.fetch_lts_data()

    if os.path.exists(bji.fn_model_output):
        recs = get_file_records(bji.fn_model_output, keep_header=True)
        header = next(iter(recs))
        inds = [header.index(x) for x in ['Feature', 'Coefficients', 'P-Value']]
        name2data = {x[inds[0]]: (x[inds[1]], x[inds[2]]) for x in recs}

        # LR Coeffs are typically natural logs, but we convert to log2 for downstream usage,
        # so report them as such for consistency.
        ln_to_log2 = 1.0 / np.log(2)

        lr_data = [name2data.get(name, (0,0)) for name in base_table['Raw Names']]
        lr_coef = [float(x[0]) * ln_to_log2 for x in lr_data]
        lr_p = [float(x[1]) if x[1] != '' else 1 for x in lr_data]

        base_table['LR Coef (lg2)'] = lr_coef
        base_table['LR P (lg10)'] = lg10(lr_p)

    if capp_jid:
        add_capp_cols(ws_id, capp_jid, base_table)

    try:
        base_table.drop(columns=['Raw Names'], inplace=True)
    except KeyError:
        pass # Got an empty table.

    return base_table, strata_table

def make_diff_table(df1, df2, sort_col, id_col, best_n, keep_cols, diff_cols, add_rank=True, abs_order=True):
    df1 = df1.copy()
    df2 = df2.copy()

    def get_best(df):
        if abs_order:
            order = (-df[sort_col].abs()).argsort()
        else:
            order = (-df[sort_col]).argsort()

        df_sort = df[[sort_col, id_col]].iloc[order].reset_index(drop=True)

        id2rank = {x:i for i, x in enumerate(df_sort.values[:, 1])}
        ranks = [id2rank[id] for id in df[id_col]]

        return df_sort[:best_n].values[:, 1], ranks
        

    df1_ids, df1_ranks = get_best(df1)
    df2_ids, df2_ranks = get_best(df2)

    if add_rank:
        if 'Rank' not in diff_cols:
            diff_cols = diff_cols + ['Rank']
        df1['Rank'] = df1_ranks
        df2['Rank'] = df2_ranks

    ids = set(df1_ids) | set(df2_ids)

    df1_rows = df1[df1[id_col].isin(ids)]
    df2_rows = df2[df2[id_col].isin(ids)]

    df = df1_rows[keep_cols].copy()
    for col in diff_cols:
        df[col + ' A'] = df1_rows[col]
        df[col + ' B'] = df2_rows[col]

    return df

def make_faers_diff_table(df1, df2, abs_order=True):
    return make_diff_table(
        df1,
        df2,
        sort_col="LR Coef (lg2)",
        id_col="Resolved Names",
        best_n=50,
        keep_cols=['Resolved Names', 'Type', 'Indication', 'Raw Counts', 'Co-Counts', 'OR (lg2)', 'Strat OR (lg2)', 'Strat P (lg10)'],
        diff_cols=['LR Coef (lg2)', 'LR P (lg10)'],
        abs_order=abs_order,
    )


@cached(version=1)
def _make_faers_indi_dose_data(cds_name, indi, ws_id):
    import numpy as np
    try:
        # Always show this workspace for this table, not the custom indis.
        di = DrugIndiDoseCount(cds=cds_name)
        rows = di.matching_rows(indi.split('|'))
        indi_rows = di.indi_drug_fm[rows]
        indi_dose_rows = di.dose_fm[rows]
    except OSError as e:
        logger.warn("Couldn't load dose table, probably old version: %s" % e)
        return []
        
    from dtk.faers import CASLookup
    caslu = CASLookup(ws_id)

    import scipy.stats

    co_counts = np.asarray(indi_rows.sum(axis=0)).reshape(-1)
    out = []
    for i, cnt in enumerate(co_counts):
        if not cnt:
            continue
        indi_drug_dose_rows = indi_dose_rows[(indi_rows[:, i] == 1).toarray().reshape(-1)]
        
        modes = []
        for col_idx in range(indi_drug_dose_rows.shape[1]):
            col = indi_drug_dose_rows[:, col_idx]
            nonzeros = col.data
            if col_idx == 0:
                nonzeros = nonzeros[nonzeros != 3]
                nonzeros = nonzeros[nonzeros != 4]
            mode, count = scipy.stats.mode(nonzeros, axis=None)
            if len(mode) == 0:
                mode = 0
            else:
                mode = mode[0]
            modes.append(mode)

                
        cas = di.meta['indi_drug_cols'][i]
        val = caslu.get_name_and_wsa(cas)
        if val:
            name, wsa = val
            route = di.meta['route'][int(modes[0])]
            out.append((name, wsa.id if wsa else None, cnt, route))
    return out

def make_faers_indi_dose_data(cds_name, indi, ws_id):
    from browse.models import WsAnnotation
    data = _make_faers_indi_dose_data(cds_name, indi, ws_id)
    out = []
    wsas = WsAnnotation.objects.filter(pk__in=[x[1] for x in data if x[1]])
    wsas = WsAnnotation.prefetch_agent_attributes(wsas, prop_names=['canonical', 'override_name'],)
    wsa_map = {x.id:x for x in wsas}
    for name, wsa_id, cnt, route in data:
        wsa = wsa_map.get(wsa_id, None)
        if wsa:
            name = wsa.html_url()
            ind = wsa.indication_label()
        else:
            ind = ''
        out.append((name, cnt, ind, route))
    return out
