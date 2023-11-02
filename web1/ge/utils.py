import logging
import os
logger = logging.getLogger(__name__)

# XXX remove_old_meta_results() causes us to wait for the worker machine
# XXX to start before refreshing the web page; maybe this could be a
# XXX settings flag which stores a timestamp, and removes any meta
# XXX results prior to that timestamp at the start of a job
def remove_old_meta_results(tissue):
    from path_helper import PathHelper
    outdir = PathHelper.publish+tissue.geoID
    from aws_op import Machine
    cmd = "rm -rf %s" % Machine.get_remote_path(outdir)
    m = Machine.name_index[PathHelper.cfg('worker_machine_name')]
    m.run_remote_cmd(cmd, venv=None)

def prep_tissue_set_id(p):
    # translate Excluded id to null field
    ts_id = int(p['tissue_set'])
    if ts_id:
        return ts_id
    return None

# This provides shared resources to TissueAction instances
class TissueActionCache:
    def __init__(self,ws_id):
        from dtk.timer import BucketTimer
        self.btmr = BucketTimer()
        from dtk.lts import LtsRepo
        from path_helper import PathHelper
        self.lts_repo = LtsRepo.get(str(ws_id),PathHelper.cfg('lts_branch'))

class TissueAction:
    '''
    description - a user-readable string describing the
        current state of meta and/or sig processing.
    primary - label for the primary action button.
    phase - a string controlling the general processing phase:
        'ext' (no valid operation except delete),
        'abort' (an operation is in progress; render an abort button
        instead of an action link),
        'meta' (meta hasn't succeeded, and needs to be run before
        anything else), or
        'edit' (sig can be run or re-run).
    ok - True indicates that the status is ok, and False indicates
        it should be highlighted in the UI.
    platforms - a list of platforms, one of which must be selected to
        re-run the failed meta (normally empty).
    time_info - time information string for sig and meta jobs
    ts - last successful job completion (for out-of-date dependency)
    last_run_time - last job start or completion (for runtime sort)
    '''
    def __init__(self,tissue,jcc,tac):
        from django.utils import timezone
        min_ts = timezone.now().replace(year=1972)
        self.ok = False
        self.platforms = []
        self.time_info = ""
        self.ts = min_ts
        self.last_run_time = min_ts
        if tissue.source in ('comb','ext'):
            # if tissue was loaded externally, no further action
            self.phase = "ext"
            self.description = tissue.source_label()
            self.ok = True
            return
        jobtypes = "meta sig".split()
        jobnames = [
                tissue.get_meta_jobname(),
                tissue.get_sig_jobname(),
                ]
        tac.btmr.start('latest_jobs')
        job_vec = [jcc.latest_jobs().get(x) for x in jobnames]
        tac.btmr.stop()
        # make sure any LTS data is present
        import os
        for job in job_vec:
            if job:
                if False:
                    # uses standard APIs, but is too slow
                    tac.btmr.start('get_bound')
                    from runner.process_info import JobInfo
                    bji = JobInfo.get_bound(tissue.ws,job)
                    tac.btmr.start('fetch_lts_data')
                    bji.fetch_lts_data()
                    tac.btmr.stop()
                else:
                    tac.btmr.start('lts_fetch')
                    # use an single shared LTS repo instance, and do
                    # fetch directly without involving the JobInfo layer
                    relpath = os.path.join(
                            job.name.split('_')[0],
                            str(job.id),
                            )
                    tac.lts_repo.lts_fetch(relpath)
                    tac.btmr.stop()
        try:
            self.last_run_time = max([
                        x.completed or x.created
                        for x in job_vec
                        if x is not None
                        ])
        except ValueError:
            pass
        labeled_jobs = filter(lambda x: x[1] and x[1].completed,
                        zip(jobtypes,job_vec),
                        )
        tac.btmr.start('extended_status')
        self.meta = jcc.extended_status(job_vec[0]).label
        self.sig = jcc.extended_status(job_vec[1]).label
        tac.btmr.stop()
        from dtk.text import fmt_time
        self.time_info = '<br>'.join([
                '%s: %s'%(n,fmt_time(j.completed,'%Y-%m-%d %H:%M'))
                for n,j in labeled_jobs
                ])
        jobtimes = [j.completed for n,j in labeled_jobs]
        self.ts = max(jobtimes) if jobtimes else min_ts
        if self.meta in jcc.active_labels or self.sig in jcc.active_labels:
            # if something is running, we want some flavor of abort
            self.phase = "abort"
            self.primary = "ABORT"
            if self.meta == "UpToDate":
                self.description = "sig:"+self.sig
            else:
                self.description = "meta:"+self.meta+" sig:"+self.sig
            return
        if self.meta != "UpToDate":
            # if meta isn't up to date, we focus on that
            self.phase = 'meta'
            self.description = "meta:"+self.meta
            self.primary = "Re-Process"
            platforms = []
            if self.meta == "NeverRun":
                self.primary = "Process"
            elif self.meta == "Failed":
                # check multiple case
                from algorithms.exit_codes import ExitCoder
                ec = ExitCoder()
                meta_job = jcc.latest_jobs()[tissue.get_meta_jobname()]
                if meta_job.exit_code == ec.encode('multiplePlatformsError'):
                    from path_helper import PathHelper
                    try:
                        f = open(PathHelper.publish
                                +tissue.geoID
                                +'/multipleGPLsToChooseFrom.err.txt'
                                )
                        self.platforms = [x.strip() for x  in f]
                        f.close()
                        self.meta = "MultiplePlatforms"
                    except IOError:
                        self.platforms = []
                        self.meta = "MultiplePlatforms (please assign manually)"
                    self.primary = ""
            return
        # else, we focus on sig
        self.phase = 'edit'
        self.description = self.sig
        self.primary = "Classify" if self.sig == "NeverRun" else "Re-Classify"
        self.ok = (self.sig == 'UpToDate')
    META_UPDATE=0
    META_REDO=1
    SIG_UPDATE=2
    SIG_REDO=3
    def batches(self):
        '''returns a set of codes indicating which batches this is is.
        '''
        from runner.process_info import JobCrossChecker
        active_labels = JobCrossChecker.active_labels
        result = set()
        try:
            if self.meta not in active_labels:
                result.add(self.META_REDO)
                if self.meta != 'UpToDate':
                    result.add(self.META_UPDATE)
            if self.meta in active_labels+['UpToDate']:
                if self.sig not in active_labels:
                    result.add(self.SIG_REDO)
                    if self.sig != 'UpToDate':
                        result.add(self.SIG_UPDATE)
        except AttributeError:
            pass # self.meta not defined for ext tissues
        return result

class SampleSorter:
    ''' Control grouping of samples for easier case/control setting
    '''
    patterns = (
        r'((sex|gender): ?)?(male|female|m|f|w|0|1|2|NA|unknown|not available)$',
        r'age: *(unknown|NA|N/A|NaN|[0-9]+(\.[0-9]+)?)$',
        r'age: *(unknown|NA|N/A|NaN|[0-9]+(\.[0-9]+)?y)$',
        r'age: *(unknown|NA|N/A|NaN|[0-9]+(\.[0-9]+)? (years|yrs)( old)?)$',
        r'age in years: *(unknown|NA|N/A|NaN|[0-9]+(\.[0-9]+)?)$',
        r'age *\(years?\): *(unknown|NA|N/A|NaN|[0-9]+(\.[0-9]+)?)$',
        r'age *\(yrs?\): *(unknown|NA|N/A|NaN|[0-9]+(\.[0-9]+)?)$',
        r'age *\(y\): *(unknown|NA|N/A|NaN|[0-9]+(\.[0-9]+)?)$',
        r'(unknown|NA|N/A|NaN|[0-9]+(\.[0-9]+)? (years|yrs)( old)?)$',
        )
    @staticmethod
    def matched_all(p,ks):
        import re
        for s in ks:
            if not s:
                continue
            if not re.match(p,s.strip(),re.IGNORECASE):
                return False
        return True
    @classmethod
    def matched_any(cls,ks):
        if not ks:
            return False
        for p in cls.patterns:
            if cls.matched_all(p,ks):
                return True
        return False
    def content_filter(self):
        new = ''
        for i,flag in enumerate(self.selector):
            if flag == 'N':
                new += 'N'
            elif self.matched_any(self.keysets[i]):
                new += 'N'
            else:
                new += 'Y'
        self.selector = new
    def __init__(self,tissue):
        # map the preferred tissue labels for this tissue set
        from browse.models import Sample
        enum = Sample.group_vals
        self.label_map = { x[0]:x[1] for x in enum.choices() }
        ts = tissue.tissue_set
        if ts:
            self.label_map[enum.CASE] = ts.case_label
            self.label_map[enum.CONTROL] = ts.control_label
        # Collect all the Sample objects into the sample array.
        # In the parallel sample_attr array, hold all the attributes
        # for the sample as a list.
        # [[sample1_attr1,sample1_attr2,...],[sample2_attr1,...],...]
        qs = Sample.objects.filter(tissue=tissue)
        sample_attr = []
        samples = []
        for sample in qs:
            if sample.attributes:
                attr = sample.attributes.split('\t')
            else:
                # legacy data
                attr = sample.primary_factor.split(';')
                attr += sample.secondary_factor.split(';')
            sample_attr.append(attr)
            samples.append(sample)
        # Now go build a selector string with one character for each
        # attribute type:
        # '_' - all samples share a single value
        # 'N' - each sample has its own unique value
        # 'Y' - there are multiple values, but fewer values than samples
        # self.desc accumulates all the '_' values
        # self.keysets holds a set for each 'Y' or 'N' value
        self.desc = []
        self.keysets = []
        self.selector = ''
        any_id_col = False
        for col_data in zip(*sample_attr):
            vals = len(set(col_data))
            if vals == 1:
                self.desc.append(col_data[0])
                self.selector += '_'
            else:
                self.keysets.append(set(col_data))
                if vals == len(sample_attr):
                    any_id_col = True
                    self.selector += 'N'
                else:
                    self.selector += 'Y'
        # remove all the '_' attribute types from sample_attr
        filtered = []
        for attr in sample_attr:
            tmp = []
            for f,v in zip(self.selector,attr):
                if f != '_':
                    tmp.append(v)
            filtered.append(tmp)
        sample_attr = filtered
        # ...and from selector
        self.selector = list(filter(lambda x: x != '_',self.selector))
        # if there wasn't any 'N' attribute type, make a fake one
        if not any_id_col:
            self.selector += 'N'
            keyset = set()
            for sample,attr in zip(samples,sample_attr):
                attr += [sample.sample_id]
                keyset.add(sample.sample_id)
            self.keysets.append(keyset)
        # now look for content-based reasons to exclude fields from
        # the default set
        self.content_filter()
        # now build the selector-dependent information
        self.samples = samples
        self.sample_attr = sample_attr
        self.build_selected(tissue.cc_selected)
    def build_selected(self,selector=''):
        # if a selector is passed in, use it
        # but only if it's compatible with the other data
        # (different meta options may change the number of attributes)
        if selector and len(selector) == len(self.selector):
            self.selector = selector
        # build a list of available attributes that the user can group on;
        # each entry in self.columns is a tuple containing:
        # - any common prefix for all values
        # - a string listing all values with the prefix removed
        # - a flag indicating whether the column is selected as part of the
        #   key in the self.groups structure below
        self.columns = []
        import os
        for f,s in zip(self.selector,self.keysets):
            pre = os.path.commonprefix(list(s))
            rest = len(pre)
            self.columns.append(
                    (pre
                    ,'; '.join([x[rest:].strip() for x in s])
                    ,f == 'Y'
                    )
                )
        # go through all samples, and construct a key tuple consisting of
        # the selected attributes for that sample, and the current
        # classification; group the samples by the key
        groups = {}
        from browse.models import Sample
        for sample,attr in zip(self.samples,self.sample_attr):
            key = []
            for a,f in zip(attr,self.selector):
                if f == 'Y':
                    key.append(a.strip())
            key.append(Sample.group_vals.get('label',sample.classification))
            key = tuple(key)
            l = groups.setdefault(key,[])
            l.append(sample)
        # Now, turn that dict into an OrderedDict in sorted key order
        # replacing the sample list value with a tuple containing:
        # - html for case/control radio buttons
        # - the sample list
        from collections import OrderedDict
        self.groups = OrderedDict()
        for i,key in enumerate(sorted(groups.keys())):
            self.groups[key] = (
                    self.radio_button_html(i,key),
                    groups[key],
                    i & 1,
                    )
    label_prefix = "cci_radio_button_"
    def radio_button_html(self,i,key):
        template='''
            <span style='white-space:nowrap'>
            <input type='radio' name='%s%s' value='%s' %s/> %s
            </span>
            '''
        l = list()
        from browse.models import Sample
        for pair in Sample.group_vals.choices():
            l.append(template %
                ( self.label_prefix
                , str(i)
                , pair[1]
                , "checked" if pair[1] == key[-1] else ""
                , self.label_map[pair[0]]
                ))
        from django.utils.safestring import mark_safe
        return mark_safe("".join(l))
    def update_records_from_key(self,key,value):
        # process a single POST value returned from a radio button
        # as constructed above; find all the corresponding samples
        # and update them
        if not key.startswith(self.label_prefix):
            return
        from browse.models import Sample
        class_idx = Sample.group_vals.find('label',value)
        group_idx = int(key[len(self.label_prefix):])
        samples = list(self.groups.values())[group_idx][1]
        for sample in samples:
            sample.classification = class_idx
            sample.save()


def generate_sample_metadata(prj_id, out_file):
    logger.info(f"Generating sample metadata for {prj_id} to {out_file}")
    assert prj_id.startswith('PRJ'), "Only implemented for PRJ; GEO & AE do this in run_meta"
    from dtk.entrez_utils import SraSearch
    from ge.models import SraRun
    data = SraRun.objects.filter(bioproject=prj_id).values_list('experiment', 'attrs_json')

    logger.info("Grabbing outlier/processing failed data")
    try:
        from path_helper import PathHelper
        from dtk.files import get_file_records
        from dtk.data import MultiMap
        pubdir = os.path.join(PathHelper.publish, prj_id)
        fn = os.path.join(pubdir, f'{prj_id}_sampleToSrrConversion.csv')
        srr_to_srx = {srr:srx for srx, srr in get_file_records(fn, keep_header=False)}
        fn = os.path.join(pubdir, 'SRRs_processed.tsv')
        good_srrs = set(list(x)[0] for x in get_file_records(fn, keep_header=None))
        # Need the 'None' because the file has some spurious non-SRR directory entries.
        good_srxs = set(srr_to_srx.get(srr, None) for srr in good_srrs)
    except (OSError, IOError) as e:
        # We don't fail here because older datasets don't have this yet.
        # Could convert to a hard fail later.
        logger.warning("Failed to pull SRR conversion / success data.")
        good_srxs = None

    all_attrs = []
    for experiment, attrs_json in data:
        import json
        attrs_obj = json.loads(attrs_json)
        attrs = [f'{k}: {v}' for k, v in attrs_obj]
        attrs = [experiment] + attrs

        if good_srxs and experiment not in good_srxs:
            attrs.append('Outlier: True')
        else:
            attrs.append('Outlier: False')
        all_attrs.append(attrs)

    from atomicwrites import atomic_write
    with atomic_write(out_file, overwrite=True) as f:
        for attrlist in all_attrs:
            f.write('\t'.join(attrlist) + '\n')


# XXX Maybe build an additional data structure in SampleSorter to
# XXX facilitate rendering repeated values using rowspans?

def upload_samples(tissue):
    geo_id = tissue.geoID
    logger.info(f"Uploading samples for {geo_id}")
    # read upload file into a dict of {id:(attributes,legacy1,legacy2)}
    found = {}
    outliers = []
    legacy_outliers = tissue.legacy_outliers()
    # first check for tsv
    out_file = tissue.metadata_path()
    logger.debug("attempting to read '%s'",out_file)

    if not os.path.exists(out_file) and tissue.is_bioproject():
        generate_sample_metadata(geo_id, out_file)

    prefix = 'Outlier: '
    with open(out_file,"r") as tsvfile:
        for line in tsvfile:
            row = line.strip().split('\t')
            if row[0] in legacy_outliers:
                outliers.append(row[0])
                continue
            if row[-1].startswith(prefix):
                if row[-1][len(prefix):] == 'True':
                    outliers.append(row[0])
                    continue
                row = row[:-1]
            found[row[0]] = ('\t'.join(row[1:]),'','')
    logger.debug("beginning upload of %d records" % len(found))
    # read through existing sample objects
    # - if in hash, delete hash entry
    # - else, delete Sample record
    from browse.models import Sample
    qs = Sample.objects.filter(tissue=tissue).all()
    for s in qs:
        if s.sample_id in found:
            logger.debug("%s should continue to exist",s.sample_id)
            # copy possibly updated classification info
            row = found[s.sample_id]
            s.attributes = row[0]
            s.primary_factor = row[1]
            s.secondary_factor = row[2]
            s.save()
            del found[s.sample_id]
        else:
            logger.debug("%s should be deleted",s.sample_id)
            s.delete()
    # insert everything left in hash to Sample table
    for sample_id in found:
        logger.debug("%s should be added",sample_id)
        row = found[sample_id]
        s = Sample(tissue=tissue
                ,sample_id=sample_id
                ,attributes = row[0]
                ,primary_factor = row[1]
                ,secondary_factor = row[2]
                ,classification=Sample.group_vals.IGNORE
                )
        s.save()
    return outliers

def find_split_btn(d):
    prefix = 'split_btn_'
    for key in d:
        if key.startswith(prefix):
            return int(key[len(prefix):])
    return None

class DbProteinMappings:
    def __init__(self,mapping=None,ws=None):
        from dtk.prot_map import DpiMapping
        if not mapping:
            from browse.default_settings import DpiDataset
            mapping = DpiDataset.value(ws=ws)
        self.dpi = DpiMapping(mapping)
        # dpi_list is:
        # [(drug,prot,ev),...]
        self.dpi_list = []
        with open(self.dpi.get_path()) as f:
            header = None
            for line in f:
                fields = line.strip('\n').split('\t')
                if header is None:
                    header = fields
                    continue
                self.dpi_list.append((fields[0],fields[1],float(fields[2])))
    def _group(self,by_drug,thresh=0,drugs_in=[]):
        (key_col,val_col) = (0,1) if by_drug else (1,0)
        thresh_col = 2
        index = {}
        for item in self.dpi_list:
            if thresh and item[thresh_col] < thresh:
                continue
            if drugs_in and item[0] not in drugs_in:
                continue
            sub_list = index.setdefault(item[key_col],[])
            sub_list.append(item[val_col])
        return index
    def get_drugs_per_prot(self,**kwargs):
        # return a dict of { prot: [drug,drug,...], ... }
        return self._group(False,**kwargs)
    def get_prots_per_drug(self,**kwargs):
        # return a dict of { drug: [prot,prot,...], ... }
        return self._group(True,**kwargs)
    def get_kts_by_prot(self,ws):
        kt_wsas = ws.get_wsa_id_set(ws.eval_drugset)
        kt_keys = set()
        for key,wsa_list in self.dpi.get_wsa_id_map(ws).items():
            if set(wsa_list) & kt_wsas:
                kt_keys.add(key)
        return self.get_drugs_per_prot(drugs_in=kt_keys)

########
# tissue stats stuff
########
def level_list(data):
    # input is a dictionary of the form:
    # {item_id:number_of_sets, ...} or
    # {item_id:[list of sets], ...} (where list of sets can be any iterable)
    # output is:
    # [ (set_count,[item_id,...]), ... ]
    level_index = {}
    for key,val in data.items():
        try:
            level = len(val)
        except TypeError:
            level = int(val)
        item_list = level_index.setdefault(level,[])
        item_list.append(key)
    level_list = []
    for i in sorted(list(level_index.keys()),reverse=True):
        level_list.append((i,level_index[i]))
    return level_list

def get_tissue_stats(ts,threshold, heading=None):
    # tissue/confidence matrix
    if heading is None:
        heading=[2.,0.999,0.99,0.95,0.9,0.8]
    rows, tissues_per_prot, prots_per_tissue, heatmap_data = _get_tissue_prot_data(heading, ts.id, threshold)
    prot_pors = {p:float(v)/len(list(prots_per_tissue.keys())) for p,v in tissues_per_prot.items()}
    tissue_prot_stats = _process_tp_stats(prots_per_tissue, prot_pors)
    from path_helper import PathHelper
    import os
    plot_heatmap(heatmap_data
                , str(ts.ws_id)
                , ofile = os.path.join(PathHelper.ws_publish(ts.ws_id), 'tp_heatmap.png')
                )
    return (heading,rows,tissues_per_prot,tissue_prot_stats)

def _process_tp_stats(prots_per_tissue, prot_pors):
    import numpy as np
    tissue_prot_stats = {}
    nprots = float(len(list(prots_per_tissue.keys())))
    bins = [float(i) / nprots for i in _get_shared_sigProt_bins(nprots) if nprots]
    from browse.models import Tissue
    for t in prots_per_tissue.keys():
        bin_pors = np.array([0.0]*len(bins))
        total_prots = len(prots_per_tissue[t])
        if total_prots == 0:
            continue
        single_portion = 1.0 / total_prots
        for p in prots_per_tissue[t]:
            p_bin_pors = [0.0] * len(bins)
            for i in range(len(p_bin_pors)):
                if prot_pors[p] >= bins[i]:
                    p_bin_pors[i] = single_portion
                    break
            p_bin_pors = np.array(p_bin_pors)
            bin_pors = np.add(bin_pors, p_bin_pors)
        tissue_prot_stats[Tissue.concisify(t)] = (total_prots, list(bin_pors))
    return tissue_prot_stats

# These evidence scores range from 0-1
# The options on the UI, range from 0-1
# So if there is a threshold above 1
# that is an indication that the tissue-specific threshold should be used
def verify_tissue_threshold(threshold, t):
    if threshold <=1.:
# The UI doesn't currently have a FC threshold option
# So return 0 as the FC threshold
        return threshold,0.
    return t.ev_cutoff,t.fc_cutoff

def _get_tissue_prot_data(heading, ts_id, threshold):
    from collections import Counter, defaultdict
    import operator
    rows=[]
    tissues_per_prot = Counter()
    prots_per_tissue = {}
    heatmap_data = defaultdict(dict)
    # The following loop is the slowest part of the tissue_stats page.
    # It was optimized quite a bit when it fetched from the database,
    # but then was converted to use the Tissue.sig_results method. There
    # may be further optimizations available.
    from browse.models import Tissue
    for t in Tissue.objects.filter(tissue_set_id=ts_id).order_by('name'):
        t_ev_threshold, t_fc_threshold = verify_tissue_threshold(threshold,t)
        prots_per_tissue[t.name] = set()
        tname = t.concise_name()
        l = t.sig_results(over_only=False)
        for rec in l:
            heatmap_data[rec.uniprot][tname] = rec.evidence * rec.direction
            if rec.evidence >= t_ev_threshold and rec.fold_change >= t_fc_threshold:
                tissues_per_prot[rec.uniprot] += 1
                prots_per_tissue[t.name].add(rec.uniprot)
        counter=[]
        for thresh in heading:
            ev,fc=verify_tissue_threshold(thresh,t)
            counter += [
                sum((x.evidence >= ev and x.fold_change >= fc for x in l))
                ]
        ev_vec = [x.evidence for x in l]
        if ev_vec:
            from dtk.num import avg
            stats = [min(ev_vec),avg(ev_vec),max(ev_vec)]
        else:
            stats = [0,0,0]
        rows.append([t, counter, stats])
    return rows, tissues_per_prot, prots_per_tissue, heatmap_data

def plot_heatmap(heatmap_data, tmpstr, ofile = 'test.png'):
    tmp_file = "/tmp/heatmap" + tmpstr + ".csv"
    from browse.utils import prep_heatmap
    plot_names, cor_mat = prep_heatmap(heatmap_data)
    if not plot_names:
        return
    with open(tmp_file, 'w') as f:
        f.write(",".join(['rownames'] + plot_names) + "\n")
        for i in range(len(cor_mat)):
            f.write(",".join([plot_names[i]] + cor_mat[i]) + "\n")
    import subprocess
    from path_helper import PathHelper
    subprocess.check_call([
            'Rscript',
            PathHelper.Rscripts+'pheatmap_plotter.R',
            tmp_file,
            ofile,
            ])

def plot_pt_stats(level_list, ws_id, tp_stats):
    from matplotlib.backends.backend_pdf import PdfPages
    from path_helper import PathHelper
    import os
    ofile = os.path.join(PathHelper.ws_publish(ws_id), 'tp_stats.pdf')
    pp = PdfPages(ofile)
    colors = ['grey', 'blue']
    import matplotlib.pyplot as plt
    fig = plot_pt_hist(level_list, colors)
    pp.savefig(fig)
    plt.close(fig)
    for fig in plot_tissue_stats(tp_stats, colors):
        pp.savefig(fig)
        plt.close(fig)
    pp.close()
def plot_tissue_stats(tp_stats, colors):
    import operator
    import matplotlib.pyplot as plt
    # for these next 2 I want to use smaller font
    plt.rcParams.update({'font.size': 10})
    vals = [(k,v[0], v[1]) for k,v in tp_stats.items()]
    vals.sort(key=operator.itemgetter(1))
    xs = [i + 1 for i in range(len(vals))]
    figs = [plot_sigProt_cnts(xs,vals, colors)]
    figs += plot_shared_sigProt_por(xs, vals, colors)
    # Do a zoom in on the last plot
    figs += plot_shared_sigProt_por(xs, vals, colors, ymax = 0.12)
    return figs
def _get_shared_sigProt_bins(ntiss):
    if ntiss <= 10:
        return [8, 6, 4, 2, 1]
    return [int(ntiss * 0.9), 8, 4, 2, 1]

def plot_shared_sigProt_por(xs, vals, colors, ymax = 1.1):
    import matplotlib.pyplot as plt
    ### this a modified (for clearer plotting) version of what is in get_tissue_stats
    bins = [str(i) for i in _get_shared_sigProt_bins(len(xs))]
    # very helpful: http://colorbrewer2.org/#type=sequential&scheme=YlGnBu&n=5
    bar_colors = ['#253494', '#2c7fb8', '#41b6c4', '#a1dab4', '#ffffcc']
    all_ys = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(bins)):
        all_ys.append([x[2][i] for x in vals])
    y_bottoms = [[0.0]*len(vals)]
    for i in range(len(all_ys[:-1])):
        y_bottoms.append( [all_ys[i][j] + y_bottoms[-1][j] for j in range(len(vals))])
    rects = (ax.bar(xs
                    , all_ys[i]
                    , color=bar_colors[i]
                    , bottom = y_bottoms[i]
                   )
              for i in range(len(bins))
            )
    ax.set_ylabel('Portion of significant proteins')
    ax.set_ylim([0.0, ymax])
    ax.set_xlabel('Gene expression datasets')
    ax.set_xticks(xs)
    xtickNames = ax.set_xticklabels([x[0] for x in vals])
    plt.setp(xtickNames, fontsize=8, rotation = 90)
    ax.legend((x[0] for x in rects)
              , bins
              , loc='upper center'
              , bbox_to_anchor=(0.5, 1.02)
              , ncol=len(bins)
              , fancybox=True
              , shadow=True
             )
    plt.title('Portion of shared significant proteins')
    plt.tight_layout()
    return [fig]
def plot_sigProt_cnts(xs,vals, colors):
    import matplotlib.pyplot as plt
    import itertools
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(xs, [x[1] for x in vals]
                    , color=colors[1]
                    , edgecolor = "none"
                   )
    for i in list(range(len(xs)))[::2]:
        rects1[i].set_color(colors[0])
    ax.set_ylabel('Number of significant proteins')
    plt.yscale('log')
    ax.set_xlabel('Gene expression datasets')
    ax.set_xticks(xs)
    xtickNames = ax.set_xticklabels([x[0] for x in vals])
    plt.setp(xtickNames, fontsize=8, rotation = 90)
    for xtick, color in zip(ax.get_xticklabels(), itertools.cycle(colors)):
        xtick.set_color(color)
    plt.tight_layout()
    return fig
def plot_pt_hist(level_list, colors):
    import numpy as np
    import matplotlib.pyplot as plt
    import itertools
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ## the data
    ys = []
    xs = []
    level_list.reverse()
    for tup in level_list:
        xs.append(tup[0])
        ys.append(len(tup[1]))
# flip back for webpage viewing
    level_list.reverse()
    rects1 = ax.bar(xs, ys
                    , color='blue'
                    , edgecolor = "none"
                   )
    for i in list(range(len(xs)))[::2]:
        rects1[i].set_color('grey')
    ax.set_ylabel('Number of significant proteins')
    plt.yscale('log')
    ax.set_xlabel('Number of tissues in which protein is misexpressed')
    ax.set_xticks(xs)
    xtickNames = ax.set_xticklabels(xs)
    for xtick, color in zip(ax.get_xticklabels(), itertools.cycle(colors)):
        xtick.set_color(color)
    combined_y = np.array([j for i in range(len(xs)) for j in [xs[i]] * ys[i]])
    annot_strs = ["Median: %.2f" % np.median(combined_y),
                  "Mean: %.2f" % np.mean(combined_y),
                 ]
    plt.annotate("\n".join(annot_strs)
                 , xy=(1, 1)
                 , xytext = (-120, -10)
                 , va = 'top'
                 , xycoords = 'axes fraction'
                 , textcoords = 'offset points'
                )
    return fig

def make_protein_plot(ev,fc,ids,thresh):
    cmax = max([abs(min(fc)), max(fc)])
    cmin = cmax * -1.0
    from browse.models import Protein
    names = Protein.make_gene_list_from_uniprot_list(ids)
    from dtk.plot import scatter2d
    pp = scatter2d('Evidence rank',
                '1 - FDR',
                zip(list(range(len(ev))), ev),
                title = 'Differential protein expression',
                text = ['<br>'.join(
                                [
                                 names[i]
                                 , "Fold change: " + str(fc[i])
                                ])
                        for i in range(len(ids))
                       ],
                ids = ('protpage', ids),
                refline = False,
                class_idx = [0] * len(ids), # filler
                classes=[('Unknown'
                          ,{
                           'color' : fc
                           , 'cmax' : cmax
                           , 'cmin' : cmin
                           ,'opacity' : 0.7
                           ,'colorbar' : {
                               'title' : 'Fold Change',
                               'len' : 0.5,
                              },
                          }
                        )],
                width = 1200,
               )
    pp._layout['shapes']=[
                 {
                  'type': 'line',
                  'x0': 0,
                  'y0': thresh,
                  'x1': len(ev),
                  'y1': thresh,
                  'line': {
                     'color': 'red',
                     'dash': 'dot',
                    },
                 }
                ]
    return pp

