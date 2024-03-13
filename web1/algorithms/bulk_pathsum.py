import logging
logger = logging.getLogger(__name__)

import six
def run_wrapper(item):
    wi=None
    # process one work item
    try:
        context,wi = item
        wi.run(context)
    except Exception as ex:
        print("GOT EXCEPTION IN SUBPROCESS")
        print(wi)
        import traceback
        traceback.print_exc()
        import sys
        sys.stdout.flush()
        raise

class WorkItem:
    @classmethod
    def pickle(cls,indir,name,data):
        import os
        fn = os.path.join(indir,name+'.pickle')
        f = open(fn,'wb')
        import pickle
        pickle.dump(data,f)
    @classmethod
    def unpickle(cls,indir,name):
        import os
        fn = os.path.join(indir,name+'.pickle')
        f = open(fn, 'rb')
        import pickle
        return pickle.load(f)
    @classmethod
    def execute(cls,max_cores,worklist,context,fake):
        # This can be executed locally or remotely;
        # it shouldn't access django or mysql
        if False: # enable for short command-line tests
            if True:
                context['inner_cycles'] = 10
                # quick turnaround for single-thread optimization
                run_wrapper((context,worklist[0]))
                return
            else:
                # measure multi-thread degradation
                context['inner_cycles'] = 100
                max_cores=8
                worklist=worklist[:max_cores]
        maplist = [(context,wi) for wi in worklist]
        if fake:
            map(run_wrapper,maplist)
        else:
            from dtk.parallel import pmap
            list(pmap(run_wrapper, maplist, num_cores=max_cores))
    def __getattr__(self,n):
        # if attribute isn't set directly in work item, fall back to context
        if 'context' in self.__dict__:
            if n in self.context:
                return self.context[n]
        raise AttributeError("no attribute '%s'"%n)
    def __str__(self):
        return 'WorkItem %d' % self.serial
    def run(self,context):
        print('In base WorkItem.run',self,context)

def filter_dpi_from_file(filename,dpi_map,thresh=None):
    f = open(filename,'r')
    is_header = True
    for line in f:
        fields=line.rstrip('\n').split('\t')
        if is_header:
            is_header = False
            yield fields
            continue
        # this pre-filters drugs that aren't in the workspace
        if fields[0] not in dpi_map:
            continue
        if thresh and float(fields[thresh[0]]) < thresh[1]:
            continue
        yield fields

def filter_from_file(filename,key_filter=None,thresh=None):
    from dtk.files import get_file_records
    source = get_file_records(filename)
    yield next(source) # header
    for fields in source:
        if key_filter and fields[key_filter[0]] not in key_filter[1]:
            continue
        if thresh and float(fields[thresh[0]]) < thresh[1]:
            continue
        yield fields

def filter_from_ppi(ppi,key_filter=None,thresh=None):
    from dtk.files import get_file_records
    thresh = thresh or 0.0
    source = ppi.get_data_records(min_evid=thresh)

    yield ppi.get_header()

    for fields in source:
        if key_filter and fields[key_filter[0]] not in key_filter[1]:
            continue
        yield list(fields)

def convert_dpi_to_wsa_keys(recs,dpi_map):
    is_header = True
    for fields in recs:
        if is_header:
            is_header = False
            yield fields
            continue
        for wsa in dpi_map[fields[0]]:
            yield [str(wsa)]+fields[1:]

def subtract_fixed_from_t2p(recs,fixed,verbose=False):
    is_header = True
    for fields in recs:
        if is_header:
            is_header = False
            yield fields
            continue
        if fields[0] in fixed:
            if verbose:
                print('skipping',fields[0],'tissue',fields[1],fields[2:])
            continue
        yield fields

def add_fixed_to_dpi(recs,fixed,verbose=False):
    extra={} # keeps final for loop from breaking when recs is empty
    cur_key=None
    is_header = True
    for row in recs:
        if is_header:
            is_header = False
            yield row
            continue
        if row[0] != cur_key:
            # we're starting a new drug
            if cur_key:
                # flush (possibly modified) base drug bindings
                # from the drug we just finished
                for k,v in six.iteritems(extra):
                    yield [cur_key,k]+[str(x) for x in v]
            # load a fresh set of base bindings for this drug
            extra={k:list(v) for k,v in six.iteritems(fixed)}
        cur_key = row[0]
        if row[1] in extra:
            # if this drug has a binding similar to the fixed
            # drug, merge it into the fixed drug info that will get
            # appended at the end:
            # - ev (v[0]) is max of this or fixed
            # - dir (v[1]) is zero if they don't agree, otherwise unchanged
            was = [float(row[2]),int(row[3])]
            v = extra[row[1]]
            v[0] = max(v[0],was[0])
            if v[1] != was[1]:
                v[1] = 0
            if verbose:
                print(row,v,'' if v == was else '***')
        else:
            # else, just pass it through
            yield row
    for k,v in six.iteritems(extra):
        yield [cur_key,k]+[str(x) for x in v]

def get_combo_fixed(d,dpi):
    # single fixed drug combo case; get the fixed drug data
    from browse.models import WsAnnotation
    wsa_id=d['wsa']
    wsa=WsAnnotation.objects.get(pk=wsa_id)
    l=dpi.get_dpi_info(wsa.agent)
    combo_fixed = {}
    for rec in l:
        combo_fixed[rec[1]] = [
                        float(rec.evidence),
                        int(rec.direction),
                        ]
    return combo_fixed

class PathsumBackground:
    def __init__(self,filenames):
        self.filenames = list(filenames)
        self._label2scores = None
    def label2scores(self):
        if self._label2scores == None:
            self._label2scores = {}
            self.runs = 0
            for path in self.filenames:
                with open(path) as f:
                    for line in f:
                        label,score = line.strip('\n').split('\t')
                        score = float(score)
                        if score == 0:
                            continue
                        l = self._label2scores.setdefault(str(label),[])
                        l.append(score)
                    self.runs += 1
        return self._label2scores
    def padded_score(self,label):
        bg_scores = self.label2scores().get(str(label),[])
        missing = self.runs - len(bg_scores)
        return bg_scores + [0.0]*missing

def direction_accumulator(ts,idxs):
    import algorithms.pathsum2 as ps
    return ps.Accumulator(
                ts.colsets('direction',idxs),
                bin_col=None,
                product=-1,
                )

def evidence_accumulator(ts,weight_map):
    import algorithms.pathsum2 as ps
    return ps.Accumulator(
                ts.colsets('evidence',list(weight_map.keys())),
                weight_map=weight_map,
                )

def get_random_prots(n):
    raise NotImplementedError('deprecated; use InPlacePathsum instead')

class PathsumBaseWorkItem(WorkItem):
    @classmethod
    def tissue_fn(cls,indir,tissue_id):
        import os
        return os.path.join(indir,'tissue'+str(tissue_id))
    @classmethod
    def build_tissue_file(cls,indir,tissue_id,ev_thresh,fc_thresh,counts=False):
        from browse.models import Tissue
        prot_counts = 0
        with open(cls.tissue_fn(indir,tissue_id),'w') as f:
            t = Tissue.objects.get(pk=tissue_id)
            for sp in t.sig_results(
                        ev_cutoff=ev_thresh,
                        fc_cutoff=fc_thresh,
                        ):
                f.write('\t'.join([
                            sp.uniprot,
                            str(sp.evidence),
                            str(sp.direction),
                            ])+'\n')
                counts += 1
        if counts:
            return counts
    @classmethod
    def build_nontissue_file(cls,indir,key,score_d):
        with open(cls.tissue_fn(indir,key),'w') as f:
            for uni,score in six.iteritems(score_d):
                f.write('\t'.join([
                            uni,
                            str(score),
                            '0', # XXX dummy direction -- suppress?
                            ])+'\n')
    @classmethod
    def dpi_map_fn(cls,indir,ws_id,p2d_choice):
        keyspace,version = p2d_choice.split('.')
        import os
        return os.path.join(indir,'dpi_map.%s.%d'%(keyspace,ws_id))
    @classmethod
    def build_dpi_map(cls,indir,ws_id,p2d_choice):
        # generate dpi mapping file
        from dtk.prot_map import DpiMapping
        dpi = DpiMapping(p2d_choice)
        fn = cls.dpi_map_fn(indir,ws_id,p2d_choice)
        with open(fn,'w') as f:
            from browse.models import Workspace
            ws = Workspace.objects.get(pk=ws_id)
            for k,v in six.iteritems(dpi.get_wsa_id_map(ws)):
                f.write('\t'.join([k]+[str(x) for x in v])+'\n')

class PathsumWorkItem(PathsumBaseWorkItem):
    # pathsum stack layout
    tissue_idx=0
    t2p_idx=1
    p2p_idx=2
    p2d_dir_idx=3
    p2d_indir_idx=4
    detail_file=False
    compress_detail_file=True
    show_stats=False
    map_to_wsa=True
    drop_ppi_pct=0
    def get_dpi_map(self):
        fn = self.dpi_map_fn(self.indir,self.ws_id,self.p2d_file)
        with open(fn) as f:
            result = {}
            for line in f:
                rec = line.strip('\n').split('\t')
                result[rec[0]] = [str(x) for x in rec[1:]]
        return result
    def run(self,context):
        self.context = context
        import algorithms.pathsum2 as ps
        self.p2d_src_dest = [(self.t2p_idx,self.p2d_dir_idx)]
        self.net = ps.Network()
        # The core pathsum algorithm is invoked via the three
        # calls to self.net.run_mapping() below, followed by
        # construction and invoking of a score accumulator.
        #
        # load tissues
        t2p = iter(self.t2p_records())
        if self.combo_with and self.combo_type=='sub':
            t2p = subtract_fixed_from_t2p(t2p,self.combo_fixed)
        self.net.run_mapping('t2p',
                t2p,
                [(self.tissue_idx,self.t2p_idx)],
                auto_add=True,
                )
        # load ppi
        if self.p2p_file:
            from dtk.prot_map import PpiMapping
            ppi = PpiMapping(self.p2p_file)
            filter_parms=dict(
                    ppi=ppi,
                    thresh=self.p2p_t,
                    )
            if self.drop_ppi_pct:
                # Column 1 is matched against the tissue side, and column
                # 0 against the drug side.  Empirically, filtering on 0
                # seems to work better, and this seems reasonable, because
                # it would remove all paths to dropped drug targets
                # simultaneously.
                filter_col = 0
                import random
                seed =  str(random.random())
                from dtk.data import RandomKeyset
                filter_parms['key_filter'] = (
                        filter_col,
                        RandomKeyset(int(100-self.drop_ppi_pct),100,seed),
                        )
            self.net.run_mapping('p2p',
                            filter_from_ppi(**filter_parms),
                            [(self.t2p_idx,self.p2p_idx)],
                            )
            self.p2d_src_dest.append((self.p2p_idx,self.p2d_indir_idx))
            if self.show_stats:
                print(len(self.net.stack[self.p2p_idx]),'ppi keys')
        # load dpi
        # The dpi_map is required even if we want to report native
        # drug keys, because it allows us to filter which DPI records
        # get processed, which can speed things up significantly.
        # So, it's always passed to filter_dpi_from_file, and a
        # separate config flag determines if it also gets passed
        # to convert_dpi_to_wsa_keys.
        dpi_map=self.get_dpi_map()
        from dtk.prot_map import DpiMapping
        dpi = DpiMapping(self.p2d_file)
        dpi_src = filter_dpi_from_file(dpi.get_path(),
                            dpi_map=dpi_map,
                            thresh=(2,float(self.p2d_t)),
                            )
        if self.combo_with and self.combo_type=='add':
            dpi_src = add_fixed_to_dpi(dpi_src,self.combo_fixed)
        if self.map_to_wsa:
            dpi_src = convert_dpi_to_wsa_keys(dpi_src,dpi_map)
        self.net.run_mapping('p2d',
                    dpi_src,
                    self.p2d_src_dest,
                    )
        if self.show_stats:
            if self.p2p_file:
                print("%d direct keys, %d indirect keys" % (
                        len(self.net.stack[self.p2d_dir_idx]),
                        len(self.net.stack[self.p2d_indir_idx]),
                        ))
            else:
                print("%d direct keys" % (
                        len(self.net.stack[self.p2d_dir_idx]),
                        ))
        # extract scores
        self.ts = ps.TargetSet()
        self.ts.load_from_network(self.net,(x[1] for x in self.p2d_src_dest))
        def extract_weights(l):
            return [float(context[x+'_w']) for x in l.split()]
        # direct score
        dir_weights = extract_weights('t2p p2d')
        accum = [evidence_accumulator(self.ts,{self.p2d_dir_idx:dir_weights})]
        direction_idxs = [self.p2d_dir_idx]
        if self.p2p_file:
            # indirect score
            indir_weights = extract_weights('t2p p2p p2d')
            accum.append(evidence_accumulator(self.ts,
                        {self.p2d_indir_idx:indir_weights}
                        ))
            direction_idxs += [self.p2d_indir_idx]
        # direction
        accum.insert(0,direction_accumulator(self.ts,direction_idxs))
        import os
        if self.detail_file:
            fn='path_detail%d.tsv'%self.serial
            if self.compress_detail_file:
                fn = fn + '.gz'
            fn=os.path.join(self.outdir,fn)
        else:
            fn=None
        self.ss = ps.ScoreSet(self.ts,accum,fn)
        if self.show_stats:
            print("%d paths examined; %d scores calculated" % (
                        self.ss.paths,
                        len(self.ss.scores),
                        ))
        #print list(self.ss.scores.iteritems())[:10]
        # The above looks like:
        # [(wsa_id_as_string,[direction,direct_score,indirect_score]), ... ]
        to_write = ['direction','direct']
        if self.p2p_file:
            to_write.append('indirect')
        for idx,stem in enumerate(to_write):
            vec = [(k,v[idx]) for k,v in six.iteritems(self.ss.scores)]
            fn=os.path.join(self.outdir,stem+'%dscore'%self.serial)
            with open(fn,'w') as f:
                for label,v in vec:
                    f.write(label+'\t'+str(v)+'\n')
    def t2p_records(self):
        t2p=['protein tissue evidence direction'.split()]
        for tissue_id in self.tissues:
            with open(self.tissue_fn(self.indir,tissue_id)) as f:
                recs = [
                    line.strip('\n').split('\t')
                    for line in f
                    ]
            if self.randomize:
                recs = [
                        [new_id]+rec[1:]
                        for new_id,rec in zip(get_random_prots(len(recs)),recs)
                        ]
            if self.show_stats:
                # NOTE: this output is needed for the method
                # extract_tissue_count_from_log() to work
                print('tissue',tissue_id,'has',len(recs),'significant proteins')
            for rec in recs:
                t2p.append([
                        rec[0], # protein id
                        str(tissue_id),
                        rec[1],
                        str(rec[2]), # direction
                        ])
        return t2p

def extract_tissue_count_from_log(job_id):
    from runner.common import LogRepoInfo
    lri = LogRepoInfo(job_id)
    lri.fetch_log()
    import re
    from dtk.files import get_file_lines
    tissue_count = 0
    for line in get_file_lines(lri.log_path()):
        m = re.match(r'tissue .* has (\d+) significant proteins',line)
        if m and m.group(1) != '0':
            tissue_count += 1
    return tissue_count

class InPlacePathsum:
    # This is an independent implementation of the pathsum algorithm,
    # designed to speed up background generation. Specifically:
    # - it's less general; the 'pathsum2' algorithm assumes any amount of
    #   data can be associated with each segment of the path, and is extracted
    #   via a very abstract scoring interface. This algorithm assumes that
    #   each segment has evidence and direction values, and that there are
    #   fixed scoring algorithms for each (although the scoring algorithms
    #   themselves are held in a separate class).
    # - it doesn't implement some older special cases, like combo therapies
    #   (although these could be added back in without affecting performance
    #   much)
    # - it doesn't do as much on-the-fly network construction and filtering
    #   as the data is read, so that what lands in memory can be re-used
    #   multiple times with different random labels applied to the tissue
    #   data
    # All this happens without much change to the external interface of the
    # WorkItem, so the code in run_gpbr.py is only minimally changed.
    #
    # The code that extracts and scores drugs (in random_cycle()) has been
    # tweaked somewhat from the simplest, most straightforward implementation,
    # but lots of more advanced optimizations were tried and proved to be
    # unsuccessful. Some of this code is left in place for reference, but
    # maybe should be removed in the future for clarity. The main issue
    # seems to be that so little of the potential network actually ends
    # up matching a target that most work done in advance turns out to
    # be wasted.
    # 
    # If more performance is needed, we should look at some more serious
    # restructurings (which would propagate up into run_gpbr). For example:
    # - rather than converting DPI/PPI data into partial paths at the start
    #   of each drug being processed, maybe that could be done once up
    #   front, and the results of that shipped to the worker as input
    # - possibly rather than have each process do a specific number of cycles
    #   on all drugs, we could shard drugs between processes, and have each
    #   process run its subset of drugs through a much larger number of
    #   cycles. If paths were accumulated across this much larger number
    #   of targets, and then run through numpy in a much larger batch, maybe
    #   the faster numpy calculations would overcome the setup cost
    def __init__(self):
        self.tissues = []
        self.verbose = False
    def load_uniprots(self,protein_names):
        # Sort for determinism
        self.all_prots = list(sorted(set(protein_names)))
        if self.verbose:
            print('loaded',len(self.all_prots),'protein names')
    def load_wsa_map(self,src):
        from dtk.data import MultiMap
        self.d2wsa_mm = MultiMap(src)
        self.valid_drugkeys = set(self.d2wsa_mm.fwd_map().keys())
        if self.verbose:
            print('loaded',len(self.valid_drugkeys),'drug keys')
            print('   for',len(self.d2wsa_mm.rev_map()),'wsas')
    def _load_pi(self,src):
        result = {}
        for rec in src:
            d = result.setdefault(rec[0],{})
            d[rec[1]] = rec[2:]
        return result
    def load_d2p_map(self,src):
        self.d2p = self._load_pi(src)
        self.valid_prot1s = set()
        for d in self.d2p.values():
            self.valid_prot1s |= set(d.keys())
        if self.verbose:
            print('loaded interactions for',len(self.d2p),'drugs')
            print('   with',len(self.valid_prot1s),'targets')
    def load_p2p_map(self,src):
        self.p2p = self._load_pi(src)
        self.valid_prot2s = set()
        for d in self.p2p.values():
            self.valid_prot2s |= set(d.keys())
        if self.verbose:
            print('loaded interactions for',len(self.p2p),'proteins')
            print('   with',len(self.valid_prot2s),'level 2 targets')
    def add_tissue(self,src):
        # build dict of tissue protein mappings; filter prots here,
        # which guarantees no tissue can have more uniprot ids than
        # the random set we're sampling from
        d = {
                prot:(ev,direction)
                for prot,ev,direction in src
                if prot in self.all_prots
                }
        if not d:
            # CAPP has empty "tissues" (actually co-morbidities);
            # skipping them up front speeds things up.
            return
        if self.verbose:
            print('loaded %d significant proteins for tissue %d' %(
                    len(d),
                    len(self.tissues),
                    ))
        self.tissues.append(d)
    def random_cycle(self,bsa):
        # create randomized tissues
        tissues = []
        tissue_prots = []
        all_tissue_prots = set()
        from random import sample
        possible_tissue_matches = self.valid_prot1s | self.valid_prot2s
        for t in self.tissues:
            prots = sample(self.all_prots,len(t))
            # any random protein key that's not in possible_tissue_matches
            # won't be in a path anyway, so just drop these here; if everything
            # in the tissue gets dropped, skip the whole tissue
            d = {
                    new_key:val
                    for (key,val),new_key in zip(t.items(),prots)
                    if new_key in possible_tissue_matches
                    }
            if not d:
                continue
            tissues.append(d)
            tissue_prots.append(set(tissues[-1].keys()))
            all_tissue_prots |= tissue_prots[-1]
        # generate and score paths for each drug
        for drug_key,drug_dpi in self.d2p.items():
            # Before iterating through the tissues, do an up-front construction
            # of all the possible direct and indirect paths (with the tissue
            # part missing). Then match the endpoints of those paths against
            # each tissue.
            dir_parts = {}
            indir_parts = {}
            for prot1,dpi_vals in drug_dpi.items():
                if prot1 in all_tissue_prots:
                    dir_parts[prot1] = dpi_vals
                for prot2,ppi_vals in self.p2p.get(prot1,{}).items():
                    if prot2 in all_tissue_prots:
                        s = indir_parts.setdefault(prot2,set())
                        s.add((dpi_vals,ppi_vals))
            # Once the partial paths are constructed, the options are:
            # - iterate through partials, and look up tissues
            # - iterate through tissues, and look up partials
            # - iterate through intersection of keys, and look up both
            # Because key matches are relatively rare, the last option
            # is fastest.
            d_keys = set(dir_parts.keys())
            i_keys = set(indir_parts.keys())
            for i,tissue in enumerate(tissues):
                tprots = tissue_prots[i]
                for prot1 in d_keys & tprots:
                    bsa.add_dpath(i,dir_parts[prot1],tissue[prot1])
                for prot2 in i_keys & tprots:
                    for dpi_vals,ppi_vals in indir_parts[prot2]:
                        bsa.add_ipath(i,dpi_vals,ppi_vals,tissue[prot2])
            bsa.roll_up_drug(drug_key)
            
    def random_cycles(self, bsa, cycles, serial):
        """A reimplementations of the random_cycle, using numpy/scipy and some other speedups."""
        import numpy as np
        from dtk.numba import accum_max_sum_row
        protverse = self.valid_prot1s | self.valid_prot2s

        # Sort for determinism.
        self.d2p = dict(sorted(self.d2p.items()))

        # Sort for determinism.
        prot2idx = {x:i for i,x in enumerate(sorted(protverse))}

        # Sparse matrices containing the 'best' weighted path from each
        # drug to each prot
        dir_mat, indir_mat = self._setup_partial_paths(prot2idx, bsa)

        num_drugs = len(self.d2p)
        accum_drug_dir_scores = np.zeros(num_drugs)
        accum_drug_indir_scores = np.zeros(num_drugs)

        for tissue in self.tissues:
            for cycle_idx in range(cycles):
                # To make this deterministic, each cycle is run with a predetermined seed based on its index.
                seed = serial * cycles + cycle_idx
                # A np.array with the tissue scores for each protein
                tissue_mat = self._make_random_tissue_mat(tissue, prot2idx, seed=seed)

                for drug_idx in range(num_drugs):
                    # For each row in the drug-prot matrix, add it to the tissue mat and accumulate.
                    accum_max_sum_row(dir_mat, tissue_mat, drug_idx, bsa.d_weights[1], out=accum_drug_dir_scores)
                    accum_max_sum_row(indir_mat, tissue_mat, drug_idx, bsa.i_weights[2], out=accum_drug_indir_scores)

        # This implementation doesn't really use bsa the way expected;
        # instead, just insert the expected values into bsa.
        for i, drug_key in enumerate(self.d2p.keys()):
            scores = [0, accum_drug_dir_scores[i], accum_drug_indir_scores[i]]
            bsa.drug_scores[drug_key] = scores
        bsa.cycle_count = cycles
    

    def _setup_partial_paths(self, prot2idx, bsa):
        from collections import defaultdict
        from scipy.sparse import lil_matrix

        num_drugs = len(self.d2p)
        num_prots =  len(prot2idx)
        dir_mat = lil_matrix((num_drugs, num_prots))
        indir_mat = lil_matrix((num_drugs, num_prots))
        for i, (drug_key,drug_dpi) in enumerate(self.d2p.items()):
            dir_row = {}
            indir_row = defaultdict(float)
            for prot1,dpi_vals in drug_dpi.items():
                prot1_idx = prot2idx[prot1]
                dir_row[prot1_idx] = dpi_vals[0] * bsa.d_weights[0]

                for prot2,ppi_vals in self.p2p.get(prot1, {}).items():
                    prot2_idx = prot2idx[prot2]
                    val = dpi_vals[0] * bsa.i_weights[0] + ppi_vals[0] * bsa.i_weights[1]
                    indir_row[prot2_idx] = max(indir_row[prot2_idx], val)

            dir_mat.rows[i] = list(dir_row.keys())
            dir_mat.data[i] = list(dir_row.values())
            indir_mat.rows[i] = list(indir_row.keys())
            indir_mat.data[i] = list(indir_row.values())
        
        return dir_mat.tocsr(), indir_mat.tocsr()
    def _make_random_tissue_mat(self, tissue, prot2idx, seed):
        from random import Random
        rng = Random(seed)
        import numpy as np
        num_prots = len(prot2idx)
        tissue_mat = np.zeros(num_prots)
        prots = rng.sample(self.all_prots,len(tissue))
        for (key,val),new_key in zip(sorted(tissue.items()),prots):
            new_key_idx = prot2idx.get(new_key, None)
            if new_key_idx is None:
                continue
            tissue_mat[new_key_idx] = val[0]
        return tissue_mat

class BackgroundScoreAccumulator:
    def __init__(self,d2p_w,p2p_w,p2t_w):
        self.d2p_w = d2p_w
        self.p2p_w = p2p_w
        self.p2t_w = p2t_w
        # scale weights by the sum in advance, so we don't need
        # an extra division inside the scoring loop
        scale = lambda v:[x/sum(v) for x in v]
        self.d_weights = scale((d2p_w,p2t_w))
        self.i_weights = scale((d2p_w,p2p_w,p2t_w))
        self._reset_drug_counters()
        self.drug_scores = {}
        self.cycle_count = 0
    def _reset_drug_counters(self):
        self.by_tissue = {}
        self.directions = 0
    def _tissue_counters(self,i):
        # by starting these lists with a 0, we don't need to special-case
        # empty lists in roll_up_drug()
        return self.by_tissue.setdefault(i,([0],[0]))
    def add_dpath(self,i,d2p,p2t):
        self.directions += -1 * d2p[1] * p2t[1]
        l = self._tissue_counters(i)
        w = self.d_weights
        l[0].append(sum((
                w[0] * d2p[0],
                w[1] * p2t[0],
                )))
    def add_ipath(self,i,d2p,p2p,p2t):
        self.directions += -1 * d2p[1] * p2p[1] * p2t[1]
        l = self._tissue_counters(i)
        w = self.i_weights
        l[1].append(sum((
                w[0] * d2p[0],
                w[1] * p2p[0],
                w[2] * p2t[0],
                )))
    def roll_up_drug(self,drug_key):
        l = self.drug_scores.setdefault(drug_key,[0,0,0])
        l[0] += self.directions
        l[1] += sum([max(x[0]) for x in self.by_tissue.values()])
        l[2] += sum([max(x[1]) for x in self.by_tissue.values()])
        self._reset_drug_counters()
    def roll_up_cycle(self):
        self.cycle_count += 1
    def extract_scores(self,wsa_map):
        drug_keyed_scores = [
                [k]+[x/self.cycle_count for x in v]
                for k,v in self.drug_scores.items()
                ]
        drug_keyed_scores.sort(key=lambda x:-x[2])
        wsa_seen = set()
        result = []
        for rec in drug_keyed_scores:
            for wsa in wsa_map[rec[0]]:
                if wsa in wsa_seen:
                    continue
                wsa_seen.add(wsa)
                result.append([wsa]+rec[1:])
        return result

class BackgroundPathsumWorkItem(PathsumBaseWorkItem):
    def get_drugkey_wsa_pairs(self):
        fn = self.dpi_map_fn(self.indir,self.ws_id,self.p2d_file)
        from dtk.files import get_file_records
        for rec in get_file_records(fn,keep_header=None,parse_type='tsv'):
            for wsa_id in rec[1:]:
                yield (rec[0],wsa_id)
    def get_dpi_records(self,valid_drugkeys,ev_thresh):
        from dtk.prot_map import DpiMapping
        dpi = DpiMapping(self.p2d_file)
        from dtk.files import get_file_records
        for rec in get_file_records(dpi.get_path(),keep_header=False):
            if rec[0] not in valid_drugkeys:
                continue
            ev = float(rec[2])
            if ev < ev_thresh:
                continue
            yield (rec[0],rec[1],ev,int(rec[3]))
    def get_ppi_records(self,valid_prot1s,ev_thresh):
        from dtk.prot_map import PpiMapping
        ppi = PpiMapping(self.p2p_file)
        for rec in ppi.get_data_records(min_evid=ev_thresh):
            if rec[0] not in valid_prot1s:
                continue
            yield rec
    def get_tissue_records(self,tissue_id):
        fn = self.tissue_fn(self.indir,tissue_id)
        from dtk.files import get_file_records
        for rec in get_file_records(fn,keep_header=None,parse_type='tsv'):
            yield (rec[0],float(rec[1]),float(rec[2]))
    def get_uniprot_ids(self):
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned(
                'uniprot',
                self.uniprot_flavor,
                self.uniprot_role,
                )
        s3f.fetch()
        from dtk.files import get_file_records
        return set(rec[0] for rec in get_file_records(s3f.path()))
    def run(self,context):
        self.context = context
        # XXX both the following can be fixed by copying the logic in
        # XXX PathsumWorkItem, but there's no immediate need...
        if self.combo_with:
            raise NotImplementedError('gpbr does not support combo modes')
        if not self.p2p_file:
            raise NotImplementedError('gpbr requires indirect data')
        #print('context:',context)
        ipp = InPlacePathsum()
        if self.serial == 0:
            ipp.verbose = True
        # load all uniprot ids, needed for protein randomization
        ipp.load_uniprots(self.get_uniprot_ids())
        # load drugkey -> wsa mapping
        # This is used for filtering which drugs in the DPI file are in
        # the workspace, but all scores are generated based on drugkeys,
        # and are mapped back to WSAs at the end. This is consistent with
        # how normal (foreground) pathsum is performed, but NOT with how
        # backgrounds used to be calculated.
        ipp.load_wsa_map(self.get_drugkey_wsa_pairs())
        # load dpi and ppi data keeping only over-threshold records and
        # keys that match the level above
        ipp.load_d2p_map(self.get_dpi_records(ipp.valid_drugkeys,self.p2d_t))
        if self.p2p_file:
            ipp.load_p2p_map(self.get_ppi_records(ipp.valid_prot1s,self.p2p_t))
        # load tissues, which are pre-thresholded
        for tissue_id in self.tissues:
            ipp.add_tissue(self.get_tissue_records(tissue_id))
            # XXX eventually add combo subtract support

        bsa = BackgroundScoreAccumulator(self.p2d_w,self.p2p_w,self.t2p_w)

        # The two implementations below give the same results (within random noise).
        # The random_cycles implementation should be faster, but it's nice to
        # have two implementations to check against.
        if True:
            ipp.random_cycles(bsa, self.inner_cycles, self.serial + self.serial_offset)
        else:
            for i in range(self.inner_cycles):
                ipp.random_cycle(bsa)
                bsa.roll_up_cycle()

        # write accumulated average scores to files
        scores = bsa.extract_scores(ipp.d2wsa_mm.fwd_map())
        # XXX bypass direction write and rollup? This list could be
        # XXX provided in the configuration.
        to_write = ['direction','direct']
        if self.p2p_file:
            to_write.append('indirect')
        import os
        for idx,stem in enumerate(to_write):
            fn=os.path.join(self.outdir,stem+'%dscore'%self.serial)
            with open(fn,'w') as f:
                for rec in scores:
                    f.write(str(rec[0])+'\t'+str(rec[idx+1])+'\n')
