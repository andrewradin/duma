#!/usr/bin/env python3

import sys
import os
import logging
logger = logging.getLogger(__name__)

from path_helper import PathHelper,make_directory
import json
import numpy as np
import random
from dtk.subclass_registry import SubclassRegistry
from multiprocessing import Pool

EPS = 1e-9
# Sets the minimum/maximum (half-)width of the 'linear'ish portion of the sigmoid
# We don't want a cliff, enforce some sort of gradualness.
# Don't set min too low or you might run into numerical precision issues and explode
# your gradients.
MIN_WFSIG_WIDTH = 0.01
MAX_WFSIG_WIDTH = 0.1

# Run SGD every this many iterations.
SGD_PER_ITERS = 10

################################################################################
# Auto-tuners
################################################################################
class AutoTuner(SubclassRegistry):
    extra_parms=[]
    @classmethod
    def get_choices(cls):
        return [('','No')]+[
                (x[1].choice_code,x[1].choice_label)
                for x in cls.get_subclasses()
                ]
    @classmethod
    def get_class_by_choice(cls,choice):
        for x in cls.get_subclasses():
            if x[1].choice_code == choice:
                return x[1]
        raise RuntimeError("unknown autotune option '%s'"%choice)

class GridWalkAutoTuner(AutoTuner):
    # - this walks an integer grid of delta values
    # - _mod_weight converts the delta vector to a weight vector, based
    #   on 'spacing'
    # - each iteration tries all the possibilities available by changing
    #   one delta value up or down from the previous best value
    # - 'cap' specifies the maximum number of deltas away from the original
    #   to examine
    # - 'step' allows specifying a larger initial step value; this must be
    #   a power of two
    # - 'min_delta' specifies a minimum metric change from the previous best
    #   value before stopping (or before decreasing the stepsize, if step > 1)
    extra_parms=['cap','spacing','min_delta','step']
    choice_code='gridwalk'
    choice_label='Grid Walk'
    def _mod_weight(self,d,w):
        return w*(pow(1+self.spacing,d))
    def _make_candidate(self,deltas):
        class Dummy: pass
        cand = Dummy()
        cand.deltas = deltas
        mod_weights = np.array([
                self._mod_weight(d,w)
                for d,w in zip(deltas,self.initial_weights)
                ])
        cand.score = self.score_func(mod_weights)
        cand.metric = self.eval_func(cand.score)
        return cand
    def _get_current_mod_wts(self):
        return [
                self._mod_weight(d,w)
                for d,w in zip(self.best.deltas,self.initial_weights)
                ]
    def __init__(self,weights,score_func,eval_func,constraints,extra_settings,score_array,cores):
        self.cycle = 0
        self.score_func = score_func
        self.eval_func = eval_func
        self.spacing = extra_settings['spacing']
        self.cap = extra_settings['cap']
        self.min_delta = float(extra_settings['min_delta'])
        self.vstep = int(extra_settings['step'])
        # evaluate initial case
        self.initial_weights = weights
        self.tried = set()
        self.best = self._make_candidate(tuple([0]*len(self.initial_weights)))
 # for this approach there is only one, but for other tuners there are several stats
        self.stats = [self.best.metric]
        print('initial metric',self.best.metric)
        self.tried.add(self.best.deltas)
    def run_iteration(self):
        from tools import sci_fmt
        self.cycle += 1
        base = self.best
        attempts = []
        for i in range(len(base.deltas)):
            for direction in (self.vstep,-self.vstep):
                delta_vec = list(base.deltas)
                delta_vec[i] += direction
                if abs(delta_vec[i]) > self.cap:
                    continue # hit cap
                delta_vec = tuple(delta_vec)
                if delta_vec in self.tried:
                    # already examined
                    continue
                attempt = self._make_candidate(delta_vec)
                attempts.append( attempt )
                self.tried.add(delta_vec)
                if attempt.metric > self.best.metric:
                    print('+++',self.cycle,delta_vec,sci_fmt(attempt.metric))
                    self.best = attempt
                    self.stats = [self.best.metric]
        if self.best == base \
                or self.best.metric < base.metric * (1+self.min_delta):
            if self.vstep > 1:
                self.vstep = self.vstep/2
                print('dropping step size to',self.vstep)
                assert self.vstep and type(self.vstep) is int
            else:
                print('halting -- no improvement')
                for x in attempts:
                    print("   ",x.deltas,x.metric)
                return True
    def output_results(self,plot_scores,score_labels,f,archive_labels):
        print('Final weights:')
        from tools import sci_fmt
        for a,l,d,w in zip(
                    archive_labels,
                    score_labels,
                    self.best.deltas,
                    self.initial_weights,
                    ):
            m = self._mod_weight(d,w)
            print(sci_fmt(m),l,'-- initial',sci_fmt(w),'delta',d)
            plot_scores[l] = m
            f.write("\t".join([a,str(m)]) + "\n")


def low_var_weighted_sample(pop, score_fn, cnt, rng):
    scores = [score_fn(x) for x in pop]
    sum_scores = sum(scores)
    width = sum_scores / cnt
    x0 = rng.random() * width
    th = x0
    x = 0

    for i, scorei in enumerate(scores):
        x += scorei
        while x >= th:
            yield pop[i]
            th += width



pool_func = None
def pool_init(func):
    global pool_func
    pool_func = func

def pool_run(*args):
    global pool_func
    return pool_func(*args)



class GeneticAutoTuner(AutoTuner):
    choice_code='genetic'
    choice_label='Genetic'
    # - each individual is a set of weights
    #   - an evaluation function caches the corresponding score and metric
    #     for each individual
    # - a population of individuals is constructed initially, and then
    #   replaced with a new population each generation
    # - the next generation includes:
    #   - the top n individuals from the previous generation
    #   - k additional individuals randomly chosen from the previous generation
    #   - enough new individuals that the population size remains constant
    # - new individuals are constructed by:
    #   - pairing random survivors
    #   - combining their weight vectors
    #   - applying mutations
    # - the initial population is created by mutating a passed-in initial
    #   weight vector
    # - so, some potential parameters (all currently hardwired) are:
    #   - number of top scorers kept
    #   - number of random individuals kept
    #   - number of new individuals created
    #   - mutation rate and mean of mutation amount
    # XXX - this might benefit from some sort of annealing, where the amount
    # XXX   of mutation decreases towards the end of the run
    # XXX - maybe prefer the parent with the higher metric in crossover?
    class Individual:
        def __init__(self,weights):
            assert len(weights) > 0
            self.weights = np.array(weights,dtype=float)
            if False:
                # XXX It seems like crossover would be more meaningful if
                # XXX weights were somehow normalized so they would more easily
                # XXX mix and match from different individuals. But trying this
                # XXX exposes some problems:
                # XXX - since in the general case some weights will be negative,
                # XXX   the magnitudes will be driven up higher than expected
                # XXX - possibly as a result of this, the heatmaps become less
                # XXX   informative
                # XXX - since the mutation offsets are independent of weight
                # XXX   scale, they affect low-magnitude weights more (this
                # XXX   might be a general issue that normalizing merely
                # XXX   draws attention to)
                # XXX In any case, this is disabled for now.
                self.weights /= sum(self.weights)
    extra_parms=[
            'top_count','extra_count','new_count',
            'init_sigma','init_freq','mut_sigma','mut_freq',
            'w_min','w_max',
            'n_parents',
            'iter',
            'seed',
            'anneal_cycles',
            'frac_children',
            'weighted_resample',
            'anneal_sigma',
            'l2_reg',
            'directed_mutate',
            'dropout',
            'reg_importance',
            ]
    def __init__(self,agg_model,eval_func,constraints,extra_settings,cores):
        self.cycle = 0
        self.agg_model = agg_model
        self.eval_func = eval_func
        self.top_count = extra_settings['top_count']
        self.extra_count = extra_settings['extra_count']
        self.new_count = extra_settings['new_count']
        self.mut_sigma = extra_settings['mut_sigma']
        self.mut_freq = extra_settings['mut_freq']
        self.w_min = extra_settings['w_min']
        self.w_max = extra_settings['w_max']
        self.n_parents = extra_settings['n_parents']
        self.num_iters = extra_settings['iter']
        self.constraints = constraints
        self.anneal_cycles = extra_settings['anneal_cycles']
        self.frac_children = extra_settings['frac_children']
        self.weighted_resample = extra_settings['weighted_resample']
        self.anneal_sigma = extra_settings['anneal_sigma']
        self.l2_reg = extra_settings['l2_reg']
        self.dropout = extra_settings['dropout']
        self.reg_importance = extra_settings['reg_importance']
        seed = extra_settings.get('seed', 0)
        if seed == -1:
            print("Randomizing our random seed")
            import time
            seed = time.time()
        self.rng = random.Random(seed)
        # generate initial population
        self.population = [
                self.Individual(self._mutate(self.agg_model.initial_weights()))
                for i in range(self.top_count+self.extra_count+self.new_count)
                ]
        self._apply_constraints_to(self.population)
        self.wt_sds=[]


        pool = Pool(initializer=pool_init, initargs=(self.opt_and_score_item,), processes=cores)
        self.pool = pool

        self._score_and_order()

    def _dist(self,p1,p2):
        diff = p1.weights - p2.weights # expects nparrays
        return np.sqrt(diff.dot(diff))
    def _show_population_spread(self,label):
        print(label,'population euclidean distances and metrics:')
        dist = [(p,self._dist(p,self.best)) for p in self.population]
        from tools import sci_fmt
        for d in sorted(dist,key=lambda x:x[1]):
            print(' ',sci_fmt(d[1]),sci_fmt(d[0].metric))
    def opt_and_score_item(self, item, do_opt, save_sgd_weights):
        if do_opt and self.agg_model.sgd_enabled():
            weights, dbg_grads, dbg_weights = self.agg_model.sgd(item.weights, output_debug=save_sgd_weights)
            item.weights = weights
            item.dbg_grads = dbg_grads
            item.dbg_weights = dbg_weights
            item.metric = None
        else:
            item.dbg_grads = None
            item.dbg_weights = None

        if getattr(item, 'metric', None) is None:
            # Compute a new score if we don't have one (or if we have
            # dropout, in which case cached scores become stale).
            # This branch is currently disabled.  Dropout is implemented in the
            # sgd path, but this version of it causes sampling issues, revisit later.
            if False and self.dropout > 0:
                weights = np.array(item.weights)
                dcnt = int(len(weights) * self.dropout * self._anneal_mult())
                dcnt = max(dcnt, 1)
                dropouts = self.rng.sample(range(len(weights)), dcnt)
                weights[dropouts] = 0
            else:
                weights = item.weights


            normed = self.agg_model.post_norm(weights)
            item.score = self.agg_model.score(normed, weights)
            item.metric = self.eval_func(item.score)
            item.orig_metric = item.metric
            if self.l2_reg > 0:
                if self.reg_importance:
                    score_weights = self.agg_model.score_weights(weights)
                    # Normally we sum across rows to create drug scores
                    # Instead, sum down columns to get feature scores.
                    importances = np.sum(normed, axis=0) * score_weights
                    mean_imp = np.mean(importances)
                    norm_imp = (importances - mean_imp) / (mean_imp + EPS)
                    reg_score = np.mean(np.abs(norm_imp))
                    from dtk.num import sigma
                    # reg_score can sometimes go large, use a sigmoid so that it
                    # just approaches 1 instead.
                    reg_score = (sigma(reg_score*2) - 0.5) * 2
                else:
                    weights = np.array(self.agg_model.score_weights(weights))
                    mid_weight = (self.w_max + self.w_min) / 2

                    keep_idxs = set(range(len(weights)))
                    extra_weights = []
                    # Treat constrained weights as a single combined weight
                    # for regularization purposes.
                    for constraint in self.constraints:
                        combo_weight = constraint.get_combined_weight(weights)
                        idxs = constraint.weight_idxs()
                        if not idxs:
                            continue
                        keep_idxs -= set(idxs)
                        extra_weights.append(combo_weight)

                    pen_score_weights = np.concatenate([
                        weights[list(keep_idxs)], extra_weights])
                    pen_weights = np.abs(mid_weight - pen_score_weights) / ((self.w_max - self.w_min) / 2)
                    reg_score = np.mean(pen_weights**2)
                l2_penalty = item.metric * reg_score * self.l2_reg
                item.metric -= l2_penalty
        return item
    def _score_and_order(self):
        do_opt = True if self.cycle % SGD_PER_ITERS == 0 else False
        do_opt = [do_opt] * len(self.population)
        save_sgd_weights = [False] * len(self.population) 
        save_sgd_weights[0] = True
        self.population = self.pool.starmap(pool_run, zip(self.population, do_opt, save_sgd_weights))
        self.dbg_weights = self.population[0].dbg_weights
        self.dbg_grads = self.population[0].dbg_grads
        self.population.sort(key=lambda x:-x.metric)
        self.best = self.population[0]
        temp = [x.metric for x in self.population]
        orig_metrics = [x.orig_metric for x in self.population]
        # self.stats ends up displayed in the 'Auto-tuned results' line plot.
        self.stats = self._calc_stats(temp) + self._calc_stats(orig_metrics)
        prefix = 'Gen %d (anneal_mult: %.2f)' % (self.cycle, self._anneal_mult())
        self._show_stats(orig_metrics, prefix+(' (Base Metrics) :'))
        self._show_stats(temp, prefix+(' (L2reg Metrics):'))
        self._get_wt_diversity()
        if self.cycle % 10 == 0:
            self._show_population_spread(prefix)
    def _get_wt_diversity(self):
        from dtk.num import avg_sd,median
        sds=[]
        for i in range(len(self.best.weights)):
            tmp = [indiv.weights[i] for indiv in self.population]
            _,s = avg_sd(tmp)
            sds.append(s)
        self.wt_sds.append([min(sds), median(sds), max(sds)])
    def _show_weight_stats(self,label_vec):
        print('Gen %d Weights:'%self.cycle)
        weights = self.agg_model.score_weights(self.best.weights)
        for i in range(len(weights)):
            self._show_stats(
                    [weights[i] for x in self.population],
                    '  '+label_vec[i],
                    )
    def _heatmap_population(self,fn,row_labels, title_prefix=''):
        from dtk.plot import plotly_heatmap
        import numpy as np
        data = None
        col_labels = []
        for i,p in enumerate(self.population):
            weights = self.agg_model.score_weights(p.weights)
            if data is None:
                data = [[w] for w in weights]
            else:
                assert len(weights) == len(data)
                for j,w in enumerate(weights):
                    data[j].append(w)
            col_labels.append('Indiv.' + str(i) + ': %.3f'%(p.metric))
        pp = plotly_heatmap(np.array(data),
                            row_labels,
                            col_labels=col_labels,
                            color_zero_centered = True,
                            Title = title_prefix+' Population weights',
                            precalcd = True
                           )
        pp.save(fn)
    
    def _heatmap_sgd(self, fn_prefix, score_labels):
        itr = self.cycle
        if not self.dbg_grads:
            # We seem to occasionally hit this, not sure why, but not a huge
            # issue so debug later.
            logger.warning("Missing saved sgd values for heatmap_sgd")
            return
        logger.info(f"Plotting gradient heatmaps to {fn_prefix}")
        fn_suffix = f'hm.plotly'

        from dtk.plot import plotly_heatmap
        import numpy as np
        grad_hm = np.nan_to_num(np.array(self.dbg_grads))
        splits = grad_hm.shape[1] // len(score_labels)
        assert splits * len(score_labels) == grad_hm.shape[1], f"Expected grad dim {grad_hm.shape[1]} to be even multiple of labels {len(score_labels)}"
        split_sz = grad_hm.shape[1] // splits
        weight_group_labels = self.agg_model.weight_group_labels
        
        for i, weight_group in zip(range(splits), weight_group_labels):
            data = grad_hm[:, i*split_sz:(i+1)*split_sz]
            grad_plot = plotly_heatmap(
                data,
                row_labels=[f'iter {i}' for i in range(data.shape[0])],
                col_labels=score_labels,
                reorder_rows=False,
                reorder_cols=False,
                color_zero_centered=True,
                Title=f'SGD Gradients For {weight_group} Iter {itr}',
            )

            grad_plot.save(f"{fn_prefix}/sgd.itr{itr:04d}.spl{i}.weights.{fn_suffix}")

        weight_hm = np.nan_to_num(np.array(self.dbg_weights))
        for i, weight_group in zip(range(splits), weight_group_labels):
            data = weight_hm[:, i*split_sz:(i+1)*split_sz]
            weight_plot = plotly_heatmap(
                data,
                row_labels=[f'iter {i}' for i in range(data.shape[0])],
                col_labels=score_labels,
                reorder_rows=False,
                reorder_cols=False,
                color_zero_centered=True,
                Title=f'SGD Weights For {weight_group} Iter {itr}',
            )
            weight_plot.save(f"{fn_prefix}/sgd.itr{itr:04d}.spl{i}.grads.{fn_suffix}")


    def _show_stats(self,vec,label):
        from tools import sci_fmt
        logger.info("%s %s", label, ' / '.join([sci_fmt(x)
                                for x in self._calc_stats(vec)
                               ]))
    def _calc_stats(self, vec):
        from dtk.num import avg
        return [max(vec), avg(vec),min(vec)]
    def _make_child(self,parents):
        num_weights = len(parents[0].weights)
        if self.frac_children:
            assert len(parents) == 2
            alphas = [self.rng.random() for _ in range(num_weights)]
            weights = [
                    parents[0].weights[i] * alpha + parents[1].weights[i] * (1-alpha)
                    for alpha, i in zip(alphas, range(num_weights))
                    ]
        else:
            weights = [
                    parents[self.rng.randrange(len(parents))].weights[i]
                    for i in range(num_weights)
                    ]
        return self.Individual(self._mutate(weights))

    def _mutate(self, weights):
        return self.agg_model.mutate(weights, self._anneal_mult())

    def _anneal_mult(self):
        if self.anneal_cycles == 0:
            return 1

        # frac_base will cycle from 1 to 0 repeatedly (self.anneal_cycles times)
        frac_base = (1.0 - ((self.cycle+1) * self.anneal_cycles / float(self.num_iters))) % 1
        # Each cycle will be less extreme than the previous.
        frac = frac_base * ((1.0 - self.cycle / float(self.num_iters)) * 0.5 + 0.5)

        high = 1.5
        low = 0.1
        return low + (high - low) * frac

    def run_iteration(self):
        # promote top and extra
        next_gen = self.population[:self.top_count]
        if self.weighted_resample:
            anneal_mult = self._anneal_mult()
            # This is a parameter that either spreads or squeezes the scores
            # Spreading the scores increases the differences between them,
            # which leads to us keeping more of the top scores.
            # Squeezing the scores reduces the differences between them which
            # means we will keep more of a random set.
            # Early on we want more diversity so we squeeze; as the process
            # goes on we want to narrow in on our top scores and tweak those
            # so we do the opposite.
            acceptance = 0.5 / anneal_mult
            next_gen += list(low_var_weighted_sample(self.population,
                                                     lambda x: x.metric ** acceptance,
                                                     self.extra_count,
                                                     rng=self.rng))
        else:
            next_gen += self.rng.sample(
                    self.population[self.top_count:],
                    self.extra_count,
                    )
        # generate children
        next_gen += [
                self._make_child(self.rng.sample(next_gen,self.n_parents))
                for i in range(self.new_count)
                ]

        # Apply any constraints to our new generation.
        self._apply_constraints_to(next_gen)
        self.population = next_gen
        self.cycle += 1

        self._score_and_order()
    def _apply_constraints_to(self, next_gen):
        for constraint in self.constraints:
            for individual in next_gen:
                constraint.apply_to(individual.weights)
    def output_results(self,plot_scores,score_labels,f,archive_labels):
        self._show_weight_stats(score_labels)
        for i,tup in enumerate(zip(score_labels,self.best.weights)):
            l,v = tup
            plot_scores[l] = v
        from tools import sci_fmt
        print('Final weights:')
        for l,v in sorted(plot_scores.items(),key=lambda x:-x[1]):
            print(' ',sci_fmt(v),l)
        self._show_population_spread('Final')
        for l,v in zip(archive_labels,self.best.weights):
            f.write("\t".join([l,str(v)]) + "\n")


def run(feature_matrix_fn, parms_fn, outdir_fn, plotdir_fn, cores):
    import multiprocessing
    if cores > multiprocessing.cpu_count():
        # This usually happens during tests or if we're running locally.
        # Running with 16+ cores on a 1 core machine makes it sad.
        logger.warning("Ignoring requested %d cores, using max machine", cores)
        cores = multiprocessing.cpu_count()
    import dtk.features as feat
    logger.info("Loading feature matrix")
    fm = feat.FMBase.load_from_file(feature_matrix_fn)

    logger.info("Loading parameters")
    with open(parms_fn) as f:
        parms = json.loads(f.read())

    auto = parms['auto_tune']

    logger.info("Autotuning")
    runner = AutoTuneRunner(parms, fm.data, fm.feature_names, outdir_fn, plotdir_fn)
    runner.auto_tune(AutoTuner.get_class_by_choice(auto), cores=cores)

    plot = plot_combined(fm, parms, runner.tuner.best.weights)
    combo_wt_plot = os.path.join(plotdir_fn, "combined.plotly")
    plot.save(combo_wt_plot)

def assess_norm(score_mat, fm, parms):
    drug_set = parms['train_wsa_ids']

    cnt=[[fm.feature_names[i],0]
            for i in range(score_mat.shape[1])
        ]
    for i,w in enumerate(parms['all_wsa_ids']):
        if w in drug_set:
            for j,v in enumerate(score_mat[i]):
                if v > 0.:
                    cnt[j][1] += 1
    maxs = np.nanmax(score_mat, axis=0)
    print("\t".join(['CM score name',
                        "Eval drugs with a norm'd score",
                        "max norm'd score"
                    ]))
    for i, l in enumerate(cnt):
        print("\t".join([str(x) for x in l+[maxs[i]]]))

def shapes_for_algo(parms, data, xaxis, yaxis, y_max, agg_model, weights, idx):
    algo = parms['algo']
    if algo == 'wts':
        wtr_cutoff = parms['wtr_cutoff']
        floor = data[wtr_cutoff] if wtr_cutoff < len(data) else 0
        return [{
            'type': 'line',
            'xref': xaxis,
            'yref': yaxis,
            'x0': floor,
            'x1': floor,
            'y0': 0,
            'y1': y_max,
            'line': {
                'color': 'rgba(0,100,250, 0.8)',
                'width': 3,
            },
            }]
    elif algo == 'wffl':
        floor = agg_model.floor_vals(weights)[idx]
        return [{
            'type': 'line',
            'xref': xaxis,
            'yref': yaxis,
            'x0': floor,
            'x1': floor,
            'y0': 0,
            'y1': y_max,
            'line': {
                'color': 'rgba(0,100,250, 0.8)',
                'width': 2,
            }
        }, {
            'type': 'line',
            'xref': xaxis,
            'yref': yaxis,
            'x0': floor,
            'x1': 1.0,
            'y0': 1.0,
            'y1': y_max,
            'line': {
                'color': 'rgba(150,150,150, 0.8)',
                'width': 1,
            },
        }]
    elif algo == 'wfsig':
        mid = agg_model.mid_vals(weights)[idx]
        width = agg_model.width_vals(weights)[idx]

        xvals = []
        x = 0
        while x < 1.01:
            xvals.append(x)
            # This determines the x resolution of the sigmoid line.
            # Most of the change occurs near the mid value of the sigmoid,
            # so we increase the resolution around it.
            if abs(x - mid) < 0.1:
                x += 0.005
            else:
                x += 0.025

        # This is a bit gross.  The graph has a log10 y-scale, but we want this
        # data to appear as linear on it.
        from dtk.num import sigma
        yvals = sigma((xvals - mid) / width)
        yvals = 10**(yvals * np.log10(y_max))

        sig_lines = [{
            'type': 'line',
            'xref': xaxis,
            'yref': yaxis,
            'x0': xvals[i],
            'x1': xvals[i+1],
            'y0': yvals[i],
            'y1': yvals[i+1],
            'line': {
                'color': f'150,150,150, 0.8)',
                'width': 1,
            }} for i in range(len(xvals)-1)
            ]

        return [{
            'type': 'line',
            'xref': xaxis,
            'yref': yaxis,
            'x0': mid,
            'x1': mid,
            'y0': 0,
            'y1': y_max,
            'line': {
                'color': 'rgba(0,100,250, 0.8)',
                'width': 2,
            }
        }] + sig_lines
    else:
        return []


# XXX there probably is a way to refactor some of this as I just copied it straight from the plot_combined below
def get_importances(fm, parms, weights):
    N = len(fm['feature_names'])
    train_wsa_ids = parms['train_wsa_ids']
    agg_model = make_agg_model(parms, auto_drug_set=train_wsa_ids, fm=fm)
    post_norm = agg_model.post_norm(weights)
    scored = agg_model.score(post_norm, weights)
    score_weights = agg_model.score_weights(weights)

# see comment below in plot_combined
    ids = fm.sample_keys
    top_K = min(1000, len(ids))
    top_mol_idxs = np.argpartition(scored, -top_K)[-top_K:]

    importances = []
    for i in range(N):
        full_norm_y_use = post_norm[:,i]
        # Now add the weight bar and importance bar
        score_weight = score_weights[i]
        top_score_mean = np.mean(full_norm_y_use[top_mol_idxs])
        importance = top_score_mean * score_weight
        importances.append(importance)
    importance_max = max(importances)
    return dict(zip(fm['feature_names'], [i/importance_max for i in importances]))

def plot_combined(fm, parms, weights):
    from dtk.plot import PlotlyPlot, Color
    import operator
    import numpy
    plots =[]
    plots2=[]
    names = []
    total_counts = []
    nonzero_counts =[]
    import numpy as np
    import math
    for i, name in enumerate(fm['feature_names']):
        y_nan = list(np.asarray(fm['data'][:,i].todense()).reshape(-1))
        nonzero_counts.append(len([x for x in y_nan if (math.isnan(x) == False) and x!=0]))
    layout = None
    N = len(fm['feature_names'])
    shapes = []

    train_wsa_ids = parms['train_wsa_ids']
    agg_model = make_agg_model(parms, auto_drug_set=train_wsa_ids, fm=fm)
    post_norm = agg_model.post_norm(weights)
    scored = agg_model.score(post_norm, weights)

    # We're using the top K scores overall to compute importance
    # Rather than doing a full sort here, argpartition is an argsort-like
    # method that ensure the K'th element is in the correct position,
    # everything before it is smaller, and everything after is larger.
    ids = fm.sample_keys
    top_K = min(1000, len(ids))
    top_mol_idxs = np.argpartition(scored, -top_K)[-top_K:]

    y_max = fm.data.shape[0]
    score_weights = agg_model.score_weights(weights)

    kt_idxs = []
    for i, id in enumerate(ids):
        if id in train_wsa_ids:
            kt_idxs.append(i)


    importances = []
    for i in range(N):
        full_norm_y_use = post_norm[:,i]
        # Now add the weight bar and importance bar
        score_weight = score_weights[i]
        top_score_mean = np.mean(full_norm_y_use[top_mol_idxs])
        importance = top_score_mean * score_weight
        importances.append(importance)

    importance_max = np.max(importances)
    importance_argsort = np.argsort(importances)

    # We can't display 0 on a log scale plot.  This is important for both scaling the axis, and for displaying the bars.
    # (Plotly has some weird fill rendering bugs if you try to draw 0 value bars in log mode)
    # So instead, everywhere we have a 0 that would get log'd, we replace it with ZERO_EQUIV.
    # Since our underlying data is counts, anything below 1 is non-real and distinguishable from real data.
    ZERO_EQUIV = 0.9

    for idx, i in enumerate(importance_argsort):
        name = fm['feature_names'][i]
        y_nan = np.asarray(fm['data'][:,i].todense()).reshape(-1)
        y_nan0 = np.nan_to_num(y_nan)
        ordering = [(x, y) for x, y in zip(ids, y_nan0)]
        ordering.sort(key=lambda x:-x[1])
        order_ids = [x[0] for x in ordering]
        znorm = get_prenorm_for_ordering(ordering,parms)
        y_use = [znorm.get(i) for i in order_ids]
        names.append(name)

        full_norm_y_use = post_norm[:,i]

        kt_scores = [znorm.get(kt_id) for kt_id in train_wsa_ids]
        trace_names = ['prenorm', 'postnorm', 'KT-prenorm']

        from dtk.plot import bar_histogram_overlay, PlotlyPlot
        traces, layout = bar_histogram_overlay(
                x_data=[y_use, full_norm_y_use, kt_scores],
                names=trace_names,
                density=False,
                x_range=(0, 1),
                bins=50,
                show_mean=False,
                )
        yaxis = f'y{idx+1}'
        xaxis = 'x1'
        for trace, trace_name in zip(traces, trace_names):
            # Replace any 0's in the plot with small values, to prevent rendering bugs in log scale.
            for j in range(len(trace['y'])):
                if trace['y'][j] == 0:
                    trace['y'][j] = ZERO_EQUIV
            trace['yaxis'] = yaxis
            trace['xaxis'] = xaxis
            trace['legendgroup'] = trace_name
            if idx != 0:
                trace['showlegend'] = False
        layout = layout
        plots.extend(traces)

        shapes.extend(shapes_for_algo(
            parms,
            y_use,
            xaxis,
            yaxis,
            y_max,
            agg_model,
            weights,
            i,
            ))

        # Now add the weight bar and importance bar
        score_weight = score_weights[i]
        importance = importances[i] / importance_max
        nonzero_frac = nonzero_counts[i] / len(ids)


        bar_xs = [nonzero_frac, score_weight, importance,]
        bar_ys = ['Nonzero', 'Weight', 'Importance', 'Nonzero',]
        colors = [Color.highlight2, Color.default, Color.highlight,]


        from dtk.plot import Color
        datas = [{
            'type': 'bar',
            'x': [x],
            'y': [name],
            'name': name,
            'marker': {
                'color': color,
                },
            'orientation': 'h',
            'xaxis': 'x2',
            'yaxis': f'y{N+idx+1}',
            'legendgroup': name,
            'showlegend': idx == 0,
            } for x, name, color in zip(bar_xs, bar_ys, colors)]
        plots.extend(datas)

    # Why is the second column reverse-indexed?
    # Unclear.  It seems to require that to work.
    subplots = [[f'x1y{i+1}', f'x2y{N + N - i}'] for i in range(N)]


    layout.update(dict(
        grid={'rows': N, 'columns': 2, 'subplots': subplots},
        title='Scores',
        showlegend=True,
        height=40*N,
        width=1200,
        shapes=shapes,
    ))
    layout['xaxis']['domain'] = [0, 0.5]
    layout['xaxis2'] = {
            }
    annotations = []
    for i in range(N):
        anno = {
            'text': names[i],
            'xref': 'paper',
            'yref': f'y{i+1}',
            'x': 0,
            'y': 1,
            'showarrow': False,
            'xanchor': 'right',
            'yanchor': 'bottom',
            'textangle': 30,

        }
        annotations.append(anno)
        layout[f'yaxis{i+1}'] = {
                'domain': [i/N, (i+0.9)/N],
                'type': 'log',
                'showticklabels': False,
                # Tick labels for log plots are specified in log space.
                'range': [np.log10(ZERO_EQUIV), np.log10(y_max)],
                }
        layout[f'yaxis{i+N+1}'] = {
                'showticklabels': False,
                }
    layout['annotations'] = annotations
    layout['margin'] = {'l': 300}
    pp1 = PlotlyPlot(plots, layout)
    return pp1

def score_weight_directed_mutate(weights, sigma, freq, rng, w_min, w_max, l2_reg, score_mat, kt_idxs):
    NUM_WEIGHTS = len(weights)
    NUM_DRUGS = score_mat.shape[0]
    NUM_REF_DRUGS = 10

    mut_count = max(1, int(NUM_WEIGHTS * freq))

    new_weights = list(weights)

    # Randomly pick a set of weights to mutate, and for each one...
    weight_idxs = rng.choice(range(NUM_WEIGHTS), mut_count, replace=False)
    w_mid = (w_max + w_min) / 2
    for weight_idx in weight_idxs:
        delta = abs(rng.normal(0, sigma))
        # Use True to disable the other branch for now, the l2 reg is handled
        # by grad descent typically.
        # This version hasn't been updated to handle importance regularization.
        if True or rng.rand() < 0.9 or l2_reg == 0:
            # Randomly pick a KT
            for _ in range(3):
                # Bias towards non-zero KTs (but don't eliminate them
                # entirely)
                kt_idx = rng.choice(kt_idxs, 1)
                kt_score = score_mat[kt_idx,weight_idx]
                if kt_score != 0:
                    break

            # Randomly pick several other drugs
            other_idxs = rng.choice(range(NUM_DRUGS), NUM_REF_DRUGS, replace=False)
            others_score = np.mean(score_mat[other_idxs,weight_idx])
            # If KT higher in this score, shift weight higher, and vice-versa.
            if kt_score > others_score:
                new_weights[weight_idx] += delta
            else:
                new_weights[weight_idx] -= delta
        else:
            # Move towards mid weight.
            if new_weights[weight_idx] < w_mid:
                new_weights[weight_idx] += delta
            else:
                new_weights[weight_idx] -= delta

    return np.clip(new_weights, w_min, w_max)

def score_weight_mutate(weights,sigma,freq,rng,w_min,w_max):
    func = lambda v: v+(0 if rng.rand()>freq else rng.normal(0,sigma))
    out = [func(v) for v in weights]
    return np.clip(out, w_min, w_max)

class AggModel:
    """
    raw_score_mat: Whatever the CMs output for each molecule.
               Score1   Score2  ... ScoreN
    Drug1       1.2     3.2     ...  NaN
    Drug2       0.01    -1.2    ...  100.2

    pre_norm_mat: Unparameterized normalization; usually min-max -> 0-1.
               Score1   Score2  ... ScoreN
    Drug1       0.2     1.0     ...  0.0
    Drug2       0.1     0.0     ...  1.0

    """
    weight_group_labels = ['score_weights']
    def __init__(self, model_parms, parms, num_features, kt_idxs, n_kts, raw_score_mat, constraints):
        self.parms = parms
        self.model_parms = model_parms

        for k, v in self.model_parms.items():
            setattr(self, k, v)

        self.num_features = num_features
        self.kt_idxs = kt_idxs
        self.n_kts = n_kts
        self.raw_score_mat = raw_score_mat
        self.wsas = [str(x) for x in self.parms['all_wsa_ids']]
        self.pre_norm_mat = self.pre_norm(raw_score_mat)
        self._latest_post_norm = None
        self.constraints = constraints
        self._cond_pre_norm_mat = None

        seed = model_parms.get('seed', 0)
        if seed == -1:
            print("Randomizing our random seed")
            seed = None
        self.rng = np.random.RandomState(seed)

        from dtk.enrichment import EnrichmentMetric
        em = EnrichmentMetric.lookup(self.model_parms.get('metric'))
        self.condensed = getattr(em, 'condensed', False)
        self.tf_eval_func = getattr(em, 'tf_eval_func', None)
        if self.tf_eval_func is not None:
            self.tf_eval_func = self.tf_eval_func()

        if 'wsa_to_group' in parms:
            # This is required for condensed optimization, but for plotting we
            # can construct without it.
            self._wsa_to_group = {str(k):int(v) for k,v in parms['wsa_to_group'].items()}

    def sgd_enabled(self):
        return (self.model_parms.get('gradient_descent', False) and
                self.tf_eval_func is not None)

    def _setup_condensed(self):
        # This list should match the matrix ordering.
        wsas = self.wsas
        assert len(wsas) == self.pre_norm_mat.shape[0]
        # We need scores for condensing.  We don't have weights yet, though.
        # We could combine them unweighted, but that prevents almost all
        # condensing from happening. (Most molecules even with same targets
        # have different indigo scores).  In practice most molecules with
        # the same targets will have the same WZS, so as an approximation
        # when doing this sgd optimization, we ignore score for condensing.
        placeholder_scores = np.zeros(len(wsas), dtype=np.float32)
        wsa_to_idx = dict((wsa, idx) for idx, wsa in enumerate(wsas))
        ordering = sorted(zip(wsas, placeholder_scores), key=lambda x: -x[1])
        from dtk.enrichment import EMInput
        kt_set = {str(x) for x in self.parms['train_wsa_ids']}
        emi = EMInput(ordering, kt_set, self._wsa_to_group)
        cemi = emi.get_condensed_emi()
        cond_idxs = []
        cond_kt_idxs = []
        for wsa, score in cemi.get_labeled_score_vector():
            #print(f'group for {wsa} is {self._wsa_to_group.get(wsa, None)}')
            if str(wsa) in kt_set:
                cond_kt_idxs.append(len(cond_idxs))
            cond_idxs.append(wsa_to_idx[wsa])

        self._cond_pre_norm_mat = np.array(self.pre_norm_mat[cond_idxs], dtype=np.float32)
        self._cond_kt_idxs = np.array(cond_kt_idxs, dtype=np.int32)

    def sgd(self, weights, output_debug=False):
        if self.condensed:
            if self._cond_pre_norm_mat is None:
                self._setup_condensed()
            pre_norm_mat = self._cond_pre_norm_mat
            self.kt_idxs = self._cond_kt_idxs
        else:
            pre_norm_mat = self.pre_norm_mat

        from dtk.rank_gradient import sigma_of_rank, train
        from functools import partial
        out = train(weights, partial(self.tf_score_func, score_mat=pre_norm_mat), self.tf_constrain_weights, iters=50, output_debug=output_debug)

        return out


    def tf_score_func(self, weights, iter_frac, score_mat):
        from dtk.rank_gradient import sigma_of_rank
        import tensorflow as tf
        weights = tf.cast(weights, tf.float32)
        score_mat = tf.cast(score_mat, tf.float32)
        score_mat = self.tf_post_norm(score_mat, weights)
        score_weights = self.score_weights(weights)

        dropout = float(self.model_parms.get('dropout', 0))
        if dropout:
            score_weights = tf.nn.dropout(x=score_weights, rate=dropout)

        scores = score_mat @ score_weights
        sigmoid_scale = 0.01
        sor = self.tf_eval_func(scores, self.kt_idxs, self.n_kts, rank_sigmoid_scale=sigmoid_scale)

        if not self.model_parms['reg_importance']:
            w_mid = (self.w_max + self.w_min) / 2
            w_range = self.w_max - self.w_min

            keep_idxs = set(range(score_mat.shape[1]))
            extra_weights = []
            for constraint in self.constraints:
                idxs = constraint.weight_idxs()
                if not idxs:
                    continue
                combo_weight = tf.reduce_sum(tf.gather(score_weights, idxs))
                extra_weights.append(combo_weight)
                keep_idxs -= set(idxs)

            pen_score_weights = tf.concat([
                    tf.gather(tf.reshape(score_weights, (-1,)), list(keep_idxs)),
                    extra_weights,
                    ], axis=0)

            pen_weights = tf.abs(pen_score_weights - w_mid) / (w_range * 0.5)
            reg_score = tf.reduce_mean(pen_weights**2)
        else:
            importances = tf.reduce_sum(score_mat, axis=0) * tf.reshape(score_weights, (-1,))
            mean_imp = tf.reduce_mean(importances)
            norm_imp = (importances - mean_imp) / mean_imp
            reg_score = tf.reduce_mean(tf.abs(norm_imp))
            # reg_score can sometimes go large, use a sigmoid so that it
            # just approaches 1 instead.
            reg_score = (tf.sigmoid(reg_score*2) - 0.5) * 2

        l2_penalty = sor * self.l2_reg * reg_score
        #print("l2 pen is ", l2_penalty, reg_score, pen_weights, pen_score_weights)
        l2_sor = sor - l2_penalty

        return (l2_sor, sor)

    def tf_constrain_weights(self, weights):
        import tensorflow as tf
        weights.assign(tf.clip_by_value(weights, clip_value_min=self.w_min, clip_value_max=self.w_max))

        for constraint in self.constraints:
            constraint.tf_apply_to(weights, self.num_features)
    def tf_post_norm(self, score_mat, weights):
        return score_mat

    def pre_norm(self, raw_score_mat):
        """Precomputable (i.e. non-fitting-dependant) normalization."""
        srcs = []
        logger.info(f"Computing norms on {raw_score_mat.shape[1]} scores")
        for i in range(raw_score_mat.shape[1]):
            keyed_scores = zip(
                    self.wsas,
                    raw_score_mat[:,i].toarray(),
                    )
            srcs.append(get_norm_for_ordering(keyed_scores,self.parms))

        score_mat = np.array([
                    [ s.get(d) for s in srcs ]
                    for d in self.wsas
                ], dtype=float)
        logger.info("Norms computed")
        return score_mat

    def post_norm(self, weights):
        """Fitted / weight-based normalization."""
        self._latest_post_norm = self.pre_norm_mat
        return self._latest_post_norm

    def score(self, norm_score_mat, weights):
        """
        Every model should come down to this - applying weights to some score
        matrix.  This concept of score weights is important for downstream and
        interpretibility.
        """
        score_weights = self.score_weights(weights)
        out = np.sum(norm_score_mat * score_weights, axis=1)
        # Sometimes we get NaNs; should probably prevent, but for now, just replace.
        out = np.nan_to_num(out)
        return out

    def score_weights(self, weights):
        #assert len(weights) >= self.num_features
        out = weights[:self.num_features]
        return out

    def initial_weights(self):
        return np.array([1.0 for _ in range(self.num_features)])

    def mutate(self, weights, anneal_mult):
        score_weights = self.score_weights(weights)
        freq, sigma = self.mut_freq, self.mut_sigma
        freq *= anneal_mult
        if self.anneal_sigma:
            sigma *= anneal_mult

        score_mat = self._latest_post_norm

        if score_mat is not None and self.directed_mutate and self.rng.rand() < self.directed_mutate:
            # Note: we typically don't end up down this codepath anymore, due
            # to score_mat not being set in the main process.  It doesn't really
            # matter, though, as the directed part of the search can be managed
            # by the sgd now.
            out = score_weight_directed_mutate(
                    score_weights, sigma, freq, self.rng,
                    self.w_min, self.w_max, self.l2_reg,
                    score_mat, self.kt_idxs)
        else:
            out = score_weight_mutate(score_weights, sigma, freq,
                    rng=self.rng,
                    w_min=self.w_min,
                    w_max=self.w_max,
                    )
        return out

class WfflAggModel(AggModel):
    """
    Similar to the AggModel, but with extra weights/parameters to search
    for a per-score floor value instead of fixing it to the top N scores.
    """
    weight_group_labels = ['score_weights', 'floors']
    def post_norm(self, weights):
        floors = self.floor_vals(weights)
        # Weights start off as 0 to 1.  Remove a floor value from each,
        # but don't let anything go negative.
        floored = np.maximum(self.pre_norm_mat - floors, 0)
        # Rescale the remaining positive values back to 0 to 1.
        rescaled = floored / (1 - floors)
        self._latest_post_norm = rescaled
        return rescaled

    def tf_constrain_weights(self, weights):
        import tensorflow as tf
        weights.assign(tf.concat([
                tf.clip_by_value(self.score_weights(weights), clip_value_min=self.w_min, clip_value_max=self.w_max),
                tf.clip_by_value(self.floor_vals(weights), clip_value_min=EPS, clip_value_max=1-EPS),
                ], axis=0
                ))
        for constraint in self.constraints:
            constraint.tf_apply_to(weights, self.num_features * 2)

    def tf_post_norm(self, score_mat, weights):
        import tensorflow as tf
        floors = tf.reshape(self.floor_vals(weights), (1, -1))
        floored = tf.maximum(score_mat - floors, 0)
        rescaled = floored / (1 - floors)
        return rescaled

    def floor_vals(self, weights):
        return weights[self.num_features:]

    def initial_weights(self):
        return np.array(
                [0.5] * self.num_features +
                [0.01] * self.num_features
                )

    def mutate_floors(self, floor_vals, anneal_mult):
        mask_rand = self.rng.rand(self.num_features)
        mask = mask_rand<anneal_mult
        delta = self.rng.normal(0, 0.1, self.num_features) * anneal_mult
        return np.clip(floor_vals + delta * mask, EPS, 1 - EPS)

    def mutate(self, weights, anneal_mult):
        N = self.num_features
        out = np.empty(len(weights))

        self.score_weights(out)[:] = super().mutate(self.score_weights(weights), anneal_mult)
        self.floor_vals(out)[:] = self.mutate_floors(self.floor_vals(weights), anneal_mult)

        return out


class WfsigAggModel(AggModel):
    """
    The model here is that for each score, somewhere between 0 and 1
    there is a transition from no signal to signal.
    'mid' searches for where that transition occurs, and 'width' searches
    for how gradually that transition occurs.
    Alternatively, you can think of this as a smoothed version of the
    'floor' model that also has a 'ceiling'.
    """
    weight_group_labels = ['score_weights', 'sig_middles', 'sig_widths']
    def post_norm(self, weights):
        mids = self.mid_vals(weights)
        widths = self.width_vals(weights)
        from dtk.num import sigma
        renormalized = sigma((self.pre_norm_mat - mids) / widths )
        self._latest_post_norm = renormalized
        return renormalized

    def tf_constrain_weights(self, weights):
        """This should be kept in-sync with np_constrain_weights."""
        # I suspect that rather than clipping we'd be better off representing
        # these as unbounded values and use sigmoids on top of them.
        # We'd need to convert into this space on start of SGD and convert
        # back out on returning the value.
        import tensorflow as tf
        clipped_widths = tf.clip_by_value(
            self.width_vals(weights),
            clip_value_min=MIN_WFSIG_WIDTH,
            clip_value_max=MAX_WFSIG_WIDTH,
            )
        weights.assign(tf.concat([
                tf.clip_by_value(self.score_weights(weights), clip_value_min=self.w_min, clip_value_max=self.w_max),
                # Note that we're setting the minimum mid value to the width.
                # This prevents it from setting the mid to near 0 and giving score to everything.
                tf.clip_by_value(self.mid_vals(weights), clip_value_min=clipped_widths, clip_value_max=1-EPS),
                clipped_widths,
                ], axis=0
                ))
        for constraint in self.constraints:
            constraint.tf_apply_to(weights, self.num_features * 3)
    
    def np_constrain_weights(self, weights):
        """This should be kept in-sync with tf_constrain_weights."""
        clipped_weights = np.clip(
            self.score_weights(weights),
            a_min=self.w_min,
            a_max=self.w_max,
            )
        clipped_widths = np.clip(
            self.width_vals(weights),
            a_min=MIN_WFSIG_WIDTH,
            a_max=MAX_WFSIG_WIDTH,
            )
        clipped_mids = np.clip(
            self.mid_vals(weights),
            a_min=clipped_widths,
            a_max=1-EPS,
            )

        self.score_weights(weights)[:] = clipped_weights 
        self.width_vals(weights)[:] = clipped_widths
        self.mid_vals(weights)[:] = clipped_mids

        # The custom constraint (WeightConstraint) get applied later
        # in the np codepath.

    def tf_post_norm(self, score_mat, weights):
        import tensorflow as tf
        mids = tf.reshape(self.mid_vals(weights), (1, -1))
        widths = tf.reshape(self.width_vals(weights), (1, -1))
        renormalized = tf.sigmoid((score_mat - mids) / widths)
        return renormalized

    def mid_vals(self, weights):
        N = self.num_features
        return weights[N:N*2]

    def width_vals(self, weights):
        N = self.num_features
        return weights[N*2:N*3]

    def initial_weights(self):
        return np.array(
                [0.5] * self.num_features +
                [0.5] * self.num_features +
                [0.05] * self.num_features
                )

    def mutate_widths(self, width_vals, anneal_mult):
        mask_rand = self.rng.rand(self.num_features)
        mask = mask_rand<anneal_mult
        delta = self.rng.normal(0, 0.01, self.num_features) * anneal_mult
        return width_vals + delta * mask

    def mutate_mids(self, mid_vals, anneal_mult):
        mask_rand = self.rng.rand(self.num_features)
        mask = mask_rand<anneal_mult
        delta = self.rng.normal(0, 0.1, self.num_features) * anneal_mult
        return mid_vals + delta * mask

    def mutate(self, weights, anneal_mult):
        N = self.num_features
        out = np.empty(len(weights))

        self.score_weights(out)[:] = super().mutate(weights, anneal_mult)
        self.mid_vals(out)[:] = self.mutate_mids(self.mid_vals(weights), anneal_mult)
        self.width_vals(out)[:] = self.mutate_widths(self.width_vals(weights), anneal_mult)

        self.np_constrain_weights(out)
        return out

def write_score(drug_ids, raw_score, fn):
    # offset everything to non-neg to make it DEA-friendly
    floor = min(raw_score)
    with open(fn,'w') as f:
        f.write('\t'.join(['wsa','wzs'])+'\n')
        for drug,x in zip(drug_ids,raw_score):
            v = x - floor
            f.write('\t'.join((str(drug),str(v)))+'\n')


def make_agg_model(parms, auto_drug_set=None, fm=None, raw_score_mat=None, feature_names=None, constraints=None):
    """Can specify an fm or (raw_score_mat + feature_names)."""

    if raw_score_mat is None:
        raw_score_mat = fm.data
    if feature_names is None:
        feature_names = fm.feature_names
    if auto_drug_set is None:
        # This isn't always present in params.
        auto_drug_set = parms.get('train_wsa_ids', [])
    if fm is None:
        all_wsa_ids = parms['all_wsa_ids']
    else:
        all_wsa_ids = fm.sample_keys
        parms['all_wsa_ids'] = all_wsa_ids
    kt_idxs = [i for i, wsaid in enumerate(all_wsa_ids)
               if wsaid in auto_drug_set]
    # This could be different from len(kt_idxs) if we're missing some.
    n_kts = len(auto_drug_set)
    model_parms = {k[len('auto_'):]:v for k,v in parms.items()
                   if k.startswith('auto_')}

    if parms['algo'] == 'wffl':
        ModelClass = WfflAggModel
    elif parms['algo'] == 'wfsig':
        ModelClass = WfsigAggModel
    else:
        ModelClass = AggModel
    agg_model = ModelClass(
            model_parms=model_parms,
            parms=parms,
            raw_score_mat=raw_score_mat,
            num_features=len(feature_names),
            kt_idxs=kt_idxs,
            n_kts=n_kts,
            constraints=constraints,
            )
    return agg_model


class AutoTuneRunner:
    def __init__(self, parms, raw_score_mat, feature_names, out_dir, plot_dir):
        make_directory(out_dir)
        make_directory(plot_dir)
        self.parms = parms
        self.feature_names = feature_names
        self.all_wsa_ids = parms['all_wsa_ids']
        # Note: JSON converts all the int keys into strings, convert it back.
        self._wsa_to_group = {int(k):int(v) for k,v in parms['wsa_to_group'].items()}
        self.score_labels = feature_names

        self.fn_score=os.path.join(out_dir, 'wz_score.tsv')
        self.wts_file=os.path.join(out_dir, 'weights.tsv')
        self.details_file=os.path.join(out_dir, 'details.json')
        self.final_wt_plot = os.path.join(plot_dir, "final_wts.plotly")
        self.atl_plot = os.path.join(plot_dir, "atl.plotly")
        self.hm_time = os.path.join(plot_dir, "hm_time.plotly")
        self.hm_init = os.path.join(plot_dir, "hm_init.plotly")
        self.hm_1qr = os.path.join(plot_dir, "hm_1qr.plotly")
        self.hm_mid = os.path.join(plot_dir, "hm_mid.plotly")
        self.hm_3qr = os.path.join(plot_dir, "hm_3qr.plotly")
        self.hm_final = os.path.join(plot_dir, "hm_final.plotly")
        self.wt_sds = os.path.join(plot_dir, "wt_sds.plotly")

        self.auto_drug_set = self.parms['train_wsa_ids']
        self.agg_model = make_agg_model(
                parms,
                raw_score_mat=raw_score_mat,
                feature_names=feature_names,
                constraints=self._make_constraints(),
                )
        
        assert len(self.agg_model.kt_idxs) > 0, "No KTs found; have you selected the right optimization target?"


    def evaluate(self,score):
        from dtk.enrichment import EMInput
        em = self.auto_metric()
        # Don't need to use sorted_ord here, we don't generate NaN scores,
        # our normalizers take care of that.
        ordering = sorted(zip(self.all_wsa_ids,score), key=lambda x:x[1], reverse=True)
        emi = EMInput(ordering,self.auto_drug_set,self._wsa_to_group)
        from dtk.files import Quiet
        em.evaluate(emi)
        return em.rating

    def _make_constraints(self):
        if not self.parms['auto_constraints'].strip():
            return []
        from dtk.score_source import ScoreSource
        import json
        constraint_groups = json.loads(self.parms['auto_constraints'])
        constraint_map = {}
        for i, constraint_group in enumerate(constraint_groups):
            for constraint_name in constraint_group:
                constraint_map[constraint_name.lower()] = i

        constraint_weight_idxs = [[] for _ in range(len(constraint_groups))]

        for i, name in enumerate(self.feature_names):
            lname = name.lower()
            for constraint_name, constraint_idx in constraint_map.items():
                if constraint_name in lname:
                    constraint_weight_idxs[constraint_idx].append(i)

        constraints = []
        for i, constraint_weight_idx_list in enumerate(constraint_weight_idxs):
            print("Found score idxs %s matching %s" % (
                constraint_weight_idx_list, constraint_groups[i]))

            if len(constraint_weight_idx_list) < 2:
                # With less than 2 scores, this would do nothing.
                continue
            constraint = WeightConstraint(constraint_weight_idx_list,
                                            self.parms['auto_w_max'])
            constraints.append(constraint)

        return constraints

    def auto_tune(self,AutoTunerClass,cores):
        constraints = self._make_constraints()

        from dtk.enrichment import EnrichmentMetric
        self.auto_metric = EnrichmentMetric.lookup(
                    self.parms['auto_metric']
                    )

        AUTO_PREFIX = 'auto_'
        max_loops = self.parms['auto_iter']
        for suffix in ['iter','drug_set','metric']:
            key = AUTO_PREFIX + suffix
            print(key,self.parms[key])
        extra_parms = {}
        for suffix in AutoTunerClass.extra_parms:
            key = AUTO_PREFIX + suffix
            print(key,self.parms[key])
            extra_parms[suffix] = self.parms[key]

        if False:
            # This is some proof-of-concept code for running scipy optimizers
            # on this problem.
            # I plan to revive this later for fine-tuning.
            best = [0, 0]
            def opt_func(x):
                score = self.get_combo_score(x)
                metric = self.evaluate(score)
                best[0] = max(best[0], metric)
                best[1] += 1
                if best[1] % 100 == 0:
                    print("Metric %.3f (%.2f)" % (metric, best[0]))
                return -metric
            import scipy.optimize as opt
            print("Running optimizer")
            N = len(self.feature_names)
            bounds = [(0, 10) for _ in range(N)] + \
                    [(0,1) for _ in range(N)] + [(0,1)]
                    #[(0,1) for _ in range(N)]
            #initial_weights = opt.differential_evolution(opt_func, bounds, disp=True)
            #initial_weights = opt.basinhopping(opt_func, initial_weights, stepsize=1.0, minimizer_kwargs=kwargs, disp=True)
            def cb(*args, **kwargs):
                print("CB", args, kwargs)
            kwargs = {'method': 'trust-constr', 'callback': cb}

            initial_weights = opt.dual_annealing(opt_func, bounds=bounds, callback=cb, no_local_search=False, maxfun=10000, local_search_options=kwargs)

            print("Got initial weights", initial_weights)

        tuner = AutoTunerClass(
                self.agg_model,
                self.evaluate,
                constraints,
                extra_parms,
                cores=cores,
                )
        self.tuner = tuner
        self.for_hm1=None
# I do it in this order to so if there are <4 loops the least important plots are written over
        plot_iters = {3*max_loops/4 : self.hm_3qr}
        plot_iters[max_loops/4] = self.hm_1qr
        plot_iters[max_loops/2] = self.hm_mid
        self.for_line_plot = []

        # Scale so that we're not plotting a crazy # of sgd debug plots.
        # However, we only run it every SGD_PER_ITEMS, so it has to be a multiple
        # of that.
        sgd_plot_iters = SGD_PER_ITERS
        while max_loops // sgd_plot_iters > 3:
            sgd_plot_iters += SGD_PER_ITERS

        for i in range(max_loops):
            if i % sgd_plot_iters == 0 and hasattr(tuner, '_heatmap_sgd'):
                dirname = os.path.dirname(self.hm_mid)
                tuner._heatmap_sgd(dirname, self.score_labels)

            done = tuner.run_iteration()
            self.for_line_plot.append(tuner.stats)
            if self.for_hm1 is None:
                if self.parms['auto_tune'] == 'genetic':
                    self.for_hm1 = [[w] for w in self.agg_model.score_weights(tuner.best.weights)]
                    tuner._heatmap_population(self.hm_init,
                                              self.score_labels,
                                              title_prefix = 'Initial'
                                             )
                elif self.parms['auto_tune'] == 'gridwalk':
                    self.for_hm1 = [[w] for w in
                                    tuner._get_current_mod_wts()
                                   ]
                else:
                    assert False
            else:
# took out b/c Gridwalk didn't have weight and it was never catchign anything
#               assert len(self.for_hm1) == len(tuner.best.weights)
                if self.parms['auto_tune'] == 'genetic':
                    for j,w in enumerate(self.agg_model.score_weights(tuner.best.weights)):
                        self.for_hm1[j].append(w)
                    if i in plot_iters:
                        tuner._heatmap_population(plot_iters[i],
                                                  self.score_labels,
                                                  title_prefix = str(i)
                                                  )
                elif self.parms['auto_tune'] == 'gridwalk':
                    wts =tuner._get_current_mod_wts()
                    for j,w in enumerate(wts):
                        self.for_hm1[j].append(w)
                else:
                    assert False
            
            if done:
                break
        logger.info("Adjust weights completed %d loops", tuner.cycle)
        plot_scores = {}
        with open(self.wts_file, 'w') as f:
            tuner.output_results(
                    plot_scores,self.score_labels,
                    f,self.score_labels,
                    )
        with open(self.details_file, 'w') as f:
            self.output_details(tuner, f)
        self._plot(plot_scores)
        if self.parms['auto_tune'] == 'genetic':
            tuner._heatmap_population(self.hm_final, self.score_labels, title_prefix = 'Final')
            self._plot_wt_sds(tuner.wt_sds)
        write_score(self.all_wsa_ids, tuner.best.score, self.fn_score)

    def output_details(self, tuner, f):
        # Save out the full set of weights (which could include e.g. fitted
        # floor values) as well as the full set of normed scores.
        # While you could recompute the normed scores given the weights and
        # parameters, it is easier if you don't have to.
        normed = self.agg_model.post_norm(tuner.best.weights)
        details = {
                    'normed': normed.tolist(),
                    'all_weights': tuner.best.weights.tolist(),
                }
        import json
        f.write(json.dumps(details))

    def _plot_wt_sds(self, sds):
        from dtk.plot import Color,PlotlyPlot
        maxs = []
        meds = []
        mins = []
        for l in sds:
            mins.append(l[0])
            meds.append(l[1])
            maxs.append(l[2])
        traces = [
                  dict(
                   x = list(range(len(maxs))),
                   y = maxs,
                   name = 'Maximum',
                   line = dict(
                       color = Color.highlight
                   )
                  )]
        if mins:
            traces.append(dict(
                       x = list(range(len(mins))),
                       y = mins,
                       name = 'Minimum',
                       line = dict(
                           color = Color.default,
                           dash = 'dash',
                       )
                      ))
            traces.append(dict(
                       x = list(range(len(meds))),
                       y = meds,
                       name = 'Average',
                       line = dict(
                           color = Color.default
                       )
                      ))
        pp = PlotlyPlot(traces,
                   dict(
                       title='Weight StDevs across iterations',
                       yaxis=dict(title='Standard Deviation'),
                       xaxis=dict(title='Number of iterations'),
                       height=600,
                       width=600
                   )
             )
        pp.save(self.wt_sds)
    def _plot(self, scores):
        from dtk.plot import PlotlyPlot, Color, plotly_heatmap
        import operator
        sorted_scores = sorted(list(scores.items()),
                               key=operator.itemgetter(1)
                              )
        xs = []
        names = []
        max_name_len=0
        last_name_len=0
        for tup in sorted_scores:
            xs.append(tup[1])
            names.append(tup[0].split(' (')[0])
            last_name_len=len(tup[0])
            max_name_len = max([max_name_len,
                                last_name_len
                               ])
        pp = PlotlyPlot([dict(
                       type = 'bar',
                       x = xs,
                       y = names,
                       orientation = 'h',
                   )],
                   dict(
                       title='Final weights',
                       yaxis=dict(tickangle=-30),
                       xaxis=dict(title='Relative weight'),
                       height=300+ len(names)*30,
                       width=400 + max_name_len*7,
                       margin=dict(
                              l=max_name_len*8,
                              r=30,
                              b=max_name_len*6,
                              t=30,
                              pad=4
                              )

                   )
             )
        pp.save(self.final_wt_plot)
        if not self.for_line_plot:
            return
        mins = []
        avgs = []
        maxs = []
        mins2 = []
        avgs2 = []
        maxs2 = []
        for l in self.for_line_plot:
            maxs.append(l[0])
            if len(l) > 1:
                avgs.append(l[1])
                mins.append(l[2])
                maxs2.append(l[3])
                avgs2.append(l[4])
                mins2.append(l[5])
        traces = [
                  dict(
                   x = list(range(len(maxs))),
                   y = maxs,
                   name = 'Maximum',
                   line = dict(
                       color = Color.highlight
                   )
                  )]
        if mins:
            traces.append(dict(
                       x = list(range(len(maxs2))),
                       y = maxs2,
                       name = 'Maximum (no L2)',
                       line = dict(
                           color = Color.highlight_light
                       )
                      ))
            traces.append(dict(
                       x = list(range(len(mins))),
                       y = mins,
                       name = 'Minimum',
                       line = dict(
                           color = Color.default,
                           dash = 'dash',
                       )
                      ))
            traces.append(dict(
                       x = list(range(len(mins2))),
                       y = mins2,
                       name = 'Minimum (no L2)',
                       line = dict(
                           color = Color.default_light,
                           dash = 'dash',
                       )
                      ))
            traces.append(dict(
                       x = list(range(len(avgs))),
                       y = avgs,
                       name = 'Average',
                       line = dict(
                           color = Color.default
                       )
                      ))
            traces.append(dict(
                       x = list(range(len(avgs2))),
                       y = avgs2,
                       name = 'Average (no L2)',
                       line = dict(
                           color = Color.default_light
                       )
                      ))
        pp = PlotlyPlot(traces,
                   dict(
                       title='Auto-tuned results',
                       yaxis=dict(title='Enrichment score'),
                       xaxis=dict(title='Number of iterations'),
                       height=600,
                       width=600
                   )
             )
        pp.save(self.atl_plot)
        import numpy as np
        pp = plotly_heatmap(np.array(self.for_hm1),
                            self.score_labels,
                            col_labels=['Iter.' + str(i) + ': %.3f'%(v)
                                        for i,v in enumerate(maxs)
                                       ],
                            color_zero_centered = True,
                            Title = 'Best weights across all iterations'
                           )
        pp.save(self.hm_time)



################################################################################
# Utilities
################################################################################
def get_prenorm_for_ordering(ordering,parms):
    '''Return a norming object wrapping 'ordering', based on norm_choice.

    Note that score calibration norming is not handled here, because it
    needs bji objects, and those aren't available on the worker, where
    this function must run. So, that norming is pre-applied before shipping
    the FM to the worker, and norm_choice is switched to 'none' by the
    time this function sees it.
    '''
    from dtk.scores import ZNorm, MMNorm, NoNorm, LMMNorm
    if parms['norm_choice'] == 'z':
        norm = ZNorm(ordering)
    elif  parms['norm_choice'] == 'mm':
        norm = MMNorm(ordering)
    elif  parms['norm_choice'] == 'lmm':
        norm = LMMNorm(ordering)
    elif  parms['norm_choice'] == 'none':
        norm = NoNorm(ordering)
    else:
        assert False
    return norm

def get_norm_for_ordering(ordering,parms):
    '''Return a norming object wrapping 'ordering'.

    This will be based both on norm_choice and algo.
    '''
    norm = get_prenorm_for_ordering(ordering,parms)
    # apply optional wrapper
    if parms['algo'] == 'wtr':
        norm = TopN(norm,parms['wtr_cutoff'])
    elif parms['algo'] == 'wts':
        norm = FloorCorrected(norm,parms['wtr_cutoff'])
    else:
        norm = norm
    return norm

def get_per_feature_norms(fm,parms,details_file):
    '''Return a list of norming objects wrapping the features of fm.

    This is used after a WZS run is complete to introspect results,
    so it may use the details file produced during the run.
    '''
    # XXX currently this is only used to implement
    # XXX get_score_weights_and_sources() in run_wzs.py, but seems
    # XXX to live here to share get_sorted_ord()
    srcs = []
    algo = parms['algo']
    if algo == 'wffl' or algo == 'wfsig':
        # These save the normed scores directly, just load them up.
        with open(details_file, 'r') as f:
            import json
            details = json.loads(f.read())
        norms = np.array(details['normed'])

        for i,n in enumerate(fm.feature_names):
            keyed_normed_scores = zip(
                    fm.sample_keys,
                    norms[:, i]
                    )
            srcs.append(dict(keyed_normed_scores))
    else:
        if parms['norm_choice'] in ['sc','scl']:
            # in the score calibration case, follow the same pattern
            # as during the WZS run itself -- update the matrix in place,
            # and then use 'none' norming in the remainder of the path
            from dtk.score_calibration import FMCalibrator
            fmc=FMCalibrator(fm=fm)
            fmc.calibrate(logscale=bool(parms['norm_choice']=='scl'))
            parms = dict(parms)
            parms['norm_choice'] = 'none'
        for i,n in enumerate(fm.feature_names):
            keyed_scores = zip(
                    fm.sample_keys,
                    fm.data[:,i].toarray(),
                    )
            keyed_scores = get_sorted_ord(keyed_scores)
            logger.info("Norm'ing %s", n)
            srcs.append(get_norm_for_ordering(keyed_scores,parms))

    return srcs

class TopN:
    def __init__(self,znorm,n):
        in_ord = get_sorted_ord(list(znorm.scores.items()))
        from dtk.scores import get_ranked_groups
        self.out = {}
        for ahead,tied in get_ranked_groups(in_ord):
            if ahead >= n:
                logging.debug('with n = %s, scores included: %s', n, ahead)
                break
            for label in tied:
                self.out[label] = max([0.,(n-ahead-len(tied)/2.)/n])
    def get(self,label):
        if label in self.out:
            return self.out[label]
        return 0

def get_sorted_ord(in_ord, reverse=True):
    from math import isnan
    if not isinstance(in_ord, list):
        in_ord = list(in_ord)
    in_ord.sort(key=lambda x: float('-inf') if isnan(x[1]) else x[1], reverse=reverse)
    return in_ord

class FloorCorrected:
    def __init__(self,znorm,n):
        from math import isnan
        in_ord = get_sorted_ord(list(znorm.scores.items()))
        self.out = {}
        if not in_ord:
            return
        floor = znorm.get([x[0] for x in in_ord[:n]][-1])
        logging.debug('floor at %s: %s', n, floor)
        for label,val in in_ord:
            val = znorm.get(label)
            if isnan(val) or val <= floor:
                break
            self.out[label] = val-floor
    def get(self,label, def_value=0):
        return self.out.get(label, def_value)


class WeightConstraint(object):
    """Combine multiple weights into a single weight, for constraints.
    This allows us to prevent highly correlated scores from being overweighted.
    e.g. we apply this to defus-related scores.

    To do this, we check the sum of their weights, and if they exceed the
    maximum, we scale them down to sum to the maximum.
    """
    def __init__(self, weight_idxs, max_weight):
        self._weight_idxs = weight_idxs
        self._max_weight = max_weight

    def apply_to(self, weights):
        weight_sum = np.sum(weights[self._weight_idxs])
        if weight_sum > self._max_weight:
            weights[self._weight_idxs] *= self._max_weight / weight_sum

    def tf_apply_to(self, weights, N):
        import tensorflow as tf
        weight_sum = tf.reduce_sum(tf.gather(weights, self._weight_idxs))

        factor = tf.maximum(1.0, tf.cast(weight_sum / self._max_weight, tf.float32))
        mask = tf.reduce_sum(tf.one_hot(self._weight_idxs, depth=N, dtype=tf.float64), axis=0)
        mask *= tf.dtypes.cast((factor - 1), tf.float64)
        mask += 1

        mask = tf.reshape(mask, weights.shape)

        weights.assign(weights / mask)

    def weight_idxs(self):
        return self._weight_idxs

    def get_combined_weight(self, weights):
        return np.sum(weights[self._weight_idxs])



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--feature-matrix", help="feature matrix file")
    parser.add_argument("--parms", help="parameters json file")
    parser.add_argument("-o", "--out", help="Directory to write outputs to")
    parser.add_argument("-p", "--plots", help="Directory to write plots to")
    parser.add_argument("-c", "--cores", type=int, help="Cores to use", default=None)
    from dtk.log_setup import addLoggingArgs, setupLogging
    addLoggingArgs(parser)
    args = parser.parse_args()
    setupLogging(args)

    import tensorflow as tf
    # These configs need to be set before tf is ever used, and ideally before we fork (to reduce
    # the amount of init spam).
    #
    # We are threading outside of this, so only use 1 thread inside tf.
    # Also, tf parallelism seems to hurt more than it helps regardless for
    # sigma_of_rank, possibly our matrices are too small, or there is too much
    # fiddly control-flow-like stuff going on (constraints & regularization).
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    run(args.feature_matrix, args.parms, args.out, args.plots, args.cores)
