#!/usr/bin/env python
from __future__ import print_function
from builtins import range
import sys
try:
    from path_helper import PathHelper,make_directory
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")
    from path_helper import PathHelper,make_directory

import os
if 'django.core' not in sys.modules:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    import django
    django.setup()

import numpy as np
from algorithms.exit_codes import ExitCoder
import time

# created 1.Sep.2016 - Aaron C Daugherty
# To do:
# clean out old code and generally refactor after verifying the settings we like

verbose = True

# adapted from: http://exploringdatablog.blogspot.sk/2012/04/david-olives-median-confidence-interval.html
def median_withCI(y, alpha = 0.05):
    import scipy.stats
    import math
    n = len(y)
    ysort = sorted(y)
    nhalf = n//2
    if (2*nhalf < n):
        #  n odd
        med = ysort[nhalf + 1]
    else:
        # n even
        med = (float(ysort[nhalf]) + float(ysort[nhalf+1]))/2.0
    #
    #  Next, compute Olive.s standard error for the median
    #
    Ln = nhalf - int(math.ceil(math.sqrt(float(n)/4.0)))
    Un = n - Ln
    # TODO: Figure out if this "min" line is correct, it is not from the paper.
    # But if n=5, then nhalf=2, ceil(sqrt(5.0/4))=2, so Ln = 0
    # If Ln is 0, Un will be n, which is out of bounds.
    # So without the line below, the function will throw on all arrays
    # len {5, 3, 2, 1}.  Note that len 4 arrays do work.
    Un = min(Un, len(ysort)-1)
    SE = 0.5*(float(ysort[Un]) - float(ysort[Ln+1]))
    #
    #  Compute the confidence interval based on Student.s t-distribution
    #  The degrees of freedom parameter p is discussed in Olive.s paper
    #
    p = Un - Ln - 1.0
    t = float(scipy.stats.t.ppf(1.0 - alpha/2.0, df = p))
    medLCI = med - t * SE
    medUCI = med + t * SE
    return med, medUCI, medLCI


class bootstrap_enrichment(object):
    def __init__(self,**kwargs):
# Ideally we'd move over to this approach, but it's not there just yet
        self.ec = ExitCoder()
        self.score_list = kwargs.get('score_list',None)
        self.set_oi = kwargs.get('set_of_interest',None)
        if not self.set_oi:
            raise ValueError('No entries in evaluation set')
        self.alpha = kwargs.get('alpha',0.01)
        self.weight = kwargs.get('weight',1.5)
        self.nPermuts = kwargs.get('nPermuts',1000)
        self.min_tie = kwargs.get('min_tie',5)
            # below here, min_tie_len is the minimum number of 'extra'
            # matching scores (e.g. a value of 2 means a tie must be
            # 3 scores in a row). above here, we convert to a more
            # intuitive value of the total number of tied scores, and
            # use a slightly different variable name to help track
            # the different meaning.
    def run(self):
        '''Set call all of the various steps appropriate for the derived class.

        Should be overridden by derived class
        '''
        raise NotImplementedError('run not overridden')
    def get_non_zeros(self):
        self.non_zero_total_len = 0
        self.non_zero_set_len = 0
        gen = (x for x in self.score_list if x[1] != 0)
        for x in gen:
            self.non_zero_total_len += 1
            if x[0] in self.set_oi:
                self.non_zero_set_len += 1
    def verify_results(self):
        if self.weight and not self.matches_wtd_score_sum:
            self._report_bad_data()
    def _report_bad_data(self):
            to_report = 'No members of the set had a non-zero score.\n'
            to_report += str(self.set_len) + ' members in the set.\n'
            to_report += str(self.non_zero_total_len) + ' members with non-zero scores.\n'
            to_report += 'exiting without generating output...\n'
            raise ValueError (to_report)
            sys.exit(self.ec.encode('unexpectedDataFormat'))
    def calc_es(self):
        self.generate_RES()
        if self.res:
            self.find_res_peak()
        else:
            self._report_bad_data()
    def generate_RES(self):
        self.setup_vectors()
        self.normalize_vectors()
        self.make_final_vector()
        if self.final_vector is None:
            self.res = None
            self.es= None
        else:
            self.res = list(np.cumsum(np.array(self.final_vector)))
    def setup_vectors(self):
        self.find_matches()
        self.set_misses = 1 - self.set_matches
        self.total_len = len(self.score_list)
        self.set_len = len(self.set_oi)
        self.set_scores()
    def find_matches(self):
        self.set_matches = np.array([1 if dl[0] in self.set_oi else 0 for dl in self.score_list])
    def set_scores(self):
        self.wtd_scores = np.array([abs(float(drug[1]))**self.weight for drug in self.score_list])
        self.score_total = float(sum(list(self.wtd_scores)))
        if self.score_total == 0:
            self._report_bad_data()
        self.wtd_scores = np.array([d/self.score_total for d in self.wtd_scores])
        self.get_misses_norm()
    def get_misses_norm(self):
        '''The normalization factor for the entries not in the set of interest vary by use case.

        Should be overridden by derived class
        '''
        raise NotImplementedError('get_misses_norm not overridden')
    def normalize_vectors(self):
        self.matches_wtd_score_sum = sum(list(np.multiply(self.set_matches, self.wtd_scores)))
        if self.matches_wtd_score_sum != 0:
            self.matches_norm_factor = 1.0/self.matches_wtd_score_sum
        else:
            print(len(list(self.wtd_scores)))
            self.matches_norm_factor = None
            return None
    def make_final_vector(self):
        if self.matches_norm_factor:
            self.process_fv()
        else:
            print("self.matches_norm_factor was not found, so final_vector not created")
            self.final_vector = None
    def process_fv(self):
        self.score_fv()
    def score_fv(self):
        self.final_vector = list(self.set_matches * self.wtd_scores * self.matches_norm_factor
                                  - self.set_misses * self.wtd_scores * self.misses_norm_factor
                                )
    def find_res_peak(self):
        if abs(min(self.res)) > max(self.res):
            self.es = min(self.res)
        else:
            self.es = max(self.res)
    def calc_nes(self):
        self.get_background()
        self.calc_enrich()
    def get_background(self):
        '''Background sampling depends on the use case.

        Should be overridden by derived class
        '''
        raise NotImplementedError('get_background not overridden')
    def calc_enrich(self):
        self.setup_for_enrich()
        self.find_nes()
        self.find_p()
    def setup_for_enrich(self):
        ## a negative value means there was never any enrichment (only depletion)
        # also, we don't want to divide by 0
        # to avoid that we'll always just shift everything the smallest amount: 2X the downstep
        # the smallest max value we can get from the bootstrap is 1x the downstep
        # for consistency we'll do it for everything
        self.es_sign = np.sign(self.es)
        self.to_avoid_zero = 2.0 / ( self.total_len - self.set_len ) * self.es_sign
        self.boot_med, self.bm_u, self.bm_l = median_withCI(self.bg, alpha = self.alpha)
        self.boot_med_sign = np.sign(self.boot_med)
    def find_nes(self):
        '''NES calculation depends on the use case.

        Should be overridden by derived class
        '''
        raise NotImplementedError('find_nes not overridden')
    def find_p(self):
        '''p-value depends on the use case.

        Should be overridden by derived class
        '''
        raise NotImplementedError('find_p not overridden')


class Runner(bootstrap_enrichment):
    deaPlotter = 'plotly' #'deaPlotter.R'
    def __init__(self,**kwargs):
        super(Runner, self).__init__(**kwargs)
        self.fhf = kwargs.get('fhf',None)
            # set fhf to None to suppress output; data is still
            # accessible as Runner properties
        assert 'fileHandle' not in kwargs
        self.png = kwargs.get('png',False)
        self.ws = kwargs.get('ws',None)
    def run(self):
        self.get_non_zeros()
        self.calc_es()
        self.calc_poly_area()
        self.verify_results()
        if self.nPermuts:
            self.calc_nes()
        if self.fhf:
            self.write_dea_files()
            self.plot_results()
    def process_fv(self):
        super(Runner,self).score_fv()
        self.process_ties()
    def set_scores(self):
        super(Runner,self).set_scores()
        self.wtd_scores = (self.wtd_scores + (1.0 / self.total_len)) /2.0
    def get_misses_norm(self):
        self.misses_wtd_score_sum = sum(list(np.multiply(self.set_misses, self.wtd_scores)))
        self.misses_norm_factor = 1.0/self.misses_wtd_score_sum
    def find_res_peak(self):
        # Here we can enable the detection of an earlier, but similar height peak
        # This might hurt our p-values, but it's more what we're interested in
        self.es = self.find_early_max()
    def find_early_max(self, min_val = False, min_max_por = 0.9):
        if min_val:
            self.res = self._flip_sign(self.res)
        max_y = max(self.res)
        max_x = self.res.index(max_y) + 1
        if verbose:
            print("actual max is " + str(max_y) + " at " + str(max_x))
        slope = float(max_y/max_x)
        diff = [self.res[i] - float((i+1.0) * slope) for i in range(max_x)]
        biggest = max([0] + [diff[i] for i in range(len(diff)) if self.res[i] >= max_y * min_max_por])
        if biggest <= 0:
            if verbose:
                print("No earlier maxes found")
            ind = max_x-1
        else:
            ind = diff.index(biggest)
            if verbose:
                print("returning max is " + str(self.res[ind]) + " at " + str(ind))
        if min_val:
            self.res = self._flip_sign(self.res)
        return self.res[ind]
    def _flip_sign(l):
        return [i*-1.0 for i in l]
    def process_ties(self):
        self.tie_inds = []
        if self.weight != 0.0:
            self.find_ties()
        if verbose:
            print(("Found", str(len(self.tie_inds)), "ties of at least", str(self.min_tie), " drugs long."))
    def find_ties(self):
        self.no_ties_fv = []
        self.last_score = None
        self.running_val = 0.0
        self.tied_count = 0
        for i in range(self.total_len):
            if self.last_score is None:
                self.running_val = self.final_vector[i]
            elif self.wtd_scores[i] == self.last_score:
                self.running_val += self.final_vector[i]
                self.tied_count += 1
            else:
                self.update_ties(i)
                self.running_val = self.final_vector[i]
                self.tied_count = 0
            self.last_score = self.wtd_scores[i]
        self.update_ties(i)
        self.final_vector = self.no_ties_fv
    def update_ties(self, i):
        self.no_ties_fv.append(self.running_val)
        self.no_ties_fv += [0] * self.tied_count
        if self.tied_count > self.min_tie:
            self.tie_inds.append([i - self.tied_count - 1, i])
    def get_background(self):
        import random
        self.gen_bg_scores()
        r = random.Random()
        self.bg = []
        ts = time.time()
        for i in range(self.nPermuts):
            if verbose and self.nPermuts > 10 and i % (self.nPermuts//10) == 0:
                print("Finished " + str(i) + " permutations")
            s = r.sample(range(self.non_zero_total_len), self.non_zero_set_len)
            if self.total_len > self.non_zero_total_len:
                s += r.sample(range(self.non_zero_total_len, self.total_len), self.set_len - self.non_zero_set_len)
            self.non_vec_permut(s)
        if verbose:
            print(('background iterations took', time.time()-ts))
            print(('background max:', max(self.bg)))
            print(('background min:', min(self.bg)))
    def calc_poly_area(self):
        self.es_index = self.res.index(self.es)
        # for the area calculation we actually want the right side of the peak
        try:
            while self.res[self.es_index] == self.res[self.es_index+1]:
                self.es_index += 1
        except IndexError:
            pass
        score_of_es_peak = self.score_list[self.es_index][1]
        self.leading_edge_cnt = len([x for x in self.score_list if x[0] in self.set_oi and x[1] >= score_of_es_peak])
        self.draw_poly()
        self.measure_polygon_area()
        self.norm_poly_area()
    def norm_poly_area(self):
        from math import log
        if self.leading_edge_cnt > 0 and self.poly_area >= 0:
            self.poly_area /= (float(self.leading_edge_cnt) / float(len(self.set_oi)))
            self.poly_area /= (float(len(self.set_oi) + 100) / 2.0)
        else:
            self.poly_area = 99999.9
        self.poly_area = 10.0 - log(self.poly_area, 1.7)
    def draw_poly(self):
        xax = list(range(1, self.es_index+1))
        xax_flip = xax[:]
        xax_flip.reverse()
        self.x = xax + xax_flip
        self.y = self.res[:self.es_index] + [1.0]*(self.es_index)
    def measure_polygon_area(self):
        self.poly_area = 0.5*np.abs(np.dot(self.x,np.roll(self.y,1))-np.dot(self.y,np.roll(self.x,1))) + 1
        # the plus one is to account for the far left, which is an issue if the peak is at the first index
    def gen_bg_scores(self):
        downs = []
        ups = []
        adder = 1.0 / self.total_len
        denom = 2.0
        for x in self.score_list:
            common = (((abs(float(x[1])) ** self.weight) / self.score_total) + adder) / denom
            downs.append( -1.0 * common / self.misses_wtd_score_sum)
            ups.append((x[0], common / self.matches_wtd_score_sum))
        self.up_scores = ups
        self.down_scores = downs
    def non_vec_permut(self, s):
        up_bumps = []
        for i in s:
            if self.up_scores[i][1] > 0:
                up_bumps.append( (i,self.up_scores[i][1]) )
        up_bumps.sort(key=lambda x:x[0]) # the sort is necessary b/c of the random selection
        self.bg.append(self.find_peak(up_bumps, self.tie_inds[:]))
    def find_peak(self, up_bumps, tie_inds, peak = 0.0, val = 0, done = 0):
        while up_bumps:
            while tie_inds and tie_inds[0][1] <= up_bumps[0][0]:
                del tie_inds[0] # ties with a down bump aren't relevant
            if tie_inds and up_bumps[0][0] >= tie_inds[0][0]:
                tie_start,tie_after = tie_inds[0]
                val += sum(self.down_scores[done:tie_start-done+1]) # add the downs up to the tie start
                done = tie_start # done is a tracker of what index we've completed, so we're setting it at the beginning of the tie
                while up_bumps and up_bumps[0][0] < tie_after: # get all of the ups in the tie
                    val += up_bumps[0][1]
                    done += 1 # note that we dealt with one more value
                    del up_bumps[0]
                val += sum(self.down_scores[done:tie_after-done+1]) # now add all of the downs in the tie
                done = tie_after
            else: # we're at a 'normal' up
                up_idx,up_val = up_bumps[0]
                val += sum(self.down_scores[done:up_idx-done+1]) # add the downs up until this point
                val += up_val # add the up
                done = up_idx+1
                del up_bumps[0]
            # check to see if this is the peak
            if val > peak:
                peak = val
        return peak
    def find_nes(self):
        self.nes = float((self.es + self.to_avoid_zero) / (self.boot_med + self.to_avoid_zero))
    def find_p(self):
        self.pval = float(sum(i > self.es for i in self.bg)) / self.nPermuts
        self.boostrappedESSummary = [self.boot_med, self.nPermuts, self.weight,
                                     self.min_tie, self.leading_edge_cnt
                                    ]
    def write_dea_files(self):
        statsFile = open(self.fhf.get_filename('stats'), 'w')
        statsFile.write("\t".join(["Size", "ES",
                                   "NES", "P-value"
                                   , 'Area'])
                          + "\n")
        statsFile.write("\t".join([str(x)
                                   for x in [
                                             self.set_len,
                                             self.es,
                                             self.nes,
                                             self.pval,
                                             self.poly_area
                                            ]
                                   ]) 
                          + "\n")
        statsFile.close()
    
        if True:#self.deaPlotter != 'plotly':
            drugScoresFile = open(self.fhf.get_filename('scores'), 'w')
            drugScoresFile.write("\n".join([str(drug[1]) for drug in self.score_list]) + "\n")
            drugScoresFile.close()
            RESFile = open(self.fhf.get_filename('res'), 'w')
            RESFile.write("\n".join([str(i) for i in self.res]) + "\n")
            RESFile.close()
            matchesFile = open(self.fhf.get_filename('matches'), 'w')
            matchesFile.write("\n".join([str(i) for i in self.set_matches]) + "\n")
            matchesFile.close()
            bootStatsFile = open(self.fhf.get_filename('boot'), 'w')
            bootStatsFile.write("\n".join([str(i) for i in [str("%.2f" % round(self.nes, 2))] 
                                                          + ["p = " + str(self.pval)]
                                                          + self.boostrappedESSummary
                                                          + [self.es, self.poly_area]
                                           ]) + "\n")
            bootStatsFile.close()
    def plot_results(self):
        import subprocess
        if self.deaPlotter == 'plotly':
            pp = self.dea_plotly()
            pp.save(self.fhf.get_filename('plotly'))
        else:
            png = ('TRUE' if self.png else 'FALSE')
            subprocess.check_call(['Rscript'
                                ,PathHelper.deaScripts + 'deaPlotter.R'
#                                ,PathHelper.deaScripts + self.deaPlotter
                                ,self.fhf.get_filename('scores')
                                ,self.fhf.get_filename('res')
                                ,self.fhf.get_filename('matches')
                                ,self.fhf.get_filename('boot')
                                ,self.fhf.get_filename('stem')
                                ,'FALSE'
                                ,png
                                ])
    def dea_plotly(self,dtc='wsa', protids=None):
        from dtk.plot import PlotlyPlot,Color
        from browse.models import WsAnnotation
        ids = []
        score_vals = []
        colors = [Color.default] * len(self.score_list)
        widths = [1] * len(self.score_list)
        for i,x in enumerate(self.score_list):
            ids.append(str(x[0]))
            score_vals.append(x[1])
            if x[0] in self.set_oi:
                colors[i] = Color.highlight
                widths[i] = 3
        # ugly way to get ws object; assume id list not empty
        if dtc == 'wsa':
            if self.ws is None:
                self.ws = WsAnnotation.all_objects.get(pk=ids[0]).ws
            name_map = self.ws.get_wsa2name_map()
            drug_names = [name_map[int(id)] for id in ids]
        elif dtc == 'uniprot':
            from browse.views import get_prot_2_gene
            name_map = get_prot_2_gene(protids)
            drug_names = [name_map[id] for id in ids]
        poly_xs = list(range(1, self.es_index+1))
        back = list(range(1, self.es_index+1))
        back.reverse()
        poly_xs += [self.es_index+1] + back
        poly_ys = self.res[:self.es_index] + [1.0]*(len(list(range(self.es_index))) + 1)
# following sprint193fixes we create what the tools returned by hand
# as we were getting formatting errors only on the platform machine
# this includes adding the axes to the traces, and making the figure
        trace0 = dict(
                     mode = 'lines',
                     yaxis = 'y1',
                     xaxis = 'x1',
                     x = [i + 1 for i in range(len(self.res))],
                     y = self.res,
                     text = drug_names,
                     line = dict(
                         color = 'red',
                         width = 2
                        ),
                 )
        trace1 = dict(
                     yaxis = 'y1',
                     xaxis = 'x1',
                     mode = 'lines',
                     x = poly_xs,
                     y = poly_ys,
                     fill='tozerox',
                     line = dict(
                               color = 'green',
                               width = 0,
                              )
                 )
        trace2 = dict(
                     yaxis = 'y2',
                     xaxis = 'x1',
                     type='bar',
                     y = score_vals,
                     text = drug_names,
                     textposition='none',
                     marker = dict(
                          color = colors,
                          line = dict(
                                 width = widths,
                                 color = colors
                             )
                         )
                    )
        fig = {'data': [trace0,trace1,trace2],
               'layout': {
                   'yaxis1': {'position': 0.0,
                              'domain': [0.495, 1.0],
                              'anchor': 'free'
                             },
                   'yaxis2': {'domain': [0.0, 0.505],
                               'anchor': 'x1'
                             },
                   'xaxis1': {'domain': [0.0, 1.0],
                              'anchor': 'y2'
                             }
               }}
        from dtk.plot import annotations
        fig['layout'].update(height = 800
                           , width = 800
                           , title = 'Drug Enrichment Analysis'
                           , showlegend = False
                           , annotations = annotations(
                                              'DEA Area: ' +
                                              str(round(self.poly_area, 3)) +
                                              '       ' +
                                              'Leading edge count: ' +
                                              str(self.leading_edge_cnt) +
                                              '       ' +
                                              'Peak location: ' +
                                              str(self.es_index)
                                           )
                           , shapes = [
                                 {
                                   'type' : 'line',
                                   'x0' : self.es_index,
                                   'y0' : 0,
                                   'x1' : self.es_index,
                                   'y1' : tup[1],
                                   'line' : {
                                          'color' : 'red',
                                          'dash' : 'dot',
                                            },
                                   'yref' : tup[0]
                                  }
                              for tup in [('y1', 1.0), ('y2',  max(score_vals))]
                             ]
                           )
        fig['layout']['yaxis1'].update(title = 'Running Enrichment Score')
        fig['layout']['yaxis2'].update(title = 'Drug score')
        fig['layout']['xaxis1'].update(title = 'Drug Rank')
        return PlotlyPlot(fig['data'],
                    fig['layout'],
                    id_lookup=[ids,None,ids],
                    click_type={
                            'wsa':'drugpage',
                            'uniprot':'protpage',
                            }[dtc]
                    )

class FileHandleFactory:
    sep='_'
    suffix={
            'stem':[],
            'stats':["summaryStats.tsv"],
            'scores':["drugScores.txt"],
            'res':["RES.txt"],
            'matches':["matches.txt"],
            'boot':["bootstrappingStats.txt"],
            'plot':["DEAPlots.pdf"],
            'plotly':["DEAPlots.plotly"],
            }
    prefix = ['dea']
    def __init__(self):
        self.dirpath = '.'
        self.parts = ['noscore','nods','nobg']
    def set_score(self,code):
        self.parts[0] = code
    def set_ds(self,code):
        self.parts[1] = code
    def set_bg(self,code):
        self.parts[2] = code
    def get_dea_name(self):
        return self.sep.join(self.parts)
    def get_filename(self,code):
        filename = self.sep.join(
                        self.prefix + self.parts + self.suffix[code]
                        )
        import os
        return os.path.join(self.dirpath,filename)
    def fixup(self,parts):
        # hook for legacy version
        return parts
    def scandir(self):
        import os
        try:
            suffix = self.suffix['stats']
            names = []
            for name in os.listdir(self.dirpath):
                parts = name.split(self.sep)
                if parts[-len(suffix):] != suffix:
                    continue
                if self.prefix != parts[:len(self.prefix)]:
                    continue
                cvt = self.fixup(parts[len(self.prefix):-len(suffix)])
                if cvt:
                    names.append(cvt)
        except OSError as ex:
            names = []
        return names
###
### Do we still need the legacy view?
###
class LegacyFactory(FileHandleFactory):
    def __init__(self,dirpath,template):
        self.dirpath = dirpath
        self.parts = template.parts
        self.prefix = []
    equiv = [
            ('kts','knownTreatments'),
            ('ktsexp','knownAndExpTreatments'),
            ('ws','wsBackground'),
            ('dpi','dpiBackground'),
            ]
    def upgrade(self,s):
        idx = [x[1] for x in self.equiv].index(s)
        return self.equiv[idx][0]
    def downgrade(self,s):
        idx = [x[0] for x in self.equiv].index(s)
        return self.equiv[idx][1]
    def get_filename(self,code):
        filename = self.sep.join([
                        self.parts[0],
                        self.downgrade(self.parts[1]),
                        self.downgrade(self.parts[2]),
                        ] + self.suffix[code]
                        )
        import os
        return os.path.join(self.dirpath,filename)
    def fixup(self,parts):
        if len(parts) != 3:
            return None
        return [
                parts[0],
                self.upgrade(parts[1]),
                self.upgrade(parts[2]),
                ]


class Options:
    @classmethod
    def get_background_choices(cls):
        result = [
                ('ws','All Workspace Drugs'),
                ('score','Drugs with assigned scores'),
                ('nonzero','Drugs with non-zero scores'),
                ]
        from dtk.prot_map import DpiMapping
        for name in DpiMapping.dpi_names():
            result.append( (name,'Drugs w/ DPI in '+name) )
        return result
    def adjust_score_for_background(self,ordering):
        from old_dea import retreiveFullBackground
        drugScoreList = list(ordering)
        scored_drugs = set([x[0] for x in drugScoreList])
        if self.bg == 'ws':
            bg = retreiveFullBackground(self.ws.id)
            # use integer rather than string keys
            bg = set([ int(x) for x in bg ])
        elif self.bg == 'score':
            bg = scored_drugs
        elif self.bg == 'nonzero':
            bg = set([
                    x[0]
                    for x in drugScoreList
                    if x[1] > 0
                    ])
        else:
            from dtk.prot_map import DpiMapping
            dpis = DpiMapping.dpi_names()
            if self.bg in dpis:
                from old_dea import anyDpiBackground
                bg = anyDpiBackground(self.ws,self.bg)
            else:
                raise Exception('unknown BG type')
        drugScoreList = [
                x
                for x in drugScoreList
                if x[0] in bg
                ]
        for drug_id in bg:
            if drug_id not in scored_drugs:
                drugScoreList.append( (drug_id,0.0) )
        return drugScoreList
    def get_drugset_choices(self):
        return self.ws.get_wsa_id_set_choices()
    def get_drugset(self):
        return self.ws.get_wsa_id_set(self.ds)
    _parms = 'bg ds weight alpha min_tie npermuts force'.split()
    def __init__(self,ws):
        self.ws = ws
        self.bg = 'ws'
        self.ds = ws.eval_drugset
        self.weight = 1.5
        self.alpha = 0.01
        self.min_tie = 5
        self.npermuts = 1000
        self.force = False
    def update_from_dict(self,d):
        for key in self._parms:
            try:
                val = d[key]
                cast = type(getattr(self,key))
                if cast == bool:
                    # default bool cast turns everything to True,
                    # so do this instead
                    val = val in ('True','1',True,1)
                else:
                    val = cast(val)
                setattr(self,key,val)
            except KeyError:
                pass
    def as_dict(self):
        from dtk.data import dict_subset
        return dict_subset(self.__dict__,self._parms)
    def get_config_html(self):
        from django import forms
        class DeaForm(forms.Form):
            bg = forms.ChoiceField(label='Background Type'
                            ,initial=self.bg
                            ,choices=self.get_background_choices()
                            )
            ds = forms.ChoiceField(label='Ranked Drugset'
                            ,initial=self.ds
                            ,choices=self.get_drugset_choices()
                            )
            weight = forms.FloatField(label='Score Weight'
                            ,initial=self.weight
                            ,min_value=0.0
                            )
            # XXX this is hidden pending a decision on whether we should 
            # XXX EVER use cached results, given that they may have been
            # XXX produced with non-standard parameters
            # The 'force' field separates fields which request runs with
            # distinct filenames (above) from runs which differ only
            # in tweaking the way to arrive at the result (below).  If
            # an existing file matches all the parameters above, and
            # force is not set, the file will be used, meaning any settings
            # below are essentially ignored.
            #force = forms.BooleanField(label='Force re-run?'
            #                ,required=False
            #                ,initial=self.force
            #                )
            alpha = forms.FloatField(label='Alpha'
                            ,initial=self.alpha
                            )
            min_tie = forms.IntegerField(label='Minimum Tie Length'
                            ,initial=self.min_tie
                            )
            npermuts = forms.IntegerField(label='Background Permutations'
                            ,initial=self.npermuts
                            )
        return DeaForm().as_table()
    def run(self,ordering,fhf):
        fhf.set_ds(self.ds)
        fhf.set_bg(self.bg)
        kts = self.ws.get_wsa_id_set(self.ds)
        drugScoreList = self.adjust_score_for_background(ordering)
        drugScoreList.sort(key=lambda x:x[1],reverse=True)
        run = Runner(
                weight = self.weight,
                nPermuts = self.npermuts,
                min_tie = self.min_tie,
                fhf=fhf,
                score_list = drugScoreList,
                set_of_interest = self.get_drugset(),
                alpha = self.alpha,
                ws = self.ws
                )
        run.run()
        return run

class EnrichmentResult:
    def __init__(self,bji,code):
        self.metric = 'Area'
        self.bji = bji
        self.fhf = FileHandleFactory()
        self.fhf.dirpath = bji.get_dea_path()
        parts = code.split('_')
        self.fhf.set_score(parts[0])
        self.fhf.set_ds(parts[1] if len(parts) > 1 else bji.ws.eval_drugset)
        self.fhf.set_bg(parts[2] if len(parts) > 2 else 'ws')
        self._summary = None
        # if we can't find the files under the 'modern' names, try the old
        # ones (but only if we're using an eval_drugset that HAD an old name)
        if not self.get_summary() and self.fhf.parts[1] in ('kts','ktsexp'):
            base_fhf = self.fhf
            self.fhf = LegacyFactory(bji.get_legacy_dea_path(),self.fhf)
            self._summary = None
            if not self.get_summary():
                self.fhf = base_fhf
    def get_summary(self):
        if self._summary is None:
            try:
                filename = self.fhf.get_filename('stats')
                f = open(filename)
                lines = f.read().split('\n')
                recs = [x.split('\t') for x in lines]
                self._summary = dict(zip(recs[0],recs[1]))
            except IOError as ex:
                self._summary = {}
        return self._summary
    def get_href(self):
        path = self.fhf.get_filename('plotly')
        if os.path.exists(path):
            return PathHelper.url_of_plot(
                        self.bji.ws.id,
                        path,
                        self.fhf.get_dea_name()+' DEA plot',
                        )
        path = self.fhf.get_filename('plot')
        if not os.path.exists(path):
            return ""
        return PathHelper.url_of_file(path)
    def get_value(self):
        d = self.get_summary()
        try:
            return float(d[self.metric])
        except (KeyError,ValueError):
            return None
    def dea_link(self):
        href = self.get_href()
        d = self.get_summary()
        if self.metric in d:
            val = "%0.1f" % (float(d[self.metric]),)
            from dtk.html import link,glyph_icon,hover
            if href:
                val = link(val,href)
            return val
        return 'N/A'
    def summary_line(self):
        s = self.get_summary()
        if s and self.metric in s:
            from tools import sci_fmt
            r='%s %s: <b>%s</b>' % (
                    self.metric,
                    self.fhf.get_dea_name(),
                    sci_fmt(s[self.metric]),
                    )
        else:
            r='(legacy) '+self.fhf.get_dea_name()
        href = self.get_href()
        if href:
            from dtk.html import link
            r += ' ' + link('DEA plot',href)
        from django.utils.safestring import mark_safe
        return mark_safe(r)

