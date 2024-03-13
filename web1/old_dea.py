#!/usr/bin/python
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

# created 4.Oct.2015 - Aaron C Daugherty, some code from Carl
# This takes a pathsum run ID and actullay runs a GSEA-like analysis
# To do:
# clean out old code and generally refactor after verifying the settings we like
# refactor into object oriented to simplify passing variables

verbose = True

def useage(*objs):
    print("USEAGE: ", *objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def printOut(*objs):
    print(*objs, file=sys.stdout)

def anyDpiBackground(ws,mapping):
    bg = set()
    from dtk.prot_map import DpiMapping
    dpi = DpiMapping(mapping)
    return set(wsa for key,wsa in dpi.get_key_wsa_pairs(ws))

def retreiveDPIBackground(wsid):
    # All of the drugs that could possibly have had a score
    # luckily we have that list in a file
    # we will retreive it here: ws/12/pathsum_matchable_drugs.txt
    # where each line is a drug ID
    matchableDrugFilehandle = PathHelper.storage + str(wsid) +"/pathsum_matchable_drugs.txt"
    allDrugsList = [line.rstrip('\n') for line in open(matchableDrugFilehandle)]
    return(allDrugsList)

def retreiveFullBackground(wsid):
    # basically just get all of the WS annotation IDs
    from browse.models import WsAnnotation
    qs = WsAnnotation.objects.filter(ws_id=wsid)
    return([str(rec.id) for rec in qs])

def formatWSAs(drugSet, backgroundList):
    toRemove = set(drugSet) - set(backgroundList)
    toReturn = list(set(drugSet) - set(toRemove))
    return(toReturn)

##########################################################################
# Everything in this section is actually used by DEA and should remain, though likely could be improved
##########################################################################

def findListMatches(drugIDList, drugSet):
    return([1 if drugList[0] in drugSet else 0 for drugList in drugIDList])

# This is numerically equivalent to coreES in the 'simple' case, but is
# optimized for background generation:
# - scores are extracted and floated once
# - random sampler generates indicies into score list
# - for each permutation, the 'down' values is l are bulk-loaded, and then
#   the 'up' values overwrite them (in a loop with relatively few iterations)
# - np.cumsum does the heavy lifting
# This is about 8x faster than calling getSimpleES repeatedly.
#
# A similar optimization could be done on coreES, but it wouldn't measurably
# affect the timing.
def getBackgroundFast(drugScoreList, setSize, weight, sumScoresMatches, score_total, nPermuts, tie_inds, nonZeroSetSize, deplet = False, verbose = True):
    import random
    scoreOnlyList = [float(x[1]) for x in drugScoreList]
    NonZeroScoreLen = len([x for x in scoreOnlyList if x!=0])
    downBump = -1.0/(len(scoreOnlyList)-setSize)
    r = random.Random()
    result =[]
    ts = time.time()
    for i in range(nPermuts):
        if verbose and nPermuts > 10 and i % (nPermuts/10) == 0:
            printOut("Finished " + str(i) + " permutations")
        s = r.sample(range(NonZeroScoreLen),nonZeroSetSize)
        if len(scoreOnlyList) > NonZeroScoreLen:
            s += r.sample(list(range(NonZeroScoreLen, len(scoreOnlyList))),setSize-nonZeroSetSize)
        result.append(non_vec_permut(s, scoreOnlyList, downBump, sumScoresMatches, score_total, weight, tie_inds, deplet = deplet))
    if verbose:
        printOut('background iterations took', time.time()-ts)
    return result

def non_vec_permut(s, scoreOnlyList, downBump, sumScoresMatches, score_total, weight, tie_inds, include_rank = True, deplet = False):
    tie_inds = list(reversed(tie_inds))
    if include_rank:
        up_bumps = [
            (x, (((scoreOnlyList[x] ** weight)/score_total + (1.0/len(scoreOnlyList))) / 2.0))#/sumScoresMatches)
#            (x, (((scoreOnlyList[x] ** weight) + 1.0) / 2.0) /sumScoresMatches)
            for x in s
            ]
    else:
        up_bumps = [
            (x, (scoreOnlyList[x] ** weight) /sumScoresMatches)
            for x in s
            ]
    up_bumps = [x for x in up_bumps if x[1] > 0]
    up_bumps.sort(key=lambda x:x[0])
    peak = downBump
    done = 0
    val = 0
    return find_peak(up_bumps, tie_inds, downBump, peak, deplet)

def find_peak(up_bumps, tie_inds, downBump, peak, deplet, val = 0, done = 0):
    while up_bumps:
        while tie_inds and tie_inds[0][1] <= up_bumps[0][0]:
            tie_inds = tie_inds[1:]
        if tie_inds and up_bumps[0][0] >= tie_inds[0][0]:
            tie_start,tie_after = tie_inds[0]
            val += (tie_start-done)*downBump
            done = tie_start
            while up_bumps and up_bumps[0][0] < tie_after:
                val += up_bumps[0][1]
                done += 1
                del up_bumps[0]
            val += (tie_after-done)*downBump
            done = tie_after
        else:
            up_idx,up_val = up_bumps[0]
            val += (up_idx-done)*downBump
            val += up_val
            done = up_idx+1
            del up_bumps[0]
        if (not deplet and val > peak) or (deplet and abs(val) > abs(peak)):
            peak = val
    return peak

###
### coreES seems to be particularly ready for an overhaul
###
def coreES(drugScoreList, drugSet, weight, simple = True, sumScoresMatches = None, min_tie_len = 3, include_rank = True, verbose = True):
    isThereAMatch = np.array(findListMatches(drugScoreList, drugSet)) # pull out just the drug IDs
    isThereNoMatch = 1 - isThereAMatch  # the inverse of the above
    listTotal = len(drugScoreList) # how many total drugs are we talking about
    setTotal = len(drugSet)
    listDrugsNotInSet =  listTotal - setTotal
    # the sum of all scores which correspond to a drug in drug.list that has a match in drug.set
    scoresVector = np.array([abs(float(drug[1])**weight) for drug in drugScoreList])
    scoreTotal = float(sum(list(scoresVector)))
    scoresVector = np.array([d/scoreTotal for d in scoresVector])
    if include_rank and len(scoresVector):
        scoresVector = (scoresVector + (1.0/len(scoresVector))) /2.0
    if not simple:
        # rather than looping through, we can just use array multiplication, via numpy, and sum the result
        sumScoresMatches = sum(list(np.multiply(isThereAMatch, scoresVector)))
    elif sumScoresMatches is None:
        warning("If running simple coreES, need to provide the sumScoresMatches from the fullES run. Quitting.")
        exitCoder = ExitCoder()
        sys.exit(exitCoder.encode('usageError'))
    if sumScoresMatches != 0:
        normalizationFactorForMatches = 1.0/sumScoresMatches
    else:
        return  [None, None], None, None
    normFactorNonMatches = 1.0/listDrugsNotInSet
    # Try numpy array multiplication
    finalVector = list(isThereAMatch * scoresVector * normalizationFactorForMatches - isThereNoMatch * normFactorNonMatches)
    tie_inds = []
    if weight != 0.0:
        finalVector, tie_inds = process_ties(scoresVector, finalVector, min_tie_len)
    if verbose:
        printOut("Found", str(len(tie_inds)), "ties of at least", str(min_tie_len + 1), " drugs long.")
    if simple:
        return finalVector
    else:
        return [finalVector, isThereAMatch], sumScoresMatches, tie_inds, scoreTotal

def process_ties(scoresVector, finalVector, min_tie_len):
    # Can we process this vector to identify ties
    no_ties_fv = []
    last_score = None
    running_val = 0.0
    tied_count = 0
    tie_inds = []
    for i in range(len(scoresVector)):
        if last_score is None:
            running_val = finalVector[i]
        elif scoresVector[i] == last_score:
            running_val += finalVector[i]
            tied_count += 1
        else:
            no_ties_fv, tie_inds = update_ties(no_ties_fv, running_val, tied_count, tie_inds, i, min_tie_len = min_tie_len)
            running_val = finalVector[i]
            tied_count = 0
        last_score = scoresVector[i]
    # catch the last one
    return update_ties(no_ties_fv, running_val, tied_count, tie_inds, len(scoresVector), min_tie_len = min_tie_len)

def update_ties(no_ties_fv, running_val, tied_count, tie_inds, i, min_tie_len = 1):
    no_ties_fv.append(running_val)
    no_ties_fv = no_ties_fv + [0] * tied_count
    if tied_count >= min_tie_len:
        tie_inds.insert(0,[i - tied_count - 1, i])
    return no_ties_fv, tie_inds

def getFullES(drugScoreList, drugSet, weight, min_tie_len = 3, deplet = False, verbose = True):
    finalVecAndMatches, sumScoresMatches, tie_inds, scoreTotal = coreES(drugScoreList, drugSet, weight, simple=False, min_tie_len=min_tie_len, verbose = verbose)
    if finalVecAndMatches[0] is None:
        return [None, None, finalVecAndMatches[1]], sumScoresMatches, tie_inds
    RES = list(np.cumsum(np.array(finalVecAndMatches[0])))
# Here we can enable the detection of an earlier, but similar height peak
# This might hurt our p-values, but it's more what we're interested in
    if deplet and abs(min(RES)) > max(RES):
# Right now only GLEE uses the deplete, and I'm not sure I'm ready to use this feature for GLEE. Still want to think about it....
        ES = min(RES)
#        ES = find_early_max(RES, min_val = True)
    elif deplet:
        ES = max(RES)
    else:
        ES = find_early_max(RES)
    return [ES, RES, finalVecAndMatches[1]], sumScoresMatches, tie_inds, scoreTotal

# I ended up not using this as is b/c it wasn't noticably changing things
def find_early_max(RES, min_val = False, min_max_por = 0.9):
    if min_val:
        RES = flip_sign(RES)
    max_y = max(RES)
    max_x = RES.index(max_y)+1
    if verbose:
        print("actual max is " + str(max_y) + " at " + str(max_x))
    slope = float(max_y/max_x)
    diff = [RES[i] - float((i+1.0) * slope) for i in range(max_x)]
    biggest = max([0] + [diff[i] for i in range(len(diff)) if RES[i] >= max_y * min_max_por])
    if biggest <= 0:
        if verbose:
            print("No earlier maxes found")
        ind = max_x
    else:
        ind = diff.index(biggest)
        if verbose:
            print("returning max is " + str(RES[ind]) + " at " + str(ind))
    if min_val:
        RES = flip_sign(RES)
    return RES[ind]
def flip_sign(RES):
    return [i*-1.0 for i in RES]

def run_enrich(nPermuts, drugScoreList, drugSet, weight, realResults, sumScoresMatches, score_total, tie_inds, alpha = 0.05, deplet = False, verbose = True):
    nonZeroSetLen = len([x for x in drugScoreList if x[0] in drugSet and x[1] != 0])
    bootstrappedES = getBackgroundFast(drugScoreList, len(drugSet),weight,sumScoresMatches,score_total, nPermuts, tie_inds, nonZeroSetLen, deplet = deplet, verbose = verbose)
    return calc_enrich(drugScoreList, drugSet, bootstrappedES, alpha, realResults, nPermuts, weight, deplet = deplet)

def calc_enrich(drugScoreList, drugSet, bootstrappedES, alpha, realResults, nPermuts, weight, deplet = False):
    ## a negative value means there was never any enrichment (only depletion)
    # also, we don't want to divide by 0
    # to avoid that we'll always just shift everything the smallest amount: 2X the downstep
    # the smallest max value we can get from the bootstrap is 1x the downstep
    # for consistency we'll do it for everything
    es_sign = np.sign(realResults[0])
    to_avoid_zero = 2.0 / ( len(drugScoreList) - len(drugSet) ) * es_sign
    boot_med, bm_u, bm_l = median_withCI(bootstrappedES, alpha = alpha)
    boot_med_sign = np.sign(boot_med)
    if not deplet or es_sign == boot_med_sign:
        nes = float((realResults[0] + to_avoid_zero) / (boot_med + to_avoid_zero))
        nes_l = float((realResults[0] + to_avoid_zero) / (bm_u + to_avoid_zero))
        nes_u = float((realResults[0] + to_avoid_zero) / (bm_l + to_avoid_zero))
    else:
        shifted_bmu = abs(abs(bm_u) - abs(boot_med))
        shifted_bml = abs(abs(bm_l) - abs(boot_med))
        shifted_es = abs(realResults[0]) + abs(boot_med) + shifted_bml
        nes = float((shifted_es + to_avoid_zero) / (to_avoid_zero + shifted_bml)) * es_sign
        bound1 = float((shifted_es + to_avoid_zero) / (to_avoid_zero + shifted_bml + shifted_bmu)) * es_sign
        bound2 = float((shifted_es + to_avoid_zero) / (to_avoid_zero)) * es_sign
        nes_l = min([bound1,bound2])
        nes_u = max([bound1,bound2])
    if deplet and es_sign == -1:
        pval = float(sum(i < realResults[0] for i in bootstrappedES)) / nPermuts
    else:
        pval = float(sum(i > realResults[0] for i in bootstrappedES)) / nPermuts
    return [boot_med, bm_u, bm_l, nPermuts, weight], [nes, nes_l, nes_u], pval

# adapted from: http://exploringdatablog.blogspot.sk/2012/04/david-olives-median-confidence-interval.html
def median_withCI(y, alpha = 0.05):
    import scipy.stats
    import math
    n = len(y)
    ysort = sorted(y)
    nhalf = n/2
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

def write_dea_files(fhf, drug_set_len, realResults, nes_list, pval, drugScoreList, boostrappedESSummary, alpha):

    statsFile = open(fhf.get_filename('stats'), 'w')
    statsFile.write("\t".join(["Size", "ES", "NES", "NES_lower", "NES_upper", "alpha", "P-value"]) + "\n")
    statsFile.write("\t".join([str(drug_set_len), str(realResults[0]), str(nes_list[0]), str(nes_list[1]), str(nes_list[2]), str(alpha), str(pval)]) + "\n")
    statsFile.close()

    drugScoresFile = open(fhf.get_filename('scores'), 'w')
    drugScoresFile.write("\n".join([str(drug[1]) for drug in drugScoreList]) + "\n")
    drugScoresFile.close()

    RESFile = open(fhf.get_filename('res'), 'w')
    RESFile.write("\n".join([str(i) for i in realResults[1]]) + "\n")
    RESFile.close()

    matchesFile = open(fhf.get_filename('matches'), 'w')
    matchesFile.write("\n".join([str(i) for i in realResults[2]]) + "\n")
    matchesFile.close()

    bootStatsFile = open(fhf.get_filename('boot'), 'w')
    bootStatsFile.write("\n".join([str(i) for i in [str("%.2f" % round(nes_list[0], 2))]
                                                 + [str((1.0 - alpha) * 100)
                                                      + "% CI: "
                                                      + str("%.2f" % round(nes_list[1], 2))
                                                      + " - "
                                                      + str("%.2f" % round(nes_list[2], 2))]
                                                 + ["p = " + str(pval)]
                                                 + boostrappedESSummary
                                                 + [realResults[0]]
                                   ]) + "\n")
    bootStatsFile.close()

def plot_results(deaPlotter, fhf, check_enrich = 'FALSE', png = 'FALSE'):
    import subprocess
    subprocess.check_call(['Rscript'
                        ,PathHelper.deaScripts + deaPlotter
                        ,fhf.get_filename('scores')
                        ,fhf.get_filename('res')
                        ,fhf.get_filename('matches')
                        ,fhf.get_filename('boot')
                        ,fhf.get_filename('stem')
                        , check_enrich
                        , png
                        ])


class Runner:
    deaPlotter = 'deaPlotter.R'
    def __init__(self,**kwargs):
        self.weight = kwargs.get('weight',1)
        self.nPermuts = kwargs.get('nPermuts',1000)
        self.min_tie = kwargs.get('min_tie',5)
            # below here, min_tie_len is the minimum number of 'extra'
            # matching scores (e.g. a value of 2 means a tie must be
            # 3 scores in a row). above here, we convert to a more
            # intuitive value of the total number of tied scores, and
            # use a slightly different variable name to help track
            # the different meaning.
        self.fhf = kwargs.get('fhf',None)
            # set fhf to None to suppress output; data is still
            # accessible as Runner properties
        assert 'fileHandle' not in kwargs
        self.png = kwargs.get('png',False)
        self.verbose = True
        self.deplet = False
    def run(self,drugScoreList,drugSet, alpha):
#        drugScoreList = [x for x in drugScoreList if x[1] > 0]
        self.realResults, self.sumScoresMatches, self.tie_inds, scoreTotal = getFullES(
                                                    drugScoreList,
                                                    drugSet,
                                                    self.weight,
                                                    self.min_tie-1,
                                                    self.deplet,
                                                    self.verbose,
                                                    )
        if not self.nPermuts:
            return
        if self.weight and not self.sumScoresMatches:
            warning('No members of drugSet had a non-zero score.')
            printOut('%d drugs in drugSet.' % len(drugSet))
            printOut('%d drugs with non-zero scores.' % len(
                                [x for x in drugScoreList if float(x[1])>0]
                                ))
            warning('exiting without generating output...')
            return

        self.boostrappedESSummary, self.nes_list, self.pval = run_enrich(
                                                    self.nPermuts,
                                                    drugScoreList,
                                                    drugSet,
                                                    self.weight,
                                                    self.realResults,
                                                    self.sumScoresMatches, scoreTotal,
                                                    self.tie_inds,
                                                    alpha,
                                                    self.deplet,
                                                    self.verbose,
                                                    )
        self.boostrappedESSummary.append(self.min_tie)
        if self.fhf:
            write_dea_files(
                        self.fhf,
                        len(drugSet),
                        self.realResults,
                        self.nes_list,
                        self.pval,
                        drugScoreList,
                        self.boostrappedESSummary,
                        alpha = alpha
                        )
            plot_results(
                        self.deaPlotter,
                        self.fhf,
                        png=('TRUE' if self.png else 'FALSE'),
                        )

class Options:
    def get_background_choices(self):
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
        self.weight = 1.0
        self.alpha = 0.01
        self.min_tie = 5
        self.npermuts = 100000
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
            # XXX the weight 0 case seems to be broken, most likely by the
            # XXX tie removal code (since in the weight 0 case, the whole thing
            # XXX is a tie).  We could comment this out to hide the option.
            # XXX Doing that still allows you to set the weight from the URL.
            weight = forms.FloatField(label='Score Weight'
                            ,initial=self.weight
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
        run = Runner(
                weight = self.weight,
                nPermuts = self.npermuts,
                min_tie = self.min_tie,
                fhf=fhf,
                )
        drugScoreList = self.adjust_score_for_background(ordering)
        drugScoreList.sort(key=lambda x:x[1],reverse=True)
        run.run(drugScoreList,self.get_drugset(),self.alpha)
        return run

class EnrichmentResult:
    def __init__(self,bji,code):
        self.bji = bji
        self.fhf = FileHandleFactory()
        self.fhf.dirpath = bji.get_dea_path()
        parts = code.split('_')
        self.fhf.set_score(parts[0])
        self.fhf.set_ds(parts[1] if len(parts) > 1 else 'kts')
        self.fhf.set_bg(parts[2] if len(parts) > 2 else 'ws')
        self._summary = None
        if not self.get_summary():
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
        path = self.fhf.get_filename('plot')
        if not os.path.exists(path):
            return ""
        return PathHelper.url_of_file(path)
    def get_value(self):
        d = self.get_summary()
        try:
            return float(d['NES'])
        except (KeyError,ValueError):
            return None
    def dea_link(self):
        href = self.get_href()
        d = self.get_summary()
        if 'NES' in d:
            val = "%0.1f" % (float(d['NES']),)
            ci = self.ci()
            from dtk.html import link,glyph_icon,hover
            if href:
                val = link(val,href)
            if ci:
                return val+hover(u'\xb1',ci)
            from django.utils.safestring import mark_safe
            return val+mark_safe('?')
        return 'N/A'
    def ci(self):
        s = self.get_summary()
        if 'alpha' in s:
            from tools import sci_fmt
            ci = 100 * (1-float(s['alpha']))
            return '%0.1f%% CI %s - %s, p=%s' % (
                    ci,
                    sci_fmt(s['NES_lower']),
                    sci_fmt(s['NES_upper']),
                    sci_fmt(s['P-value']),
                    )
        return ''
    def summary_line(self):
        s = self.get_summary()
        if not s:
            return ''
        from tools import sci_fmt
        r='%s NES: <b>%s</b>' % (self.fhf.get_dea_name(),sci_fmt(s['NES']),)
        ci = self.ci()
        if ci:
            r += ' (%s)' % ci
        href = self.get_href()
        if href:
            from dtk.html import link
            r += ' ' + link('DEA plot',href)
        from django.utils.safestring import mark_safe
        return mark_safe(r)

