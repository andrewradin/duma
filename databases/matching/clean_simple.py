#!/usr/bin/env python3

from __future__ import print_function
from builtins import str
from builtins import object
import sys
import os

MATCHING_DIR = os.path.dirname(os.path.abspath(__file__))

class dpi_counts(object):
    def __init__(self,data):
        self.drugs=len(set([x[0] for x in data]))
        self.prots=len(set([x[1] for x in data]))
        self.uniq_dpi=len(data)
        self.total_dpi=len([x for v in list(data.values()) for x in v])

class bioChemDPICleaner(object):
    def __init__(self,
                 in_iter,
                 tmp_r_file=os.path.join(MATCHING_DIR,'for_r.tmp'),
                 R_dir=MATCHING_DIR,
                 ):
        self.input_iterator = in_iter
        self.tmp_r_file=tmp_r_file
        self.R_dir=R_dir
        self.header = None
        self.data = {}
        self.all_vals=[]
        self.counts=[]
    def load(self):
        for frs in self.input_iterator:
            if self.header is None:
                self.header = frs
                self.check_dpi_header()
                continue
            k = (frs[0], frs[1])
            if k not in self.data:
                self.data[k] = []
            self.data[k].append((float(frs[2]),int(frs[3])))
            self.all_vals.append(float(frs[2]))
    def check_dpi_header(self):
        assert self.header[0].endswith('_id')
        self.drug_collection = '_'.join(self.header[0].split('_')[:-1])
        assert self.header[1] == 'uniprot_id'
        self.dpi_type = self.header[2]
        assert self.header[3] == 'direction'
# this one is optional for other DPI files
        assert len(self.header) == 4
    def filter(self):
### This is not the most efficient approach by any means
### It was written to mimic a previous R script
### It also allows for step by step viewing of the deltas
        self.get_counts('baseline')
        self.data=self.remove_outliers()
        self.get_counts('outliers')
        self.data=self.collapse_rptd_measures()
        self.get_counts('repeated measures')
        self.data=self.collapse_direction()
        self.get_counts('directions')
    def remove_outliers(self):
        self._get_ceiling()
        return self._remove_outliers()
    def _remove_outliers(self):
        new_data = {}
        for k,l in self.data.items():
            for vals in l:
                if vals[0] < self.upper_ceil and  vals[0] > 0:
                    if k not in new_data:
                        new_data[k] = []
                    new_data[k].append(vals)
        return new_data
    def _get_ceiling(self):
        from numpy import percentile, exp
        import subprocess as sp
        with open(self.tmp_r_file, 'w') as f:
            f.write("\n".join([str(x) for x in self.all_vals])+"\n")
        p = sp.Popen(['Rscript', os.path.join(self.R_dir,'medCouple.R'), self.tmp_r_file],stdout = sp.PIPE)
        for x in p.stdout:
            mc=float(x.split()[1])
            break
        (q1,q3) = percentile(self.all_vals, [25,75])
        iqr = q3-q1
# this outlier definition is sourced from here
# https://wis.kuleuven.be/stat/robust/papers/2008/adjboxplot-revision.pdf
        assert mc >= 0
        self.upper_ceil = q3 + 1.5*exp(3*mc)*iqr
    def collapse_rptd_measures(self):
        from scipy.stats.mstats import gmean
        from numpy import std
        new_data = {}
        for k,vals in self.data.items():
            dirs = {}
            for (evid,dir) in vals:
                if dir not in dirs:
                    dirs[dir] = []
                dirs[dir].append(evid)
            if dirs:
                new_data[k] = []
            for d,l in dirs.items():
                cnt = len(l)
                if cnt > 1:
# ddof=1 is to ensure we take the sample SD, not the population SD
# this is particularly relevant given the low number of samples we're dealing with
                    sd = round(std(l,ddof=1),2)
                    e = round(gmean(l),2)
                else:
                    sd = 0.
                    e = l[0]
                new_data[k].append((e, d, cnt, sd))
        return new_data
    def collapse_direction(self):
        new_data = {}
        for k,vals in self.data.items():
            new_vals = None
            max_cnt = 0
            for (e,d,c,s) in vals:
# take whichever direction has the most measurements...
                if c > max_cnt:
                    new_vals = (e,d,c,s)
                    max_cnt = c
# in the case of a tie, go with the smaller StDev
                elif c == max_cnt:
# and if there is still a tie take the lower value,
# though if that is a tie we just take the first one
                    if s == new_vals[-1]:
                        if e < new_vals[0]:
                            new_vals = (e,d,c,s)
                    elif s < new_vals[-1]:
                        new_vals = (e,d,c,s)
                    
            if new_vals:
                new_data[k] = [new_vals]
        return new_data
    def get_counts(self, label, d=None):
        if d is None:
            d = self.data
        self.counts.append((label,dpi_counts(d)))
    def report(self, fn):
        self._report_results(fn)
        self._report_counts()
    def _report_counts(self):
        print("\n".join(self._org_count_report()))
    def _org_count_report(self):
        to_ret=[]
        (init_label,init_o) = self.counts.pop(0)
        initial_d=init_o.drugs
        initial_p=init_o.prots
        initial_u=init_o.uniq_dpi
        initial_t=init_o.total_dpi
        to_ret+=[init_label,
                 '\tDrugs:     {:,}'.format(initial_d),
                 '\tProteins:  {:,}'.format(initial_p),
                 '\tUniq. DPI: {:,}'.format(initial_u),
                 '\tTotal DPI: {:,}'.format(initial_t),
                 ""
                ]
        steps = []
        d_percs = []
        p_percs = []
        u_percs = []
        t_percs = []
        for l,o in self.counts:
            steps.append(l)
            d_percs.append(stringify_round_percentage(o.drugs,initial_d))
            p_percs.append(stringify_round_percentage(o.prots,initial_p))
            u_percs.append(stringify_round_percentage(o.uniq_dpi,initial_u))
            t_percs.append(stringify_round_percentage(o.total_dpi,initial_t))
        to_ret+=['\t'.join(['']*2+steps)]
        to_ret+=['\t\t'.join(['Drugs    ']+d_percs)]
        to_ret+=['\t\t'.join(['Proteins ']+p_percs)]
        to_ret+=['\t\t'.join(['Uniq. DPI']+u_percs)]
        to_ret+=['\t\t'.join(['Total DPI']+t_percs)]
        to_ret+=['',
                 'Final totals',
                 '\tDrugs:     {:,}'.format(o.drugs),
                 '\tProteins:  {:,}'.format(o.prots),
                 '\tUniq. DPI: {:,}'.format(o.uniq_dpi),
                 '\tTotal DPI: {:,}'.format(o.total_dpi),
                ]
        return to_ret
    def _report_results(self, fn):
        with open(fn, 'w') as f:
            f.write("\t".join([self.drug_collection+'_id',
                               'uniprot_id',
                               'direction',
                               self.dpi_type+'_final',
                               'n_independent_measurements',
                               'stdDev'
                              ])+"\n")
            for k,v in self.data.items():
                (drug,uni) = k
                assert len(v) == 1
                (evid,dir,cnt,sd) = v[0]
                e = int(evid) if (evid).is_integer() else evid
                s = int(sd) if (sd).is_integer() else sd
                f.write("\t".join([str(x) for x in
                                   [drug, uni, dir, e, cnt, s]
                                  ])+"\n")

def stringify_round_percentage(n,d):
    return str(round((float(n)/d)*100.,2))

if __name__ == '__main__':
    from dtk.files import get_file_records
    import argparse
    parser = argparse.ArgumentParser(
        description='Combine and clean biochemical DPI data.')
    parser.add_argument('i', help='input file')
    parser.add_argument('o', help='output file')
    args = parser.parse_args()

    bcdc = bioChemDPICleaner(get_file_records(args.i,
                                              parse_type='tsv',
                                              keep_header=True,
                             ))
    bcdc.load()
    bcdc.filter()
    bcdc.report(args.o)
