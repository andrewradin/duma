from dtk.duma_view import DumaView,list_of,boolean
from dtk.table import Table,SortHandler
from django import forms
from django.http import HttpResponseRedirect

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse

import logging

logger = logging.getLogger(__name__)

# Create your views here.
class OngoCTView(DumaView):
    template_name='xws/ongoct.html'
    phase_choices=[
            ('ph2','Phase 2'),
            ('ph3','Phase 3'),
            ]
    button_map={
            'search':['filt'],
            }
    GET_parms={
            'job_list':(list_of(int),[]),
            'as_tsv':(boolean,False),
            'phase':(str,'ph2'),
            'sort':(SortHandler,'completion'),
            }
    def custom_setup(self):
        from .utils import prep_job_list
        filtered, options, messages = prep_job_list(self.job_list)
        self.job_list = filtered
        self.job_options = options
        for msg in messages:
            self.message(msg)
    def custom_context(self):
        if self.phase == 'ph2':
            target_ds = 'ct-ph2ongo'
        elif self.phase == 'ph3':
            target_ds = 'ct-ph3ongo'
        else:
            self.message(f"Unknown phase: '{self.phase}'")
            return
        # get molecules with data from ctretro runs
        from .utils import get_molecule_lists
        molecule_lists = get_molecule_lists(self.job_options,self.job_list)
        # build a lookup table across all workspaces
        self.retro_mol_cache = {}
        for l in molecule_lists.values():
            for rmd in l:
                self.retro_mol_cache[rmd.wsa_id] = rmd
        # retrieve all WSAs under the target ds name in each workspace
        ws_list = [x.ws_id for x in self.job_options if x.id in self.job_list]
        from browse.models import Workspace
        wsa_ids = set()
        for ws_id in ws_list:
            ws = Workspace.objects.get(pk=ws_id)
            wsa_ids |= ws.get_wsa_id_set(target_ds)
        # filter the wsa list by the stuff we have RetroMolData for
        # (Note that this will toss anything that didn't map to an MOA
        # molecule at the time the ctretro job was run.)
        wsa_ids = set(x for x in wsa_ids if x in self.retro_mol_cache)
        # only show the download link if there's some data
        if wsa_ids:
            from dtk.text import fmt_time
            ts = fmt_time(self.context["now"],"%Y-%m-%d")
            self.context_alias(
                    download_link=self.here_url(as_tsv=1),
                    tsv_filename=f'{target_ds}-{ts}.tsv',
                    )
        # instantiate data retrieval wrappers
        from .utils import CTData, CTDataCache
        cache = CTDataCache(wsa_ids=wsa_ids,phase=self.phase, find_ct_evidence=False,)
        rows = [CTData(wsa_id=x,cache=cache) for x in wsa_ids]
        # annotate rows with blinded rank
        for ctd in rows:
            rmd = self.retro_mol_cache[ctd.wsa_id]
            ctd.blinded_rank = rmd.get_combined_rank(True)[0]
        # return data in requested format
        if self.as_tsv:
            return self.send_tsv(rows)
        self.context_alias(
                trial_table = self.make_table(rows)
                )
    def send_tsv(self,rows):
        tsv_spec = [
                ('WS ID',lambda x:str(x.ws.id)),
                ('WS Name',lambda x:x.ws.name),
                ('Drug',lambda x:x.wsa.get_name(False)), # is_demo checked below
                ('Drug Page', lambda x:self.request.build_absolute_uri(
                        x.wsa.drug_url()
                        )),
                ('Targets',lambda x: ' '.join(
                        g+'-?+'[int(1+d)] # gene + direction suffix
                        for u,g,d in x.target_data
                        )),
                ('Blinded Rank',lambda x:str(x.blinded_rank)),
                ('CT Link',lambda x:x.ct_data.url if x.ct_data else ''),
                ('Completion',lambda x:x.completion),
                ('Sponsor',lambda x:x.sponsor),
                ]
        return self.tsv_response(tsv_spec,rows,self.tsv_filename)
    def get_sort_key(self):
        colspec = self.sort.colspec
        if colspec == 'workspace':
            return lambda x:x.workspace
        if colspec == 'name':
            return lambda x:x.wsa.get_name(False)
        if colspec == 'targets':
            return lambda x:tuple((g,d) for u,g,d in x.target_data)
        if colspec == 'blinded_rank':
            return lambda x:x.blinded_rank
        if colspec == 'ct_link':
            return lambda x:x.ct_data.label if x.ct_data else ''
        if colspec == 'completion':
            return lambda x:x.completion
        if colspec == 'sponsor':
            # Only return sponsor if we have a CT id. This prevents the
            # string 'Unavailable' from sorting in the middle of the
            # list of sponsor names.
            return lambda x:x.sponsor.lower() if x.ct_id else ''
        # else return None and don't sort
    def make_table(self,rows):
        key = self.get_sort_key()
        if key:
            rows = sorted(rows,key=key,reverse=self.sort.minus)
        from dtk.table import Table
        from dtk.html import link
        return Table(rows,columns=[
                Table.Column('Workspace',
                        sort='l2h',
                        ),
                Table.Column('Name',
                        extract=lambda x:link(
                                x.wsa.get_name(self.is_demo()),
                                x.wsa.drug_url(),
                                new_tab=True,
                                ),
                        sort='l2h',
                        ),
                Table.Column('Targets',
                        sort='l2h',
                        ),
                Table.Column('Blinded Rank',
                        sort='l2h',
                        ),
                Table.Column('CT Link',
                        sort='l2h',
                        ),
                Table.Column('Completion',
                        sort='l2h',
                        ),
                Table.Column('Sponsor',
                        sort='l2h',
                        ),
                ],
                url_builder=self.here_url,
                sort_handler=self.sort,
                )
    def make_filt_form(self,data):
        from .utils import get_job_choices
        class MyForm(forms.Form):
            job_list = forms.MultipleChoiceField(
                    label='Input Jobs',
                    choices=get_job_choices(self.job_options),
                    required=True,
                    initial=self.job_list,
                    widget=forms.SelectMultiple(
                            attrs={'size':20}
                            ),
                    )
            phase = forms.ChoiceField(
                    choices=self.phase_choices,
                    initial=self.phase,
                    )
        return MyForm(data)
    def search_post_valid(self):
        p = self.context['filt_form'].cleaned_data
        p['job_list'] = ','.join(p['job_list'])
        return HttpResponseRedirect(self.here_url(**p))

class RetroCTView(DumaView):
    template_name='xws/retroct.html'
    button_map={
            'calc':['select'],
            }
    GET_parms={
            'job_list':(list_of(int),[]),
            'as_tsv':(boolean,False),
            }
    def custom_setup(self):
        from .utils import prep_job_list
        filtered, options, messages = prep_job_list(self.job_list)
        self.job_list = filtered
        self.job_options = options
        for msg in messages:
            self.message(msg)
    def custom_context(self):
        from dtk.ct_predictions import UNCLEAR_THRESHOLD,SUPPORTED_THRESHOLD
        self.context_alias(output_list=[],unclr_thrsh=UNCLEAR_THRESHOLD,supd_thrsh=SUPPORTED_THRESHOLD)
        self.get_molecule_lists()
        if self.molecule_lists:
            self.context_alias(
                    download_link=self.here_url(as_tsv=1),
                    tsv_filename=f'retro_cts.tsv',
                    )
            if self.as_tsv:
                return self.send_tsv()
            self.get_overall_stats()
            self.get_per_ws_stats()
            self.get_raw_counts()
    def send_tsv(self):
        # XXX Eventually, we'll want to download ph3 data as well.
        # XXX At that point, consider re-doing the tsv such that it includes
        # XXX a row for any drug with either a ph2 or ph3 trial id, and
        # XXX replacing the passed/failed boolean with the 4-valued status.
        # XXX Some of the prep for this already happens in get_raw_counts(),
        # XXX so factor out a single data retrieval function
        # bulk-load CT data
        from .utils import CTDataCache,CTData
        ct_cache = CTDataCache(
                wsa_ids=[x.wsa_id for x in self.all_molecules],
                phase='ph2',
                find_ct_evidence=True,
                )
        # Filter and further annotate the list.
        from browse.models import WsAnnotation
        from dtk.ct_predictions import label_molecules
        keep = []
        for lm in label_molecules(self.all_molecules,2,True):
            rmd = lm.mol
            # find clinical trial and completion date
            ctd = CTData(wsa_id = rmd.wsa_id,cache=ct_cache)
            if not ctd.ct_data:
                continue
            if not ctd.ct_data.trial_id:
                continue
            # pre-load wsa (which can't be lazy-loaded because rmd doesn't
            # have a bulk_fetch member)
            rmd.wsa = ctd.wsa
            # annotate RMD with passed and supported booleans and completion
            rmd.passed = 1 if lm.outcome == 'pass' else 0
            rmd.supported = 1 if lm.support == 'supported' else 0
            rmd.completion = ctd.completion
            keep.append(rmd)
        columns = [
                ('ws_id',lambda x:str(x.wsa.ws_id)),
                ('drugname',lambda x:str(x.wsa.get_name(False))),
                ('drug page', lambda x:self.request.build_absolute_uri(
                        x.wsa.drug_url()
                        )),
                ('ph2_link',lambda x:x.ph2_link),
                ('completion',lambda x:str(x.completion)),
                ('passed',lambda x:str(x.passed)),
                ('supported',lambda x:str(x.supported)),
                ]
        return self.tsv_response(columns,keep,self.tsv_filename)
    def make_select_form(self,data):
        from .utils import get_job_choices
        class MyForm(forms.Form):
            job_list = forms.MultipleChoiceField(
                    label='Input Jobs',
                    choices=get_job_choices(self.job_options),
                    required=True,
                    initial=self.job_list,
                    widget=forms.SelectMultiple(
                            attrs={'size':20}
                            ),
                    )
        return MyForm(data)
    def calc_post_valid(self):
        p = self.context['select_form'].cleaned_data
        p['job_list'] = ','.join(p['job_list'])
        return HttpResponseRedirect(self.here_url(**p))
    def get_molecule_lists(self):
        from .utils import get_molecule_lists
        self.molecule_lists = get_molecule_lists(self.job_options,self.job_list)
        self.all_molecules = []
        for l in self.molecule_lists.values():
            self.all_molecules += l
    def get_raw_counts(self):
        # XXX this re-does some work that happens elsewhere in the view,
        # XXX but efficiency is not an issue yet
        # bulk-load CT data
        from .utils import CTDataCache,CTData
        ct_caches = {
                2:CTDataCache(
                        wsa_ids=[x.wsa_id for x in self.all_molecules],
                        phase='ph2',
                        find_ct_evidence=False,
                        ),
                3:CTDataCache(
                        wsa_ids=[x.wsa_id for x in self.all_molecules],
                        phase='ph3',
                        find_ct_evidence=False,
                        ),
                }
        from dtk.ct_predictions import label_molecules
        from collections import Counter
        rows = []
        def filtered_counts(mols,cache):
            url_count = 0
            id_count = 0
            for lm in mols:
                ctd = CTData(wsa_id = lm.mol.wsa_id,cache=cache)
                if not ctd.ct_data:
                    continue
                url_count += 1
                if not ctd.ct_data.trial_id:
                    continue
                id_count += 1
            return (url_count,id_count)
        for job in self.job_options:
            if job.id not in self.job_list:
                continue
            cm_data = self.molecule_lists[job.id]
            for phase in (2,3):
                subset = label_molecules(cm_data,phase,True)
                passed = [x for x in subset if x.outcome=='pass']
                failed = [x for x in subset if x.outcome=='fail']
                cache = ct_caches[phase]
                url_passes,id_passes = filtered_counts(passed,cache)
                url_fails,id_fails = filtered_counts(failed,cache)
                rows.append((
                        job.ws_id,
                        phase,
                        len(cm_data),
                        (len(passed),len(failed)),
                        (url_passes,url_fails),
                        (id_passes,id_fails),
                        ))
        def p_f_format(data):
            return f'{data[0]} / {data[1]}'
        from dtk.table import Table
        self.output_list.append((
                'Raw blinded pass-fail counts',
                Table(rows,[
                Table.Column('WS ID',
                        idx=0,
                        ),
                Table.Column('Phase',
                        idx=1,
                        ),
                Table.Column('Total records',
                        idx=2,
                        ),
                Table.Column('Pass / Fail',
                        idx=3,
                        cell_fmt=p_f_format,
                        ),
                Table.Column('w/ URL',
                        idx=4,
                        cell_fmt=p_f_format,
                        ),
                Table.Column('w/ Trial ID',
                        idx=5,
                        cell_fmt=p_f_format,
                        ),
                ]),
                None,
                ))
    def yield_cases(self):
        from dtk.ct_predictions import BulkFetcher
        for phase in BulkFetcher.phases:
            for blinded in (True,False):
                yield (phase,blinded)
    def _calc_baseline_stats(self, pass_cnts, fail_cnts):
        total_pass = sum([v for k,v in pass_cnts.items() if k[1]=='sum'])
        total_fail = sum([v for k,v in fail_cnts.items() if k[1]=='sum'])
        total = total_pass+total_fail
        baseline_pass_rate = total_pass/total
        return (baseline_pass_rate, total)
    def _calc_supported_coverage(self, pass_cnts):
        supported = pass_cnts[('supported', 'sum')]
        total = sum([v for k,v in pass_cnts.items() if k[1]=='sum'])
        return supported/total
    def get_per_ws_stats(self):
        from dtk.ct_predictions import label_molecules,calc_subset_stats,format_stats_for_display,calc_enrich
        job_lookup = {x.id:x for x in self.job_options}
        rows = []
        for job_id in self.job_list:
            job = job_lookup[job_id]
            molecules = self.molecule_lists[job_id]
            for phase,blinded in self.yield_cases():
                labeled = label_molecules(molecules,phase,blinded)
                pass_cnts,fail_cnts,fracs = calc_subset_stats(labeled)
                fracs_w_cnts = format_stats_for_display(pass_cnts,
                                                        fail_cnts,
                                                        fracs
                                                        )
# think about adding an enrichment test for each of the support groups to see if passes are enirched in that groups vs all predictions
# will need to get the total number of predictions to make that happen though
                supd_OR, supd_p = calc_enrich(pass_cnts,fail_cnts)
                supd_n_unclr_OR, supd_and_unclr_p = calc_enrich(pass_cnts,
                                                                fail_cnts,
                                                                unclear='supported'
                                                               )
                if blinded:
                    # no need to do this 2x (bc it'd be the same), so just do it for the blinded group
                    baseline_stats = self._calc_baseline_stats(pass_cnts, fail_cnts)
                    supported_tp_coverage=self._calc_supported_coverage(pass_cnts)
                    rows.append((phase,
                                 job.ws_id,
                                 fracs_w_cnts,
                                 (supd_OR,supd_p),
                                 (supd_n_unclr_OR, supd_and_unclr_p),
                                 baseline_stats,
                                 supported_tp_coverage
                                ))
        rows.sort()
        self.build_median_ws_table_and_plots(rows)
        self.output_list.append((
                'Blinded per-workspace stats',
                build_retro_ct_table(rows,
                    second_col_title='WS ID'
                ),
                None,
                ))
    def _prep_fracs_for_plots(self, d):
        to_return = [(l,vals) for l,vals in d.items()]
        to_return.sort(key = lambda x: x[0])
        return to_return
    def build_median_ws_table_and_plots(self,rows):
        from collections import defaultdict
        from statistics import median
        # extract data
        # key is phase, then it's a list of disease-specific counter objects
        fracs_w_cnts = defaultdict(list)
        supd = defaultdict(list)
        supd_n_unclr = defaultdict(list)
        baseline = defaultdict(list)
        coverage = defaultdict(list)
        for r in rows:
            fracs_w_cnts[r[0]].append(r[2])
            supd[r[0]].append(r[3])
            supd_n_unclr[r[0]].append(r[4])
            baseline[r[0]].append(r[5])
            coverage[r[0]].append(r[6])
        #build table input
        rows=[]
        fracs_for_plots = []
        normalized_for_plots = []
        for phase in fracs_w_cnts:
            median_fracs={}
            median_supd = (median([x[0] for x in supd[phase]]),
                           median([x[1] for x in supd[phase]])
                           )
            median_supd_n_unclr = (median([x[0] for x in supd_n_unclr[phase]]),
                                   median([x[1] for x in supd_n_unclr[phase]])
                                  )
            per_label = {}
            # normalized is each frac, minus the baseline, and dividied by the baseline
            # i.e. how much better than baseline, as a ratio, is it
            normalized_per_label = {}
            for i,disease_d in enumerate(fracs_w_cnts[phase]):
                baseline_val = baseline[phase][i][0]
                for support_label in disease_d:
                    if support_label not in per_label:
                        per_label[support_label] = []
                        normalized_per_label[support_label] = []
                    per_label[support_label].append(disease_d[support_label][0])
                    normalized_per_label[support_label].append( (disease_d[support_label][0] - baseline_val) / baseline_val)
            # in some cases support labels are missing and assumed 0, need to address that here
            total = len(fracs_w_cnts[phase])
            for support_label in per_label:
                pl_cnt = len(per_label[support_label])
                if pl_cnt != total:
                    per_label[support_label] += [0]*(total-pl_cnt)
                    assert len(per_label[support_label]) == total
                median_fracs[support_label] = median(per_label[support_label])
            rows.append((phase,
                        median_fracs,
                        median_supd,
                        median_supd_n_unclr
                       ))

            list_of_vals = self._prep_fracs_for_plots(per_label)
            list_of_normd_vals = self._prep_fracs_for_plots(normalized_per_label)

            fracs_for_plots.append((phase,list_of_vals))
            normalized_for_plots.append((phase,list_of_normd_vals))

        self._build_plots(fracs_for_plots,supd,supd_n_unclr, normalized_for_plots, coverage)
        # build the actual table
        from dtk.table import Table
        self.output_list.append((
                'Median blinded per-workspace stats',
                Table(rows,[
                Table.Column('Phase',
                        idx=0,
                        ),
                Table.Column('CT % passed, supported',
                        extract=lambda x:x[1]['supported'],
                        cell_fmt=fmt_pct,
                        ),
                Table.Column('CT % passed, unsupported',
                        extract=lambda x:x[1]['unsupported'],
                        cell_fmt=fmt_pct,
                        ),
                Table.Column('% passed supported - % passed unsupported',
                        extract=lambda x:x[1]['supported'] - x[1]['unsupported'],
                        cell_fmt=fmt_pct,
                        ),
                Table.Column('OR passed in supported vs unsupported',
                        extract=lambda x:x[2],
                        cell_fmt=fmt_OR_w_p,
                        ),
                Table.Column('CT % passed, unclear',
                        extract=lambda x:x[1]['unclear'],
                        cell_fmt=fmt_pct,
                        ),
                Table.Column('OR passed in supported & unclear vs unsupported',
                        extract=lambda x:x[3],
                        cell_fmt=fmt_OR_w_p,
                        ),
                ]),
                None,
                ))
    def _build_plots(self, fracs_for_plots,supd,supd_n_unclr, normd_for_plots, coverage):
        self.plotly_plots=[]
        self._build_frac_boxplots(fracs_for_plots,
                                  'portion passed, by support type',
                                  'fracs',
                                  xaxis_range=[0.,1.]
                                 )
        self._build_frac_boxplots(normd_for_plots,
                                  'improvement over baseline as a ratio',
                                  'normd'
                                 )
        self._build_coverage_boxplot(coverage)
        self._build_enrichment_scatter_plots(supd,supd_n_unclr)
        self.context_alias(plotly_plots=self.plotly_plots)
    def _build_frac_boxplots(self, fracs_for_plots, title, type, xaxis_range=None):
        from dtk.plot import boxplot_stack
        for phase, label_vals_list in fracs_for_plots:
            self.plotly_plots.append((f'{type}{phase}',
                                      boxplot_stack([(vals,k)
                                                     for k,vals in label_vals_list
                                                    ],
                                                   height_per_dataset = 110,
                                                   title = f'Per WS, Blinded Ph{phase}, {title}',
                                                   xaxis_range = xaxis_range
                                                   )
                                    ))

    def _build_coverage_boxplot(self, coverage):
        from dtk.plot import boxplot_stack
        print(coverage)
        self.plotly_plots.append(('coverage',
                                  boxplot_stack([(vals,f'Ph. {phase}')
                                                 for phase,vals in coverage.items()
                                                ],
                                               height_per_dataset = 150,
                                               title = f'Per WS and Phase, portion of passed CTs that were supported',
                                               xaxis_range = [0.,1.]
                                               )
                                ))

    def _build_enrichment_scatter_plots(self, supd,supd_n_unclr):
        from dtk.plot import scatter2d
        from math import log
        titled_data = [('Supported vs unsupported',supd),
                       ('Supported & unclear vs unsupported', supd_n_unclr)
                      ]
        for title,d in titled_data:
            for phase in d:
                points = []
                for tup in d[phase]:
                    if tup[1] == 1:
                        points.append((tup[0],0.))
                    else:
                        points.append((tup[0],-1. * log(tup[1], 10)))
# may want to add a minimum range at some point, would take updating scatter2d
                self.plotly_plots.append((f'{phase}{title}',
                                      scatter2d('Odds Ratio',
                                                '-Log10(p)',
                                                points,
                                                refline = False,
                                                title = f'Per WS, Ph{phase} {title}'
                                               )
                                     ))
    def get_overall_stats(self):
        from dtk.ct_predictions import label_molecules,calc_subset_stats, format_stats_for_display,calc_enrich
        rows = []
        for phase,blinded in self.yield_cases():
            labeled = label_molecules(self.all_molecules,phase,blinded)
            pass_cnts,fail_cnts,fracs = calc_subset_stats(labeled)
            fracs_w_cnts = format_stats_for_display(pass_cnts,fail_cnts,fracs)
            supd_OR, supd_p = calc_enrich(pass_cnts,fail_cnts)
            supd_n_unclr_OR, supd_and_unclr_p = calc_enrich(pass_cnts, fail_cnts, unclear='supported')
            baseline_stats = self._calc_baseline_stats(pass_cnts, fail_cnts)
            supported_tp_coverage=self._calc_supported_coverage(pass_cnts)
            rows.append((phase,
                         blinded,
                         fracs_w_cnts,
                         (supd_OR, supd_p),
                         (supd_n_unclr_OR, supd_and_unclr_p),
                         baseline_stats,
                         supported_tp_coverage
                        ))
        self.output_list.append((
                'Combined stats across all molecules',
                build_retro_ct_table(rows),
                None,
                ))


fmt_pct = lambda x:f'{x*100:.1f}'
fmt_pct_w_cnt = lambda x:f'{x[0]*100:.1f} (N={x[1]})'
fmt_OR_w_p = lambda x:f'{x[0]:.1f} (p={x[1]:.2e})'

def build_retro_ct_table(rows, second_col_title='Blinded'):
    from dtk.table import Table
    return Table(rows,[
                Table.Column('Phase',
                        idx=0,
                        ),
                Table.Column(second_col_title,
                        idx=1,
                       ),
                Table.Column('Baseline % passed',
                        extract=lambda x:x[5],
                        cell_fmt=fmt_pct_w_cnt,
                        ),
                Table.Column('CT % passed, supported',
                        extract=lambda x:x[2].get('supported', (0,0)),
                        cell_fmt=fmt_pct_w_cnt,
                        ),
                Table.Column('CT % passed, unsupported',
                        extract=lambda x:x[2].get('unsupported', (0,0)),
                        cell_fmt=fmt_pct_w_cnt,
                        ),
                Table.Column('% passed supported - baseline % passed',
                        extract=lambda x:x[2].get('supported', (0,0))[0] - x[5][0],
                        cell_fmt=fmt_pct,
                        ),
                Table.Column('% passed supported - % passed unsupported',
                        extract=lambda x:x[2].get('supported', (0,0))[0] - x[2].get('unsupported', (0,0))[0],
                        cell_fmt=fmt_pct,
                        ),
                Table.Column('OR passed in supported vs unsupported',
                        extract=lambda x:x[3],
                        cell_fmt=fmt_OR_w_p,
                        ),
                Table.Column('% of passed CTs that were supported',
                        extract=lambda x:x[6],
                        cell_fmt=fmt_pct,
                        ),
                Table.Column('CT % passed, unclear',
                        extract=lambda x:x[2].get('unclear', (0, 0)),
                        cell_fmt=fmt_pct_w_cnt,
                        ),
                Table.Column('OR passed in supported & unclear vs unsupported',
                        extract=lambda x:x[4],
                        cell_fmt=fmt_OR_w_p,
                        ),
                ])
