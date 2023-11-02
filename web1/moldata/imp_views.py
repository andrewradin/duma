from dtk.duma_view import DumaView,list_of,boolean
from dtk.table import Table,SortHandler
from django import forms
from django.http import HttpResponseRedirect
import logging
logger = logging.getLogger(__name__)

# Create your views here.

# To add a new CM to the TrgImpView page, you need to do the following:
# 1. Add the CM to the dict in get_jid_name_map
#      The key is just used for this page.
#        The norm is to use three letters specific to the CM
#        plus '_jid' (for jobID)
#      The value is the name of the class in the
#        extract_importance file (see below for details)
# 2. Add a field in the make_calc_form.
#     The jid needs to be the same as in get_jid_name_map
#     Also add the appropriate label.
# 3. Create a class in extract_importance to actually do
#          the scoring.

def get_jid_name_map():
    return {'psm_jid' : 'path',
            'dpd_jid' : 'depend',
            'cds_jid' : 'codes',
            'esg_jid' : 'esga',
            'gph_jid' : 'gpath',
            'cap_jid' : 'capp',
            'dfs_jid' : 'defus'
           }
class TrgImpView(DumaView):
    template_name='moldata/trgimp.html'
    index_dropdown_stem='rvw:review'
    GET_parms={
            x:(list_of(int),None)
            for x in get_jid_name_map()
            }
    GET_parms['exclude'] = (list_of(str),[])
    GET_parms['prescreen_id'] = (int,None)
    button_map={
            'calc':['calc'],
            }
    # XXX Potential improvements:
    # XXX - skip inapplicable (score,method) combinations
    # XXX - don't run peel twice when scoring by both methods
    def make_calc_form(self, data):
        class MyForm(forms.Form):
            psm_jid = forms.IntegerField(
                    required=False,
                    label='PathSum JobID'
                 )
            dpd_jid = forms.IntegerField(
                    required=False,
                    label='DEEPEnD JobID',
                 )
            cds_jid = forms.IntegerField(
                    required=False,
                    label='CoDES JobID',
                 )
            esg_jid = forms.IntegerField(
                    required=False,
                    label='ESGA JobID',
                 )
            gph_jid = forms.IntegerField(
                    required=False,
                    label='gPath JobID',
                 )
            cap_jid = forms.IntegerField(
                    required=False,
                    label='CAPP JobID',
                 )
            dfs_jid = forms.IntegerField(
                    required=False,
                    label='DEFUS JobID',
                 )
        return MyForm(data)
    def calc_post_valid(self):
        p = self.context['calc_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))
    def custom_context(self):
        self._get_drug_name()
        self._setup()
        if self.are_no_jids():
            self.message("No Job IDs provided")
        else:
            self._get_label_dict()
            for k in ['peel_cumulative']:
                self.timelog('starting %s',k)
                self.get_imp_data(k)
                if k not in self.raw_data:
                    self.message("No data available")
                    break
                self.timelog('aggregating %s',k)
                self.aggregate_data(k)
                self.timelog('plotting %s',k)
                self.plot(k)
            self.timelog('calculations complete')
        self.context_alias(
                drug_name=self.drug_name,
                scores=list(self.scores),
                plotly_plots=self.plotly_plots
                )
    def are_no_jids(self):
        for x in self.jid_name_map:
            if getattr(self,x):
                return False
        return True
    def plot(self, k):
        from dtk.plot import PlotlyPlot
        ymin = 0.0
        ymax = 1.0
        dat = []
        box_data = {}
        max_label_len = 0
        for s in sorted(self.plot_d[k]):
            if not self.plot_d[k][s]['x']:
                continue
            label = condense_jr(s)
            max_label_len = max([max_label_len, len(label)])
            for x,y in zip(self.plot_d[k][s]['x'], self.plot_d[k][s]['y']):
                box_data.setdefault(x, []).append(y)
            dat.append(dict(type='bar',
                          x=self.plot_d[k][s]['x'],
                          y=self.plot_d[k][s]['y'],
                          name=label
                         )
                      )
            if min(self.plot_d[k][s]['y']) < ymin:
                ymin = min(self.plot_d[k][s]['y'])
            if max(self.plot_d[k][s]['y']) > ymax:
                ymax = max(self.plot_d[k][s]['y'])
        if dat:
            from dtk.data import merge_dicts
            base_layout ={'title':'Target importance - ' + k,
                           'yaxis':{'title':k+' importance score',
                                    'range':[ymin,ymax]
                                   }
                          }
            self.plotly_plots.append((k+'box',PlotlyPlot(
                                [dict(type='box',
                                        y = l,
                                        name = p,
                                        boxpoints = 'all',
                                        jitter = 0.5,
                                        boxmean = 'sd',
                                        marker = dict(size = 3, opacity = 0.5)
                                       )
                                 for p,l in box_data.items()
                                ],
                                base_layout
                             )))
            bar_plot_width = max([(100 +
                                    (len(self.plot_d[k])*12 + 50) * len(self.prots)
                                   ),
                                  500])
            # the height needed for each column / an approximation of the number of columns
            bot_margin = (5*len(self.plot_d[k])+20) / ((bar_plot_width - 100) / max_label_len/40.)
            print(bar_plot_width, bot_margin)
            self.plotly_plots.append((k,PlotlyPlot(
                                           dat,
                                           merge_dicts(base_layout,
                                                      {'barmode':'group',
                                                       'legend' : {'orientation' : "h",
                                                                   'x' : -0.2,
                                                                   'y' : ymin - (ymax-ymin)/15.
                                                                  },
                                                       'margin' : {'b' : bot_margin, 't' : 40},
                                                       'height' : 450 + bot_margin,
                                                       'width' : bar_plot_width
                                                      }

                                                      )
                             )))
        else:
            self.message('The jobIDs provided did not have usable data.')
    def _get_label_dict(self):
        from dtk.scores import get_sourcelist
        sl = get_sourcelist(
                self.ws,
                self.request.session,
                prescreen = self.wsa.marked_prescreen
                )
        self.label_d = {x.job_id():x.label() for x in sl.sources()}
    def aggregate_data(self, k):
        from dtk.html import decimal_cell
        from browse.models import Protein
        get_gene_list = Protein.make_gene_list_from_uniprot_list
        # first find all targets for which any significant data exists
        self.prots = set()
        for s in self.raw_data[k]:
            for x in self.raw_data[k][s]:
                if abs(self.raw_data[k][s][x]) > 0.01: # >1%
                    self.prots.add(x)
        # now build plot data, skipping any targets not included above
        self.plot_d[k] = {}
        for s in self.raw_data[k]:
            self.plot_d[k][s] = {'x':[], 'y':[]}
            for x in sorted(self.raw_data[k][s]):
                if x not in self.prots:
                    continue
                self.plot_d[k][s]['x'] += get_gene_list([x])
                self.plot_d[k][s]['y'].append(self.raw_data[k][s][x])
        self.scores = list(self.raw_data[k].keys())
        self.prots = list(self.prots)
    def get_imp_data(self, k):
        from dtk.target_importance import TrgImpJobResults, Cache
        from runner.process_info import JobInfo
        compute_cache = Cache()
        for id_type in self.jid_name_map:
            jids = getattr(self, id_type)
            if jids is None:
                continue
            for jid in jids:
                self.timelog('starting %d %s',jid,id_type)
                try:
                    tijr = TrgImpJobResults(
                            ws_id=self.ws.id,
                            wsa_id=self.wsa.id,
                            job_or_id=jid,
                            key_exclude=self.exclude,
                            )
                    d = tijr.get_importance(k, compute_cache=compute_cache)
                except (
                        JobInfo.AmbiguousKeyError,
                        AssertionError,
                        IOError,
                        ValueError,
                        KeyError,
                        ) as ex:
                    self.message('skipping %s %d: %s %s'%(
                            id_type,
                            jid,
                            ex.__class__.__name__,
                            str(ex),
                            ))
                    # don't repeat error for other methods
                    jids.remove(jid)
                    continue
                bji=JobInfo.get_bound(self.ws,jid)
                prefix = self.label_d.get(jid,bji.role_label()) + " "
                if k not in self.raw_data:
                    self.raw_data[k] = {}
                self.raw_data[k].update({
                        prefix+score:results
                        for score,results in d.items()
                        })
    def _setup(self):
        self.plot_d = {}
        self.plotly_plots = []
        self.raw_data = {}
        self.scores = set()
        self.jid_name_map = get_jid_name_map()
        if self.prescreen_id and self.are_no_jids():
            # if no job ids are specified, but a prescreen_id is, set all
            # the job_id attributes as if they were set based on the inputs
            # to the prescreen
            from browse.models import Prescreen
            pscr = Prescreen.objects.get(pk=self.prescreen_id)
            from runner.process_info import JobInfo
            bji = JobInfo.get_bound(self.ws,pscr.primary_job_id())
            from .utils import TrgImpParmBuilder
            tipb = TrgImpParmBuilder()
            tipb.build_from_downstream_bji(bji)
            tipb.extract_as_attributes(self)
    def _get_drug_name(self):
        from browse.models import WsAnnotation
        self.drug_name = self.wsa.get_name(self.is_demo())
def condense_jr(s):
    s = s.replace("Case/Control", "C/C")
    s = s.replace("OpenTargets", "OTarg")
    s = s.replace("SigDiffuser", "SigDif")
    s = s.replace("codesDir", "direction")
    s = s.replace("codesMax", "score")
    s = s.replace("Known Drug", "KT")
    s = s.replace("Literature", "Lit.")
    return s


def convert_jid_to_role(k, ws_id):
    from runner.process_info import JobInfo
    parts = k.split('_')
    job_id = int(parts[0].lstrip('j'))
    bji = JobInfo.get_bound(ws_id,job_id)
    return '_'.join([bji.job.role]+parts[1:])

class ScrImpView(DumaView):
    template_name='moldata/scrimp.html'
    index_dropdown_stem='rvw:review'
    GET_parms={
           'wzs_jid':(int, None),
            }
    button_map={
            'calc':['calc'],
            }
    def make_calc_form(self, data):
        class MyForm(forms.Form):
            wzs_jid = forms.ChoiceField(
                    label='WZS JobID',
                    choices = self.ws.get_prev_job_choices('wzs'),
                    required=True,
                 )
        return MyForm(data)
    def calc_post_valid(self):
        p = self.context['calc_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))
    def custom_context(self):
        self._get_drug_name()
        if not self.wzs_jid:
            self.message("No Job IDs provided")
        else:
            try:
                self._setup()
                self._process()
            except Exception as ex:
                # raise # uncomment for debugging
                self.message(
                    "Can't extract data from that WZS job; got %s %s"
                            %(ex.__class__.__name__,ex)
                    )
            else:
                self._plot_setup()
                self._plot()
                self.context_alias(
                    drug_name=self.drug_name,
                    plotly_plots=self.plotly_plots
                    )
    def _get_drug_name(self):
        from browse.models import WsAnnotation
        self.drug_name = self.wsa.get_name(self.is_demo())
    def _setup(self):
        self._load_wzs_settings()
    def _process(self):
        from dtk.score_importance import ScoreImportance
        si = ScoreImportance(self.ws.id, self.wzs_jid)
        w,n,f = si.get_score_importance(self.wsa.id)
        self.wts = w
        self.just_normd = n
        self.final_wts = f
    def _load_wzs_settings(self):
        from runner.process_info import JobInfo
        self.wzs_bji = JobInfo.get_bound(self.ws,self.wzs_jid)
        self.wzs_settings = self.wzs_bji.job.settings()
    def _plot_setup(self):
        from dtk.plot import Color
        color_list = Color.plotly_defaults
        assert set(self.wts.keys()) == set(self.final_wts.keys())
        # because there are multiple job roles from GWAS data
        gwas_color = color_list[1]
        external_color = color_list[4]
        pheno_color = color_list[5]
        self.source_colors = [
            ('cc', color_list[0]),
            ('gpath', gwas_color),
            ('esga', gwas_color),
            ('gwasig', gwas_color),
            ('otarg', color_list[2]),
            ('faers', color_list[3]),
            ('agr', external_color),
            ('dgn', external_color),
            ('misig', pheno_color),
            ('mips', pheno_color),
            ('tcgamut', color_list[6]),
            ('mirna', color_list[7]),
            ('other', color_list[8]),
        ]
        self.source_types = [x[0] for x in self.source_colors]
        self.default = self._process_wts(self.wts)
        self.normd_drug = self._process_wts(self.just_normd)
        self.wtd_drug = self._process_wts(self.final_wts)
    def _plot(self):
        l = []
        for k,color in self.source_colors:
            if k not in self.default and k not in self.normd_drug:
                continue
            assert len(self.default[k]) == len(self.normd_drug[k])
            for i in range(len(self.default[k])):
                assert self.default[k][i][0] == self.normd_drug[k][i][0]
                l.append((
                           self.default[k][i][0],
                           [self.default[k][i][1],
                            self.normd_drug[k][i][1],
                            self.wtd_drug[k][i][1]
                           ],
                           color,
                           self.default[k][i][2],
                         ))
        self.plotly_plots = []
        self._plot_bar(l,
                       str(self.wzs_jid),
                       ['WZS weights',
                        self.drug_name + ' normalized',
                        self.drug_name + ' weighted'
                       ],
                       title = 'Relative score importance'
                      )
    def _plot_bar(self, l, name, xs, title=''):
        from dtk.plot import PlotlyPlot
        dat = []
        seen = set()
        for tup in l:
            dat.append(dict(type='bar',
                           x = xs,
                           y = tup[1],
                           text = ["<br>".join([tup[0],"%0.4f" % tup[1][i]])
                                   if tup[1][i] > 0 else None
                                   for i in range(len(tup[1]))
                                  ], # a bunch of 0 weight scores were cluttering the hover
                           legendgroup = tup[3],
                           name = tup[3],
                           showlegend = tup[3] not in seen,
                           hoverinfo = 'text',
                           marker = dict(color = tup[2],
                                         line = dict(
                                                     color = 'white',
                                                     width = 0.5
                                                )
                                    )
                           ))
            if tup[1]:
                seen.add(tup[3])
        self.plotly_plots.append((name,
               PlotlyPlot(dat,
                          {
                           'width' : 400 + 200*len(xs),
                           'height' : 700,
                           'title' : title,
                           'barmode':'relative',
                           'yaxis':{'hoverformat':'.4f'},
                           # Add some margin at the bottom so we can see
                           # pathway labels.
                           # TODO: When upgrade plotly, we can use automargin.
                           'margin':{'b': 150}
                          }
                         )))
    def _process_wts(self, d, report=False):
        denom = sum([abs(v) for v in d.values()])
        normd = {}
        for k,v in d.items():
            source_type = self._find_source_type(k)
            if source_type is None:
                if report:
                    self.message("Ignoring unexpected role '%s'"%k)
                continue
            if source_type[0] not in normd:
                normd[source_type[0]] = []
            val = v/denom if denom else 0
            normd[source_type[0]].append((k,val,source_type[1]))
        return {k:sorted(l, key=lambda x: x[0]) for k,l in normd.items()}
    def _find_source_type(self, k):
        if k.startswith('j') and k[1].isdigit(): # this suggests the job ID is being used
            k = convert_jid_to_role(k, self.ws.id)
        from dtk.score_source import ScoreSource
        return ScoreSource(k, self.ws.get_cds_choices()).source_type()

class TrgScrImpView(ScrImpView):
    template_name='moldata/trg_scr_imp.html'
    index_dropdown_stem='rvw:review'
    GET_parms={'wzs_jid':(int, None),
               'method':(str,None)
              }
    button_map={
            'calc':['calc'],
            }
    def make_calc_form(self, data):
        from dtk.target_importance import METHOD_CHOICES
        class MyForm(forms.Form):
            wzs_jid = forms.ChoiceField(
                    label='WZS JobID',
                    choices = self.ws.get_prev_job_choices('wzs'),
                    required=True,
                    initial=self.wzs_jid,
                 )
            method = forms.ChoiceField(
                    label='Target importance scoring method',
                    choices = METHOD_CHOICES,
                    required=True,
                    initial=self.method,
                 )
        return MyForm(data)
    def custom_context(self):
        self._get_drug_name()
        if not self.wzs_jid:
            self.message("No Job IDs provided")
        else:
            try:
                self._setup()
                self._process()
            except Exception as ex:
                import traceback as tb
                tb.print_exc()
                # raise # uncomment for debugging
                self.message(
                    "Can't extract data from that WZS job; got %s %s"
                            %(ex.__class__.__name__,ex)
                    )
            else:
                self._prep_trg_data()
                self._plot_setup()
                self._plot()
                self.context_alias(
                    drug_name=self.drug_name,
                    plotly_plots=self.plotly_plots
                    )
    def _prep_trg_data(self):
        self._get_jids()
        self.trg_data = {}
        self._prot_imp_data, self._pathway_imp_data = self._load_trg_imp()
    def _separate_trgs(self, trg_data):
        sep_trg_data = {}
        for j,d in trg_data.items():
            for t,v in d.items():
                if t not in sep_trg_data:
                    sep_trg_data[t] = {}
                k = j.replace(' ', '_')
                sep_trg_data[t][k.lower()] = v
        return sep_trg_data

    def _get_jids(self):
        from dtk.target_importance import get_wzs_jids, filter_accepted_jobs
        self.jids = get_wzs_jids(self.ws.id, self.wzs_settings)
        self.jids = filter_accepted_jobs(self.jids)

    def _load_trg_imp(self):
        from dtk.target_importance import TrgImpJobResults, Cache
        prot_imp_data = {}
        pathway_imp_data = {}
        compute_cache = Cache()
        from runner.process_info import JobInfo
        for jid, jr in self.jids.items():
            self.timelog('starting %d %s',jid,jr)
            try:
                tijr = TrgImpJobResults(
                        ws_id=self.ws.id,
                        wsa_id=self.wsa.id,
                        job_or_id=jid,
                        name=jr,
                        )
                prots = tijr.get_importance(self.method, data_type='prots', compute_cache=compute_cache)
                pathways = tijr.get_importance(self.method, data_type='pathways', compute_cache=compute_cache)
            except (
                    JobInfo.AmbiguousKeyError,
                    AssertionError,
                    IOError,
                    ValueError,
                    KeyError,
                    ) as ex:
                import traceback
                traceback.print_exc()
                self.message('skipping %s %d: %s %s'%(
                        jr,
                        jid,
                        ex.__class__.__name__,
                        str(ex),
                        ))
                continue
            prefix = jr + " "
            prot_imp_data.update({
                    prefix+score:results
                    for score,results in prots.items()
                    })
            pathway_imp_data.update({
                    prefix+score:results
                    for score,results in pathways.items()
                    })
        return prot_imp_data, pathway_imp_data
    def _check_wtd_drug_keys(self):
        for k, l in self.wtd_drug.items():
            for i,x in enumerate(l):
                # this suggests the job ID is being used
                if x[0].startswith('j') and x[0][1].isdigit():
                    new = convert_jid_to_role(x[0], self.ws.id)
                else:
                    new = x[0]
                l[i] = [new.lower()]+list(x)[1:]

    MISSING = 'unattributed'
    def _plot(self):
        self.plotly_plots = []
        self.context['tables'] = []
        self._plot_imp(self.MISSING, self._prot_imp_data, 'target')
        # Thresholding is problematic, the "Other" bar it creates is huge and also causes
        # problems with renaming table columns.  Fixable, but just dont' use it for now.
        self._plot_imp('NonPathway or Unattributed', self._pathway_imp_data, 'pathway', threshold=None)

    def _aggregate_trg_data(self, trg_data, fillin_name=None):
        """
        Target Importance, (targ X job_role), each jr sums to 1 (minus missing)
              otarg_codesmax  cc_path_dir   faers_depend
        P001  0.8             0.7           0.1
        Q021  0.2             0.3           0.9
        sep_trg_data stores this as a dict of targ -> jr -> weight

        Score Importance, [job_role], sums to 1
        otarg_codemax  cc_path_dir   faers_depend
        0.5            0.2           0.3

        trg_scr_imp is trg_imp * scr_imp(row-extended)

        Here we are computing trg_scr_imp, and filling in missing weight.
        Then we return this data grouped by target.
        """
        fillin_name = fillin_name or self.MISSING
        # Returns {trg: {total: x, vals: [(name, x), ...]}, ...}
        # Also adds in the 'fillin_name' target for any unaccount weight
        sep_trg_data = self._separate_trgs(trg_data)
        if fillin_name not in sep_trg_data:
            print('Adding the fill-in name to the score importance data:', fillin_name)
            sep_trg_data[fillin_name] = {}
        sep_trg_data_keys = list(sep_trg_data.keys())
        fillin_idx = sep_trg_data_keys.index(fillin_name)

        totals = [0.0]*len(sep_trg_data_keys)
        # An array for each targ, containing the contributions to it from
        # jobs
        details = [[] for _ in range(len(sep_trg_data_keys))]
        for score_type, job_weights in self.wtd_drug.items():
            for job_role, job_weight, source_type in job_weights:
                vals = [[trg, job_weight * sep_trg_data[trg].get(job_role, 0)]
                         for trg in sep_trg_data_keys]

                missing_val = job_weight - sum([x[1] for x in vals])
                # there was some float rounding that was leading to unecessary reporting
                if missing_val > 0.00009:
                    print(job_role, 'unaccounted for:', missing_val, [(k, v) for k, v in vals if v != 0])
                    vals[fillin_idx][1] += missing_val

                for i, (prot, val) in enumerate(vals):
                    if val > 0:
                        if prot == 'missing':
                            # If there was a hardcoded 'missing' prot, reassign it to the fillin_idx
                            vals[fillin_idx][1] += val
                            vals[i][1] = 0
                        else:
                            details[i].append([job_role, val, source_type, score_type])

                totals = [x + y[1] for x, y in zip(totals, vals)]
        return {p: {'total': totals[i], 'details': details[i]}
                for i, p in enumerate(sep_trg_data_keys) if totals[i] > 0}

    def _apply_threshold(self, agg_trg_data, threshold):
        squish = []
        keep = []
        for trg, trg_data in agg_trg_data.items():
            if trg_data['total'] < threshold:
                if trg_data['total'] > 0:
                    squish.append((trg, trg_data))
            else:
                keep.append((trg, trg_data))

        # Anything we're not keeping is going to go into its own bucket.
        # We need to make sure that there's a unique name for each one,
        # so add _trg to the end of each of the job role names in there.
        for trg, to_squish in squish:
            for details in to_squish['details']:
                details[0] += '_' + trg

        squish_total = sum([v['total'] for k, v in squish])
        squish_details = [x for k, v in squish for x in v['details']]

        if squish_total > 0:
            keep.append(('Other', {'total': squish_total, 'details': squish_details}))

        return dict(keep)

    def _make_bar_data(self, agg_trg_data):
        """
        Returns a list, with one element per job_role.
        Each entry contains a list of values, one element per target (in order)
        """
        jr_to_prots = {}

        data = sorted(agg_trg_data.items(),
                      key=lambda x: (x[0] == 'Other', -x[1]['total']))
        MAX_BARS=100
        data = data[:MAX_BARS]
        job_type_to_color = dict(self.source_colors)
        ordered_trgs = []
        unknown_job_types = set()
        for i, (trg, trg_data) in enumerate(data):
            ordered_trgs.append(trg)
            for job_role, val, source_type, job_type in trg_data['details']:
                if not job_role in jr_to_prots:
                    if job_type.lower() not in job_type_to_color:
                        unknown_job_types.add(job_type)
                        color = list(job_type_to_color.values())[0]
                    else:
                        color = job_type_to_color[job_type.lower()]
                    jr_to_prots[job_role] = [
                            job_role,
                            [0] * len(data),
                            color,
                            source_type
                            ]

                jr_to_prots[job_role][1][i] = val
        if unknown_job_types:
            self.message("Unknown job types: %s" % unknown_job_types)
        return jr_to_prots, ordered_trgs

    def _reactome_id_to_name(self):
        from dtk.gene_sets import get_pathway_id_name_map
        return get_pathway_id_name_map()


    def _plot_imp(self, fillin_name, trg_data, score_type, threshold=None):
        self._check_wtd_drug_keys()
        from browse.models import Protein
        agg_trg_data = self._aggregate_trg_data(trg_data, fillin_name)
        if threshold:
            unthresholded = agg_trg_data.copy()
            agg_trg_data = self._apply_threshold(agg_trg_data, threshold)
        else:
            unthresholded = agg_trg_data

        self._rct_id_to_name = self._reactome_id_to_name()

        bar_data, targs = self._make_bar_data(agg_trg_data)
        uniprot2gene_map = Protein.get_uniprot_gene_map(targs)
        self._add_table(unthresholded, score_type, uniprot2gene_map)
        names = [uniprot2gene_map.get(u, u) for u in targs]
        # Pathway names are way too long, try to make them fit better.
        names = [x.replace('REACTOME_', 'R_') for x in names]

        names = [self._rct_id_to_name.get(x, x) for x in names]

        title = 'Relative contribution per %s per score' % score_type
        # Sort bars by color
        sorted_bars = sorted(list(bar_data.values()), key=lambda x: x[2])
        self._plot_bar(sorted_bars,
                       str(self.wzs_jid) + "_" + score_type,
                       names,
                       title = title
                      )

    def _add_table(self, agg_trg_data, score_type, uniprot2gene_map):
        from dtk.url import pathway_url_factory
        from dtk.job_prefix import jr_interpreter
        pathway_url = pathway_url_factory()
        def fmt_target(x):
            # There are some results with the word 'missing' baked in - they seem
            # to mostly be 0, so not going to bother rewriting them, but they will
            # crash here if we try to do a protein lookup for them.
            if x == self.MISSING or x == 'missing':
                return x
            from dtk.html import link
            if score_type == 'target':
                gene = uniprot2gene_map.get(x, x)
                prot_url = self.ws.reverse('protein',x)
                return link(gene, prot_url)
            elif score_type == 'pathway':
                if x.startswith('REACTOME'):
                    return link(x,pathway_url(x, init_wsa=self.wsa.id))
                else:
                    id = x
                    name = self._rct_id_to_name.get(id, id)
                    return link(name, pathway_url(id, init_wsa=self.wsa.id))
            else:
                raise Exception("Unhandled score type " + score_type)


        cols = [
                Table.Column('Target', idx='target', cell_fmt=fmt_target),
                Table.Column('Total',
                    idx='total',
                    cell_fmt=lambda x:"%0.3f"%x,
                    )
                ]
        rows = []
        data = sorted(agg_trg_data.items(), key=lambda x: -x[1]['total'])
        MAX_ROWS = 100
        data = data[:MAX_ROWS]
        col_tots={}
        for _, trg_data in data:
            for jr, val, _, _ in trg_data['details']:
                if jr not in col_tots:
                    col_tots[jr]=0.
                col_tots[jr]+=val
        cols += [Table.Column(jr_interpreter(jr), idx=jr,
                 cell_fmt=lambda x:"%.3g"%x)
                 for jr,_ in sorted(col_tots.items(),
                                    key=lambda x: x[1],
                                    reverse=True)
                ]
        for trg, trg_data in data:
            from collections import defaultdict
            row_data = defaultdict(float)
            row_data.update({
                'target': trg,
                'total': trg_data['total']
                })

            for jr, val, _, _ in trg_data['details']:
                row_data[jr] = val
            rows.append(row_data)

        self.context['tables'].append(Table(rows, cols))

class IndTrgImpView(DumaView):
    template_name='moldata/ind_trg_imp.html'
    index_dropdown_stem='rvw:review'
    GET_parms={'wzs_jid':(int, None),
               'method':(str,None),
               'target':(str,None),
               'use_cache':(boolean,True),
              }
    button_map={
            'calc':['calc'],
            }
    def make_calc_form(self, data):
        from dtk.target_importance import METHOD_CHOICES
        class MyForm(forms.Form):
            wzs_jid = forms.ChoiceField(
                    label='WZS JobID',
                    choices = self.ws.get_prev_job_choices('wzs'),
                    required=True,
                    initial=self.wzs_jid,
                 )
            method = forms.ChoiceField(
                    label='Target importance scoring method',
                    choices = METHOD_CHOICES,
                    required=True,
                    initial=self.method,
                 )
            target = forms.ChoiceField(
                    label='Target',
                    choices = self._get_targets(),
                    initial=self.target,
                )
            use_cache = forms.BooleanField(
                    label='Use Cache',
                    required=False,
                    initial=self.use_cache,
                )
        return MyForm(data)

    def calc_post_valid(self):
        p = self.context['calc_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))

    def custom_setup(self):
        from browse.models import Protein
        self.prot2gene = Protein.get_uniprot_gene_map()

    def custom_context(self):
        self._get_drug_name()
        if not self.wzs_jid:
            self.message("No Job IDs provided")
            return
        if not self.target:
            self.message("Select a target")
            return

        self._load_wzs_settings()
        jids = self._get_jids()
        self._load_imp(jids)
        self._make_tables()
        #self._plot()
        self.context_alias(
            drug_name=self.drug_name,
            jids=jids,
            #plotly_plots=self.plotly_plots
            )

    def _get_targets(self):
        from dtk.target_importance import get_all_targets
        self.targets = get_all_targets(self.ws.id, [self.wsa.id])
        from browse.models import Protein
        return [(x, "%s (%s)" % (self.prot2gene.get(x, x), x)) for x in self.targets]

    def _load_wzs_settings(self):
        from runner.process_info import JobInfo
        self.wzs_bji = JobInfo.get_bound(self.ws,self.wzs_jid)
        self.wzs_settings = self.wzs_bji.job.settings()

    def _get_drug_name(self):
        from browse.models import WsAnnotation
        self.drug_name = self.wsa.get_name(self.is_demo())


    def _get_jids(self):
        from dtk.target_importance import get_wzs_jids, find_indirect_jobs
        jids = get_wzs_jids(self.ws.id, self.wzs_settings)
        jids = find_indirect_jobs(self.ws.id, jids)
        return jids

    def _load_imp(self, jids):
        from dtk.target_importance import TrgImpJobResults, Cache
        compute_cache = Cache()
        self.prot_imp_data = {}
        self.sig_data = {}
        prot_imp_data = {}
        from runner.process_info import JobInfo
        for jid, jr in jids.items():
            self.timelog('starting %d %s',jid,jr)
            try:
                print("Loading trgimp data")
                tijr = TrgImpJobResults(
                        ws_id=self.ws.id,
                        target_id=self.target,
                        job_or_id=jid,
                        name=jr,
                        skip_cache=not self.use_cache,
                        )
                prots = tijr.get_importance(self.method, data_type='prots',
                                            cache_only=self.use_cache,
                                            compute_cache=compute_cache)
                if not prots:
                    raise IOError("%s not available in cache" % jr)

                print("Loading disease sig")
                bji = JobInfo.get_bound(self.ws, jid)
                ip_code = bji.parms['input_score']
                from dtk.scores import JobCodeSelector
                cat = JobCodeSelector.get_catalog(self.ws.id,ip_code)
                ordering = cat.get_ordering(ip_code,True)
                self.sig_data[jid] = {
                        x[0]: float(x[1]) for x in ordering
                        if float(x[1]) > 0
                        }
                print("Done")

            except (
                    JobInfo.AmbiguousKeyError,
                    AssertionError,
                    IOError,
                    ValueError,
                    KeyError,
                    ) as ex:
                import traceback
                traceback.print_exc()
                self.message('skipping %s %d: %s %s'%(
                        jr,
                        jid,
                        ex.__class__.__name__,
                        str(ex),
                        ))
                continue
            prefix = jr + " "
            prot_imp_data.update({
                    prefix+score:(results, jid)
                    for score,results in prots.items()
                    })
        self.prot_imp_data = prot_imp_data
    def _make_tables(self):
        print("Making tables")
        self.context['tables'] = []
        for title, (results, jid) in self.prot_imp_data.items():
            self._add_table("%s (%s)" % (title, jid), results, jid)
        print("Done")

    def _add_table(self, title, results, jid):
        from dtk.html import link
        cols = [
                Table.Column('Prot', idx='prot',
                    cell_fmt=lambda x: link(x, self.ws.reverse('protein',x)),
                    ),
                Table.Column('Gene', idx='gene'),
                Table.Column('Importance',
                    idx='imp',
                    cell_fmt=lambda x:"%0.3f"%x,
                    ),
                Table.Column('Input Weight',
                    idx='sig',
                    cell_fmt=lambda x:"%0.3g"%x,
                    ),
                Table.Column('Input Weight (norm)',
                    idx='signorm',
                    cell_fmt=lambda x:"%0.1f%%"%x,
                    ),
                Table.Column('Input Rank (out of %d)' % len(results),
                    idx='sigrank',
                    cell_fmt=lambda x:"%d"%(x+1), # Make it 1-indexed
                    ),
                ]
        rows = []
        prot2gene = self.prot2gene
        srt_results = sorted(results.items(), key=lambda x: -x[1])
        sig = self.sig_data[jid]

        input_rank = [(prot, weight) for prot, weight in sig.items()
                      if prot in results]
        input_rank = enumerate(sorted(input_rank, key=lambda x: -x[1]))
        input_rank = {prot_val[0]: rank for rank, prot_val in input_rank}
        print("%s Sig says there are %d things, results say %d" % (title, len(sig), len(srt_results)))

        import numpy as np
        norm = np.sum(list(sig.values()))
        rows = [{
            'prot': prot,
            'imp': imp,
            'gene': prot2gene.get(prot, prot),
            'sig': sig[prot],
            'signorm': 100*sig[prot]/norm,
            'sigrank': input_rank[prot],
            } for prot, imp in srt_results]
        self.context['tables'].append((title, Table(rows, cols)))

