from dtk.duma_view import DumaView,list_of,boolean
from dtk.table import Table,SortHandler
from django import forms
from django.http import HttpResponseRedirect

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse

import logging

logger = logging.getLogger(__name__)


# Create your views here.

# options for x and y axis data
# name, label, extract function
# we use name rather than an index to make the URL more readable
pair_plot_data = [
        ('add','Add Boost',lambda x:x.add_boosts),
        ('mult','Mult Boost',lambda x:x.mult_boosts),
        ('base','Max Base Metric',lambda x:x.pair_base),
        ('f1met','Feature 1 Metric',lambda x:x.pair_f1_metric),
        ('f2met','Feature 2 Metric',lambda x:x.pair_f2_metric),
        ('hitov','Hit Overlap',lambda x:x.pair_hit_overlap),
        ('corr','Overall Correlation',lambda x:x.pair_corr),
        ('hitcorr','Hit Correlation',lambda x:x.pair_hit_corr),
        ('hitexcorr','Excess Hit Correlation',lambda x:x.pair_ex_corr),
        ]
def pair_plot_data_lookup(name):
    idx = [t[0] for t in pair_plot_data].index(name)
    return pair_plot_data[idx]

class FeaturePairHeatView(DumaView):
    template_name= 'score/shared_single_plot.html'
    GET_parms = {
            'fm_code':(str,None),
            'data_src':(str,None),
            }
    button_map={
            'display':['option'],
            }
    def custom_context(self):
        self.context['headline'] = "Feature Pair Heatmap"
        if self.fm_code and self.data_src:
            fm = self.ws.get_feature_matrix(self.fm_code)
            from dtk.score_pairs import FeaturePairs
            self.pairs = FeaturePairs.from_fm(fm)
            data_info = pair_plot_data_lookup(self.data_src)
            import numpy as np
            n = len(self.pairs.feature_names)
            grid = np.zeros((n,n))
            data = data_info[2](self.pairs)
            for v,(i1,i2) in zip(data,self.pairs.pairs):
                grid[i1,i2] = v
                grid[i2,i1] = v
            from dtk.plot import plotly_heatmap
            self.context['plot'] = plotly_heatmap(
                    grid,
                    col_labels=self.pairs.feature_names,
                    row_labels=self.pairs.feature_names,
                    color_zero_centered=True,
                    )
    def make_option_form(self,data):
        class MyForm(forms.Form):
            fm_code = forms.ChoiceField(
                label = 'Feature Matrix',
                choices = self.ws.get_feature_matrix_choices(),
                initial = self.fm_code,
                )
            data_src = forms.ChoiceField(
                label = 'Data Source',
                choices = [t[:2] for t in pair_plot_data],
                initial = self.data_src,
                )
        return MyForm(data)
    def display_post_valid(self):
        p = self.option_form.cleaned_data
        return HttpResponseRedirect(self.here_url(**p))

class FeaturePairView(DumaView):
    template_name= 'score/shared_single_plot.html'
    GET_parms = {
            'fm_code':(str,None),
            'x':(str,None),
            'y':(str,None),
            'highlight1':(str,None),
            'highlight2':(str,None),
            }
    button_map={
            'display':['option'],
            }
    def custom_setup(self):
        if self.fm_code:
            self.fm = self.ws.get_feature_matrix(self.fm_code)
    def custom_context(self):
        self.context['headline'] = "Feature Pairs"
        if self.fm_code and self.x and self.y:
            from dtk.score_pairs import FeaturePairs
            self.pairs = FeaturePairs.from_fm(self.fm)
            x_data = pair_plot_data_lookup(self.x)
            y_data = pair_plot_data_lookup(self.y)
            self.context['plot'] = self.make_plot(
                    x_data[1],x_data[2](self.pairs),
                    y_data[1],y_data[2](self.pairs),
                    [x.replace(' ','<br>') for x in self.pairs.pair_labels],
                    )
    def make_plot(self,x_label,x_vals,y_label,y_vals,point_names):
        points = zip(x_vals,y_vals)
        kwargs = dict(
                refline = False,
                text = point_names,
                ids = ('synergypage',[
                        (self.pairs.sources[i1],self.pairs.sources[i2])
                        for i1,i2 in self.pairs.pairs
                        ])
                )
        from dtk.plot import scatter2d,Color
        if self.highlight1 or self.highlight2:
            kwargs['classes'] = [
                    ('other',{'color':Color.default_light}),
                    ('feat1',{'color':Color.highlight2}),
                    ('feat2',{'color':Color.highlight4}),
                    ('both',{'color':Color.highlight}),
                    ]
            def hl_to_idx(highlight):
                if not highlight:
                    return None
                return self.pairs.sources.index(highlight)
            i1_target = hl_to_idx(self.highlight1)
            i2_target = hl_to_idx(self.highlight2)
            kwargs['class_idx'] = [
                    1*(i1 is i1_target)+2*(i2 is i2_target)
                    for i1,i2 in self.pairs.pairs
                    ]
        return scatter2d(x_label,y_label,points,**kwargs)
    def make_option_form(self,data):
        highlight_choices = [('','None')]
        if self.fm_code:
            highlight_choices += sorted(zip(
                    self.fm.spec.get_codes(),
                    self.fm.feature_names,
                    ),key=lambda x:x[1])
        class MyForm(forms.Form):
            fm_code = forms.ChoiceField(
                label = 'Feature Matrix',
                choices = self.ws.get_feature_matrix_choices(),
                initial = self.fm_code,
                )
            x = forms.ChoiceField(
                label = 'Plot X Data',
                choices = [t[:2] for t in pair_plot_data],
                initial = self.x,
                )
            y = forms.ChoiceField(
                label = 'Plot Y Data',
                choices = [t[:2] for t in pair_plot_data],
                initial = self.y,
                )
            highlight1 = forms.ChoiceField(
                label = 'Highlight 1',
                choices = highlight_choices,
                initial = self.highlight1,
                required=False,
                )
            highlight2 = forms.ChoiceField(
                label = 'Highlight 2',
                choices = highlight_choices,
                initial = self.highlight2,
                required=False,
                )
        return MyForm(data)
    def display_post_valid(self):
        p = self.option_form.cleaned_data
        return HttpResponseRedirect(self.here_url(**p))

# Later:
# - norming options? (none (or max->1), sc, sclog, some flooring?)
# - factor out plotly interface?
# - add some stats?
# - add optimum weight calculator and indication
class ScoreSynergyView(DumaView):
    template_name= 'score/score_synergy.html'
    GET_parms = {
            'x':(str,None),
            'y':(str,None),
            'ds':(str,None),
            }
    def make_score_fetcher(self,global_code):
        from dtk.score_calibration import ScoreFetcher
        sf = ScoreFetcher.from_global_code(global_code,self.ws)
        sf.sc = self.sc
        return sf
    def custom_context(self):
        self.context['headline'] = "Score Synergy"
        if not self.x or not self.y:
            self.message('URL parameters missing')
            return
        if not self.ds:
            self.ds = self.ws.eval_drugset
        from dtk.score_calibration import ScoreCalibrator
        self.sc = ScoreCalibrator()
        self.kts = self.ws.get_wsa_id_set(self.ds)
        self.x_f = self.make_score_fetcher(self.x)
        self.y_f = self.make_score_fetcher(self.y)
        self.context['plot'] = self.make_weighting_plot()
    def make_weighting_plot(self):
        wrapper = lambda x:x
        # Standard averaging is good at removing uncorrelated non-KTs from
        # the top ranks, allowing lower-scoring but better-correlated KTs
        # to move up in rank. But we don't yet have a known mechanism for
        # increasing the aggregate value of drugs with support in only a
        # single score. The disabled code below was an attempt to visualize
        # some candidate mechanisms:
        # - flooring seemed like a possibility because more drugs could be
        #   pushed at or near the 0 -> 0 line, but drugs of interest seem
        #   to be affected proportionally for the most part; possibly this
        #   would help in some special cases
        # - similarly with log scaling
        if False:
            def floored(ordering,floor):
                # expand range between floor and 1
                scale = 1/(1-floor)
                return [
                        (k,0 if v <= floor else scale*v-floor)
                        #(k,0 if v <= floor else v)
                        for k,v in ordering
                        ]
            wrapper = lambda x:floored(x,0.6)
        if False:
            import math
            #wrapper = lambda x:[(k,-math.log10(1-v)) for k,v in x]
            wrapper = lambda x:[(k,-math.log2(1-v)) for k,v in x]
        from dtk.score_pairs import RankTracer
        rt = RankTracer(
                x_ord=wrapper(self.x_f.calibrated_ordering),
                y_ord=wrapper(self.y_f.calibrated_ordering),
                )
        from dtk.plot import PlotlyPlot
        layout = {
                'xaxis':{'title':'Weight'},
                'yaxis':{'title':'Combined score'},
                }
        trace = {
                'mode': 'lines+markers',
                'name': f'Rank {rt.rank}',
                }
        trace['x'],trace['y'] = zip(*rt.trace)
        data = [trace]
        if True:
            # add average traces
            from dtk.num import avg
            if rt.x_ord:
                a1 = avg([v for k,v in rt.x_ord])
                k1 = avg([v for k,v in rt.x_ord if k in self.kts])
            else:
                a1 = k1 = 0
            if rt.y_ord:
                a2 = avg([v for k,v in rt.y_ord])
                k2 = avg([v for k,v in rt.y_ord if k in self.kts])
            else:
                a2 = k2 = 0
            data += [
                    {'x':[0,1], 'y':[a1,a2], 'name':'Average'},
                    {'x':[0,1], 'y':[k1,k2], 'name':'KT Average'},
                    ]
            if False:
                # this trace ends up being an exponential function,
                # often very linear-looking, but always monotonic
                k_val = lambda x:k1 + x*(k2-k1)
                a_val = lambda x:a1 + x*(a2-a1)
                vec = []
                steps = 100
                for w in range(steps):
                    x = 0 + w/steps
                    y = k_val(x)/a_val(x)
                    vec.append((x,y))
                vec.append((1,k2/a2))
                x,y = zip(*vec)
                data += [
                        {'x':x, 'y':y, 'name':'KT Avg/Avg'},
                        ]
            if True:
                # sample the actual SoR value at several weights:
                a_dict = dict(rt.x_ord)
                b_dict = dict(rt.y_ord)
                key_vec = list(set(a_dict) | set(b_dict))
                import numpy as np
                a_vec = np.array([a_dict.get(k,0) for k in key_vec])
                b_vec = np.array([b_dict.get(k,0) for k in key_vec])
                from dtk.enrichment import SigmaOfRank1000,EMInput
                metric = SigmaOfRank1000()
                vec = []
                steps = 100
                for w in range(steps+1):
                    x = 0 + w/steps
                    combined = b_vec*x + a_vec*(1-x)
                    ordering = sorted(zip(key_vec,combined),key=lambda x:-x[1])
                    emi = EMInput(ordering,self.kts)
                    metric.evaluate(emi)
                    y = metric.rating
                    vec.append((x,y))
                x,y = zip(*vec)
                data += [
                        {'x':x, 'y':y, 'name':'SoR1000'},
                        ]
        # add KT traces
        wsa2name = self.ws.get_wsa2name_map()
        from dtk.text import limit
        for key in self.kts:
            data.append({
                    'opacity':0.2,
                    'x':[0,1],
                    'y':[
                        rt.x_lookup.get(key,0),
                        rt.y_lookup.get(key,0),
                        ],
                    'name':limit(wsa2name[key],15),
                    })
        return PlotlyPlot(data, layout)

class MRMRCmpView(DumaView):
    # XXX Next Steps, if we decide this is worth pursuing:
    # XXX - convert this page to a single flexible scatterplot (choose X and
    # XXX   Y from dropdowns like PCA). This would allow this to be used for
    # XXX   other experimentation:
    # XXX   - do weights and floor correlate with each other?
    # XXX   - how do rankings compare when changing different MRMR plugin
    # XXX     functions?
    # XXX - convert experiments directory to use dtk; push implementation
    # XXX   differences into dtk as options
    template_name= 'score/mrmr_cmp.html'
    GET_parms = {
            'wzs_job':(int,None),
            'grouping':(str,'none'),
            }
    button_map={
            'display':['job'],
            }
    def custom_setup(self):
        result = self.ws.get_prev_job_choices('wzs')
        self.job_choices = result or [(None,"No WZS jobs in this workspace")]
    def custom_context(self):
        if self.wzs_job:
            from runner.process_info import JobInfo
            self.bji = JobInfo.get_bound(self.ws,self.wzs_job)
            # construct MRMR object from this job's FM
            fm = self.ws.get_feature_matrix(self.bji.job.settings()['fm_code'])
            from dtk.mrmr import MRMRBasic
            mrmr = MRMRBasic.from_fm(fm)
            mrmr.cycle_to_end()
            self.context['plot'] = self.make_plot(
                    mrmr,
                    'WZS Weight',
                    dict(self.bji.get_score_weights()),
                    )
            floor_lookup = dict(self.bji.get_score_floors())
            if floor_lookup:
                self.context['floor_plot'] = self.make_plot(
                        mrmr,
                        'WZS Floor',
                        floor_lookup,
                        )
    def make_plot(self,mrmr,label,lookup):
        labels = mrmr.get_ordered_labels()
        points = [
                (i,lookup[label])
                for i,label in enumerate(labels)
                ]
        kwargs = dict(
                refline = False,
                text = labels,
                )
        # if we want to partition the points into classes, class_parts
        # holds the class name for each label (or None for no grouping)
        class_parts = dict(
                none=None,
                source=[x.split('_')[0] for x in labels],
                path=[x.split('_')[-2] for x in labels],
                )[self.grouping]
        from dtk.plot import scatter2d,Color
        if class_parts is not None:
            class_names = sorted(set(class_parts))
            color_list = list(Color.ordered_colors)
            while len(class_parts) > len(color_list):
                color_list += Color.ordered_colors
            kwargs['classes'] = [
                            (n,{'color':c})
                            for n,c in zip(class_names,color_list)
                            ]
            kwargs['class_idx'] = [class_names.index(x) for x in class_parts]
        return scatter2d('MRMR Rank',label,points,**kwargs)
    def make_job_form(self,data):
        class MyForm(forms.Form):
            wzs_job = forms.ChoiceField(
                label = 'WZS Job',
                choices = self.job_choices,
                initial = self.wzs_job,
                )
            grouping = forms.ChoiceField(
                choices = [
                        ('none','None'),
                        ('source','Data Source'),
                        ('path','Data Path'),
                        ],
                initial = self.grouping,
                )
        return MyForm(data)
    def display_post_valid(self):
        p = self.job_form.cleaned_data
        return HttpResponseRedirect(self.here_url(**p))

class WeightCmpView(DumaView):
    template_name= 'score/weight_cmp.html'
    GET_parms = {
            'x':(int,None),
            'y':(int,None),
            }
    button_map={
            'display':['jobs'],
            }
    def custom_setup(self):
        result = self.ws.get_prev_job_choices('wzs')
        self.job_choices = result or [(None,"No WZS jobs in this workspace")]
    def custom_context(self):
        if self.x and self.y:
            from runner.process_info import JobInfo
            self.job_ids = [self.x,self.y]
            self.bjis = [JobInfo.get_bound(self.ws,j) for j in self.job_ids]
            from dtk.text import compare_wzs_settings
            self.context['settings_diff'] = compare_wzs_settings(
                    'X Job',          
                    *[x.job.settings() for x in self.bjis]
                    )
            self._setup()
            plotlist = (
                    'weights',
                    'floor_starts',
                    'importances',
                    'ranks'
                    )
            plotly_plots=[]
            import os
            from dtk.plot import PlotlyPlot
            for plot in plotlist:
                func = getattr(self,'build_'+plot)
                pp = func()
                if pp is None:
                    continue
                plotly_plots.append(pp)
            self.context['plotly_plots']=[
                                    ('pldiv_%d'%i,x)
                                    for i,x in enumerate(plotly_plots)
                                    ]
    def _setup(self):
        self.weights={}
        self.full_weights={}
        for x in self.bjis:
            self.full_weights[x] = x.get_full_weights()
            self.weights[x] = x.get_score_weights()
    def build_ranks(self):
        import pandas as pd
        from dtk.scores import Ranker
        ranks=[]
        for x in self.bjis:
            cat = x.get_data_catalog()
            ord=cat.get_ordering('wzs', True)
            names=[t[0] for t in ord]
            rankr = Ranker(ord)
            ranks.append(dict(zip(names,rankr.get_all(names))))
        df = pd.DataFrame(dict(zip(
                self.job_ids,
                ranks,
                )))
        return self._finish_scatter(df, 'WSA Ranks')

    def build_floor_starts(self):
        import pandas as pd
        floors=[]
        for x in self.bjis:
            vals=self.full_weights[x][len(self.weights[x]):]
            names=[tup[0] for tup in self.weights[x]]
            floors.append(dict(zip(names,vals)))
        df = pd.DataFrame(dict(zip(
                self.job_ids,
                floors,
                )))
        return self._finish_scatter(df, 'Floor starting point')

    def build_importances(self):
        from scripts.wzs import get_importances
        import pandas as pd
        import os, json
        importances=[]
        for x in self.bjis:
            try:
                # The parms.json file only exists on the machine where
                # the job was run (and maybe not even there), but if
                # it exists it contains the hardcoded WSA ids that
                # reflect the training set at the time the job was run.
                parms_fn = os.path.join(x.indir, 'parms.json')
                with open(parms_fn) as f:
                    parms = json.loads(f.read())
            except FileNotFoundError:
                # oh, well, fall back to the drugset name
                parms = x.parms
                parms['train_wsa_ids'] \
                        = self.ws.get_wsa_id_set(parms['auto_drug_set'])
            if 'fm_code' in parms:
                fm = x.ws.get_feature_matrix(parms['fm_code'])
            else:
                return None
            importances.append(get_importances(fm, parms, self.full_weights[x]))
        df = pd.DataFrame(dict(zip(
                self.job_ids,
                importances,
                )))
        return self._finish_scatter(df, 'Importances')

    def build_weights(self):
        import pandas as pd
        df = pd.DataFrame(dict(zip(
                self.job_ids,
                [dict(self.weights[x]) for x in self.bjis],
                )))
        return self._finish_scatter(df, 'Weights')

    def _finish_scatter(self, df, title=''):
        import plotly.express as px
        fig = px.scatter(df,x=self.x,y=self.y,hover_name=df.index, title=title)
        from dtk.plot import PlotlyPlot
        return PlotlyPlot.build_from_plotly_figure(fig)

    def make_jobs_form(self,data):
        class MyForm(forms.Form):
            x = forms.ChoiceField(
                label = 'X Axis Job',
                choices = self.job_choices,
                initial = self.x,
                )
            y = forms.ChoiceField(
                label = 'Y Axis Job',
                choices = self.job_choices,
                initial = self.y,
                )
        return MyForm(data)
    def display_post_valid(self):
        p = self.jobs_form.cleaned_data
        return HttpResponseRedirect(self.here_url(
                    x=p['x'],
                    y=p['y'],
                    ))

