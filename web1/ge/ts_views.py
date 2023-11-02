from dtk.duma_view import DumaView,list_of,boolean
from dtk.table import Table,SortHandler
from django import forms
from django.http import HttpResponseRedirect

# the following are needed for old-style views
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from browse.views import make_ctx
from browse.views import post_ok

import logging
logger = logging.getLogger(__name__)

@login_required
def tissue_set(request,ws_id,ts_id=0):
    from browse.models import Workspace,TissueSet,Tissue
    ws = Workspace.objects.get(pk=ws_id)
    ts = None if ts_id == 0 else TissueSet.objects.get(pk=ts_id)
    from .forms import TissueSetForm
    from .utils import find_split_btn
    if request.method == 'POST':
        if 'save_btn' in request.POST:
            ts_form = TissueSetForm(request.POST)
            if ts_form.is_valid():
                p = ts_form.cleaned_data
                if ts:
                    # edit existing
                    ts.name = p['name']
                    ts.case_label = p['case_label']
                    ts.control_label = p['control_label']
                    ts.miRNA = p['miRNA']
                    ts.species = p['species']
                    ts.save()
                else:
                    # create new
                    TissueSet.objects.create(
                        ws=ws,
                        name=p['name'],
                        case_label=p['case_label'],
                        control_label=p['control_label'],
                        miRNA = p['miRNA'],
                        species = p['species'],
                        )
                return HttpResponseRedirect(ws.reverse('ge:tissues'))
        elif 'delete_btn' in request.POST:
            # XXX disabled for now; see comment in template
            assert ts and not ts.tissue_set.exists()
            ts.delete()
            return HttpResponseRedirect(ws.reverse('ge:tissues'))
        elif 'combine_btn' in request.POST:
            # verify name and at least one checkbox
            name = request.POST['combine_name']
            inputs = set([
                    int(key[2:])
                    for key in request.POST
                    if key.startswith('t_')
                    ])
            errors = []
            if not name:
                errors.append("A name must be specified.")
            if len(inputs) < 2:
                errors.append("At least 2 tissues must be selected.")
            if not errors:
                try:
                    Tissue.combine_tissues(name,inputs,request.user.username)
                except RuntimeError as ex:
                    errors.append(ex.message)
            if errors:
                errors.append("Hit Back, correct the form, and try again.")
                from django.utils.html import format_html_join
                return render(request,'error.html'
                            ,make_ctx(request,ws,'ge:tissues',
                                 {'message':"Please correct form input."
                                 ,'detail':format_html_join('','{}<br>',[
                                            (x,)
                                            for x in errors
                                            ])
                                 }
                                )
                            )
            # redirect back to here to load updated configuration
            return HttpResponseRedirect(ws.reverse('ge:tissue_set',ts.id))
        elif find_split_btn(request.POST):
            r = Tissue.objects.get(pk=find_split_btn(request.POST))
            r.split_combined_tissue()
            # redirect back to here to load updated configuration
            return HttpResponseRedirect(ws.reverse('ge:tissue_set',ts.id))
        else:
            raise Exception("unimplemented post operation")
    else:
        ts_form = TissueSetForm(instance=ts)
    return render(request
                ,'ge/tissue_set.html'
                , make_ctx(request,ws,'ge:tissues'
                    ,{'ts':ts
                     ,'ts_form':ts_form
                     }
                    )
                )

class TissueCorrView(DumaView):
    template_name='ge/tissue_corr.html'
    from dtk.duma_view import boolean

    GET_parms = {
        'tissue_set_id':(int,None),
        'corr_type':(str,'spearman'),
        'tissue_frac':(float,0.8),
        'use_prot_cutoffs':(boolean,False),
        'use_direction':(boolean,True),
        }

    button_map={
        'save':['options'],
        }

    heatmap_types = (
        ('pearson', "Pearson Corr"),
        ('spearman', "Spearman Corr"),
        ('prot_overlap', "Protein Overlap"),
    )
    def make_options_form(self,data):
        class MyForm(forms.Form):
            corr_type = forms.ChoiceField(
                label = 'Heatmap Type',
                choices = self.heatmap_types,
                initial = self.corr_type
                )
            tissue_frac=forms.FloatField(
                label='Only use proteins present in this fraction of tissue sets',
                initial=self.tissue_frac
                )
            use_prot_cutoffs=forms.BooleanField(
                label='Use ev and fc cutoffs when picking prots to compare',
                initial=self.use_prot_cutoffs,
                required=False
                )
            use_direction=forms.BooleanField(
                label="Use +/- evidence for correlation depending on dir",
                initial=self.use_direction,
                required=False
                )
        return MyForm(data)

    def save_post_valid(self):
        p = self.context['options_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))

    def custom_setup(self):
        # POST requests just redirect to GET requests with the right params,
        # let's avoid recomputing everything and just throwing it away.
        if self.request.method == "POST":
            return

        import numpy as np

        prot_tissue_map, names = self.load_prot_tissue_map()

        if len(names) == 1:
            self.message("Only 1 tissue, not going to be useful")
            # Compare it against itself to reduce # of special cases in the
            # code below.
            names = [names[0], names[0]]

        if self.corr_type == 'prot_overlap':
            corr_mat = self.make_protein_overlap(prot_tissue_map, names)
        else:
            good_prots = self.pick_good_prots(prot_tissue_map, names)
            if len(good_prots) == 0:
                self.message('There were no proteins matching your criteria')
            tissue_prot_obs_mat = self.make_obs_mat(good_prots, names, prot_tissue_map)

        if self.corr_type == 'spearman':
            from scipy.stats import spearmanr
            corr_mat = spearmanr(tissue_prot_obs_mat, axis=1)[0]
        elif self.corr_type == 'pearson':
            from numpy import corrcoef
            corr_mat = corrcoef(tissue_prot_obs_mat, rowvar=True)
        elif self.corr_type == 'prot_overlap':
            pass
        else:
            raise Exception("%s isn't a known correlation type" % self.corr_type)

        # Handle a couple of degenerate cases where we can't really compute
        # a correlation matrix (e.g. 0 overlapping prots).
        # We just set everything to 0.
        if isinstance(corr_mat, float) or not corr_mat.shape:
            self.message('Degenerate results, your query has no data')
            corr_mat = np.zeros((len(names), len(names)))
        import numpy as np
        corr_mat = np.nan_to_num(corr_mat)

        title = "Protein Evidence Correlation"
        if names:
            # build outputs, but only if there are some non-empty tissues
            self.plot(corr_mat, names, title)
            self.build_table(corr_mat, names)

    def make_protein_overlap(self, tissue_prot_obs_mat, names):
        print("Computing protein overlap")
        from collections import defaultdict
        import numpy as np
        tissue_prots = defaultdict(set)
        for prot, tissues in tissue_prot_obs_mat.items():
            for tissue in tissues.keys():
                tissue_prots[tissue].add(prot)

        out = np.zeros((len(names), len(names)))
        for r, row_name in enumerate(names):
            row_prots = tissue_prots[row_name]
            for c, col_name in enumerate(names):
                col_prots = tissue_prots[col_name]
                isect = len(row_prots & col_prots)
                # Avoid divide by 0 if we have no prots.
                union = len(row_prots | col_prots) + 1e-9
                out[r][c] =  isect / float(union)
        print("Protein overlap done")
        return out
    def build_table(self, heatmap_mat, names):
        from dtk.table import Table
        from dtk.html import tag_wrap
        from tools import sci_fmt
        rows=[]
        cols=[Table.Column('',
                extract=lambda x:x[1],
                cell_fmt=lambda x:tag_wrap('b',x),
                )]
        from dtk.html import link
        from dtk.duma_view import qstr
        for r, name in enumerate(names):
            rows.append((r,name))
            def col_val(r):
                return lambda x: sci_fmt(heatmap_mat[r][x[0]])
            cols.append(Table.Column(name,
                    extract=col_val(r)
                    ))
        self.corr_table = Table(rows,cols)

    def plot(self, heatmap_mat, names, title):
        from dtk.plot import plotly_heatmap
        import numpy as np
        self.heatmap_mat = heatmap_mat
        rows,cols = heatmap_mat.shape
        padding_needed = max([len(i) for i in names]) + 15
        heatmap = plotly_heatmap(
                       heatmap_mat,
                       names,
                       Title = title,
                       color_bar_title = "Cor.",
                       col_labels = names,
                       height = rows*13 + padding_needed*11,
                       width = cols*13 + padding_needed*11,
                       zmin=-1,
                       zmax=1,
                       reorder_cols=True,
              )

        from dtk.plot import boxplot_stack
        plot_data = [(heatmap_mat[i][0:i].tolist() + heatmap_mat[i][i+1:].tolist(), names[i]) for i in range(len(names))]

        bp = boxplot_stack(plot_data, height_per_dataset=50)
        bp.update_layout({
            'title': 'Tissue Correlations',
            'xaxis': {'range': [-1.0, 1.0]}
            })
        self.plotly = [('Heatmap', heatmap), ('BP', bp)]

    def make_obs_mat(self, good_prots, names, prot_tissue_map):
        import numpy as np
        dat_mat = np.zeros((len(names), len(good_prots)))
        for r, t in enumerate(names):
            for c, p in enumerate(good_prots):
                dat_mat[r][c] = prot_tissue_map[p].get(t, 0)
        return dat_mat

    def pick_good_prots(self, prot_tissue_map, names):
        minnum = len(names) * self.tissue_frac
        from collections import defaultdict
        counts = defaultdict(int)
        # Skip any proteins that don't show up in at least K of our datasets.
        prots = [p for p, v in prot_tissue_map.items() if len(v) >= minnum]
        print("Found %d of %d prots" % (len(prots), len(list(prot_tissue_map.keys()))))
        return prots
    def get_tissues(self):
        from browse.models import Tissue
        ts_id=self.tissue_set_id
        assert ts_id is not None, "Must specify a tissue set ID"
        return Tissue.objects.filter(tissue_set_id=ts_id).order_by('name')

    def load_prot_tissue_map(self):
        from collections import defaultdict
        prot_tissue_map = defaultdict(dict)
        names = []

        for t in self.get_tissues():
            tname = t.concise_name()
            if tname in names:
                tname += "(%d)" % t.id
            print("Loading in %s" % tname)
            num_recs = 0
            l = t.sig_results(over_only=self.use_prot_cutoffs)
            for rec in l:
                num_recs += 1
                score = rec.evidence
                if self.use_direction:
                    score *= rec.direction
                prot_tissue_map[rec.uniprot][tname] = score
            if num_recs == 0:
                print("Skipping empty dataset %s" % tname)
            else:
                names.append(tname)
            print("Loaded %s with %d proteins" % (tname, len(l)))
        return prot_tissue_map, names

class TissueSetAnalysisView(DumaView):
    template_name = 'ge/tissue_set_analysis.html'

    GET_parms = {
        'tissue_set_id':(int,None),
        }

    def custom_context(self):
        tissue_names, scores_by_type, score_names = self._get_scores()
        self.plot(scores_by_type, score_names)
        self.build_table(scores_by_type, tissue_names, score_names)

    def get_tissues(self):
        from browse.models import Tissue
        ts_id=self.tissue_set_id
        return Tissue.objects.filter(tissue_set_id=ts_id).order_by('name')

    def _get_scores(self):
        from collections import defaultdict
        scores_by_type = defaultdict(list)
        tissue_names = []

        score_name_mapping = [
            ('Case samples','caseSampleScores'),
            ('Control samples','contSampleScores'),
            ('Concordance','concordScore'),
            ('Probes >= 0.95','sigProbesScore'),
            ('Probes = 1.0','perfectProbesScore'),
            ('Case correlation','caseCorScore'),
            ('Control correlation','controlCorScore'),
            ('Consistent Direction','consistDirScore'),
            ('Mapping','mappingScore'),
            ('Overall score','finalScore'),
            ]
        # Store these separately to keep the ordering
        score_names = [x[0] for x in score_name_mapping]
        score_name_mapping = dict([(x2, x1) for x1, x2 in score_name_mapping])

        for t in self.get_tissues():
            tname = t.concise_name()
            scores = t.sig_qc_scores()
            if len(scores) == 0:
                continue
            tissue_names.append(tname)

            for score_name, value in scores.items():
                scores_by_type[score_name_mapping[score_name]].append(value)
        return tissue_names, scores_by_type, score_names

    def plot(self, scores_by_type, score_names):
        from dtk.plot import boxplot_stack
        plot_data = [(scores_by_type[x], x) for x in score_names]

        if len(plot_data) > 0:
            bp = boxplot_stack(plot_data, height_per_dataset=80)
            self.plotly = [('BP', bp)]


    def build_table(self, scores_by_type, tissue_names, score_names):
        from dtk.table import Table
        from dtk.html import tag_wrap
        from tools import sci_fmt
        rows=[]
        cols=[Table.Column('',
                extract=lambda x:x[1],
                cell_fmt=lambda x:tag_wrap('b',x),
                )]
        from dtk.html import link
        from dtk.duma_view import qstr

        for r, tissue_name in enumerate(tissue_names):
            rows.append((r,tissue_name))

        def fmt(x):
            try:
                return sci_fmt(x)
            except ValueError:
                # Can contain strings like 'NA'
                return x

        for c, name in enumerate(score_names):
            def col_val(x, c, name):
                idx = x[0]
                scores = scores_by_type[name]
                if idx < len(scores):
                    out = scores[idx]
                else:
                    out = float('nan')
                return fmt(out)
            from functools import partial
            bound_col_val = partial(col_val, c=c, name=name)
            cols.append(Table.Column(name,
                    extract=bound_col_val,
                    ))
        self.table = Table(rows,cols)


def tiss_pathsum_histo(ts, kts, all_prots):
    from browse.models import Tissue
    from algorithms.run_path import get_tissue_settings_keys

    qs = Tissue.objects.filter(tissue_set=ts)
    
    from collections import defaultdict, Counter
    scores = defaultdict(int)
    for t in qs:
        ev_thresh = t.ev_cutoff
        fc_thresh = t.fc_cutoff
        for sp in t.sig_results(ev_cutoff=ev_thresh, fc_cutoff=fc_thresh):
            scores[sp.uniprot] += 1
    
    kt_out = []
    other_out = []
    for prot in all_prots:
        val = scores[prot]
        if prot in kts:
            kt_out.append(val)
        else:
            other_out.append(val)
            
    return kt_out, other_out, len(qs)

def prot_stats_plot(ws, ts):
    from browse.models import Protein
    all_prots = list(Protein.objects.all().values_list('uniprot', flat=True))
    kt_prots = ws.get_uniprot_set(ws.get_nonnovel_ps_default())
    
    
    kt_data, other_data, N = tiss_pathsum_histo(ts, kt_prots, all_prots)

    import plotly.graph_objects as go
    from dtk.plot import PlotlyPlot, bar_histogram_overlay, fig_legend
    annot = [fig_legend(["Histogram of how many tissues each protein shows up in significantly.<br>Ideally you want to see the non-novel distribution shifted right as compared to the other."], -0.1)]

    data, layout = bar_histogram_overlay([kt_data, other_data], names=['nonnovel', 'other'], bins=N+1, x_range=(-0.5,N+0.5), density=True, annotations=annot)
    layout.update({
        'title': "Protein Tissue Thresholded Overlap",
    })
    layout['xaxis']['title'] = '# Tissues'
    layout['yaxis'] = {'title': 'Portion of Prots'}
    return PlotlyPlot(data, layout)



@login_required
def tissue_stats(request,ws_id,ts_id):
    from browse.models import Workspace,TissueSet
    ws = Workspace.objects.get(pk=ws_id)
    ts = TissueSet.objects.get(pk=ts_id)
    from .utils import DbProteinMappings
    db=DbProteinMappings(ws=ws)
    from browse.utils import ProteinEvidenceOption, extract_list_option
    evidence_options = [2.,0.999,0.99,0.95,0.9,0.8]
    peo = ProteinEvidenceOption(request,ws,evidence_options)
    from .utils import level_list,get_tissue_stats,plot_pt_stats
    ts_heading,ts_rows,pt_counts,tp_stats = get_tissue_stats(ts,peo.selected, heading = peo.options)
    peo.level_list = level_list(pt_counts)
    if tp_stats:
        plot_pt_stats(peo.level_list, ws_id, tp_stats)
    ppd = db.get_prots_per_drug()
    wsa_id_map = db.dpi.get_wsa_id_map(ws)
    mapped_ppd = {}
    for k,v in ppd.items():
        for wsa_id in wsa_id_map.get(k,[]):
            mapped_ppd[wsa_id] = v
    plots = [('ProtStats', prot_stats_plot(ws, ts))]
    display_heading = [str(x) if x <=1. else 'Dataset-specific thresholds' for x in ts_heading]
    return render(request
                ,'ge/tissue_stats.html'
                , make_ctx(request,ws,'ge:tissues'
                    ,{'page_label':ts.name+' Tissue Set Stats'
                     ,'heading':display_heading
                     ,'rows':ts_rows
                     ,'peo':peo
                     ,'show':extract_list_option(request,'show')
                     ,'drug_prot_levels':level_list(mapped_ppd)
                     ,'kt_prot_levels':level_list(db.get_kts_by_prot(ws))
                     ,'ts':ts
                     , 'plots':plots
                     }
                    )
                )

@login_required
def kt_tiss(request,ws_id,ts_id):
    from browse.models import Workspace,TissueSet
    ws = Workspace.objects.get(pk=ws_id)
    ts = TissueSet.objects.get(pk=ts_id)
    from .utils import DbProteinMappings
    db=DbProteinMappings(ws=ws)
    # csv gen
    from browse.utils import extract_bool_option,extract_float_option
    all_drugs = extract_bool_option(request,'all_drugs')
    thresh = extract_float_option(request,'thresh')
    from .utils import get_tissue_stats
    ts_heading,ts_rows,thash,tp_stats = get_tissue_stats(ts,thresh)
    if all_drugs:
        dhash = db.get_drugs_per_prot()
        page_label = 'Drug'
    else:
        dhash = db.get_kts_by_prot(ws)
        page_label = 'KT'
    page_label += ' / %s Tissue Set Comparison' % ts.name
    from path_helper import PathHelper, make_directory
    pub_dir = PathHelper.ws_publish(ws.id)
    make_directory(pub_dir)
    infile = pub_dir+"kt_tiss.csv"
    outfile = pub_dir+"proteinsNumberOfTissuesVsNumberOfDrugs.pdf"
    import csv
    with open(infile,"w") as csvfile:
        wr = csv.writer(csvfile,lineterminator="\n")
        for prot in set(list(thash.keys())+list(dhash.keys())):
            wr.writerow((prot,thash.get(prot,0) ,len(dhash.get(prot,[]))))
    import subprocess
    subprocess.check_call(['sh','-c','(cd '+PathHelper.MLscripts+' && Rscript proteinsNumberOfTissuesVsNumberOfDrugs.R %s %s)' % (infile,pub_dir)])
    #
    # prepare data for crosstab display:
    # - put each prot list in a hash by key pair
    # - determine max drug and tissue counts
    max_tissues = 0
    max_drugs = 0
    fmt="%d,%d"
    index = {}
    for prot in set(list(thash.keys())+list(dhash.keys())):
        tcnt = thash.get(prot,0)
        if tcnt > max_tissues:
            max_tissues=tcnt
        dcnt = len(dhash.get(prot,[]))
        if dcnt > max_drugs:
            max_drugs=dcnt
        key = fmt % (tcnt,dcnt)
        group = index.setdefault(key,[])
        group.append(prot)
    # organize data for easy display in a template
    display=[]
    for di in range(max_drugs,-1,-1):
        row=[]
        display.append((di,row))
        for ti in range(0,max_tissues+1):
            key = fmt % (ti,di)
            row.append(index.get(key,[]))
    return render(request
                ,'ge/kt_tiss.html'
                , make_ctx(request,ws,'ge:tissues'
                    ,{'page_label':page_label
                     ,'heading':['vD/T>']+list(range(0,max_tissues+1))
                     ,'display':display
                     ,'graph_link':PathHelper.url_of_file(outfile)
                     }
                    )
                )

