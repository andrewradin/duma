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

class SigProtView(DumaView):
    template_name='ge/sigprot.html'
    index_dropdown_stem='ge:tissues'
    GET_parms={
            'display_cutoff':(float,0.7),
            'ev_cutoff':(float,None),
            'fc_cutoff':(float,None),
            }
    button_map={
            'display':['options'],
            'save':['options'],
            'reset':[]
            }
    def make_options_form(self,data):
        t = self.tissue
        # if these weren't present in GET parms, use DB values
        if self.ev_cutoff is None:
            self.ev_cutoff = t.ev_cutoff
        if self.fc_cutoff is None:
            self.fc_cutoff = t.fc_cutoff
        # build and return form
        from tools import sci_fmt
        class MyForm(forms.Form):
            ev_cutoff=forms.FloatField(
                label='Evidence threshold (saved=%s)'%sci_fmt(t.ev_cutoff),
                initial=self.ev_cutoff,
                # these steps are convenient, but they prevent manually
                # entering a value that doesn't fall on the step
                #widget=forms.NumberInput(attrs={'step': 0.01}),
                )
            fc_cutoff=forms.FloatField(
                label='Log 2 Fold Change threshold (saved=%s)'
                        %sci_fmt(t.fc_cutoff),
                initial=self.fc_cutoff,
                #widget=forms.NumberInput(attrs={'step': 0.025}),
                )
            display_cutoff=forms.FloatField(
                label='Min evidence to display',
                initial=self.display_cutoff,
                #widget=forms.NumberInput(attrs={'step': 0.1}),
                )
        return MyForm(data)
    def display_post_valid(self):
        p = self.context['options_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))
    def save_post_valid(self):
        p = self.context['options_form'].cleaned_data
        self.tissue.ev_cutoff = p['ev_cutoff']
        self.tissue.fc_cutoff = p['fc_cutoff']
        self.tissue._recalculate_sig_result_counts()
        self.tissue.cutoff_job_id = self.tissue.sig_result_job_id
        self.tissue.save()
        return HttpResponseRedirect(self.here_url(**p))
    def reset_post_valid(self):
        return HttpResponseRedirect('?')
    def custom_context(self):
        self.context_alias(
                stats=[],
                plotly_plots=[],
                )
        self.all_data = self.tissue.sig_results(over_only=False)
        self.get_protein_stats()
        self.build_main_plot()
        self.build_ev_vs_fc_plot()
    def get_protein_stats(self):
        from tools import sci_fmt
        self.stats.append(('Proteins over threshold',str(len([
                x
                for x in self.all_data
                if x.evidence >= self.ev_cutoff
                and x.fold_change >= self.fc_cutoff
                ]))))
        self.stats.append(('... over ev only',str(len([
                x
                for x in self.all_data
                if x.evidence >= self.ev_cutoff
                ]))))
        self.stats.append(('... over fc only',str(len([
                x
                for x in self.all_data
                if x.fold_change >= self.fc_cutoff
                ]))))
        self.stats.append(('Total proteins',str(len(self.all_data))))
        if not self.all_data:
            return
        from dtk.num import avg_sd
        for i,(field,label) in enumerate((
                        ('evidence','Evidence'),
                        ('fold_change','Log 2 Fold Change'),
                        )):
            vals=[getattr(x,field) for x in self.all_data]
            if vals:
                self.stats.append((label+' range','%s to %s'%(
                        sci_fmt(min(vals)),
                        sci_fmt(max(vals)),
                        )))
                avg,sd = avg_sd(vals)
                self.stats.append((label+' average',sci_fmt(avg)))
                self.stats.append((label+' standard deviation',sci_fmt(sd)))
    def build_main_plot(self):
        from tools import sci_fmt
        self.stats.append(('Display cutoff',sci_fmt(self.display_cutoff)))
        # get data
        tuples = [(
                    float(x.evidence),
                    float(x.fold_change) * float(x.direction),
                    x.uniprot,
                    )
            for x in self.all_data
            if x.evidence >= self.display_cutoff
            ]
        self.stats.append(('Proteins displayed',len(tuples)))
        if len(tuples) > 1:
            tuples.sort(reverse=True)
            ev, fc, ids = zip(*tuples)
            from .utils import make_protein_plot
            pp = make_protein_plot(ev,fc,ids,self.ev_cutoff)
            self.plotly_plots.append(('scat',pp))
    def build_ev_vs_fc_plot(self):
        tuples = [(
                    float(x.evidence),
                    float(x.fold_change) * float(x.direction),
                    x.uniprot,
                    )
            for x in self.all_data
            if x.evidence >= self.ev_cutoff
            and x.fold_change >= self.fc_cutoff
            ]
        if len(tuples) > 1:
            from dtk.plot import scatter2d
            ev, fc, ids = zip(*tuples)
            from browse.models import Protein
            names = Protein.make_gene_list_from_uniprot_list(ids)
            from dtk.plot import scatter2d
            pp = scatter2d('Log 2 Fold Change',
                    'Evidence',
                    zip(fc, ev),
                    title = 'Over-threshold Evidence vs Log 2 Fold Change',
                    text = [names[i] for i in range(len(ids))],
                    ids = ('protpage', ids),
                    refline = False,
                    class_idx = [0] * len(ids), # filler
                    classes=[('Unknown' ,{'opacity' : 0.3})],
                    )
            self.plotly_plots.append(('scat2',pp))

class sigQcView(DumaView):
    '''
        Display plots generated during the running of
        the sig gene expression process.
    '''
    template_name = 'ge/sig_qc.html'
    index_dropdown_stem = 'ge:tissues'
    class iframe:
        def __init__(self, src, **kwargs):
            self.src = src
            self.height = kwargs.get('height', 520)
            self.width = kwargs.get('width', 500)

    def custom_context(self):
        error = None
        bji = None
        try:
            bji = self.tissue.sig_bji()
            from runner.models import Process
            enum = Process.status_vals
            if bji.job.status != enum.SUCCEEDED:
                error = "Sig job "+enum.get('label',bji.job.status)
        except ValueError as ex:
            error = str(ex)
        if error:
            self.message(error)
        if not bji:
            return

        # try loading tsv scores file
        qc_scores = self.tissue.sig_qc_scores()
        input_scores = [
            ('Case samples','caseSampleScores'),
            ('Control samples','contSampleScores'),
            ]
        diff_scores = [
            ('Concordance','concordScore'),
            ('Probes >= 0.95','sigProbesScore'),
            ('Probes = 1.0','perfectProbesScore'),
            ]
        map_scores = [
            ('Case correlation','caseCorScore'),
            ('Control correlation','controlCorScore'),
            ('Consistent Direction','consistDirScore'),
            ('Mapping','mappingScore'),
            ]
        top_scores = [('Overall score','finalScore'),
                     ]+input_scores+diff_scores+map_scores
        from path_helper import PathHelper
        self.url_base = PathHelper.url_of_file(bji.final_pubdir)
        self.url_stem = PathHelper.url_of_file(
                               bji.final_pubdir
                               +self.tissue.base_geo()
                               +self.tissue.tissue_suffix()
                           )
        src_link = self.tissue.source_url()
        if src_link:
            from dtk.html import link
            src_link = link('',src_link,True)
        if self.tissue.source.endswith('-seq'):
# TODO are there relevant plots from EdgeR?
            diff_data = self._rnaseq_diff()
            import os
            if os.path.isfile(
                 os.path.join(bji.final_pubdir,
                              "case.model.fits.pdf")
               ):
                diff_data += self._sc_diff()
        else:
            diff_data = self._diff_data()
        from dtk.html import bar_chart_from_dict,table_from_dict
        self.context_alias(sections = [['input', 'Input Quality',
                                        table_from_dict(qc_scores,input_scores),
                                        self._input_data()
                                       ],
                                       ['diff', 'Differential Calling Quality',
                                        table_from_dict(qc_scores,diff_scores),
                                        diff_data
                                       ],
                                       ['map', 'Mapping and Final Separation',
                                        table_from_dict(qc_scores,map_scores),
                                        self._map_data()
                                       ]],
                           top_table = bar_chart_from_dict(qc_scores,top_scores),
                           tissue = self.tissue
                          )
    def _map_data(self):
        return [[self.iframe(self.url_stem+"_sigPCAMDS.pdf",
                        width = 800, height = 9300
                        ),"\n".join([
        'This series of plots shows the separation of case and control',
            'using only significant probes called by each program.',
        'Each program has 3 plots: MDS, PCA, and correlation heatmap.',
        'If a program did not call any probes significant,',
            'there will be no plots for that program.',
        'The first set of plots is from the concordance of the',
            '3 programs, and is what we are actually using in',
            'gene expression-based component methods.',
        'For each set of plots, the first plot is a multidimensional scaling',
            '(MDS) plot, and as such includes all of the variance',
            'in the data visualized in just 2 dimensions.',
        'The second plot is a PCA, and is similar to the first plot,',
            'but only shows the 2 dimensions with the most variance.',
        'In both of the first two plots we expect to see',
            'a clear separation of case and controls.',
        'The third plot is a heatmap of the correlation (Pearson Rho)',
            'between all samples using the expression data',
            'from only the significant probes.',
        'Note that the colors autoscale, so check the legend on the right.',
        'Here, we hope to see that all of the cases and controls',
            'are in their own cluster.',
        'Note that for all of these plots, but particularly',
            'the first set of plots,we have the option to try and improve',
            'separation and clustering of cases and controls',
            'by removing unwanted variation using SVA.',
        ])],
        ]
    def _rnaseq_diff(self):
        return []
    def _sc_diff(self):
        return [[self.iframe(self.url_base+"case.model.fits.pdf",
                        width = 1000, height = 1000
                        ),"\n".join([
        'scde, the package used for single cell RNA-seq analysis,',
            'fits individual error models for single cells using counts',
            'derived from single-cell RNA-seq data to estimate drop-out',
            'and amplification biases on gene expression magnitude.'
        'The cells (within each group) are first cross-compared to',
            'determine a subset of genes showing consistent expression.',
            'The set of genes is then used to fit a mixture model',
            '(Poisson-NB mixture,with expression-dependent concomitant)'
        'The first PDF is of all "case" cells, while the lower plot are "controls".'
                 ])],
        [self.iframe(self.url_base+"control.model.fits.pdf",
                        width = 1000, height = 1000
                        ),'']
               ]
    def _diff_data(self):
        return [[self.iframe(self.url_stem+"_rankProdPlot.png"),"\n".join([
        'The Y-axis of RankProd plots show the FDR',
            '(AKA PFP - portion false positive) of each probe,',
            'and the probes are ordered by their fold change (most to least).',
        'The upper plot is ordered by the fold up regulated, while',
            'the lower plot is ordered by fold down regulated.',
        'In both cases we expect to see the plot start around the origin',
            'and then fairly quickly ramp up before plateauing.',
        'Issues could include:',
        '1) The data points not increasing smoothly and steadily.',
            'That might suggest unusual input data.',
        '2) Many points around 0 on the y-axis.',
            'That would suggest too many probes being called significant.',
            'One solution might be increasing the threshold for this dataset.',
        '3) No or few points below 0.05.',
            'This could be because there are not changes between case and control,',
            'or that we need to lower the threshold for this dataset.'
        ])],
        [self.iframe(self.url_stem+"_qqplot.png"),"\n".join([
        'The SamR plot (AKA the gummy worm) is a Q-Q plot where each point is a probe.',
        'You can think of the solid black line as expected,',
        'with the dotted lines being 95% confidence intervals (CIs).',
        'Probes that fall outside of those CIs are significant, and get colored.',
        'We expect this plot to be largely linear, with down and up curves on either end.',
        'Issues could include:',
        '1) The plots not increasing smoothly and steadily.',
            'That might suggest unusual input data.',
        '2) The line of dots not being close to parallel to the expected line,',
            'and as a result many probes being called significant.',
            'That might suggest we should increase the threshold for this dataset.',
        '3) No or few dots falling outside the CIs.',
            'This could be because there are not changes between case and control,',
            'or that we need to lower the threshold for this dataset.',
        ])],
        [self.iframe(self.url_stem+"_concordance.pdf",
                width = 700, height = 4000
                ),"\n".join([
        'The concordance plot shows the overlap or agreement between all of the programs',
            'we use to call differentially expressed probes (DEPs).',
        'The data labeled "Concord" is the median of the other programs,',
            'and is currently what we are using for our final DEP call.',
        'The first plot is a heatmap showing the number of DEPs common to method pairs.',
        'Note that self comparisons (e.g. Concord vs Concord)',
            'shows the total number of DEPs called by that program.',
        'The second plot shows the same data as the first plot,',
            'but as portions of the total number of DEPs called by',
            'the program listed in the row.',
        'Note that if a row is red (i.e. numbers close to 1.0),',
            'that program is approximately a subset of everything else,',
            'while if a column is red that program is mostly',
            ' a superset of the other programs.',
        'The last heatmap is the Spearman rank correlation',
            'of the directional significance values',
            '(e.g. FDR * direction (1 or -1)) for just the significant intersections.',
        'Here we are hoping for high values in all comparisons.',
        'The venn diagram, which does NOT scale to reflect',
            'the numbers in the intersections,',
            'shows the agreement in DEP calls between all of the programs.',
        'The final plot is a simple bar plot showing what portion',
             'of all probes on the microarray',
            'are called significant by each program.',
        'Note that this autoscales, so check the Y-axis!',
        ])],
        ]
    def _input_data(self):
        return [[self.iframe(self.url_stem+"_ECDFs.png"),"\n".join([
         'ECDFs (Empirical cumulative distribution function)',
            'for each microarray individually.',
         'This is just a sanity check to make sure the distribution of',
            'probe intensity is as expected and that there are no outliers.',
        'Here and for the other plots, black = control and red = case.',
        'Potential issues:',
        '1) Negative values.',
            'We expect that the data is Log2 values of intensity.',
        '2) Big values.',
            'We expect the X-axis to stop before 20 or so.',
            'If not, it might suggest taking Log2 of the values did not work.',
        '3) Curves that are not right on top of each other.',
            'This would suggest systematic differences between microarrays.',
            'This is especially an issue if many of the red lines are shifted',
                'from the black lines.',
            'If this happens, it may suggest that normalization',
                'of the microarrays was not completed successfully.',
        ])],
        [self.iframe(self.url_stem+"_cv.png"),"\n".join([
        'The coefficient of variation (CV) is a unit-less measure of variance,',
            'here for a single probe.',
        'For a given probe, the CV is the standard deviation across all cases',
            'and controls, divided by the mean of the same values.',
        'As a result the CV can be thought of as the portion of the average',
            'that the standard deviation represents.',
        'The probes are ordered from lowest to highest CV,',
            'and the Y scale is in log space.',

        ])],
        [self.iframe(self.url_stem+"_rawMDSPCA.pdf",
                width=800, height=2400
                ),"\n".join([
        'These plots demonstrate the separation of the case and control samples',
            'using all probes on the microarray, not just the significant probes.',
        'The first plot is a Principal Component Analysis (PCA).',
        'The two axis represent the dimensions which explain',
            'the most variability in the data.',
        'Because we are using all of the probes, we do not necessarily expect',
            'a clear separation of the case (red) and controls (black),',
            'but they should be trending towards a separation.',
        'The second plot shows the variance explained by each dimension in the PCA.',
        'Here we are hoping the first 1 (X-axis in the plot above) or',
            '2 (y-axis in the plot above) points are much higher than the rest.',
        'The final plot is similar to the PCA, but a multidimensional scaling',
            '(MDS) plot includes all variance visualized in just 2 dimensions.'
        ])],
        [self.iframe(self.url_stem+"_sampleCorrelationHeatmap.png",
                width=960, height=960
                ),"\n".join([
        'This is a heatmap of the correlation (Pearson Rho) between all samples',
            'using all expression data from all probes.',
        'A value of 1 means the datasets are perfectly, linearly, correlated.',
        'Here, we hope to see that all of the cases and controls',
            'are in their own cluster.',
        'For example, if all controls are well correlated,',
            'they should group near each other on the axes and form a block',
            'that is more red than blue when cross-compared.',
        'Note that the colors autoscale, according to the legend on the right.',
        ])],
        ]

@login_required
def ge_overlap(request,ws_id):
    from browse.models import Workspace,Sample,Tissue
    ws = Workspace.objects.get(pk=ws_id)
    # find any GSM that gets used more than once in this workspace
    tissue_qs = Tissue.objects.filter(ws=ws).exclude(
                                    tissue_set_id=0,
                                    invalid=True,
                                    )
    samples = {}
    for sample in Sample.objects.filter(
                                tissue__in=tissue_qs,
                                ).exclude(
                                classification=0,
                                ):
        samples.setdefault(sample.sample_id,[]).append(sample)
    # now for all the duplications, find the set of tissue ids involved
    tissues = {}
    for k,v in samples.items():
        if len(v) == 1:
            continue
        t_ids = set([s.tissue_id for s in v])
        key = tuple(sorted(t_ids))
        d = tissues.setdefault(key[0],{})
        for t2 in key[1:]:
            l = d.setdefault(t2,[[],[],[]])
            cset = set([s.classification for s in v])
            if cset == set([1]):
                l[0] += v
            elif cset == set([2]):
                l[1] += v
            else:
                l[2] += v
    # pre-load all affected tissues for fast lookup
    needed_ids = set(tissues.keys())
    for d in tissues.values():
        needed_ids |= set(d.keys())
    tissue_lookup = {
            t.id:t
            for t in Tissue.objects.filter(id__in=needed_ids)
            }
    # format the output table
    rows = []
    def desc(t_id):
        t = tissue_lookup[t_id]
        return "%s (%d) %s" % (t.geoID,t.id,t.name)
    def tset(t_id1,t_id2):
        t1 = tissue_lookup[t_id1]
        t2 = tissue_lookup[t_id2]
        if t1.tissue_set_id == t2.tissue_set_id:
            return t1.tissue_set_id
        return "None"
    for t1,v1 in tissues.items():
        for t2,v2 in v1.items():
            rows.append( (
                        desc(t1),
                        desc(t2),
                        tset(t1,t2),
                        len(v2[0]),
                        len(v2[1]),
                        len(v2[2])
                    ) )
    header=['tissue','shares with','common tissue set','cases','controls','mixed']
    from dtk.html import pad_table
    return render(request
                ,'ge/ge_overlap.html'
                , make_ctx(request,ws,'ge:ge_overlap',
                        {
                            'table':pad_table(header,rows),
                        }
                    )
                )

