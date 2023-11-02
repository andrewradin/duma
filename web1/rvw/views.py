from dtk.duma_view import DumaView,list_of,boolean
from dtk.table import Table,SortHandler
from django import forms
from django.http import HttpResponseRedirect

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse

import logging

from runner.process_info import JobInfo
logger = logging.getLogger(__name__)

class ReviewView(DumaView):
    template_name='rvw/review.html'
    index_dropdown_stem='rvw:review'
    GET_parms = {
            'flavor':(str,''),
            'search':(str,''),
            'ever_ind':(int,None),
            }
    button_map={
            'search':['search'],
            }

    def make_search_form(self, data):
        from browse.models import WsAnnotation
        choices = WsAnnotation.grouped_choices()
        class MyForm(forms.Form):
            ever_ind = forms.ChoiceField(
                        label='Show Ever Reached',
                        initial=self.ever_ind or WsAnnotation.indication_vals.REVIEWED_PREDICTION,
                        choices=choices,
                        )
        return MyForm(data)

    def search_post_valid(self):
        p = self.context['search_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))
    def custom_context(self):
        from browse.models import WsAnnotation,Election
        qs_ws = WsAnnotation.objects.filter(ws=self.ws)
        enum=WsAnnotation.indication_vals
        cands = (enum.INITIAL_PREDICTION,)
        rvwd = (enum.REVIEWED_PREDICTION,)
        pats = (
                enum.CANDIDATE_PATENTED,
                enum.PATENT_PREP,
                enum.HIT,
                enum.IN_VITRO_1,
                enum.IN_VITRO_2,
                enum.IN_VIVO_1,
                enum.IN_VIVO_2,
                enum.LEAD_OP,
                )
        kt_ids = self.ws.get_wsa_id_set(self.ws.eval_drugset)
        elections=[]
        ds_prefix = 'ds'
        fs_prefix = 'flagset_'
        use_id_prefetch = False
        if self.search:
            search = self.search.strip()
            from browse.utils import drug_search_wsa_filter
            qs = drug_search_wsa_filter(qs_ws,search)
            use_id_prefetch = True
            headline="Search for "+search
        elif self.ever_ind is not None:
            from browse.models import DispositionAudit
            ever_wsas = DispositionAudit.objects.filter(
                    indication=self.ever_ind).values_list('wsa', flat=True)

            qs = qs_ws.filter(pk__in=ever_wsas)
            ivals = WsAnnotation.indication_vals
            label = ivals.get('label', self.ever_ind)
            headline = f"Molecules that reached '{label}'"
        elif self.flavor=='hidden':
            qs = qs_ws.filter(agent__hide = True)
            headline="Hidden Drugs"
        elif self.flavor=='all':
            qs = qs_ws.filter(marked_on__isnull = False)
            headline="Current and Past Predictions"
        elif self.flavor=='patent':
            qs = qs_ws.filter(indication__in=pats)
            headline="Hits"
        elif self.flavor=='reviewed':
            qs = qs_ws.filter(indication__in=rvwd)
            headline="Reviewed Predictions"
        elif self.flavor=='KT':
            qs = qs_ws.filter(pk__in=kt_ids)
            headline="Known Treatments"
        elif self.flavor.startswith(ds_prefix):
            ds_id = int(self.flavor[len(ds_prefix):])
            from browse.models import DrugSet
            ds = DrugSet.objects.get(pk=ds_id)
            qs = ds.drugs.all()
            headline = "In DrugSet "+ds.name
        elif self.flavor.startswith(fs_prefix):
            from flagging.models import FlagSet,Flag
            fs_id = int(self.flavor[len(fs_prefix):])
            fs = FlagSet.objects.get(pk=fs_id)
            headline = "Flagged for "+fs.source
            qs = qs_ws.filter(pk__in=Flag.objects.filter(
                        run_id=fs_id
                        ).values_list('wsa_id',flat=True))
        else:
            qs = qs_ws.filter(indication=enum.INITIAL_PREDICTION)
            headline="Under Review"
            from browse.models import Election
            for e in Election.objects.filter(ws=self.ws):
                elections.append((
                        self.ws.reverse('rvw:election',e.id),
                        e.elec_label(),
                        ))
            if self.in_group('duma_admin'):
                new_href = self.ws.reverse('rvw:election',0)
                elections += [
                        (new_href+'?flavor=pass1',"Create new Preliminary"),
                        (new_href+'?flavor=pass2',"Create new Secondary"),
                        ]
        qs = WsAnnotation.prefetch_agent_attributes(
                qs,
                use_id_prefetch=use_id_prefetch,
                )
        if self.is_demo():
            qs = []
        self.context['qs'] = qs
        self.context['headline'] = headline
        self.context['elections'] = elections
        from dtk.table import Table
        from dtk.html import link,hover,note_format,tag_wrap,glyph_icon
        from dtk.prot_map import DpiMapping, AgentTargetCache

        dpi = DpiMapping(self.ws.get_dpi_default())
        base_dpi = dpi.get_baseline_dpi()
        targ_cache = AgentTargetCache(
                mapping=base_dpi,
                agent_ids=[x.agent_id for x in qs],
                dpi_thresh=self.ws.get_dpi_thresh_default(),
                )
        orig_targ_cache = AgentTargetCache(
                mapping=dpi,
                agent_ids=[x.agent_id for x in qs],
                dpi_thresh=self.ws.get_dpi_thresh_default(),
                )
        
        def dpi_for_agent(agent):
            if agent.is_moa():
                lst = orig_targ_cache.info_for_agent(agent.id)
            else:
                lst = targ_cache.info_for_agent(agent.id)
            return lst


        def format_dpi(dpilst):
            from dtk.html import join,link
            from dtk.plot import dpi_arrow
            from django.utils.safestring import mark_safe
            return join(*[
                        join(
                            link(x[1],self.ws.reverse('protein',x[0])),
                            dpi_arrow(x[2]),
                            )
                        for x in dpilst
                ],sep=mark_safe(u'<br>'))

        wsa2dpilst = {}
        wsa2targets = {}
        wsa2protnames = {}
        allprots = set()
        for wsa in qs:
            dpilst = dpi_for_agent(wsa.agent)
            wsa2dpilst[wsa.id] = dpilst
            wsa2targets[wsa.id] = format_dpi(dpilst)
            allprots.update([x[0] for x in dpilst])
        
        from browse.models import ProteinAttribute
        uni2name = dict(ProteinAttribute.objects.filter(prot__uniprot__in=allprots, attr__name='Protein_Name').values_list('prot__uniprot', 'val'))

        for wsa_id, dpilst in wsa2dpilst.items():
            from django.utils.safestring import mark_safe
            names = [uni2name.get(uni, f'({uni})') for uni, *rest in dpilst]
            wsa2protnames[wsa_id] = mark_safe('<br>'.join(names))

        def name_column_extractor(wsa):
            from dtk.html import link
            from .utils import get_needs_vote_icon
            vote_icon = get_needs_vote_icon(wsa, self.request.user)
            drug_name = link(
                    wsa.get_name(self.is_demo()),
                    wsa.drug_url(),
                    )
            from dtk.text import fmt_time
            if wsa.marked_on:
                drug_name = hover(drug_name,"%s %s %s"%(
                            wsa.marked_because,
                            fmt_time(wsa.marked_on),
                            wsa.marked_by,
                            ))
            from django.utils.html import format_html
            return format_html(u'{}{}',
                    vote_icon,
                    drug_name,
                    )
        table=Table(qs,[
                Table.Column('Molecule name',
                        extract=name_column_extractor,
                        ),
                Table.Column('Indication',
                        extract=lambda x:x.indication_link(),
                        ),
                Table.Column('Molecule Targets',
                        extract=lambda x:wsa2targets.get(x.id)
                        ),
                Table.Column('Target Names',
                        extract=lambda x:wsa2protnames.get(x.id)
                        ),
                Table.Column('Workspace Note',
                        extract=lambda x:note_format(x.get_study_text())
                        ),
                Table.Column('Global Note',
                        extract=lambda x:note_format(x.agent.get_bd_note_text())
                        ),
                ])
        self.context['table'] = table
        self.context['drug_counts'] = [
                (len(kt_ids),'clinically investigated or used treatments'),
                ]+[
                (qs_ws.filter(indication__in=s).count(),label)
                for s,label in [
                        (cands,'initial predictions'),
                        (rvwd,'reviewed predictions'),
                        (pats,'hits'),
                        ]
                ]

class ReviewSummaryView(DumaView):
    template_name='rvw/review_summary.html'
    index_dropdown_stem='rvw:review'

    def count_ws_values(self, data):
        from collections import Counter
        from .utils import prioritize_demerits
        demerit_ranks = prioritize_demerits()
        demerit2rank = {i+1:v for i,v in enumerate(demerit_ranks)}
        rank2demerit = {v: k for k, v in demerit2rank.items()}

        #Change demerit key to highest value based on demerit hierarchy
        for k, v in data.items():
            if v != 'Inactive Prediction':
                continue
            max_num = []
            for item in v:
                if item in demerit2rank:
                    max_num.append(demerit2rank[item])

            if max_num:
                highest_num = max(max_num)
                data[k] = [rank2demerit[highest_num]]
        for key, value in data.items():
            data[key] = data[key][0]

        return dict(Counter(list(data.values())))

    def count_election_values(self, data):
        from collections import Counter
        master_dict = {}
        secondary_dict = {}
        active = []
        e_dicts = {}
        from browse.models import WsAnnotation,Election
        for e in Election.objects.filter(ws = self.ws):
            if not e.id:
                continue
            e_dict = {}
            vote_qs = e.vote_set.filter(disabled = 0)
            assert e.ws_id == self.ws.id
            active_drugs = vote_qs.values_list('drug', flat = True).distinct()
            for item in active_drugs:
                if item not in data:
                    continue
                active.append(item)
                e_dict.setdefault(item, [])
                e_dict[item].append(data[item])
                e_dicts.setdefault(item, [])
                e_dicts[item].append(data[item])
            e_count_dict = self.count_ws_values(e_dict)
            master_dict.update(e_dict)
            if e.flavor=='pass2':
                secondary_dict.update(e_dict)
        qs_ws = WsAnnotation.objects.filter(ws=self.ws)
        from browse.models import Prescreen
        sb_dict = {}
        for wsa in qs_ws:
            pscr = wsa.marked_prescreen
            if not pscr:
                continue
            if wsa.id not in active:
                continue
            sb_dict.setdefault(pscr.id, [])
            sb_dict[pscr.id].append(wsa.id)
        sb_count_dict = {}
        for k, v in sb_dict.items():
            sb_count_dict[k] = len(v)

        master_sb_dict = []
        for k, v in sb_dict.items():
            tmp_dict = {}
            tmp_dict.setdefault(k, [])
            for item in v:
                tmp_dict[k].append(data[item])
            master_sb_dict.append(tmp_dict)

        tmp_dict = {}
        for k, v in sb_dict.items():
            tmp_dict.setdefault(k, [])
            for item in v:
                tmp_dict[k].append(e_dicts[item])

        for k, v in tmp_dict.items():
            tmp_dict[k] = [item for sublist in v for item in sublist]

        master_counter = {}
        for k, v in tmp_dict.items():
            master_counter.setdefault(k, {})
            master_counter[k].update(dict(Counter(v)))

        return (Counter(list(master_dict.values())),
                Counter(list(secondary_dict.values())),
                sb_count_dict,
                master_counter
               )

    def get_election_data(self):
        from browse.models import Election,Vote
        from collections import Counter
        e_dict = {}
        voter_dict = {}
        for e in Election.objects.filter(ws = self.ws):
            if not e.id:
                continue
            vote_qs = e.vote_set.filter(disabled = 0)
            assert e.ws_id == self.ws.id
            active_drugs = vote_qs.values_list('drug', flat = True).distinct()
            e_dict[e.id] = active_drugs
            vote_lists = {}
            for drug in active_drugs:
                qs = Vote.objects.filter(drug = drug)
                for v in qs:
                    if not v.election.active():
                        self.vote = v
                        voter_dict.setdefault(self.vote.reviewer, [])
                        voter_dict[self.vote.reviewer].append(self.vote.recommended)
        # utility: convert True/False/None votes to plot-friendly strings
        def convert_votes(votes):
            return [
                    'Yes' if vote else 'No'
                    for vote in votes
                    if vote is not None
                    ]
        # This block is for total votes pie chart
        total_votes = Counter(convert_votes([
                vote for person in voter_dict.values() for vote in person
                ]))

        # The rest of this function is for personal vote stacked bar graph
        personal_votes = {}
        for key, value in voter_dict.items():
            personal_votes[key] = Counter(convert_votes(value))

        return (total_votes, personal_votes)
    def _set_preferred_order(self, demerits):
        possible=demerits + [
                     v for v in self.indication_options
                     if v != 'Inactive Prediction'
                    ]
        self.final_inds = [
            'Patent submitted',
            'Preparing patent',
            'Hit'
            ]
        self.preferred_order = self.final_inds + [
# our predictions/hits
            'Initial Prediction',
            'Reviewed Prediction',
            'Hit',
            'In Vitro 1',
            'In Vitro 2',
            'In Vivo 1',
            'In Vivo 2',
# Non-novels
            'FDA Approved Treatment',
            'Clinically used treatment',
            'Phase 3 treatment',
            'Phase 2 treatment',
            'Phase 1 treatment',
            'Clinically investigated treatment',
            'Researched as treatment',
            'Hypothesized treatment',
            'Patented',
            'Non-novel class',
# exacerbating
            'FDA Documented Cause',
            'Clinically indicated cause',
            'Candidate Cause',
            'Researched as cause',
            'Exacerbating',
            'Tox',
#Other
            'Unavailable',
            'Modality',
            'Non-unique',
            'No MOA',
            'Data Quality',
            'Ubiquitous',
            'Unclassified'
           ]
        self.preferred_order += list(set(possible) - set(self.preferred_order))
    def plot(self, dictionary):
        from dtk.plot import PlotlyPlot, Color
        from collections import Counter, OrderedDict
        self.context['vote_plots'] = []
        self.context['summary_plots'] = []
        self.context['election_plots'] = []
        self.context['scoreboard_plots'] = []
        from .utils import prioritize_demerits
        demerits = prioritize_demerits()
        color_lst = Color()
        colors = color_lst.many_pie_slice_colors
        colors *= 2
        # Plot indications by ws pie chart
        indication_labels = []
        indication_values = []
        ws_indications = self.count_ws_values(dictionary)
        for key, value in ws_indications.items():
            indication_labels.append(key)
            indication_values.append(value)
        indication_zip = list(zip(indication_labels, indication_values))
        indication_zip.sort(key = lambda x: x[1], reverse = True)
        indication_labels = [x[0] for x in indication_zip]
        indication_values = [x[1] for x in indication_zip]
        self._set_preferred_order(demerits)
        color_zip = list(zip(self.preferred_order,
                    colors
                   ))
        assert len(color_zip) == len(self.preferred_order), "Need more colors"
        color_dict = {}
        for i, (label, value) in enumerate(color_zip):
            color_dict.setdefault(label, 0)
            color_dict[label] = colors[i]
        colors = [color_dict[label] for label in indication_labels]
        indication_layout = {'title' :
            'Total Workspace Indication Categories, n = {}'.format(len(dictionary))}
        total_indication_plot = ('indication_pie', PlotlyPlot(
                            [dict(labels = indication_labels,
                                  values = indication_values,
                                  marker = dict(colors=colors),
                                  textinfo = 'percent',
                                  hoverinfo = 'value+label',
                                  type='pie'
                                  )
                            ], indication_layout))
        e_total, second_total,sb_dict, sb_dicts = self.count_election_values(dictionary)
        # Plot bar chart of scoreboard IDs
        yes_preparing_patent = []
        no_preparing_patent = []
        patent_dict = {}
        rvwd_dict = {}
        non_patent_dict = {}

        for mk,mv in sb_dicts.items():
            patent_dict.setdefault(mk, 0)
            rvwd_dict.setdefault(mk, 0)
            non_patent_dict.setdefault(mk, 0)
            for k,v in mv.items():
                if k in ['Preparing patent', 'Hit']:
                    patent_dict[mk] += v
                elif k == 'Reviewed Prediction':
                    rvwd_dict[mk] += v
                else:
                    non_patent_dict[mk] += v

        patent_dict = OrderedDict(sorted(patent_dict.items()))
        rvwd_dict = OrderedDict(sorted(rvwd_dict.items()))
        non_patent_dict = OrderedDict(sorted(non_patent_dict.items()))

        names = [k for k in patent_dict]
        yes_patent_labels = [v for v in patent_dict.values()]
        rvwd_labels = [v for v in rvwd_dict.values()]
        no_patent_labels = [v for v in non_patent_dict.values()]
        all_labels = yes_patent_labels + rvwd_labels + no_patent_labels
        data = [yes_patent_labels, rvwd_labels, no_patent_labels]

        trace1 = dict(
            type='bar',
            x=names,
            y=yes_patent_labels,
            name='Selected')

        trace2 = dict(
            type='bar',
            x=names,
            y=rvwd_labels,
            name='Reviewed')

        trace3 = dict(
            type='bar',
            x=names,
            y=no_patent_labels,
            name='Rejected')

        data = [trace3, trace2, trace1]
        layout = dict(xaxis=dict(title='Prescreen ID',
                                 type='category',
                                 tickangle=0),
                      yaxis=dict(title='Number of Drugs'),
                      barmode = 'stack',
                      width=750,
                      title = 'Drugs Per Prescreen ID, n = {}'.format(sum((all_labels)))
                     )

        scoreboard_id_plot = ('patent_scoreboard_bar', PlotlyPlot(data, layout))

        # Plot pie chart of indications for every election
        if e_total:
            colors = [color_dict[label] for label in e_total.keys()]
            total_election_plot = ('election_total_pie', PlotlyPlot(
                            [dict(type='pie',
                                    labels = list(e_total.keys()),
                                    values = list(e_total.values()),
                                    textinfo = 'percent',
                                    hoverinfo = 'value+label',
                                    marker = dict(colors=colors),
                            )], {'sort':False,
                                 'title' : 'Total Election Indication Categories, n = {}'.format(
                                               sum(e_total.values()))}))
            if second_total:
                x = ['Prescreen', 'Prelim.', 'Secondary', 'Final']
                inputs = [ws_indications,
                           e_total,
                           second_total,
                           {k:v for k,v in second_total.items()
                            if k in self.final_inds
                           }
                         ]
                x_inset=0.333
            else:
                x = ['Prescreen', 'Prelim.', 'Final']
                inputs = [ws_indications,
                           e_total,
                           {k:v for k,v in e_total.items()
                            if k in self.final_inds
                           }
                         ]
                x_inset=0.5
            data = []
            data_zoom = []
            # our version of plotly wasn't stacking appropriately, so I did it myself
            running_total=[0.]*len(inputs)
            for i in self.preferred_order:
                y=[d.get(i,0) for d in inputs]
                if sum(y) == 0:
                    continue
                running_total=[y[j]+running_total[j]
                               for j in range(len(inputs))
                              ]
                data.append(dict(
                        x=x,
                        y=running_total,
                        name=i,
                        hoverinfo='text',
                        text=["%s: %d" %(i,yi) if yi else "" for yi in y],
                        mode='none',
                        fillcolor=color_dict[i],
                        fill='tonexty',
                    ))
                if sum(y[1:]) == 0:
                    continue
                data_zoom.append(dict(
                        x=x[1:],
                        y=running_total[1:],
                        name=i,
                        hoverinfo='text',
                        text=["%s: %d" %(i,yi) if yi else "" for yi in y][1:],
                        mode='none',
                        fillcolor=color_dict[i],
                        fill='tonexty',
                        xaxis='x2',
                        yaxis='y2',
                        showlegend=False,
                    ))
            funnel_plot = ('funnel', PlotlyPlot(
                            data+data_zoom,
                            {'title' : 'Indications across reviews',
                             'yaxis':{
                                       'title':'Number of predictions'
                                      },
                             'xaxis':{
                                       'title':'Review stage'
                                      },
                             'yaxis2':{'domain':[.25,1.],
                                       'anchor':'x2',
                                       'linewidth':1.5,
                                       'mirror':True,
                                       'rangemode':'tozero'
                                      },
                             'xaxis2':{'domain':[x_inset,1.],
                                       'anchor':'y2',
                                       'showticklabels':False,
                                       'linewidth':1.5,
                                       'mirror':True,
                                      },
                             'width':1000,
                             'height':550,
                            }
                           ))
            self.context['summary_plots'].extend((funnel_plot,
                                                  total_indication_plot,
                                                  total_election_plot,
                                                  scoreboard_id_plot))
        for key, value in sb_dicts.items():
            sb_labels = []
            sb_values = []
            title = key
            for k, v in value.items():
                sb_labels.append(k)
                sb_values.append(v)
            colors = [color_dict[label] for label in sb_labels]
            self.context['scoreboard_plots'].append(
                            ('scoreboard_pie{}'.format(title),
                            PlotlyPlot(
                        [dict(type='pie',
                                labels = sb_labels,
                                values = sb_values,
                                textinfo = 'percent',
                                hoverinfo = 'value+label',
                                marker = dict(colors=colors),
                        )], {'sort':False,
                             'title' : 'Prescreen {} Indication Categories, n = {}'.format(
                                           title, sum(value.values()))})))

        # Plot total votes pie chart
        total_votes, personal_votes = self.get_election_data()
        total_votes_labels = []
        total_votes_values = []
        if total_votes:
            for key, value in total_votes.items():
                total_votes_labels.append(key)
                total_votes_values.append(value)
            total_votes_layout = {'sort':False,
                                  'title' : 'Total Workspace Votes, n = {}'.format(
                                                sum(total_votes.values()))}
            self.context['vote_plots'].append(('total votes_pie', PlotlyPlot(
                            [dict(type='pie',
                                  labels = total_votes_labels,
                                  values = total_votes_values,
                                  textinfo = 'percent',
                                  hoverinfo = 'value+label'
                                  )], total_votes_layout)))
        # Plot stacked bar chart of individual voting records
        names = []
        no_values = []
        yes_values = []
        if personal_votes:
            for k, v in personal_votes.items():
                if v:
                    names.append(k)
                    no_values.append(v['No'])
                    yes_values.append(v['Yes'])
            trace1 = dict(type='bar',x = names, y = no_values, name = 'No')
            trace2 = dict(type='bar',x = names, y = yes_values, name = 'Yes')
            data = [trace1, trace2]
            layout = dict(xaxis=dict(title='Names'),
                          yaxis=dict(title='Number of Votes'),
                          barmode = 'stack',
                          title = 'User Votes Summary'
                         )
            self.context['vote_plots'].append(('personal_votes_bar',
                                                PlotlyPlot(
                                                 data, layout
                                                )
                                             ))
    def custom_context(self):
        from browse.models import WsAnnotation, Demerit
        wsa2ind = {}
        relevant_demerits = {
                d.id:d.desc
                for d in Demerit.objects.filter(
                        )
                }

        qs_ws = WsAnnotation.objects.filter(ws=self.ws)
        self.indication_options = None
        for wsa in qs_ws:
            if self.indication_options is None:
                self.indication_options = [x[1] for x
                                           in wsa.indication_vals.choices()
                                          ]
            ind = wsa.indication_label()
            if ind in ('Unclassified','Reviewed As Molecule'):
                continue
            if ind == 'Inactive Prediction':
                ind = []
                for d_id in wsa.demerits():
                    if int(d_id) in relevant_demerits:
                        ind.append(relevant_demerits[d_id])
                if not ind:
                    continue
            else:
                ind = [ind]
            wsa2ind[wsa.id] = ind
        self.plot(wsa2ind)

class ElectionView(DumaView):
    template_name = 'rvw/election.html'
    button_map={
            'save':['election'],
            'force':[],
            'reclassify':['reclassify'],
            }
    GET_parms = {
            'flavor':(str,''),
            }
    def custom_setup(self):
        self.context_alias(
                can_force_completion = self.in_group('duma_admin'),
                )
        # set up shortcut buttons
        # - preliminaries
        from browse.models import WsAnnotation,Election
        enum = WsAnnotation.indication_vals
        from nav.views import ReclassifyFormFactory
        rff = ReclassifyFormFactory()
        self.context_alias(shortcuts=[])


        if self.elec_id:
            e = Election.objects.get(id=self.elec_id)
            shortcut_id = 'primary_shortcut'
            indication = e.flavor_info.output
            shortcut_text = enum.get('label', indication)

            # - dynamically add handler for preclin shortcut
            self.add_shortcut_handler(
                    shortcut_id,
                    self.reclassify_post_valid,
                    dict(
                            indication=indication,
                            demerit=[],
                            indication_href='',
                            ),
                    )
            self.shortcuts+=[
                    ('primary',shortcut_id,shortcut_text),
                    ('','',''), # line break
                    ]
        # - dynamically add handlers and list entries for demerits
        self.context['reclassify_form_factory'] = rff
        for did,label in rff.demerit_choices:
            btn_name = u'demerit_shortcut_%d'%did
            self.shortcuts.append(('info',btn_name,label))
            self.add_shortcut_handler(
                    btn_name,
                    self.reclassify_post_valid,
                    dict(
                            indication=enum.INACTIVE_PREDICTION,
                            demerit=[did],
                            indication_href='',
                            ),
                    )
        # - add button map entries for all buttons
        self.button_map = dict(self.button_map)
        for _,name,_ in self.shortcuts:
            self.button_map[name] = ['reclassify']
    def custom_context(self):
        e_form=self.election_form
        if self.elec_id and e_form.instance.status() == 'Active':
            # pre-format candidate progress, which needs the demo flag
            self.context_alias(
                cand_progress=
                    e_form.instance.candidate_progress(self.request.user,self.is_demo()),
                )
        # attempt to retrieve candidate results
        try:
            results = e_form.instance.candidate_results(self.is_demo())
        except AttributeError:
            results = []

        wsa_ids = [x.id for x in e_form.wsa_list]
        import json
        self.context_alias(
                e_form=e_form, # alias for older template
                results=results,
                page_tab='selection(%d)' % self.ws.id,
                wsa_ids=json.dumps(wsa_ids)
                )

    def force_post_valid(self):
        if self.can_force_completion:
            from browse.models import Election
            e = Election.objects.get(id=self.elec_id)
            e.vote_set.filter(recommended__isnull=True).update(
                                                        disabled=True
                                                        )
            e.update_note_permissions()
        return HttpResponseRedirect(self.ws.reverse('rvw:election',self.elec_id))
    def make_reclassify_form(self,data):
        # this is tightly coupled with browse_tags.disposition_form
        if not data:
            # this case doesn't do anything; the template renders separate
            # copies of the empty forms using the disposition form tag.
            # this is here to decode on POST only (which is the only case
            # where we have a wsa_id available)
            return None
        wsa_id = int(data['wsa_id'])
        from browse.models import WsAnnotation
        wsa = WsAnnotation.objects.get(pk=wsa_id)
        from nav.views import ReclassifyFormFactory
        rff = ReclassifyFormFactory()
        FormClass = rff.get_form_class(wsa)
        return FormClass(data)
    # When the 'reclassify' button is clicked, the overrides parameter
    # will default. When a shortcut button is clicked, overrides holds
    # field values to assume in the POST.
    def reclassify_post_valid(self,overrides={}):
        p = self.reclassify_form.cleaned_data
        if overrides:
            p.update(overrides)
        wsa_id = p['wsa_id']
        indication = p['indication']
        demerit = p['demerit']
        indication_href = p['indication_href']
        from browse.models import WsAnnotation
        wsa=WsAnnotation.objects.get(pk=wsa_id)
        try:
            wsa.update_indication(
                        indication,
                        demerit,
                        self.request.user.username,
                        "election %d"%self.elec_id,
                        indication_href,
                        from_prescreen=wsa.marked_prescreen,
                        )
        except ValueError as ex:
            self.message(
                'Update failed: %s; re-enter change and try again'%str(ex)
                )
        return HttpResponseRedirect(self.ws.reverse('rvw:election',self.elec_id))
    def make_election_form(self,data):
        from .forms import ElectionForm
        return ElectionForm(
                self.ws,
                self.elec_id,
                self.is_demo(),
                self.flavor,
                data,
                )
    @classmethod
    def setup_election(cls, selected_users, selected_drugs, election):
        e = election
        # make sure every drug being reviewed has a review code
        for wsa in selected_drugs:
            if not wsa.review_code:
                wsa.set_review_code()
                wsa.save()
        # make sure all active user/drug combos have vote records
        from browse.models import Vote
        for u in selected_users:
            for wsa in selected_drugs:
                v,new = Vote.objects.get_or_create(
                            election=e,
                            reviewer=u.username,
                            drug=wsa,
                            )
                if v.disabled:
                    v.disabled=False
                    v.save()
        # now, disable any vote records outside the active lists
        e.vote_set.exclude(
                reviewer__in=[u.username for u in selected_users]
                ).update(disabled=True)
        e.vote_set.exclude(
                drug_id__in=[d.id for d in selected_drugs]
                ).update(disabled=True)
        # make sure any note permissions are correct
        e.update_note_permissions()

    def save_post_valid(self):
        p = self.election_form.cleaned_data
        from browse.models import Election,WsAnnotation
        if self.elec_id:
            e = Election.objects.get(id=self.elec_id)
            e.due = p['due_date']
            e.save()
        else:
            e = Election.objects.create(ws=self.ws
                                    ,due=p['due_date']
                                    ,flavor=self.flavor
                                    )
        from django.contrib.auth.models import User
        selected_users = [ User.objects.get(id=int(k[2:]))
                    for k,v in p.items()
                    if v and k.startswith('u_')
                    ]
        selected_drugs = [ WsAnnotation.objects.get(id=int(k[2:]))
                    for k,v in p.items()
                    if v and k.startswith('d_')
                    ]
        self.setup_election(selected_users, selected_drugs, e)
        return HttpResponseRedirect(self.ws.reverse('rvw:election',e.id))


@login_required
def get_drug_summary(request, ws_id, wsa_ids):
    import json
    from browse.models import WsAnnotation, Workspace
    from dtk.prot_map import AgentTargetCache, DpiMapping
    wsa_ids = wsa_ids.split(',')
    wsas = WsAnnotation.objects.filter(pk__in=wsa_ids)
    assert len(wsas) == len(wsa_ids)
    wsas = sorted(wsas, key=lambda x: wsa_ids.index(str(x.id)))
    ws = Workspace.objects.get(pk=ws_id)
    targ_cache = AgentTargetCache(
            mapping=DpiMapping(ws.get_dpi_default()),
            agent_ids=[x.agent_id for x in wsas],
            dpi_thresh=ws.get_dpi_thresh_default(),
            )

    from browse.models import Prescreen
    from runner.process_info import JobInfo
    from dtk.scores import Ranker
    psqs = Prescreen.objects.filter(ws=ws).order_by('id')
    def get_ranker(ps):
        bji = JobInfo.get_bound(ws,ps.primary_job_id())
        cat = bji.get_data_catalog()
        return Ranker(cat.get_ordering(ps.primary_code(),True))
    rankers = [get_ranker(ps) for ps in psqs]

    data = []
    for wsa in wsas:
        lst = targ_cache.info_for_agent(wsa.agent_id)
        lst = dict((x[1], ws.reverse('protein',x[0])) for x in lst)

        mark_pscr = wsa.marked_prescreen
        is_mark_pscr = [x == mark_pscr for x in psqs]
        ranks = [str(ranker.get(wsa.id)).center(7) for ranker in rankers]
        data.append({
            'wsa_id': wsa.id,
            'wsa_href': wsa.drug_url(),
            'canonical': wsa.agent.canonical,
            'targets': list(lst.items()),
            'ranks': ranks,
            'is_mark_pscr': is_mark_pscr,
            'prescreen_ids': [ps.id for ps in psqs],
            })

    # Find all elections that aren't this one, and find all drugs, targs
    from collections import defaultdict
    reviewed_targets = defaultdict(list)
    reviewed_targets_prelim = defaultdict(list)
    all_elected_wsas = WsAnnotation.objects.filter(vote__election__ws=ws).distinct()
    targ_cache = AgentTargetCache(
            mapping=DpiMapping(ws.get_dpi_default()),
            agent_ids=[x.agent_id for x in all_elected_wsas],
            dpi_thresh=ws.get_dpi_thresh_default(),
            )
    for wsa in all_elected_wsas:
        if wsa in wsas:
            continue
        lst = targ_cache.info_for_agent(wsa.agent_id)
        prelim_elecs = list(set(wsa.vote_set.filter(election__flavor='pass1').values_list('election_id', flat=True)))
        elecs = list(set(wsa.vote_set.exclude(election__flavor='pass1').values_list('election_id', flat=True)))
        for gene in set(x[1] for x in lst):
            if elecs:
                reviewed_targets[gene].append([
                    wsa.agent.canonical,
                    ws.reverse('rvw:election', elecs[0]),
                    ])
            elif prelim_elecs:
                # Only include this drug in prelim if it's not in final
                reviewed_targets_prelim[gene].append([
                    wsa.agent.canonical,
                    ws.reverse('rvw:election', prelim_elecs[0]),
                    ])

    return JsonResponse({
        'drug_summary_data': data,
        'reviewed_targets': reviewed_targets,
        'reviewed_targets_prelim': reviewed_targets_prelim,
        })



class AllReviewNotesView(DumaView):
    template_name='rvw/all_review_notes.html'
    def custom_context(self):

        user = self.request.user


        self.context_alias(
                mol_table=self.make_mol_table(user),
                prot_table=self.make_prot_table(user),
                )


    def _filter_non_latest(self, notes):
        # Filter out non-latest notes.  There might be a faster way to do this via
        # django filtering, but seems hard.
        from dtk.data import MultiMap
        timestamps = MultiMap((x['version_of'], x['created_on']) for x in notes)
        max_timestamps = {k: max(v) for k, v in timestamps.fwd_map().items()}
        notes = [x for x in notes if x['created_on'] == max_timestamps[x['version_of']]]
        return notes

    def make_prot_table(self, user):
        from notes.models import NoteVersion, Note
        from dtk.table import Table
        from django.db.models import F
        from django.urls import reverse
        notes = NoteVersion.objects.filter(created_by=user, version_of__targetreview__isnull=False).values(
                'text',
                'created_on',
                'version_of',
                ws_name=F('version_of__targetreview__target__ws__name'),
                ws_id=F('version_of__targetreview__target__ws__id'),
                uniprot=F('version_of__targetreview__target__uniprot'),
                )
        notes = self._filter_non_latest(notes)


        uniprots = [x['uniprot'] for x in notes]
        from browse.models import Protein
        u2g = Protein.get_uniprot_gene_map(uniprots)

        def extract_target(x):
            uniprot = x['uniprot']
            name = u2g.get(uniprot, uniprot)
            ws_id = x['ws_id']
            url = reverse('protein', args=[ws_id, uniprot])
            from dtk.html import link
            return link(name, url, new_tab=True)

        from django.utils.html import urlize
        note = Table.Column('Note', extract=lambda x: urlize(x['text']))
        date = Table.Column('Date', extract=lambda x: x['created_on'].strftime('%Y/%m/%d'))
        ws = Table.Column('WS', idx='ws_name')
        obj = Table.Column('Target', extract=extract_target)

        cols = [note, obj, date, ws]

        table = Table(notes, cols)
        return table

    def make_mol_table(self, user):
        from notes.models import NoteVersion, Note
        from browse.models import WsAnnotation
        from dtk.table import Table
        from django.db.models import F
        from django.urls import reverse
        notes = NoteVersion.objects.filter(created_by=user, version_of__vote__isnull=False).values(
                'text',
                'created_on',
                'version_of',
                ws_name=F('version_of__vote__drug__ws__name'),
                wsa_id=F('version_of__vote__drug'),
                ws_id=F('version_of__vote__drug__ws__id'),
                )


        notes = self._filter_non_latest(notes)

        wsa_ids = [x['wsa_id'] for x in notes]
        wsaid2name = dict(WsAnnotation.objects.filter(
                pk__in=wsa_ids,
                agent__tag__prop__name='canonical'
                ).values_list('id', 'agent__tag__value'))

        def extract_wsa(x):
            wsa_id = x['wsa_id']
            ws_id = x['ws_id']
            name = wsaid2name[wsa_id]
            url = reverse('moldata:annotate', args=[ws_id, wsa_id])
            from dtk.html import link
            return link(name, url, new_tab=True)

        from django.utils.html import urlize
        note = Table.Column('Note', extract=lambda x: urlize(x['text']))
        date = Table.Column('Date', extract=lambda x: x['created_on'].strftime('%Y/%m/%d'))
        ws = Table.Column('WS', idx='ws_name')
        obj = Table.Column('Molecule', extract=extract_wsa)

        cols = [note, obj, date, ws]

        table = Table(notes, cols)
        return table


class Prescreen2View(DumaView):
    template_name='rvw/prescreen.html'

    button_map={
            'markreview':['markreview'],
            }

    GET_parms = {
            'prescreen_id':(int,None),
        }

    def make_markreview_form(self, data):
        from browse.models import WsAnnotation
        choices = WsAnnotation.grouped_choices()
        class MyForm(forms.Form):
            review_note = forms.CharField(
                label='Notes For Reviewers',
                required=False,
                widget=forms.Textarea(attrs={'rows':'4','cols':'60'})
            )
        return MyForm(data)
    def make_reclassify_form(self,data):
        from nav.views import ReclassifyFormFactory
        rff = ReclassifyFormFactory()
        from browse.models import WsAnnotation
        enum = WsAnnotation.indication_vals
        FormClass = rff.get_form_class(
                self.wsa,
                top=(
                        enum.INITIAL_PREDICTION,
                        enum.INACTIVE_PREDICTION,
                        ),
                )
        # Build a list of demerits to pre-set based on drug flags.
        # Since demerits are defined dynamically through a table,
        # we can't hard-code ids here. Instead, create a list of
        # demerit names to set, and then convert them to ids.
        added_demerit_names = set()
        # TODO: Redo initial selection.
        """
        for flag in self.unformatted_prescreen_flags:
            # copy certain demerits from other workspaces
            if flag.category == 'Demerit':
                if flag.detail in (
                        'Ubiquitous',
                        'Unavailable',
                        ):
                    added_demerit_names.add(flag.detail)
            # flag unavailable if not for sale in ZINC
            if flag.category == 'ZINC labels':
                if flag.detail == 'Zinc ID but no label' \
                        or 'not-for-sale' in flag.detail:
                    added_demerit_names.add('Unavailable')
            # flag non-novel if there are unwanted important proteins
            if flag.category == 'Unwanted Important Protein':
                added_demerit_names.add('Non-novel class')
        """
        initial_overrides = {}
        if added_demerit_names:
            initial_overrides['demerit'] = \
                    self.wsa.demerits() | set([
                            code
                            for code,desc in rff.demerit_choices
                            if desc in added_demerit_names
                            ])
            initial_overrides['indication'] = enum.INACTIVE_PREDICTION

        return FormClass(
                data,
                initial=initial_overrides,
                )
    
    def _next_wsa(self):
        from .prescreen import PrescreenOrdering
        return PrescreenOrdering.next_mol_to_prescreen(self.pscr)

    def reclassify_post_valid(self,overrides={}):
        p = self.reclassify_form.cleaned_data
        wsa=self.wsa
        if overrides:
            p.update(overrides)
        wsa.update_indication(
                    p['indication'],
                    p['demerit'],
                    self.request.user.username,
                    self.pscr.marked_because(),
                    p['indication_href'],
                    from_prescreen=self.pscr,
                    )
        next_wsa = self._next_wsa()
        next_url = f"{self.ws.reverse('rvw:prescreen', next_wsa)}?prescreen_id={self.pscr.id}"
        return HttpResponseRedirect(next_url)

    def markreview_post_valid(self):
        p = self.markreview_form.cleaned_data
        wsa=self.wsa
        from browse.models import WsAnnotation
        ind = WsAnnotation.indication_vals.INITIAL_PREDICTION
        review_note = p['review_note']
        if review_note:
            from notes.models import Note
            Note.set(wsa,
                     'study_note',
                     self.request.user.username,
                     review_note,
                     )
        wsa.update_indication(
                    ind,
                    None,
                    self.request.user.username,
                    self.pscr.marked_because(),
                    '',
                    from_prescreen=self.pscr,
                    )
        next_wsa = self._next_wsa()
        next_url = f"{self.ws.reverse('rvw:prescreen', next_wsa)}?prescreen_id={self.pscr.id}"
        return HttpResponseRedirect(next_url)
    

    def custom_setup(self):
        if self.prescreen_id:
            from browse.models import Prescreen
            self.pscr = Prescreen.objects.get(pk=self.prescreen_id)
        else:
            self.pscr = None
        self.setup_ind_shortcuts()
    
    def setup_ind_shortcuts(self):
        from nav.views import ReclassifyFormFactory
        from browse.models import WsAnnotation
        iv = WsAnnotation.indication_vals
        rff = ReclassifyFormFactory()
        shortcut_list = []
        for did,label in rff.demerit_choices:
            button_name = 'demerit_shortcut_%d'%did
            shortcut_list.append(('info',button_name,label))
            self.add_shortcut_handler(
                    button_name,
                    self.reclassify_post_valid,
                    overrides=dict(
                            indication=iv.INACTIVE_PREDICTION,
                            demerit=[did],
                            indication_href='',
                            )
                    )
        self.context_alias(shortcut_list=shortcut_list)
        self.button_map['reclassify'] = ['reclassify']
        for _,name,_ in shortcut_list:
            self.button_map[name] = ['reclassify']
        

    def custom_context(self):
        from .prescreen import PrescreenData, PrescreenCacher
        data = PrescreenData.data_for_moa(self.wsa, self.pscr)
        self.context_alias(**data, prefetched_count=PrescreenCacher.prefetched_count(self.pscr))
        PrescreenCacher.launch_precache(self.pscr)


class HitClusterView(DumaView):
    template_name='rvw/hitclusters.html'

    button_map={
            'show':['config'],
            }

    GET_parms = {
            'ps':(list_of(str),[]),
            'ds':(list_of(str),[]),
            'prot_sig_id':(str,None),
            'path_sig_id':(str,None),
            'top_n_prots':(int,500),
            'top_n_pws':(int,200),
            'apply_dis_score':(boolean,False),
        }
    
    def make_protsig_choices(self):
        choices = []
        glf_choices = self.ws.get_prev_job_choices('glf')
        for jid, _ in glf_choices:
            bji = JobInfo.get_bound(self.ws, jid)
            try:
                input_jid, input_code = bji.upstream_jid(bji.job.settings())
            except KeyError:
                # This job doesn't know its own input score, skip it.
                self.message("Skipping glf job %s, no input"% jid)
                continue
            input_bji = JobInfo.get_bound(self.ws, input_jid)
            choice = (
                f'{input_jid}_{input_code}',
                f'{input_bji.role_label()} {input_code} ({input_jid})',
                )
            if choice not in choices:
                choices.append(choice)
        return choices


    def make_config_form(self, data):
        from browse.models import WsAnnotation
        choices = WsAnnotation.grouped_choices()
        ps_choices = self.ws.get_uniprot_set_choices()
        ds_choices = self.ws.get_wsa_id_set_choices(retro=True)
        prot_sig_choices = self.make_protsig_choices()
        path_sig_choices = self.ws.get_prev_job_choices('glf')
        class MyForm(forms.Form):
            ps = forms.MultipleChoiceField(
                        label='Prot Sets',
                        initial=self.ps,
                        choices=ps_choices,
                        required=False,
                        widget=forms.SelectMultiple(
                                attrs={'size':min(8, len(ps_choices))}
                                ),
            )
            ds = forms.MultipleChoiceField(
                        label='Mol Sets',
                        initial=self.ds,
                        choices=ds_choices,
                        required=False,
                        widget=forms.SelectMultiple(
                                attrs={'size':min(8, len(ps_choices))}
                                ),
            )
            prot_sig_id = forms.ChoiceField(
                label='Prot Sigs',
                initial=self.prot_sig_id,
                choices=prot_sig_choices,
                help_text='For identifying which indirect prots are worth comparing (top N prots from this score)',
            )
            path_sig_id = forms.ChoiceField(
                label='Pathway Sigs',
                initial=self.path_sig_id,
                choices=path_sig_choices,
                help_text='For identifying which pathways are worth comparing (top N pathways from this score)',
            )
            top_n_prots = forms.IntegerField(
                label='Top N prots',
                initial=self.top_n_prots,
            )
            top_n_pws = forms.IntegerField(
                label='Top N pathways',
                initial=self.top_n_pws,
            )
            apply_dis_score = forms.BooleanField(
                label='Apply Disease Scores',
                initial=self.apply_dis_score,
                required=False,
                help_text='Multiplies disease scores into heatmap outputs',
            )
        return MyForm(data)

    def show_post_valid(self):
        p = self.config_form.cleaned_data
        p['ps'] = ','.join(p['ps'])
        p['ds'] = ','.join(p['ds'])
        return HttpResponseRedirect(self.here_url(**p))
    
    def make_prot_sig(self):
        jid,code = self.prot_sig_id.split('_')
        bji = JobInfo.get_bound(self.ws, jid)
        cat = bji.get_data_catalog()
        dis_prot_sig = {prot:ev for prot, ev in cat.get_ordering(code,True)}
        return dis_prot_sig

    def make_path_sig(self):
        jid = self.path_sig_id
        code = 'wFEBE'
        bji = JobInfo.get_bound(self.ws, jid)
        cat = bji.get_data_catalog()
        out = {pw:ev for pw, ev in cat.get_ordering(code,True)}

        ps_id = bji.parms['std_gene_list_set']
        return out, ps_id

    def custom_context(self):
        import dtk.hit_clustering as hc

        plots = []
        if (self.ds or self.ps) and self.prot_sig_id:
            dis_prot_sig = self.make_prot_sig()
            dis_path_sig, ps_id = self.make_path_sig()

            from dtk.scores import Ranker
            prot_ranker = Ranker(sorted(dis_prot_sig.items(), key=lambda x: -x[1]))
            path_ranker = Ranker(sorted(dis_path_sig.items(), key=lambda x: -x[1]))

            name_to_prots = {}
            name_to_wsas = {}
            for ps in self.ps:
                prots = self.ws.get_uniprot_set(ps)
                for prot in prots:
                    name_to_prots[prot] = [prot]
            

            for ds in self.ds:
                wsa_ids = self.ws.get_wsa_id_set(ds)
                from browse.models import WsAnnotation
                wsas = WsAnnotation.objects.filter(pk__in=wsa_ids)
                from dtk.prot_map import AgentTargetCache, MultiAgentTargetCache
                atc = AgentTargetCache.atc_for_wsas(wsas, ws=self.ws)
                matc = MultiAgentTargetCache(atc)
                for wsa in wsas:
                    cur_atc = matc.atc_for_agent(wsa.agent_id)
                    prots = [x[1] for x in cur_atc.raw_info_for_agent(wsa.agent_id)]
                    name = wsa.get_name(False)
                    name_to_prots[name] = prots
                    name_to_wsas[name] = wsa.id

            plots = []
            for feature in [hc.IndirectTargets, hc.Pathways, hc.RelatedDiseases, hc.StructSimilarity]:
                if feature == hc.IndirectTargets:
                    click_type = 'protpage'
                elif feature == hc.Pathways:
                    click_type = 'pathwaypage'
                else:
                    click_type = None
                df = hc.cluster(
                    ws=self.ws,
                    dis_prot_sig=dis_prot_sig,
                    dis_path_sig=dis_path_sig,
                    name_to_prots=name_to_prots,
                    name_to_wsas=name_to_wsas,
                    top_n_pws=self.top_n_pws,
                    top_n_prots=self.top_n_prots,
                    ps_id=ps_id,
                    feature_types=[feature],
                    apply_dis_score=self.apply_dis_score,
                )
                from browse.models import Protein
                u2g = Protein.get_uniprot_gene_map()
                from dtk.gene_sets import get_pathway_id_name_map
                pw2name = get_pathway_id_name_map()

                def do_u2g(l):
                    def lu(x):
                        if x in u2g:
                            name = u2g[x]

                            rank = prot_ranker.get(x)
                            return f'({rank}) {name}'
                        return x
                    return [lu(x) for x in l]

                from dtk.url import pathway_url_factory
                pwurl = pathway_url_factory()
                def do_pw2name(l):
                    def lu(x):
                        if x in pw2name:
                            name = pw2name[x]

                            rank = path_ranker.get(x)
                            return f'({rank}) {name}'
                            # This seems to really break the autoscaling of the image and over-reserves space
                            # for text.
                            #url = pwurl(x)
                            #return f"<a href='{url}'>{name}</a>"
                        return x

                    return [lu(x) for x in l]
                
                def do_names(l):
                    return do_pw2name(do_u2g(l))

                if len(df.columns) == 0 or len(df.index) == 0:
                    self.message(f"Skipping  {feature.__name__}, shape is {df.values.shape}")
                    continue
                from dtk.plot import plotly_heatmap
                cur_plots = []
                plot = plotly_heatmap(
                    df.values.T,
                    col_labels=do_names(df.index.tolist()),
                    row_labels=do_names(df.columns.tolist()),
                    colorscale='Picnic',
                    color_zero_centered=True,
                    reorder_cols=True,
                    reorder_rows=True,
                    click_type=click_type,
                    id_lookup=df.columns.tolist(),
                )
                plot._layout['xaxis']['tickangle'] = 45
                cur_plots.append(plot)

                if feature != hc.StructSimilarity:
                    from scipy.spatial.distance import pdist,squareform
                    # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
                    # for different distance metrics.
                    # Ideally we want one that outputs a 0-1 distance (so we can easily invert to get a similarity).
                    # cosine distance might be nice, except it has some degeneracies that heatmap doesn't like.
                    # braycurtis is pretty close to a continuous version of 1.0-dice which is similar to our usual jaccard.
                    import numpy as np
                    plot2 = plotly_heatmap(
                        1.0 - np.nan_to_num(squareform(pdist(df.values, 'braycurtis'))),
                        col_labels=do_names(df.index.tolist()),
                        row_labels=do_names(df.index.tolist()),
                        colorscale='Picnic',
                        color_zero_centered=True,
                        reorder_cols=True,
                        reorder_rows=True,
                    )
                    cur_plots.append(plot2)

                plots.append((feature.__name__, cur_plots, feature.description))
        self.context_alias(plotly_plots=plots)

class AnimalModelCompareView(DumaView):
    template_name='rvw/animal_model_compare.html'

    button_map={
            'show':['config'],
            }

    GET_parms = {
            'ds':(str,'selected'),
            'dpi':(str, None),
            'path_sig_id':(str,None),
            'amod_ss_id':(str,None),
            'top_n_prots':(int,1000),
            'top_n_pws':(int,1000),
        }

    def make_amod_choices(self):
        from dtk.text import fmt_time
        from browse.models import ScoreSet
        choices = []
        wf_choices = self.ws.get_prev_job_choices('wf')
        for jid, desc in wf_choices:
            bji = JobInfo.get_bound(self.ws, jid)
            if 'AnimalModelFlow' not in bji.job.name:
                continue

            if not bji.get_scoreset():
                continue

            choices.append((bji.get_scoreset().id, f'AnimalModelWF {jid} {bji.job.started}'))
        return choices

    def make_config_form(self, data):
        from browse.models import WsAnnotation
        from dtk.prot_map import DpiMapping
        choices = WsAnnotation.grouped_choices()
        ds_choices = self.ws.get_wsa_id_set_choices(retro=True)
        path_sig_choices = self.ws.get_prev_job_choices('glf')
        class MyForm(forms.Form):
            dpi = forms.ChoiceField(
                label = 'DPI dataset',
                choices = DpiMapping.choices(self.ws),
                initial = self.dpi or self.ws.get_dpi_default(),
                )
            ds = forms.ChoiceField(
                        label='Mol Set',
                        initial=self.ds,
                        choices=ds_choices,
                        required=True,
            )
            path_sig_id = forms.ChoiceField(
                label='Human Pathways Score',
                initial=self.path_sig_id,
                choices=path_sig_choices,
            )
            amod_ss_id = forms.ChoiceField(
                label='AnimalModel Scoreset',
                initial=self.amod_ss_id,
                choices=self.make_amod_choices(),
            )
            top_n_prots = forms.IntegerField(
                label='Top N prots',
                initial=self.top_n_prots,
            )
            top_n_pws = forms.IntegerField(
                label='Top N pathways',
                initial=self.top_n_pws,
            )
        return MyForm(data)

    def show_post_valid(self):
        p = self.config_form.cleaned_data
        return HttpResponseRedirect(self.here_url(**p))

    def custom_context(self):
        if not self.amod_ss_id:
            return
        from rvw.animal_model_compare import compare
        from browse.default_settings import DpiDataset, PpiDataset, GeneSets

        ppi = PpiDataset.value(self.ws)
        pathways = GeneSets.value(self.ws)

        cmp_table, details, _, pathway_corrs, n_paths = compare(
            ws=self.ws,
            ds=self.ds,
            dpi=self.dpi,
            ppi=ppi,
            pathways=pathways,
            amod_ss_id=self.amod_ss_id,
            glf_jid=self.path_sig_id,
            top_n_prots=self.top_n_prots,
            top_n_pathways=self.top_n_pws,
        )
        from dtk.html import glyph_icon
        from django.utils.safestring import mark_safe
        total_glyph=mark_safe(glyph_icon('info-sign',hover=f'Number of pathways with human score >0: {"{:,}".format(n_paths[0])}'))
        top_glyph=mark_safe(glyph_icon('info-sign',
                    hover=f'Number of pathways with human score >0: and ranked in the top {"{:,}".format(self.top_n_pws)} human pathways: {"{:,}".format(n_paths[1])}'))

        self.context_alias(
            cmp_table=cmp_table,
            details=details,
            pathway_corrs=pathway_corrs,
            total_glyph=total_glyph,
            top_glyph=top_glyph,
            )


class DefusDetailsView(DumaView):
    template_name='rvw/defus_details.html'

    button_map={
            'show':['config'],
            }

    GET_parms = {
            'ref_jid':(int,None),
            'wsa_id':(int,None),
        }

    def make_config_form(self, data):
        from dtk.html import WsaInput
        job_choices = self.ws.get_prev_job_choices('defus')
        class MyForm(forms.Form):
            ref_jid = forms.ChoiceField(
                label = 'Ref JID',
                choices = job_choices,
                initial = self.ref_jid,
                )
            wsa_id = forms.IntegerField(
                        label='Molecule',
                        initial=self.wsa_id,
                        widget=WsaInput(ws=self.ws),
                        required=True,
            )
        return MyForm(data)

    def show_post_valid(self):
        p = self.config_form.cleaned_data
        return HttpResponseRedirect(self.here_url(**p))
    
    def custom_context(self):
        if not self.ref_jid or not self.wsa_id:
            return
        
        from scripts.newdefus import make_metasim, run_data
        from algorithms.run_defus import gen_defus_settings
        from browse.models import WsAnnotation

        bji = JobInfo.get_bound(self.ws, self.ref_jid)

        import os
        if not os.path.exists(bji.outsims):
            self.message("This DEFUS job doesn't have similarity metadata, try a newer one or rerunning")
            return

        ref_scores = bji._gen_ref_scores()

        wsa = WsAnnotation.all_objects.get(pk=self.wsa_id)
        agent_id = wsa.agent_id
        ref_sim_fn = bji.outsims
        from dtk.arraystore import list_arrays, get_array
        arr_names = list_arrays(ref_sim_fn)
        
        from dtk.table import Table

        ref_agent_ids = set()

        ref_urls = {}

        def ref_extract(x):
            return ref_urls.get(x, x)

        cols = [
            Table.Column('Input ID', extract=ref_extract),
            Table.Column('Input pval', extract=lambda x: f"{ref_scores.get(x, '')[0]:.3g}"),
            Table.Column('Input coef', extract=lambda x: f"{ref_scores.get(x, '')[1]:.3g}"),
        ]

        all_col_ids = set()

        data = {}
        details = {}
        for arr_name in arr_names:
            mat, meta = get_array(ref_sim_fn, arr_name)
            from dtk.features import SparseMatrixWrapper
            data[arr_name] = SparseMatrixWrapper(mat, meta['row_names'], meta['col_names'])
            ref_agent_ids.update(meta['row_names'])
            all_col_ids.update(meta['col_names'])
            if 'details' in meta:
                details[arr_name] = meta['details']
                
        from functools import lru_cache

        @lru_cache()
        def extract_sim(ref_agent_id, data_arr):
            try:
                return f'{data_arr[ref_agent_id, agent_id]:.2}'
            except KeyError:
                return ''

        from scripts.newdefus import single_score
        @lru_cache()
        def extract_score(ref_agent_id, data_arr):
            try:
                sim = data_arr[ref_agent_id, agent_id]
                score = single_score(ref_agent_id, ref_scores, sim)
                return f'{score:.2}'
            except KeyError:
                return ''
        
        def extract_details(ref_agent_id, data_arr):
            try:
                from drugs.models import Blob, Drug
                smiles = data_arr[str(ref_agent_id)][str(agent_id)]
                
                drug_ids = Blob.objects.filter(value=smiles).values_list('drug_id', flat=True)
                wsas = list(WsAnnotation.all_objects.filter(ws=self.ws, agent__in=drug_ids))
                if wsas:
                    return wsas[0].html_url()
                else:
                    return Drug.objects.get(pk=drug_ids[0]).canonical
            except KeyError:
                return ''
            except Exception as e:
                return str(e)

        from functools import partial
        for arr_name in arr_names:
            col = Table.Column(arr_name + ' sim', extract=partial(extract_sim, data_arr=data[arr_name]))
            cols.append(col)
            col = Table.Column(arr_name + ' score', extract=partial(extract_score, data_arr=data[arr_name]))
            cols.append(col)
            if arr_name in details:
                col = Table.Column(arr_name + ' details', extract=partial(extract_details, data_arr=details[arr_name]))
                cols.append(col)
        
        
        ref_wsas = WsAnnotation.all_objects.filter(ws=self.ws, agent__in=ref_agent_ids)
        ref_wsas = WsAnnotation.prefetch_agent_attributes(ref_wsas, prop_names=['canonical', 'override_name'])
        for wsa in ref_wsas:
            ref_urls[wsa.agent_id] = wsa.html_url()
        
        table = Table(ref_agent_ids, cols)
        self.context_alias(
            defus_table=table,
            wsa_name=wsa.agent.canonical,
            )
