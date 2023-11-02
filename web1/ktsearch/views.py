from django import forms
from django.http import HttpResponseRedirect

from dtk.duma_view import DumaView,list_of,boolean
from dtk.table import Table,SortHandler

from dtk.table import Table
from dtk.html import link
from dtk.duma_view import qstr

from .models import KtSearch,KtSource,KtSearchResult,KtResultGroup
from .sources import KtSourceType

import logging
import six
logger = logging.getLogger(__name__)

from collections import namedtuple
IndData=namedtuple('IndData','ind_val sort_key ind_label count')

def build_ind_data_list(ind_count_pairs):
    from .models import IndicationMapper
    im = IndicationMapper()
    from browse.models import WsAnnotation
    enum=WsAnnotation.indication_vals
    l = [
            IndData(
                ind_val,
                im.sort_key(ind_val),
                enum.get('label',ind_val),
                count,
            )
            for ind_val,count in ind_count_pairs
            ]
    l.sort(key=lambda x:x.sort_key,reverse=True)
    return l

class BulkNameResolveView(DumaView):
    template_name='ktsearch/bulk_name_resolve.html'
    index_dropdown_stem='kts_search'
    button_map={
            'reject':[],
            }
    GET_parms={
            'limit':(int, 500),
            }
    def custom_context(self):
        name_list = KtSearchResult.ordered_unmatched_names(self.search)
        # We only need one proxy record per name, but if the list limiting
        # kicks in, we want to select the first N names that we'd see in
        # the individual resolution flow
        filtered = []
        seen = set()
        for item in name_list:
            if item.name in seen:
                continue
            seen.add(item.name)
            filtered.append(item)
        name_list = filtered
        # figure out what we're displaying
        got = len(name_list)
        if got > self.limit:
            label = f'First {self.limit} of {got} Distinct'
        elif got == 0:
            label = f'No'
        else:
            label = f'All {got} Distinct'
        # now apply the limit and put it in alpha order
        name_list = sorted(name_list[:self.limit],key=lambda x:x.name)
        if name_list:
            # extend returned namedtuple with additional info;
            # this is protected by the if to assure name_list[0] is valid
            from collections import namedtuple
            RowType = namedtuple(
                    'RowType',
                    name_list[0]._fields+('link',),
                    )
        from dtk.url import google_search_url
        from dtk.html import link
        name_list = [
                RowType(
                        *x,
                        link(
                                'Google search',
                                google_search_url([x.name]),
                                new_tab=True,
                                ),
                        )
                for x in name_list
                ]
        self.context_alias(
                name_list = name_list,
                quant_label = label,
                )
    def get_rejections(self):
        prefix = 'rej_'
        result = []
        for k,v in self.request.POST.items():
            v = v.strip()
            if not k.startswith(prefix) or not v:
                continue
            result.append((int(k[len(prefix):]),v))
        return result
    def reject_post_valid(self):
        for key,reason in self.get_rejections():
            # retrieve stand-in result
            item=KtSearchResult.objects.get(pk=key)
            # mark all matching names
            KtSearchResult.objects.filter(
                        query__search=self.search,
                        drugname=item.drugname,
                        group__isnull=True,
                        unmatch_confirmed=False,
                    ).update(
                        group=None,
                        unmatch_confirmed=True,
                    )
        return HttpResponseRedirect(self.here_url())

class NameResolveView(DumaView):
    template_name='ktsearch/name_resolve.html'
    index_dropdown_stem='kts_search'
    GET_parms = {
            'search_term':(str,''),
            'sort':(SortHandler,'name'),
            'pattern_anywhere':(boolean,False),
            }
    button_map = {
            'unmatch':[],
            'search':['search'],
            'match':['wsa'],
            }
    def custom_setup(self):
        name_list = KtSearchResult.ordered_unmatched_names(self.search)
        if name_list:
            self.drugname = name_list[0].name
        else:
            self.drugname = None
        if not self.search_term and self.drugname:
            self.search_term = self.drugname
    def custom_context(self):
        if self.drugname:
            import dtk.url as durl
            from dtk.html import link
            self.context['search_links'] = [
                    link('Pubchem',
                            durl.pubchem_search_url([self.drugname]),
                            new_tab=True
                            ),
                    link('Wikipedia',
                            durl.wikipedia_drug_search_url(self.drugname),
                            new_tab=True
                            ),
                    link('Google',
                            durl.google_search_url([self.drugname]),
                            new_tab=True
                            ),
                    link('GlobalData',
                            durl.globaldata_drug_search_url(
                                    self.drugname.replace('-',''),
                                    ),
                            new_tab=True
                            ),
                    ]
            self.context_alias(
                    shared_name_srs = [(link(sr.label(), sr.href, new_tab=True)
                                        ,sr
                                        )
                                        for sr in KtSearchResult.objects.filter(
                                            query__search=self.search,
                                            drugname=self.drugname,
                                        )
                                      ],
                    )
            from .models import IndicationMapper
            im = IndicationMapper()
            from browse.models import WsAnnotation
            enum=WsAnnotation.indication_vals
            for sr in self.shared_name_srs:
                sr[1].sort_key = im.sort_key(sr[1].ind_val)
                sr[1].indication_label = enum.get('label',sr[1].ind_val)
            self.shared_name_srs.sort(
                    key=lambda x:x[1].sort_key,
                    reverse=True,
                    )
        if self.search_term:
            from browse.models import WsAnnotation
            from browse.utils import drug_search
            MAX_MATCHES = 200
            drugs = drug_search(
                    version=self.ws.get_dpi_version(),
                    pattern=self.search_term,
                    pattern_anywhere=self.pattern_anywhere,
                    )
            drugs = drugs[:MAX_MATCHES]
            if len(drugs) == MAX_MATCHES:
                self.context_alias(too_many_results = True)
            self.build_search_result_table(drugs)
    def unmatch_post_valid(self):
        self.update_matching_search_results(None,True)
        return HttpResponseRedirect(self.here_url(search_term=None,sort=None))
    def make_search_form(self,data):
        class MyForm(forms.Form):
            search_term = forms.CharField(
                    initial=self.search_term,
                    )
            pattern_anywhere = forms.BooleanField(
                        label='Search non-prefixes (slow)',
                        required=False,
                        initial=self.pattern_anywhere,
                    )

        return MyForm(data)
    def search_post_valid(self):
        p = self.search_form.cleaned_data
        return HttpResponseRedirect(self.here_url(**p))
    def make_wsa_form(self,data):
        # this form is only used for parsing the POST data; the HTML is
        # rendered manually in the template with a copy of the form for
        # each search result row
        class MyForm(forms.Form):
            wsa_id = forms.IntegerField(
                    widget=forms.HiddenInput(),
                    )
        return MyForm(data)
    def match_post_valid(self):
        # Link wsa with search result; redirect should get next name.
        # Note that this doesn't touch the wsa indication. That gets
        # handled in the original resolve page.
        wsa_id = self.wsa_form.cleaned_data['wsa_id']
        grp,created = KtResultGroup.objects.get_or_create(
                search=self.search,
                wsa_id=wsa_id,
                )
        self.update_matching_search_results(grp,False)
        return HttpResponseRedirect(self.here_url(search_term=None,sort=None))
    def update_matching_search_results(self,group,confirmed):
        # update all remaining unresolved mismatches for the same drugname
        KtSearchResult.objects.filter(
                    query__search=self.search,
                    drugname=self.drugname,
                    group__isnull=True,
                    unmatch_confirmed=False,
                ).update(
                    group=group,
                    unmatch_confirmed=confirmed,
                    )
    def build_search_result_table(self, drugs):
        # XXX This is a very basic initial display of search results. There
        # XXX are many possible extensions:
        # XXX - add columns for more drug attributes; maybe even make the
        # XXX   columns dynamically configurable
        # XXX - add a column showing which attribute matched the search term
        # XXX - possibly pre-populate the table with a drug name search;
        # XXX   this would help if the drug wasn't matched due to multiple
        # XXX   results
        # XXX It may also be helpful to show agents that match the search,
        # XXX but aren't included in the workspace.

        from drugs.views import DrugSearchView
        table = DrugSearchView.make_search_table(
                drugs,
                self.ws.id,
                show_import=True,
                show_collection=True,
                )
        self.context_alias(
                search_table = table
                )

class ResolveView(DumaView):
    template_name='ktsearch/resolve.html'
    index_dropdown_stem='kts_search'
    button_map = {
            'update':['check_id','note'],
            'ignore':['check_id','note'],
            'reclassify':['check_id', 'indi','note','cta'],
            }
    def get_previous(self):
        # figure out the previous resolution by this user, if any
        try:
            last_group = KtResultGroup.objects.filter(
                    ktsearchresult__query__search=self.search,
                    user=self.request.user.username,
                    ).order_by('-timestamp')[0]
        except IndexError:
            return
        wsa = last_group.wsa
        from dtk.html import link
        self.previous_drug = link(
                        wsa.get_name(self.is_demo()),
                        wsa.drug_url(),
                        new_tab=True,
                        )
    def custom_setup(self):
        self.get_previous()
        # get all unresolved groups
        rg_qs = KtResultGroup.objects.filter(
                    ktsearchresult__query__search=self.search,
                    resolution=KtResultGroup.resolution_vals.UNRESOLVED,
                    ).distinct()
        group_by_id={
                rg.id:rg
                for rg in rg_qs
                }
        # sort the groups by proposed indication
        from dtk.data import MultiMap
        mm = MultiMap(rg_qs.values_list('id','ktsearchresult__ind_val'))
        id2inds = mm.fwd_map()
        from .models import IndicationMapper
        im = IndicationMapper()
        ordering = sorted(
                list(group_by_id.keys()),
                key=lambda x:max(im.sort_key(y) for y in id2inds[x]),
                reverse=True,
                )
        groups = [group_by_id[x] for x in ordering]
        # go through the unresolved groups, auto-resolving them if
        # possible, and stop at the first group needing manual intervention
        # (note in this case we don't call mark_resolver)
        for group in groups:
            group.cache_evidence()
            if group.proposed_indication == group.wsa.indication:
                group.resolution = group.resolution_vals.MATCHED_EXISTING
                group.save()
                continue
            # XXX if a group exists for this name in a previous search,
            # XXX auto-resolve with a new "previously_skipped" status,
            # XXX unless the current search contains any new evidence
            self._prep_selected_group(group)
            break
        if not hasattr(self, 'group'):
            self.button_map = {}
    def make_check_id_form(self,data):
        class MyForm(forms.Form):
            group_id = forms.IntegerField(
                    initial=self._get_group_id_initial(),
                    widget=forms.HiddenInput(),
                    )
        return MyForm(data)
    def _get_group_id_initial(self):
        try:
            return self.group.id
        except AttributeError:
            return None
    def make_indi_form(self,data):
        wsa = self.group.wsa
        indi_choices=wsa.grouped_choices()
        class FormClass(forms.Form):
            indication = forms.ChoiceField(
                        choices=indi_choices,
                        initial=wsa.indication,
                        )
            indication_href = forms.CharField(
                        initial=wsa.indication_href,
                        required=False,
                            # this is no longer required, because it
                            # can be imputed; if it isn't supplied when
                            # needed, that's now caught by having
                            # update_indication inside the try block
                        )
        return FormClass(data)
    def make_cta_form(self,data):
        class FormClass(forms.Form):
            # XXX these fields are copied from the moldata annotation view
            ph2_status = forms.ChoiceField(
                        label = 'Phase 2 Status',
                        choices = self.cta.ct_status_vals.choices(),
                        initial = self.cta.ph2_status
                        )
            ph2_url = forms.CharField(
                        required=False,
                        label = 'Phase 2 URL',
                        initial = self.cta.ph2_url,
                        widget = forms.TextInput(attrs={'size':'60'}),
                        )
            ph3_status = forms.ChoiceField(
                        label = 'Phase 3 Status',
                        choices = self.cta.ct_status_vals.choices(),
                        initial = self.cta.ph3_status
                        )
            ph3_url = forms.CharField(
                        required=False,
                        label = 'Phase 3 URL',
                        initial = self.cta.ph3_url,
                        widget = forms.TextInput(attrs={'size':'60'}),
                        )
        return FormClass(data)
    def make_note_form(self,data):
        class FormClass(forms.Form):
            # XXX these fields are copied from the moldata annotation view
            study_note = forms.CharField(
                    widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
                    required=False,
                    initial = self.group.wsa.get_study_text(),
                    label='Workspace note',
                    )
        return FormClass(data)
    def add_imputed_indication(self,ind_data,cta_data,wsa):
        # only impute if an indication hasn't been set
        if wsa.indication != wsa.indication_vals.UNCLASSIFIED:
            return
        # don't impute if indication or href are explicitly set in this post
        if int(ind_data['indication']) != wsa.indication:
            return
        if ind_data['indication_href'] != wsa.indication_href:
            return
        # Impute indication from Ph2/3 status
        trialed = set([
                self.cta.ct_status_vals.PASSED,
                self.cta.ct_status_vals.ONGOING,
                self.cta.ct_status_vals.FAILED,
                ])
        # find max completed (2 or 3)
        if int(cta_data['ph3_status']) in trialed:
            imputed = wsa.indication_vals.TRIALED3_TREATMENT
            imputed_href = cta_data['ph3_url']
        elif int(cta_data['ph2_status']) in trialed:
            imputed = wsa.indication_vals.TRIALED2_TREATMENT
            imputed_href = cta_data['ph2_url']
        else:
            return # no imputation
        ind_data['indication'] = imputed
        ind_data['indication_href'] = imputed_href
    def reclassify_post_valid(self,overrides=None):
        # NOTE: the overrides parameter allows us to use this code to handle
        # the 'Researched as Treatment' buttons as well as the Reclassify
        # button. If overrides is not None, it contains information that
        # is used to partially fill in the indi_form fields needed to update
        # the indication. But the other indi_form fields are undefined, so
        # we need to skirt some of the save logic at the bottom.
        wsa = self.group.wsa
        if overrides:
            matched = [
                item
                for item in self.group.evidence
                if item.id == overrides['item_id']
                ]
            assert len(matched) == 1
            mode = overrides['mode']
            if mode == 'rat':
                ind_val = wsa.indication_vals.EXP_TREATMENT
            elif mode == 'pro':
                ind_val = matched[0].ind_val
            else:
                raise RuntimeError(f"unknown override mode '{mode}'")
            ind_data = dict(
                    indication = ind_val,
                    indication_href = matched[0].href,
                    )
            cta_data = None
        else:
            ind_data = self.context['indi_form'].cleaned_data
            cta_data = self.context['cta_form'].cleaned_data
            self.add_imputed_indication(ind_data,cta_data,wsa)
        note_data = self.context['note_form'].cleaned_data
        self.update_group_status(
                self.group.resolution_vals.ACCEPTED,
                ind_data=ind_data,
                cta_data=cta_data,
                note_data=note_data,
                )
        return HttpResponseRedirect(self.here_url())
    def update_post_valid(self):
        # the following should be prevented by the template, which should
        # disable the update button in this case, but make sure
        assert self.group.href_usable()
        self.update_group_status(
                self.group.resolution_vals.ACCEPTED,
                ind_data=dict(
                        indication=self.group.proposed_indication,
                        indication_href=self.group.best_indication_href(),
                        ),
                note_data = self.context['note_form'].cleaned_data,
                )
        return HttpResponseRedirect(self.here_url())
    def ignore_post_valid(self):
        self.update_group_status(
                self.group.resolution_vals.SKIPPED,
                note_data = self.context['note_form'].cleaned_data,
                )
        return HttpResponseRedirect(self.here_url())
    def update_group_status(self,resolution,
            ind_data=None,
            cta_data=None,
            note_data=None,
            ):
        p = self.check_id_form.cleaned_data
        assert p['group_id'] == self.group.id
        # update_indication will throw an error message before committing
        # if it encounters a problem. Call that in a try block, preceded by
        # any other needed error checks. Once the try block has ended, none
        # of the other writes should fail, so we can execute them in any order.
        try:
            if cta_data:
                self.cta.check_post_data(cta_data)
            if ind_data:
                detail = "KT search %d" % self.search.id
                extra = ind_data.get('detail')
                if extra:
                    detail += f" ({extra})"
                self.group.wsa.update_indication(
                        ind_data['indication'],
                        demerits=set(), # this clears any demerits
                        href=ind_data['indication_href'],
                        user=self.username(),
                        detail=detail,
                        )
        except ValueError as ex:
            self.message(str(ex))
            return
        if cta_data:
            self.cta.update_from_post_data(cta_data,self.username())
        if note_data:
            from notes.models import Note
            Note.set(self.group.wsa
                        ,'study_note'
                        ,self.username()
                        ,note_data['study_note']
                        )
        self.group.resolution = resolution
        self.group.mark_resolver(self.request.user.username)
        self.group.save()
    def _prep_selected_group(self,group):
        # we've identified the next drug to resolve
        drugname = group.wsa.get_name(self.is_demo())
        self.context_alias(
                group=group,
                drugname=drugname,
                wsa_link=link(
                        drugname,
                        group.wsa.drug_url(),
                        new_tab=True,
                        ),
                )
        # copy button map so we can customize it
        self.button_map = dict(self.button_map)
        enum = group.wsa.indication_vals
        for item in group.evidence:
            item.link_html = link(item.label(), item.href, new_tab=True)
            if not item.href_usable():
                continue # no shortcut buttons for unusable hrefs
            btn_name = f'rat{item.id}'
            self.button_map[btn_name] = ['check_id','note']
            self.add_shortcut_handler(
                    btn_name,
                    self.reclassify_post_valid,
                    dict(mode='rat',item_id=item.id),
                    )
            # include a button for the proposed indication, but only if
            # it doesn't duplicate the button above
            if item.ind_val != enum.EXP_TREATMENT:
                item.ind_label = enum.get('label',item.ind_val)
                btn_name = f'pro{item.id}'
                self.button_map[btn_name] = ['check_id','note']
                self.add_shortcut_handler(
                        btn_name,
                        self.reclassify_post_valid,
                        dict(mode='pro',item_id=item.id),
                        )
        from moldata.models import ClinicalTrialAudit
        self.cta = ClinicalTrialAudit.get_latest_wsa_record(
                group.wsa.id,
                self.username(),
                )

class SummaryView(DumaView):
    template_name='ktsearch/summary.html'
    index_dropdown_stem='kts_search'
    GET_parms = {
            'show':(list_of(str),[]),
            'msort':(SortHandler,'wsa'),
            'usort':(SortHandler,'name'),
            }
    button_map={
        'create':['autods'],
        }
    def custom_setup(self):
        self.msort.sort_parm='msort'
        self.usort.sort_parm='usort'
        self.context_alias(stats=[])
    def custom_context(self):
        self.load_search_config()
        self.load_match_data()
        self.calculate_match_stats()
        self.build_match_table()
        self.load_unmatch_data()
        self.build_unmatch_table()
        self.build_auto_drugsets()
        self.context['show']=self.show
    
    def make_autods_form(self, data):
        indi_choices=['tts', 'p2ts', 'p3ts', 'kts']
        indi_choices = [(x, x) for x in indi_choices]
        class FormClass(forms.Form):
            name = forms.CharField(label='DrugSet Name')
            indication = forms.ChoiceField(
                        choices=indi_choices,
                        label='Minimum Indication',
                        )
        return FormClass(data)
    def create_post_valid(self):
        p = self.context['autods_form'].cleaned_data
        name = p['name']
        indication = p['indication']
        from ktsearch.search import drugs_from_results, make_auto_drugset
        wsas = drugs_from_results(self.search, indication)
        ds = make_auto_drugset(self.ws, name, wsas, self.request.user.username)
        from dtk.prot_map import DpiMapping
        ws_mol_dpi = DpiMapping(self.ws.get_dpi_default()).get_baseline_dpi().choice
        url = self.ws.reverse('drugset') + f'?dpi={ws_mol_dpi}&drugset=ds{ds.id}'
        return HttpResponseRedirect(url)

    def _report_stat(self,label,count):
        self.stats.append('%d %s'%(count,label))
    @staticmethod
    def _grouped_indication_counts(key_ind_iterator):
        from collections import Counter
        result = {}
        for key,ind in key_ind_iterator:
            c = result.setdefault(key,Counter())
            c[ind] += 1
        return result
    @classmethod
    def _assemble_ind_data(cls,key_ind_iterator):
        # build {key:{ind:count,ind:count,...},...}
        ind_counts=cls._grouped_indication_counts(key_ind_iterator)
        # now reformat the counts into IndData namedtuples for each key
        result = {}
        for key,d in six.iteritems(ind_counts):
            result[key] = build_ind_data_list(six.iteritems(d))
        return result
    @staticmethod
    def format_evidence(l):
        from dtk.html import ulist,join
        return ulist(
                join(info.ind_label,':',info.count)
                for info in l
                )
    def load_search_config(self):
        self.search_detail = [
                (KtSourceType.lookup(src.source_type).src_name, src.config)
                for src in KtSource.objects.filter(search=self.search)
                ]
    def load_unmatch_data(self):
        # If an unmatched drugname gets matched to a WSA during the
        # manual resolution process, its group will no longer be null,
        # and it will disappear from this list. If it is confirmed as
        # not being a match to anything, it remains in the list but
        # the Reviewed column shows as 'True'. These confirmed non-matches
        # are also not included in the 'unmatched' count in the context,
        # which is displayed in the link to the name resolve page.
        sr_qs=KtSearchResult.objects.filter(
                query__search=self.search,
                group__isnull=True,
                )
        # build name->ind_data map for unmatched results
        self.name_ind_data = self._assemble_ind_data(
                sr_qs.values_list('drugname','ind_val')
                )
        # build name->sr map
        from dtk.data import MultiMap
        self.sr_by_name = MultiMap((sr.drugname,sr) for sr in sr_qs).fwd_map()
        # assemble data and sort it
        self.unmatch_data = [
                (
                    name,
                    self.name_ind_data[name][0].ind_label,
                    self.name_ind_data[name][0].sort_key,
                    self.name_ind_data[name],
                    all(x.unmatch_confirmed for x in self.sr_by_name[name]),
                )
                for name in self.sr_by_name.keys()
                ]
        if self.usort.colspec == 'name':
            key_idx=0
        elif self.usort.colspec == 'proposed_indication':
            key_idx=2
        self.unmatch_data.sort(
                key=lambda x:x[key_idx],
                reverse=self.usort.minus,
                )
        # calculate stats
        self._report_stat('results with unmatched drugnames', len(sr_qs))
        self.context['unmatched'] = len(set([
                x.drugname for x in sr_qs
                if not x.unmatch_confirmed
                ]))
    def build_unmatch_table(self):
        self.context_alias(
                unmatch_table = Table(self.unmatch_data,[
                        Table.Column('Name',
                                idx=0,
                                sort='l2h',
                                ),
                        Table.Column('Proposed Indication',
                                idx=1,
                                sort='h2l',
                                ),
                        Table.Column('Evidence',
                                idx=3,
                                cell_fmt=self.format_evidence,
                                ),
                        Table.Column('Reviewed',
                                idx=4,
                                ),
                        ],
                        url_builder=self.url_builder_factory('unmatched'),
                        sort_handler=self.usort,
                        ),
                )
    def load_match_data(self):
        # define base querysets for matched and unmatched results
        rg_qs=KtResultGroup.objects.filter(search=self.search)
        # build wsa->ind_data map
        self.wsa_ind_data = self._assemble_ind_data(
                rg_qs.values_list('wsa_id','ktsearchresult__ind_val')
                )
        # now retrieve all result groups into a dict
        group_by_wsa={x.wsa_id:x for x in rg_qs}
        # annotate each group with a proposed_indication value
        # (the ind_val from the first ind_data object in that group's list)
        for wsa_id in group_by_wsa:
            rg = group_by_wsa[wsa_id]
            ind_data = self.wsa_ind_data[wsa_id]
            rg.proposed_indication = ind_data[0].ind_val
        # build a map from wsa_id to drugname to speed page rendering
        self.wsa_name_helper=dict(
                rg_qs.filter(
                        wsa__agent__tag__prop__name='canonical',
                        ).values_list('wsa_id','wsa__agent__tag__value')
                )
        # build a list of (wsa_id,sort_key) pairs based on the requested
        # sort type for the match table
        if self.msort.colspec == 'wsa':
            ordering=list(self.wsa_name_helper.items())
        elif self.msort.colspec == 'cind':
            from .models import IndicationMapper
            im = IndicationMapper()
            ordering=[
                    # Current indication might not be a treatment
                    # indication; all of these get a sort key of
                    # 0. So, add the indication code itself as a
                    # secondary key, so different types of
                    # non-treatment group together
                    (wsa_id,(im.sort_key(ind_val),ind_val))
                    for wsa_id,ind_val in rg_qs.values_list('wsa_id','wsa__indication')
                    ]
        elif self.msort.colspec == 'proposed_indication':
            ordering=[
                    (rg.wsa_id,self.wsa_ind_data[rg.wsa_id][0].sort_key)
                    for rg in group_by_wsa.values()
                    ]
        else:
            # assume code is a KtResultGroup attribute name
            ordering=[
                    (rg.wsa_id,getattr(rg,self.msort.colspec))
                    for rg in group_by_wsa.values()
                    ]
        # build a list of result groups in the specified order
        from dtk.data import TypesafeKey
        ordering.sort(
                key=lambda x:TypesafeKey(x[1]), # handle columns with nulls
                reverse=self.msort.minus,
                )
        self.groups = [group_by_wsa[x[0]] for x in ordering]
    def calculate_match_stats(self):
        self._report_stat('matched drugs', len(self.groups))
        from browse.models import WsAnnotation
        enum=WsAnnotation.indication_vals
        self.context['unresolved'] = len([
                group
                for group in self.groups
                if group.resolution == group.resolution_vals.UNRESOLVED
                ])
        self._report_stat('new classifications', len([
                group
                for group in self.groups
                if group.wsa.indication == enum.UNCLASSIFIED
                ]))
        self._report_stat('matched classifications', len([
                group
                for group in self.groups
                if group.wsa.indication == group.proposed_indication
                ]))
        self._report_stat('modified classifications', len([
                group
                for group in self.groups
                if group.wsa.indication != group.proposed_indication
                    and group.wsa.indication != enum.UNCLASSIFIED
                ]))
    def build_auto_drugsets(self):
        from ktsearch.search import drugs_from_results
        groups = ['tts', 'p2ts', 'p3ts', 'kts']
        rows = []
        from django.utils.safestring import mark_safe
        for group in groups:
            auto_ds = drugs_from_results(self.search, group)
            rows.append((group, len(auto_ds)))
        from dtk.table import Table
        cols = [
            Table.Column('Indication Group', idx=0),
            Table.Column('# of Drugs', idx=1),
        ]
        table = Table(rows, cols)
        self.context_alias(autods_table=table)
    def build_match_table(self):
        from browse.models import WsAnnotation
        ind_vals=WsAnnotation.indication_vals
        res_vals = KtResultGroup.resolution_vals
        from dtk.html import join
        from django.utils.safestring import mark_safe
        from dtk.text import fmt_time
        self.context_alias(
                match_table = Table(self.groups,[
                        Table.Column('Drug',
                                code='wsa',
                                cell_fmt=lambda d:link(
                                        self.wsa_name_helper[d.id],
                                        d.drug_url(),
                                        new_tab=True,
                                        ),
                                sort='l2h',
                                ),
                        Table.Column('Current Indication',
                                # we need a distinct code from the Drug
                                # column for sorting, so make one up and
                                # supply an extract method to return
                                # the wsa
                                code='cind',
                                extract=lambda r:r.wsa,
                                cell_fmt=lambda d:d.indication_link(),
                                sort='h2l',
                                ),
                        Table.Column('Proposed Indication',
                                cell_fmt=lambda x:ind_vals.get('label',x),
                                sort='h2l',
                                ),
                        Table.Column('Evidence',
                                extract=lambda r:self.wsa_ind_data[r.wsa_id],
                                cell_fmt=self.format_evidence,
                                ),
                        Table.Column('Resolution',
                                cell_fmt=lambda x:res_vals.get('label',x),
                                sort='h2l',
                                ),
                        Table.Column('Resolved By',
                                code='user',
                                sort='l2h',
                                ),
                        Table.Column('Resolved At',
                                code='timestamp',
                                sort='h2l',
                                cell_fmt=lambda x:fmt_time(x,fmt='%Y-%m-%d %H:%M')
                                ),
                        ],
                        url_builder=self.url_builder_factory('matched'),
                        sort_handler=self.msort,
                        ),
                )

class SearchViewFaersTable(DumaView):
    template_name='ktsearch/_faers_section.html'
    def make_drug_table(self):
        from dtk.faers import make_faers_indi_dose_data, get_vocab_for_cds_name
        from dtk.table import Table 
        from browse.default_settings import faers
        cds_name = f'faers.{faers.latest_version()}'
        vocab = get_vocab_for_cds_name(cds_name)
        indi = self.ws.get_disease_default(vocab)
        out = make_faers_indi_dose_data(cds_name, indi, self.ws.id)
        columns = [
            Table.Column("Drug", idx=0),
            Table.Column("Count", idx=1),
            Table.Column("Indication", idx=2),
            Table.Column("Route", idx=3),
        ]
        return Table(out, columns)
    def custom_context(self):
        self.context_alias(
                drug_table = self.make_drug_table(),
        )

class SearchView(DumaView):
    template_name='ktsearch/search.html'
    index_dropdown_stem='kts_search'
    def custom_setup(self):
        # build button map and make_xxx_form members dynamically,
        # based on available KtSourceType classes
        self.sources=KtSourceType.get_subclasses()
        self.button_map={
                'search':[name for name,src in self.sources],
                }
        for name,src in self.sources:
            setattr(self,
                    'make_'+src.form_name(),
                    src.make_form_wrapper(self.ws),
                    )
    def custom_context(self):
        self.context_alias(
                past_searches = KtSearch.objects.filter(
                        ws=self.ws,
                        ).order_by('-id'),
                faers_data_url = self.ws.reverse('kts_search_faers_data'),
                )
    def form_list(self):
        return [
                getattr(self,src.form_name())
                for name,src in self.sources
                ]
    def search_post_valid(self):
        supplied_sources = []
        for name,SrcClass in self.sources:
            form = getattr(self,SrcClass.form_name())
            config = SrcClass.extract_config(**form.cleaned_data)
            try:
                SrcClass.append_parsed_config(supplied_sources,name,config)
            except ValueError as ex:
                # this was originally set up to catch ValueErrors thrown
                # by convert_records_using_colmap(), but each
                # append_parsed_config() implementation can use this to
                # report its own errors
                self.message(f'Error parsing {name}; {ex}')
                return
        if not supplied_sources:
            self.message('No search data supplied')
            return
        search = KtSearch(
                ws=self.ws,
                user=self.username(),
                )
        search.save()
        vfd = self.ws.get_versioned_file_defaults()
        import json
        for name,SrcClass,config,transient in supplied_sources:
            query = KtSource(
                    search=search,
                    source_type=name,
                    config=json.dumps(config),
                    )
            query.save()
            SrcClass.load_results(query,transient,vfd)
        return HttpResponseRedirect(self.ws.reverse('kts_summary',search.id))
