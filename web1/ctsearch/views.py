from django import forms
from django.http import HttpResponseRedirect

from dtk.duma_view import DumaView,list_of,boolean
from dtk.table import Table,SortHandler

from .models import CtSearch,CtDrugName
from .utils import ClinicalTrialsSearch,format_ct_study_table,ct_study_stats

import logging
logger = logging.getLogger(__name__)

class TrialDrugsView(DumaView):
    template_name='ctsearch/trial_drugs.html'
    index_dropdown_stem='cts_trial_drugs'
    GET_parms = {
            'phases':(list_of(str),['Phase 3','Phase 4']),
            'disease':(str,None),
            'drug':(str,None),
            'after':(int,None),
            'completed':(boolean,None),
            }
    button_map={
            'display':['search'],
            'save':[],
            }
    def criteria_hash(self):
        return dict(
                disease=self.disease,
                phases=self.phases,
                after=self.after,
                completed=self.completed,
                drug=self.drug,
                )
    def custom_setup(self):
        if self.disease and self.phases:
            self.search = ClinicalTrialsSearch(**self.criteria_hash())
    def custom_context(self):
        self.context_alias(top_links=[])
        if self.disease and self.phases:
            if self.drug:
                self.detail_context()
            else:
                self.list_context()
        else:
            self.context['page_label'] = 'Enter Disease'
        if not self.drug:
            self.context['saved_searches'] = list(
                    CtSearch.objects.filter(ws=self.ws).order_by('-id')
                    )
    def detail_context(self):
        from dtk.html import link
        from dtk.duma_view import qstr
        from dtk.url import google_search_url
        self.top_links.append(link(
                        'See all drugs for '+self.disease,
                        self.here_url(drug=None),
                        ))
        self.top_links.append(link(
                        'search Duma for drug',
                        self.ws.reverse('review')+qstr({},search=self.drug)
                        ))
        self.top_links.append(link(
                        'drug+disease on google',
                        google_search_url([self.drug,self.ws.name]),
                        new_tab=True,
                        ))
        ct_url="https://clinicaltrials.gov/ct2/show/"
        show_set=lambda x:', '.join(x)
        self.context['table'] = format_ct_study_table(self.search.study_list)
        self.context['page_label'] = 'Studies for %s, %s'%(
                        self.drug,
                        self.disease,
                        )
    def list_context(self):
        by_drug = self.search.by_drug
        # the following also enables the save button if non-zero
        self.list_size = len(by_drug)
        from dtk.table import Table
        from dtk.duma_view import qstr
        from dtk.url import google_search_url
        from dtk.html import link
        self.context['table'] = Table(by_drug,[
                Table.Column('Drug',
                        idx=0,
                        ),
                Table.Column('',
                        idx=0,
                        cell_fmt=lambda x:link('see detail',
                                    self.here_url(drug=x)
                                    ),
                        ),
                Table.Column('',
                        idx=0,
                        cell_fmt=lambda x:link('search Duma for drug',
                                    self.ws.reverse('review')+qstr({},search=x)
                                    ),
                        ),
                Table.Column('',
                        idx=0,
                        cell_fmt=lambda x:link('drug+disease on google',
                                    google_search_url([x,self.ws.name]),
                                    new_tab=True,
                                    ),
                        ),
                Table.Column('Study Count',
                        extract=lambda x:len(x[1]),
                        ),
                ])
        self.context['page_label'] = 'Trial Drugs for '+self.disease
    def make_search_form(self,data):
        class MyForm(forms.Form):
            disease = forms.CharField(
                    initial=self.disease \
                        or self.ws.get_disease_default('ClinicalTrials'),
                    )
            from dtk.html import WrappingCheckboxSelectMultiple
            phases = forms.MultipleChoiceField(
                    label='Include phases:',
                    choices=[(x,x) for x in ClinicalTrialsSearch.phases],
                    widget=WrappingCheckboxSelectMultiple,
                    initial=self.phases,
                    )
            completed = forms.BooleanField(
                    label='Completed studies only',
                    required=False,
                    initial=self.completed,
                    )
            after = forms.IntegerField(
                    label='Ignore studies before (YYYY)',
                    required=False,
                    initial=self.after,
                    )
        return MyForm(data)
    def display_post_valid(self):
        p = self.context['search_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(
                disease=p['disease'],
                completed=p.get('completed'),
                phases=','.join(p.get('phases')),
                after=p.get('after'),
                ))
    def save_post_valid(self):
        # record search criteria and results in database
        import json
        srch = CtSearch(
                ws=self.ws,
                user=self.request.user.username,
                config=json.dumps(
                        self.criteria_hash(),
                        separators=(',',':'),
                        sort_keys=True,
                        ),
                )
        srch.save()
        for drug_name,_ in self.search.by_drug:
            dn = CtDrugName(search=srch,drug_name=drug_name)
            dn.save()
        return HttpResponseRedirect(self.ws.reverse('cts_resolve',srch.id))

class CtSummaryView(DumaView):
    '''Show status of a saved CT search.'''
    template_name='ctsearch/ct_summary.html'
    index_dropdown_stem='cts_trial_drugs'
    def handle_search_id_arg(self,search_id):
        self.search = CtSearch.objects.get(pk=search_id)
        dn_set = self.search.ctdrugname_set
        self.drugnames = dn_set.order_by('status','id')
        self.stats = ct_study_stats(dn_set)
    def custom_context(self):
        self.context['page_label'] = 'CT Search Summary'
        from dtk.table import Table
        from dtk.duma_view import qstr
        from dtk.url import google_search_url
        from dtk.html import link
        self.context['table'] = Table(self.drugnames,[
                Table.Column('Drug Name',
                        ),
                Table.Column('Status',
                        cell_fmt=lambda x:CtDrugName.status_vals.get('label',x)
                        ),
                Table.Column('',
                        code='drug_name',
                        cell_fmt=lambda x:link('search Duma for drug',
                                    self.ws.reverse('review')+qstr({},search=x)
                                    ),
                        ),
                Table.Column('',
                        code='id',
                        cell_fmt=lambda x:link('see detail',
                                    self.ws.reverse(
                                            'cts_resolve',
                                            self.search.id,
                                            )+qstr({},dn=x)
                                    ),
                        ),
                ])

# XXX Possible future additions:
# XXX - drugnames in the CT database match at least one collection name, but
# XXX   not necessarily a name in the current workspace; we could do a more
# XXX   elaborate query to look for a workspace drug in the same cluster as
# XXX   the one(s) that match the CT name
# XXX - for the case of real matches that haven't already been seen, an
# XXX   alternative to completely manual resolution would be to manually
# XXX   associate a wsa with the drugname record, and then feed back through
# XXX   the auto-resolve loop; it's hard to see how this would save much
# XXX   time, however
# XXX     - maybe, the page tries more aggressively to resolve the name:
# XXX       - link through non-imported collections
# XXX       - does fuzzy match against names of all marked indications in
# XXX         workspace
class CtResolveView(DumaView):
    '''Resolve next drugname in a saved CT search.'''
    template_name='ctsearch/ct_resolve.html'
    index_dropdown_stem='cts_trial_drugs'
    GET_parms = {
            'dn':(int,None),
            }
    def custom_context(self):
        self.context['page_label'] = 'CT Search Resolution'
    def find_wsa_match_info(self,dn):
        from browse.models import WsAnnotation
        qs_ws = WsAnnotation.objects.filter(ws=self.ws)
        from browse.utils import drug_search_wsa_filter
        qs=drug_search_wsa_filter(qs_ws,dn.drug_name)
        name_matches = [
                x
                for x in qs
                if x.agent.canonical.lower() == dn.drug_name
                ]
        return set(qs),name_matches
    def auto_resolve(self,dn):
        # given an unresolved CtDrugName object, either return a matching
        # WSA record for manual resolution, or resolve it automatically
        # and return None
        s,name_matches = self.find_wsa_match_info(dn)
        dn_stat = CtDrugName.status_vals
        if not s:
            dn.status = dn_stat.UNMATCHED
            logger.info('auto_resolve setting %s to %s',
                    dn.drug_name,
                    dn_stat.get('label',dn.status),
                    )
            dn.save()
            return
        if len(s) > 1 and len(name_matches) != 1:
            dn.status = dn_stat.AMBIGUOUS
            logger.info('auto_resolve setting %s to %s',
                    dn.drug_name,
                    dn_stat.get('label',dn.status),
                    )
            dn.save()
            return
        # At this point, since we failed both if's above,
        # either there's exactly one name match,
        # or there's no name match but exactly one drug in s.
        # Extract the single drug, preferring the name match.
        if name_matches:
            match = name_matches[0]
        else:
            match = s.pop()
        from browse.models import WsAnnotation
        wsa_stat = WsAnnotation.indication_vals
        premarked_set = [
                wsa_stat.FDA_TREATMENT,
                wsa_stat.KNOWN_TREATMENT,
                wsa_stat.TRIALED3_TREATMENT,
                wsa_stat.TRIALED2_TREATMENT,
                wsa_stat.TRIALED1_TREATMENT,
                wsa_stat.TRIALED_TREATMENT,
                ]
        if match.indication in premarked_set:
            dn.status = dn_stat.PREMARKED
            logger.info('auto_resolve setting %s (%d) to %s',
                    dn.drug_name,
                    match.id,
                    dn_stat.get('label',dn.status),
                    )
            dn.save()
            return
        return match
    def handle_search_id_arg(self,search_id):
        self.search = CtSearch.objects.get(pk=search_id)
    def custom_setup(self):
        dn_set = self.search.ctdrugname_set
        self.stats = ct_study_stats(dn_set)
        if self.dn:
            self.selected_dn=CtDrugName.objects.get(pk=self.dn)
            s,name_matches = self.find_wsa_match_info(self.selected_dn)
            if len(name_matches) == 1:
                match = name_matches[0]
            elif len(s) == 1:
                match = s.pop()
            else:
                match = None
                self.button_map = {
                        'manual':[],
                        }
            self.label = 'Reviewing'
        else:
            # set defaults in case loop runs out without match
            self.selected_dn = None
            match = None
            for dn in dn_set.filter(
                            status=CtDrugName.status_vals.UNRESOLVED
                            ).order_by('id'):
                match = self.auto_resolve(dn)
                if match:
                    self.selected_dn = dn
                    self.label = 'Resolving'
                    break
        self.selected_wsa = match
        if match:
            from dtk.url import ext_drug_links
            self.context['ext_drug_links'] = ext_drug_links(match)
            self.button_map = {
                    'reclassify':['indi'],
                    'phase1':['study_id'],
                    'phase2':['study_id'],
                    'phase3':['study_id'],
                    'reject':[],
                    }
        import json
        d=json.loads(self.search.config)
        if self.selected_dn:
            d['drug'] = self.selected_dn.drug_name
            ct_srch = ClinicalTrialsSearch(**d)
            self.study_list = ct_srch.study_list
            self.context['table'] = format_ct_study_table(self.study_list)
    def make_indi_form(self,data):
        wsa = self.selected_wsa
        from dtk.html import move_to_top
        indi_choices=move_to_top(wsa.indication_vals.choices(),(
                wsa.indication_vals.INITIAL_PREDICTION,
                wsa.indication_vals.INACTIVE_PREDICTION,
                ))
        class FormClass(forms.Form):
            indication = forms.ChoiceField(
                        choices=indi_choices,
                        initial=wsa.indication,
                        )
            indication_href = forms.CharField(
                        initial=wsa.indication_href,
                        )
        return FormClass(data)
    def make_study_id_form(self,data):
        # this is used for input parsing only; the form is
        # rendered directly in the template (actually, one
        # copy for each study, but only one will post back
        # at a time)
        class FormClass(forms.Form):
            study = forms.CharField()
        return FormClass(data)
    def post_classify_redirect(self):
        if self.dn:
            # we came here from the summary page for a specific
            # drug; return there
            return HttpResponseRedirect(
                    self.ws.reverse('cts_summary',self.search.id)
                    )
        else:
            # we're in auto-resolve mode; go to next drugname
            return HttpResponseRedirect(self.here_url())
    def phase1_post_valid(self):
        from browse.models import WsAnnotation
        return self.mark_indication(
                WsAnnotation.indication_vals.TRIALED1_TREATMENT
                )
    def phase2_post_valid(self):
        from browse.models import WsAnnotation
        return self.mark_indication(
                WsAnnotation.indication_vals.TRIALED2_TREATMENT
                )
    def phase3_post_valid(self):
        from browse.models import WsAnnotation
        return self.mark_indication(
                WsAnnotation.indication_vals.TRIALED3_TREATMENT
                )
    def mark_indication(self,ind_val):
        p = self.context['study_id_form'].cleaned_data
        wsa = self.selected_wsa
        from dtk.url import clinical_trials_url
        wsa.update_indication(
                ind_val,
                None,
                self.request.user.username,
                "accepted clinical trial",
                clinical_trials_url(p['study']),
                )
        dn = self.selected_dn
        dn.study_id = p['study']
        dn.status = dn.status_vals.ASSIGNED
        dn.save()
        return self.post_classify_redirect()
    def manual_post_valid(self):
        dn = self.selected_dn
        dn.status = dn.status_vals.MANUAL
        dn.save()
        return self.post_classify_redirect()
    def reject_post_valid(self):
        dn = self.selected_dn
        dn.status = dn.status_vals.REJECTED
        dn.save()
        return self.post_classify_redirect()
    def reclassify_post_valid(self):
        p = self.context['indi_form'].cleaned_data
        wsa = self.selected_wsa
        wsa.update_indication(
                p['indication'],
                {},
                self.request.user.username,
                "set manually",
                p['indication_href'],
                )
        dn = self.selected_dn
        dn.status = dn.status_vals.ASSIGNED
        dn.save()
        return self.post_classify_redirect()

