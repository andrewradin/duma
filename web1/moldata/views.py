from dtk.duma_view import DumaView,list_of,boolean
from dtk.table import Table,SortHandler
from django import forms
from django.http import HttpResponseRedirect, HttpResponse
from django.db import transaction

import logging
logger = logging.getLogger(__name__)

class AnnotateView(DumaView):
    template_name='moldata/annotate.html'
    GET_parms={
            'scorejobs':(str,''),
            'prescreen_id':(int,None),
            'show':(list_of(str),''),
            'dpi':(str,None),
            'dpi_thresh':(float,None),
            'patent_edit':(int,None),
            'ver_override':(int,None),
            'only_scorebox':(boolean,None),
            'only_moa_mols':(boolean,None),
            'only_faers':(boolean,None),
            'only_cta_ds':(boolean,None),
            }
    # override to display 'invalid' drugs for audit purposes
    def handle_wsa_id_arg(self,wsa_id):
        from browse.models import WsAnnotation
        self.context_alias(wsa = WsAnnotation.all_objects.get(pk=wsa_id))
    def custom_setup(self):
        from dtk.prot_map import AgentTargetCache, DpiMapping
        if self.only_scorebox:
            self.template_name = 'moldata/_scorebox_section.html'
        elif self.only_moa_mols:
            self.template_name = 'moldata/_moa_mols_section.html'
        elif self.only_faers:
            self.template_name = 'moldata/_mol_faers_section.html'
        elif self.only_cta_ds:
            self.template_name = 'moldata/_cta_ds_section.html'
        else:
            self.template_name = 'moldata/annotate.html'

        assert self.wsa.ws_id == self.ws.id
        self.button_map={
                'target':['target'],
                'indication':['ann','demerit'],
                }
        if not self.dpi:
            self.dpi = DpiMapping(self.ws.get_dpi_default()).dpimap_for_wsa(self.wsa).choice
        if self.dpi_thresh is None:
            self.dpi_thresh = self.ws.get_dpi_thresh_default()
        self.targ_cache = AgentTargetCache(
                mapping = DpiMapping(self.dpi),
                agent_ids = [self.wsa.agent_id],
                dpi_thresh = self.dpi_thresh,
                )
        self.get_election_data()
        if hasattr(self,'vote'):
            self.button_map.update(vote=['vote','protein_notes'])
        from .models import ClinicalTrialAudit
        self.cta = ClinicalTrialAudit.get_latest_wsa_record(
                self.wsa.id,
                self.username(),
                )
    def custom_context(self):
        self.context['drug_ws'] = self.wsa
        self.context['show'] = self.show
        self.load_cluster_members()
        self.get_other_notes()
        self.get_scorebox()
        self.get_cluster_mates()
        self.get_cta_status()
        from dtk.url import ext_drug_links,chembl_drug_url,bindingdb_drug_url,multiprot_search_links,globaldata_drug_url
        if self.wsa.is_moa():
            self.context['show_faers'] = False
            from browse.models import Protein
            prots = Protein.objects.filter(uniprot__in=self.targ_cache.all_prots)
            links = list(multiprot_search_links(self.ws, prots).items())
            # Rename it from 'srch' to something more user friendly.
            links[0] = ['All Genes Search', links[0][1]]
        else:
            links = ext_drug_links(self.wsa, split=True)
            self.context['show_faers'] = True
        if self.is_demo():
            from django.utils.safestring import mark_safe
            import re
            ttab = str.maketrans({x:'*' for x in '0123456789'})
            clean_links = []
            repl_href="https://google.com/"
            for heading,l in links:
                clean_l = []
                for s in l:
                    s = s.translate(ttab)
                    m = re.match(r'<a href="(.*)" target=',s)
                    if not m:
                        continue
                    s = s[:m.start(1)]+repl_href+s[m.end(1):]
                    clean_l.append(mark_safe(s))
                clean_links.append([heading,clean_l])
            links = clean_links
        self.context['ext_drug_links'] = links
        from dtk.html import link
        self.context['condensed_links'] = [
                link(chembl_id,chembl_drug_url(chembl_id),new_tab=True)
                for chembl_id in sorted(
                        self.wsa.agent.shadowed_chembl_id_set,
                        key=lambda x:int(x[6:]),
                        )
                ]
        self.context['condensed_links'] += [
                link(id,bindingdb_drug_url(id),new_tab=True)
                for id in sorted(
                        self.wsa.agent.shadowed_bindingdb_id_set,
                        key=lambda x:int(x[4:]),
                        )
                ]
        self.context['condensed_links'] += [
                link(id,globaldata_drug_url(id),new_tab=True)
                for id in sorted(
                        self.wsa.agent.shadowed_globaldata_id_set,
                        key=lambda x:int(x[2:]),
                        )
                ]
        self.get_binding()
        self.load_attributes()
        self.get_pending_proposals()
        if self.only_cta_ds:
            self.context['dss']=self.check_ct_drugsets()
        self.context['page_tab']='drug(%d) %s' % (
                                self.ws.id,
                                self.drugname(),
                                )
        from patsearch.patent_search import related_searches_for_drug
        self.context['patent_searches'] = related_searches_for_drug(self.wsa.id)
        self.context['scorebox_url'] = self.here_url(only_scorebox=True)
        self.context['moa_mols_url'] = self.here_url(only_moa_mols=True)
        self.context['faers_url'] = self.here_url(only_faers=True)
        self.context['cta_ds_url'] = self.here_url(only_cta_ds=True)

        cts = self._get_trials_with_sponsors()
        self.context['clinical_trials'] = cts
        self.context['ct_table'] = self._make_ct_table(cts)
        if self.only_faers:
            self.context['faers_table'] = self._make_faers_table(self.cas_ids)

    def check_ct_drugsets(self):
        if self.wsa.is_moa():
            base_choices = [('moa-' + x[0], 'MoA ' + x[1]) for x in self._get_ds_choices()]
        else:
            base_choices = self._get_ds_choices()
        choices = []
        for ds_id, ds_name in base_choices:
            choices += [
                        ('split-train-%s' % ds_id, '%s (Train)' % ds_name)
                    ]
            choices += [
                        ('split-test-%s' % ds_id, '%s (Test)' % ds_name)
                    ]
        dss = []
        for id, name in choices:
            wsas = self.ws.get_wsa_id_set(id)
            if wsas and self.wsa.id in wsas:
                dss.append(name)
        if dss:
            return dss

    def _get_ds_choices(self):
        to_return = self.ws.get_ct_drugset_choices()
# this leaves the option to extend this to other types of drugsets,
# though at this time these are the only ones we care about
        return to_return

    def _get_trials_with_sponsors(self):
        from dtk.aact import lookup_trials_by_molecule
        cts = lookup_trials_by_molecule(self.wsa)
        if not cts:
            return cts
        # merge in sponsor data
        example = next(iter(cts))
        from collections import namedtuple
        TrialRecord = namedtuple('TrialRecord',example._fields+('sponsor',))
        trials = [x.study for x in cts]
        from dtk.aact import lookup_trial_sponsors
        trial2sponsor = lookup_trial_sponsors(trials)
        return set(
                TrialRecord(*(tuple(ct)+(trial2sponsor.get(ct.study,''),)))
                for ct in cts
                )

    def _make_faers_table(self, cas_ids):
        if not cas_ids:
            return None

        from dtk.faers import DrugIndiDoseCount
        from dtk.cache import cached
        from browse.default_settings import faers
        import numpy as np
        cds = faers.value(ws=self.ws)
        ver = cds.split('v')[1]
        if int(ver) < 7:
            # This data doesn't exist pre-v7.
            ver = 7
        cds = f'faers.v{ver}'

        didc = DrugIndiDoseCount(cds=cds)
        col_names = didc.meta['indi_drug_cols']
        cas_ids = [x for x in cas_ids if x in col_names]
        if not cas_ids:
            return None


        @cached(version=1, argsfunc=lambda: (sorted(cas_ids), cds))
        def faers_table_data():
            fm = didc.indi_drug_fm
            rows = didc.matching_rows(cas_ids)
            counts = np.asarray(fm[rows].sum(axis=0)).reshape(-1)
            dis_data = [(name, count) for name, count in zip(col_names, counts) if count and name not in cas_ids]
            return dis_data

        data = faers_table_data()

        from dtk.table import Table
        cols = [
            Table.Column('Indication Used For', idx=0),
            Table.Column('Count', idx=1),
        ]
        table = Table(data, cols)
        return table

    def _make_ct_table(self, cts):
        from dtk.table import Table
        from dtk.url import clinical_trials_url
        from dtk.html import link
        return Table(cts, [
            Table.Column('Study', cell_fmt=lambda x: link(x, clinical_trials_url(x), new_tab=True)),
                    Table.Column('Intervention'),
                    Table.Column('Phase'),
                    Table.Column('Start', code='start_year'),
                    Table.Column('Status'),
                    Table.Column('Completion', code='completion_date'),
                    Table.Column('Sponsor', code='sponsor'),
                    Table.Column('Disease', cell_fmt=lambda x: ' | '.join(x)),
                    Table.Column('Drugs', cell_fmt=lambda x: ', '.join(x)),
                ])

    def get_cta_status(self):
        if self.cta.pk:
            self.context['cta_status'] = \
                    '; '.join(self.cta.stat_summary()) \
                    + f' ({self.cta.metadata()})'
    def get_cluster_mates(self):
        # Create links to all the drugs that are in the same cluster as
        # this drug. In normal circumstances there shouldn't be any,
        # because import should only pull in one drug per cluster.
        # But if clustering changes, and multiple drugs in the new cluster
        # have meaningful history, they will all be preserved.
        from drugs.models import DpiMergeKey
        ver = self.wsa.ws.get_dpi_version()
        try:
            cluster_key = DpiMergeKey.objects.get(
                    version=ver,
                    drug_id=self.wsa.agent_id,
                    ).dpimerge_key
        except DpiMergeKey.DoesNotExist:
            return
        cluster_agent_ids = DpiMergeKey.objects.filter(
                version=ver,
                dpimerge_key=cluster_key,
                ).values_list('drug_id',flat=True)
        if not cluster_agent_ids:
            return
        from browse.models import WsAnnotation
        mate_wsas = WsAnnotation.objects.filter(
                ws_id=self.wsa.ws_id,
                agent_id__in=cluster_agent_ids,
                )
        from dtk.html import link
        self.cluster_mate_links = [
                link(
                        getattr(wsa.agent,wsa.agent.collection.key_name),
                        wsa.drug_url(),
                        )
                for wsa in mate_wsas
                if wsa != self.wsa
                ]
    def make_ann_form(self,data):
        wsa = self.wsa
        agent = wsa.agent
        ps_list = wsa.ws.get_prescreen_choices()
        replacements_str = '\n'.join(str(x.id) for x in wsa.replacements.all())
        from dtk.html import MultiWsaField
        class MyForm(forms.Form):
            indication = forms.ChoiceField(
                    choices = wsa.grouped_choices(),
                    initial = wsa.indication,
                    )
            indication_href = forms.CharField(
                    required=False,
                    label = 'Indication URL',
                    initial = wsa.indication_href,
                    widget = forms.TextInput(attrs={'size':'60'}),
                    )
            doc_href = forms.CharField(
                    required=False,
                    label = 'Write-up URL',
                    initial = wsa.doc_href,
                    )
            why = forms.ChoiceField(
                    label = 'Marking Detail',
                    choices = [(0,'Set manually')]+ps_list,
                    initial = wsa.marked_prescreen.id if wsa.marked_prescreen else None,
                    )
            # XXX these fields are duplicated in the ktsearch resolve view;
            # XXX factor out somehow? This would involve splitting this
            # XXX form into 3 parts, and importing the middle part from a
            # XXX shared place, maybe like the VoteForm
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
            study_note = forms.CharField(
                    widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
                    required=False,
                    initial = wsa.get_study_text(),
                    label='Workspace note',
                    )
            hide = forms.BooleanField(
                    required=False,
                    label='Hide (all WS)',
                    initial=agent.hide,
                    )
            ubiquitous = forms.BooleanField(
                    required=False,
                    label='Ubiquitous (all WS)',
                    initial=agent.ubiquitous,
                    )
            bd_note = forms.CharField(
                    widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
                    required=False,
                    initial = wsa.agent.get_bd_note_text(),
                    label='Global note (all WS)',
                    )
            override_name = forms.CharField(
                    required=False,
                    initial = agent.override_name,
                    label='Override Name (all WS)',
                    widget = forms.TextInput(attrs={'size':'60'}),
                    )
            replaced_by = MultiWsaField(
                    ws_id=self.ws.id,
                    required=False,
                    initial = replacements_str,
                    label='Replaced By',
                    )
            txr_id = forms.CharField(
                    required=False,
                    label = 'TXR ID',
                    initial = wsa.txr_id,
                    )
        return MyForm(data)
    def make_demerit_form(self,data):
        from browse.models import Demerit
        choices = [
                (x.id,x.desc)
                for x in Demerit.objects.filter(active=True).order_by('desc')
                ]
        from dtk.html import WrappingCheckboxSelectMultiple
        class MyForm(forms.Form):
            demerits = forms.MultipleChoiceField(
                    label='Reject Reasons',
                    choices=choices,
                    widget=WrappingCheckboxSelectMultiple,
                    required=False,
                    initial=self.wsa.demerits(),
                    )
        return MyForm(data)
    def indication_post_valid(self):
        ap = self.ann_form.cleaned_data
        dp = self.demerit_form.cleaned_data
        try:
            prescreen_id = int(ap['why'])
            if prescreen_id:
                from browse.models import Prescreen
                pscr = Prescreen.objects.get(pk=prescreen_id)
                marked_because = pscr.marked_because()
                marking_prescreen = pscr
            else:
                marked_because = 'set manually'
                marking_prescreen = None
            self.cta.check_post_data(ap)
            self.wsa.update_indication(
                    ap['indication'],
                    dp['demerits'],
                    self.request.user.username,
                    marked_because,
                    ap['indication_href'],
                    from_prescreen=marking_prescreen,
                    )
        except ValueError as ex:
            self.message(str(ex))
            self.show.append('indication')
            return
        self.wsa.doc_href = ap['doc_href']
        self.wsa.txr_id = ap['txr_id']

        def parse_wsa_list(valstr):
            if not valstr:
                return []
            import re
            wsas = re.split('[,\s]+', valstr)
            return [int(x.strip()) for x in wsas if x.strip()]

        replacements = parse_wsa_list(ap['replaced_by'])
        self.wsa.replacements.set(replacements)
        self.wsa.save()
        self.cta.update_from_post_data(ap,self.username())
        from notes.models import Note
        Note.set(self.wsa
                    ,'study_note'
                    ,self.username()
                    ,ap['study_note']
                    )
        agent = self.wsa.agent
        agent.hide = ap['hide']
        agent.ubiquitous = ap['ubiquitous']
        agent.save()
        Note.set(agent
                    ,'bd_note'
                    ,self.username()
                    ,ap['bd_note']
                    )
        override_name = ap['override_name']
        from drugs.models import Prop
        if override_name:
            agent.set_prop(Prop.OVERRIDE_NAME,override_name)
        else:
            agent.del_prop(Prop.OVERRIDE_NAME)
        if 'indication' not in self.show:
            self.show.append('indication')
        return HttpResponseRedirect(self.here_url(show=self.show))
    def make_target_form(self,data):
        from nav.views import dpi_mapping_form_class
        TargetForm=dpi_mapping_form_class(self.dpi,self.dpi_thresh)
        return TargetForm(data)
    def target_post_valid(self):
        p = self.target_form.cleaned_data
        self.show.append('targ')
        return HttpResponseRedirect(self.here_url(show=self.show,**p))
        return HttpResponseRedirect(self.here_url(show='target',**p))
    def make_vote_form(self,data):
        from .forms import VoteForm
        return VoteForm(self.request.user,data,instance=self.vote)
    def make_protein_notes_form(self,data):
        from dtk.dynaform import FormFactory
        ff = FormFactory()
        for prot,gene,note_id,text in self.prot_note_data:
            ff.add_field(prot,forms.CharField(
                    required=False,
                    widget=forms.Textarea(attrs={'rows':'1','cols':'40'}),
                    initial=text,
                    label='%s (%s)'%(gene,prot),
                    ))
        FormClass = ff.get_form_class()
        return FormClass(data)
    def vote_post_valid(self):
        user = self.request.user.username
        from browse.models import TargetReview
        for uniprot,text in self.protein_notes_form.cleaned_data.items():
            TargetReview.save_note(self.ws,uniprot,user,text)
        p = self.vote_form.cleaned_data
        self.vote.recommended = p['recommended']
        from notes.models import Note
        Note.set(self.vote,'note',user,p['note'])
        self.vote.save()
        # this may have changed completion status; make sure
        # note permissions are correct
        self.vote.election.update_note_permissions()
        return HttpResponseRedirect(self.here_url())
    def load_cluster_members(self):
        # Load agent ids of all cluster members.
        # - this includes the agent id of this wsa, even if it isn't
        #   clustered with anything else
        # - this uses the possibly-overridden dpimerge version
        from drugs.models import Drug
        my_agent_id = self.wsa.agent_id
        self.all_cluster_members = Drug.matched_id_mm(
                [my_agent_id],
                version=self._get_mol_version(),
                ).fwd_map()[my_agent_id]
    def get_pending_proposals(self):
        # Create links for any Duma collection proposals that are active
        # but not yet fully uploaded. Unlike the previous approach, this
        # code doesn't assume the WSA being viewed needs to be from the
        # duma collection.
        from drugs.models import DrugProposal,Tag
        from dtk.html import link
        from django.urls import reverse
        pending = []
        for dp in DrugProposal.objects.filter(
                ref_drug_id__in=self.all_cluster_members,
                state__in=DrugProposal.active_states,
                ):
            if Tag.objects.filter(
                    prop__name='duma_proposal_id',
                    value=str(dp.id),
                    ).exists():
                continue # this one's been uploaded; ignore
            pending.append(link(
                    f'{dp.ref_drug.get_key()} has pending edits',
                    reverse('drug_edit_review_view') + f"?prop_id={dp.id}"
                    ))
        self.context['pending_edits'] = pending
    def load_attributes(self):
        from drugs.models import Collection,Drug,Prop
        from collections import defaultdict
        name_attrs = defaultdict(lambda: defaultdict(list))
        struct_attrs = defaultdict(lambda: defaultdict(list))
        mol_vals = defaultdict(lambda: defaultdict(list))
        struct_alerts = set()
        max_phase=set()
        cas_ids = set()
        # build list of (keyspace,prop_name,value) for all properties of
        # all matched agents
        exclude_props = [
                'synthesis_protection',
                Prop.OVERRIDE_NAME,
                ]
        attrs=[]
        name_struc_props = ['Tag', 'Blob']
        other_id_attrs = ['synonym','pubchem_cid','cas','canonical','atc','kegg']
        struc_attrs = ['smiles', 'inchi']
        other_struct_attrs = []
        for p in Prop.prop_types.choices():
            prop_type = p[0]
            prop = p[1]
            cls = Prop.cls_from_type(prop_type)
            for x in cls.objects.filter(
                            drug_id__in=self.all_cluster_members,
                            ).values_list(
                            'drug__collection__name','prop__name','value'
                            ):
                attr_name= x[1]
                attr_parts = attr_name.split('_')
                source = x[0].split('.')[0]
                val = x[2]
                if attr_name in exclude_props or attr_name.startswith(Collection.foreign_key_prefix):
                    continue
                if attr_name == 'cas':
                    cas_ids.add(val)
                if prop in name_struc_props and (attr_name.endswith('_id') or attr_name in other_id_attrs):
                    name_attrs[attr_name][val].append(source)
                elif prop in name_struc_props and any((a in attr_parts for a in struc_attrs)):
                    struct_attrs[attr_name][val].append(source)
                elif prop == 'Flag' and attr_name.startswith('sa_'):
                    struct_alerts.add((" ".join(attr_parts[1:]),source))
                elif attr_name == 'max_phase':
                    max_phase.add(int(val))
                elif prop in ['Metric', 'Index']:
                    k=self._process_pc_names(attr_parts)
                    mol_vals[k][val].append(source)
        self.cas_ids = cas_ids
        self.context['name_attrs'] = self._process_ddls(name_attrs)
        self.context['struct_attrs'] = self._process_ddls(struct_attrs)
        self.context['mol_vals'] = self._process_ddls(mol_vals)
        self.context['struct_alerts']=sorted(list(struct_alerts))
        if max_phase:
            val = max(max_phase)
            if val == 4:
                term = 'Approved'
            elif val == 0:
                term = 'Experimental'
            else:
                term = f'Ph.{val}'
            self.context['max_phase'] = term
    def _process_pc_names(self, attr_parts):
        default=" ".join(attr_parts).capitalize()
        if default == 'Full mwt':
            return "Molecular weight"
        elif default == 'Alogp  hydrophobicity':
            return "Hydrophobicity: ALogP"
        elif default.endswith(' pka'):
            return " ".join(attr_parts[:-1]).capitalize() + ' pKa'
        elif default == 'Num lipinski rule5 violations':
            return "Lipinski rule of 5 violations"
        elif default == 'Num rule of 5 violations':
            return "Rule of 5 violations"
        elif default == 'Num structural alerts':
            return "Structural alerts count"
        elif default == 'Wtd qed  drug likliness':
            return "Weighted quantitative estimate of drug-likeness (wQED)"
        return default
    def _get_mol_version(self):
        if self.ver_override:
            return self.ver_override
        else:
            return self.wsa.ws.get_dpi_version()
    def _process_ddls(self, ddl):
       to_ret = []
       for k in sorted(ddl.keys()):
           l = [(v, ", ".join(set(sl))) for v,sl in ddl[k].items()]
           to_ret.append((k,l))
       return to_ret
    def get_binding(self):
        from dtk.prot_map import DpiMapping, AgentTargetCache, protein_link
        targ_cache = self.targ_cache
        note_cache = targ_cache.build_note_cache(
                self.ws,
                self.request.user.username,
                )
        by_key = {}
        prots = set()
        protdirs = set()
        for x in targ_cache.full_info_for_agent(self.wsa.agent_id):
            l = by_key.setdefault(x[0],[])
            l.append(x[1:])
            prots.add(x[1])
            protdirs.add((x[1], x[4]))
        bindings = []
        for native_key in sorted(by_key.keys()):
            # sort by evidence
            l = sorted(by_key[native_key],key=lambda x:x[2],reverse=True)
            bindings.append((native_key,[
                    protein_link(
                            uniprot,
                            gene,
                            self.ws,
                            note_cache=note_cache,
                            direction=direction,
                            )
                    for uniprot,gene,evidence,direction in l
                    ]))

        from rvw.prescreen import PrescreenData
        prot_table = PrescreenData.data_for_prots(self.ws, prots, prot_imp={}, clin_imp=0)
        prot_table.remove_columns({'Importance'})

        if self.only_moa_mols:
            if self.wsa.is_moa():
                self.context_alias(
                    mols_table=PrescreenData.mols_for_prots(self.ws, protdirs),
                )
            else:
           # This comment block was from a previous implementation
           # now that we've moved this to load on demand it's not really an issue,
           # but I left it here for now
                # XXX If this block is enabled, an MOA link will appear near
                # XXX the top of the Molecule Targets section, enabling a
                # XXX quick jump to the MOA molecule from a 'real' molecule.
                # XXX However, this more than doubles the annotate page load
                # XXX time when the caches are hot (.3 to .7 seconds on my
                # XXX machine, and 20% of the load time even when the caches
                # XXX are cold), and nobody really asked for this feature, so
                # XXX I'm leaving it disabled.
                # XXX
                # XXX A better implementation might be to have a dummy URL
                # XXX that takes a wsa_id, calculates the MOA wsa_id, and
                # XXX redirects to that page. Then the expensive logic would
                # XXX only get invoked when the MOA link was clicked. Consider
                # XXX this if anyone asks for this feature.
                # get corresponding MOA molecules
                from dtk.prot_map import DpiMapping
                from dtk.moa import make_wsa_to_moa_wsa,moa_dpi_variant
                mm = make_wsa_to_moa_wsa(
                        [self.wsa.id],
                        ws=self.ws,
                        dpi_mapping=DpiMapping(moa_dpi_variant(self.dpi)),
                        )
                self.context_alias(mols_table = None,
                                   moa_wsa_ids = list(mm.fwd_map().get(self.wsa.id,[])),
                                  )

        self.context_alias(
            prot_table=prot_table,
            )
        self.context['bindings']=bindings
        self.context['binding_list']=",".join(targ_cache.all_prots)
        self.context['dpi']=self.dpi
        self.context['dpi_thresh']=self.dpi_thresh
        self.timelog("got targets")
    def get_scorebox(self):
        from .utils import ScoreBox
        self.context_alias(scorebox = ScoreBox())
        self.scorebox.set_non_novel(self.ws.get_wsa_id_set('related'))
        from browse.models import Prescreen
        pscr = self.wsa.marked_prescreen
        if pscr and self.only_scorebox:
            from dtk.scores import SourceList
            src = SourceList.SourceProxy(
                    self.ws,
                    job_id=pscr.primary_job_id(),
                    label="PRESCREEN",
                    )
            self.scorebox.add_from_source(
                    src,
                    pscr.primary_code(),
                    self.wsa.id,
                    '*%d',
                    )
        # The actual marking prescreen is always the one that shows up with
        # the PRESCREEN label. But if there's a prescreen_id qparm, use that
        # to select the other scores to display.
        if self.prescreen_id:
            pscr = Prescreen.objects.get(pk=self.prescreen_id)
        from dtk.scores import get_sourcelist
        sl = get_sourcelist(
                self.ws,
                self.request.session,
                joblist=self.scorejobs,
                prescreen=pscr,
                )
        from .utils import TrgImpParmBuilder,get_wzs_jid_qparm
        tipb = TrgImpParmBuilder()
        tipb.build_from_source_list(sl)
        self.context['trgimp_qparms'] = tipb.extract_as_qparms()
        wzs_jid_qparam = get_wzs_jid_qparm(sl, pscr)
        self.context['scrimp_qparms'] = wzs_jid_qparam
        if wzs_jid_qparam:
            compev_qparms = wzs_jid_qparam+"&wsa="+str(self.wsa.id)
            orig = self.wsa.replacement_for.all()
            if orig:
                orig_wsa_ids = ','.join(str(wsa.id) for wsa in orig)
                compev_qparms += "&orig_wsa="+orig_wsa_ids
                orig_jids = []
                for orig_wsa in orig:
                    orig_sl = get_sourcelist(
                                        self.ws,
                                        self.request.session,
                                        joblist=self.scorejobs,
                                        prescreen=orig_wsa.marked_prescreen,
                                        )
                    orig_jids.append(str(get_wzs_jid_qparm(orig_sl, orig_wsa.marked_prescreen, jid_only=True)))
                compev_qparms += "&orig_wzs_jid="+','.join(orig_jids)
            self.context['compev_qparams'] = compev_qparms
        self._get_trg_scr_imp_qparms()
        # TrgImpParmBuilder already infers a prescreen id if there's
        # a single scorejobs parameter, and uses an explict prescreen
        # id from the querystring or if the drug is marked. So, base
        # the ScoreDetail qparm off that.
        if tipb._prescreen_id:
            self.context['score_detail_qparms']=(
                    '?prescreen_id=%d'%tipb._prescreen_id
                    )
        if self.only_scorebox:
            from .utils import DrugRunInfo
            dri = DrugRunInfo(self.wsa,sl)
            self.timelog('DrugRunInfo created')
            dri.append_job_scores(self.scorebox)
            self.timelog('job scores set')
            from .utils import append_tox_scores
            append_tox_scores(self.scorebox,self.wsa)
            self.timelog('tox scores set')
    def _get_trg_scr_imp_qparms(self, method = 'peel_cumulative'):
        from dtk.duma_view import qstr
        method_qp = qstr({'method':'peel_cumulative'})
        self.context['trg_scr_imp_qparms'] =  ''.join([self.context['scrimp_qparms'],
                                                  method_qp.replace('?', '&')
                                                 ])
    def get_election_data(self):
        '''Set self.vote and inactive_elections.

        self.vote points to an active vote record for this drug and user,
        if any (there should never be more than one).

        inactive_elections is a list of past elections, with each election
        tuple holding a list of votes, with each vote tuple holding a
        list of protein notes by that user. It is formatted as:
        [
            (Election object,[
                (Vote object,[
                    (prot,gene,note_id,txt),
                    ...
                    ]),
                ...
                ]),
            ...
        ]
        '''
        vote_lists = {}
        from browse.models import Vote
        qs = Vote.objects.filter(drug=self.wsa)
        qs = qs.prefetch_related('election')
        username=self.request.user.username
        active_vote = None
        for v in qs:
            if v.election.active():
                if v.reviewer == username and not v.disabled:
                    active_vote = v
            else:
                if v.note or v.recommended != None:
                    vote_lists.setdefault(v.election,[]).append(v)
        if active_vote or vote_lists:
            # Find all the protein notes for this drug; in the vote case,
            # the notes will be private, so pass the user name. This looks
            # somewhat redundant with get_binding, but that code allows
            # user overrides for dpi & thresh, where this forces defaults.
            # NOTE: We used to do that, but it is a problem if the ws dpi has
            # to be overridden for whatever reason, so switch back to the molecule
            # specific/selected target cache.
            targ_cache = self.targ_cache
            protgene = set(
                    (x[0],x[1])
                    for x in targ_cache.info_for_agent(self.wsa.agent_id)
                    )
            protgene = sorted(protgene,key=lambda x:x[1])
            user = self.request.user.username
            note_cache = targ_cache.build_note_cache(self.ws,user)
            if active_vote:
                # In the vote case, set up info needed by protein_notes form
                self.vote = active_vote
                self.prot_note_data = []
                for prot,gene in protgene:
                    note_id,text = note_cache.get(prot,{}).get(user,(None,''))
                    self.prot_note_data.append((prot,gene,note_id,text))

            if vote_lists:
                related_prot_wsas = self.get_related_prot_wsas(protgene)


                # In the vote_lists case, organize notes by user name for
                # merging into inactive_elections list.
                notes_for_user = {}
                # by scaning the note_cache in gene-sorted order, the
                # list values in notes_for_user are also gene-sorted
                for prot, gene in protgene:
                    if prot not in note_cache:
                        continue
                    for user,(note_id,text) in note_cache[prot].items():
                        l = notes_for_user.setdefault(user,[])
                        l.append((prot,gene,note_id,text,related_prot_wsas.get(prot, [])))
                self.context_alias(inactive_elections = [])
                for k,v in vote_lists.items():
                    vlist = []
                    self.inactive_elections.append((k,vlist))
                    for vote in sorted(v,key=lambda x:x.reviewer.lower()):
                        vlist.append((
                                vote,
                                notes_for_user.get(vote.reviewer,[]),
                                ))

    def get_related_prot_wsas(self, cur_protgenes):
        from browse.models import WsAnnotation
        vote_wsas = WsAnnotation.objects.filter(vote__election__ws=self.ws).distinct()
        from dtk.prot_map import AgentTargetCache, DpiMapping
        from browse.default_settings import DpiDataset, DpiThreshold
        # Use the non-combo DPI if in a combo, otherwise every drug has overlapping prots.
        dpi = DpiMapping(DpiDataset.value(ws=self.ws)).get_noncombo_dpi()
        atc = AgentTargetCache.atc_for_wsas(
            wsas=vote_wsas,
            dpi_mapping=dpi,
            dpi_thresh=DpiThreshold.value(ws=self.ws),
        )

        cur_prots = set(x[0] for x in cur_protgenes)
        from collections import defaultdict
        out = defaultdict(list)

        for wsa in vote_wsas:
            if wsa == self.wsa:
                continue
            related_prots = set(
                    x[0]
                    for x in atc.info_for_agent(wsa.agent_id)
                    )

            for prot in (related_prots & cur_prots):
                out[prot].append(wsa)
            
        return out

    def get_other_notes(self):
        # find notes and indications outside this workspace
        from browse.models import WsAnnotation
        enum = WsAnnotation.indication_vals
        qs = WsAnnotation.objects.filter(agent_id=self.wsa.agent_id)
        qs = qs.exclude(
                study_note__isnull=True,
                indication=enum.UNCLASSIFIED,
                )
        qs = qs.exclude(ws = self.ws)
        self.context_alias(other_notes = [])
        # now sort by review status and indication
        from dtk.data import MultiMap
        indi_map = MultiMap(
                ((bool(x.review_code),x.max_discovery_indication()),x)
                for x in qs
                ).fwd_map()
        # and construct an indication ordering
        order = WsAnnotation.ordered_selection_indications()
        order += WsAnnotation.ordered_kt_indications()
        # make sure we show everything (the WSA may be unclassified but
        # have a note, or maybe there's an indication val that doesn't
        # appear in the ordering lists)
        missed_indis = set(key[1] for key in indi_map) - set(order)
        order += sorted(missed_indis)
        # now add rows for each indication that appears in the map
        for reviewed in [True,False]:
            for indi in order:
                key = (reviewed,indi)
                if key in indi_map:
                    label_parts = []
                    if reviewed:
                        label_parts.append('Reviewed')
                    label_parts.append(enum.get('label',indi))
                    self.other_notes.append((
                            ' '.join(label_parts),
                            indi_map[key],
                            ))

class DrugRunDetailView(DumaView):
    @property
    def template_name(self):
        if self.section == 'pathprots':
            return 'moldata/_paths_section.html'
        if self.section == 'sbranks':
            return 'moldata/_sb_ranks_section.html'
        else:
            return 'moldata/drug_run_detail.html'
    GET_parms={
            'scorejobs':(str,''),
            'prescreen_id':(int,None),
            'pathway_set':(str,''),
            'mapping':(str,None),
            'show':(list_of(str),[]),
            'pbmsort':(SortHandler, 'is_cell_potency'),
            'section':(str, None),
            'section_info':(str, None),
            }
    def custom_setup(self):
        self.pbmsort.sort_parm='pbmsort'
    def custom_context(self):
        from dtk.html import decimal_cell
        from dtk.scores import Ranker, SourceList
        from dtk.files import get_file_records
        from collections import defaultdict
        import runner.data_catalog as dc
        self.context['show'] = self.show
        self.context['page_tab'] = 'detail(%d) %s' % (
                                self.ws.id,
                                self.wsa.get_name(self.is_demo()),
                                )
        if self.prescreen_id:
            from browse.models import Prescreen
            pscr = Prescreen.objects.get(pk=self.prescreen_id)
        else:
            pscr = self.wsa.marked_prescreen
        from dtk.scores import get_sourcelist
        sl = get_sourcelist(
                self.ws,
                self.request.session,
                joblist=self.scorejobs,
                prescreen=pscr,
                )
        from .utils import DrugRunInfo
        dri = DrugRunInfo(self.wsa, sl)
        if self.section == 'pathprots':
            protlist = dri.make_path_protlist(self.section_info, self.mapping)
            self.context_alias(prot_list=protlist)
        elif self.section == 'sbranks':
            sb_ranks = dri.get_sb_ranks()
            self.context_alias(sb_ranks = sb_ranks)
        else:
            meta_prot_options = dri.get_protmap_options()
            gsig_list=[]
            dri.append_gsig_data(gsig_list, self.mapping, 'gesig')
            dri.append_gsig_data(gsig_list, self.mapping, 'gwasig')
            dri.append_gsig_data(gsig_list, self.mapping, 'otarg')
            dri.append_gsig_data(gsig_list, self.mapping, 'tcgamut')
            dri.append_gsig_data(gsig_list, self.mapping, 'agr')
            dri.append_gsig_data(gsig_list, self.mapping, 'dgn')
            esga_list=[]
            dri.append_esga_data(esga_list, self.mapping)
            defus_list=[]
            dri.append_defus_data(defus_list)
            drug_score_list=[]
            dri.append_drug_score_data(drug_score_list, 'faers')
            from dtk.html import pad_table
            self.context_alias(
                    meta_prot_options = meta_prot_options,
                    gsig_list = gsig_list,
                    esga_list = esga_list,
                    defus_list = defus_list,
                    probeminer_data_table = self.get_probeminer_table(),
                    drug_score_list = drug_score_list,
                    )
        from browse.models import Protein
        self.context_alias(
            uniprot2gene_map = Protein.get_uniprot_gene_map(
                                    dri.used_uniprots
                                    ),
        )
    def get_probeminer_table(self):
        ver = self.ws.get_dpi_version()
        chembl_ids = self.wsa.agent.external_ids('chembl', ver)
        drugbank_ids = self.wsa.agent.external_ids('drugbank', ver)
        if not chembl_ids and not drugbank_ids:
            print('no chembl or drugbank ID found, so no probe miner data displayed')
            return []
        from dtk.s3_cache import S3MiscBucket,S3File
        from dtk.files import get_file_records
        from dtk.table import Table
        from dtk.html import link
        from browse.models import Protein

        def isfloat(value):
            try:
               float(value)
               return True
            except ValueError:
                return False
        ids = [str(x) for x in chembl_ids] + [str(x) for x in drugbank_ids]
        create_fn = 'probeMiner_create.tsv'
        create_file = S3File(S3MiscBucket(),create_fn)
        create_file.fetch(unzip=True)
        create_gen = get_file_records(create_file.path(),
                                      select=(ids,2),
                                      keep_header = False,
                                     )
        cansar_ids = set([x[0] for x in create_gen])
        if not cansar_ids:
            print('no cansar_ids found')
            return
        data_fn = 'probeMiner_data.tsv'
        data_file = S3File(S3MiscBucket(),data_fn)
        data_file.fetch(unzip=True)
        probeminer_lst = []
        data_gen = get_file_records(data_file.path(),
                                    select=(cansar_ids,0),
                                    keep_header = False,
                                    )
        for x in data_gen:
            p = Protein.get_canonical_of_uniprot(x[1])
            if not p:
                self.message("Probeminer returned unknown uniprot '%s'"%x[1])
                continue
            probeminer_lst.append((
                    (str(x[0]), str(p.uniprot)),
                    p,
                    bool(int(x[2])),
                    bool(int(x[3])),
                    int(x[4]),
                    bool(int(x[5])),
                    "%.3f" % float(x[6]),
                    "%.3f" % float(x[7]),
                    "%.3f" % float(x[8]),
                    "%.3f" % float(x[9]),
                    ))
        if not probeminer_lst:
            print('no probeminer data found')
            return
        def get_probeminer_url(tup):
            probeminer_url = 'https://probeminer.icr.ac.uk/#/'
            return link(tup[1] + '/' + tup[0], probeminer_url + tup[1] + '/' + tup[0], new_tab=True)

        data_columns = [
            Table.Column('ProbeMiner Link',
                    idx = 0,
                    cell_fmt = lambda x: get_probeminer_url(x),
                    sort = '12h',
                    code = 'cansar_id',
                    ),
            Table.Column('Target',
                    idx = 1,
                    cell_fmt = lambda x: x.get_uniprot_url(),
                    sort = '12h',
                    code = 'uniprot_id',
                    ),
            Table.Column('Is Cell Potency',
                    idx = 2,
                    sort = 'h2l',
                    code = 'is_cell_potency',
                    ),
            Table.Column('Is PAINS-free',
                    idx = 3,
                    sort = 'h2l',
                    code = 'is_pains',
                    ),
            Table.Column('Is Suitable Probe',
                    idx = 5,
                    sort = 'h2l',
                    code = 'is_suitable_probe',
                    ),
            Table.Column('Raw SAR',
                    idx = 4,
                    sort = 'h2l',
                    code = 'sar_raw',
                    ),
           Table.Column('Global',
                    idx = 6,
                    sort = 'h2l',
                    code = 'global',
                    ),
            Table.Column('Minimum SIC',
                    idx = 7,
                    sort = 'h2l',
                    code = 'sic_min',
                    ),
            Table.Column('Maximum SIC',
                    idx = 8,
                    sort = 'h2l',
                    code = 'sic_max',
                    ),
            Table.Column('Selectivity',
                    idx = 9,
                    sort = 'h2l',
                    code = 'selectivity',
                    ),
                  ]

        print('colspec', self.pbmsort.colspec)
        sort_idx = 0
        for x in data_columns:
            if x.code == self.pbmsort.colspec:
                sort_idx = x.idx
        probeminer_lst.sort(
                key=lambda x:sort_idx,
                reverse=self.pbmsort.minus,
                )

        return Table(
                     probeminer_lst,
                     data_columns,
                     url_builder=self.here_url,
                     sort_handler=self.pbmsort,
                    )
        probeminer_column_info = [
            ('Is Cell Potency:'
                ' True means that the compound binds to its target of interest'
                ' in cell lines with a median activity of less than 10 uM.'
                ),
            ('Is PAINS-free:'
                ' True means that the compound does not interfere with'
                ' detection methods of screening assays.'
                ),
            ('Is Suitable Probe:'
                ' True means 3 criteria are satisfied: 100 nM or better'
                ' on-target activity, 10-fold selective against other targets,'
                ' and permeable at less than 10 uM.'
                ),
            ('Raw SAR (Structure-Activity Relationship):'
                ' A measure of the confidence that the biological effect of'
                ' a given compound is achieved via the modulation of the'
                ' reference target. A higher value is a better score.'
                ),
            ('Global:'
                ' Combination of pains, interactive analog score, sar score,'
                ' cell score, selectivity score, and potency score. A higher'
                ' value is a better score.'
                ),
            ('SIC:'
                ' The summary of the differences beteween the median activity'
                ' of the reference target and the activity of each off-target'
                ' minus one.'
                ),
            ('Selectivity:'
                ' If the compound has been screened against other targets and'
                ' has 10-fold specificity for this targetSAR score.'
                ),
            ]



class MolCmpView(DumaView):
    template_name='moldata/molcmp.html'
    GET_parms={
            'search_wsa':(int,None),
            'search_prots':(str, ''),
            'ds':(str,''),
            }

    button_map={
            'search':['search'],
            'protsearch': ['protsearch'],
            'add': ['drugset'],
    	    'create': ['newdrugset'],
            }

    def get_dpi_map(self):
        from dtk.prot_map import DpiMapping
        from browse.default_settings import DpiDataset
        # Don't use combo DPIs here, or we'll find that every molecule is similar to
        # all others.
        # Similarly, moa-dpi isn't what we want.
        dpi_map = DpiMapping(DpiDataset.value(ws=self.ws)).get_baseline_dpi()
        return dpi_map


    def wsa_to_protlu(self, wsa_id):
        from dtk.prot_map import AgentTargetCache
        from browse.default_settings import DpiThreshold
        dpi_map = self.get_dpi_map()
        atc = AgentTargetCache.atc_for_wsas(wsas=[wsa_id], dpi_mapping=dpi_map, dpi_thresh=DpiThreshold.value(ws=self.ws))

        prot_lu = set()
        for key, prot, gene, ev, dr in atc.full_info_for_agent(atc.agent_ids[0]):
            # We're explicitly ignoring direction here.
            # Often these dir annotations are incorrect and we're searching for opposite.
            prot_lu.add((prot, gene, 0))
        return prot_lu

    def protlu_to_wsas(self, protlu):
        dpi_map = self.get_dpi_map()
        prot_to_dr = {prot:dr for prot, gene, dr in protlu}
        prots = prot_to_dr.keys()

        bindings = dpi_map.get_drug_bindings_for_prot_list(prot_list=prots)
        agentkeys = []
        for agentkey, prot, ev, dr in bindings:
            dr = int(dr)
            protdr = int(prot_to_dr[prot])
            if protdr == 0 or dr == 0 or protdr == dr:
                agentkeys.append(agentkey)

        key_wsa = dict(dpi_map.get_key_wsa_pairs(self.ws, keyset=agentkeys))

        wsas = [key_wsa[key] for key in agentkeys if key in key_wsa]
        return wsas

    def add_post_valid(self):
        p = self.context['drugset_form'].cleaned_data
        ds = p['ds']

        to_add = set()
        for name, value in self.request.POST.items():
            parts = name.split('_')
            if parts[0] == 'check':
                if value == 'on':
                    to_add.add(int(parts[1]))


        logger.info(f"Adding {to_add} to {ds}")
        from browse.models import DrugSet
        DrugSet.objects.get(pk=ds[2:]).drugs.add(*to_add)


        return HttpResponseRedirect(self.ws.reverse('moldata:hit_selection') + f'?ds={p["ds"]}')

    def make_newdrugset_form(self, data):
        initial = []
        if self.protlu:
            for prot, gene, dr in self.protlu:
                initial.append([prot, gene])

        from dtk.html import MultiField

        class MyForm(forms.Form):
            name = forms.CharField(label='Name')
            description = forms.CharField(
                    label='Description',
                    widget=forms.Textarea(attrs={'rows':'4','cols':'30'}),
                    required=False,
                    )
            prots=MultiField(label='MoA', initial=initial, jstype='ProtSearch')
        return MyForm(data)

    @transaction.atomic
    def create_post_valid(self):
        p = self.context['newdrugset_form'].cleaned_data
        from browse.models import DrugSet, Protein
        ds=DrugSet.objects.create(
                ws=self.ws,
                name=p['name'],
                description=p['description'],
                created_by=self.request.user.username,
                )
        import json
        prots = {prot for prot, gene in json.loads(p['prots'])}
        from moldata.models import MolSetMoa
        moa = MolSetMoa.objects.create(
                ds=ds,
                )
        moa.proteins.set(Protein.objects.filter(uniprot__in=prots))
        moa.save()
        return HttpResponseRedirect(self.here_url())



    def make_drugset_form(self, data):
        class MyForm(forms.Form):
            ds = forms.ChoiceField(
                    label = '',
                    choices = self.ws.get_wsa_id_set_choices(ds_only=True),
                    initial=self.ds,
                    )
            ds.widget.attrs.update({'class': 'form-control input-sm'})
        return MyForm(data)

    def search_post_valid(self):
        p = self.context['search_form'].cleaned_data
        # Override this.
        p['search_prots']=None

        return HttpResponseRedirect(self.here_url(**p))

    def make_search_form(self, data):
        from dtk.html import WsaInput
        class MyForm(forms.Form):
            search_wsa=forms.IntegerField(label='Similar To Molecule', initial=self.search_wsa, widget=WsaInput(ws=self.ws))
        return MyForm(data)

    def make_protsearch_form(self, data):
        initial = []
        if self.protlu:
            for prot, gene, dr in self.protlu:
                initial.append([prot, gene, dr])

        from dtk.html import MultiField
        class MyForm(forms.Form):
            search_prots=MultiField(label='Prots', initial=initial, jstype='ProtAndDirSelect')
        return MyForm(data)

    def protsearch_post_valid(self):
        p = self.context['protsearch_form'].cleaned_data
        # Override this.
        p['search_wsa']=None

        return HttpResponseRedirect(self.here_url(**p))

    def custom_setup(self):
        if self.search_wsa:
            self.protlu = self.wsa_to_protlu(self.search_wsa)
        elif self.search_prots:
            import json
            self.protlu = json.loads(self.search_prots)
        else:
            self.protlu = None

    def custom_context(self):
        if self.protlu:
            wsa_ids = self.protlu_to_wsas(self.protlu)

            from browse.models import WsAnnotation
            import dtk.molecule_table as MT
            from dtk.table import Table

            # There tends to be a tail of sparse DPI data - rather than
            # making the table grid huge with lots of empties, use a heuristic to
            # put all the tail DPI into an "Other" column.
            def other_col_filter(in_ref, idx, num_mols):
                return not in_ref and (idx >= 15 and num_mols < 10 or num_mols <= 1)

            ref_prots = [x[0] for x in self.protlu]
            col_groups = [
                    MT.Checkbox('Select'),
                    MT.Name(),
                    MT.StructureAndCore('Struct', dims=(400,200)),
                    MT.CommAvail(),
                    MT.MaxPhase(),
                    MT.Indication(),
                    MT.DpiSim(self.search_wsa, ref_prots=ref_prots, dpi=self.get_dpi_map()),
                    MT.EffRankColumn(),
                    MT.ReviewedInWs(),
                    MT.Dpi(self.search_wsa, ref_prots=ref_prots, other_col_filter=other_col_filter, dpi=self.get_dpi_map()),
                    ]

            wsas = WsAnnotation.objects.filter(pk__in=wsa_ids)
            wsas = wsas.exclude(agent__hide=True)
            wsas = WsAnnotation.prefetch_agent_attributes(wsas)


            cols = MT.resolve_cols(col_groups, wsas)
            self.context_alias(
                    table=Table(wsas, cols),
                    ds=self.ds,
                    )




class DrugCmpView(DumaView):
    template_name='moldata/drugcmp.html'
    index_dropdown_stem='rvw:review'
    GET_parms={
            'ids':(str,None),
            'search':(str,None),
            'dpi':(str,None),
            'dpi_t':(float,None),
            'ppi':(str,None),
            'ppi_t':(float,None),
            'max':(int,None),
            'prsim':(float,1.0),
            'dirJac':(float,0.5),
            'indJac':(float,0.5),
            'rdkit':(float,1.0),
            'indigo':(float,1.0),
            'do_plot':(boolean,False),
            }
    button_map={
            'search':['search'],
            'find':['target'],
            }
    def make_target_form(self, data):
        from dtk.prot_map import DpiMapping,PpiMapping
        class MyForm(forms.Form):
            dpi = forms.ChoiceField(
                label = 'DPI dataset',
                choices = DpiMapping.choices(self.ws),
                initial = self.ws.get_dpi_default(),
                )
            dpi_t = forms.FloatField(
                label = 'Min DPI evidence',
                initial = self.ws.get_dpi_thresh_default(),
                )
            ppi = forms.ChoiceField(
                label = 'PPI dataset',
                choices = PpiMapping.choices(),
                initial = self.ws.get_ppi_default(),
                )
            ppi_t = forms.FloatField(
                label = 'Min PPI evidence',
                initial = self.ws.get_ppi_thresh_default(),
                )
            max = forms.IntegerField(
                label = 'Maximum number of drugs to show',
                initial = 10,
                )
            prsim = forms.FloatField(
                label = 'PRSim weight',
                initial = 0,
                )
            dirJac = forms.FloatField(
                label = 'Direct JacSim weight',
                initial = 1,
                )
            indJac = forms.FloatField(
                label = 'Indirect JacSim weight',
                initial = 0,
                )
            indigo = forms.FloatField(
                label = 'Indigo weight',
                initial = 0,
                )
            rdkit = forms.FloatField(
                label = 'RDKit weight',
                initial = 0,
                )
        return MyForm(data)
    def make_search_form(self, data):
        class MyForm(forms.Form):
            search=forms.CharField(
                label = 'Search for more drugs by name',
                max_length = 50,
                )
        return MyForm(data)
    def find_post_valid(self):
        p = self.context['target_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))
    def search_post_valid(self):
        p = self.context['search_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))
    def custom_context(self):
        try:
            self.id_list = self.ids.split(",")
        except AttributeError:
            return
        self.load_drugs()
        self.prots = []
        self.pph = []
        if self.search:
            self.make_query()
            if self.qs:
                return self.process_search()
            else:
                self.message = "No drugs found matching '%s'" % self.search
        elif self.dpi and self.ppi:
            from scripts.metasim import metasim
            self.ms = metasim(refset = [x.id for x in self.drugs],
                             dpi = self.dpi,
                             ppi = self.ppi,
                             ws = self.ws,
                             dpi_t = self.dpi_t,
                             ppi_t = self.ppi_t,
                             wts = {'rdkit': self.rdkit,
                                    'prMax': self.prsim,
                                    'indigo': self.indigo,
                                    'dirJac': self.dirJac,
                                    'indJac': self.indJac
                                   },
                             std_gene_list_set=None,
                             )
            self.ms.setup()
            self.ms.run()
            self.ms.full_combine_scores()
            return self.process_matches()
        else:
            if self.dpi_t is None:
                self.dpi_t = self.ws.get_dpi_thresh_default()
            from .utils import find_prots
            self.protsD = find_prots(self.ws,self.drugs, self.dpi, self.dpi_t)
            self.order_prots()
            from dtk.html import link
            if self.do_plot:
                self.get_score_rank()
                self.plot()
                tog_link=link(
                        'Redisplay without heatmap',
                        self.here_url(do_plot=0),
                        )
            else:
                tog_link=link(
                        'Redisplay with heatmap',
                        self.here_url(do_plot=1),
                        )
            self.context['plot_toggle_link'] = tog_link
            self.message = ''
        self.context_alias(
                ids=self.id_list,
                drugs=self.drugs,
                prots=self.prots,
                message=self.message,
                plotly_plots=self.pph
                )
    def load_drugs(self):
        from browse.models import WsAnnotation
        self.drugs = [WsAnnotation.objects.get(id=d) for d in self.id_list if d]
        self.drug_names = {d.id:d.get_name(self.is_demo()) for d in self.drugs}
        ### deal with repetitive drug names (should only happen if is_demo is true)
        from dtk.data import uniq_name_list
        uniq_keys = uniq_name_list(list(self.drug_names.values()))
        for i in range(len(uniq_keys)):
            self.drug_names[list(self.drug_names.keys())[i]] = uniq_keys[i]
    def add_found_drugs(self, new_ids):
        return HttpResponseRedirect(self.request.path+'?ids='+new_ids)
    def process_matches(self):
        self.ms.matches.sort(key=lambda x: x[1], reverse=True)
        if len(self.ms.matches) > self.max:
            self.ms.matches = self.ms.matches[:self.max]
        return self.add_found_drugs(",".join([str(x[0]) for x in self.ms.matches]))
    def order_prots(self):
        self.prots = []
        import math
        for p,v in self.protsD.items():
            k = 0
            for i,f in enumerate(v):
                if f:
                    k += math.pow(2,2-i)
            self.prots.append([p,k,v])
        self.prots.sort(key=lambda x: x[1], reverse=True)
    def get_score_rank(self):
        from dtk.scores import Ranker
        self.scores = {}
        self.ranks = {}
        from dtk.scores import SourceList
        import runner.data_catalog as dc
        sl = SourceList(self.ws)
        sl.load_from_session(self.request.session)
        for job_id in (int(x.job_id()) for x in sl.sources()):
            src = SourceList.SourceProxy(self.ws,job_id=job_id)
            bji = src.bji()
            cat = bji.get_data_catalog()
            for code in cat.get_codes('wsa','score'):
                    drug_scores = [
                            cat.get_cell(code,d.id)[0]
                            for d in self.drugs
                            ]
                    if not drug_scores == [None]*len(drug_scores):
                        k = " ".join([src.label(), code])
                        self.scores[k] = {}
                        self.ranks[k] = {}
                        ranker = Ranker(cat.get_ordering(code,True))
                        for i,s in enumerate(drug_scores):
                            d_id = self.drugs[i].id
                            d = self.drug_names[d_id]
                            if s is None:
                                self.scores[k][d] = "NA"
                            else:
                                self.scores[k][d] = "%0.2f" % s
                            self.ranks[k][d] = ranker.get_details(d_id)
    def plot(self):
        if not self.scores:
            return
        from dtk.plot import plotly_heatmap
        from browse.utils import build_dat_mat
        import numpy as np
        self._prep_Ds()
        plot_data, col_names = build_dat_mat(
                                    self.rank_score,
                                    self.score_names
                                )
        hover_text = [[" ".join(['Ranked higher:',
                                 str(self.ranks[k][d][0]),
                                 '<br>Tied:',
                                 str(self.ranks[k][d][1]),
                                 '<br>Score:',
                                 self.scores[d][k]
                                ])
                        for d in col_names]
                      for k in self.score_names
                     ]
        x,y = np.array(plot_data).shape
        padding_needed = max([len(i) for i in col_names])
        self.pph = [('heatmap',
                    plotly_heatmap(
                     np.array(plot_data)
                       , self.score_names
                       , Title = 'Drug-Compare Score-Rank Heatmap'
                       , color_bar_title = "Log10(Rank)"
                       , col_labels = col_names
                       , hover_text = np.array(hover_text)
                       , invert_colors = True
                       , zmin = 0.0
                       , height = x*24
                       , width = 800 + padding_needed*11
                   )
                   )]
    def _prep_Ds(self):
        from math import log
        self.score_names = list(self.scores.keys())
        new_sd = {}
        new_rd = {}
        for d in self.drug_names.values():
            new_sd[d] = {}
            new_rd[d] = {}
            for k in self.score_names:
                new_sd[d][k] = self.scores[k][d]
                new_rd[d][k] = log(self.ranks[k][d][0] + (self.ranks[k][d][1]/2) + 1, 10)
        self.scores = new_sd
        self.rank_score = new_rd
    def make_query(self):
        from browse.models import WsAnnotation
        qs_ws = WsAnnotation.objects.filter(ws=self.ws)
        from browse.utils import drug_search_wsa_filter
        self.qs = drug_search_wsa_filter(qs_ws,self.search)
    def process_search(self):
        if len(self.qs) == 1:
            return self.add_found_drugs(self.add_one_match())
        else:
            self.do_search()
        return
    def add_one_match(self):
        # if one found, add id to the list
        # and re-direct back here with the search string removed
        d = list(self.qs)[0]
        self.drugs.append(d)
        return ','.join([str(d.id)
                         for d in self.drugs
                        ])
    def do_search(self):
            # if more than one found, render a selection list, with all links
            # pointing back here with an updated id string; also have a
            # 'cancel' link on the selection page that returns here with
            # the original id string.
        self.template_name = 'moldata/drugcmp_list.html'
        self.context_alias(ids=",".join(self.id_list),
                           matches=self.qs,
                           search=self.search
                          )

class AssaysView(DumaView):
    template_name='moldata/assays.html'
    GET_parms = {'sort' : (SortHandler, 'nm')}
    def setup_raw_table(self):
        aa = self.wsa.get_raw_assay_info()
        from collections import namedtuple
        gene_idx = aa.info_cols.index('gene')
        prot_idx = aa.info_cols.index('protein')
        RowType = namedtuple('RowType',['gene_protein']+aa.info_cols)
        rows = [RowType(
                        '%s (%s)'%(x[gene_idx],x[prot_idx]),
                        *x)
                for x in aa.assay_info()
                ]
        rows.sort(
                key=lambda x:(
                        getattr(x,self.sort.colspec),
                        # in case of ties on the primary key,
                        # present matching rows in a reasonable order:
                        x.drug_key,
                        x.protein,
                        x.nm,
                        ),
                reverse=self.sort.minus,
                )
        from dtk.url import chembl_assay_url, bindingdb_drug_url
        from dtk.html import link
        def drug_link(key):
            if key.startswith('CHEMBL'):
                return link(key,chembl_assay_url(key),new_tab=True)
            elif key.startswith('BDBM'):
                return link(key,bindingdb_drug_url(key),new_tab=True)
            else:
                return key
        from dtk.table import Table
        columns = [
                Table.Column('Assay Type',
                        ),
                Table.Column('Drug Key',
                        cell_fmt=drug_link,
                        ),
                Table.Column('Gene/Protein',
                        cell_html_hook=lambda data,row,table:link(
                                data,
                                self.ws.reverse('protein',row.protein),
                                new_tab=True,
                                ),
                        ),
                Table.Column('Direction',),
                Table.Column('Relation',),
                Table.Column('nM',
                        ),
                Table.Column('Format', code='assay_format'),
                ]
        self.context['raw_table'] = Table(
                rows,
                columns,
                sort_handler = self.sort,
                url_builder = self.here_url,
                )

    def custom_context(self):
        self.setup_raw_table()
        aa = self.wsa.get_assay_info()
        from collections import namedtuple
        gene_idx = aa.info_cols.index('gene')
        prot_idx = aa.info_cols.index('protein')
        RowType = namedtuple('RowType',['gene_protein']+aa.info_cols)
        rows = [RowType(
                        '%s (%s)'%(x[gene_idx],x[prot_idx]),
                        *x)
                for x in aa.assay_info()
                ]
        rows.sort(
                key=lambda x:(
                        getattr(x,self.sort.colspec),
                        # in case of ties on the primary key,
                        # present matching rows in a reasonable order:
                        x.drug_key,
                        x.protein,
                        x.nm,
                        ),
                reverse=self.sort.minus,
                )
        from dtk.url import chembl_assay_url, bindingdb_drug_url
        from dtk.html import link
        def drug_link(key):
            if key.startswith('CHEMBL'):
                return link(key,chembl_assay_url(key),new_tab=True)
            elif key.startswith('BDBM'):
                return link(key,bindingdb_drug_url(key),new_tab=True)
            else:
                return key
        from dtk.table import Table
        columns = [
                Table.Column('Assay Type',
                        sort='l2h',
                        ),
                Table.Column('Drug Key',
                        cell_fmt=drug_link,
                        sort='l2h',
                        ),
                Table.Column('Gene/Protein',
                        cell_html_hook=lambda data,row,table:link(
                                data,
                                self.ws.reverse('protein',row.protein),
                                new_tab=True,
                                ),
                        sort='l2h',
                        ),
                Table.Column('Direction',
                        sort='l2h',
                        ),
                Table.Column('nM',
                        sort='l2h',
                        ),
                Table.Column('Assay Count',
                        sort='l2h',
                        ),
                Table.Column('Std. Dev.',
                        code='std_dev',
                        sort='l2h',
                        ),
                ]
        self.context['table'] = Table(
                rows,
                columns,
                sort_handler = self.sort,
                url_builder = self.here_url,
                )

class NoneffAssaysView(DumaView):
    template_name='moldata/noneff_assays.html'
    GET_parms={
            'ver_override':(int,None),
            }
    def custom_context(self):
        import dtk.assays as assays
        ver = self._get_mol_version()
        # 'ver' is the dpi/matching version used to find the chembl ids
        # associated with this agent; the chembl assays file retrieved
        # is the one from the input version of chembl used to make the
        # matching file
        adme_data, err = assays.load_dmpk_assays(self.wsa.agent, 'adme', ver)
        if err:
            self.message(err)
        pc_data, err = assays.load_dmpk_assays(self.wsa.agent, 'pc', ver)
        if err:
            self.message(err)
        tox_data, err = assays.load_dmpk_assays(self.wsa.agent, 'tox', ver)
        if err:
            self.message(err)

        adme_parse = [assays.interpret_dmpk_assay(x) for x in adme_data]
        pc_parse = [assays.interpret_dmpk_assay(x) for x in pc_data]
        tox_parse = [assays.interpret_dmpk_assay(x) for x in tox_data]

        def desc_url(assay):
            desc = assay.description
            qs = f'molecule_chembl_id:{assay.chembl_id} AND assay_chembl_id:{assay.assay_chembl_id}'
            href = 'https://www.ebi.ac.uk/chembl/g/#browse/activities/filter/'+qs
            from dtk.html import link
            return link(desc, href, new_tab=True)

        adme_rows = [
                (desc_url(a), a.assay_type, a.relation, a.value, a.unit, a.organism, parse.category)
                for a, parse in zip(adme_data, adme_parse)
                ]
        pc_rows = [
                (desc_url(a), a.assay_type, a.relation, a.value, a.unit, a.organism, parse.category)
                for a, parse in zip(pc_data, pc_parse)
                ]
        tox_rows = [
                (desc_url(a), a.assay_type, a.relation, a.value, a.unit, a.organism, parse.category)
                for a, parse in zip(tox_data, tox_parse)
                ]
        columns = [{'title': col} for col in ['Assay Description', 'Type', 'Relation', 'Value', 'Unit', 'Organism', 'Category']]
        self.context_alias(
                    tables = [
                        ('ADME', 'adme', adme_rows, columns, [[0, 'desc']]),
                        ('Physicochemical', 'pc', pc_rows, columns, [[0, 'desc']]),
                        ('Toxicity', 'tox', tox_rows, columns, [[0, 'desc']]),
                        ],
                    )
    def _get_mol_version(self):
        if self.ver_override:
            return self.ver_override
        else:
            return self.wsa.ws.get_dpi_version()

class PatentDetailView(DumaView):
    template_name = 'moldata/patent_detail.html'
    GET_parms = {'sort' : (SortHandler, 'patent_id')}

    def plot(self, df):
        from dtk.plot import PlotlyPlot, Color
        import pandas as pd
        self.context['patent_plots'] = []

        plot_values = sorted(list(zip(df['date'].value_counts().index,
                                      df['date'].value_counts().values)))
        x = [x[0] for x in plot_values]
        y = [y[1] for y in plot_values]
        data = [dict(x=x, y=y,type='bar')]

        self.context['patent_plots'].append(('patent_count_bar', PlotlyPlot(
                                    [dict(x = x, y = y, type='bar')],
                                    {'title':'Number of Patents Over Time, n = {}'.format(sum(y)),
                                     'yaxis':dict(title='Number of Patents'), 'xaxis':dict(title='Year')})))

    def custom_context(self, empty_cell = '-'):
        from dtk.table import Table
        from dtk.s3_cache import S3File,S3MiscBucket
        from dtk.files import get_file_records
        from dtk.url import google_patent_url
        from dtk.html import link
        import pandas as pd
        import numpy as np

        template_name='browse/patent_detail.html'
        GET_parms = {'sort' : (SortHandler, 'patent_id')}

        # Get the disease counts (from USPTO)
        tmp_disease_lst = []
        # We don't fill this out anymore, but could grab it from existing data.

        # Get the drug counts (from SureChEMBL)
        self.surechembl_ids = self.wsa.get_surechembl_id()
        if self.surechembl_ids is None:
            self.surechembl_ids = []
        self.context['surechembl_ids'] = ' '.join(self.surechembl_ids)
        s3f = S3File(S3MiscBucket(),'patent.surechembl.full.tsv.gz')
        s3f.fetch()
        gen = get_file_records(s3f.path(),
                  select=(self.surechembl_ids,0),
                  keep_header = False)
        tmp_drug_lst = []
        for item in gen:
            item[1] = item[1].replace('-', '')
            item[2] = int(item[2])
            item[3] = int(item[3])
            item[4] = int(item[4])
            tmp_drug_lst.append(item)

        # Merge the disease and drug counts
        drug_lst = [[i[1], i[0], i[2], i[3], i[4], i[5]] for i in tmp_drug_lst]
        disease_lst = [[i[1], i[0], i[2], i[3], i[4], i[5]] for i in tmp_disease_lst]
        for i, sublist in enumerate(disease_lst):
             if sublist[5] == '':
                 disease_lst[i][5] = empty_cell
        drug_dict = {i[0]:i[1:] for i in drug_lst}
        disease_dict = {i[0]:i[1:] for i in disease_lst}

        for k, v in disease_dict.items():
            if k in drug_dict:
                drug_dict[k].extend(v)

        tmp_master_lst = [[k] + v for k, v in drug_dict.items()]
        master_lst = []
        for i in tmp_master_lst:
            row = [i[0], i[5], i[2], i[3], i[4]]
            if len(i) > 6:
                row += [int(i[7]), int(i[8]), int(i[9]), i[10]]
            else:
                row += [empty_cell]*4
            master_lst.append(row)
        df = pd.DataFrame(master_lst, columns = ['patent_id', 'date', 'drug_title_count',
                                                 'drug_abstract_count', 'drug_claims_count',
                                                 'disease_title_count', 'disease_abstract_count',
                                                 'disease_claims_count', 'assignee'])
        df['patent_abbreviation'] = df['patent_id'].apply(lambda x: x[:-3])
        df.sort_values(by='patent_id', inplace=True)
        df.drop_duplicates(subset='patent_abbreviation', keep='last', inplace=True)
        df = df[['patent_id', 'date', 'drug_title_count', 'drug_abstract_count', 'drug_claims_count',
                 'disease_title_count', 'disease_abstract_count', 'disease_claims_count', 'assignee']]
        df['date'] = df['date'].apply(lambda x: x[0:4])
        lst = df.values.tolist()
        if lst:
            self.plot(df)

        columns = [
            Table.Column('Patent ID',
                    idx = 0,
                    cell_fmt = lambda x: link(x, google_patent_url(str(x)), new_tab=True),
                    sort = '12h',
                    code = 'patent_id',
                    ),
            Table.Column('Publication Date',
                    idx = 1,
                    sort = 'h2l',
                    code = 'publication_date',
                    ),
            Table.Column('Drug Title Count',
                    idx = 2,
                    sort = 'h2l',
                    code = 'drug_title_count',
                    ),
            Table.Column('Drug Abstract Count',
                    idx = 3,
                    sort = 'h2l',
                    code = 'drug_abstract_count',
                    ),
            Table.Column('Drug Claims Count',
                    idx = 4,
                    sort = 'h2l',
                    code = 'drug_claims_count',
                    ),
            Table.Column('Disease Title Count',
                    idx = 5,
                    sort = 'h2l',
                    code = 'disease_title_count',
                    ),
            Table.Column('Disease Abstract Count',
                    idx = 6,
                    sort = 'h2l',
                    code = 'disease_abstract_count',
                    ),
            Table.Column('Disease Claims Count',
                    idx = 7,
                    sort = 'h2l',
                    code = 'disease_claims_count',
                    ),
            Table.Column('Assignee',
                    idx = 8,
                    sort = 'h2l',
                    code = 'assignee',
                    ),

            ]
        sort_idx=1
        for x in columns:
           if x.code == self.sort.colspec:
               sort_idx=x.idx
        lst.sort(key=lambda x: (x[sort_idx] is not '-', x[sort_idx]), reverse=self.sort.minus)
        self.context['surechembl_table'] = Table(
                        lst,
                        columns,
                        sort_handler = self.sort,
                        url_builder = self.here_url,
                        )

class DispositionAuditView(DumaView):
    template_name='moldata/dispositionaudit.html'
    button_map={
            'ignore': [],
            'unignore': [],
            }

    def ignore_post_valid(self):
        da_id = self.request.POST['da_id']
        logger.info("Ignoring DA %s", da_id)
        from browse.models import DispositionAudit
        da = DispositionAudit.all_objects.get(pk=da_id)
        da.ignore = True
        da.save()
        return HttpResponseRedirect(self.here_url())

    def unignore_post_valid(self):
        da_id = self.request.POST['da_id']
        logger.info("Unignoring DA %s", da_id)
        from browse.models import DispositionAudit
        da = DispositionAudit.all_objects.get(pk=da_id)
        da.ignore = False
        da.save()
        return HttpResponseRedirect(self.here_url())

    def custom_context(self):
        from browse.models import DispositionAudit
        das = DispositionAudit.all_objects.filter(wsa=self.wsa).order_by('timestamp')
        self.context_alias(
                das=das
                )


class HitSelectionReportView(DumaView):
    template_name='moldata/hit_selection_report.html'
    GET_parms={
            'ds':(str,None),
            }

    button_map={
            'save':['note'],
            }
    def make_note_form(self,data):
        from notes.models import Note
        dsmoa = self.dsmoa
        preclin_note = Note.get(dsmoa, 'preclin_note', '')
        class NoteForm(forms.Form):
            note = forms.CharField(
                    widget=forms.Textarea(attrs={'rows':'6','cols':'60'}),
                    required=False,
                    initial=preclin_note,
                    )
        return NoteForm(data)
    def save_post_valid(self):
        p = self.note_form.cleaned_data
        from notes.models import Note
        Note.set(self.dsmoa,
                'preclin_note',
                self.request.user.username,
                p['note'],
                )
        return HttpResponseRedirect(self.here_url())

    def custom_setup(self):
        from moldata.models import MolSetMoa
        self.dsmoa = MolSetMoa.objects.get(ds_id=self.ds[2:])

    def custom_context(self):
        # For each molecule, present the information available.
        from browse.models import WsAnnotation
        self.wsas = WsAnnotation.objects.filter(pk__in=self.ws.get_wsa_id_set(self.ds))

        from notes.models import Note
        dsmoa = self.dsmoa
        moa_note = Note.get(dsmoa, 'hitsel_note', '')
        genes = dsmoa.proteins.all().values_list('gene', flat=True)

        wsas = WsAnnotation.prefetch_agent_attributes(self.wsas)
        import dtk.molecule_table as MT
        from django.utils.safestring import mark_safe
        max_phase_data = WsAnnotation.bulk_max_phase(wsas)
        score_cols = MT.EditableScoreColumns(max_phase_data)

        score_data, overall_scores = score_cols.make_datas(wsas)

        report = []

        for wsa in sorted(wsas, key=lambda x: -overall_scores[x.id]):
            overall = overall_scores[wsa.id]
            img = MT.Structure()._extract(wsa)
            wsa_data = [('', mark_safe(f'<div class="score"><b>Overall Score</b>: {overall:.2f} / 4</div>{img}'))]

            for extractor, col in score_data:
                note, score, detail_data = extractor(wsa)

                if (score is None or score == '') and not note:
                    continue
                section = col.label

                detail_list = ['<li>' + x for x in detail_data]
                score_el = f'<div class="score"><b>Score</b>: {score:.1f} / 4</div>' if score is not None else ''
                details_el = f'''
                    <h4>Automated Details</h4>
                    <ul>
                    {''.join(detail_list)}
                    </ul>
                ''' if detail_list else ''

                from django.utils.html import urlize, linebreaks
                note = linebreaks(urlize(note))

                entry = mark_safe(f'''
                    {score_el}
                    {note}<br>
                    {details_el}
                ''')

                wsa_data.append((section, entry))


            report.append((wsa, wsa_data))


        self.context_alias(
            report=report,
            moa_note=moa_note,
            genes=' '.join(genes),
            )


class HitSelectionView(DumaView):
    template_name='moldata/hit_selection.html'
    GET_parms={
            'ds':(str,None),
            }

    button_map={
            'select':['moleculeset'],
            'save': [],
            'moa': ['moa'],
            'addmol': ['addmol'],
            'delwsa': [],
            'savehitselnote': ['hitselnote'],
            }

    def get_dsmoa(self):
        from moldata.models import MolSetMoa
        moa, new = MolSetMoa.objects.get_or_create(ds_id=self.ds[2:])
        return moa

    def make_moa_form(self, data):
        initial = []
        if self.ds:
            moa = self.get_dsmoa()
            for prot in moa.proteins.all():
                initial.append([prot.uniprot, prot.gene])

        from dtk.html import MultiField
        class MyForm(forms.Form):
            prots=MultiField(label='Prots', initial=initial, jstype='ProtSearch')
        return MyForm(data)

    def make_addmol_form(self, data):
        from dtk.html import MultiWsaField
        class MyForm(forms.Form):
            mols=MultiWsaField(label='Add New Molecules To Group', ws_id=self.ws.id)
        return MyForm(data)
    
    def make_hitselnote_form(self, data):
        from notes.models import Note
        class MyForm(forms.Form):
            hitsel_note=forms.CharField(
                    widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
                    required=False,
                    initial = Note.get(self.ws, 'hitsel_note', self.request.user.username),
                    label='Note',
            )
        return MyForm(data)

    def savehitselnote_post_valid(self):
        from notes.models import Note
        p = self.context['hitselnote_form'].cleaned_data
        Note.set(self.ws, 'hitsel_note', user=self.request.user.username, text=p['hitsel_note'])
        return HttpResponseRedirect(self.here_url())


    def moa_post_valid(self):
        from moldata.models import MolSetMoa
        from browse.models import Protein
        p = self.context['moa_form'].cleaned_data
        import json

        moa = self.get_dsmoa()
        prots = {prot for prot, gene in json.loads(p['prots'])}
        moa.proteins.set(Protein.objects.filter(uniprot__in=prots))
        moa.save()

        return HttpResponseRedirect(self.here_url())

    @transaction.atomic
    def addmol_post_valid(self):
        from browse.models import DrugSet
        p = self.context['addmol_form'].cleaned_data
        ds = DrugSet.objects.get(pk=self.ds[2:])
        mols = p['mols'].split(',')
        ds.add_mols(mols, self.request.user)
        return HttpResponseRedirect(self.here_url())

    def delwsa_post_valid(self):
        from browse.models import DrugSet
        wsa = self.request.POST['wsa_id']
        ds = DrugSet.objects.get(pk=self.ds[2:])
        ds.remove_mols([wsa], self.request.user)
        return HttpResponseRedirect(self.here_url())


    def make_moleculeset_form(self, data):
        class MyForm(forms.Form):
            ds = forms.ChoiceField(
                    label = 'Molecule Set',
                    choices = self.ws.get_wsa_id_set_choices(ds_only=True),
                    initial = self.ds,
                    )
        return MyForm(data)

    def select_post_valid(self):
        p = self.context['moleculeset_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))

    def save_post_valid(self):
        
        from notes.models import Note
        text = self.request.POST['hitsel_note']
        dsmoa = self.get_dsmoa()
        old_hitsel_note = Note.get(dsmoa, 'hitsel_note', '')
        if text != old_hitsel_note:
            Note.set(dsmoa, 'hitsel_note', self.request.user.username, text)
        

        from collections import defaultdict
        data = defaultdict(dict)
        errors = []
        for key, val in self.request.POST.items():
            parts = key.split('_')
            if parts[0] == 'score' or parts[0] == 'note':
                part, wsaid, scoretype = parts
                data[(wsaid, scoretype)][part] = val
                try:
                    if part == 'score' and val and (float(val) > 4 or float(val) < 0):
                        errors.append(f"Score {val} must be between 0 and 4")
                except ValueError:
                    errors.append("Score {val} is not a number")

        if errors:
            for error in errors:
                self.message(error)
            return

        from moldata.models import HitScoreValue
        from dtk.molecule_table import EditableHitScoreValue
        for (wsaid, scoretype), value in data.items():
            if value['note'].startswith(EditableHitScoreValue.AUTO_PREFIX):
                # Don't explicitly save auto values.
                continue
            HitScoreValue.update(wsaid, scoretype, value.get('score', 0), value['note'], self.request.user.username)

        return HttpResponseRedirect(self.here_url())

    def _make_hitsel_summary(self):
        from moldata.models import MolSetMoa
        from notes.models import Note
        moas = MolSetMoa.objects.filter(ds__ws=self.ws)

        def ds_link(moa):
            url = f"{self.ws.reverse('moldata:hit_selection_report')}?ds=ds{moa.ds.id}"
            return f'{moa.ds.name}\n<a href="{url}">View Report</a>'
        
        def mol_data(moa):
            wsas = moa.ds.drugs.all()

            return '\n'.join(f'{x.html_url()} ({x.indication_label()})' for x in wsas)


        data = [{
            'name': ds_link(moa),
            'genes': '\n'.join(x.get_html_url(self.ws.id) for x in moa.proteins.all()),
            'hitsel_note': Note.get(moa, 'hitsel_note', self.request.user),
            'preclin_note': Note.get(moa, 'preclin_note', self.request.user),
            'mols': mol_data(moa),
        } for moa in moas]

        from dtk.table import Table

        cols = [
            Table.Column('Name', idx='name'),
            Table.Column('Genes', idx='genes'),
            Table.Column('Hit Selection Note', idx='hitsel_note'),
            Table.Column('Preclinical Team Note', idx='preclin_note'),
            Table.Column('Molecules', idx='mols'),
        ]

        self.context_alias(
            summary_table=Table(data, cols)
        )
        

    def custom_context(self):
        if self.ds:
            from browse.models import WsAnnotation
            from notes.models import Note
            self.wsas = WsAnnotation.objects.filter(pk__in=self.ws.get_wsa_id_set(self.ds))
            wsas = WsAnnotation.prefetch_agent_attributes(self.wsas)

            dsmoa = self.get_dsmoa()
            hitsel_note = Note.get(dsmoa, 'hitsel_note', '')


            import dtk.molecule_table as MT
            from dtk.table import Table

            max_phase_data = WsAnnotation.bulk_max_phase(wsas)

            col_groups = [
                    MT.DeleteColumn(),
                    MT.Name(),
                    MT.DpiColumn(),
                    MT.Structure(),
                    MT.MaxPhase(max_phase_data),
                    MT.EditableScoreColumns(max_phase_data),
                    ]

            cols = MT.resolve_cols(col_groups, wsas)
            from browse.models import DrugSet
            ds_name = DrugSet.objects.get(pk=self.ds[2:]).name
            self.context_alias(
                    table=Table(wsas, cols),
                    ds_name=ds_name,
                    hitsel_note=hitsel_note,
                    )
        else:
            self._make_hitsel_summary()

