from django.shortcuts import render

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, HttpResponseRedirect
from dtk.duma_view import DumaView,list_of,boolean
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.db import transaction
from django import forms
from dtk.url import multiprot_search_links
from notes.models import Note
import json



class ConsultantView(DumaView):
    template_name='consultant/index.html'

    def custom_context(self):
        from browse.models import Vote
        votes = Vote.user_votes(self.request.user)
        from dtk.data import MultiMap

        election_to_votes = MultiMap((v.election, v) for v in votes
                                     if v.election.active()).fwd_map()

        # Works in py3.7+ due to dict insertion ordering guarantees.
        election_to_votes = dict(sorted(
            election_to_votes.items(),
            key=lambda x: x[0].due
            ))
        
        self.context_alias(election_to_votes=election_to_votes)


class ConsultantMoleculeView(DumaView):
    template_name='consultant/molecule.html'
    def custom_setup(self):
        self.button_map={}
        self.get_election_data()
        if not self.vote:
            # Consultants have access only when they have a vote.
            return HttpResponseRedirect('/consultant/')

        self.button_map.update(vote=['vote','protein_notes'])

    def custom_context(self):
        from dtk.url import ext_drug_links
        self.get_binding()

        if self.wsa.is_moa():
            links = list(multiprot_search_links(self.ws, self.prots).items())
            # Rename it from 'srch' to something more user friendly.
            links[0] = ['All Genes Search', links[0][1]]
        else:
            links = ext_drug_links(self.wsa, split=True)
        
        self.context_alias(
                drug_ws=self.wsa,
                ext_drug_links=links,
                )
    def get_election_data(self):
        from browse.models import Vote
        qs = list(Vote.user_votes(self.request.user).filter(drug=self.wsa))
        if len(qs) > 0:
            self.vote = qs[0]
        else:
            self.vote = None


        self.prot_note_data = []
        targ_cache = self.ws.get_canonical_target_cache([self.wsa.agent_id])
        protgene = set(
                (x[0],x[1])
                for x in targ_cache.info_for_agent(self.wsa.agent_id)
                )
        protgene = sorted(protgene,key=lambda x:x[1])
        user = self.request.user.username
        note_cache = targ_cache.build_note_cache(self.ws,user)
        for prot,gene in protgene:
            note_id,text = note_cache.get(prot,{}).get(user,(None,''))
            self.prot_note_data.append((prot,gene,note_id,text))

    def make_vote_form(self,data):
        from moldata.forms import VoteForm
        return VoteForm(self.request.user,data,instance=self.vote)
    def vote_post_valid(self):
        user = self.request.user.username
        from browse.models import TargetReview
        for uniprot,text in self.protein_notes_form.cleaned_data.items():
            TargetReview.save_note(self.ws,uniprot,user,text)
        p = self.vote_form.cleaned_data
        self.vote.recommended = p['recommended']
        Note.set(self.vote,'note',user,p['note'])
        self.vote.save()
        # this may have changed completion status; make sure
        # note permissions are correct
        self.vote.election.update_note_permissions()
        return HttpResponseRedirect(self.here_url())
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

    def get_binding(self):
        from dtk.prot_map import DpiMapping, AgentTargetCache, protein_link
        self.dpi = self.ws.get_dpi_default()
        self.dpi_thresh = self.ws.get_dpi_thresh_default()
        targ_cache = AgentTargetCache(
                mapping = DpiMapping(self.dpi),
                agent_ids = [self.wsa.agent_id],
                dpi_thresh = self.dpi_thresh,
                )
        note_cache = targ_cache.build_note_cache(
                self.ws,
                self.request.user.username,
                )
        by_key = {}
        protdirs =set()
        for x in targ_cache.full_info_for_agent(self.wsa.agent_id):
            l = by_key.setdefault(x[0],[])
            l.append(x[1:])
            protdirs.add((x[1], x[4]))
        from browse.models import Protein
        uniprots = [x[0] for x in protdirs]
        self.prots = Protein.objects.filter(uniprot__in=uniprots)
        bindings = []
        for native_key in sorted(by_key.keys()):
            # sort by evidence
            l = sorted(by_key[native_key],key=lambda x:x[2],reverse=True)
            bindings.append((native_key,[
                    protein_link(
                            uniprot,
                            gene,
                            self.ws,
                            note_cache={}, # don't show other people's prot notes for consultants
                            direction=direction,
                            consultant_link=True
                            )
                    for uniprot,gene,evidence,direction in l
                    ]))
        self.context['bindings']=bindings

        from rvw.prescreen import PrescreenData
        from browse.models import WsAnnotation
        wsa_ids = PrescreenData.protlu_to_wsas(self.ws, protdirs)
        wsas = WsAnnotation.objects.filter(id__in=wsa_ids)
        self.context['mol_list'] = ', '.join(x.agent.canonical for x in wsas)


class ConsultantProteinView(DumaView):
    template_name='consultant/protein.html'

    def custom_context(self):
        from dtk.url import ext_prot_links
        for cat, cat_links in ext_prot_links(self.ws,self.protein).items():
            self.context[f'{cat}_links'] = cat_links
        self.load_ppi()
        self.load_pathways()

    def load_ppi(self):
        from dtk.prot_map import PpiMapping
        pm = PpiMapping(self.ws.get_ppi_default())
        protlist = pm.get_ppi_info_for_keys([self.protein.uniprot], min_evid=0.9)
        protlist.sort(key=lambda x:x.evidence,reverse=True)
        from dtk.plot import dpi_arrow
        self.context_alias(
                protlist = [
                        [x, dpi_arrow(float(x.direction))]
                        for x in protlist
                        ]
                )
    
    def load_pathways(self):
        from dtk.gene_sets import get_pathways_for_prots, get_pathway_id_name_map, get_prots_for_pathways
        from dtk.url import ext_pathway_url
        from dtk.html import link
        from browse.default_settings import GeneSets
        gs_choice = GeneSets.latest_version()
        prot2pw = get_pathways_for_prots(gs_choice, [self.protein.uniprot]).fwd_map()
        pw2name = get_pathway_id_name_map(gs_choice)
        pathway_ids = prot2pw.get(self.protein.uniprot, [])
        pathway_names = [pw2name.get(id, id) for id in pathway_ids]
        def get_source(x):
            if x.startswith('R-'):
                return 'Reactome'
            if x.startswith('GO:'):
                return 'Gene Ontology'
            return ''

        sources = [get_source(id) for id in pathway_ids]
        pathway_links = [link(id, ext_pathway_url(id)) for id in pathway_ids]

        from dtk.table import Table

        cols = [
            Table.Column("ID", idx=0),
            Table.Column("Name", idx=1),
            Table.Column("Source", idx=2),
        ]
        rows = list(zip(pathway_links, pathway_names, sources))
        table = Table(rows, cols)

        self.context_alias(
            pathways=table
        )

