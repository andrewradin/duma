from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, HttpResponseRedirect
from django import forms
from dtk.duma_view import DumaView,list_of,boolean
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.db import transaction
from django.urls import reverse
import json
import logging
logger = logging.getLogger(__name__)

# See:
# https://ariapharmaceuticals.atlassian.net/wiki/spaces/TD/pages/2727084033/Updating+Duma+Drug+Collection
# for an overview of how this works.

class DrugEditReviewView(DumaView):
    template_name='drugs/review.html'
    button_map = {'publish': []}
    GET_parms = {
            'prop_id':(int,None),
            }
    def custom_context(self):
        from drugs.models import DrugProposal
        qs = DrugProposal.objects.all().order_by('-id')
        values = qs.values('user', 'timestamp', 'ref_drug', 'state', 'id', 'drug_name')
        values = [{
            'user': x['user'],
            'date': x['timestamp'].strftime("%Y-%m-%d %H:%M"),
            'state': DrugProposal.states.get('label', x['state']),
            'drug_name': x['drug_name'],
            'id': x['id'],
            } for x in values]


        from drugs.drug_edit import DumaCollection
        col = DumaCollection()
        ood = col.out_of_date_info()

        self.context_alias(
                proposals=json.dumps(values),
                preselect=self.prop_id,
                **ood,
                )

    def publish_post_valid(self):
        from drugs.drug_edit import DumaCollection
        col = DumaCollection()
        col.publish()


def make_dpi_data(drug):
    from dtk.prot_map import DpiMapping
    from browse.default_settings import DpiDataset
    dpi_choice = DpiDataset.value(ws=None)
    dpi = DpiMapping(dpi_choice)
    dpi_info = dpi.get_dpi_info(drug) if drug else []
    from browse.models import Protein

    uniprots = [x.uniprot_id for x in dpi_info]
    gene_map = Protein.get_uniprot_gene_map(uniprots)

    from collections import defaultdict
    prot2type2value = defaultdict(dict)
    if drug:
        aa = drug.get_assay_info(version=None if dpi.legacy else dpi.version)
        prot_idx = aa.info_cols.index('protein')
        type_idx = aa.info_cols.index('assay_type')
        value_idx = aa.info_cols.index('nm')
        for info in aa.assay_info():
            prot, type, value = info[prot_idx], info[type_idx], info[value_idx]
            prot2type2value[prot][type] = value

    rows = [
        {
            'source': '%s (%s)' % (x[0], dpi_choice),
            'uniprot': x.uniprot_id,
            'gene': gene_map.get(x.uniprot_id, ''),
            'evidence': x.evidence,
            'direction': x.direction,
            'c50': prot2type2value[x.uniprot_id].get('c50', ''),
            'ki': prot2type2value[x.uniprot_id].get('ki', ''),
        } for x in dpi_info
        ]

    return rows

def make_attrs_data(drug):
    # We only want to be able to specify things the platform actually uses.
    # Anything else can be looked up easily via cross matching page.
    keys = [
            ('canonical', False),
            ('synonym', True),
            ('cas', True),
            ('atc', True),
            ('smiles_code', False),
            ('inchi', False),
            ]

    if drug is None:
        rows=[
                {
                    'name': x[0],
                    'value': [] if x[1] else '',
                    'other': [],
                    'source': '',
                }
                for x in keys
                ]
        return rows

    from drugs.models import Drug, Prop, Collection

    from browse.default_settings import DpiDataset
    from dtk.prot_map import DpiMapping
    version = DpiMapping(DpiDataset.value(ws=None)).version

    drug_ids = Drug.matched_id_mm([drug.id], version=version).fwd_map()[drug.id]
    # build list of (keyspace,prop_name,value) for all properties of
    # all matched agents
    exclude_props = [
            'synthesis_protection',
            Prop.OVERRIDE_NAME,
            ]
    attrs=[]
    for prop_type in [x[0] for x in Prop.prop_types.choices()]:
        cls = Prop.cls_from_type(prop_type)
        attrs+=[
                (x[0].split('.')[0],x[1],x[2],x[3])
                for x in cls.objects.filter(
                        drug_id__in=drug_ids
                        ).values_list(
                        'drug__collection__name','prop__name','value','prop__multival',
                        )
                if x[1] not in exclude_props and not
                        x[1].startswith(Collection.foreign_key_prefix)
                ]
    # reformat into MultiMap to group by (prop,value)
    from dtk.data import MultiMap
    mm = MultiMap(
            ((prop,multival),(value,keyspace))
            for keyspace,prop,value,multival in attrs
            )

    def get_value(fullkey):
        key, multival = fullkey
        if multival:
            return list(getattr(drug, key + "_set"))
        else:
            out = getattr(drug, key)
            if out is None:
                return ''
            else:
                return out

    drug_key = drug.get_key()
    rows=[
            {
                'name': x[0],
                'value':get_value(x),
                'other': sorted(mm.fwd_map().get(x, [])),
                'source': drug_key,
            }
            for x in keys
            ]
    return rows


class DrugChangesView(DumaView):
    template_name='drugs/changes.html'
    GET_parms = {
            'startver':(int,None),
            'endver':(int,None),
            }
    button_map = {
            'diff': ['versions'],
            'prop': [],
            }

    def get_diff_data(self):
        def compute():
            from scripts.duma_version_audit import diff_drug
            from drugs.models import Drug
            duma_drugs = Drug.objects.filter(collection__name='duma.full')
            out = []
            from browse.models import Protein
            prot2gene = Protein.get_uniprot_gene_map()
            for drug in duma_drugs:
                diff = diff_drug(drug, self.startver, self.endver, prot2gene)
                # We store drug.id because it can be weird caching models if they
                # change.
                out.append((drug.id, diff))
            return out
        from dtk.cache import Cacher
        c = Cacher('audit_mol_changes.v1')
        key = f'{self.startver},{self.endver}'
        return c.check(key, compute)


    def prop_post_valid(self):
        import json
        from browse.models import Protein
        from drugs.models import Drug, DrugProposal
        query_str = self.request.POST['query']
        query = json.loads(query_str)

        # Load up the corresponding proposal
        drug = Drug.objects.get(pk=query['drug_id'])
        orig_proposal = DrugProposal.objects.get(pk=drug.duma_proposal_id)
        # confirm that the thing we're working on is aleady a duma drug
        # with an assigned duma drug id
        assert orig_proposal.collection_drug_id
        orig_prop_data = json.loads(orig_proposal.data)
        attrs_data, dpi_data = orig_prop_data['attrs'], orig_prop_data['dpi']

        orig_entry_by_prot = {row['uniprot']:row for row in dpi_data}
        orig_order_by_prot = {row['uniprot']:i for i, row in enumerate(dpi_data)}

        new_dpi = []


        # Generate new DPI data from the selected lines.
        # There should be one entry for each prot that we kept or added.
        selected = query['selected']
        from .drug_edit import get_computed_evidence
        for entry in selected:
            prot, evid, direction, sources = entry.split('|')

            orig_entry = orig_entry_by_prot.get(prot, None)
            orig_ev = get_computed_evidence(orig_entry['evidence'], orig_entry['c50'], orig_entry['ki']) if orig_entry else None
            if orig_ev and orig_ev == float(evid) and orig_entry['direction'] == direction:
                # Didn't change, just keep the original.
                new_dpi.append(orig_entry)
            else:
                isNew = not orig_entry

                gene_map = Protein.get_uniprot_gene_map([prot])
                # Make a new entry.
                new_entry = {
                    'source': f'{sources} (v{self.endver})',
                    'uniprot': prot,
                    'gene': gene_map.get(prot, ''),
                    'evidence': evid,
                    'direction': direction,
                    'c50': '',
                    'ki': '', # These can be pulled from original source as needed.
                    'isNew': isNew,
                    'isChanged': not isNew,
                    'keep': True
                        }
                new_dpi.append(new_entry)

        # Generate keep=False entries for unselected, just for reference.
        unselected = query['unselected']
        for entry in unselected:
            prot, evid, direction, sources = entry.split('|')

            orig_entry = orig_entry_by_prot.get(prot, None)
            gene_map = Protein.get_uniprot_gene_map([prot])
            # Make a new entry.
            new_entry = {
                'source': f'{sources} (v{self.endver})',
                'uniprot': prot,
                'gene': gene_map.get(prot, ''),
                'evidence': evid,
                'direction': direction,
                'c50': '',
                'ki': '',
                'keep': False,
                    }
            new_dpi.append(new_entry)

        new_dpi.sort(key=lambda x: (orig_order_by_prot.get(x['uniprot'], 1e99), x['gene'], x['evidence']))


        new_prop_data = {'attrs': attrs_data, 'dpi': new_dpi}

        drug_proposal = DrugProposal.objects.create(
                data=json.dumps(new_prop_data),
                user=self.username(),
                ref_drug_id=drug.id,
                ref_proposal_id=orig_proposal.id,
                collection_drug_id=orig_proposal.collection_drug_id,
                drug_name=drug.canonical,
                )


    def diff_post_valid(self):
        p = self.context['versions_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))


    def custom_context(self):
        if not self.startver or not self.endver:
            return

        if self.startver >= self.endver:
            self.message("End Version must be larger than Start Version")
            return

        diff_data = self.get_diff_data()
        from drugs.models import Drug
        drugtables = []
        for drug_id, entry in diff_data:
            drug = Drug.objects.get(pk=drug_id)
            drugdata = {
                'name': drug.canonical,
                'prop_url': f"{reverse('drug_edit_view')}?prop_id={drug.duma_proposal_id}",
                'wsa_url': drug.any_wsa_url(),
                'drug_id': drug.id,
            }
            drugtables.append((drugdata, entry))


        from dtk.drug_clusters import RebuiltCluster
        start_clust = RebuiltCluster(version=self.startver).match_inputs
        end_clust = RebuiltCluster(version=self.endver).match_inputs
        all_keys = start_clust.keys() | end_clust.keys()
        versions = {
                key: [start_clust.get(key, ''), end_clust.get(key, '')]
                for key in all_keys
                }

        self.context_alias(
                drugtables=drugtables,
                versions=versions,
                )


    def make_versions_form(self,data):
        from browse.default_settings import DpiDataset
        from dtk.prot_map import DpiMapping
        from django import forms
        choices = DpiMapping.dpi_names()
        dpis = [DpiMapping(x) for x in choices]
        choices = [(dpi.version, dpi.choice) for dpi in dpis if dpi.version and 'DNChBX' in dpi.choice and dpi.get_baseline_dpi() == dpi]

        class MyForm(forms.Form):
            startver = forms.ChoiceField(
                    label='Start Version',
                    choices=choices,
                    required=True,
                    initial=self.startver,
                    )
            endver = forms.ChoiceField(
                    label='End Version',
                    choices=choices,
                    required=True,
                    initial=self.endver,
                    )

        return MyForm(data)

class DrugEditView(DumaView):
    template_name='drugs/edit.html'
    GET_parms = {
            'wsa_id':(int,None),
            'prop_id':(int, None),
            }
    button_map = {'propose': []}
    def custom_context(self):
        self.drug = None
        self.ref_proposal = None
        self.best_duma_key = None
        # note that in a dev environment, if you've uploaded match files
        # that are created after your database (and so may contain different
        # duma ids than those in your DrugProposal table), chaos can ensue.
        # So we provide an override here for development.
        from path_helper import PathHelper
        from drugs.models import DpiMergeKey
        self.cluster_version = PathHelper.cfg(
                'drug_edit_cluster_version'
                ) or DpiMergeKey.max_version()

        # There will either be:
        # - a wsa_id, if we're coming from the drug page,
        # - a prop_id, if we're coming from the review edits page,
        # - neither, if we're coming from the New Drug Proposal link
        # There will never be both a wsa_id and prop_id.
        #
        # On the output side:
        # - a new drug will have neither a ref_drug or a ref_proposal,
        #   and its collection_drug_id will always be built from the
        #   proposal record id (so, it must have done a 2-part write).
        # - a newly-modified drug (first drug proposal for that agent)
        #   will have a ref_drug but not a ref_proposal
        # In either case:
        # - a subsequent modification via the proposal page will have
        #   ref_proposal set to the proposal being modified, and
        #   ref_drug copied from that proposal.
        # - a subsequent modification via the annotate page will have
        #   ref_drug set to the agent for that annotate page, and
        #   ref_proposal pointing at the last proposal with ref_drug
        #   pointing at that annotate page's agent (if any)
        #
        # If duma drugs cluster together, this code should always set
        # the collection_drug_id to the latest duma drug in the cluster.
        #
        # If the edit originates from an annotate page, and there are
        # any proposals in the cluster, the latest will be used as the
        # baseline.
        #
        # Any proposal can be selected as the baseline from the review
        # page, but the user will be warned if the baseline isn't the
        # latest, or if multiple proposals are pending.

        if self.wsa_id:
            # In this case, the best proposal to use is the latest one
            # for any drug in the cluster
            from browse.models import WsAnnotation
            self.drug = WsAnnotation.objects.get(pk=self.wsa_id).agent
            rbc = self.drug.get_molecule_matches(self.cluster_version)
            self.best_duma_key = rbc.best_duma_key()
            if self.best_duma_key:
                self.ref_proposal = self.duma_key_to_prop(self.best_duma_key)
        elif self.prop_id:
            # in this case, use data from the selected prop, but use the
            # priority DUMA id for the cluster, so dpi_merge selects this
            # proposal
            from drugs.models import DrugProposal
            self.ref_proposal = DrugProposal.objects.get(pk=self.prop_id)
            self.best_duma_key = self.prop_to_duma_key(self.ref_proposal)

        # Now we've got ref_proposal and best_duma_key; set up data for
        # drug_edits.js
        import json
        note = ''

        if self.ref_proposal:
            prop = self.ref_proposal
            self.drug = prop.ref_drug # might be None
            prop_data = json.loads(prop.data)
            attrs_data, dpi_data = prop_data['attrs'], prop_data['dpi']
            note = prop_data.get('note', '')
        else:
            attrs_data = make_attrs_data(self.drug)
            dpi_data = make_dpi_data(self.drug)

        self.context_alias(
                ref_drug=self.drug,
                ref_drug_id=self.drug.id if self.drug else 'null',
                ref_proposal_id=self.ref_proposal.id if self.ref_proposal else 'null',
                best_duma_key=f'"{self.best_duma_key or str()}"',
                attr_json=json.dumps(attrs_data),
                dpi_json=json.dumps(dpi_data),
                note=note,
                conversion_info_help=self._make_conversion_info_help,
                )
    
    def duma_key_to_prop(self,duma_key):
        accepted = []
        proposed = []
        from drugs.models import DrugProposal
        for dp in DrugProposal.objects.filter(collection_drug_id=duma_key):
            if dp.state == DrugProposal.states.ACCEPTED:
                accepted.append(dp)
            elif dp.state == DrugProposal.states.PROPOSED:
                proposed.append(dp)
        if proposed:
            # if there are already active proposals, return the latest;
            # warn about multiples
            if len(proposed) > 1:
                self.message(
                        'WARNING: multiple proposals pending for this drug',
                        )
            return sorted(proposed,key=lambda x:x.id)[-1]
        assert len(accepted) == 1
        return accepted[0]
    def prop_to_duma_key(self,prop):
        from dtk.drug_clusters import RebuiltCluster
        if prop.ref_drug:
            rbc = prop.ref_drug.get_molecule_matches(self.cluster_version)
        else:
            rbc = RebuiltCluster(
                    base_key=('duma_id',prop.collection_drug_id),
                    version=self.cluster_version,
                    )
        best_key = rbc.best_duma_key()
        # cross-check
        best_prop = self.duma_key_to_prop(best_key)
        if best_prop and prop != best_prop:
            from dtk.html import link,join
            self.message(join(
                'WARNING: you are not editing the ',
                link('latest proposal',self.here_url(prop_id=best_prop.id)),
                ))
        return best_key
    def _make_conversion_info_help(self):
        from dtk.affinity import C50_CONV, KI_CONV
        from dtk.html import glyph_icon

        help = f'''If evidence values are provided, they will be used directly (typical values are 0.5, 0.7, 0.9, or 1.0)
        Any evidence lower than 0.5 is largely unused.
        If no explicit evidence is provided, we will apply automatic conversion from C50 or Ki values to evidence.
        C50: <{C50_CONV['for_hi']}nM maps to {C50_CONV['hi_evid']} evid, otherwise <{C50_CONV['for_lo']}nM to {C50_CONV['lo_evid']} evid
        Ki:  <{KI_CONV['for_hi']}nM maps to {KI_CONV['hi_evid']} evid.
        '''

        return glyph_icon(
            'info-sign',
            html=True,
            hover=help,
        )

    def propose_post_valid(self):
        # NOTE that the javascript does this postback without the
        # original query parms, so we can't use custom_setup as
        # a common place to pre-calculate needed values. Instead,
        # they all get passed in json and re-extracted here.
        import json
        from drugs.models import DrugProposal
        query_str = self.request.POST['query']
        query = json.loads(query_str)
        drugData = query['newDrugData']
        bestDumaKey = query['bestDumaKey']

        drugName = ''
        for attr in drugData['attrs']:
            if attr['name'] == 'canonical':
                drugName = attr['value']

        drug_proposal = DrugProposal.objects.create(
                data=json.dumps(drugData),
                user=self.username(),
                ref_drug_id=query['refDrugId'],
                ref_proposal_id=query['refProposalId'],
                drug_name=drugName,
                collection_drug_id=bestDumaKey or '',
                )
        if not bestDumaKey:
            # this is a new duma drug; construct a key to match record id
            drug_proposal.collection_drug_id = 'DUMA%05d' % drug_proposal.id
            drug_proposal.save()


@login_required
def proposal_data(request, proposal_id):
    from drugs.models import DrugProposal
    dp = DrugProposal.objects.get(pk=proposal_id)

    ref_drug = dp.ref_drug
    ref_prop = dp.ref_proposal
    import json
    if ref_prop:
        if ref_drug:
            ref_url = ref_drug.any_wsa_url()
        else:
            ref_url = None

        ref_data = json.loads(ref_prop.data)['dpi']
    elif ref_drug:
        ref_url = ref_drug.any_wsa_url()
        ref_data = make_dpi_data(ref_drug)
    else:
        ref_data = []
        ref_url = None
    
    drug_prop_data = json.loads(dp.data)

    from drugs.drug_edit import validate_smiles
    try:
        smiles = [x for x in drug_prop_data['attrs'] if x['name'] == 'smiles_code']
        if smiles and smiles[0]['value']:
            validation = validate_smiles(smiles[0]['value'])
        else:
            validation = {}
    except Exception as e:
        validation = {'error': str(e)}

    return JsonResponse({
        'newDrug': drug_prop_data,
        'refDrug': ref_data,
        'refUrl': ref_url,
        'validation': validation,
        })


@require_POST
@login_required
@transaction.atomic()
def resolve_proposal(request, proposal_id, resolution):
    if request.user.groups.filter(name='button_pushers').count() == 0:
        raise Exception("Requires 'button_pushers' group")
    from .models import DrugProposal
    resolution = int(resolution)
    dp = DrugProposal.objects.get(pk=proposal_id)
    dp.state = resolution
    if resolution == DrugProposal.states.ACCEPTED:
        for related_prop in dp.related_proposals():
            if related_prop.state == DrugProposal.states.ACCEPTED:
                related_prop.state = DrugProposal.states.OUT_OF_DATE
                related_prop.save()

    dp.save()
    return JsonResponse({'newStatus': dp.state_text})

@login_required
def twoxar_attrs(request):
    from .drug_edit import collection_attrs
    attrs = collection_attrs()
    return HttpResponse('\n'.join('\t'.join(x) for x in attrs),
                        content_type='text/plain')

@login_required
def twoxar_dpi(request):
    from .drug_edit import collection_dpi
    attrs = collection_dpi()
    return HttpResponse('\n'.join('\t'.join(x) for x in attrs),
                        content_type='text/plain')


class DrugSearchView(DumaView):
    template_name='drugs/search.html'
    button_map = {'search': ['search']}
    GET_parms = {
            'ws_id':(str,''),
            'pattern':(str,''),
            'search_blobs':(boolean,True),
            'pattern_anywhere':(boolean,False),
            'page':(int,0),
            }

    def make_search_form(self, data):
        class MyForm(forms.Form):
            ws_id = forms.IntegerField(
                    label='Workspace',
                    help_text="Workspace ID",
                    required=True,
                    initial=self.ws_id,
                    )
            pattern = forms.CharField(
                    label='Search',
                    required=True,
                    initial=self.pattern,
                    help_text="Search terms"
                    )
            search_blobs = forms.BooleanField(
                        label='Search all properties',
                        required=False,
                        initial=self.search_blobs,
                        help_text="Include 'long' properties (synonyms, smiles, inchi)",
                    )
            pattern_anywhere = forms.BooleanField(
                        label='Search non-prefixes (slow)',
                        required=False,
                        initial=self.pattern_anywhere,
                        help_text="Looks for your search term anywhere in the name, not just the start"
                    )
        return MyForm(data)
    @classmethod
    def make_search_table(cls, search_results, ws_id,
            show_import=False,
            show_collection=False,
            un_moa=True, # The only places I see this used when would this behavior, but I want to preserve the option to not
            ):
        """If using show_import, be sure to include the javascript to make the button work."""
        from browse.models import WsAnnotation, Workspace, Protein
        from drugs.models import DpiMergeKey, Drug
        ws = Workspace.objects.get(pk=ws_id)
        version = DpiMergeKey.max_version()
        dpimergekeys = [x[0] for x in search_results if x[0]]

        wsas = WsAnnotation.all_objects.filter(
                ws=ws_id,
                agent__dpimergekey__version=version,
                agent__dpimergekey__dpimerge_key__in=dpimergekeys)

        wsas_by_key = {key: wsa for key, wsa in zip(wsas.values_list('agent__dpimergekey__dpimerge_key', flat=True), wsas)}


        from dtk.prot_map import DpiMapping, AgentTargetCache, protein_link
        from browse.default_settings import DpiDataset, DpiThreshold
        if un_moa:
            from dtk.moa import un_moa_dpi_variant
            dpi_choice = un_moa_dpi_variant(DpiDataset.value(ws=ws_id))
        else:
            dpi_choice = DpiDataset.value(ws=ws_id)
        dpi_thresh = DpiThreshold.value(ws=ws_id)
        dpi = DpiMapping(dpi_choice)
        from dtk.data import MultiMap
        dpi_entries = dpi.get_dpi_info_for_keys(dpimergekeys, min_evid=dpi_thresh)
        key2dpi = MultiMap((x[0], x) for x in dpi_entries).fwd_map()
        prots = {x[1] for x in dpi_entries}
        u2g = Protein.get_uniprot_gene_map(prots)

        search_results = [x + (wsas_by_key.get(x[0]),) for x in search_results]

        from django.utils.safestring import mark_safe
        from dtk.html import link,tie_icon,join,glyph_icon,popover
        import traceback as tb
        def targets_col(entry):
            key = entry[0]
            return join(*[ protein_link(
                            uniprot,
                            u2g.get(uniprot, uniprot),
                            ws_id,
                            direction=direction,
                            )
                    for _,uniprot,evid,direction in key2dpi.get(key, [])
                    ])
        def mol_col(entry):
            key = entry[0]
            wsa = entry[-1]
            if wsa:
                url = link(wsa.agent.canonical, wsa.drug_url())
            else:
                drug_id = entry[3]
                name = Drug.objects.get(pk=drug_id).canonical
                url = f'{name} ({key})'
                if show_import:
                    btn = f'<button type="button" name="agent_import" agent={drug_id} class="btn btn-sm btn-default">Import</button>'
                    url = mark_safe(f'{url} {btn}')
            return url

        def collect_col(entry):
            drug_id = entry[3]
            return Drug.objects.get(pk=drug_id).collection.name
        def matched_col(entry):
            return mark_safe(f'{entry[2]}&nbsp;<span class="label label-info customlabel">{entry[1]}</span>')


        from dtk.table import Table
        columns = [
                Table.Column('Molecule', extract=mol_col),
                Table.Column('Matched', extract=matched_col),
                Table.Column('Targets', extract=targets_col),
                ]
        if show_collection:
            columns.append(
                    Table.Column('Collection', extract=collect_col)
                    )

        table = Table(search_results, columns=columns)
        return table

    def custom_context(self):
        if not self.ws_id:
            self.ws_id = None
        else:
            from browse.models import Workspace
            self.ws = Workspace.objects.get(pk=self.ws_id)
        from browse.utils import drug_search
        if self.pattern:
            search_results = drug_search(
                    version=None, # Latest
                    pattern=self.pattern,
                    pattern_anywhere=self.pattern_anywhere,
                    )

            from dtk.table import Pager, Table
            pager = Pager(
                    self.here_url,
                    len(search_results),
                    page_size=50,
                    page=self.page,
                    )
            search_results = search_results[pager.page_start:pager.page_end]

            table = self.make_search_table(search_results, self.ws_id, show_import=True)

            self.context_alias(
                    pager=pager,
                    table=table,
                    searched=self.pattern,
                    )

    def search_post_valid(self):
        p = self.context['search_form'].cleaned_data
        return HttpResponseRedirect(self.here_url(**p))

def get_default_version():
    from browse.default_settings import DpiDataset
    choice = DpiDataset.value(ws=None)
    from dtk.prot_map import DpiMapping
    return DpiMapping(choice).version

@login_required
def mol_chem_image(request, drug_id):

    from drugs.models import Drug
    drug = Drug.objects.get(pk=drug_id)

    smiles = drug.smiles_code
    from browse.views import is_demo
    if is_demo(request.user):
        smiles = 'OCC1OC(O)(CO)C(O)C1O' # substitute chemical structure
    if not smiles:
        version = get_default_version()
        drug_ids = Drug.matched_id_mm([drug.id], version=version).fwd_map()[drug.id]
        for drug_id in drug_ids:
            # If we're going back to the cluster to get the smiles code, we
            # will use the standardized one.
            smiles = Drug.objects.get(pk=drug_id).std_smiles
            if smiles:
                break

    if not smiles:
        logger.debug("No smiles codes found for %s", drug_id)
        # No smiles code.
        blank_svg = '<svg xmlns="http://www.w3.org/2000/svg"/>'
        return HttpResponse(blank_svg, content_type="image/svg+xml")

    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor, AllChem

    mol = Chem.MolFromSmiles(smiles)

    core_bonds = []
    bond_colors = {}
    atom_colors = {}
    sym_to_color = {
            'C': (0.3, 1.0, 0.7),
            'N': (0.8, 0.8, 1),
            'F': (0.8, 1, 1),
            'S': (1, 1, 0.8),
            'O': (1.0, 0.8, 0.8),
            }
    if request.GET.get('core', False):
        from rdkit.Chem.Scaffolds import MurckoScaffold
        core = MurckoScaffold.GetScaffoldForMol(mol)
        match = list(mol.GetSubstructMatch(core))
        for atm in match:
            sym = mol.GetAtomWithIdx(atm).GetSymbol()
            if sym != 'C':
                atom_colors[atm] = sym_to_color.get(sym, (.5, 0.5, 0.5))
        for bond in core.GetBonds():
            a1 = match[bond.GetBeginAtomIdx()]
            a2 = match[bond.GetEndAtomIdx()]
            mbond = mol.GetBondBetweenAtoms(a1, a2).GetIdx()
            core_bonds.append(mbond)
            bond_colors[mbond] = (0.6, 1.0, 0.6)


    rdDepictor.Compute2DCoords(mol)

    align = request.GET.get('align', None)
    if align:
        try:
            align_to = Chem.MolFromSmiles(align)
            AllChem.Compute2DCoords(align_to)
            AllChem.GenerateDepictionMatching2DStructure(mol, align_to)
        except ValueError:
            # Couldn't find a substructure match between the molecule the
            # requested alignment molecule.  Oh well, ignore.
            pass

    width = int(request.GET.get('width', 600))
    height = int(request.GET.get('height', 600))

    # Apparently this is new, poorly document drawing code.
    # Works better than Chem.MolToImage
    drawer = Draw.MolDraw2DSVG(width, height)

    from rdkit.Chem import rdDepictor
    rdDepictor.SetPreferCoordGen(True)

    drawer.drawOptions().minFontSize = 18

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol,
            highlightBonds=core_bonds,
            highlightBondColors=bond_colors,
            highlightAtoms=atom_colors.keys(),
            highlightAtomColors=atom_colors,
            )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:','')

    return HttpResponse(svg, content_type="image/svg+xml")

@login_required
@require_POST
def import_molecule(request):
    agent_id = request.POST.get('agent_id')
    ws_id = request.POST.get('ws_id')

    from browse.models import Workspace
    Workspace.objects.get(pk=ws_id).import_single_molecule(agent_id, request.user.username)


    return HttpResponse("OK", content_type='text/plain')
