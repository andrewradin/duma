from django.db import models
from django.db import transaction
from django.utils.safestring import mark_safe

from tools import Enum
from browse.models import WsAnnotation, DrugSet, Protein
import logging
logger = logging.getLogger(__name__)

ScoreTypes = Enum([], [
        ('PRECLINICAL_NOVELTY',),
        ('COMMERCIAL_AVAILABILITY',),
        ('TOLERABILITY',),
        ('INTELLECTUAL_PROPERTY',),
        ('DOSING_ROA','Dosing, RoA'),
        ('PK', 'Pharmacokinetics'),
        ('CHEMISTRY',),
        ('POTENCY',),
        ('SELECTIVITY',),
        ('HITSEL_NOTE',),
        ])


class ScoreTypeData:
    can_autogenerate = False
    must_nonzero = False

    @classmethod
    def autogenerate(cls, ws, wsas):
        return {}

    @classmethod
    def from_scoretype(cls, scoretype):
        ST = ScoreTypes
        lu = {
            ST.PRECLINICAL_NOVELTY: PreclinicalNovelty,
            ST.COMMERCIAL_AVAILABILITY: CommercialAvailability,
            ST.TOLERABILITY: Tolerability,
            ST.INTELLECTUAL_PROPERTY: IntellectualProperty,
            ST.DOSING_ROA: DosingRoa,
            ST.PK: Pharmacokinetics,
            ST.CHEMISTRY: Chemistry,
            ST.POTENCY: Potency,
            ST.SELECTIVITY: Selectivity,
            ST.HITSEL_NOTE: HitSelNote,
        }
        return lu[scoretype]



def disease_molecule_lit_overlap(ws, wsas):
    if not wsas:
        return 0, []

    ws = wsas[0].ws

    from dtk.entrez_utils import PubMedSearch
    pms = PubMedSearch()

    from drugs.models import Drug
    drugs = [wsa.agent_id for wsa in wsas]
    version = ws.get_dpi_version()
    drug_syns = Drug.bulk_synonyms(drugs, version).fwd_map()

    disease_syns = ws.get_disease_default('WebSearch').split('|')

    def good_syn(x):
        # Some heuristics to filter out IUPAC names and other synonyms that
        # could be problematic or overly verbose.
        return x.count('(') < 2 and x.count('[') < 2 and x.count('-') < 4 and x.count('[') == 0 and x.count(',') == 0

    out = {}
    for wsa in wsas:
        # On replacing spaces with hyphens, that seems to be the best way to do things according to
        # https://pubmed.ncbi.nlm.nih.gov/help/#searching-for-a-phrase
        #
        # PubMed doesn't really know how to search for phrases unless the phrase already exists in its phrase
        # index.  By using hyphens, we prevent fallback to automatic term mapping which falls
        # back to searching word-by-word.  word-by-word becomes problematic when searching for synonyms like
        # "C 12354", as it will match any article with a "C" in it.

        mol_syns = [x.replace('"', '').replace(' ', '-') for x in drug_syns[wsa.agent_id] if good_syn(x)]
        mol_q = '"' + '"[Title/Abstract] OR "'.join(sorted(mol_syns)) + '"[Title/Abstract]'
        dis_q = '"' + '" OR "'.join(sorted(disease_syns)) + '"'
        query = f"({mol_q}) AND ({dis_q})"
        res = pms.esearch(query)
        from urllib.parse import quote
        url = f"http://www.ncbi.nlm.nih.gov/pubmed/?term={quote(query)}"
        out[wsa.id] = res.count, res.ids, url
    return out

class PreclinicalNovelty(ScoreTypeData):
    scoring_help = """
1 – many papers
2 – papers from several* labs or different studies
3 – a single paper; only a few* in vitro studies;
4 – no papers
"""
    weight = 0.1
    @classmethod
    def autogenerate(cls, ws, wsas):
        overlap = disease_molecule_lit_overlap(ws, wsas)

        wsa_map = {}

        for wsa_id, (count, ids, url) in overlap.items():
            if count >= 10:
                score = 1
            elif count >= 1:
                score = 2
            else:
                score = 4
            descr = f'{count} PubMed disease+molecule papers'
            from dtk.html import link
            details = [link('PubMed', url, new_tab=True)]
            wsa_map[wsa_id] = (score, descr, details)
        return wsa_map

class CommercialAvailability(ScoreTypeData):
    scoring_help = """0 – Not available from a vendor
1 – available from commercial vendors, but requires synthesis
4 – available off the shelf
"""
    weight = 0.1
    must_nonzero = True
    @classmethod
    def autogenerate(cls, ws, wsas):
        from dtk.comm_avail import wsa_comm_availability
        comm_avails = wsa_comm_availability(ws, wsas)
        wsa_map = {}
        for wsa, comm_avail in zip(wsas, comm_avails):
            if comm_avail.has_commdb:
                wsa_map[wsa.id] = (3.0, 'In commercial database', comm_avail.details)
            elif comm_avail.has_cas:
                wsa_map[wsa.id] = (3.0, 'Has CAS number', comm_avail.details)
            elif comm_avail.has_zinc:
                wsa_map[wsa.id] = (2.0, f'Has ZINC data ({comm_avail.zinc_reason})', comm_avail.details)
        return wsa_map



class Tolerability(ScoreTypeData):
    scoring_help = "Disease-specific, TBD"
    weight = 0.15
    @classmethod
    def get_assay_list(cls, wsa, attr):
        # XXX copied from Pharmacokinetics -- s/b/ in base class?
        from dtk.assays import load_dmpk_assays
        ver = wsa.ws.get_dpi_version()
        return load_dmpk_assays(wsa.agent, attr, ver)

    @classmethod
    def autogenerate(cls, ws, wsas, max_phase_data):
        from dtk.html import link
        wsa_map = {}
        for wsa in wsas:
            data = max_phase_data[wsa.id]
            max_phase = data['overall_max_phase']
            details = []

            if data['trials']:
                name = f'CT Max Phase: {data["trials_max_phase"]}'
                url = ws.reverse('moldata:annotate', wsa.id) + '?show=trials'
                details.append(link(name, url, new_tab=True))

            if data['max_phase_drugid']:
                name = f'ChEMBL Max Phase: {data["max_phase"]}'
                from dtk.url import chembl_drug_url
                from drugs.models import Drug
                url = chembl_drug_url(Drug.objects.get(pk=data['max_phase_drugid']).chembl_id)
                details.append(link(name, url, new_tab=True))

            if data['approved']:
                details.append('Drugbank: Approved')

            score = 0
            if max_phase >= 2:
                score = 2
            elif max_phase >= 1:
                score = 1

            assays, err = cls.get_assay_list(wsa, 'tox')
            if assays:
                details.append(link(
                        f'{len(assays)} tox assays',
                        ws.reverse('moldata:noneff_assays', wsa.id),
                        new_tab=True))
            wsa_map[wsa.id] = (score, f'Max Phase={max_phase}', details)
        return wsa_map


class HitSelNote(ScoreTypeData):
    scoring_help = """
    Overview and/or non-category-specific notes about this molecule for hit selection.
"""
    weight = 0

class IntellectualProperty(ScoreTypeData):
    scoring_help = """
1 – Pharma-sponsored patent naming the molecule and disease specifically
2 – Non-specific or “loose” pharma patents or non-industry patents
3 – only mention of disease and molecule is in a laundry list
4 – no mention of the molecule name or structure and the disease
"""
    weight = 0.1

class DosingRoa(ScoreTypeData):
    scoring_help = """
0 – no dosing data available
1 – only limited in vitro data available
2 – single dose, in vivo data available
3 – Limited oral dosing data over a time course
4 – Several experiments of oral dosing data over a time course
"""
    weight = 0.15
    must_nonzero = True
    @classmethod
    def autogenerate(cls, ws, wsas, max_phase_data):
        from dtk.html import link
        wsa_map = {}
        for wsa in wsas:
            data = max_phase_data[wsa.id]
            max_phase = data['overall_max_phase']
            details = []

            if data['trials']:
                name = f'CT Max Phase: {data["trials_max_phase"]}'
                url = ws.reverse('moldata:annotate', wsa.id) + '?show=trials'
                details.append(link(name, url, new_tab=True))

            if data['max_phase_drugid']:
                name = f'ChEMBL Max Phase: {data["max_phase"]}'
                from dtk.url import chembl_drug_url
                from drugs.models import Drug
                url = chembl_drug_url(Drug.objects.get(pk=data['max_phase_drugid']).chembl_id)
                details.append(link(name, url, new_tab=True))

            if data['approved']:
                details.append('Drugbank: Approved')

            score = 0
            if max_phase >= 2:
                score = 4
            elif max_phase >= 1:
                score = 3

            wsa_map[wsa.id] = (score, f'Max Phase={max_phase}', details)
        return wsa_map

class Pharmacokinetics(ScoreTypeData):
    scoring_help = """
1 – no data available and any predictions are low confidence
2 – no experimental data, but predictions are high confidence
3 – limited PK data, supplemented with predictions
4 – extensive PK data readily available
"""
    weight = 0.2

    @classmethod
    def get_assay_list(cls, wsa, attr):
        from dtk.assays import load_dmpk_assays
        ver = wsa.ws.get_dpi_version()
        return load_dmpk_assays(wsa.agent, attr, ver)

    @classmethod
    def load_props(cls, ws, wsas):
        out = {}
        from drugs.models import Metric, Drug
        ver = ws.get_dpi_version()
        agents = [x.agent_id for x in wsas]
        out['logd7_4'] = Drug.bulk_prop(agents, ver, 'logd7_4', Metric).fwd_map()
        return out

    @classmethod
    def autogenerate(cls, ws, wsas):
        from dtk.assays import interpret_dmpk_assays, DmpkAssay, short_organism
        wsa_map = {}
        props = cls.load_props(ws, wsas)

        def pk_entry(assay, parse):
            rating = '(' + parse.score_rating + ')' if parse.score_rating else ''
            return f'''
            <div class='pk-entry'>
                <div class='organism'>{short_organism(assay.organism)} {rating}</div>
                {parse.std_value:.1f} {parse.std_unit}
            </div>
            '''


        for wsa in wsas:
            assays, err = cls.get_assay_list(wsa, 'pc')
            assays2, err = cls.get_assay_list(wsa, 'adme')
            assays += assays2

            logd = props['logd7_4'].get(wsa.agent_id, None)
            if logd is not None:
                for logd_val in logd:
                    assays += [DmpkAssay('ChEMBL Computed LogD7.4', 'logd7.4', '=', logd_val, 'None', 'None', 'None', 'None')]

            if assays:
                import numpy as np
                categorized = interpret_dmpk_assays(assays)

                reasons = []
                scores = []
                for category, samples in categorized.items():
                    if not category:
                        continue

                    samples = sorted(samples, key=lambda x: x[0].std_value)
                    cat_scores = [x[0].score for x in samples if isinstance(x[0].score, float)]
                    if cat_scores:
                        score = np.mean(cat_scores)
                        scores.append(score)

                    entries = [pk_entry(assay, parse) for parse, assay in samples]
                    reasons.append(mark_safe(f'<b>{category}</b>:<br>{"".join(entries)}'))

                if scores:
                    score = np.mean(scores)
                else:
                    score = 0
                from dtk.html import link
                reasons.append(link(f'{len(assays)} noneff assays', ws.reverse('moldata:noneff_assays', wsa.id), new_tab=True))
                wsa_map[wsa.id] = (score, f'From {len(assays)} pc/adme assays', reasons)
        return wsa_map

class Chemistry(ScoreTypeData):
    scoring_help = "Ease of synthesis"
    weight = 0.05


def prot_potency(assays):
    # Assume all biochem for now
    best = (-1, 0, 'No assays')
    for assay in assays:
        assay_type = assay.assay_type
        if assay_type == 'ki':
            if assay.nm < 100:
                best = max(best, (1, -assay.nm, f'ki={assay.nm}nm'))
            else:
                best = max(best, (0, -assay.nm, f'ki={assay.nm}nm'))
        elif assay_type == 'c50':
            if assay.nm < 10:
                score = 3
            elif assay.nm < 100:
                score = 2
            elif assay.nm < 200:
                score = 1
            else:
                score = 0

            best = max(best, (score, -assay.nm, f'c50={assay.nm}nm'))
        else:
            raise Exception(f"Implement type {assay_type}")

    if best[0] == -1:
        best = (0, 0, 'No assays')
    return (best[0], best[2])


# In order to use a consistent potency scoring method elsewhere,
# where MolSetMoas aren't applicable
# I pulled out the potency scoring
# It makes it a little clunkier for the original use,
# but has the upside of being usable elsewhere for a single WSA
def score_potency(wsa, prots=[]):
    from dtk.data import MultiMap
    aa = wsa.get_assay_info()
    from collections import namedtuple
    RowType = namedtuple('RowType',aa.info_cols)
    rows = [RowType(*x) for x in aa.assay_info()]

    if rows:
        datamm = MultiMap((row.protein, row) for row in rows)

        scores = []
        reasons = []
        if not prots:
            prots = list(datamm.fwd_map().keys())

        from browse.models import Protein
        prot2gene = Protein.get_uniprot_gene_map(prots)

        for prot in prots:
            rows = list(datamm.fwd_map().get(prot, []))
            score, reason = prot_potency(rows)
            reasons.append(f'{prot2gene[prot]}: ({score}) - {reason}')
            scores.append(score)

        import numpy as np
        return np.mean(scores), reasons
    return 0., []

class Potency(ScoreTypeData):
    # TODO: Bring this back once we're distinguishing cell assays.
    cell_scoring = """
1 - <600nM C50 cell. C50; <200nM Biochem C50; <100nM Ki
2 - <200nM C50 cell. C50; <100nM Biochem C50; <10nM Ki
3 - <100nM cell. C50; <10nM biochem. C50
4 - < 10nM cell. C50
"""
    scoring_help = """
1 - <200nM Biochem C50; <100nM Ki
2 - <100nM Biochem C50; <10nM Ki
3 - <10nM Biochem C50
4 - <10nM cell C50
    """
    weight = 0.05
    @classmethod
    def autogenerate(cls, ws, wsas):
        # Find the drugset with all of these, pull out an MoA.
        qs = MolSetMoa.objects
        for wsa in wsas:
            qs = qs.filter(ds__drugs=wsa)
        if len(qs) != 1:
            logger.warning(f'Ambiguous or missing MoA {list(qs)}')
            return {}
        moa = list(qs)[0]

        prots = [prot.uniprot for prot in moa.proteins.all()]

        # For each prot in the MoA, compute a score.
        # Score as the mean of the MoA scores.
        # Report each prot's individual & overall score in details, along with link.
        wsa_map = {}
        for wsa in wsas:
            score,reasons=score_potency(wsa, prots)
            from dtk.html import link
            reasons.append(link('assays', ws.reverse('moldata:assays', wsa.id), new_tab=True))
            wsa_map[wsa.id]=(score, 'From assays', reasons)
        return wsa_map


# see note for score_potency
def score_selectivity(wsa, moaprots=[]):
    from dtk.data import MultiMap
    aa = wsa.get_assay_info()
    from collections import namedtuple
    RowType = namedtuple('RowType',aa.info_cols)
    rows = [RowType(*x) for x in aa.assay_info()]

    final_score = 0
    if rows:
        datamm = MultiMap((row.protein, row) for row in rows)
        scores = []
        reasons = []
        for prot, rows in datamm.fwd_map().items():
            if prot in moaprots:
                continue
            rows = list(rows)
            gene = rows[0].gene
            score, reason = prot_potency(rows)
            reasons.append(f'{gene}: ({score}) - {reason}')
            scores.append(score)

        num_pot = len([x for x in scores if x >= 2])
        num_nonpot = len([x for x in scores if x <= 1])
        num_superpot = len([x for x in scores if x == 4])
        num_tested = num_pot + num_nonpot

# see scoring_help in the Selectivity Class
        if num_nonpot > 10 and num_pot == 0:
            final_score = 4
        elif (num_tested >= 5 and num_pot == 0) or (num_tested > 10 and num_pot <= 2):
            final_score = 3
        elif (num_tested >= 2 and num_pot == 0) or (num_tested >= 5 and num_pot <= 2):
            final_score = 2
        elif (num_tested >= 1 and num_pot == 0) or (num_tested >= 2 and num_pot == 1 and num_superpot == 0):
            final_score = 1

        return (final_score, f'offtarget {num_pot} potent, {num_nonpot} nonpotent', reasons)

    return (final_score, '', [])


class Selectivity(ScoreTypeData):
# note the scoring is now done in score_selectivity
    scoring_help = """
1 – tested against 1 non-targets w/potency <= 1; or 2+ w/ 1 having a potency 2 or 3
2 – tested against 2-5 non-targets all w/potency <= 1; or 5+ w/ 1-2 having a potency >1
3 – tested against 5-10 non-targets all w/potency <= 1; or >10 w/ 1-2 having a potency >1
4 – tested against >10 non-targets all w/potency <= 1
"""
    weight = 0.1

    @classmethod
    def autogenerate(cls, ws, wsas):
        # Find the drugset with all of these, pull out an MoA.
        qs = MolSetMoa.objects
        for wsa in wsas:
            qs = qs.filter(ds__drugs=wsa)

        if len(qs) != 1:
            logger.warning(f'Ambiguous or missing MoA {list(qs)}')
            return {}
        moa = list(qs)[0]

        moaprots = [prot.uniprot for prot in moa.proteins.all()]
        prot2gene = {prot.uniprot: prot.gene for prot in moa.proteins.all()}

        # For each prot in the MoA, compute a score.
        # Score as the mean of the MoA scores.
        # Report each prot's individual & overall score in details, along with link.
        wsa_map = {}
        for wsa in wsas:
            (final_score, txt, reasons) = score_selectivity(wsa, moaprots)
            if reasons:
                from dtk.html import link
                reasons.append(link('assays', ws.reverse('moldata:assays', wsa.id), new_tab=True))
            wsa_map[wsa.id] = (final_score, txt, reasons)
        return wsa_map





class HitScoreValue(models.Model):
    class Meta:
        unique_together=[['wsa', 'scoretype']]
    wsa = models.ForeignKey(WsAnnotation, on_delete=models.CASCADE)
    scoretype = models.IntegerField(choices=ScoreTypes.choices())
    value = models.FloatField(null=True)
    note = models.TextField(blank=True)

    @classmethod
    @transaction.atomic
    def update(cls, wsa, scoretype, value, note, user):
        existing = HitScoreValue.objects.filter(wsa=wsa, scoretype=scoretype)
        if value == '' and note == '' and not existing:
            # This was unset before and continues to be unset
            return

        if value == '':
            value = 0

        if existing and existing[0].value == float(value) and existing[0].note == note:
            # Unchanged.
            return

        from browse.models import WsAnnotation
        if isinstance(wsa, WsAnnotation):
            wsa = wsa.id
        logger.info(f"Updating {wsa} {scoretype} to {value} {note}")
        hsv, new = HitScoreValue.objects.get_or_create(wsa_id=wsa, scoretype=scoretype)
        hsv.value = float(value)
        hsv.note = note
        hsv.save()
        HitScoreValueLog.objects.create(hsv=hsv, user=user, value=value, note=note)

    @classmethod
    def computed_value(cls, wsa, scoretype):
        return None

class HitScoreValueLog(models.Model):
    hsv = models.ForeignKey(HitScoreValue, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now=True)
    user = models.CharField(max_length=50)
    value = models.FloatField()
    note = models.TextField()




from notes.models import Note
class MolSetMoa(models.Model):
    ds = models.OneToOneField(DrugSet, on_delete=models.CASCADE, primary_key=True)
    proteins = models.ManyToManyField(Protein)
    hitsel_note = models.ForeignKey(Note,null=True,blank=True, on_delete=models.CASCADE, related_name='hitsel_note')
    preclin_note = models.ForeignKey(Note,null=True,blank=True, on_delete=models.CASCADE, related_name='preclin_note')
    def note_info(self,attr):
        return {}


class ClinicalTrialAudit(models.Model):
    ct_status_vals = Enum([], [
            ('UNKNOWN',),
            ('FAILED',),
            ('ONGOING',),
            ('PASSED',),
            ])
    wsa = models.ForeignKey(WsAnnotation, on_delete=models.CASCADE)
    ph2_status = models.IntegerField(choices=ct_status_vals.choices())
    ph2_url = models.TextField(blank=True)
    ph3_status = models.IntegerField(choices=ct_status_vals.choices())
    ph3_url = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now=True)
    user = models.CharField(max_length=50)
    def metadata(self):
        from dtk.text import fmt_time
        return f'{self.user}@{fmt_time(self.timestamp,"%Y-%b-%d")}'
    def stat_summary(self):
        data = [
                (phase,getattr(self,f'ph{phase}_status'))
                for phase in (2,3)
                ]
        return [
                f'Ph{ph}: {self.ct_status_vals.get("label",val)}'
                for ph,val in data
                ]
    @classmethod
    def get_latest_wsa_record(cls,wsa_id,user):
        qs = cls.objects.filter(wsa_id=wsa_id).order_by('-id')[:1]
        if qs:
            return qs[0]
        # record doesn't exist; return a mock-up to populate form
        return cls(
                wsa_id=wsa_id,
                ph2_status=cls.ct_status_vals.UNKNOWN,
                ph3_status=cls.ct_status_vals.UNKNOWN,
                user=user,
                )
    @classmethod
    def get_latest_ws_records(cls,ws_id):
        from browse.models import WsAnnotation
        from django.db.models import Max
        # Django aggregation has some limitations, so we do this in two
        # parts, first fetching the ids of the latest row for each
        # wsa in the workspace, then fetching the rows themselves.
        # Luckily, we can use Max(id) as a proxy for Max(timestamp),
        # since the id is actually what we need for the 2nd fetch.
        # Since CTAs are hand-annotated, this will always be small/fast.
        ids_to_fetch = set(WsAnnotation.objects.filter(
                ws_id=ws_id,
                clinicaltrialaudit__id__isnull=False,
                ).annotate(
                Max('clinicaltrialaudit__id')
                ).values_list('clinicaltrialaudit__id__max',flat=True))
        return cls.objects.filter(id__in=ids_to_fetch)
    def check_post_data(self,post):
        for phase in (2,3):
            stat = int(post[f'ph{phase}_status'])
            if stat != self.ct_status_vals.UNKNOWN:
                if not post[f'ph{phase}_url']:
                    raise ValueError(f'A Phase {phase} URL must be provided')
    def update_from_post_data(self,post,user):
        # This holds some shared logic for updating the CTA table from
        # a Django form. 'post' is a dictionary containing the post
        # output for the 4 trial status fields (which are expected to
        # be named as in the list below). 'self' is expected to be a
        # CTA record returned by get_latest_wsa_record, i.e. populated
        # with the current values. If the post data is different from
        # the record data, a new record will be written and attributed
        # to the passed-in username.
        need_cta_update = False
        for field in ['ph2_status','ph2_url','ph3_status','ph3_url']:
            # str needed because form field output is always a string
            if str(getattr(self,field)) != post[field]:
                setattr(self,field,post[field])
                need_cta_update = True
        if need_cta_update:
            # Require a URL for any known status. We do this here so we
            # enforce the rule on any update, but don't trip over legacy
            # data. Normally check_post_data will already have been called
            # and any exception caught to provide a user-friendly reminder,
            # but this is just a backstop.
            self.check_post_data(post)
            # write validated result
            self.pk = None # force new record
            self.user = user
            self.save()
