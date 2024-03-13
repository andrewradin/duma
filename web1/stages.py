from __future__ import print_function
from collections import OrderedDict
from tools import linecount, percent
from path_helper import PathHelper
from browse.models import Tissue,Sample,WsAnnotation
from django.db.models import Max
import csv
import logging

logger = logging.getLogger(__name__)

def get_jobs_by_name(name):
    from runner.models import Process
    return Process.objects.filter(
            name=name,
            status=Process.status_vals.SUCCEEDED,
            )

class WorkflowStage(object):
    @staticmethod
    def get_or_create_obj(ws, stage_name):
        from browse.models import StageStatus
        # Explicitly iterating here is faster than filtering if you've
        # prefetched (which you should if you're calling this on lots of ws).
        for obj in ws.stagestatus_set.all():
            if obj.stage_name == stage_name:
                return obj
        obj, is_new = StageStatus.objects.get_or_create(
                ws=ws, stage_name=stage_name)
        return obj


    def __init__(self,ws,name=None,link="workflow"):
        self._name = name if name else self.id()
        self._ws = ws
        if isinstance(link, tuple):
            self._href = ws.reverse(*link)
        else:
            self._href = ws.reverse(link) if link else ''
        self._quality_detail = OrderedDict()
        self._loaded = False
        self._status_obj = None
    def status_obj(self):
        if self._status_obj is None:
            self._status_obj = self.get_or_create_obj(self._ws, self.id())
        return self._status_obj
    def id(self):
        return self.__class__.__name__
    def name(self):
        return self._name
    def href(self):
        return self._href
    def quality_detail(self):
        if not self._loaded:
            self._loaded = True
            import time
            start = time.time()
            self.load_quality()
            end = time.time()
            # Can uncomment this to diagnose why workflow page loads slowly.
            # logger.info(f'{self.name()} took {(end-start)*1000:.2f}ms')
        return self._quality_detail
    def load_quality(self):
        pass

    def completion(self):
        status = self.status_obj().status
        if self.complete(status) or self.verified(status):
            return 1.0
        elif self.active(status):
            return 0.5
        else:
            return 0

    def is_complete(self):
        status = self.status_obj().status
        if self.complete(status) or self.verified(status):
            return True
        return False


    @classmethod
    def button_classes_for_status(cls, status):
        if cls.complete(status):
            return 'btn-success'
        elif cls.verified(status):
            return 'btn-primary'
        elif cls.active(status):
            return 'btn-warning'
        else:
            return 'btn-default'

    def button_classes(self):
        status = self.status_obj().status
        return self.button_classes_for_status(status)

    @classmethod
    def status_text_for_status(cls, status):
        if cls.verified(status):
            return 'Verified'
        elif cls.complete(status):
            return 'Complete'
        elif cls.active(status):
            return 'Active'
        else:
            return ''

    def status_text(self):
        status = self.status_obj().status
        return self.status_text_for_status(status)

    @classmethod
    def verified(cls, status):
        from browse.models import StageStatus
        statuses = StageStatus.statuses
        return status == statuses.VERIFIED

    @classmethod
    def complete(cls, status):
        from browse.models import StageStatus
        statuses = StageStatus.statuses
        return status == statuses.COMPLETE

    @classmethod
    def active(cls, status):
        from browse.models import StageStatus
        statuses = StageStatus.statuses
        return status == statuses.ACTIVE


class DrugCollections(WorkflowStage):
    def __init__(self,ws,quick=False):
        super(DrugCollections, self).__init__(ws, "Molecule Collections",'nav_col2')
        self.quick = quick
    def weight(self):
        return 0.1
    def load_quality(self):
        from drugs.models import Prop
        self._quality = 1
        self._quality_detail = OrderedDict()
        ws_qs = WsAnnotation.objects.filter(ws=self._ws,agent__removed=False)
        self._quality_detail['Molecules'] = ws_qs.count()
        if self.quick:
            # only show mapping counts for workspace default dpi mapping
            ns_list = [self._ws.get_dpi_default().split('.')[0]]
        else:
            # show all mapping counts
            from dtk.prot_map import DpiMapping
            ns_list = sorted(set([
                    x[0].split('.')[0]
                    for x in DpiMapping.choices(self._ws)
                    ]))
        if False:
            for ns in ns_list:
                self._quality_detail['D2P '+ns] = ws_qs.filter(
                            agent__tag__prop__in=[
                                            Prop.get(ns+'_id'),
                                            Prop.get('m_'+ns+'_id'),
                                            ]
                            ).distinct().count()

class IndicationsStage(WorkflowStage):
    def __init__(self,ws):
        super(IndicationsStage, self).__init__(ws, "Investigated Treatments","kts_search")
        self._ws = ws
    def weight(self):
        return 1
    def load_quality(self):
        eval_len = len(self._ws.get_wsa_id_set(self._ws.eval_drugset))
        self._quality_detail['Eval Drugset'] = "%s (n=%d)" % (self._ws.eval_drugset, eval_len)

        # Compute Missing DPI
        ws_qs = WsAnnotation.objects.filter(ws=self._ws, indication__gt=0)

        # If it's an moa-based drugset, we still want to list missing DPI for the underlying molecules.
        treat_wsas = self._ws.get_wsa_id_set(self._ws.eval_drugset.replace('moa-', ''))
        treat_qs = WsAnnotation.objects.filter(id__in=treat_wsas)
        from dtk.prot_map import DpiMapping, AgentTargetCache
        dpi = DpiMapping(self._ws.get_dpi_default()).get_baseline_dpi()
        
        atc = AgentTargetCache.atc_for_wsas(treat_qs, ws=self._ws, dpi_mapping=dpi)
        missing_dpi = 0
        for agent in treat_qs.values_list('agent_id', flat=True):
            if not atc.info_for_agent(agent):
                missing_dpi += 1
        
        missing_dpi_msg = f'{missing_dpi} / {len(treat_qs)}'
        if missing_dpi > 0:
            missing_dpi_msg = f"<b style='color:#a00'>{missing_dpi_msg}</b>"
        from django.utils.html import mark_safe
        self._quality_detail['-- Missing DPI'] = mark_safe(missing_dpi_msg)

        # pre-fetch counts grouped by indication in a single query
        from django.db.models import Count
        lookup = {d['indication']:d['count']
                for d in ws_qs.values('indication').annotate(count=Count('*'))
                }
        # now report out subsets
        enum=WsAnnotation.indication_vals
        entries = [
                ('Known Treatments',[
                        enum.KNOWN_TREATMENT,
                        enum.FDA_TREATMENT,
                        ]),
                ('Clinically Investigated Treatments',[
                        enum.TRIALED_TREATMENT,
                        enum.TRIALED1_TREATMENT,
                        enum.TRIALED2_TREATMENT,
                        enum.TRIALED3_TREATMENT,
                        ]),
                ('Phase 3+',[
                        enum.KNOWN_TREATMENT,
                        enum.FDA_TREATMENT,
                        enum.TRIALED3_TREATMENT,
                        ]),
                ('Phase 2+',[
                        enum.KNOWN_TREATMENT,
                        enum.FDA_TREATMENT,
                        enum.TRIALED3_TREATMENT,
                        enum.TRIALED2_TREATMENT,
                        ]),
                ('Phase 1+',[
                        enum.KNOWN_TREATMENT,
                        enum.FDA_TREATMENT,
                        enum.TRIALED3_TREATMENT,
                        enum.TRIALED2_TREATMENT,
                        enum.TRIALED1_TREATMENT,
                        enum.TRIALED_TREATMENT,
                        ]),
                ('Researched Treatments',[
                        enum.EXP_TREATMENT,
                        ]),
                ]
        for label,val_list in entries:
            self._quality_detail[label] = sum([
                    lookup.get(v,0)
                    for v in val_list
                    ])


        # for badge, rank based on number of KTs (25 == 100%)
        self._quality = self._quality_detail['Known Treatments'] * 4



class SearchTissueStage(WorkflowStage):
    def __init__(self,ws):
        super().__init__(ws, "Search Omics","ge:ae_search")
        self._ws = ws
    def weight(self):
        return 2
    def load_quality(self):
        from browse.models import AeSearch
        num_searches = AeSearch.objects.filter(ws=self._ws).count()
        self._quality_detail['Searches'] = str(num_searches)

class TissuesStage(WorkflowStage):
    def __init__(self,ws):
        super(TissuesStage, self).__init__(ws, "Gene Expression","ge:tissues")
        self._ws = ws
    def weight(self):
        return 1
    def load_quality(self):
        vals=Sample.group_vals
        assigned=[vals.CASE
            ,vals.CONTROL
            ]
        ws_tissues = Tissue.objects.filter(ws=self._ws)
        self._quality_detail['All Tissues'] = ws_tissues.count()
        from browse.models import TissueSet
        total_active=0
        # Tricky: we're going to repeat fields for each tissue set, but
        # _quality_detail wants each key to be unique.  So, we prepend each
        # key with some number of spaces, different for each tissue set.
        # These disappear in the HTML formatting.
        prefix = ''
        for ts in TissueSet.objects.filter(ws=self._ws):
            prefix=prefix+' '
            self._quality_detail[ts.name] = ''
            act_tiss_ids = set()
            protein_cnt = 0
            for t in ts.tissue_set.all():
                over,_,_,_ = t.sig_result_counts()
                if over:
                    act_tiss_ids.add(t.id)
                    protein_cnt += over
            cases = Sample.objects.filter(tissue_id__in=act_tiss_ids,
                    classification=vals.CASE,
                    ).count()
            controls = Sample.objects.filter(tissue_id__in=act_tiss_ids,
                    classification=vals.CONTROL,
                    ).count()
            for key,value in (
                        ('Active Tissues',len(act_tiss_ids)),
                        (ts.case_label+'s',cases),
                        (ts.control_label+'s',controls),
                        ('Proteins > thresh',protein_cnt),
                        ):
                self._quality_detail[prefix+'--'+key] = value
            total_active += len(act_tiss_ids)
        self._quality = total_active * 0.125

class PathsStage(WorkflowStage):
    def __init__(self,ws):
        super(PathsStage, self).__init__(ws, "Scores","nav_scoreboard")
        self._ws = ws
    def load_quality(self):
        from runner.process_info import JobCrossChecker
        jcc=JobCrossChecker()
        from dtk.scores import SourceList
        sl = SourceList(self._ws,jcc)
        sl.load_defaults()
        any = False
        for src in sl.sources():
            bji = src.bji()
            cat = bji.get_data_catalog()
            for code in cat.get_codes('wsa','score'):
                from dea import EnrichmentResult
                er = EnrichmentResult(bji,code)
                label = ' '.join([src.label(),cat.get_label(code)])
                self._quality_detail[label] = er.dea_link()
                any = True
        self._quality = 1 if any else 0

class ClassifierStage(WorkflowStage):
    def __init__(self,ws):
        super(ClassifierStage, self).__init__(ws, "Classifier","ml")
        self._ws = ws
    def load_quality(self):
        self._quality = 0
        outdir=PathHelper.ws_ml_publish(self._ws.pk)
        path=outdir+"results/full_vector/PlotsAndFinalPredictions/rankedListOfKnownRelations.csv"
        try:
            scores = []
            with open(path,"rb") as f:
                inp = csv.reader(f)
                for row in inp:
                    #print repr(row)
                    scores.append(float(row[1]))
            for cutoff in [0.75,0.90]:
                idx = int(cutoff*len(scores))
                self._quality_detail[percent(cutoff)+' KT cutoff'] = scores[idx]
            self._quality = self._quality_detail['75% KT cutoff']
        except Exception as ex:
            print("got exception: " + repr(ex))

class ReviewStage(WorkflowStage):
    def __init__(self,ws):
        super(ReviewStage, self).__init__(ws, "Review","rvw:review")
        self._ws = ws
    def load_quality(self):
        self._quality = 0
        enum=WsAnnotation.indication_vals
        ours=WsAnnotation.objects.filter(ws=self._ws)
        cand=enum.INITIAL_PREDICTION
        self._quality_detail['marked'] = ours.filter(marked_on__isnull=False).count()
        self._quality_detail['active'] = ours.filter(indication=cand).count()
        p = (enum.CANDIDATE_PATENTED,enum.PATENT_PREP)
        self._quality_detail['patents'] = ours.filter(indication__in=p).count()
        self._quality = self._quality_detail['patents']

class WorkspaceVDefaults(WorkflowStage):
    def __init__(self,ws):
        super(WorkspaceVDefaults, self).__init__(ws, "Workspace Default Settings",'ws_vdefaults')
    def weight(self):
        return 0.2
    def load_quality(self):
        self._quality_detail = OrderedDict()
        self._quality = 0
        self._quality_detail["Non-Default Settings"] = ""
        from browse.models import VersionDefault
        from browse.default_settings import Defaultable

        ws_defaults = VersionDefault.get_defaults(self._ws.id)
        defaults = VersionDefault.get_defaults(None)
        for key in ws_defaults:
            cls = Defaultable.lookup(key)
            if not getattr(cls, 'visible', True):
                continue

            if not cls.has_global_default():
                # This is a per-workspace setting, no sensible global default.
                continue

            if defaults[key] != ws_defaults[key]:
                self._quality_detail[' - ' + key] = cls.display_value(self._ws)


class DiseaseNames(WorkflowStage):
    def __init__(self,ws):
        super(DiseaseNames, self).__init__(ws, "Disease Names",'nav_disease_names')
    def weight(self):
        return 0.2

    def load_quality(self):
        from dtk.vocab_match import DiseaseVocab

        for name, cls in DiseaseVocab.get_subclasses():
            initial,dd = self._ws.get_disease_default(
                            name,
                            return_detail=True,
                            )
            if dd:
                # Add optional line breaks after these characters so that
                # the column width doesn't blow out.
                breakable = [',', '|', ':']
                for breakchar in breakable:
                    initial = initial.replace(breakchar, f'{breakchar}\u200B')

                self._quality_detail[name] = initial


class GeneExpressionData(TissuesStage):
    pass

class GWASData(WorkflowStage):
    def __init__(self,ws):
        super(GWASData, self).__init__(ws, "GWAS Data",'gwas_search')
    def weight(self):
        return 0.5
    def load_quality(self):
        self._quality_detail['# Datasets'] = len(self._ws.get_gwas_dataset_qs())

class DataStatusPage(WorkflowStage):
    def __init__(self,ws):
        super(DataStatusPage, self).__init__(ws, "Data Status Page Ready",'data_status')
    def weight(self):
        return 0.1
    def load_quality(self):
        from dtk.html import link
        faers_name = 'faers_%d' % self._ws.id
        dgn_name = 'dgn_%d' % self._ws.id
        faers_href = self._ws.reverse('nav_job_start', faers_name)
        dgn_href = self._ws.reverse('nav_job_start', dgn_name)
        faers_job = len(self._ws.get_prev_job_choices(faers_name))
        dgn_job = len(self._ws.get_prev_job_choices(dgn_name))

        self._quality_detail['FAERS job run'] = link(faers_job, faers_href)
        self._quality_detail['DisGeNet job run'] = link(dgn_job, dgn_href)

class KTSearch(IndicationsStage):
    pass

class ProteinSets(WorkflowStage):
    def __init__(self,ws):
        super(ProteinSets, self).__init__(ws, "Non-Novel Targets",'nav_ps')
    def weight(self):
        return 0.5
    def load_quality(self):
        def protset_detail(ps_id):
            name = self._ws.get_uniprot_set_name(ps_id)
            prot_len = len(self._ws.get_uniprot_set(ps_id))
            return '%s (%d prots)' % (name, prot_len)

        ps_intol = self._ws.get_intolerable_ps_default()
        ps_nonnovel = self._ws.get_nonnovel_ps_default()

        self._quality_detail['Non-novel'] = protset_detail(ps_nonnovel)
        self._quality_detail['Intolerable'] = protset_detail(ps_intol)


class GEOptimization(WorkflowStage):
    def __init__(self,ws):
        link_name = 'wf_%d_CombinedGEEvalFlow' % ws.id
        self._link_name = link_name
        super(GEOptimization, self).__init__(ws, "GE Optimization",('nav_job_start',link_name))
    def weight(self):
        return 0.2
    def load_quality(self):
        runs = get_jobs_by_name(self._link_name)
        self._quality_detail['# Successful Jobs'] = len(runs)

class GWASOptimization(WorkflowStage):
    def __init__(self,ws):
        link_name = 'wf_%d_CombinedGWASEvalFlow' % ws.id
        self._link_name = link_name
        super(GWASOptimization, self).__init__(ws, "GWAS Optimization",('nav_job_start',link_name))
    def weight(self):
        return 0.2
    def load_quality(self):
        runs = get_jobs_by_name(self._link_name)
        self._quality_detail['# Successful Jobs'] = len(runs)

class RefreshWorkflow(WorkflowStage):
    def __init__(self,ws):
        link_name = 'wf_%d_RefreshFlow' % ws.id
        self._link_name = link_name
        super(RefreshWorkflow, self).__init__(ws, "Refresh Workflow",('nav_job_start',link_name))
    def weight(self):
        return 0.25
    def load_quality(self):
        runs = get_jobs_by_name(self._link_name)
        self._quality_detail['# Successful Jobs'] = len(runs)

class RefreshQC(WorkflowStage):
    def __init__(self,ws):
        super(RefreshQC, self).__init__(ws, "Refresh QC",'nav_refresh_qc')
    def weight(self):
        return 0.25

class WzsTuning(WorkflowStage):
    def __init__(self,ws):
        link_name = 'wzs_%d' % ws.id
        self._link_name = link_name
        super(WzsTuning, self).__init__(ws, "WZS Tuning",('nav_job_start', link_name))
    def weight(self):
        return 0.25
    def load_quality(self):
        runs = get_jobs_by_name(self._link_name)
        self._quality_detail['# Successful Jobs'] = len(runs)

class ReviewWorkflow(WorkflowStage):
    def __init__(self,ws):
        link_name = 'wf_%d_ReviewFlow' % ws.id
        self._link_name = link_name
        super(ReviewWorkflow, self).__init__(ws, "Review Workflow",('nav_job_start',link_name))
    def weight(self):
        return 0.25
    def load_quality(self):
        runs = get_jobs_by_name(self._link_name)
        self._quality_detail['# Successful Jobs'] = len(runs)

class Prescreen(WorkflowStage):
    def __init__(self,ws):
        super(Prescreen, self).__init__(ws, "Prescreening",'nav_prescreen_list')
    def weight(self):
        return 4.5
    def load_quality(self):
        from browse.models import Prescreen, DispositionAudit
        pscrs = Prescreen.objects.filter(ws=self._ws)
        self._quality_detail['Prescreens'] = len(pscrs)
        screened_wsas = WsAnnotation.objects.filter(ws=self._ws, indication__gt=0)
        marked_wsas = screened_wsas.filter(marked_on__isnull=False)
        self._quality_detail['Molecules screened'] = len(screened_wsas)
        # This should be the same thing in most cases... but the computation
        # below is more appropriate.
        #self._quality_detail['Molecules marked'] = len(marked_wsas)
        ivals = WsAnnotation.indication_vals
        initial = DispositionAudit.objects.filter(indication=ivals.INITIAL_PREDICTION, wsa__ws=self._ws).values_list('wsa_id').distinct()
        self._quality_detail['Initial Predictions'] = initial.count()

class PatentSearch(WorkflowStage):
    def __init__(self,ws):
        super(PatentSearch, self).__init__(ws, "Patent Review",'pats_search')
    def weight(self):
        return 2
    def load_quality(self):
        from browse.models import WsAnnotation
        wsas = WsAnnotation.objects.filter(drugdiseasepatentsearch__wsa__ws=self._ws)
        self._quality_detail['Molecules Searched'] = len(wsas)

class ReviewRounds(WorkflowStage):
    def __init__(self,ws):
        super(ReviewRounds, self).__init__(ws, "Prelim Review Round",'rvw:review')
    def weight(self):
        return 60
    def load_quality(self):
        from browse.models import Election, DispositionAudit
        elections = Election.objects.filter(ws=self._ws)
        from collections import defaultdict
        counts = defaultdict(int)
        for election in elections:
            counts[election.flavor_info.label + ' Rounds'] += 1



        for label, count in counts.items():
            if 'Prelim' in label:
                self._quality_detail[label] = count

        ivals = WsAnnotation.indication_vals
        reviewed = DispositionAudit.objects.filter(indication=ivals.REVIEWED_PREDICTION, wsa__ws=self._ws).values_list('wsa_id').distinct()
        self._quality_detail["Reviewed Predictions"] = reviewed.count()

class SecondaryReviewRound(WorkflowStage):
    def __init__(self,ws):
        super().__init__(ws, "Second Review Round",'rvw:review')
    def weight(self):
        return 20
    def load_quality(self):
        from browse.models import Election, DispositionAudit
        elections = Election.objects.filter(ws=self._ws)
        from collections import defaultdict
        counts = defaultdict(int)
        for election in elections:
            counts[election.flavor_info.label + ' Rounds'] += 1

        for label, count in counts.items():
            if 'Prelim' not in label:
                self._quality_detail[label] = count

        ivals = WsAnnotation.indication_vals
        hits = DispositionAudit.objects.filter(indication=ivals.HIT, wsa__ws=self._ws).values_list('wsa_id').distinct()
        self._quality_detail["Hits"] = hits.count()


class CandidateWorkflow(WorkflowStage):
    def __init__(self,ws):
        link_name = 'wf_%d_CandidateFlow' % ws.id
        self._link_name = link_name
        super(CandidateWorkflow, self).__init__(ws, "Hit Workflow",('nav_job_start',link_name))
    def weight(self):
        return 0.1
    def load_quality(self):
        runs = get_jobs_by_name(self._link_name)
        self._quality_detail['# Successful Jobs'] = len(runs)

class HitSelection(WorkflowStage):
    def __init__(self,ws):
        super().__init__(ws, "Hit Selection",'moldata:hit_selection')
    def weight(self):
        return 8
# TBD not sure what to quantify here
#    def load_quality(self):

class TestTrainSplit(WorkflowStage):
    def __init__(self,ws):
        super(TestTrainSplit, self).__init__(ws, "Test/Train Split",'drugset')
    def weight(self):
        return 0.1
    def load_quality(self):
        ds = self._ws.eval_drugset
        test = len(self._ws.get_wsa_id_set('split-test-' + ds))
        train = len(self._ws.get_wsa_id_set('split-train-' + ds))
        self._quality_detail['# Train Set'] = "%d treatments" % train
        self._quality_detail['# Test Set'] = "%d treatments" % test

class HitReport(WorkflowStage):
    def __init__(self,ws):
        super().__init__(ws, "Hit Report",'rvw:review')
        self._href += "?flavor=patent"
    def weight(self):
        return 12
    def load_quality(self):
        from browse.models import WsAnnotation
        ival=WsAnnotation.indication_vals
        reported = self._ws.wsannotation_set.exclude(indication=ival.UNCLASSIFIED).exclude(doc_href='')

        reports = set([report.doc_href for report in reported])
        self._quality_detail['Hits with Reports'] = len(reported)
        from dtk.html import link
        for i, report in enumerate(reports):
            self._quality_detail[f'Report {i+1}'] = link('link', report)

def completion(steps, weighted=False):
    completed_steps, total_steps = completion_data(steps, weighted)
    return completed_steps / total_steps


def completion_data(steps, weighted=False):
    total_steps = 0
    completed_steps = 0
    for step in steps:
        weight = step.weight() if weighted else 1.0
        total_steps += weight
        completed_steps += step.completion() * weight
    return (completed_steps, total_steps)


def overall_progress(workflow):
    overall_completion = 0
    overall_total = 0
    for name, parts in workflow:
        completion, total = completion_data(parts, weighted=True)
        overall_completion += completion
        overall_total += total

    return overall_completion / overall_total


class AbundanceValidated(WorkflowStage):
    def __init__(self, ws):
        super().__init__(ws, "Data Abundance",'data_status')
    def weight(self):
        return 1
    def load_quality(self):
        date_changed = self.status_obj().changed_on
        self._quality_detail['Last Updated'] = date_changed.strftime("%Y-%m-%d %R")

def make_workflow(ws):
    workflow = [
            ('Configuration', [
                    AbundanceValidated(ws),
                    WorkspaceVDefaults(ws),
                    DrugCollections(ws, quick=True),
                    DiseaseNames(ws),
                ]),
            ('Data Selection/Annotation', [
                    SearchTissueStage(ws),
                    GeneExpressionData(ws),
                    GWASData(ws),
                    KTSearch(ws),
                    ProteinSets(ws),
                    DataStatusPage(ws),
                    TestTrainSplit(ws),
                    GEOptimization(ws),
                    GWASOptimization(ws),
                ]),
            ('Predictions', [
                    RefreshWorkflow(ws),
                    RefreshQC(ws),
                    WzsTuning(ws),
                    ReviewWorkflow(ws),
                ]),
            ('Hit Review', [
                    Prescreen(ws),
                    ReviewRounds(ws),
                    CandidateWorkflow(ws),
                    SecondaryReviewRound(ws),
                    HitSelection(ws),
                    PatentSearch(ws),
                    HitReport(ws),
                ]),
        ]
    return workflow
