# Tools related to composing patent and review notes

# Testing Notes:
# - this page is slow to load, so testing changes can be frustrating
# - it's faster to target a single wsa; you can do this with something like:
#   http://127.0.0.1:8000/88/patent_notes/?wsa_list=13211692&recompute=1
# - Note the 'recompute=1' above is also essential if you don't want
#   internal caching to hide the effect of code changes
# - these classes have been partially converted to LazyLoader; this means
#   that you can comment out some of the self.format_ calls in
#   extract_evidence, and it will avoid preparing for work it doesn't
#   end up doing; since this was a retrofit, it can probably be improved
# - adding &profile to the URL is a good way of tracking down remaining
#   overhead to improve testing turnaround

from dtk.lazy_loader import LazyLoader

class DrugNoteCollection:
    '''Holds drugs and notes for review_notes.html template.'''
    def __init__(self,wsa,demo,name_override=None):
        if name_override:
            self.name = name_override
        else:
            self.name = wsa.get_name(demo) if wsa is not None else None
        self.notes = []
        from collections import OrderedDict
        self.evidence = OrderedDict()
        self.tables = []

    @staticmethod
    def from_dict(data, demo):
        dnc = DrugNoteCollection(None, demo)
        from collections import OrderedDict
        dnc.name = data['name']
        dnc.evidence = OrderedDict(data['evidence'])
        dnc.tables = data['tables']
        for title, text in data['notes']:
            dnc.add_note(title, text)
        return dnc

    def to_dict(self):
        out = {
                'name': self.name,
                'tables': self.tables,
                'notes': self.notes,
                # We want to maintain ordering, so we can't save this out
                # as a dict, but instead we do a list of tuples.
                'evidence': list(self.evidence.items()),
                }
        return out

    def add_table(self,title,header=None):
        self.tables.append((title,header,[],[]))
    def add_table_row(self,*args):
        self.tables[-1][2].append(args)
    def add_table_footnote(self,text):
        self.tables[-1][3].append(text)
    def add_evidence(self,title,text,subbullets=[]):
        l = self.evidence.setdefault(title,[])
        l.append([text,subbullets])
    def add_note(self,title,text):
        from collections import namedtuple
        Note=namedtuple('Note','title txt')
        self.notes.append(Note(title,text))
    def add_vote(self,vote,user):
        from browse.models import Note
        title = vote.reviewer
        if vote.recommended is not None:
            title += " (%s)" % ('yes' if vote.recommended else 'no')
        self.add_note(
                title,
                Note.get(vote,'note',user),
                )

class DrugNoteCache(object):
    @staticmethod
    def get_cache():
        from django.core.cache import caches
        return caches['drugnotes']

    @classmethod
    def store(cls, wsa, dnc, wzs_jid=None, appendix=False):
        import json
        key = cls.key(wsa, wzs_jid, appendix)
        cls.get_cache().set(key, json.dumps(dnc.to_dict()))

    @classmethod
    def load(cls, wsa, wzs_jid=None, appendix=False):
        import json
        key = cls.key(wsa, wzs_jid, appendix)
        data = json.loads(cls.get_cache().get(key))
        return DrugNoteCollection.from_dict(data, False)

    @staticmethod
    def key(wsa, wzs_jid, appendix):
        if not wzs_jid:
            from browse.models import Prescreen
            score_src = wsa.marked_prescreen
            if not score_src:
                # This can happen if it was set manually instead of via
                # a prescreen
                wzs_jid = None
            else:
                wzs_jid = score_src.primary_job_id()
                # If it's not wzs, append the primary_code to the key name.
                if score_src.primary_code() != 'wzs':
                    wzs_jid = '%s%s' % (score_src.primary_code(), wzs_jid)

        return ','.join((str(wsa.id), str(wzs_jid), str(appendix)))

    @classmethod
    def has(cls, wsa, wzs_jid=None, appendix=False):
        key = cls.key(wsa, wzs_jid, appendix)
        return key in cls.get_cache()

    @classmethod
    def get_or_compute(cls, wsa, func, wzs_jid=None, force_compute=False, appendix=False):
        if cls.has(wsa, wzs_jid, appendix) and not force_compute:
            return cls.load(wsa, wzs_jid, appendix)
        else:
            data = func()
            cls.store(wsa, data, wzs_jid=wzs_jid, appendix=appendix)
            return data



def link(text,href):
    # like dtk.html.link, but always force a new tab (since this page
    # takes a long time to render, we don't want to accidentally jump
    # away), but without adding a 'new-window' icon (since glyph icons
    # are not enabled on this page)
    from dtk.html import tag_wrap
    return tag_wrap('a',text,dict(href=href,target='_blank'))

def uni_sci(val):
    from tools import sci_fmt
    s = sci_fmt(val)
    parts = s.split('e')
    if len(parts)>1:
        x10='\u00D710'
        parts[1] = x10+uni_sup(parts[1])
    return ''.join(parts)

def uni_sup(digit_string):
    # map each exponent char to unicode superscript
    sup = {
            '+':'\u207A',
            '-':'\u207B',
            '0':'\u2070',
            '1':'\u00B9',
            '2':'\u00B2',
            '3':'\u00B3',
            '4':'\u2074',
            '5':'\u2075',
            '6':'\u2076',
            '7':'\u2077',
            '8':'\u2078',
            '9':'\u2079',
            }
    return ''.join(sup[x] for x in digit_string)

def uni_sub(digit_string):
    sub = {
            '0':u'\u2080',
            '1':u'\u0081',
            '2':u'\u0082',
            '3':u'\u0083',
            '4':u'\u2084',
            '5':u'\u2085',
            '6':u'\u2086',
            '7':u'\u2087',
            '8':u'\u2088',
            '9':u'\u2089',
            }
    return u''.join(sub[x] for x in digit_string)

class WzsSources:
    def __init__(self,ws,jid):
        self.ws = ws
        self.jid = jid
        self.id = 'wzs_%s' % jid

    def source_list_job_ids(self):
        from runner.process_info import JobInfo
        bji = JobInfo.get_bound(self.ws, self.jid)
        return bji.get_all_input_job_ids()

class ScoreboardData(LazyLoader):
    def __init__(self,score_sources):
        super().__init__() # no LazyLoader kwargs functionality
        self.scoresrcs = score_sources
    def _roles_loader(self):
        # create map from role name to a bound JobInfo for each job id
        # listed in the saved scoreboard
        roles = {}
        from runner.process_info import JobInfo
        for job_id in self.scoresrcs.source_list_job_ids():
            bji = JobInfo.get_bound(self.scoresrcs.ws,job_id)
            role = bji.job.role
            if role:
                roles[role] = bji
        # now try to infer missing roles
        for cds in ('FAERS',):
            base_key = cds+'_faers'
            if base_key not in roles:
                self._extract_role_from_downstream(
                        roles,
                        base_key,
                        base_key+'_capp',
                        lambda x:int(x['faers_run']),
                        )
        suffix='_codes'
        for role in list(roles.keys()):
            if not role.endswith(suffix):
                continue
            base = role[:-len(suffix)]
            if base not in roles:
                self._extract_role_from_downstream(
                        roles,
                        base,
                        role,
                        lambda x:int(x['input_score'].split('_')[0]),
                        )
        return roles
    def _extract_role_from_downstream(self,roles,base,downstream,extract):
        try:
            ds_bji=roles[downstream]
        except KeyError:
            return False
        try:
            job_id=extract(ds_bji.job.settings())
        except (KeyError,ValueError):
            return False
        from runner.process_info import JobInfo
        bji = JobInfo.get_bound(self.scoresrcs.ws,job_id)
        roles[base]=bji
        return True
    def _drug_scores_loader(self):
        result = {}
        for role,bji in self.roles.items():
            cat = bji.get_data_catalog()
            for code in cat.get_codes('wsa','score'):
                result[(role,code)] = self.top_pct(cat,code)
        return result
    def _pathway_scores_loader(self):
        '''Return {role:ranker,...} for each role supplying pathways.

        Ranker is ranking the scores as pre-processed for GLF, rather
        than a raw score.
        '''
        result = {}
        from algorithms.run_depend import glf_input_from_bji
        from dtk.scores import Ranker
        for role,bji in self.roles.items():
            if not role.endswith('_glf'):
                continue
            # the -2.0 is copied from the default on the DEEPEND settings form
            scores = [(k,s) for k,s,d in glf_input_from_bji(bji,-2.0)]
            scores.sort(key=lambda x:x[0],reverse=True)
            result[role] = Ranker(scores)
        return result
    def _protein_score_sets_loader(self):
        '''protein_score_sets is a list of tuples, where each tuple is a label,
        a boolean flag indicating whether PPI information was used,
        and protein information.

        The protein information is either a Ranker, or a dictionary
        of Rankers keyed by the label of the score subtype.
        '''
        # get labels for otarg scores
        from runner.process_info import JobInfo
        uji=JobInfo.get_unbound('otarg')
        cat=uji.get_data_catalog()
        otarg_labels={
                code:get_full_otarg_label(cat.get_label(code))
                for code in cat.get_codes('uniprot','score')
                if code not in ['overall']
                }
        # scan scoreboard and collect interesting protein scores by group
        out=[]

        prot_score_types = ['agr', 'dgn']

        otarg_scores={}
        otarg_ppi_scores={}
        # Similar to otarg, but without sublabels.
        single_prot_scores = dict()
        single_prot_ppi_scores = dict()
        for role,bji in self.roles.items():
            cat = bji.get_data_catalog()
            group = role # unless overridden
            if role == 'gwasig_sigdif':
                out.append(('GWAS',True,self.top_pct(cat,'difEv')))
            elif role == 'otarg':
                for code in cat.get_codes('uniprot','score'):
                    try:
                        label = otarg_labels[code]
                    except KeyError:
                        continue
                    otarg_scores[label] = self.top_pct(cat,code)
            elif role in prot_score_types:
                for code in cat.get_codes('uniprot','score'):
                    single_prot_scores[code] = self.top_pct(cat,code)
            elif role.endswith('_otarg_sigdif'):
                src = role.split('_')[0]
                label = otarg_labels[src]
                otarg_ppi_scores[label] = self.top_pct(cat,'difEv')
            elif role.endswith('_sigdif'):
                # Non-opentargets sigdif
                src = role.split('_')[-2]
                single_prot_ppi_scores[src] = self.top_pct(cat, 'difEv')
            elif role == 'tcgamut':
                out.append((role, False, self.top_pct(cat, 'muten')))
            elif role.endswith('_gesig'):
                out.append((role + '|up', False, self.top_pct(cat, 'ev')))
                out.append((role + '|down', False, self.top_pct(cat, 'ev', descending=False)))
        if otarg_scores:
            out.append(('OpenTargets',False,otarg_scores))
        if otarg_ppi_scores:
            out.append(('OpenTargets',True,otarg_ppi_scores))
        for label, scores in single_prot_scores.items():
            out.append((label, False, scores))
        for label, scores in single_prot_ppi_scores.items():
            out.append((label, True, scores))
        return out
    def top_pct(self,cat,code,descending=True):
        ordering = cat.get_ordering(code,descending)
        from dtk.scores import Ranker
        return Ranker(ordering)
    def top_drug_scores(self,wsa,min_top_rnk=100):
        hits = {}
        for k,rnkr in self.drug_scores.items():
            ahead, tied, behind = rnkr.get_details(wsa)
            if ahead >= min_top_rnk or ahead >= rnkr.total or behind == 0:
                continue
            # In the case of ties, this reports us as the highest rank.
            hits[k]=(ahead + 1, rnkr.total, tied)
        return hits
    def top_pathway_scores(self, drug_pathways):
        '''Return [(role,[(path_id,%rank),...]),...]

        For each role supplying disease pathways, return any pathways from
        the set of drug_pathways that rank in the top percent_cutoff %.
        Both main list and sublists are ordered best match first.
        '''
        percent_cutoff = 10
        result = []
        for role,rnkr in self.pathway_scores.items():
            route_pct = [
                    (drug_pw,rnkr.get_pct(drug_pw))
                    for drug_pw in drug_pathways
                    ]
            drug_matches = [x for x in route_pct if x[1] <= percent_cutoff]
            drug_matches.sort(key=lambda x:x[1])
            if drug_matches:
                result.append((role,drug_matches[:5]))
        # order sources by lowest pathway percent
        result.sort(key=lambda x:x[1][0][1])
        return result
    def top_protein_scores(self,prots, min_top_pct=10):
        # Anything <= this rank will be kept regardless of pct.
        kRankKeepThreshold = 10

        for label,ppi,scores in self.protein_score_sets:
            if type(scores) == dict:
                hits=[]
                for prot in prots:
                    for subtype,score in scores.items():
                        pct = score.get_pct(prot)
                        rank = score.get(prot)
                        if pct > min_top_pct and rank > kRankKeepThreshold:
                            continue
                        hits.append((prot,pct,subtype,score.total))
                if hits:
                    yield dict(
                            prots_d={(x[0],x[2]):x[1] for x in hits},
                            pct=max([x[1] for x in hits]),
                            total=min([x[3] for x in hits]),
                            source=label,
                            ppi=ppi,
                            subtypes=sorted(set([x[2] for x in hits])),
                            )
            else:
                hits=[]
                for prot in prots:
                    pct = scores.get_pct(prot)
                    rank = scores.get(prot)
                    if pct > min_top_pct and rank > kRankKeepThreshold:
                        continue
                    hits.append((prot,pct, scores.total))
                if hits:
                    yield dict(
                            prots_d={(x[0],None):x[1] for x in hits},
                            pct=max([x[1] for x in hits]),
                            total=min([x[2] for x in hits]),
                            source=label,
                            ppi=ppi,
                            )
def get_full_otarg_label(s):
    if s == 'Gene Assoc':
        return 'Genetic Association'
    elif s == 'RNA Expr':
        return 'RNA Expression'
    elif s == 'Somatic Mut':
        return 'Somatic Mutation'
    return s

class DrugData(LazyLoader):
    def __init__(self,ec,wsa,dnc,wzs_jid=None):
        super().__init__() # no LazyLoader kwargs functionality
        self.wsa = wsa
        self.ws = self.wsa.ws
        self.dnc = dnc
        self.warnings = []
        self.ec = ec
        from browse.models import Prescreen
        if wzs_jid:
            score_src = WzsSources(self.ws, wzs_jid)
        else:
            score_src = wsa.marked_prescreen
        self.score_src = score_src
        self.dm=ec.dm
        self.all_dpi_targs = ec.all_dpi_targs
        self.dpi_thresh=ec.dpi_thresh
        self.pm=ec.pm
        self.ppi_thresh=ec.ppi_thresh
        self.all_ppi_targs = ec.all_ppi_targs
        self.load_direct_targets()
        self.load_indirect_targets()
        self.all_possible_proteins = set(self.all_ppi_targs) | set(self.all_dpi_targs)

    @property
    def scoreboard(self):
        if not hasattr(self, '_scoreboard'):
            ec = self.ec
            score_src = self.score_src
            self._scoreboard = ec.get_scoreboard_data(score_src) if score_src and ec else None
        return self._scoreboard

    def _pathways_loader(self):
        '''Return set of pathway ids matching this drug's targets.
        '''
        return set(p
                for p,s in self.ec.pathway_db.protsets.items()
                if set(s) & self.direct_targets
                )

    def load_direct_targets(self):
        dm = self.dm.dpimap_for_wsa(self.wsa)
        self.direct_targets = set(x[1]
                for x in dm.get_dpi_info(self.wsa.agent)
                if float(x[2]) >= self.dpi_thresh
                )
        self.n_dts = len(self.direct_targets)
    def load_indirect_targets(self):
        self.indirect_targets = {}
        self.all_indirect_targets = set()
        if not self.pm:
            return
        all_ind = self.pm.get_ppi_info_for_keys(self.direct_targets,
                                                min_evid = self.ppi_thresh
                                               )
        for tup in all_ind:
            if tup[0] not in self.direct_targets:
                continue
            if tup[0] not in self.indirect_targets:
                self.indirect_targets[tup[0]] = set()
            self.indirect_targets[tup[0]].add(tup[1])
            self.all_indirect_targets.add(tup[1])

def format_cnt(cnt):
    return "{:,}".format(cnt)
# lifted from https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
def fmt_rnk(rnk):
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])
    return ordinal(rnk)
def fmt_pct(pct):
    import math
    exp = int(math.log(pct,10))
    if exp <= -3:
        return 0.0001
    if exp <= -2:
        return '%0.3f'%pct
    if exp <= -1:
        return '%0.2f'%pct
    if exp < 1:
        return '%0.1f'%pct
    return '%0.0f'%pct

class EvidenceComposer(LazyLoader):
    # A class for managing the creation of textual descriptions of drug
    # evidence. Usage is:
    # - instatiate the class once per workspace
    # - call extract_evidence for each drug, which adds 'evidence' text to a
    #   DrugNoteCollection object
    # - pass a list of DrugNoteCollection objects to the review_notes.html
    #   template to create a file of text that can be pasted into a document
    #
    # The general design is:
    # - this class does any generic workspace setup (i.e. not drug specific)
    #   in the __init__ method
    # - the ScoreboardData class holds data specific to a saved scoreboard,
    #   but possibly shared by many drugs from that scoreboard
    # - the DrugData class holds drug-specific data
    # - each extract_evidence call begins by instantiating a DrugData object
    #   and setting it as self.drug; it then calls functions to format
    #   each data section, and finally clears self.drug before returning
    # - in addition to holding workspace-wide data, this class provides:
    #   - a cache of previously-built ScoreboardData objects
    #   - low-level formatting functions
    def __init__(self,ws,appendix_only=False):
        super().__init__() # no LazyLoader kwargs functionality
        self.ws = ws
        self.scoreboards = {}
        self.load_dpi_data()
        self.load_ppi_data()
    def load_dpi_data(self):
        from dtk.prot_map import DpiMapping
        self.dm = DpiMapping(self.ws.get_dpi_default())
        if self.dm.is_combo():
            self.dm = self.dm.get_baseline_dpi()
        self.dpi_thresh = self.ws.get_dpi_thresh_default()
    def _all_dpi_targs_loader(self):
        return self.dm.get_filtered_uniq_target(min_evid=self.dpi_thresh)
    def load_ppi_data(self):
        from dtk.prot_map import PpiMapping
        self.ppi_thresh = self.ws.get_ppi_thresh_default()
        self.pm = PpiMapping(self.ws.get_ppi_default())
    def _all_ppi_targs_loader(self):
        return self.pm.get_filtered_uniq_target(min_evid=self.ppi_thresh)
    def _pathway_db_loader(self):
        '''Return a class holding get_pathway_data results as members.

        This loads the global pathway database into memory for access
        elsewhere in the composer.
        '''
        class Dummy: pass
        db = Dummy()
        from dtk.gene_sets import get_pathway_data
        db.protsets, db.pathways_data, db.hier = get_pathway_data()
        return db
    def extract_evidence(self,wsa,dnc,wzs_jid=None,appendix=False):
        print('extracting',wsa)
        self.drug = DrugData(self,wsa,dnc,wzs_jid)
        if appendix:
            self.format_ortholog_table()
            self.format_patent_table(wsa)
            return
        if self.drug.scoreboard:
            self.format_target_table()
            self.format_ancillary_table()
            self.format_top_scores()
            self.format_clinical()
            self.format_direct_omics()
            self.format_indirect_omics()
            self.format_additional_targets()
            self.format_pathways()
        else:
            self.warn("Can't locate prescreen for drug")
        self.format_warnings()
        self.drug = None
    ####
    # high-level formatting
    ####
    def format_warnings(self):
        if self.drug.warnings:
            for txt in self.drug.warnings:
                self.drug.dnc.add_evidence('Data Retrieval Warnings',txt)
    def format_ortholog_table(self):
        from browse.models import Protein
        from dtk.orthology import get_ortho_records
        data = []
        prots = list(self.drug.direct_targets)
        u2g = Protein.get_uniprot_gene_map(uniprots=prots)

        header = ['Human Gene', 'Organism', 'Organism Gene', 'Similarity (%)']
        for rec in get_ortho_records(self.ws, uniprots=prots):
            data.append((
                u2g.get(rec['uniprot'], rec['uniprot']),
                rec['organism'],
                rec['organism_gene'],
                f'{rec["similarity(%)"]:.1f}',
                ))
        self.drug.dnc.add_table('Target Orthologs', header)
        for row in data:
            self.drug.dnc.add_table_row(*row)
    
    def format_patent_table(self, wsa):
        header = ['Status', 'ID', 'Title']
        self.drug.dnc.add_table('Related Patents', header)
        from patsearch.models import DrugDiseasePatentSearch, PatentSearchResult
        searchs = DrugDiseasePatentSearch.objects.filter(wsa=wsa)
        res_vals = PatentSearchResult.resolution_vals
        resolutions = [res_vals.RELEVANT, res_vals.NEEDS_MORE_REVIEW]
        rows = []
        from dtk.url import google_patent_search_url
        for search in searchs:
            for patent_res in search.patentsearchresult_set.filter(resolution__in=resolutions):
                pat_id = patent_res.patent.pub_id
                url = patent_res.patent.href
                row = [
                    patent_res.resolution_text,
                    # Replace hyphen with non-breaking-hyphen in pat id.
                    link(pat_id.replace('-', '‑'), url),
                    patent_res.patent.title,
                    ]
                self.drug.dnc.add_table_row(*row)

            search_url = wsa.ws.reverse('pats_resolve', search.id)
            self.drug.dnc.add_table_footnote(link(f'Platform results for {search.drug_name}', search_url))


    def format_target_table(self):
        sub50 = uni_sub('50')
        self.drug.dnc.add_table(
                'Candidate drug target profile',
                ['Target',u'Biochemical EC%s/IC%s'%(sub50,sub50)],
                )
        # provide a link to chembl literature for manual interpretation
        chembl_ids = self.drug.wsa.agent.chembl_id_set
        chembl_ids |= self.drug.wsa.agent.m_chembl_id_set
        from dtk.url import chembl_lit_url
        from dtk.html import join
        urls = [chembl_lit_url(x) for x in chembl_ids]
        for x in chembl_ids:
            self.drug.dnc.add_table_footnote(link(
                    join(u'literature for ',x),
                    chembl_lit_url(x),
                    ))
        # get data for each direct target, ordered by gene name
        from browse.models import Protein
        prot_data = []
        for uniprot in self.drug.direct_targets:
            try:
                prot = Protein.objects.get(uniprot=uniprot)
            except Protein.DoesNotExist:
                prot = Protein(uniprot=uniprot, gene="(%s)" % uniprot)
            prot_data.append((prot,prot.get_name()))
        prot_data.sort(key=lambda x:x[0].gene)
        # build table with one row for each protein in the prot_data
        # list, adding in any supporting assay data
        aa = self.drug.wsa.get_assay_info()
        for prot,prot_name in prot_data:
            assays = [x for x in aa.assays if x[2] == prot.uniprot]
            assays.sort(key=lambda x:x[4]) # sort by uM
            self.drug.dnc.add_table_row(
                    u'%s (%s)'%(prot.get_name(),prot.gene),
                    self.format_assays(assays),
                    )
        # XXX we could attempt to construct this field from chembl
        # XXX activity data, but that would be difficult.
        # XXX - there's an API client available:
        # XXX   https://github.com/chembl/chembl_webresource_client
        # XXX   per issue 38 on github, I needed to force gevent==1.2.2,
        # XXX   greenlet==0.4.12
        # XXX - can do retrieval like:
        # XXX   new_client.activity.filter(molecule_chembl_id='CHEMBL22242')
        # XXX   but then need to
        # XXX   - filter (target_organism, standard_type)
        # XXX   - group by target_chembl_id and match to gene
        # XXX   - consolidate potentially very different values across
        # XXX     relation, value, and standard_units
    def format_ancillary_table(self):
        self.drug.dnc.add_table('Ancillary candidate drug information')
        self.drug.dnc.add_table_row(
                '<i>Indication (status)</i>',
                '???', # GlobalData? can't find on ChEMBL
                )
        self.drug.dnc.add_table_row(
                '<i>ROA</i>',
                '???', # GlobalData
                )
        self.drug.dnc.add_table_row(
                '<i>Originator</i>',
                '???', # GlobalData
                )
        self.drug.dnc.add_table_row(
                '<i>Developer</i>',
                '???', # GlobalData
                )
        self.drug.dnc.add_table_row(
                '<i>Adverse Events (grade 1-2)</i>',
                '???', # SIDER? lit? drugbank toxicity?
                )
        self.drug.dnc.add_table_row(
                '<i>Adverse Events (grade 3)</i>',
                '???', # SIDER? lit? drugbank toxicity?
                )
        self.drug.dnc.add_table_row(
                '<i>Dose level</i>',
                '???', # GlobalData or lit?
                )
    def format_clinical(self, significance_threshold = 0.05):
        pval = self.get_score('FAERS_faers','lrpvalue')
        if pval is not None and pval <= significance_threshold:
            direction = self.get_score('FAERS_faers','lrdir')
            if direction == -1:
                effect = self.format(u'''
                    This analysis indicates that {drug} may have
                    a protective effect, though other interpretations
                    of that result are possible.
                    ''',
                    )
            else:
                effect = ''
            self.bullet('Clinical data',u'''
                {drug} is significantly associated with {disease}
                based on clinical data analysis (p = {pval}).
                {effect}
                ''',
                pval=uni_sci(pval),
                effect=effect,
                # XXX add sentence if there's a similar clinical
                # XXX signature for other drugs in this class
                )
    def format_direct_omics(self):
        from browse.models import Protein
        for prot in self.drug.direct_targets:
            gene = Protein.get_gene_of_uniprot(prot)
            for d in self.ge_results(prot):
                self.bullet('Target gene expression data',u'''
                        {gene} expression was {signif}
                        altered in {count} of the {total}
                        {tiss_set} gene expression datasets analyzed
                        that measured {gene} expression.
                        ''',
                        gene=gene,
                        **d
                        )
            if prot in self.all_gwas_prots:
                self.bullet('Target genetics data',u'''
                        {gene} was detected in {count} of the {total}
                        GWAS datasets analyzed.
                        ''',
                        gene=gene,
                        total = len(self.gwas_data),
                        count = len([1 for d in self.gwas_data
                                     if prot in d
                                    ])
                        )
    def format_indirect_omics(self):
        self.format_ge_indirect()
        self.format_gwas_indirect()
    def format_gwas_indirect(self, enrich_thresh = 0.05):
        from browse.models import Protein
        for d_prot, i_prot_set in self.drug.indirect_targets.items():
            overlap = i_prot_set & self.all_gwas_prots
            if overlap:
                count = len(overlap)
                were_was = 'was' if count == 1 else 'were'
                tup = self.calc_fishers_exact(count,
                                          len(i_prot_set),
                                          len(self.all_gwas_prots),
                                          len(self.all_gwas_prots | self.drug.all_possible_proteins)
                                         )
                if tup[1] <= enrich_thresh:
                    self.bullet('Target genetics data',u'''
                        {count} of the {total} indirect targets which
                        interact with {gene} {were_was} detected in at least
                        one GWAS datasets analyzed ({p},
                        One-tailed Fischer's exact test).
                        ''',
                        gene=Protein.get_gene_of_uniprot(d_prot),
                        total = format_cnt(len(i_prot_set)),
                        count = count,
                        were_was = were_was,
                        p = u'p = %s'%uni_sci(tup[1])
                        )
    def format_ge_indirect(self, enrich_thresh = 0.05):
        from browse.models import Protein
        for ts,tissues in self.sigprot_data:
            all_me_proteins = set([k for d in tissues for k in d])
            n_all_me_prots = len(all_me_proteins)
            n_universe = len(self.all_ge_prots[ts.id] | self.drug.all_possible_proteins)
            indirect_me_targets = self.drug.all_indirect_targets & all_me_proteins
            n_indirect_me_targets = len(indirect_me_targets)
            n_its = len(self.drug.all_indirect_targets)
            perc = 0.0
            if n_its:
                perc = round(float(n_indirect_me_targets)/n_its*100.0, 1)
            tup = self.calc_fishers_exact(n_indirect_me_targets,
                                          n_its,
                                          n_all_me_prots,
                                          n_universe
                                         )
            pval = tup[1]
            l = []
            dprots = set()
            for d_prot, i_prot_set in self.drug.indirect_targets.items():
                dprots.add(d_prot)
                tup = self.calc_fishers_exact(len(i_prot_set & all_me_proteins),
                                           len(i_prot_set),
                                           n_all_me_prots,
                                           n_universe
                                          )
                if tup[1] <= enrich_thresh:
                    g = Protein.get_gene_of_uniprot(d_prot)
                    l.append(' '.join([g, u'(p = %s, One-tailed Fischer\'s exact test)'%uni_sci(tup[1])]))
            last_sentence = ''
            if l and len(dprots) > 1:
                last_sentence = 'This includes significant portions for'
                if len(l) == 1:
                    last_sentence += ' ' + ", ".join(l)+"."
                else:
                    last_sentence += ':'
            if last_sentence or perc > 50 or pval <= enrich_thresh:
                k = 'Target gene expression data'
                subbullets=[]
                if len(l) > 1:
                    subbullets=l
                self.bullet(k,u'''
                         {perc}% of the {n_its} indirect targets (proteins which
                         directly interact with the drug's direct targets;
                         ({pval}, One-tailed Fischer's exact test))
                         have significantly altered expression in at least one
                         {tiss_name} dataset. {last_sentence}
                       ''',
                       subbullets=subbullets,
                       perc = perc,
                       n_its = format_cnt(n_its),
                       pval =  u'p = %s'%uni_sci(pval),
                       tiss_name = ts.ts_label(),
                       last_sentence = last_sentence
                       )
    def calc_fishers_exact(self, me_in, total_in, all_me_proteins, total_universe):
        import scipy.stats as sps
        nme_in = total_in - me_in
        return sps.fisher_exact([
                             [me_in, nme_in],
                             [all_me_proteins - me_in
                              , total_universe - all_me_proteins - nme_in
                             ]
                            ],
                            alternative = 'greater'
                           )
    def format_top_scores(self):
        hits = self.drug.scoreboard.top_drug_scores(self.drug.wsa.id)
        for k,(rank, total, ties) in hits.items():
            # Ties normally includes itself, -1 to remove.
            ties_txt = '' if ties <= 1 else ' (tied with %d others)' % (ties - 1,)

            self.bullet('Top-ranked algorithms',
                        u'''Ranked {rnk}{ties_txt} in {role} - {code}.''',
                        role=k[0],
                        code=k[1],
# the total turned out to be more confusing than helpful
                        #total=format_cnt(v[1]),
                        rnk=fmt_rnk(rank),
                        ties_txt=ties_txt,
                       )
    def format_pathways(self):
        for role,match_list in self.drug.scoreboard.top_pathway_scores(
                self.drug.pathways,
                ):
            sub_bullets = [
                    self.pathway_db.pathways_data[pw_id]['name'] \
                        + f' ({pw_id}), in top {int(pct_rank)+1}%'
                    for pw_id,pct_rank in match_list
                    ]
            self.bullet('Pathway data',
                    f'{role} matched these pathways:',
                    subbullets=sub_bullets,
                    )
    def format_additional_targets(self):
        from browse.models import Protein
        for d in self.drug.scoreboard.top_protein_scores(
                self.drug.direct_targets
                ):
            # format genes
            genes_d = {(Protein.get_gene_of_uniprot(k[0]),k[1]):v
                       for k,v in d['prots_d'].items()
                      }
            subtypes = d.get('subtypes')
            ppi = d.get('ppi')
            d.update(pct=fmt_pct(d['pct']))
            d.update(total=format_cnt(d['total']))
            if d['source'] == 'GWAS':
                k = 'Target genetics data'
            elif '_gesig' in d['source']:
                k = 'Target gene expression data'
            else:
                k = 'Additional target data'
            verb = ' was'
            if not subtypes:
                genes = [(x[0],v) for x,v in genes_d.items()]
                genes.sort(key = lambda x: x[0])
                if len(genes) > 1:
                    gene_phrase = [x[0]+" ("+fmt_pct(x[1])+"%)"
                                   for x in genes
                                  ]
                else:
                    gene_phrase = [x[0] for x in genes]
                gene_phrase = self.and_list(gene_phrase)
                if len(genes) > 2:
                    verb = ' were all'
                elif len(genes) > 1:
                    verb = ' were both'
                gene_phrase += verb
               # format basis phrase
                if ppi:
                    basis='based on the integration of {source} and PPI data.'
                else:
                    basis='based on {source} data.'
                # assemble
                if '_gesig' not in d['source']:
                    self.bullet(k,u'''
                            {gene_phrase} ranked
                            in the top {pct}% of all {total} proteins
                            '''+basis,
                            gene_phrase=gene_phrase,
                            **d
                            )
                else:
                    role, direction = d['source'].split('|')

                    if role == 'cc_gesig':
                        type = 'Case/Control'
                    elif role == 'mirna_gesig':
                        type = 'miRNA'
                    else:
                        type = role

                    self.bullet(k,u'''
                            {gene_phrase} among the most consistently
                            {direction}-regulated genes measured in a meta-analysis of
                            all {type} datasets (top {pct}% of all {total}).
                            ''',
                            gene_phrase=gene_phrase,
                            direction=direction,
                            type=type,
                            **d
                            )
            else:
                new_d={}
                for x,p in genes_d.items():
                    if x[0] not in new_d:
                        new_d[x[0]] = {}
                    new_d[x[0]][x[1]] = p
                for g in new_d:
                    gene_phrase = g+verb
                    # format basis phrase
                    if ppi:
                        basis= u'''
                        when the following data types from {source}
                        were integrated with PPI data: '''
                    else:
                        basis= u'''
                        based on the following data types from {source}: '''
                    # assemble
                    subbullets=[]
                    if len(new_d[g]) > 1:
                        for s in sorted(new_d[g], key=lambda x: new_d[g][x]):
                            subbullets.append(s+" (top "+fmt_pct(new_d[g][s])+"%)")
                    else:
                        basis += list(new_d[g].keys())[0]+"."
                    self.bullet(k,u'''
                        {gene_phrase} ranked
                        in the top {mypct}% of all {total} proteins
                        '''+basis,
                        subbullets=subbullets,
                        gene_phrase=gene_phrase,
                        mypct = fmt_pct(max([p for p in new_d[g].values()])),
                        **d
                        )
    ####
    # scoreboard cache
    ####
    def get_scoreboard_data(self,score_src):
        if score_src.id not in self.scoreboards:
            print('building scoreboard data for prescreen/wzs',score_src.id)
            self.scoreboards[score_src.id] = ScoreboardData(score_src)
        return self.scoreboards[score_src.id]
    ####
    # GWAS data access
    ####
    def _gwas_data_loader(self):
        from dtk.gwas import gwas_codes, scored_gwas
        return [
                {u:s
                    for u,s in scored_gwas(ds).items()
                    if u in self.drug.all_possible_proteins
                    }
                for ds in gwas_codes(self.ws)
                ]
    def _all_gwas_prots_loader(self):
        result = set()
        for d in self.gwas_data:
            result |= set(d.keys())
        return result
    ####
    # GE data access
    ####
    def _ts_tissues_loader(self):
        # tissues for each tissue set
        return {
                ts:ts.tissue_set.exclude(over_proteins=0)
                for ts in self.ws.tissueset_set.all()
                }
    def _sigprot_data_loader(self):
        # [(ts,[{uniprot:ssr},...]),...}
        # returns list of (ts,[per_tissue_data,...])
        # where per_tissue_data is {uniprot:srr,...} for over-threshold results
        result = []
        for ts,tissues in self.ts_tissues.items():
            l = [
                    {srr.uniprot:srr for srr in tissue.sig_results()}
                    for tissue in tissues
                    ]
            if l:
                result.append((ts,l))
        return result
    def _tissue_cnts_loader(self):
        # {ts_id:{uniprot:cnt,...},...} with count of results over all tissues,
        # not just over-threshold
        result = {}
        for ts,tissues in self.ts_tissues.items():
            result[ts.id] = {}
            for tissue in tissues:
                for srr in tissue.sig_results(over_only=False):
                    if srr.uniprot not in result[ts.id]:
                        result[ts.id][srr.uniprot] = 0
                    result[ts.id][srr.uniprot] += 1
        return result
    def _all_ge_prots_loader(self):
        # set of uniprots seen in each tissue set (including under-threshold)
        return {
                ts_id:set(cnt_d.keys())
                for ts_id,cnt_d in self.tissue_cnts.items()
                }
    def ge_results(self,prot):
        for ts,tissues in self.sigprot_data:
            class EvCounter:
                def __init__(self):
                    self.ev_list=[]
                def add_ev(self,ev):
                    self.ev_list.append(ev)
                def any(self):
                    return bool(self.ev_list)
                def readout(self):
                    ev = min(self.ev_list)
                    count=len(self.ev_list)
                    comp = '=' if count == 1 else '<='
                    p = 1-ev
                    if p:
                        signif = u'significantly (p %s %s, BH corrected)'%(comp, uni_sci(1-ev))
                    else:
                        signif = u'significantly (p %s %s, BH corrected)'%('<', 0.0001)
                    return dict(
                            count=count,
                            signif=signif,
                            )
            up = EvCounter()
            down = EvCounter()
            updown = EvCounter()
            for tissue in tissues:
                try:
                    srr = tissue[prot]
                except KeyError:
                    continue
                if srr.direction > 0:
                    up.add_ev(srr.evidence)
                else:
                    down.add_ev(srr.evidence)
                updown.add_ev(srr.evidence)
            d = dict(
                    total= 0 if prot not in self.tissue_cnts[ts.id] else self.tissue_cnts[ts.id][prot],
                    tiss_set=ts.ts_label(),
                    )
            if False:
                # We no longer include direction, but leaving this here in ase
                # we change our mind.
                if up.any():
                    d.update(up_dn='up', **up.readout())
                    yield d
                if down.any():
                    d.update(up_dn='down', **down.readout())
                    yield d

            if updown.any():
                d.update(**updown.readout())
                yield d
    ####
    # data catalog access (drug scores)
    ####
    def get_score(self,role,code):
        try:
            bji = self.drug.scoreboard.roles[role]
        except KeyError:
            self.warn(
                    "No role {role} in prescreen ({pscr_id})",
                    role=role,
                    )
            return None
        cat = bji.get_data_catalog()
        try:
            val,attrs = cat.get_cell(code,self.drug.wsa.id)
        except ValueError:
            self.warn(
                    "No {code} scores in role {role} ({job_id})",
                    code=code,
                    role=role,
                    job_id=bji.job.id,
                    )
            return None
        if val is None:
            self.warn(
                    "No {code} score for this drug in role {role} ({job_id})",
                    code=code,
                    role=role,
                    job_id=bji.job.id,
                    )
        return val
    ####
    # low-level formatting
    ####
    def format_assays(self,assays):
        # - return one line per assay, separated by <br>
        # - line consists of <assay_type> <conc> uM (<detail>)
        # - detail is a link to the reference, optionally followed by:
        #   <count> assays, std. dev. <sd>
        #   in cases where multiple assays are condensed
        from dtk.html import join
        lines = []
        fmt_type = {
                'c50':'C'+uni_sub('50'),
                'ki':'Ki',
                }
        for atype,drug_key,protein,direction,nm,count,std_dev in assays:
            from dtk.url import chembl_drug_url, bindingdb_drug_url
            if drug_key.startswith('CHEMBL'):
                url = chembl_drug_url(drug_key)
            elif drug_key.startswith('BDBM'):
                url = bindingdb_drug_url(drug_key)
            else:
                url = ''
            detail=[link(drug_key,url) if url else drug_key]
            if count > 1:
                detail += [
                        str(count),
                        'assays, std. dev.',
                        uni_sci(std_dev),
                        ]
            lines.append(join(
                    fmt_type[atype],
                    ' ',
                    uni_sci(nm),
                    u' nM (',
                    join(*detail),
                    ')',
                    sep=u'',
                    ))
        from django.utils.safestring import mark_safe
        return join(*lines,sep=mark_safe(u'<br>'))
    def bullet(self,heading,template,subbullets=[],**kwargs):
        self.drug.dnc.add_evidence(
                heading,
                self.format(template,**kwargs),
                subbullets = [self.format(s) for s in subbullets]
                )
    def warn(self,template,**kwargs):
        self.drug.warnings.append(self.format(template,**kwargs))
    def and_list(self,l):
        if len(l) == 2:
            return u' '.join(l[:-1])+u', and '+l[-1]
        if len(l) > 1:
            return u', '.join(l[:-1])+u', and '+l[-1]
        return l[0]
    def format(self,template,**kwargs):
        kwargs.update(
                drug=self.drug.wsa.agent.canonical,
                disease=self.ws.name,
                )
        if self.drug:
            if self.drug.scoreboard:
                kwargs.update(
                    pscr_id=self.drug.scoreboard.scoresrcs.id,
                    )
        return template.format(**kwargs)
