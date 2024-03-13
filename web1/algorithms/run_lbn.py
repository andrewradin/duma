#!/usr/bin/env python3

from __future__ import print_function
import sys
import six
# retrieve the path where this program is located, and
# use it to construct the path of the website_root directory
try:
    from path_helper import PathHelper,make_directory
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")
    from path_helper import PathHelper,make_directory

import os
import django
if 'django.core' not in sys.modules:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    django.setup()

from django import forms
from django.http import HttpResponseRedirect

from tools import ProgressWriter
from runner.process_info import JobInfo
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager
import runner.data_catalog as dc
from dtk.prot_map import DpiMapping

import json
import logging
logger = logging.getLogger("algorithms.run_lbn")

class ConfigForm(forms.Form):
    job_id = forms.IntegerField(
                        label='Drug Ordering Job',
                        )
    score = forms.CharField(
                        label='Drug Ordering Score',
                        )
    start = forms.IntegerField(
                        label='Initial Drugs to skip',
                        initial=0,
                        )
    count = forms.IntegerField(
                        label='Drugs to examine',
                        initial=200,
                        )
    condensed = forms.BooleanField(
            label='Count via condensed',
            initial=True,
            required=False,
            )
    add_drugs = forms.ChoiceField(label='Additional Drugs')
    dpi_file = forms.ChoiceField(label='DPI dataset')
    dpi_t = forms.FloatField(label='Min DPI evidence')
    _subtype_name = "job_subtype"
    def __init__(self, ws, copy_job, *args, **kwargs):
        super(ConfigForm,self).__init__(*args, **kwargs)
        self.ws = ws
        # reload choices on each form load
        f = self.fields['add_drugs']
        f.choices = [
                ('none','None'),
                ]+self.ws.get_wsa_id_set_choices()
        f.initial = 'none' #self.ws.eval_drugset
        f = self.fields['dpi_file']
        f.choices = DpiMapping.choices(ws)
        f.initial = self.ws.get_dpi_default()
        # use special high-confidence default threshold
        self.fields['dpi_t'].initial = 0.9 #self.ws.get_dpi_thresh_default()
        if copy_job:
            self.from_json(copy_job.settings_json)
    
    def as_html(self):
        from django.utils.html import format_html
        return format_html('''
                <div class="well">
                Reset PubMed Cache:
                <button name='reset_cache_btn' type='submit'>Reset</button>
                </div>
                <input name="{}" type="hidden" value="{}"/>
                <table>{}</table>
                '''
                ,self._subtype_name
                ,None
                ,self.as_table()
                )
        
    def as_dict(self):
        # this returns the settings_json for this form; it may differ
        # from what appears in the user interface; there are 2 use
        # cases:
        # - extracting default initial values from an unbound form
        #   for rendering the settings comparison column
        # - extracting user-supplied values from a bound form
        if self.is_bound:
            src = self.cleaned_data
        else:
            src = {fld.name:fld.field.initial for fld in self}
        p ={}
        for f in self:
            key = f.name
            value = src[key]
            p[key] = value
        return p
    def from_json(self,init):
        p = json.loads(init)
        for f in self:
            if f.name in p:
                f.field.initial = p[f.name]

class MyJobInfo(JobInfo):
    def description_text(self):
        return '''
        <b>Literature Based Novelty</b> report the enrichment of the
        drug and disease co-occurance in Pubmed.
        '''
    def settings_defaults(self,ws):
        cfg=ConfigForm(ws,None)
        return {
                'default':cfg.as_dict(),
                }
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        form = ConfigForm(ws, copy_job)
        return form.as_html()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        form = ConfigForm(ws, None,post_data)
        if not form.is_valid():
            return (form.as_html(),None)

        settings = form.as_dict()
        settings['ws_id'] = ws.id
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(settings)
                            )
        next_url = ws.reverse('nav_progress',job_id)
        return (None, next_url)

    def get_buttons(self):
        return [{
                    'name': 'reset_cache',
                    'action': lambda: self.reset_cache()
                }]

    def reset_cache(self):
        from dtk.entrez_utils import EClientWrapper
        logger.info("Clearing eutils cache")
        EClientWrapper.clear_cache()
        return HttpResponseRedirect('')

    def role_label(self):
        return self._upstream_role_label()

    def get_data_code_groups(self):
        return [
                dc.CodeGroup('wsa',self._std_fetcher('outfile'),
                        dc.Code('lbnOR',
                                    label='Odds Ratio',
                                    efficacy=False,
                                    novelty=True,
                                    fmt='%0.2f',
                                    ),
                        dc.Code('lbnP',
                                    efficacy=False,
                                    novelty=True,
                                    label='p-val',
                                    fmt='%0.4e',
                                    ),
                        dc.Code("disPorWDrug",
                                    label='Disease portion w/drug',
                                    efficacy=False,
                                    novelty=True,
                                    fmt='%0.4e',
                                    ),
                        dc.Code("drugPorWDis",
                                    label='Drug portion w/disease',
                                    efficacy=False,
                                    novelty=True,
                                    fmt='%0.4e',
                                    ),
                        dc.Code("targLogOdds",
                                    label='Best target log-odds',
                                    efficacy=False,
                                    novelty=True,
                                    fmt='%0.4e',
                                    ),
                        dc.Code("targPortion",
                                    label='Best target portion w/disease',
                                    efficacy=False,
                                    novelty=True,
                                    fmt='%0.4e',
                                    ),
                        ),
                dc.CodeGroup('uniprot',self._std_fetcher('target_outfile'),
                        dc.Code('lbnTargOR',
                                    label='Target Odds Ratio',
                                    efficacy=False,
                                    novelty=True,
                                    fmt='%0.2f',
                                    ),
                        dc.Code('lbnTargP',
                                    efficacy=False,
                                    novelty=True,
                                    label='Target p-val',
                                    fmt='%0.4e',
                                    ),
                        dc.Code("disPorWTarg",
                                    label='Disease portion w/targ',
                                    efficacy=False,
                                    novelty=True,
                                    fmt='%0.4e',
                                    ),
                        dc.Code("targPorWDis",
                                    label='Targ portion w/disease',
                                    efficacy=False,
                                    novelty=True,
                                    fmt='%0.4e',
                                    ),
                        ),
                ]
    def __init__(self,ws=None,job=None):
        # base class init
        self.use_LTS=True
        super(MyJobInfo,self).__init__(
                    ws,
                    job,
                    __file__,
                    "Lit-based Novelty",
                    "Literature-based Novelty",
                    )
        # any base class overrides for unbound instances go here
        self.publinks = (
                )
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            self.ec = ExitCoder()
            self.outfile = os.path.join(self.lts_abs_root, 'lbn.tsv')
            self.target_outfile = os.path.join(self.lts_abs_root, 'target_lbn.tsv')
    def get_warnings(self):
        return super().get_warnings(
                 ignore_conditions=self.base_warning_ignore_conditions+[
                        # eutils limits results, but we're only counting anyhow.
                        # so no need to be notified about truncation
                         lambda x:'; see https://github.com/biocommons/eutils/issues/124/' in x,
                         ],
                )
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        make_directory(self.lts_abs_root)
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                "calculating novelty",
                "clean up"
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.run_lbn()
        p_wr.put("calculating novelty", 'complete')
        self.finalize()
        p_wr.put("clean up","complete")
        
    def run_lbn(self):
        from dtk.entrez_utils import EClientWrapper
        EClientWrapper.expire_old_cache_data()

        disease = self.ws.name
        from flagging.utils import get_target_wsa_ids,show_list_progress
        wsa_ids = get_target_wsa_ids(
                                     self.ws,
                                     self.parms['job_id'], 
                                     self.parms['score'],
                                     self.parms['start'],
                                     self.parms['count'],
                                     self.parms['condensed'],
                                     )
        if self.parms['add_drugs'] != 'none':
            print(len(wsa_ids),'drugs within range')
            wsa_ids = set(wsa_ids)
            wsa_ids |= self.ws.get_wsa_id_set(self.parms['add_drugs'])
            print(len(wsa_ids),'drugs including',self.parms['add_drugs'])
        from browse.models import WsAnnotation
        wsas = list(WsAnnotation.objects.filter(pk__in=wsa_ids))
        target_dict = self.calc_target_novelty(wsas)
        res_dict = calculate_novelty(
                self.ws.name,
                show_list_progress(wsas),
                )
        wsa_lookup={x.id:x for x in wsas}
        with open(self.outfile, 'w') as o:
            header = ["wsa",
                    "lbnOR",
                    "lbnP",
                    "disPorWDrug",
                    "drugPorWDis",
                    "targLogOdds",
                    "targPortion",
                    ]
            o.write("\t".join(header))
            o.write("\n")
            for key,val in six.iteritems(res_dict):
                wsa = wsa_lookup[key]
                o.write("\t".join([str(key), 
                                    str(val['log_odds']), 
                                    str(val['pvalue']),
                                    str(val['por_o_disease_w_drug']),
                                    str(val['por_o_drug_w_disease']),
                                    str(wsa.target_log_odds),
                                    str(wsa.target_portion),
                                    '\n']))
        with open(self.target_outfile, 'w') as o:
            header = ["uniprot",
                    "lbnTargOR",
                    "lbnTargP",
                    "disPorWTarg",
                    "targPorWDis",
                    ]
            o.write("\t".join(header))
            o.write("\n")
            for key,val in six.iteritems(target_dict):
                o.write("\t".join([str(key), 
                                    str(val['log_odds']), 
                                    str(val['pvalue']),
                                    str(val['por_o_disease_w_gene']),
                                    str(val['por_o_gene_w_disease']),
                                    '\n']))

    def get_uniprot_to_gene(self, wsas):
        # get collection keys for each wsa
        from dtk.prot_map import DpiMapping
        dm = DpiMapping(self.parms['dpi_file'])
        from dtk.data import MultiMap
        mm = MultiMap(
                (wsa_id,key)
                for key,wsa_id in dm.get_key_wsa_pairs(self.ws)
                )
        all_collection_keys = set()
        for wsa in wsas:
            wsa.collection_keys = mm.fwd_map().get(wsa.id,set())
            all_collection_keys |= wsa.collection_keys
        print('Got',len(all_collection_keys),'collection keys')
        # get dpi bindings
        collkey2uniprot = MultiMap(
                (binding[0],binding[1])
                for binding in dm.get_dpi_info_for_keys(
                        all_collection_keys,
                        min_evid = self.parms['dpi_t'],
                        )
                )
        bindings = sum([len(x) for x in collkey2uniprot.fwd_map().values()])
        print('Got',bindings,'DPI bindings')
        # get gene names for all uniprots
        all_uniprots = set(collkey2uniprot.rev_map().keys())
        from browse.models import Protein
        uniprot2gene = Protein.get_uniprot_gene_map(all_uniprots)
        return uniprot2gene, collkey2uniprot

    def calc_target_novelty(self,wsas,max_pvalue=0.05):
        print('Processing',len(wsas),'drugs')
        uniprot2gene, collkey2uniprot = self.get_uniprot_to_gene(wsas)
        # get pubmed counts for each gene
        from dtk.entrez_utils import PubMedSearch
        from eutils.exceptions import EutilsNCBIError
        from time import sleep
        pubmedsearch = PubMedSearch()
        disease = self.ws.name
        total = pubmedsearch.size()
        disease_cnt = pubmedsearch.count_frequency([disease])
        genes = set(uniprot2gene.values())
        print('Scanning pubmed for',len(genes),'genes')
        from flagging.utils import show_list_progress
        gene2counts = {}
        import numpy as np
        for gene in show_list_progress(genes):
            gene_cnt = pubmedsearch.count_frequency([gene])
            if gene_cnt:
                both_cnt = pubmedsearch.count_frequency([gene, disease])
                cm = np.array([
                    [ both_cnt, disease_cnt-both_cnt ],
                    [ gene_cnt-both_cnt, total-disease_cnt-gene_cnt+both_cnt],
                    ])
                log_odds,pvalue = contingency2odds_pv(cm)
                if pvalue > max_pvalue:
                    continue
                gene2counts[gene] = dict(
                        contingency_matrix=cm,
                        log_odds=log_odds,
                        pvalue=pvalue,
                        por_o_disease_w_gene=outcomes2portion(cm[0,:]),
                        por_o_gene_w_disease=outcomes2portion(cm[:,0]),
                        )
        print('Found',len(gene2counts),'significant gene associations')
        if True:
            sort_key='log_odds'
            for gene,d in sorted(
                    six.iteritems(gene2counts),
                    key=lambda x:-x[1][sort_key],
                    ):
                print('  ',gene,d[sort_key],d['contingency_matrix'].tolist())
        # assign score to each wsa
        for wsa in wsas:
            scores = {}
            uniprots = {}
            for collkey in wsa.collection_keys:
                for uniprot in collkey2uniprot.fwd_map().get(collkey,[]):
                    try:
                        gene = uniprot2gene[uniprot]
                    except KeyError:
                        continue
                    try:
                        d = gene2counts[gene]
                    except KeyError:
                        continue
                    scores[gene] = d
                    uniprots.setdefault(gene,set()).add(uniprot)
            if False:
                for gene in scores:
                    print(gene,scores[gene],uniprots[gene])
            for attr,key in (
                    ('target_log_odds','log_odds'),
                    ('target_portion','por_o_disease_w_gene'),
                    ):
                try:
                    v = max(d[key] for d in scores.values())
                except ValueError:
                    v = 0
                setattr(wsa,attr,v)

        from dtk.data import MultiMap
        gene2uniprot = MultiMap((v, k) for k, v in six.iteritems(uniprot2gene)).fwd_map()
        uniprot2counts = {}
        for gene, data in six.iteritems(gene2counts):
            for uniprot in gene2uniprot[gene]:
                uniprot2counts[uniprot] = gene2counts[gene]
        return uniprot2counts

def calculate_novelty(disease,wsas):
    import numpy as np
    novelty_dict = {}
    from dtk.entrez_utils import PubMedSearch
    pubmedsearch = PubMedSearch()
    total = pubmedsearch.size()
    disease_cnt = pubmedsearch.count_frequency([disease])
    print('Beginning novelty calculations...')
    for wsa in wsas:
        name = wsa.get_name(False)
        drug_cnt = pubmedsearch.count_frequency([name])
        if drug_cnt:
            both_cnt = pubmedsearch.count_frequency([name, disease])
        else:
            both_cnt = 0
        # assemble contingency matrix
        cm = np.array([
                [ both_cnt, disease_cnt-both_cnt ],
                [ drug_cnt-both_cnt, total-disease_cnt-drug_cnt+both_cnt],
                ])
        log_odds,pvalue = contingency2odds_pv(cm)
        novelty_dict[wsa.id] = dict(
                drug_name=name,
                disease_name=disease,
                contingency_matrix=cm,
                log_odds=log_odds,
                pvalue=pvalue,
                por_o_disease_w_drug=outcomes2portion(cm[0,:]),
                por_o_drug_w_disease=outcomes2portion(cm[:,0]),
                )
    return novelty_dict

def contingency2odds_pv(cm,odds_ratio_min=0.0001, odds_ratio_max=10000.):
    # return log-odds and pvalue for a contingency matrix
    import scipy.stats as stats
    import numpy as np
    odds_ratio, pvalue = stats.fisher_exact(cm)
    if np.isnan(odds_ratio):
        odds_ratio = 0
    # prevent domain error at zero, while staying in a reasonable range
    if odds_ratio > odds_ratio_max:
        odds_ratio = odds_ratio_max
    # prevent domain error at zero, while staying in a reasonable range
    elif odds_ratio < odds_ratio_min:
        odds_ratio = odds_ratio_min
    log_odds = np.log(odds_ratio)
    return log_odds,pvalue

def outcomes2portion(outcomes,idx=0):
    # return fraction of trials with a particular outcome
    total = sum(outcomes)
    if total == 0:
        return 0
    return float(outcomes[idx])/total

if __name__ == "__main__":
    MyJobInfo.execute(logger)
