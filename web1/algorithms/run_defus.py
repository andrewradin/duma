#!/usr/bin/env python3

import sys
from path_helper import PathHelper,make_directory

import os
import django
import django_setup

import logging
logger = logging.getLogger("algorithms.run_defus")

from django import forms

import json
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo, StdJobInfo, StdJobForm
import runner.data_catalog as dc
from algorithms.exit_codes import ExitCoder
from browse.models import WsAnnotation
from dtk.files import get_file_records

import pickle

def extract_job_id(p):
    if p.get('molgsig_run', None):
        return int(p['molgsig_run'])
    # PLAT-1716 changed faers_run from a path to a job id
    try:
        return int(p['faers_run'])
    except ValueError:
        parts = p['faers_run'].split('/')
        return int(parts[7])

def gen_smiles(ws):
    from scripts.metasim import load_wsa_smiles_pairs
    # We filter out the MoA collection, it never has SMILES codes.
    # We remap to MoA scores at the end (if appropriate)
    smiles = {x.id:None for x 
              in WsAnnotation.objects.filter(ws_id=ws.id).exclude(agent__collection__name='moa')
             }
    smiles.update(dict(load_wsa_smiles_pairs(ws)))
    return smiles

def gen_faers_data(ws_or_id, faers_run_id, mask_kts):
    from runner.process_info import JobInfo
    faers_bji = JobInfo.get_bound(ws_or_id, faers_run_id)
    cat = faers_bji.get_data_catalog()
    input = {tup[0]:[tup[1]]
             for tup in cat.get_ordering('lrpvalue',True)
            }
    for tup in cat.get_ordering('lrenrichment',True):
        input[tup[0]].append(tup[1])

    if mask_kts:
        ws = faers_bji.ws
        kts = ws.get_wsa_id_set('wseval')
        for kt in kts:
            if kt in input:
                logger.info("Masking out KT %s from inputs", kt)
                del input[kt]
    logger.info("Generated %d KT FAERS inputs", len(input))
    return input

def gen_molgsig_data(ws_or_id, job_id):
    from runner.process_info import JobInfo
    faers_bji = JobInfo.get_bound(ws_or_id, job_id)
    cat = faers_bji.get_data_catalog()
    input = {tup[0]:[0.0]
             for tup in cat.get_ordering('sigCorr',True)
            }
    for tup in cat.get_ordering('sigCorr',True):
        input[tup[0]].append(tup[1])

    logger.info("Generated %d KT MolGSig inputs", len(input))
    return input

def gen_defus_settings(parms):
    settings = {'dpi':parms['p2d_file'],
                'dpi_t':parms['p2d_t'],
                'ppi':parms['p2p_file'],
                'ppi_t':parms['p2p_t'],
                }
    sim_names = [
                 'rdkit',
                 'indigo',
                 'prMax',
                 'dirJac',
                 'indJac',
                 'pathway',
                ]
    RDKIT_ONLY = False 
    if RDKIT_ONLY:
        settings['sim_wts'] = {k:0.0 for k in sim_names}
        settings['sim_wts']['rdkit'] = 1.0
    else:
        settings['sim_wts'] = {k:1.0 for k in sim_names}

    settings['min_sim'] = {k:parms.get(k, 0) for k in sim_names}

    for k, v in settings['min_sim'].items():
        # Disable anything with min_sim>1.0
        if v > 1.0:
            settings['sim_wts'][k] = 0

    # This gets called via target importance, make sure we handle older data
    # that doesn't have this parm set.
    settings['rdkit_use_bit_smiles_fp'] = parms.get('rdkit_use_bit_smiles_fp', False)
    settings['std_gene_list_set'] = parms.get('std_gene_list_set', None)
    settings['d2ps_threshold'] = parms.get('d2ps_threshold', 0)
    settings['d2ps_method'] = parms.get('d2ps_method', None)

    from browse.default_settings import similarity
    settings['sim_db_choice'] = similarity.value(ws=parms.get('ws_id'))

    return settings


class MyJobInfo(StdJobInfo):
    descr = '''
        The main motivation is to extrapolate findings from FAERS data 
        to similar drugs. FAERS information is great, but is only 
        available for clinically approved drugs. DEFUS takes the betas 
        (importance score connecting drug and disease) from FAERS. It 
        multiplies these betas by a measure of similarity between two 
        drugs. These similarity scores are derived from either protein 
        target similarity or chemical structure similarity.

        Target similarity is calculated using the Jaccard index (how 
        much do these two lists overlap compared to how much could 
        overlap?) of 1. direct targets 2. indirect targets 3. PRSim 

        Chemical structure similarity uses 1. bits (Extended 
        Connectivity Fingerprints: for every atom in a small molecule, 
        figure out what other atoms are within two chemical bonds and 
        make that a "bit", then compare all the bits between two 
        molecules), 2. predefined features (indigo: does it have a 
        cyclopentadiene? How about a p-benzoquinone? check like 60 
        different fragment types per molecule and then compare the two 
        molecules)
        '''
    short_label = 'DEFUS'
    page_label = 'DEFUS'

    def make_job_form(self, ws, data):
        from dtk.prot_map import DpiMapping, PpiMapping
        from browse.default_settings import GeneSets
        from dtk.d2ps import D2ps
        dpi_choices = DpiMapping.choices(ws)
        ppi_choices = PpiMapping.choices()
        faers_choices = [(None, 'None')] + ws.get_prev_job_choices('faers')
        initial_faers = faers_choices[1][0] if len(faers_choices) > 1 else None
        molgsig_choices = [(None, 'None')] + ws.get_prev_job_choices('molgsig')
        geneset_choices = GeneSets.choices(ws)
        d2ps_method_choices = D2ps.enabled_method_choices

        class ConfigForm(StdJobForm):
            faers_run = forms.ChoiceField(
                                label='FAERS Job ID to use',
                                choices=faers_choices,
                                required=False,
                                initial=initial_faers,
                                )
            molgsig_run = forms.ChoiceField(
                                label='MolGSig Job ID to use',
                                choices=molgsig_choices,
                                required=False,
                                )
            p2d_file = forms.ChoiceField(
                label='DPI dataset',
                choices=dpi_choices,
                initial=ws.get_dpi_default(),
                help_text="DPI file for looking up targets of outputs.  Can be set to a uniprot dpi file.",
                )
            ref_p2d_file = forms.ChoiceField(
                label='Ref. DPI dataset',
                choices=dpi_choices,
                initial=ws.get_dpi_default(),
                help_text="DPI file for looking up targets of reference/input molecules. Should not be a uniprot dpi file.",
                )
            p2d_t = forms.FloatField(
                label='Min DPI evidence',
                initial=ws.get_dpi_thresh_default(),
                )
            p2p_file = forms.ChoiceField(
                label='PPI Dataset',
                choices=ppi_choices,
                initial=ws.get_ppi_default(),
                )
            p2p_t = forms.FloatField(
                label='Min PPI evidence',
                initial=ws.get_ppi_thresh_default(),
                )
            prMax = forms.FloatField(
                        label = 'PRSim Max min. sim.',
                        initial = 0.0,
                        )
            dirJac = forms.FloatField(
                        label = 'Direct JacSim min. sim.',
                        initial = 0.0,
                        )
            indJac = forms.FloatField(
                        label = 'Indirect JacSim min. sim.',
                        initial = 0.0,
                        )
            indigo = forms.FloatField(
                        label = 'Indigo min. sim.',
                        initial = 0.5,
                        )
            rdkit = forms.FloatField(
                        label = 'RDKit min. sim.',
                        initial = 0.5,
                        )
            pathway = forms.FloatField(
                        label = 'Pathway min. sim.',
                        initial = 0.0,
                        )
            std_gene_list_set = forms.ChoiceField(
                        label='Standard gene list set',
                        initial=GeneSets.value(ws=ws),
                        choices=geneset_choices,
                        )
            d2ps_threshold = forms.FloatField(
                                label = 'Filter out mol-pathway associations under this value',
                                # This helps out runtime a lot, as otherwise we have a lot of weak associations we have to account for.
                                initial = 0.3,
                                help_text='This filters out at the d2ps level before similarity is computed, unlike min sim',
                                )
            d2ps_method = forms.ChoiceField(label='Mol-Pathway connection scorer',
                                initial=D2ps.default_method,
                                choices=d2ps_method_choices,
                                )
            rdkit_use_bit_smiles_fp = forms.BooleanField(
                                label='Use BitSmiles for the RDKit fingerprint',
                                initial=False,
                                required=False,
                                )
            mask_out_kts = forms.BooleanField(
                                label='Mask out known treatments from input',
                                initial=False,
                                required=False,
                                )
        return ConfigForm(ws, data)
            

    def build_role_code(self,jobname,settings_json):
        import json
        d = json.loads(settings_json)
        job_id = extract_job_id(d)
        return self._upstream_role_code(job_id)

    def role_label(self):
        # this doesn't use _upstream_role_label because of the odd way
        # FAERS role labels are constructed
        job_id = extract_job_id(self.job.settings())
        src_bji = self.get_bound(self.ws,job_id)
        return ' '.join([src_bji.role_label(),self.short_label])

    def __init__(self,ws=None,job=None):
        super().__init__(ws=ws, job=job, src=__file__)
        # job-specific properties
        if self.job:
            # input files
            self.inputpickle = self.indir+'input.pkl'
            self.settingspickle = self.indir+'settings.pkl'
            self.smilespickle = self.indir+'smiles.pkl'
            self.dpimappickle = self.indir+'dpimap.pkl'
            # output files
            self.tmp_outfile = self.outdir+'defus.tsv'
            self.tmp_outpkl = self.outdir+'output.pkl.gz'
            self.tmp_outsims = self.outdir+'output-sim.zarr'
            self.outfile = self.lts_abs_root+'defus.tsv'
            self.outfile_unremapped = self.lts_abs_root+'defus_unremapped.tsv'
            self.outpickle = self.lts_abs_root+'output.pkl.gz'
            self.outsims = self.lts_abs_root+'output-sim.zarr'
            # published output files
            if os.path.exists(self.outsims):
                url = f"{self.ws.reverse('rvw:defus_details')}?ref_jid={self.job.id}"
                self.otherlinks = [
                    ('DEFUS Details', url),
                ]
    

    
    def do_wsa_remap(self, all_wsa_ids):
        from dtk.prot_map import DpiMapping
        dpi_map = DpiMapping(self.parms['p2d_file'])
        dpi_type = dpi_map.get_dpi_type()
        if dpi_type != 'moa':
            return

        # Load up outfile, move to outfile_unmapped (if doesn't exist).
        if not os.path.exists(self.outfile_unremapped):
            os.rename(self.outfile, self.outfile_unremapped)

        all_wsa_ids = [int(x) for x in all_wsa_ids]
        from dtk.moa import make_wsa_to_moa_wsa
        logger.info(f"Computing WSA remapping for {len(all_wsa_ids)} wsas")
        wsa2moawsa = make_wsa_to_moa_wsa(all_wsa_ids, pick_canonical=True, dpi_mapping=dpi_map)

        logger.info(f"Found {len(wsa2moawsa)}, applying remapping")
        rows = get_file_records(self.outfile_unremapped, keep_header=True)
        header = next(iter(rows))

        from dtk.parallel import chunker
        # Parse types out of the header; we could get it elsewhere, but we depend on the
        # file structure below anyway.
        # Ordering here is important, it should match the header, as we output in this order too.
        types = [x[0].split('Score')[0] for x in chunker(header[1:], chunk_size=2)]

        from collections import defaultdict
        best = defaultdict(lambda: defaultdict(float))
        connects = defaultdict(lambda: defaultdict(lambda: '-'))
        
        missing_wsas = []
        for row in rows:
            wsa = row[0]
            moawsa = wsa2moawsa.get(int(wsa))
            if not moawsa:
                missing_wsas.append(wsa)
                continue
            
            for type, (score, connection) in zip(types, chunker(row[1:], chunk_size=2)):
                score = float(score)
                if score > best[moawsa][type]:
                    best[moawsa][type] = score
                    connects[moawsa][type] = connection
        
        logger.info(f"Writing remapped output for {len(best)} wsas")
        with open(self.outfile, 'w') as f:
            f.write('\t'.join(header) + '\n')
            for wsa, scores in best.items():
                connection = connects[wsa]
                row = [str(wsa)]
                for type in types:
                    row += [str(scores[type]), connection[type]]
                f.write('\t'.join(row) + '\n') 
        
        if missing_wsas:
            print(f"Missing mapping for {len(missing_wsas)} wsas: (first 10): {missing_wsas[:10]}")


    def get_data_code_groups(self):
        from math import log
        codes = [
            # dirJac is too direct, it's not going to find anything novel, so
            # we don't include it as an efficacy score.
            dc.Code('dirJacScore',label='Direct JacSim', fmt='%0.2f', efficacy=False),
            dc.Code('indJacScore',label='Indirect JacSim', fmt='%0.2f'),
            dc.Code('prMaxScore',label='PRSim Max', fmt='%0.2f'),
            dc.Code('indigoScore',label='Indigo', fmt='%0.2f'),
            dc.Code('rdkitScore',label='RDKit', fmt='%0.2f'),
            dc.Code('pathwayScore',label='Pathway', fmt='%0.2f'),
            dc.Code('defuskey',valtype='str',hidden=True),
            ]
        codetype = self.dpi_codegroup_type('p2d_file')
        return [
                dc.CodeGroup(codetype,self._std_fetcher('outfile'), *codes),
                ]
    def get_target_key(self,wsa):
        cat = self.get_data_catalog()
        try:
            val,_ = cat.get_cell('defuskey',wsa.id)
            return val
        except ValueError:
            return super().get_target_key(wsa)
    def run(self):
        self.make_std_dirs()

        self.run_steps([
            ('wait for resources', self.reserve_step([1])),
            ('new setup', self.new_setup),
            ('wait for remote resources', self.reserve_step(
                    lambda: [0, self.remote_cores_wanted],
                    slow=True,
                    )),
            ('new run remote', self.new_run_remote),
            ('finalize', self.finalize),
        ])
    def run_remote(self):
        remote_cores_to_use = self.reserved[1]
        from pathlib import Path
        options = [
                  Path(self.inputpickle),
                  Path(self.smilespickle),
                  Path(self.dpimappickle),
                  Path(self.settingspickle),
                  Path(self.tmp_outfile),
                  str(remote_cores_to_use),
                  ]
        self.run_remote_cmd('scripts/defus.py', options)
        self._mv_outfile()
        all_wsa_ids = self.smiles.keys()
        self.do_wsa_remap(all_wsa_ids)

    def _mv_outfile(self):
        import shutil
        shutil.move(self.tmp_outfile, self.outfile)
    def _get_ws_agents(self):
        from dtk.prot_map import DpiMapping
        dpi = DpiMapping(self.parms['p2d_file'])
        moa_filter = {'agent__collection__name': 'moa.full'}
        if dpi.get_dpi_type() == 'moa':
            qs = WsAnnotation.objects.filter(ws=self.ws, **moa_filter)
        else:
            qs = WsAnnotation.objects.filter(ws=self.ws).exclude(**moa_filter)
        return qs.values_list("agent_id", flat=True)
    
    def generate_input_data(self, sim_methods=None, ws_agents=None):
        ws_agents = ws_agents if ws_agents is not None else self._get_ws_agents()

        # Grab the ref agent ids and scores.
        ref_scores = self._gen_ref_scores()

        logger.info(f'{len(ref_scores)} ref scores')

        # Map agents to similarity keys (i.e. SMILES or MoA)
        from dtk.metasim import MetaSim, get_sim_methods, StructKey
        from dtk.prot_map import DpiMapping
        sim_methods = sim_methods or get_sim_methods()
        dpi_map = DpiMapping(self.parms['p2d_file'])

        # Remove structural methods when doing uniprot.
        if dpi_map.mapping_type() == 'uniprot':
            sim_methods = [x for x in sim_methods if x.key_type != StructKey]

        logger.info("Making reference keys")
        # We didn't always have a separate reference DPI file, fallback to the one we always had.
        ref_dpi = self.parms.get('ref_p2d_file', self.parms['p2d_file'])
        ref_sim_keys = MetaSim.make_similarity_keys(
            agents=ref_scores.keys(),
            methods=sim_methods,
            dpi_choice=ref_dpi,
            )

        logger.info("Making ws keys")
        ws_sim_keys = MetaSim.make_similarity_keys(
            agents=ws_agents,
            methods=sim_methods,
            dpi_choice=self.parms['p2d_file'],
            )
        
        methods = [x.name for x in sim_methods]

        return {
            'ref_sim_keys': ref_sim_keys,
            'ws_sim_keys': ws_sim_keys,
            'ref_scores': ref_scores,
            'methods': methods,
        }

    def new_setup(self):
        in_data = self.generate_input_data()
        with open(self.inputpickle, 'wb') as handle:
            pickle.dump(in_data, handle)

        self._save_settings()

        max_wanted = 1
        for key_type, keys in in_data['ref_sim_keys'].items():
            max_wanted = max(max_wanted, len(keys) // 10)
            logger.info(f"{len(keys)} ref keys for {key_type}")

        for key_type, keys in in_data['ws_sim_keys'].items():
            logger.info(f"{len(keys)} output keys for {key_type}")

        # Don't try to run defus with very few cores, takes too long.
        self.remote_cores_wanted=(min(max_wanted, 12),max_wanted)
        
    def new_run_remote(self):
        remote_cores_to_use = self.reserved[1]
        from pathlib import Path
        options = [
                  Path(self.inputpickle),
                  Path(self.settingspickle),
                  Path(self.tmp_outpkl),
                  str(remote_cores_to_use),
                  '--out-sims', Path(self.tmp_outsims),
                  ]
        self.run_remote_cmd('scripts/newdefus.py', options, local=False)
        os.replace(self.tmp_outpkl, self.outpickle)
        os.replace(self.tmp_outsims, self.outsims)

        self.write_results_file()
        
    def write_results_file(self):
        import isal.igzip as gzip
        import pickle
        with gzip.open(self.outpickle, 'rb') as f:
            data = pickle.load(f)
        scores = data['scores']
        connections = data['connections']

        from dtk.prot_map import DpiMapping
        dpi = DpiMapping(self.parms['p2d_file'])
        codetype = dpi.mapping_type()

        settings = gen_defus_settings(self.parms)

        conn_agent2wsa = dict(WsAnnotation.all_objects.filter(ws=self.ws).values_list('agent_id', 'id'))

        if codetype == 'uniprot':
            # For uniprots our keys are just uniprots (though connections are still agents).
            agent2wsa = {x:x for x in dpi.get_uniq_target()}
        else:
            agent2wsa = conn_agent2wsa

        header = [codetype]
        header += [x for t in settings['sim_wts'] for x in [t+'Score', t+'ConnectingDrug']]
        with open(self.outfile, 'w') as f:
            f.write("\t".join(header) + "\n")
            for agent_id, score_type_d in scores.items():
                if agent_id not in agent2wsa:
                    logger.warning(f"No wsa for agent {agent_id}")
                    continue
                out = [agent2wsa[agent_id]]
                for st in settings['sim_wts']:
                    if st in connections[agent_id]:
                        v = conn_agent2wsa[connections[agent_id][st]]
                        n = score_type_d[st]
                    else:
                        v = '-'
                        n = 0.0
                    out += [n, v]
                f.write("\t".join([str(x) for x in out]) + "\n")

    def setup(self):
        self._pickle_faers_data()
        self._save_settings()
        self.smiles = self._load_smiles()
        self._load_dpi()
        # We batch up drugs into sets of 500, use that as the max core count.
        max_wanted = int(3 + len(self.smiles) // 500)
        # Don't try to run defus with very few cores, takes too long.
        self.remote_cores_wanted=(12,max_wanted)
    def _save_settings(self):
        settings = gen_defus_settings(self.parms)
        with open(self.settingspickle, 'wb') as handle:
            pickle.dump(settings, handle)
    def _load_dpi(self):
        from dtk.prot_map import DpiMapping
        # Always want baseline DPI for defus internals - it works on molecules.
        # If MoA dpi is selected, outputs will later be remapped to MoAs.
        dpi_obj = DpiMapping(self.parms['p2d_file']).get_baseline_dpi()
        dpi_map = dpi_obj.get_wsa_id_map(self.ws)
        with open(self.dpimappickle, 'wb') as handle:
            pickle.dump(dpi_map, handle)


    def _load_smiles(self):
        codetype = self.dpi_codegroup_type('p2d_file')
        if codetype == 'uniprot':
            # NOTE: defus uniprot scoring doesn't quite work yet.
            # The problem is that in metasim we need to use two different
            # DPI files, one with molecules (for faers) and the other with
            # protein-mols (for this).
            from dtk.prot_map import DpiMapping
            dpi_obj = DpiMapping(self.parms['p2d_file'])
            dpi_map = dpi_obj.get_wsa_id_map(self.ws)
            all_prots = set()
            for prots in dpi_map.values():
                all_prots.update(prots)
            smiles = {prot: None for prot in all_prots}
        else:
            smiles = gen_smiles(self.ws)
        with open(self.smilespickle, 'wb') as handle:
            pickle.dump(smiles, handle)
        return smiles
    def _pickle_faers_data(self):
        if self.parms.get('molgsig_run', None):
            faers_data = gen_molgsig_data(self.ws, self.parms['molgsig_run'])
        else:
            faers_data = gen_faers_data(self.ws, self.parms['faers_run'], self.parms.get('mask_out_kts', False))
        with open(self.inputpickle, 'wb') as handle:
            pickle.dump(faers_data, handle)
        self.input_len = len(faers_data)
    def _gen_ref_scores(self):
        if self.parms.get('molgsig_run', None):
            faers_data = gen_molgsig_data(self.ws, self.parms['molgsig_run'])
        else:
            faers_data = gen_faers_data(self.ws, self.parms['faers_run'], self.parms.get('mask_out_kts', False))
        
        wsa2agent = dict(WsAnnotation.all_objects.filter(pk__in=faers_data.keys()).values_list('id', 'agent_id'))
        return {wsa2agent[wsa]: value for (wsa, value) in faers_data.items()}

    def get_warnings(self):
        return super().get_warnings(
                 ignore_conditions=self.base_warning_ignore_conditions+[
                        # Seems to be an invalid std_smiles in our db.
                        lambda x:'Error computing similarity for smiles [H]C(O)(CO)C([H])(O)C([H])(O)C([H])(=O)CO' in x,
                        lambda x:'WARNING: not removing hydrogen atom without neighbors' in x,
                         ],
                )

    def add_workflow_parts(self,ws,parts):
        uji = self # simplify access inside MyWorkflowPart
        class MyWorkflowPart:
            def __init__(self,label,cds):
                self.label=label
                self.cds=cds
                # Note only the FAERS cds has a data status
                self.enabled_default=uji.data_status_ok(
                        ws,
                        'Faers',
                        'Complete Clinical Values',
                        ) if self.cds.startswith('faers.v') else False
            def add_to_workflow(self,wf):
                cm_info = wf.std_wf.get_main_cm_info(uji.job_type)
                faers_name = cm_info.pre.add_pre_steps(wf,self.cds)
                assert faers_name
                from dtk.workflow import DefusStep
                my_name = faers_name+'_'+uji.job_type
                DefusStep(wf,my_name,
                        inputs={faers_name:True},
                        )
                cm_info.post.add_post_steps(wf,my_name)
        for choice in ws.get_cds_choices():
            parts.append(MyWorkflowPart(
                    choice[1]+' '+self.short_label,
                    choice[0],
                    ))

if __name__ == "__main__":
    MyJobInfo.execute(logger)
