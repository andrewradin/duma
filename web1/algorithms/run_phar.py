#!/usr/bin/env python3

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

import subprocess
import json
from tools import ProgressWriter
from reserve import ResourceManager

import logging
logger = logging.getLogger("algorithms.run_phar")

from runner.process_info import JobInfo

################################################################################
debug = True
# Step-by-step plug in guide
#
# Also see the comments in the JobInfo base class definition in
# runner/process_info.py.
################################################################################
# 10) XXX call check_enrichment() and finalize() from inside run() to run DEA, and
#     to copy any published files into their final position
# 11) XXX implement get_data_code_groups() to return score and feature vector
#     results
# XXX I have not yest called check_inrichment() from inside run()
# XXX I have not yet implemented get_data_code_groups to return score and feature vector

class ConfigForm(forms.Form):
    drug_set = forms.ChoiceField(label='Drug Set',initial=''
                        ,choices=(('','None'),)
                        )
    make_global_drugset = forms.BooleanField(initial=False
            , label='Make global 3D drugset (takes 30 minutes, drugspace agnostic)'
            , required=False
            )
    score_type = forms.ChoiceField(label='Score Type',initial='Tversky'
                        ,choices=(('tversky','Tversky, no penalty for mismatched phars'),
                            ('tanimoto','Tanimoto, penalize for mismatch phars'),)
                        )
    def __init__(self, ws, copy_job, *args, **kwargs):
        super(ConfigForm,self).__init__(*args, **kwargs)
        self.ws = ws
        # ...then drugset
        f = self.fields['drug_set']
        #f = self.fields[]
        f.choices = self.ws.get_wsa_id_set_choices()
        f.initial = f.choices[0][0]
        f.label = 'Drug Set'
        if copy_job:
            self.from_json(copy_job.settings_json)
    def as_html(self):
        from django.utils.html import format_html
        return format_html('<h4>{}</h4>{}'
                ,'Similarity scores will compare to:'
                ,self.as_p()
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
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        form = ConfigForm(ws, copy_job)
        return form.as_html()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        form = ConfigForm(ws, None, post_data)
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
    def get_data_code_groups(self):
        import runner.data_catalog as dc
        import dtk.readtext as rt
        sim_cps = []
        mb_cps = []
        if self.job:
            # for a bound instance, load the _cps arrays with
            # the data columns that are actually available in
            # the files
            try:
                sim_wsa_ids = []
                with open(self.fn_scores_to_export) as f:
                    header = next(rt.parse_delim(f,delim=','))
                    for colname in header[1:]:
                        assert colname.startswith('like_')
                        sim_wsa_ids.append(int(colname[5:]))
                name_map = self.ws.get_wsa2name_map()
                for wsa_id in sim_wsa_ids:
                    colname = 'like%d'%wsa_id
                    drugname = name_map[wsa_id]
                    sim_cps.append(dc.Code(colname,label='like '+drugname))
            except IOError:
                pass
        result = []
        if sim_cps or not self.job:
            def my_fetcher(keyset):
                f = open(self.fn_scores_to_export)
                g = rt.parse_delim(f,delim=',')
                next(g) # skip header
                return rt.dc_file_fetcher(keyset,g,key_mapper=int)
            def argsum(*args): return sum(args)
            result += [
                    dc.CodeGroup('wsa',my_fetcher,
                            dc.Code('ktsim',
                                    calc=(argsum,'similarity'),
                                    label="Overall KT similarity",
                                    ),
                            dc.Code('similarity',valtype='alias',
                                    label="Per-KT similarity scores",
                                    ),
                            *sim_cps
                            )
                    ]
        return result
    def __init__(self,ws=None,job=None):
        # base class init
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "Pharmacophore",
                "Pharmacophore 3D Alignment Scoring",
                )
        # any base class overrides for unbound instances go here
        self.publinks = (
                ('Score Distribution','scores_graph.png'),
                ('ATC Enrichment', 'atc_enrichment_graph.png'),
                ('Scores.csv','scores_3D.csv')
                )
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            # gloabale variable
            self.known_treatment_ids = None
            # input files
            self.fn_smiles = self.indir+"smiles.smi"
            self.fn_treatments = self.indir+"treatments.tsv"
            # R script to run align-it
            self.fn_align = "scripts/phar_align.R"
            self.fn_graph = "scripts/phar_graph.R"
            # intermediate files
            self.fn_3D_smiles = self.outdir+'drugset_3D.sdf'
            self.phars = 'drugset_3D.phar'
            self.fn_phars_out = self.outdir+self.phars
            self.fn_phars_tmp = self.indir+self.phars
            self.fn_3D_warnings = self.outdir+'3D_warnings.txt'
            self.fn_db_to_wsa = self.indir+'db_to_wsa.csv'
            # published output files
            self.fn_atc = self.indir+'wsa_to_atc.csv'
            self.fn_atc_graph = self.tmp_pubdir+'atc_enrichment_graph.png'
            self.fn_scores_graph = self.tmp_pubdir+'scores_graph.png'
            self.fn_scores_to_graph = self.outdir+'scores_3D_tidy_format.csv'
            self.fn_scores_to_export = self.outdir+'scores_3D.csv'
    def write_treatments_file(self):
        self.known_treatment_ids = [str(x) for x in self.ws.get_wsa_id_set(self.parms['drug_set'])]
        self.kt_count = len(self.known_treatment_ids)
        self.info('kt_count %d',self.kt_count)
        if not self.kt_count:
            self.fatal('No qualifying known treatments found')
        with open(self.fn_treatments,'w') as f:
            wsa_to_db = dict(self.ws.wsa_prop_pairs('drugbank_id'))
            i = 0
            for k,v in six.iteritems(wsa_to_db):
                wsa_to_db[k] = v + '\n'
            for kt in self.known_treatment_ids:
                f.write(wsa_to_db.get(int(kt),''))
    def make_drugset_phars(self):
        if self.parms['make_global_drugset']:
            wsa_to_db = dict(self.ws.wsa_prop_pairs('drugbank_id'))
            # get smiles codes with current wsa id
            from dtk.data import MultiMap
            mm = MultiMap(self.ws.wsa_prop_pairs('smiles_code'))
            # retrieve smiles codes and convert to 3D structure (.sdf)
                #ran obabel individually to skip seg faults
            with open(self.fn_3D_smiles,'w') as sdf:
                for k,v in MultiMap.flatten(mm.fwd_map()):
                    drug_bank_id = wsa_to_db[k]
                    smi_path = self.indir + drug_bank_id + '.txt'
                    sdf_path = self.indir
                    with open(smi_path, 'w') as f:
                        f.write("%s\t%s\n" % (v,wsa_to_db[k]))
                    cmd = ['obabel', '-ismi', smi_path, '-osdf',sdf_path, '--gen3D']
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
                    out, err = p.communicate()
                    if len(out) != 0:
                        sdf.write(out)
            # convert 3D .sdf to pharmacophore file with Align-it
            subprocess.call(['align-it', '-d', self.fn_3D_smiles, '--dbType', 'SDF',
                             '-p', self.fn_phars_tmp])
            subprocess.call(['s3cmd','put',self.fn_phars_tmp,'s3://duma-datasets/'])
        else:
            subprocess.call(['s3cmd','get','s3://duma-datasets/'+self.phars,
                self.fn_phars_tmp])

    def find_similarities(self):
        # doing it on remote server
        rp = self.mch.get_full_remote_path
        wsa_to_db = dict(self.ws.wsa_prop_pairs('drugbank_id'))
        with open(self.fn_db_to_wsa, 'w') as f:
            for k,v in six.iteritems(wsa_to_db):
                f.write(','.join([v,str(k)])+'\n')
        self.copy_input_to_remote()
        self.make_remote_directories([
            self.outdir,
            self.tmp_pubdir,
            ])
        script = rp(PathHelper.website_root+self.fn_align)
        script_with_args = " ".join([script,
            '--phars', rp(self.fn_phars_tmp),
            '--input_treat', rp(self.fn_treatments),
            '--graph', rp(self.fn_scores_to_graph),
            '--input_dir', rp(self.indir),
            '--num_cores', str(self.remote_cores_got),
            '--db_to_wsa', self.fn_db_to_wsa,
            '--scores', self.fn_scores_to_export,
            '--score_type', self.parms['score_type']])
        self.mch.check_remote_cmd(script_with_args)
        self.copy_output_from_remote()
    def generate_graph(self):
        atc_codes = dict(self.ws.wsa_prop_pairs('atc'))
        with open(self.fn_atc,'w') as f:
            for k,v in six.iteritems(atc_codes):
                f.write(','.join([str(k),v])+'\n')
        subprocess.check_call([self.fn_graph,
            '-i', self.fn_scores_to_export , '-o',
            self.fn_scores_graph, '--atc', self.fn_atc, '--graph', self.fn_atc_graph])
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        p_wr = ProgressWriter(self.progress
                , [ "wait for resources"
                  , "setup"
                  , "wait for remote resources"
                  , "calculate similarities"
                  , "check enrichment"
                  , "generate graph"
                  ]
                )
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.write_treatments_file()
        self.make_drugset_phars()
        p_wr.put("setup","complete")
        if self.kt_count < 35:
            self.remote_cores_wanted=(1,self.kt_count+1) # up to known treatment count
        else:
            self.remote_cores_wanted=(1,0) # as many as possible
        got = self.rm.wait_for_resources(self.job.id,[
                                    0,
                                    self.remote_cores_wanted,
                                    ])
        self.remote_cores_got = got[1]
        p_wr.put("wait for remote resources"
                        ,"complete; got %d cores"%self.remote_cores_got
                        )
        self.find_similarities()
        p_wr.put("calculate similarities","complete")
        self.check_enrichment()
        p_wr.put("check enrichment","complete")
        self.rm.wait_for_resources(self.job.id,[1])
        self.generate_graph()
        self.finalize()
        p_wr.put("generate graph","complete")
    def get_kt_similarity_matrix(self):
        '''Return a similarity matrix from a struct job.
        '''
        cat=self.get_data_catalog()
        # get codes of similarity scores
        cols = [
                code
                for code in cat.get_codes('wsa','score',include_hidden=True)
                if code.startswith('like')
                ]
        if not cols:
            return None
        # extract wsa_ids from similarity scores
        ordered_wsa_ids = [int(x[4:]) for x in cols]
        # fetch feature vectors; only keep those for treatments
        kt_set = set(ordered_wsa_ids)
        _,gen = cat.get_feature_vectors(*cols)
        rows = {
                wsa:vec
                for wsa,vec in gen
                if wsa in kt_set
                }
        # make the row order match the column order
        matrix = [
                rows[key]
                for key in ordered_wsa_ids
                ]
        from dtk.similarity import SimilarityMatrix
        return SimilarityMatrix(ordered_wsa_ids,matrix)

if __name__ == "__main__":
    MyJobInfo.execute(logger)
