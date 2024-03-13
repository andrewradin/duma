#!/usr/bin/env python3

from __future__ import print_function
from builtins import range
import sys
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
logger = logging.getLogger("algorithms.run_struct")

from runner.process_info import JobInfo

class ConfigForm(forms.Form):
    drug_set = forms.ChoiceField(label='Drug Set',initial=''
                        ,choices=(('','None'),)
                        )
    package = forms.ChoiceField(label='Package',initial='RDKit'
                        ,choices=(('rdkit','RDKit'),('indigo','Indigo'),)
                        )
    fingerprint = forms.IntegerField(label='RDKit only: Fingerprint radius',initial=2)
    bits = forms.BooleanField(initial=True
                            ,label='RDKit only: Report bits'
                            ,required=False
                            )
    bitslr = forms.BooleanField(initial=True
                            ,label='RDKit only: LR score bits'
                            ,required=False
                            )
    min_portion = forms.FloatField(initial=0.01
                            ,label='Min bit portion'
                            ,required=False
                            )
    def __init__(self, ws, copy_job, *args, **kwargs):
        super(ConfigForm,self).__init__(*args, **kwargs)
        self.ws = ws
        # ...then drugset
        f = self.fields['drug_set']
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
        p ={'ws_id':self.ws.id}
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
    def settings_defaults(self,ws):
        cfg=ConfigForm(ws,None)
        return {
                'default':cfg.as_dict(),
                }
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        form = ConfigForm(ws, copy_job)
        return form.as_html()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        form = ConfigForm(ws, None, post_data)
        if not form.is_valid():
            return (form.as_html(),None)
        settings = form.as_dict()
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
                with open(self.fn_similar) as f:
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
            try:
                with open(self.fn_bits) as f:
                    header = next(rt.parse_delim(f,delim=','))
                    # for legacy reasons, the molecule bits file ends every
                    # column name with a _<int> to guarantee uniqueness;
                    # strip that here and use the SMILES part of the name
                    # as the feature vector label
                    import re
                    m = re.compile(r'(.*)_[0-9]+$')
                    mb_cps = [
                            dc.Code('bit%d'%x,
                                        valtype='bool',
                                        fv_label=m.match(header[x]).group(1),
                                        )
                            for x in range(1,len(header))
                            ]
            except IOError:
                pass
        result = []
        # return actual results, or dummy columns if unbound instance
        result += [
                dc.CodeGroup('wsa',self._std_fetcher('fn_bitslr'),
                        dc.Code('ktsimlr',label='LR KT Similarity'),
                        ),
                ]
        # pylint: disable=function-redefined
        if sim_cps or not self.job:
            def my_fetcher(keyset):
                f = open(self.fn_similar)
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
        if mb_cps or not self.job:
            def my_fetcher(keyset):
                f = open(self.fn_bits)
                g = rt.parse_delim(f,delim=',')
                next(g) # skip header
                return rt.dc_file_fetcher(keyset,g,
                            key_mapper=int,
                            data_mapper=lambda x:x=='True',
                            )
            result += [
                    dc.CodeGroup('wsa',my_fetcher,
                            dc.Code('bits',valtype='alias',
                                    label="Molecule Bits",
                                    efficacy=False,
                                    ),
                            *mb_cps
                            )
                    ]
        return result
    def __init__(self,ws=None,job=None):
        # base class init
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "Structure",
                "Molecular Structure Scoring",
                )
        # any base class overrides for unbound instances go here
        self.publinks = (
                ('KT Similarity','KTSimilarityHeatmap.png'),
                )
        # job-specific properties
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
            # files marshalled for external processing
            self.fn_smiles = self.indir+"smiles.tsv"
            self.fn_treatments = self.indir+"treatments.tsv"
            # output files
            self.fn_similar = self.outdir+"similarities.csv"
            self.fn_bits = self.outdir+"moleculeBits.csv"
            self.fn_bitslr = self.outdir+"lr_score.tsv"
            # intermediate files
            self.fn_heat_inp = self.root+"forKTSimilarityHeatmap.csv"
            self.fn_atc = self.indir+'wsa_to_atc.csv'
            # published output files
            self.fn_heat_outp = self.tmp_pubdir+"KTSimilarityHeatmap.png"
    def write_smiles_file(self):
        # a smiles code followed by an identifier, in tsv format
        from dtk.data import MultiMap
        mm = MultiMap(self.ws.wsa_prop_pairs('smiles_code'))
        with open(self.fn_smiles,'w') as f:
            for k,v in MultiMap.flatten(mm.fwd_map()):
                f.write("%s\t%d\n" % (v,k))
    def write_treatments_file(self):
        self.ds = [str(x) for x in self.ws.get_wsa_id_set(self.parms['drug_set'])]
        self.kt_count = len(self.ds)
        self.info('kt_count %d',self.kt_count)
        if not self.kt_count:
            self.fatal('No qualifying known treatments found')
        with open(self.fn_treatments,'w') as f:
            for kt in self.ds:
                f.write(kt + "\n")
    def find_similarities(self):
        libraryToUse = self.parms['package']
        self.debug("compute structural distance csv")
        local = False
        if local:
            cvt = lambda x:x
        else:
            cvt = self.mch.get_remote_path
        script_dir = cvt(PathHelper.fingerprint)
        script = script_dir + "similarity_csv.py"
        # build up the command depending on the settings from above
        script_with_args_list = [script]
        if libraryToUse == "rdkit":
            if self.parms['min_portion']:
                script_with_args_list+=["--minBitPortion",str(self.parms['min_portion'])]
            if self.parms['bits']:
                script_with_args_list.append("-b " + cvt(self.fn_bits))
            if self.parms['bitslr']:
                script_with_args_list+=["--lr_score_tsv",cvt(self.fn_bitslr)]
            script_with_args_list += ["-l", "rdkit", "-f", str(self.parms['fingerprint'])]
        elif libraryToUse is "indigo":
            script_with_args_list.append("-l indigo")
        # These are common to all approaches
        script_with_args_list.append("-s")
        script_with_args_list.append(cvt(self.fn_smiles))
        script_with_args_list.append("-o")
        script_with_args_list.append(cvt(self.fn_similar))
        script_with_args_list.append(cvt(self.fn_treatments))
        script_with_args_list.append("--cores")
        script_with_args_list.append(str(self.remote_cores_got))
        #
        script_with_args = " ".join(script_with_args_list)
        print('command options',script_with_args)
        if local:
            print("running locally")
            make_directory(self.tmp_pubdir)
            cmd = "cd %s && %s" % (script_dir,script_with_args)
            subprocess.check_call(cmd,shell=True)
            return
        self.copy_input_to_remote()
        self.make_remote_directories([
                                self.tmp_pubdir,
                                ])
        self.mch.check_remote_cmd(script_with_args)
        self.copy_output_from_remote()
    def generate_heatmap(self):
        if self.kt_count < 2:
            self.info('bypassing heatmap generation')
            return
        # input to R script is the subset of lines from the similarities
        # file that match known treatments; also, strip the 'like_' from
        # the header line so column labels and row labels match
        inp = open(self.fn_similar)
        outp = open(self.fn_heat_inp,'w')
        treatments = None
        for line in inp:
            fields = line.strip().split(',')
            if treatments is None:
                # this is the first line
                # strip 'like_'
                l = [x[5:] for x in fields[1:]]
                treatments = set(l)
                outp.write(','.join(fields[:1]+l)+'\n')
            else:
                # every other line
                if fields[0] in treatments:
                    outp.write(line)
        inp.close()
        outp.close()
        subprocess.check_call([
                'Rscript',
                PathHelper.Rscripts+'pheatmap_plotter.R',
                self.fn_heat_inp,
                self.fn_heat_outp,
                ])
    def run(self):
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.tmp_pubdir)
        p_wr = ProgressWriter(self.progress
                , [ "wait for resources"
                  , "setup"
                  , "wait for remote resources"
                  , "prepare structural data"
                  , "check enrichment"
                  , "generate heatmap"
                  ]
                )
        if self.parms['package'] == 'indigo':
            self.remote_cores_wanted=1
        else:
            self.remote_cores_wanted=(1,10) # up to 10
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.write_smiles_file()
        self.write_treatments_file()
        p_wr.put("setup","complete")
        got = self.rm.wait_for_resources(self.job.id,[
                                    0,
                                    self.remote_cores_wanted,
                                    ])
        self.remote_cores_got = got[1]
        p_wr.put("wait for remote resources"
                        ,"complete; got %d cores"%self.remote_cores_got
                        )
        self.find_similarities()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("prepare structural data","complete")
        self.check_enrichment()
        p_wr.put("check enrichment","complete")
        self.generate_heatmap()
        self.finalize()
        p_wr.put("generate heatmap","complete")
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
