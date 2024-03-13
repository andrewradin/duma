#!/usr/bin/env python3

from builtins import range
import sys
from path_helper import PathHelper,make_directory

import os
import django
import django_setup

from django import forms

from browse.models import Tissue
from tools import ProgressWriter
from runner.process_info import JobInfo
import runner.data_catalog as dc
from drugs.models import Prop
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager

import json
import logging
logger = logging.getLogger("algorithms.run_gesig")

BG_ITERS = 100

from collections import defaultdict
class ConfigForm(forms.Form):
    randomize = forms.BooleanField(initial=False
                            ,label = 'Randomize the gene names to generate a negative control dataset.'
                            ,required = False
                            )
    plot_all = forms.BooleanField(initial=True
                            ,label = 'Plot a scatter plot summary. This takes a few minutes.'
                            ,required = False
                            )
    trsig_sub = forms.IntegerField(
                            initial=None,
                            label = 'Treatment-Response signature (job id) to remove from this one',
                            required = False,
                            )
    _subtype_name = "job_subtype"
    def __init__(self, ts_id, copy_job, *args, **kwargs):
        if copy_job:
            ts_id = copy_job.name.split('_')[1]
        elif ts_id is None:
            # if both copy_job and ts_id are None, there must be POST data
            ts_id=args[0][self._subtype_name]
        super(ConfigForm,self).__init__(*args, **kwargs)
        self.ts_id = ts_id
        # build individual threshold fields for each tissue
        qs = Tissue.objects.filter(tissue_set_id=ts_id)
        for t in qs:
            _,_,_,total = t.sig_result_counts()
            if not total:
                continue
            self.fields['t_'+str(t.pk)] = forms.BooleanField(
                                label = t.name+'('+str(t.id)+')',
                                initial = True,
                                required = False,
                                )
        self.fields['min_tis_por'] = forms.FloatField(
                                 label = 'Min. portion of datasets a protein '+
                                         "must be detected in.",
                                 initial = 0.25
                                 )
        from browse.default_settings import GESigBaseComboSig
        from browse.models import TissueSet
        ws = TissueSet.objects.get(pk=self.ts_id).ws
        self.fields['trsig_sub'].initial = GESigBaseComboSig.value(ws)

        if copy_job:
            self.from_json(copy_job.settings_json)
    def as_html(self):
        from django.utils.html import format_html
        return format_html(u'''
                <input name="{}" type="hidden" value="{}"/>
                <table>{}</table>
                '''
                ,self._subtype_name
                ,self.ts_id
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
        from browse.models import TissueSet
        p = {
            'ws_id':TissueSet.objects.get(pk=self.ts_id).ws_id,
            }
        for f in self:
            key = f.name
            value = src[key]
            if key.startswith('t_') and value > 1:
                continue # exclude from dictionary
            p[key] = value
        return p
    def from_json(self,init):
        p = json.loads(init)
        for f in self:
            if f.name in p:
                f.field.initial = p[f.name]

def assemble_tissue_data(inputs):
    # returns a data structure like:
    # {tid:{uniprot:[ev,fc],...},...}
    # where ev and fc are both signed to indicate direction
    tge_dd = {}
    from collections import Counter
    for tid,wt in inputs:
        if wt <= 0.0:
            continue
        tge_dd[tid] = defaultdict(dict)
        errors = Counter()
        t=Tissue.objects.get(pk=tid)
        for rec in t.sig_results(over_only=False):
            out = [float(rec.evidence) * float(rec.direction) * wt]
            try:
                out += [float(rec.fold_change) * float(rec.direction) * wt]
            except TypeError:
                errors['missing fold change'] += 1
                out += [0.0]
            tge_dd[tid][rec.uniprot] = out
        print(("Tissue",tid,"weight",wt,"loaded",len(tge_dd[tid]),"proteins"))
        if len(errors) > 0:
            print(('error counts for tid',tid,errors))
    return tge_dd

def combine_tissue_data(tge_dd,min_tis):
    # returns a structure like:
    # {uniprot:[ev,fc,tc,dir],...}
    # where:
    # - ev is mean signed evidence
    # - fc is mean signed fold change
    # - tc is count of tissues contributing to means
    # - dir is mean direction (so, not an integer)
    dl = defaultdict(lambda: defaultdict(list))
    for tid in tge_dd.keys():
        for uni in tge_dd[tid].keys():
            dl[uni][0].append(tge_dd[tid][uni][0])
            dl[uni][1].append(tge_dd[tid][uni][1])
    if len(list(dl.keys())) == 0:
        raise RuntimeError("no significant proteins")
    signature={}
    gen = (i for i in dl.keys() if len(dl[i][0]) >= min_tis)
    from statistics import mean
    import numpy as np
    for uni in gen:
### Currently we are taking the mean as this did better than median and max seemed inappropriate
### This approach also implicitly defaults to 0 if a protein is not detected in a given tissue.
### That may not be what we want.
### The direction is reported as the average direction, but again we may want to try something else.
        signature[uni] = [mean(dl[uni][0]),
                          mean(dl[uni][1]),
                          len(dl[uni][0]),
                          mean([np.sign(x) for x in dl[uni][1]])
                          ]
    print(("signature calculated;",len(signature),"proteins"))
    return signature

def apply_tr_signature(signature, tr_signature, use_direction):
    # This removes a treatment-response signature from a computed signature
    for uni, data in signature.items():
        trdata = tr_signature.get(uni)

        if not use_direction:
            # In this approach we toss out direction while doing this.
            if trdata:
                trval = abs(trdata[0])
            else:
                trval = 0
            data[0] = max(0, abs(data[0]) - trval)
        else:
            if trdata:
                data[0] = data[0] - trdata[0]

def load_tr_signature(fn):
    sig = {}
    header = None
    from dtk.files import get_file_records
    for rec in get_file_records(fn):
        if header is None:
            header = rec
            continue
        uniprot, ev, fold, cnt, dir = rec
        sig[uniprot] = [float(ev), float(fold), float(cnt), float(dir)]
    return sig


class MyJobInfo(JobInfo):
    def description_text(self):
        return '''
        GESig combines multiple GE datasets into a single signature: a list
        where each protein has a single value.  The output from this method
        is used by several other methods including CoDES and GLEE.
        '''
    def settings_defaults(self,ws):
        result = {}
        from browse.models import TissueSet
        for ts in TissueSet.objects.filter(ws=ws):
            form = ConfigForm(ts.id,None)
            result[ts.name] = form.as_dict()
        return result
    def source_label(self,jobname):
        ts_id,ws_id = self._parse_jobname(jobname)
        from browse.models import TissueSet
        ts = TissueSet.objects.get(pk=ts_id)
        return self.get_source_label_for_tissue_set(ts)
    def get_source_label_for_tissue_set(self,ts):
        return ts.ts_label()+' GESig'
    def build_role_code(self,jobname,settings_json):
        ts_id,_ = self._parse_jobname(jobname)
        return self._tissue_set_role_code(ts_id)
    def out_of_date_info(self,job,jcc):
        ts_id,ws_id = self._parse_jobname(job.name)
        return self._out_of_date_from_tissue_set(job,ts_id)
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        if copy_job or not job_type:
            ts_id = None
        else:
            ts_id = job_type.split('_')[1]
        form = ConfigForm(ts_id,copy_job)
        return form.as_html()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        form = ConfigForm(None,None,post_data)
        if not form.is_valid():
            return (form.as_html(),None)
        p = form.as_dict()
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(p)
                            )
        next_url = ws.reverse('nav_progress',job_id)
        return (None, next_url)
    def _parse_jobname(self,jobname):
        fields = jobname.split('_')
        try:
            return (int(fields[1]),int(fields[2]))
        except ValueError as e:
            from future.utils import raise_with_traceback
            raise_with_traceback(ValueError(e.message + ' jobname is %s' % jobname))
    def __init__(self,ws=None,job=None):
        # base class init
        self.use_LTS=True
        super(MyJobInfo,self).__init__(
                    ws,
                    job,
                    __file__,
                    "GEsig",
                    "Gene Expression SIGnature",
                    )
        # any base class overrides for unbound instances go here
        self.publinks = (
                (None,'scatter.plotly'),
                )
        self.qc_plot_files = (
                'scatter.plotly',
                )
        # job-specific properties
        if self.job:
            self.log_prefix = self.job.name+':'
            self.debug("setup")
            self.ec = ExitCoder()
            # stash common ordered list of tissue ids here
            self.tissue_ids = [
                    int(x[2:])
                    for x in self.parms
                    if x.startswith('t_')
                    ]
            self.tissue_ids.sort()
            # output files
            self.signature_file = self.lts_abs_root+"signature.tsv"
            # published output files
            self.scatter = self.tmp_pubdir+"scatter.plotly"
    def get_jobnames(self,ws):
        '''Return a list of all jobnames for this plugin in a workspace.'''
        return [
                "%s_%d_%d" % (self.job_type,ts.id,ws.id)
                for ts in ws.get_tissue_sets()
                ]
    def is_a_source(self):
        return True
    def get_data_code_groups(self):
        import runner.data_catalog as dc
        return [
                dc.CodeGroup('uniprot',self._std_fetcher('signature_file'),
                        dc.Code('ev',label='Evidence'),
                        dc.Code('fold',label='Fold Change'),
                        dc.Code('tisscnt',label='Tissue Count'),
                        dc.Code('avgDir',label='Mean direction'),
                        ),
                ]
    def run(self):
        from collections import defaultdict
        make_directory(self.indir)
        make_directory(self.outdir)
        make_directory(self.lts_abs_root)
        make_directory(self.tmp_pubdir)
        p_wr = ProgressWriter(self.progress
                , [ "wait for resources"
                  , "setup"
                  , "distilling to a single signature"
                  , "plotting"
                  ]
                )
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        self.get_tissue_data()
        p_wr.put("setup","complete")
        self.distill_to_signature()
        if self.parms['trsig_sub']:
            self.apply_tr()
        self.save_signature()
        p_wr.put("distilling to a single signature","complete" + self.rand_msg)
        status = self.plot()
        p_wr.put("plotting",status)
        self.finalize()

    def incorporates_combo_base(self):
        if self.parms.get('hide_combo', False):
            return False

        return bool(self.parms.get('trsig_sub'))

    def get_tr_from_job(self, jobid):
        from runner.models import Process
        from runner.process_info import JobInfo
        p = Process.objects.get(pk=jobid)
        ws = p.settings()['ws_id']
        bji = JobInfo.get_bound(ws, jobid)
        return bji.signature_file

    def apply_tr(self):
        logger.info("Subtracting out base drug signature")
        sig_file = self.get_tr_from_job(self.parms['trsig_sub'])
        tr_sig = load_tr_signature(sig_file)
        apply_tr_signature(self.signature, tr_sig, self.parms.get('use_direction', True))

    def get_tissue_data(self):
        inputs = [
                  (tid,self.parms['t_'+str(tid)])
                  for tid in self.tissue_ids
                  if self.parms['t_'+str(tid)]
                 ]
        self.tge_dd = assemble_tissue_data(inputs)
    def distill_to_signature(self):
        if self.parms['randomize']:
            self.setup_for_shuffle()
            for i in range(self.bg_iters):
                self.shuffle_protein_names()
                self.single_distill()
                for uni in self.signature.keys():
                    for i in range(len(self.signature[uni])):
                        try:
                            self.shuffled_sigs[uni][i].append(self.signature[uni][i])
                        except IndexError:
                            self.shuffled_sigs[uni].append([self.signature[uni][i]])
            self.signature={}
            # for each protein report the average value, where a missing score is assumed to be 0
            for uni in self.shuffled_sigs.keys():
                self.signature[uni] = [sum(self.shuffled_sigs[uni][i]) / float(self.bg_iters)
                                       for i in range(len(self.shuffled_sigs[uni]))
                                      ]
        else:
            self.rand_msg = ""
            self.single_distill()
    def single_distill(self):
        try:
            self.signature = combine_tissue_data(
                                self.tge_dd,
                                int(round(self.parms['min_tis_por'] * len(list(self.tge_dd.keys())))),
                                )
        except RuntimeError:
            sys.exit(self.ec.encode('unexpectedDataFormat'))


    def setup_for_shuffle(self):
        from collections import defaultdict
        self.bg_iters = BG_ITERS # this could/should be a UI variable
        self.rand_msg = "d background: " + str(self.bg_iters) + " iterations of shuffling proteins"
        from browse.default_settings import uniprot
        from dtk.s3_cache import S3File
        s3f=S3File.get_versioned(
                'uniprot',
                uniprot.value(self.ws),
                'Protein_Entrez',
                )
        s3f.fetch()
        from dtk.files import get_file_records
        self.prot_ids = list(set(
                [rec[0] for rec in get_file_records(s3f.path)]
                ))
        self.shuffled_sigs = defaultdict(list)
    def shuffle_protein_names(self):
        import random
        random.shuffle(self.prot_ids)
        for tid in self.tge_dd.keys():
            random_index = map(int,
                               random.sample( (range( len( list(self.tge_dd[tid].keys()) ))),
                                              len( list(self.tge_dd[tid].keys()))
                                            )
                              )
            assert len(set(random_index)) == len(list(self.tge_dd[tid].values()))
            self.tge_dd[tid] = dict(zip([self.prot_ids[x] for x in random_index], list(self.tge_dd[tid].values())))
    def save_signature(self):
        with open(self.signature_file, 'w') as f:
            header = ['uniprot', 'ev', 'fold', 'tisscnt', 'avDir']
            f.write("\t".join(header) + "\n")
            for uni in self.signature.keys():
                line = "\t".join([str(i) for i in [uni] + self.signature[uni]])
                f.write(line + "\n")
    def plot(self):
        if not self.parms['plot_all']:
            return 'skipped'
        from dtk.plot import scatter2d,fig_legend
        from math import log
        xys = []
        cols = []
        ns = []
        for x in self.signature.values():
            xys.append((x[1],x[0]))
            cols.append(x[3])
            ns.append(x[2])
        from browse.models import Protein
        prot_qs = Protein.objects.filter(uniprot__in=list(self.signature.keys()))
        uni_2_gene = { x.uniprot:x.gene for x in prot_qs }
        names = []
        for id in self.signature:
            try:
                names.append(uni_2_gene[id])
            except KeyError:
                 names.append(id)
        pp = scatter2d('Log2(fold change)',
                'Evidence score',
                xys,
                title = 'Uniprot evidence vs fold change',
                text = ['<br>'.join(
                                [
                                 names[i]
                                 , "Mean direction: " + str(cols[i])
                                 , "n datasets: " + str(ns[i])
                                ])
                        for i in range(len(list(self.signature.keys())))
                       ],
                ids = ('protpage', list(self.signature.keys())),
                refline = False,
                class_idx = [0] * len(list(self.signature.values())), # filler
                bins = True,
                classes=[('Unknown'
                          ,{
                           'color' : cols
                           ,'size' : [log(x,2) + 5 for x in ns]
                           ,'opacity' : 0.3
                           ,'colorbar' : {
                               'title' : 'Mean Direction',
                               'len' : 0.5,
                              },
                          }
                        )],
                height = 800,
               )
        pp._layout['annotations'] = [fig_legend([
                                      'The mean Evidence score is plotted as a'
                                      +' function of the mean Log2(Fold Change) for'
                                      ,'each protein that was detected in the minimum'
                                      +' number of datasets. Dots are colored by the'
                                      ,'average direction of change and the size is'
                                      +' determined by the number of datasets the'
                                      ,'protein was detected in. Clicking a dot '
                                      +"will load that proteins' page"
                                     ],-0.11)]
        pp._layout['margin']=dict(
                              l=50,
                              r=20,
                              b=140,
                              t=30,
                              pad=4
                             )
        pp.save(self.scatter)
        return 'complete'
    def add_workflow_parts(self,ws,parts, nonhuman=False):
        uji = self # simplify access inside MyWorkflowPart
        class MyWorkflowPart:
            def __init__(self,label,ts):
                self.label=label
                self.ts=ts
                self.enabled_default=uji.data_status_ok(ws,'GE',ts.ts_label())
            def add_to_workflow(self,wf):
                cm_info = wf.std_wf.get_main_cm_info(uji.job_type)
                cm_info.pre.add_pre_steps(wf)
                from dtk.workflow import GESigStep
                my_name = uji._tissue_set_role_code(self.ts.id)
                GESigStep(wf,my_name,
                        ts=self.ts,
                        thresh_overrides={}
                        )
                cm_info.post.add_post_steps(wf,my_name,'ev')
        from browse.models import Species
        for ts in ws.tissueset_set.all():
            is_human = ts.species == Species.HUMAN
            if ts.tissue_set.exists() and ts.num_valid_tissues() > 0 and is_human != nonhuman:
                parts.append(MyWorkflowPart(
                        self.get_source_label_for_tissue_set(ts),
                        ts,
                        ))


if __name__ == "__main__":
    MyJobInfo.execute(logger)
