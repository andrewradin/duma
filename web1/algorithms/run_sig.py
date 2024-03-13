#!/usr/bin/env python3

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

import subprocess
import shutil
from browse.models import Tissue,Sample
from tools import ProgressWriter
from runner.process_info import JobInfo
from aws_op import Machine
from algorithms.exit_codes import ExitCoder
from reserve import ResourceManager

import logging
logger = logging.getLogger("algorithms.run_sig")

class MyJobInfo(JobInfo):
    def description_text(self):
        return '''
        After users assign cases and controls, <b>sig</b> identifies which
        genes are differentially expressed. This also creates a number
        of quality controls plots for evaluation.
        '''
    def __init__(self,ws=None,job=None):
        self.use_LTS=True
        # base class init
        super(MyJobInfo,self).__init__(
                    ws,
                    job,
                    __file__,
                    "Sig",
                    "Significant Protein Extraction",
                    )
        # any base class overrides for unbound instances go here
        self.in_run_menu = False
        # job-specific properties
        if self.job:
            self.tissue_id = self.parms['tissue_id']
            self.log_prefix = '%s tissue %d:' %(str(self.ws),self.tissue_id)
            self.fn_settings = self.indir+'settings.R'
            self.fn_sigprot = self.outdir+'browse_significantprotein.tsv'
            self.fn_dcfile = self.lts_abs_root+'sigprot.tsv'
            self.fn_tmpsigqc = self.outdir+'sigqc.tsv'
            self.fn_sigqc = self.lts_abs_root+'sigqc.tsv'
            self.debug("setup")
    def __getattr__(self,attr):
        if attr == 'tissue':
            self.tissue = Tissue.objects.get(pk=self.tissue_id)
            return self.tissue
        raise AttributeError("unknown property '%s'" % attr)
    def get_jobnames(self,ws):
        '''Return a list of all jobnames for this plugin in a workspace.'''
        result = [
                self.get_jobname_for_tissue(t)
                for t in ws.tissue_set.all()
                ]
        return result
    def get_jobname_for_tissue(self,t):
        return "%s_%s_%d_%d" % (self.job_type,t.geoID,t.id,t.ws_id)
    def out_of_date_info(self,job,jcc):
        parms = job.settings()
        nameset = set()
        try:
            tissue = Tissue.objects.get(pk=parms['tissue_id'])
            nameset.add(tissue.get_meta_jobname())
        except (KeyError,Tissue.DoesNotExist):
            pass
        return self._out_of_date_from_names(job,nameset)
    def blocking_jobnames(self,jobname,jcc=None):
        result = set()
        # prevent two copies of the same sig job from running at once
        result.add(jobname)
        return result
    def ordering_jobnames(self,jobname,jcc=None):
        result = set()
        # wait for the matching meta job to complete
        sig,geo,tid,ws_id = jobname.split('_')
        tissue = Tissue.objects.get(pk=tid)
        result.add(tissue.get_meta_jobname())
        return result
    def get_data_code_groups(self):
        import runner.data_catalog as dc
        return [
                dc.CodeGroup('uniprot',self._std_fetcher('fn_dcfile'),
                            dc.Code('ev',label='Evidence',
                                    hidden=True,
                                    ),
                            dc.Code('fold',label='Fold Change',
                                    hidden=True,
                                    ),
                            dc.Code('dir',label='Direction',
                                    hidden=True,
                                    ),
                            ),
                ]
    def run(self):
        make_directory(self.root)
        make_directory(self.lts_abs_root)
        self.setup()
        self.run_sig_remote()
        return 0
    def setup(self):
        # initialize progress reporting
        self.p_wr = ProgressWriter(self.progress
                , [ 'wait for resources'
                  , 'setup'
                  , 'sig'
                  ]
                )
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[0,1])
        self.p_wr.put("wait for resources","complete")
        self.tissue._start_sig_result_job(self.job.id)
        make_directory(self.indir)
        # write case/control files
        qs = Sample.objects.filter(tissue=self.tissue).order_by('sample_id')
        casefile = self.indir+'/cases.tsv'
        controlfile = self.indir+'/controls.tsv'
        with open(casefile,'w') as cases:
            with open(controlfile,'w') as controls:
                enum = Sample.group_vals
                for s in qs:
                    if s.classification == enum.CONTROL:
                        controls.write(s.sample_id+'\n')
                    elif s.classification == enum.CASE:
                        cases.write(s.sample_id+'\n')
        self.write_settings_file()
        self.p_wr.put("setup","complete")
    def write_settings_file(self):
        with open(self.fn_settings,'w') as sf:
            # Write out the paths to the converter files.
            from algorithms.run_meta import get_and_sync_worker_converters
            converters = get_and_sync_worker_converters(self.ws, self.mch,
                                                        species= self.parms['species'].split('_')[0].lower()
                                                       )
            cvt=self.mch.get_remote_path
            for name, fn in converters:
                sf.write(f'{name} <- "{fn}"\n')

            sf.write("algoToUse <- '%s'\n"%self.parms['algo'])
            def r_bool(x):
                return 'TRUE' if x else 'FALSE'
            sf.write("miRNA <- %s\n"%r_bool(self.parms['miRNA']))
            sf.write("scRNAseq <- %s\n"%r_bool(self.parms['scRNAseq']))
            sf.write("runSVA <- %s\n"%r_bool(self.parms['runSVA']))
            sf.write("debug <- %s\n"%r_bool(self.parms['debug']))
            sf.write("top1PercentSignalThreshold <- %f\n"
                    % self.parms['top1thresh']
                    )
            sf.write("permNum <- %f\n" % self.parms['permut'])
            sf.write("absoluteMinUniprotProportion <- %f\n"
                    % self.parms['minUniPor']
                    )
            sf.write("minProbePortionAgreeingOnDirection <- %f\n"
                    % self.parms['minDirPor']
                    )
            sf.write("minSamplePortionWithReads <- %f\n"
                    % self.parms['minReadPor']
                    )
            sf.write("minCPM <- %f\n" % self.parms['minCPM'])
            self._test_add_extra_settings(sf)

    def _test_add_extra_settings(self, sf):
        # We mock this out in tests to inject extra settings.
        pass

    def run_sig_remote(self):
        m = self.mch
        self.copy_input_to_remote()
        self.make_remote_directories([
                self.tmp_pubdir,
                self.outdir,
                ])
        cmd = 'cd %s && Rscript sigGEO.R %s %s %d %s %s %s' % (
                m.get_remote_path(PathHelper.Rscripts),
                self.tissue.source,
                self.tissue.geoID,
                self.tissue.id,
                m.get_full_remote_path(self.indir),
                m.get_full_remote_path(self.outdir),
                m.get_full_remote_path(self.tmp_pubdir),
                )
        #XXX cmd += ' testing'
        return_code = m.run_remote_cmd(cmd)
        need_upload = False
        ec = ExitCoder()
        if ec.decodes_to('noGenesFromGeoDE',return_code):
            self.warning("no proteins found")
            # follow normal success path, but nothing to upload
        elif return_code != 0:
            self.error("sig failed; error code %d",return_code)
            sys.exit(return_code)
        else:
            need_upload = True
        self.copy_output_from_remote()
        if need_upload:
            self._refresh_db_connection()
            self.convert_sig_result()
            shutil.copyfile(self.fn_tmpsigqc,self.fn_sigqc)
        self.finalize()
        self.p_wr.put("sig","complete")
    def convert_sig_result(self):
        self._convert_sig_result(self.fn_sigprot)
    def _convert_sig_result(self,fn):
        from dtk.files import get_file_records
        l = [
            (rec[1],float(rec[3]),int(rec[4]),float(rec[6]))
            for rec in get_file_records(fn)
            ]
        self.write_sig_file(l)
        self.tissue._recalculate_sig_result_counts()
        self.tissue.save()
    def write_sig_file(self,l):
        # l is a list of tuples (which can be SigResultRec namedtuples)
        # containing uniprot, ev, dir, fold (last 3 as numerics);
        # after return, caller needs to:
        # - push to LTS
        # - update tissue counts
        l.sort(key=lambda x:x[1],reverse=True)
        fmt = '%s\t%.3e\t%d\t%.3e\n'
        with open(self.fn_dcfile,'w') as f:
            f.write('uniprot\tev\tdir\tfold\n')
            for rec in l:
                f.write(fmt%rec)

if __name__ == "__main__":
    job = JobInfo.get_my_job()
    tissue_id = job.settings()['tissue_id']
    t = Tissue.objects.get(pk=tissue_id)
    MyJobInfo.execute(logger, ws_id=t.ws_id)
