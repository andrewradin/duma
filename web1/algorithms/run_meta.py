#!/usr/bin/env python3

from __future__ import print_function
import sys
# retrieve the path where this program is located, and
# use it to construct the path of the website_root directory
try:
    from path_helper import PathHelper,make_directory
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")
    from path_helper import PathHelper,make_directory
from algorithms.exit_codes import ExitCoder
import os
import django
if 'django.core' not in sys.modules:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    django.setup()

import subprocess
import csv
import shutil
from runner.process_info import JobInfo
from browse.models import Tissue,Sample
from tools import ProgressWriter
from aws_op import Machine
from reserve import ResourceManager

import logging
logger = logging.getLogger("algorithms.run_meta")


"""
RNASeq processing currently works as follows:
    - Grab data about the list of runs/experiments needed
        - For GEO this is done in metaGEO.R
        - For BioProject, pull out data from the db that we stored during the search from BigQuery
        - The SRX/Experiment list converts to an SRR/Run list, as well as a combine list
            - In the case of multiple SRR's per SRX, they are marked for combining
    - As of plat4027 the RNA-seq process is now all handed within full_rnaseq_parallel_pipeline.sh
        - Download all of the raw data from those runs
            - rnaSeqBashPipeline_parallel.sh, which calls getRawData_parallel.sh
                - This invokes sratoolkit's prefetch command to get all the data
                - (Note prefetch can handle SRX's directly only when they have only a single SRR,
                  so for consistency we now use SRRs here)
        - Process all of the raw data
            - finish_rnaSeqBashPipeline_parallel.sh
                - convert SRA to FastQ  (sraToFastq_parallel.sh)
                - Merge any FastQ SRR's from the same SRX (combine_redundant_fqs)
                - Process the data (qcFastqs_parallel.sh, runSalmon.sh)
                    - runSalmon runs the salmon binary, then invokes condenseSalmonResults.R at the end
    - Generate a metadata tsv file
        - Used for sample case/control selection for sig
        - For GEO, this is pulled out via R and pulled over from worker
        - For Bio, the metadata is already in the SraRun table from BigQuery
            - (For Bio this runs when viewing the case/control page, rather than here)
    - (For Bio) Generate a list of successfully processed samples
        - In the GEO case, condense_SalmonResults will mark failed downloads as outliers in
          the R metadata rds, which gets converted into the sample metadata tsv file
        - For Bio, we don't create the sample tsv until later, so we need to store the list of
          successful samples here, so that we can use it later (in ge.utils.generate_sample_metadata).
"""


# get exit codes
exitCoder = ExitCoder()

class MyJobInfo(JobInfo):
    def description_text(self):
        return '''
        <b>meta</b> downloads a gene exression dataset, extracts metadata
        needed for case/control assignments, and prepares the dataset
        for use in sig.  When possible meta also normalizes the data and
        produces QC plots.
        '''
    def blocking_jobnames(self,jobname,jcc=None):
        result = set()
        # prevent two copies of the same meta job from running at once
        result.add(jobname)
        return result
    def __init__(self,ws=None,job=None):
        # base class init
        super(MyJobInfo,self).__init__(
                    ws,
                    job,
                    __file__,
                    "Meta",
                    "Extract GE Meta-data",
                    )
        # any base class overrides for unbound instances go here
        self.in_run_menu = False
        # job-specific properties
        if self.job:
            # Although this is currently prevented by special-case code
            # in the workflow view function, it was the case that the
            # tissue_id passed here could be from any workspace that
            # uses the geoID.  Also, metaGEO.R requires a tissue_id parameter
            # because it's parsed by common code with sigGEO.R, even though
            # the parameter isn't used.
            #
            # To minimize propagation of a possibly incorrect workspace,
            # the geoID is extracted here and the tissue_id is otherwise
            # unused.  A zero is passed as the tissue id parameter to
            # metaGEO.R.
            #
            # Note that, as well as appearing in the settings json, there
            # is a tissue_id command line parameter to run_meta.py.  But
            # that parameter will always reflect the workspace the job
            # is launched from, not something copied from another job.
            try:
                tissue_id = self.parms['tissue_id']
            except KeyError:
                # really old meta jobs have the tissue id only on the command
                # line, not in the settings
                tissue_id = int(self.job.cmd.split()[-1])
            try:
                self.tissue = Tissue.objects.get(pk=tissue_id)
            except Tissue.DoesNotExist:
                self.tissue = None
                return
            self.geoID = self.tissue.geoID
            self.source = self.tissue.source
            self.log_prefix = self.geoID+":"
            self.fn_settings = self.indir+'settings.R'
            self.debug("setup")
    def get_jobnames(self,ws):
        '''Return a list of all jobnames for this plugin in a workspace.'''
        return [
                self.get_jobname_for_tissue(t)
                for t in ws.tissue_set.all()
                ]
    def get_jobname_for_tissue(self,t):
        return "%s_%s" % (self.job_type,t.geoID)
    def get_progress(self):
        progress = self._get_progress_parts()
        if len(progress[1]) > 0:
            if progress[1][0][0] == "RNAseq":
                try:
                    outdir=PathHelper.publish+self.geoID
                    remote_od =  self.mch.get_remote_path(outdir)
                    n_samples=self._get_sample_count(outdir)
                    # start by counting the samples that are totally finished
                    final_file_command = f'find {remote_od} -print | grep quant.sf | wc -l'
                    self.mch.check_remote_cmd(final_file_command, hold_output=True)
                    count = int(self.mch.stdout.strip())
                    # I thought about giving a more fine grained estimate by checking for intermediate
                    # files, but that was going to be more of a challenge and this still gets us an estimate
                    progress[1][0][1] = f'>= {str(int(100*count/n_samples))}% complete'
                except:
                    progress[1][0][1] = "???% complete"
        return progress[0]+progress[1]
    def run(self):
        make_directory(self.root)
        make_directory(self.indir)
        self.setup()
        self.run_meta_remote()
        return 0
    def setup(self):
        # initialize progress reporting
        self.p_wr = ProgressWriter(self.progress
                , [ 'wait for resources'
                  , 'setup'
                  , 'meta'
                  , 'RNAseq'
                  , 'cleanup'
                  ]
                )
        self.rna_seq = self.source.endswith('seq')
        self.rm = ResourceManager()
        if self.rna_seq:
            remote_cores = (1,3)
        else:
            remote_cores = 1
        geo_dnld_slots = 1 if self.source in ('geo','geo-orig') else 0
        self.rm.wait_for_resources(
                self.job.id,
                [0,remote_cores,geo_dnld_slots],
                slow=self.rna_seq,
                )
        self.p_wr.put("wait for resources","complete")
        self.write_settings_file()
        self.p_wr.put("setup","complete")
    def write_settings_file(self):
        cvt=self.mch.get_remote_path
        converters = get_and_sync_worker_converters(self.ws, self.mch, self.get_species_name())
        def r_bool(x):
            return 'TRUE' if x else 'FALSE'
        with open(self.fn_settings,'w') as sf:
            sf.write('debug <- %s\n'%r_bool(True))
            sf.write('ignoreMissing <- %s\n'%r_bool(self.tissue.ignore_missing))
            for name, fn in converters:
                sf.write(f'{name} <- "{fn}"\n')

            self._test_add_extra_settings(sf)

    def  _test_add_extra_settings(self, sf):
        # We mock this out in tests to inject extra settings.
        pass

    def get_species_name(self):
        from browse.models import Species
        return Species.get('label', self.get_species()).lower()
    def get_species(self):
        return self.tissue.tissue_set.species

    @property
    def is_bioproject(self):
        return self.geoID.startswith('PRJ')

    def _run_metaGEO(self, m):
        cvt=m.get_remote_path
        # Override the locale for meta geo, there is a bug in gridSVG
        # that prevents it from handling an NA charset and a bug in
        # R that makes localeToCharset unable to interpret C.UTF-8.
        cmd = 'cd %s && LANG=en_US.utf8 Rscript metaGEO.R %s %s 0 %s' % (
                cvt(PathHelper.Rscripts),
                self.source,
                self.geoID,
                m.get_full_remote_path(self.indir),
                )
        return_code = m.run_remote_cmd(cmd)
        return return_code

    def _get_sample_count(self,outdir):
        if self.is_bioproject:
            from ge.models import SraRun
            srxs = SraRun.objects.filter(bioproject=self.geoID).values_list('experiment', flat=True)
            return len(srxs)
        else:
            dnld_file = '%s/%s_%ssToDownload.csv' % (
                            outdir,
                            self.geoID,
                            'FASTQ' if self.source == 'ae-seq' else 'SRA',
                            )
            with open(dnld_file) as f:
                samples = f.readline().split()
            return len(samples)

    def _get_srx_list(self, m, outdir):
        """Constructs the list of experiments to download and process.

        In the case of bioproject, we generate this from the SraRun table,
        using the sample data we pulled from BigQuery.

        In the GEO case, we grab the file from worker that was generated by
        the metaGEO R script.

        In the AE case, it is a list of raw FASTQ files to download.
        """
        cvt=m.get_remote_path
        dnld_file = '%s/%s_%ssToDownload.csv' % (
                            outdir,
                            self.geoID,
                            'FASTQ' if self.source == 'ae-seq' else 'SRA',
                            )
        dnld_rmt = cvt(dnld_file)

        if self.is_bioproject:
            from ge.models import SraRun
            srxs = SraRun.objects.filter(bioproject=self.geoID).values_list('experiment', flat=True)
            # Write the remote download file.
            to_download_str = ' '.join(srxs)
            logger.info(f"Writing {dnld_file} for remote download {dnld_rmt}: {to_download_str}")
            make_directory(os.path.dirname(dnld_file))
            with open(dnld_file, 'w') as f:
                f.write(to_download_str)
            m.copy_to(dnld_file,dnld_rmt)
            return srxs, dnld_rmt
        else:
            m.copy_from(dnld_rmt,dnld_file)
            with open(dnld_file) as f:
                samples = f.readline().split()
            return samples, dnld_rmt

    def _get_kmer_dir(self, species='human'):
        from browse.default_settings import salmon
        from dtk.s3_cache import S3File
        ws_v = salmon.value(self.ws)
# pick one of the flavors and then strip the count off
# it will be dynamically added on the worker
        s3f = S3File.get_versioned(
                'salmon',
                f'{species}.{ws_v}',
                role='kmer31',
                )
        full_path = s3f.path()
        # strip 31.tgz
        return full_path[:-6]
    def _make_srr_and_combine_lists(self, srx_list):
        """Creates a conversion from experiment (SRX) list to list of read files to download (SRRs).

        In some cases, each experiment will have multiple SRRs, which we mark for combining later.
        """

        from dtk.data import MultiMap
        if self.geoID.startswith('E-'):
            # ArrayExpress - it seems like these are already operating at the SRR level
            # rather than SRX?
            return srx_list, [], MultiMap((x,x) for x in srx_list)
        from dtk.sra_bigquery import SraBigQuery
        bq = SraBigQuery()
        srx_to_srr = bq.get_srx_srr_mapping(srx_list)
        assert len(srx_to_srr.fwd_map()) == len(srx_list)
        all_srr = list(srx_to_srr.rev_map().keys())
        to_combine = []
        for srrs in srx_to_srr.fwd_map().values():
           # it's a bit of a misnomer,
           # but we now include all SRRs in this file
           # and the determination of if anything needs to be combined
           # is done in the bash script
            to_add = list(srrs)
            if len(srrs) == 1:
                to_add.append('None') # add a marker that there is nothing to combine
            to_combine.append(to_add)
        return all_srr, to_combine, srx_to_srr

    def make_sample_srr_conversion(self, m, srx_to_srr, to_combine, outdir):
        """Outputs the conversion from sample ID (SRX or GSM) to combined SRR.

        Using the output from make_srr_and_combine above, this take it one step
        further if needed and maps from GSM ID to SRR ID.

        There is only one output per GSM/SRX, to the canonical SRR that other SRR
        data will have been merged into.

        This will output the file into the publish directory on the worker machine.
        """
        cvt=m.get_remote_path
        sample_to_srr = {}

        if self.is_bioproject or self.geoID.startswith('E-'):
            # No gsm conversion, just use identity SRX -> SRX
            gsm_to_srx = {x:x for x in srx_to_srr.fwd_map().keys()}
        else:
            # Pull gsm conversion; that will map GSM -> SRX
            gsm_file = os.path.join(outdir, f'{self.geoID}_gsmToSrxConversion.csv')
            gsm_rmt = cvt(gsm_file)
            m.copy_from(gsm_rmt,gsm_file)


            gsm_to_srx = {}
            from dtk.files import get_file_records
            for gsm, srx in get_file_records(gsm_file, keep_header=False):
                gsm_to_srx[gsm] = srx


        srx_to_combinedsrr = {}
        # for each SRX, find the SRR it converts to, using to_combine to disambiguate
        for srx, srrs in srx_to_srr.fwd_map().items():
            # This is how we pick the canonical srr above, so copy it.
            srx_to_combinedsrr[srx] = list(srrs)[0]

        # Output conversion file, push to machine.
        sample_file = os.path.join(outdir, f'{self.geoID}_sampleToSrrConversion.csv')
        sample_rmt = cvt(sample_file)
        with open(sample_file, 'w') as f:
            f.write('V1,V2\n')
            for gsm, srx in gsm_to_srx.items():
                f.write(f'{gsm},{srx_to_combinedsrr[srx]}\n')
        m.copy_to(sample_file,sample_rmt)


    def write_remote_lists(self, m, srrs, to_combine, outdir, dnld_rmt):
        """Takes the SRR & toCombine lists and writes them out onto the worker."""
        if self.geoID.startswith('E-'):
            return dnld_rmt

        cvt=m.get_remote_path

        local_file = os.path.join(outdir, f'{self.geoID}_SRR_SRAsToDownload.csv')
        print("Creating ", local_file, cvt(local_file))
        with open(local_file, 'w') as f:
            f.write(' '.join(srrs))
        m.copy_to(local_file,cvt(local_file))
        rmt_srrs = cvt(local_file)

        local_file = os.path.join(outdir, f'{self.geoID}_SRRsToCombine.tsv')
        print("Creating ", local_file, cvt(local_file))
        with open(local_file, 'w') as f:
            for entry in to_combine:
                f.write(f'{entry[0]}\t{",".join(entry[1:])}\n')
        m.copy_to(local_file,cvt(local_file))
        return rmt_srrs

    def pull_successful_sample_list(self, m):
        """Gets the list of successfully processed samples from the worker.
        """
        pass

    def run_meta_remote(self):
        m = self.mch
        cvt=m.get_remote_path
        self.copy_input_to_remote()
        if not self.is_bioproject:
            return_code = self._run_metaGEO(m)
        else:
            return_code = 0
        outdir = PathHelper.publish+self.geoID
        self.p_wr.put("meta","complete")

        if self.rna_seq:
            if return_code == exitCoder.encode('alreadyDone'):
                self.p_wr.put("RNAseq","Already Done")
            else:
                if return_code != 0:
                    print('1st phase got exit code',return_code)
                    exit(return_code)
                # Copy the settings file over to the directory that RNASeq knows to look in.
                m.copy_to(self.fn_settings, cvt(outdir + '/settings.R'))
                # There are two more stages to RNAseq processing, and each has
                # different resource constraints, so we manage them here.
                # First, retrieve the number of samples to process.
                samples, dnld_rmt = self._get_srx_list(m, outdir)
                assert samples, "No samples found"
                srrs, to_combine, srx_to_srr = self._make_srr_and_combine_lists(samples)
                srrs_rmt = self.write_remote_lists(m, srrs, to_combine, outdir, dnld_rmt)
                self.make_sample_srr_conversion(m, srx_to_srr, to_combine, outdir)
                n_samples = len(samples)

                # Now do the download & processing together
                # Note that we used to allocate a special RNAseq download
                # slot for each sample to control the rate that we hit the server,
                # but that no longer seems necessary
                out_rmt = cvt(outdir)
                kmer_dir = cvt(self._get_kmer_dir(self.get_species_name()))
                analysis_dir = out_rmt+'/rnaSeqAnalysis'
                dnld_dir = analysis_dir+'/rawFastq'
                home = '$HOME/'
                # Now do the processing phase. We use the 'slow' flag because
                # these cores might be tied up for a while, and we don't want
                # to prevent all other processing
                # Request a big chunk of cores here, we can make use of lots of cores per sample, and
                # we don't want a small number of jobs to linger for days on the worker.
# XXX 337sprintfixes, trying a different balance of cores. Turns out how the cores are used are overwritten in the bash script anyhow, so just going for one core per sample
                remote_cores = (8,n_samples)
                got = self.rm.wait_for_resources(
                        self.job.id,
                        [0,0,0,0,remote_cores],
                        slow=True,
                        )

                print('got',got,'as processing resource vector; samples',n_samples)
                cmd = f'cd {cvt(PathHelper.Rscripts)}RNAseq && bash complete_rnaseq_parallel_pipeline.sh {home+srrs_rmt} {home+dnld_dir} {home+kmer_dir} {home+analysis_dir} {got[4]}'
                print('executing',cmd)
                return_code = m.run_remote_cmd(cmd)

                self.p_wr.put("RNAseq","complete")
        else:
            self.p_wr.put("RNAseq","N/A")
        self.divider('start copying back results')
        xferfile = '/tmp/meta.%s.cpio' % self.geoID
        remote_xferfile = m.get_remote_path(xferfile)
        # currently, the multipleGPLsToChooseFrom.err.txt file
        # is the only .txt file actually expected, but leave
        # the filter as-is
        cmd = 'cd %s && find . %s | cpio -o > %s' % (
                cvt(outdir),
                '-maxdepth 1 -type f'
                    ' \('
                    ' -name "multipleGPLsToChooseFrom.*"'
                    ' -o -name "*_metadata.tsv"'
                    ' -o -name "SRRs_processed.tsv"'
                    ' -o -name "*_sampleToSrrConversion.csv"'
                    ' -o -name "outlierGsms.csv"'
                    ' -o -name "arrayQualityMetrics.*"'
                    ' -o -name "index.html"'
                    ' -o -name "*.pdf"'
                    ' -o -name "*.png"'
                    ' \)'
                    ,
                remote_xferfile,
                )
        m.check_remote_cmd(cmd)
        m.copy_from(remote_xferfile,xferfile)
        subprocess.call(["rm","-rf",outdir])
        subprocess.check_call(["mkdir","-p",outdir])
        subprocess.check_call(["sh"
                , "-c"
                , "cd "+outdir+"&& cpio -i <"+xferfile
                ])
        subprocess.check_call(["rm",xferfile])
        m.run_remote_cmd("rm "+remote_xferfile)
        self.divider('done copying back results')
        if return_code != 0 and return_code != exitCoder.encode('alreadyDone'):
            exit(return_code)
        self.p_wr.put("cleanup","complete")


def get_and_sync_worker_converters(ws, mch, species=None):
    cvt = mch.get_remote_path
    result = []
    cmds = []
    from path_helper import PathHelper
    move_cmd = PathHelper.databases + 'matching/move_s3_files.py'
    if species and species != 'human':
        from browse.default_settings import homologene
        file_type = homologene
        mapping = {
                    'EntrezToUniprotMap':['entrez',species],
                    'EnsemblTRSToUniprotMap':['ensembltrs',species],

# XXX this not currently supported by homolgene,
# XXX but if we start using this more we can extend homologene
#                    'EnsemblToUniprotMap':'Protein_Ensembl',
                    }

    else:
        from browse.default_settings import uniprot
        file_type = uniprot
        mapping = {
                    'EntrezToUniprotMap':['Protein_Entrez',None],
                    'EnsemblToUniprotMap':['Protein_Ensembl',None],
                    'EnsemblTRSToUniprotMap':['Protein_EnsemblTRS',None],
                    }

    # Get the path for the each file and make sure it is
    # fetched onto worker
    for label,l in mapping.items():
        role,flavor=l
        if flavor:
            from dtk.s3_cache import S3File
            choice = f'{flavor}.{file_type.value(ws)}'
            s3f = S3File.get_versioned(
                file_type.name(),
                choice,
                role=role,
                )
        else:
            s3f = file_type.get_s3_file(ws=ws, role=role, fetch=False)
        # Needs the explicit absolute path here.
        path = '/home/ubuntu/' + cvt(s3f.path())
        result.append((label,path))
        cmds.append(move_cmd + ' "" ' + path)
    mch.run_remote_cmd(' && '.join(cmds))
    return result


if __name__ == "__main__":
    job = JobInfo.get_my_job()
    tissue_id = job.settings()['tissue_id']
    t = Tissue.objects.get(pk=tissue_id)
    MyJobInfo.execute(logger, ws_id=t.ws_id)
