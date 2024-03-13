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

import logging
logger = logging.getLogger("algorithms.run_ltsconv")

import json
from tools import ProgressWriter
from reserve import ResourceManager
from runner.process_info import JobInfo

from django import forms

class ConfigForm(forms.Form):
    plugins = forms.CharField(
                label='Plugins to convert',
                initial='faers_pickle',
                )
    batch_size = forms.IntegerField(initial=10)

# XXX manual cleanup after testing (w/ asserts commented out):
# XXX from dtk.tests.tests import rm_readonly_tree, rm_bare_lts_repo, rm_lts_s3_repo
# XXX rm_readonly_tree('/home/carl/2xar/lts/5')
# XXX rm_bare_lts_repo('5')
# XXX rm_lts_s3_repo('5')

import shutil
from dtk.lts import LtsRepo

################################################################################
# low-level file converters
################################################################################
def decompress(old,new):
    import subprocess
    print('decompress',old,new)
    subprocess.check_call(['zcat',old],stdout=open(new,'w'))

def dump_candidate_table(old,new):
    print('extract candidates to',new)
    job_id = new.split('/')[-2]
    outf = open(new,'w')
    outf.write('\t'.join([
            'wsa','direct','indirect','direction','pathkey',
            ])+'\n')
    from browse.models import Candidate
    for cand in Candidate.objects.filter(run_id=job_id):
        outf.write('\t'.join([str(x) for x in [
                cand.drug_ws_id,
                cand.direct_score,
                cand.indirect_score,
                cand.direction_sum,
                cand.path_key,
                ]])+'\n')

def rewrite_capp_header(old,new):
    print('rewrite_capp_header',old,new)
    outf = open(new,'w')
    inf = open(old)
    parts = inf.next().rstrip('\n').split('\t')
    assert parts[0] == 'drug_id'
    parts[0] = 'wsa'
    assert parts[1] == 'direct'
    parts[1] = 'capds'
    assert parts[2] == 'indirect'
    parts[2] = 'capis'
    outf.write('\t'.join(parts)+'\n')
    for line in inf:
        outf.write(line)

def rewrite_gesig_header(old,new):
    print('rewrite_gesig_header',old,new)
    outf = open(new,'w')
    inf = open(old)
    parts = inf.next().rstrip('\n').split('\t')
    for old,new in [
                ('evidence','ev'),
                ('log2 fold change','fold'),
                ('tissue_count','tisscnt'),
                ('average direction','avDir'),
                ]:
        try:
            idx = parts.index(old)
        except ValueError:
            continue
        parts[idx] = new
    outf.write('\t'.join(parts)+'\n')
    for line in inf:
        outf.write(line)

def rewrite_header(oldf,newf,conversions):
    outf = open(newf,'w')
    inf = open(oldf)
    parts = inf.next().rstrip('\n').split('\t')
    for old,new in conversions:
        try:
            idx = parts.index(old)
        except ValueError:
            continue
        parts[idx] = new
    outf.write('\t'.join(parts)+'\n')
    for line in inf:
        outf.write(line)

def add_header(oldf,newf,header):
    outf = open(newf,'w')
    inf = open(oldf)
    outf.write('\t'.join(header)+'\n')
    for line in inf:
        outf.write(line)

def rewrite_gpath_header(old,new):
    print('rewrite_gpath_header',old,new)
    rewrite_header(old,new,[
            ('drug_id','wsa'),
            ('direct','gds'),
            ('indirect','gis'),
            ])

def rewrite_gwasig_header(old,new):
    print('rewrite_gwasig_header',old,new)
    rewrite_header(old,new,[
            ('evidence','ev'),
            ('gwdsCount','gwascnt'),
            ])

def copy(old,new):
    print('copy',old,new)
    shutil.copyfile(old,new)

def dircopy(old,new):
    print('dircopy',old,new)
    shutil.copytree(old,new)

def sig_publish_dircopy(old,new):
    print('sig publish dircopy',old,new)
    shutil.copytree(old,new,ignore=shutil.ignore_patterns(
                        '*.tsv',
                        'GeneOntology*',
                        ))

def write_sigprot_common(new,l):
    l.sort(key=lambda x:x[1],reverse=True)
    fmt = '%s\t%.3e\t%d\t%.3e\n'
    with open(new,'w') as f:
        f.write('uniprot\tev\tdir\tfold\n')
        for rec in l:
            f.write(fmt%rec)

def dump_sigprot_table(tid,new):
    print('extract sigprot to',new,'from tissue',tid)
    from browse.models import SignificantProtein
    l = [
        (
            sp.protein_id,
            float(sp.evidence),
            int(sp.direction),
            float(sp.fold_change or 0),
        )
        for sp in SignificantProtein.objects.filter(tissue_id=tid)
        ]
    write_sigprot_common(new,l)

def rewrite_sigprot(old,new):
    print('rewrite sigprot',old,new)
    with open(old) as f:
        from dtk.readtext import parse_delim
        l = []
        for rec in parse_delim(f):
            if len(rec) < 7:
                rec.append('0')
            l.append(
                (rec[1],float(rec[3]),int(rec[4]),float(rec[6]))
                )
    write_sigprot_common(new,l)

################################################################################
# Converter base class
################################################################################
class Converter(object):
    name_index = {}
    def __init__(self,plugin_with_optional_suffix):
        self._plugin = plugin_with_optional_suffix.split('_')[0]
        self.name_index[plugin_with_optional_suffix] = self
        self.updated_repos = set()
    def timestamp(self):
        import datetime
        print(datetime.datetime.now())
    def _group_jobs_by_ws(self):
        self._ws_jobs = {}
        from runner.models import Process
        for job in Process.objects.filter(
                    status=Process.status_vals.SUCCEEDED,
                    name__startswith=self._plugin+'_',
                    ).order_by('-id'):
            ws_id = self.ws_of_job(job)
            self._ws_jobs.setdefault(ws_id,[]).append(job)
        print('got %d jobs across %d workspaces' % (
                sum([len(x) for x in self._ws_jobs.values()]),
                len(self._ws_jobs)
                ))
    def convert(self,batch_size):
        jobs_examined = 0
        file_stats = [0,0,0]
        self._group_jobs_by_ws()
        for ws_id, job_list in six.iteritems(self._ws_jobs):
            repo = LtsRepo(str(ws_id),PathHelper.cfg('lts_branch'))
            if not repo.is_replicated():
                # it's a dev repo, but do all the S3 and sync stuff
                # anyway, so we can get accurate timings
                repo.force_replication()
            waiting = 0
            push_list = []
            need_sync = False
            for job in job_list:
                self.cur_job = job # for derived function access
                old_root = PathHelper.storage+'%d/%s/%d/' % (
                                            ws_id,
                                            job.job_type(),
                                            job.id,
                                            )
                job_rel = os.path.join(job.job_type(),str(job.id))
                ws_rel = os.path.join(job.job_type())
                print('######',ws_id,job_rel)
                self.timestamp()
                job_abs = os.path.join(repo.path(),job_rel)
                print('######','about to convert')
                if self.convert_one_job(old_root,repo,job_rel,file_stats):
                    if batch_size == 1:
                        print('######','about to push')
                        repo.lts_push(job_rel)
                    else:
                        waiting += 1
                        push_list.append(job_rel)
                        if waiting == batch_size:
                            print('######','about to push')
                            repo.lts_add(ws_rel)
                            waiting = 0
                    self.updated_repos.add(repo._repo_name)
                    need_sync = True
                print('######','convert complete',file_stats)
                jobs_examined += 1
            if batch_size != 1:
                # XXX This is currently very fast, but may be more complex
                # XXX than it needs to be. Since adding --fast mode on the
                # XXX send, it may be ok to just do an entire push for
                # XXX each batch, rather than saving them all until the end.
                print('######','final ws push',ws_id)
                if waiting:
                    repo.lts_add(ws_rel)
                waiting = 0
                #for d in push_list:
                #    repo.lts_send(d)
                if push_list:
                    self.timestamp()
                    repo.lts_send(ws_rel)
            if need_sync:
                repo.lts_sync()
        return "converted %d files in %d jobs (%d previous, %d no input); touched %d repos"%(
                file_stats[0],
                jobs_examined,
                file_stats[1],
                file_stats[2],
                len(self.updated_repos),
                )

################################################################################
# old-style plugin-level converters
################################################################################
class CappConverter(Converter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def convert_one_job(self,old_root,repo,job_rel,file_stats):
        need_push = False
        for target in ('capp.tsv','path_detail.tsv','publish'):
            new_path = os.path.join(repo.path(),job_rel,target)
            if os.path.lexists(new_path):
                file_stats[1] += 1
                continue
            if target == 'capp.tsv':
                old_path = os.path.join(old_root,'output',target)
                method = rewrite_capp_header
            elif target == 'path_detail.tsv':
                old_path = os.path.join(old_root,'output','path_detail0.tsv.gz')
                method = decompress
            elif target in ['publish']:
                old_path = os.path.join(
                        PathHelper.publish,
                        repo._repo_name,
                        job_rel,
                        )
                method = dircopy
            if not os.path.exists(old_path):
                file_stats[1] += 1
                continue
            make_directory(os.path.dirname(new_path))
            method(old_path,new_path)
            file_stats[0] += 1
            need_push = True
        return need_push

CappConverter('capp')

class GpathConverter(Converter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def convert_one_job(self,old_root,repo,job_rel,file_stats):
        need_push = False
        for target in ('gpath.tsv','path_detail.tsv','publish'):
            new_path = os.path.join(repo.path(),job_rel,target)
            if os.path.lexists(new_path):
                file_stats[1] += 1
                continue
            if target == 'gpath.tsv':
                old_path = os.path.join(old_root,'output',target)
                method = rewrite_gpath_header
            elif target == 'path_detail.tsv':
                old_path = os.path.join(old_root,'output','path_detail0.tsv.gz')
                method = decompress
            elif target in ['publish']:
                old_path = os.path.join(
                        PathHelper.publish,
                        repo._repo_name,
                        job_rel,
                        )
                method = dircopy
            if not os.path.exists(old_path):
                file_stats[1] += 1
                continue
            make_directory(os.path.dirname(new_path))
            method(old_path,new_path)
            file_stats[0] += 1
            need_push = True
        return need_push

GpathConverter('gpath')

class GesigConverter(Converter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def convert_one_job(self,old_root,repo,job_rel,file_stats):
        need_push = False
        for target in ('signature.tsv','publish'):
            new_path = os.path.join(repo.path(),job_rel,target)
            if os.path.exists(new_path):
                file_stats[1] += 1
                continue
            if target == 'signature.tsv':
                old_path = os.path.join(old_root,'output',target)
                method = rewrite_gesig_header
            elif target in ['publish']:
                old_path = os.path.join(
                        PathHelper.publish,
                        repo._repo_name,
                        job_rel,
                        )
                method = dircopy
            if not os.path.exists(old_path):
                file_stats[2] += 1
                print(old_path,'does not exist')
                continue
            make_directory(os.path.dirname(new_path))
            method(old_path,new_path)
            file_stats[0] += 1
            need_push = True
        return need_push

GesigConverter('gesig')

class GwasigConverter(Converter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def convert_one_job(self,old_root,repo,job_rel,file_stats):
        need_push = False
        for target in ('signature.tsv','publish'):
            new_path = os.path.join(repo.path(),job_rel,target)
            if os.path.exists(new_path):
                file_stats[1] += 1
                continue
            if target == 'signature.tsv':
                old_path = os.path.join(old_root,'output',target)
                method = rewrite_gwasig_header
            elif target in ['publish']:
                old_path = os.path.join(
                        PathHelper.publish,
                        repo._repo_name,
                        job_rel,
                        )
                method = dircopy
            if not os.path.exists(old_path):
                file_stats[2] += 1
                print(old_path,'does not exist')
                continue
            make_directory(os.path.dirname(new_path))
            method(old_path,new_path)
            file_stats[0] += 1
            need_push = True
        return need_push

GwasigConverter('gwasig')

################################################################################
# new-style converter base class
################################################################################
class NewConverter(Converter):
    def convert(self,batch_size):
        jobs_examined = 0
        file_stats = [0,0,0]
        self._group_jobs_by_ws()
        for ws_id, job_list in six.iteritems(self._ws_jobs):
            repo = LtsRepo(str(ws_id),PathHelper.cfg('lts_branch'))
            if not repo.is_replicated():
                # it's a dev repo, but do all the S3 and sync stuff
                # anyway, so we can get accurate timings
                repo.force_replication()
            waiting = 0
            push_list = []
            need_sync = False
            for job in job_list:
                self.cur_job = job # for derived function access
                job_rel = os.path.join(job.job_type(),str(job.id))
                ws_rel = os.path.join(job.job_type())
                print('######',ws_id,job_rel)
                self.timestamp()
                if self.convert_one_job(ws_id,job_rel,repo,file_stats):
                    if batch_size == 1:
                        print('######','about to push')
                        repo.lts_push(job_rel)
                    else:
                        waiting += 1
                        push_list.append(job_rel)
                        if waiting == batch_size:
                            print('######','about to add')
                            repo.lts_add(ws_rel)
                            waiting = 0
                    self.updated_repos.add(repo._repo_name)
                    need_sync = True
                print('######','convert complete',file_stats)
                jobs_examined += 1
            if batch_size != 1:
                # XXX This is currently very fast, but may be more complex
                # XXX than it needs to be. Since adding --fast mode on the
                # XXX send, it may be ok to just do an entire push for
                # XXX each batch, rather than saving them all until the end.
                print('######','final ws push',ws_id)
                if waiting:
                    repo.lts_add(ws_rel)
                waiting = 0
                #for d in push_list:
                #    repo.lts_send(d)
                if push_list:
                    self.timestamp()
                    repo.lts_send(ws_rel)
            if need_sync:
                repo.lts_sync()
        return "converted %d files in %d jobs (%d previous, %d no input); touched %d repos"%(
                file_stats[0],
                jobs_examined,
                file_stats[1],
                file_stats[2],
                len(self.updated_repos),
                )
    def convert_one_job(self,ws_id,job_rel,lts_repo,file_stats):
        need_push = False
        for target in self.target_list():
            new_path = target.new_path(ws_id,job_rel,lts_repo)
            if not new_path:
                file_stats[1] += 1
                continue
            old_path = target.old_path(ws_id,job_rel,lts_repo)
            if not old_path:
                file_stats[2] += 1
                continue
            make_directory(os.path.dirname(new_path))
            target.process(old_path,new_path)
            file_stats[0] += 1
            need_push = True
        return need_push

################################################################################
# new-style single-file converters
################################################################################
class PubdirTarget:
    def new_path(self,ws_id,job_rel,lts_repo):
        new_path = os.path.join(lts_repo.path(),job_rel,'publish')
        if os.path.exists(new_path):
            return None
        return new_path
    def old_path(self,ws_id,job_rel,lts_repo):
        old_path = os.path.join(
                        PathHelper.publish,
                        str(ws_id),
                        job_rel,
                        )
        if not os.path.exists(old_path) or not os.listdir(old_path):
            return None
        return old_path
    def process(self,old_path,new_path):
        dircopy(old_path,new_path)

class OutfileCopyTarget:
    def __init__(self,target):
        self.target = target
    def new_path(self,ws_id,job_rel,lts_repo):
        new_path = os.path.join(lts_repo.path(),job_rel,self.target)
        if os.path.exists(new_path):
            return None
        return new_path
    def old_path(self,ws_id,job_rel,lts_repo):
        old_path = os.path.join(
                        PathHelper.storage,
                        str(ws_id),
                        job_rel,
                        'output',
                        self.target,
                        )
        if not os.path.exists(old_path):
            return None
        return old_path
    def process(self,old_path,new_path):
        copy(old_path,new_path)


class OutfileTsvTarget:
    def __init__(self,target,header_rewrites=None,add_header=None):
        self.target = target
        self.header_rewrites=header_rewrites
        self.add_header=add_header
    def new_path(self,ws_id,job_rel,lts_repo):
        new_path = os.path.join(lts_repo.path(),job_rel,self.target)
        if os.path.exists(new_path):
            return None
        return new_path
    def old_path(self,ws_id,job_rel,lts_repo):
        old_path = os.path.join(
                        PathHelper.storage,
                        str(ws_id),
                        job_rel,
                        'output',
                        self.target,
                        )
        if not os.path.exists(old_path):
            return None
        return old_path
    def process(self,old_path,new_path):
        if self.add_header:
            print('add_header',old_path,new_path)
            add_header(old_path,new_path,self.add_header)
        elif self.header_rewrites:
            print('rewrite_header',old_path,new_path)
            rewrite_header(old_path,new_path,self.header_rewrites)
        else:
            copy(old_path,new_path)

class MlPofATarget:
    def new_path(self,ws_id,job_rel,lts_repo):
        new_path = os.path.join(lts_repo.path(),job_rel,'allPofA.tsv')
        if os.path.exists(new_path):
            return None
        return new_path
    def old_path(self,ws_id,job_rel,lts_repo):
        old_path = os.path.join(
                        PathHelper.storage,
                        str(ws_id),
                        job_rel,
                        'output',
                        'allPofA.csv',
                        )
        if not os.path.exists(old_path):
            return None
        return old_path
    def process(self,old_path,new_path):
        # copy to LTS directory and add header
        from dtk.files import get_file_records,FileDestination
        with FileDestination(new_path,['wsa','ml']) as dest:
            for rec in get_file_records(old_path):
                dest.append(rec)

################################################################################
# new-style plugin-level converters
################################################################################
class CodesConverter(NewConverter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def target_list(self):
        return [
                PubdirTarget(),
                OutfileTsvTarget('codes.tsv',
                        header_rewrites=[
                                ('drug_id','wsa'),
                                ('codesDir','negDir'),
                                ('codesCor','posCor'),
                                ],
                        )
                ]

CodesConverter('codes')

class EsgaConverter(NewConverter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def target_list(self):
        return [
                PubdirTarget(),
                OutfileTsvTarget('esga.tsv',
                        header_rewrites=[
                                ('drug_id','wsa'),
                                ('protrank_max','prMax'),
                                ],
                        )
                ]

EsgaConverter('esga')

class GpbrConverter(NewConverter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def target_list(self):
        return [
                PubdirTarget(),
                ]+[
                OutfileTsvTarget(stem+'_scores.tsv',
                        add_header=[
                                'wsa',stem+'bg',stem+'bgnormed',stem+'pval'
                                ],
                        )
                for stem in ['direct','indirect','direction']
                ]

GpbrConverter('gpbr')

class DependConverter(NewConverter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def target_list(self):
        return [
                PubdirTarget(),
                OutfileTsvTarget('depend.tsv',
                        header_rewrites=[
                                ('drug_id','wsa'),
                                ],
                        ),
                ]

DependConverter('depend')

class GleeConverter(NewConverter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def target_list(self):
        targets = []
        try:
            skip = self.cur_job.settings()['input_score'].startswith('single')
        except KeyError:
            skip = False
        if not skip:
            targets.append(OutfileTsvTarget('glee.tsv',
                        header_rewrites=[
                                ('List_name','uniprotset'),
                                ('NES_lower','NESlower'),
                                ('NES_upper','NESupper'),
                                ('p-value','pvalue'),
                                ('P-value','pvalue'),
                                ('q-value','qvalue'),
                                ],
                        ))
        return targets+[PubdirTarget()]

GleeConverter('glee')

class JacSimConverter(NewConverter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def target_list(self):
        return [
                PubdirTarget(),
                OutfileTsvTarget('jacsim.tsv',
                        header_rewrites=[
                                ('drug_id','wsa'),
                                ('jaccard','DirectJacSim'), # legacy format
                                ('dt_jaccard','DirectJacSim'),
                                ('it_jaccard','IndirectJacSim'),
                                ],
                        ),
                ]

JacSimConverter('jacsim')

class PrSimConverter(NewConverter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def target_list(self):
        return [
                PubdirTarget(),
                OutfileTsvTarget('prsim.tsv',
                        header_rewrites=[
                                ('drug_id','wsa'),
                                ('protrank_max','prSimMax'),
                                ('protrank_median','prSimMed'),
                                ('protrank_mean','prSimMean'),
                                ],
                        ),
                ]

PrSimConverter('prsim')

class UphdConverter(NewConverter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def target_list(self):
        return [
                PubdirTarget(),
                OutfileTsvTarget('uniprot_IDs.tsv'),
                ]

UphdConverter('uphd')

class WsCopyConverter(NewConverter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def target_list(self):
        return [
                PubdirTarget(),
                OutfileTsvTarget('wscopy.tsv',
                        header_rewrites=[
                                ('drug_id','wsa'),
                                ],
                        ),
                ]

WsCopyConverter('wscopy')

class SigDifConverter(NewConverter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def target_list(self):
        return [
                PubdirTarget(),
                OutfileTsvTarget('signature.tsv',
                        ),
                ]

SigDifConverter('sigdif')

class WzsConverter(NewConverter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def target_list(self):
        return [
                PubdirTarget(),
                OutfileTsvTarget('wz_score.tsv',
                        add_header=['wsa','wzs'],
                        ),
                OutfileTsvTarget('weights.tsv',
                        ),
                ]

WzsConverter('wzs')

class MlConverter(NewConverter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def target_list(self):
        return [
                PubdirTarget(),
                MlPofATarget(),
                ]

MlConverter('ml')

class FaersConverter(NewConverter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def target_list(self):
        return [
                PubdirTarget(),
                OutfileTsvTarget('faers_output.tsv',
                        header_rewrites=[
                                ('drug_id','wsa'),
                                ('q-value','dcoq'),
                                ('faers_enrichment','dcoe'),
                                ('faers_portion','dcop'),
                                ('drug_portion','drugPor'),
                                ],
                        ),
                OutfileTsvTarget('lr_faers_output.tsv',
                        header_rewrites=[
                                ('drug_id','wsa'),
                                ('lr_p_value','lrpvalue'),
                                ('lr_enrichment','lrenrichment'),
                                ('lr_direction','lrdir'),
                                ],
                        ),
                OutfileTsvTarget('coindications.tsv',
                        header_rewrites=[
                                ('Feature','mindi'),
                                ('Odds Ratio','coior'),
                                ('Q-Value','coiqv'),
                                ],
                        ),
                ]

FaersConverter('faers')

class FvsConverter(NewConverter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def target_list(self):
        return [
                PubdirTarget(),
                OutfileCopyTarget('feature_matrix.npz'),
                ]

FvsConverter('fvs')

class EsgaPickleConverter(NewConverter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def target_list(self):
        return [
                OutfileCopyTarget('out.pickle'),
                ]

EsgaPickleConverter('esga_pickle')


class FaersPickleConverter(NewConverter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def target_list(self):
        return [
                OutfileCopyTarget('important_stats.pkl'),
                ]

FaersPickleConverter('faers_pickle')

################################################################################
# more old-style plugin-level converters
################################################################################
class PathConverter(Converter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def convert_one_job(self,old_root,repo,job_rel,file_stats):
        need_push = False
        for target in ('path_scores.tsv','path_detail.tsv','publish'):
            new_path = os.path.join(repo.path(),job_rel,target)
            if os.path.exists(new_path):
                file_stats[1] += 1
                continue
            if target == 'path_scores.tsv':
                # this data is in the database, so there is no "old_path",
                # but the code below requires a valid existing path in
                # order to trigger the conversion
                old_path = PathHelper.storage
                method = dump_candidate_table
            elif target in ['path_detail.tsv']:
                parts = old_root.split('/')
                old_path = os.path.join(
                        PathHelper.storage,
                        parts[-4], # ws_id
                        'paths.%s.tsv.gz'%parts[-2] # job_id
                        )
                method = decompress
            elif target in ['publish']:
                old_path = os.path.join(
                        PathHelper.publish,
                        repo._repo_name,
                        job_rel,
                        )
                method = dircopy
            if not os.path.exists(old_path):
                file_stats[2] += 1
                print(old_path,'does not exist')
                continue
            make_directory(os.path.dirname(new_path))
            method(old_path,new_path)
            file_stats[0] += 1
            need_push = True
        return need_push

PathConverter('path')

class LogConverter(Converter):
    # not ws based; override base class conversion
    def convert(self,batch_size):
        jobs_examined = 0
        file_stats = [0,0,0]
        from runner.common import LogRepoInfo
        # first build list of old progress files and organize by job_id
        job2prog = {}
        import glob
        for fn in glob.glob(PathHelper.storage+'*/*/*/progress'):
            parts = fn.split('/')
            job2prog[int(parts[-2])] = fn
        # group jobs by repo,hash directory
        job_groups = {}
        from runner.models import Process
        for job_id in Process.objects.values_list('id',flat=True):
            lri = LogRepoInfo(job_id)
            d = job_groups.setdefault(lri._path_parts[1],{})
            d.setdefault(lri._path_parts[2].split('/')[0],[]).append(job_id)
        # now process one shard at a time
        for repo_name in sorted(job_groups.keys()):
            shard_map = job_groups[repo_name]
            repo = LtsRepo(repo_name,PathHelper.cfg('lts_branch'))
            if not repo.is_replicated():
                # it's a dev repo, but do all the S3 and sync stuff
                # anyway, so we can get accurate timings
                repo.force_replication()
            for shard in sorted(shard_map.keys()):
                print('######','starting shard',shard)
                self.timestamp()
                job_list = sorted(shard_map[shard])
                copy_list = []
                for job_id in job_list:
                    jobs_examined += 1
                    lri = LogRepoInfo(job_id)
                    old_log = PathHelper.publish+('bg_logs/%d.txt'%job_id)
                    if not os.path.exists(old_log):
                        file_stats[2] += 1
                    else:
                        copy_list.append((old_log,lri.log_path()))
                    old_prog = job2prog.get(job_id)
                    if not old_prog:
                        file_stats[2] += 1
                    else:
                        copy_list.append((old_prog,lri.progress_path()))
                print('######','scan done',len(copy_list),'found')
                self.timestamp()
                need_push = False
                for old_path,new_path in copy_list:
                    if os.path.exists(new_path):
                        file_stats[1] += 1
                    else:
                        make_directory(os.path.dirname(new_path))
                        copy(old_path,new_path)
                        need_push = True
                        file_stats[0] += 1
                if need_push:
                    print('######','starting push shard',shard)
                    self.timestamp()
                    repo.lts_add(shard)
            print('######','starting final send',repo_name)
            self.timestamp()
            repo.lts_send('.')
            repo.lts_sync()
            print('######','completed final send',repo_name)
            self.timestamp()
        return "converted %d files in %d jobs (%d previous, %d no input)"%(
                file_stats[0],
                jobs_examined,
                file_stats[1],
                file_stats[2],
                )

LogConverter('log')

class SigConverter(Converter):
    def ws_of_job(self,job):
        return int(job.name.split('_')[-1])
    def tissue_of_job(self,job):
        return int(job.name.split('_')[-2])
    # For sig, we need to populate new Tissue fields as well as reformat
    # output. Since the Tissue record is affected by the most recent job
    # even if it fails, and since we can leverage existing code to populate
    # the tissue record if we do it after the job conversion, we use a
    # custom _group_jobs_by_ws to include failed-but-relevant jobs.
    #
    # We also want to process comb and ext tissues, and in order to do that
    # we need them to have dummy jobs. So create the jobs in those cases
    # if they don't already exist.
    def _group_jobs_by_ws(self):
        self._ws_jobs = {}
        self._dummy_tissue_jobs = {}
        self._tissues_seen = set()
        from runner.models import Process
        for job in Process.objects.filter(
                    name__startswith=self._plugin+'_',
                    ).order_by('-id'):
            ws_id = self.ws_of_job(job)
            t_id = self.tissue_of_job(job)
            if not job.cmd:
                if t_id in self._dummy_tissue_jobs:
                    raise RuntimeError(
                            'tissue %d has multiple dummy jobs'%t_id
                            )
                self._dummy_tissue_jobs[t_id] = job.id
            if t_id in self._tissues_seen:
                # we've already got the most recent job for this tissue;
                # now only keep the successful ones
                if job.status != Process.status_vals.SUCCEEDED:
                    continue
            self._tissues_seen.add(t_id)
            self._ws_jobs.setdefault(ws_id,[]).append(job)
        print('got %d jobs across %d workspaces' % (
                sum([len(x) for x in self._ws_jobs.values()]),
                len(self._ws_jobs)
                ))
        self._tissues_seen = set() # reset this to be used again by convert_one
        # now go back and make sure there are dummy jobs for all the
        # comb and ext tissues; any that already exist have been noted above
        from browse.models import Tissue
        for t in Tissue.objects.filter(source__in=('comb','ext')):
            if t.id not in self._dummy_tissue_jobs:
                print('create dummy job, ws',t.ws_id,'tissue',t.id)
                bji = t._prep_foreground_sig('ltsconv')
                # and push the job on the front of the correct ws list
                l = self._ws_jobs.setdefault(ws_id,[])
                l.insert(0,bji.job)
    def convert_one_job(self,old_root,repo,job_rel,file_stats):
        from browse.models import Tissue
        need_push = False
        t_id = self.tissue_of_job(self.cur_job)
        first_job = (t_id not in self._tissues_seen)
        self._tissues_seen.add(t_id)
        for target in ('sigprot.tsv','sigqc.tsv','publish'):
            new_path = os.path.join(repo.path(),job_rel,target)
            if os.path.exists(new_path):
                file_stats[1] += 1
                continue
            if target == 'sigprot.tsv':
                old_path = os.path.join(
                            old_root,
                            'output/browse_significantprotein.tsv',
                            )
                if first_job and not os.path.exists(old_path):
                    old_path = PathHelper.storage # random valid path
                    # note this picks up dummy jobs as well;
                    # we want to extract data from the sigprot table,
                    # which requires a tissue id, but that doesn't
                    # get passed in; create a wrapper
                    def dump_sigprot_table_factory(tid):
                        def wrapper(old,new):
                            dump_sigprot_table(tid,new)
                        return wrapper
                    method = dump_sigprot_table_factory(t_id)
                else:
                    # old_path may not exist; that gets caught below
                    method = rewrite_sigprot
            elif target in ['sigqc.tsv']:
                old_path = os.path.join(
                            old_root,
                            'output',
                            target,
                            )
                method = copy
            elif target == 'publish':
                if not first_job:
                    continue
                from runner.models import Process
                if self.cur_job.status != Process.status_vals.SUCCEEDED:
                    continue
                parts = self.cur_job.name.split('_')
                if len(parts) != 4:
                    raise RuntimeError(self.cur_job.name)
                geoID = parts[1]
                old_path = os.path.join(
                        PathHelper.publish,
                        '%s_%d'%(geoID,t_id),
                        )
                method = sig_publish_dircopy
            if not os.path.exists(old_path):
                file_stats[2] += 1
                print(old_path,'does not exist')
                continue
            make_directory(os.path.dirname(new_path))
            method(old_path,new_path)
            if first_job and target == 'sigprot.tsv':
                # If we just converted the first (most recent) job for
                # a tissue, we should also update the tissue record at
                # this point.
                try:
                    t = Tissue.objects.get(pk=t_id)
                    print('updating tissue',t_id)
                    t.sig_result_job_id = self.cur_job.id
                    t._recalculate_sig_result_counts()
                    t.save()
                except Tissue.DoesNotExist:
                    print('skipping deleted tissue',t_id)
            file_stats[0] += 1
            need_push = True
        return need_push

SigConverter('sig')
################################################################################
# LTS Convert plugin
################################################################################
# This plugin acts on all workspaces, no matter which workspace it runs in.
# It seemed too daunting to get all the plugin logic to handle a case with
# no workspace, so this was the most expedient alternative. It's not registered
# in level_names, so it doesn't appear in the run dropdown. You need to type
# in the correct URL. This is probably a good thing. Use something like:
# /cv/6/job_start/ltsconv/
#
# The only negative consequence is that progress page links never appear.

class MyJobInfo(JobInfo):
    def get_config_html(self,ws,job_type,copy_job,sources=None):
        if copy_job:
            initial=copy_job.settings()
        else:
            initial=None
        form = ConfigForm(initial=initial)
        return form.as_p()
    def handle_config_post(self,jobname,jcc,user,post_data,ws,sources=None):
        form = ConfigForm(post_data)
        if not form.is_valid():
            return form.as_p()
        settings = dict(form.cleaned_data)
        settings['ws_id'] = ws.id
        job_id = jcc.queue_job(self.job_type,jobname
                            ,user=user.username
                            ,settings_json=json.dumps(settings)
                            )
        next_url = ws.reverse('nav_progress',job_id)
        return (None, next_url)
    def __init__(self,ws=None,job=None):
        # base class init
        super(MyJobInfo,self).__init__(
                ws,
                job,
                __file__,
                "LTS Convert",
                "LTS Legacy Output Conversion",
                )
        if self.job:
            self.log_prefix = str(self.ws)+":"
            self.debug("setup")
    def get_jobnames(self,ws):
        return [self.job_type] # no ws_id in name
    def run(self):
        make_directory(self.root)
        plugins=self.job.settings()['plugins'].split()
        p_wr = ProgressWriter(self.progress, [
                "wait for resources",
                ]+plugins+[
                "cleanup",
                ])
        self.rm = ResourceManager()
        self.rm.wait_for_resources(self.job.id,[1])
        p_wr.put("wait for resources","complete")
        touched_repos = set()
        batch_size=self.job.settings()['batch_size']
        for plugin in plugins:
            converter = Converter.name_index[plugin]
            status = converter.convert(batch_size=batch_size)
            p_wr.put(plugin,status or "complete")
            touched_repos |= converter.updated_repos
        p_wr.put("cleanup","marked %d repos"%len(touched_repos))
        return 0

if __name__ == "__main__":
    MyJobInfo.execute(logger)
