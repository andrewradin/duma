# This file implements the Long-term Storage (LTS) function, which stores CM
# inputs and outputs using S3 and git-annex.  This provides:
# - resilient, inexpensive long-term storage on S3
#   - scores on local disks essentially operate as a cache, so disk
#     requirements are reduced
# - automatic compression, encryption, and de-duplication
#   - as in git, files are stored based on a hash of their contents, so files
#     with identical content are only stored once
#   - on S3, files are automatically compressed and encrypted
# - uniform mechanism for transferring files to/from workers
#   - inputs are checked in to LTS on the web machine, and checked out on the
#     worker; for outputs this is reversed. This extends the above caching and
#     deduplication advantages to web/worker communications.
# - easy sharing of production data with dev environments
#   - all environments share a single storage space
#   - production data uses the 'master' branch
#   - dev environments create new branches from master when loading DB
#     snapshots; this gives them access to all previous jobs as reflected
#     in their databases, and provides an isolated space for storing new
#     CM data
# - more scalable
#   - all data is stored in a single S3 bucket
#   - each workspace has a separate repo for managing meta-data
#
# Theory of Operation
# - git-annex extends git to allow external storage of large files
#   - inside git itself, the file is represented by a symlink
#   - git-annex keeps the file content separately
#   - not every remote needs to have every file
#     - git-annex tracks which files are on which remotes
#     - if the file is not present locally
#       - the symlink doesn't point to anything
#       - git-annex can retrieve the content and fix the symlink on request
#   - git-annex can use S3 as a "special remote"
#     - i.e. it's not really a git repo
#     - but it can be used to store content blobs (compressed and encrypted)
# - LTS is a thin python wrapper around git-annex
#   - it allows creating multiple repos (each with a unique name)
#   - it assumes an existing S3 bucket, which all repos share (using the repo
#     name as a prefix for all repo-specific content within the bucket)
#   - it assumes an existing server for holding 'bare' copies of all the
#     git repos (but not the git-annex content)
#     - this acts as a central point for machines to exchange repo updates,
#       without all machines needing to be visible to each other
# - a repo within LTS is accessed by creating an LtsRepo instance
#   - this will implicitly create the repo itself, if necessary, and assure
#     that it's instantiated locally and on the correct branch
#   - the path() method on the instance returns the absolute path of the local
#     git repository
#   - by convention, the repo name will be the workspace number, and content
#     within the repo will be organized as <plugin>/<job_id>/<input_or_output>
# - the typical life-cycle is:
#   - plugin instantiates an LtsRepo for its workspace
#   - plugin interacts with the worker, and gathers all output files locally
#   - plugin moves all permanent files in LTS repo, and calls the LtsRepo
#     lts_push() method to check in the output and send to S3
# - creating an LtsRepo instance requires a repo name and a branch name
#   - on web instances, the branch name is retrieved from local_settings to
#     minimize errors
# - branches are managed as follows:
#   - if a branch does not exist, it is assumed to be a new dev branch, and it
#     is automatically created as follows:
#     - a timestamp is extracted from the branch name
#     - the timestamp is used to find a 'last commit' on the master branch
#     - the new branch will be created based on that commit
#   - XXX once a dev environment is overwritten with a new DB snapshot, the
#     contents of the previous development branch is no longer useful; these
#     could be deleted and compressed away if necessary
# - log files and progress files are checked into a separate set of repos,
#   managed at the process level rather than the plugin level; this assures
#   they get pushed even if the job fails
# - after some experimentation, it turns out locking is needed to prevent
#   git failures, which propagate out. This is implemented by having a
#   lock file for each repo. The lock file is outside the repo itself, so
#   it can even be used during creation. Operations which should be atomic
#   (like the entire create sequence, or add followed by commit) are done
#   under a single lock. Since code may be invoked as part of different
#   atomic operations, the basic unlocked sequences are implemented as
#   private (leading _) functions, with the public versions wrapping them
#   in locking.
# - even with locking, "git annex sync" is especially prone to failure. It's
#   not fully protected by locking because it interacts with the bare repo,
#   which may be changed by another machine. It's not essential to any
#   atomic sequence, so it's implemented as a separate function with its
#   own lock, and a retry on failure.
#
# Manual one-time setup:
# - create lts server using aws_op.py
# - push credentials using scripts/authkeys.py
# - log in to lts server and install needed packages:
#   sudo apt-get git git-annex
#
# Test environments
# - If the repo name doesn't conflict with a production repo name
#   (currently, an integer or the word 'log' with an integer suffix)
#   then it can share the production support resources without issue.
#   This is done in LtsTestCase (which creates two repos called 'test'
#   and 'test2').
# - The underlying resouces for LTS are:
#   - a place to store the bare repo. This is defined by _lts_machine and
#     _bare_path.
#   - a bucket for building s3 remotes. This is coded in _get_s3_options().
#   - an s3 object name prefix. This is coded in _check_exists().
#   None of these things is easily overridden unless we add optional
#   init parameters for doing so.
#
# Useful tools:
# - 'git annex info' dumps lots of info about a repo.
# - 'git annex info <remote>' dumps lots of info about a remote
# - 'git cat-file -p git-annex:remote.log' is an easy way to look at the
#   remote.log file (or other things on the git-annex branch)

import logging
logger = logging.getLogger(__name__)

import os
import subprocess
from path_helper import PathHelper
from dtk.lock import FLock

class LtsRepo:
    _in_use_check=False
    _lts_path = PathHelper.lts
    from aws_op import Machine
    _lts_machine = Machine.name_index['lts']
    _placeholder_file='empty'
    _s3_remote_name='s3'
    CURRENT_BRANCH='CURRENT_BRANCH'
    @classmethod
    def aws_key(cls):
        return [
                x.rstrip('\n')
                for x in open(PathHelper.website_root+'.lts_keys')
                ][:2]
        # Note that this function is only used when running on machines
        # outside AWS. For running on AWS instances, S3 is accessed with
        # role-based keys.
    def __init__(self,repo_name,branch,local_lts_root=None):
        self._repo_name = repo_name
        self._cached_replication = None
        # always calculate bare path based on standard root
        self._bare_path=self._lts_machine.get_remote_path(self.path())+'.git'
        self._lock_path = PathHelper.timestamps+'lts_%s_lock'%self._repo_name
        self._token_expiry = None
        if local_lts_root:
            # allow multiple local repos for testing
            assert os.path.isabs(local_lts_root)
            self._lts_path=local_lts_root
        # set specified branch; allow the client to request the existing
        # branch (useful for lts_status script), but don't make it easy
        # to do by accident
        if branch == self.CURRENT_BRANCH:
            branch = self._current_branch()
        self._branch = branch
        # The placeholder file exists in every repository as an artifact of
        # needing an initial checkin to create the master branch.  It's also
        # held open here to prevent another process from switching branches.
        with FLock(self._lock_path) as lock:
            self._guard_file = os.path.join(self.path(),self._placeholder_file)
            self._check_exists()
            self._check_branch()
            if self._in_use_check:
                self._guard = open(self._guard_file)

    _cache = {}
    @classmethod
    def get(cls, repo_name, branch):
        """Returns a shared LtsRepo object that can be used for this repo."""
        key = (repo_name, branch)
        if key not in cls._cache:
            cls._cache[key] = LtsRepo(repo_name, branch)
        return cls._cache[key]

    #######
    # High-level interface
    #######
    def path(self):
        return os.path.join(
                self._lts_path,
                self._repo_name,
                )
    def lts_push(self,rel_path):
        with FLock(self._lock_path) as lock:
            self._add(rel_path, lock)
            if not self.is_replicated():
                return
            self._send(rel_path,lock)
        self.lts_sync()
    def lts_fetch(self,rel_path):
        # the following is a slight speed optimization: don't shell out
        # to git unless we see a broken symlink. It makes a difference
        # on the tissue page, where we try to fetch directories for
        # every tissue.
        if self._all_present(rel_path):
            return
        with FLock(self._lock_path) as lock:
            self._retry_git_cmd(['annex','get',rel_path],lock)
    #######
    # Low-level interface (mostly for ltsconv)
    #######
    def is_replicated(self):
        if self._cached_replication == None:
            pushed_ref = 'refs/remotes/origin/'+self._branch
            try:
                self._git_cmd(['show-ref',pushed_ref])
                self._cached_replication = True
            except subprocess.CalledProcessError:
                self._cached_replication = False
        return self._cached_replication
    def force_replication(self):
        self._git_cmd(['push','origin','-u',self._branch])
        self._cached_replication = None
    def lts_sync(self):
        with FLock(self._lock_path) as lock:
            self._retry_git_cmd(['annex','sync'],lock)
    def lts_add(self,rel_path):
        with FLock(self._lock_path) as lock:
            self._add(rel_path, lock)
    def lts_send(self,rel_path):
        with FLock(self._lock_path) as lock:
            self._send(rel_path,lock)
    #######
    # Internal methods
    #######
    def _add(self,rel_path,lock):
        self._retry_git_cmd(['annex','add',rel_path], lock)
        self._retry_git_cmd(['commit','-m','LTS push'], lock)
    def _send(self,rel_path,lock):
        self._retry_git_cmd(['annex','copy',rel_path,
                '--to='+self._s3_remote_name,
                '--fast',
                '--quiet',
                ],lock)
    def _bare_url(self):
        return self._lts_machine._ssh_dest()+':'+self._bare_path
    def _git_cmd(self,cmd,env=None):
        return subprocess.check_call(['git']+cmd, cwd=self.path(), env=env)
    def _all_present(self,rel_path):
        abs_path=os.path.join(self.path(),rel_path)
        for root,folders,files in os.walk(abs_path):
            for fn in files:
                path=os.path.join(root,fn)
                if os.path.islink(path) and not os.path.exists(path):
                    return False
        return True
    def _retry_git_cmd(self,cmd,lock):
        retries = 5
        while(retries):
            try:
                self._git_cmd(cmd,env=self._aws_env())
                break
            except subprocess.CalledProcessError:
                retries -= 1
                if not retries:
                    raise
                import time
                import random
                delay = 3*random.random()
                print('###########')
                print('# sleep %.2f and try again(%d)' % (delay,retries))
                print('###########')
                lock.release()
                time.sleep(delay)
                lock.acquire()
    # For normal IAM keys, the S3 special remote caches the keys in
    # .git/annex/creds. For role keys, this doesn't really work -- it
    # caches the access key and secret key, but not the session token.
    # So, to support role keys, we always pass the key data in the
    # environment to every git command. (The session token will be
    # properly handled by the S3 special remote if it's in the
    # environment.) See:
    # https://git-annex.branchable.com/forum/temporary_AWS_credentials/
    #
    # This change was committed on 2018-09-05 15:53:57 -0400.
    # This is supported in ubuntu 20.04, but not in 16.04.
    #
    # The code below handles constructing the environment, caching the
    # role tokens, and fetching new ones when they expire.
    def _check_role_token(self):
        import datetime
        now = datetime.datetime.utcnow()
        limit = datetime.timedelta(seconds=60*60)
        if self._token_expiry and self._token_expiry - now > limit:
            self._token_uses += 1
            return # token loaded and not expiring
        from dtk.aws_api import RoleAccessToken
        rat=RoleAccessToken(role='platform')
        self._role_token = rat.keys
        exp_str = rat.expiration.rstrip('Z')
        logger.debug(
                "%s loading role key %s with expiry %s after %d uses",
                self.path(),
                self._role_token['AWS_ACCESS_KEY_ID'],
                exp_str,
                0 if self._token_expiry is None else self._token_uses,
                )
        self._token_uses = 1
        self._token_expiry = datetime.datetime.fromisoformat(exp_str)
    def _aws_env(self):
        d = dict(os.environ)
        if PathHelper.cfg('LTS_role_keys'):
            self._check_role_token()
            d.update(**self._role_token)
        else:
            instance_key = self.aws_key()
            d.update(
                    AWS_ACCESS_KEY_ID=instance_key[0],
                    AWS_SECRET_ACCESS_KEY=instance_key[1],
                    )
        return d
    def _check_in_use(self):
        if not self._in_use_check:
            return
        try:
            subprocess.check_call(['lsof',self._guard_file])
        except subprocess.CalledProcessError:
            return # no other users
        raise RuntimeError('current branch in use')
    def _current_branch(self):
        # verify on correct branch
        self._last_symbolic_ref_output=subprocess.check_output([
                'git','symbolic-ref','HEAD',
                ],
                cwd=self.path(),
                )
        import re
        m=re.match('refs/heads/(.*)\n',self._last_symbolic_ref_output.decode('utf8'))
        return m.group(1)
    def _check_branch(self):
        if self._current_branch() == self._branch:
            return # repo exists and branch is correct
        logger.debug(
                "symbolic-ref returned '%s' from %s",
                repr(self._last_symbolic_ref_output),
                self.path(),
                )
        # verify no other users prior to switching branches
        self._check_in_use()
        # make sure local repo is up to date
        self._git_cmd(['fetch'])
        # attempt branch checkout, return if successful
        try:
            self._git_cmd(['checkout',self._branch])
            return
        except subprocess.CalledProcessError:
            pass
        # extract base date from branch name (which should end with
        # _before_YYYY.MM.DD.HH.MM)
        regex=r'.*_before_(\d{4})\.(\d\d)\.(\d\d)\.(\d\d)\.(\d\d)$'
        import re
        m=re.match(regex,self._branch)
        if not m:
            raise RuntimeError(
                    "can't extract date from '%s'"%self._branch,
                    )
        date='%s-%s-%s %s:%s'%(
                m.group(1),
                m.group(2),
                m.group(3),
                m.group(4),
                m.group(5),
                )
        # find corresponding master commit
        output=subprocess.check_output([
                'git','rev-list',
                '--before="%s 0000"'%date,
                '-n','1',
                'origin/master',
                ],
                cwd=self.path(),
                )
        commit = output.strip()
        if not commit:
            # This can happen when creating a workspace in a dev environment.
            # The base master commit gets created successfully, but it has
            # a date after the one the database snapshot is named for. At
            # this point it should be safe to fall back to using job0 as
            # the commit.
            commit = 'job0'
            logger.warning(
                    "no commit before '%s' falling back to %s"%(date,commit)
                    )
        logger.debug(
                "%s creating branch from %s for date %s",
                self.path(),
                commit,
                date,
                )
        # create branch from tag
        self._git_cmd(['checkout','-b',self._branch,commit])
        return
    def _clone(self):
        subprocess.check_call([
                'git',
                'clone',
                self._bare_url(),
                self.path(),
                ])
        from path_helper import current_user_name
        if current_user_name() == 'www-data':
            # avoid annoying "not configured" messages under www-data
            self._git_cmd(['config','--local','user.name','lts'])
            self._git_cmd(['config','--local','user.email','lts@twoxar.com'])
    def _get_s3_options(self):
        return [
                'type=S3',
                'chunk=1MiB',
                'datacenter=us-west-2',
                'encryption=shared',
                'bucket=2xar-duma-lts',
                ]
    def _check_exists(self):
        if os.path.isdir(self.path()):
            # repo already exists
            return
        if False:
            # This can be enabled for testing to prevent the code
            # from pulling down any workspace it doesn't already have,
            # and to help track down where the request came from.
            if not self._repo_name.startswith('log'):
                raise RuntimeError('new repos disabled')
        # In either case below, we need an AWS key for the commands to
        # succeed. Pre-fetch it here, so if it's not configured, we don't
        # end up with a half-created environment.
        aws_env = self._aws_env()
        # has the bare repo been created yet?
        m = self._lts_machine
        if m.run_remote_cmd('test -d '+self._bare_path, venv=None):
            # need to create bare repo
            m.check_remote_cmd('mkdir -p '+os.path.dirname(self._bare_path), venv=None)
            m.check_remote_cmd('git init --bare '+self._bare_path, venv=None)
            # clone repo
            self._clone()
            # create something to check in
            open(os.path.join(self.path(),self._placeholder_file),'w')
            # add, commit, and sync
            self._git_cmd(['add',self._placeholder_file])
            self._git_cmd(['commit','-m','create master branch'])
            self._git_cmd(['annex','init'])
            self._git_cmd(['annex','initremote','--fast',self._s3_remote_name]
                    + self._get_s3_options() + [
                    'fileprefix='+self._repo_name+'/',
                    ],
                    env=aws_env,
                    )
            self._git_cmd(['annex','sync'])
            # assure there's always one job tag
            tagname = 'job0' # for legacy reasons
            self._git_cmd(['tag',tagname])
            self._git_cmd(['push','origin',tagname])
        else:
            self._clone()
            self._git_cmd(['annex','enableremote',self._s3_remote_name],
                    env=aws_env,
                    )

################################################################################
# Maintenance classes
################################################################################
class PathCompressor:
    def __init__(self,keep=2):
        self.keep = keep
        self.paths = set()
        self.inputs = 0
    def add(self,path):
        self.inputs += 1
        self.paths.add('/'.join(path.split('/')[:self.keep]))
    def get(self):
        return sorted(self.paths)
    @property
    def outputs(self): return len(self.paths)

class RepoChecker:
    force = False
    unclean_report = []
    def __init__(self,repo_path,fix,remove):
        self.repo_path = repo_path
        self.repo_name = self.repo_path.split('/')[-1]
        self.fix = fix
        self.remove = remove
    def scan(self):
        steps = [
                'clean',
                'job0',
                'sync',
                's3',
                ]
        fixes = []
        for step in steps:
            # run the check function, and proceed to the next step if it
            # succeeds
            check_func = getattr(self,'check_'+step)
            error = check_func()
            if not error:
                continue
            # the check failed; do we have a fix for this failure?
            fix_name = 'fix_'+error.replace(' ','_')
            try:
                fix_func = getattr(self,fix_name)
            except AttributeError:
                fix_func = None
            if not self.fix or not fix_func:
                break # no fix, or not in fix mode; give up
            # try the fix function, record results
            fixes.append(fix_func())
            # is the check now working?
            error = check_func()
            if error:
                error += '; fix failed'
                break # no, give up
            # yes, keep going
        # report any fixes, and/or the error that aborted the cycle
        return '; '.join(fixes + [error or 'ok'])
    def check_clean(self):
        # Normally, everything in a repo is checked in by lts_push
        # at the end of a job, and then those files are never altered
        # again. The dev branch mechanism counts on this, creating
        # branches after the production database snapshot date.
        #
        # But, the plotly thumbnail mechanism creates png files on
        # demand, and these are located in the lts directory. So,
        # they will show up as errors here. On production machines,
        # it's ok to fix this by pushing the new files 'late'. They
        # may not show up on dev branches, but that's ok because the
        # dev environment will re-create them locally. On dev machines,
        # it's ok to fix this either by pushing the files or by
        # removing them. In the later case, they'll also get re-created
        # on demand.
        try:
            output = subprocess.check_output(
                        ['git','status','--porcelain'],
                        cwd=self.repo_path,
                        encoding='utf8'
                        )
        except subprocess.CalledProcessError:
            return 'not a repo'
        if output:
            l = [(x[:2],x[3:]) for x in output.split('\n')[:-1]]
            # no fix unless everything is an untracked file
            if set([x[0] for x in l]) != set(['??']):
                return 'unexpected file status'
            # get list of file names
            self.untracked_files = [x[1] for x in l]
            extra = self.log_unclean_files()
            if extra and not self.force:
                # this prevents the default fix logic from running
                return f'not clean {extra}'
            return 'not clean'
    def log_unclean_files(self):
        full_jobs = []
        unclassified = []
        thumbs = PathCompressor()
        bad_job_id = False
        import re
        for name in self.untracked_files:
            m = re.match(r'[^/]+/(\d+)/$',name)
            if m:
                # it's a completely unpushed job
                if int(m.group(1)) < 5000:
                    bad_job_id = True
                full_jobs.append(name)
                continue
            m = re.match(r'.+\.png$',name)
            if m:
                # it's an unpushed thumbnail
                thumbs.add(name)
                continue
            # it's an unexpected pattern
            unclassified.append(name)
        # report
        self.unclean_report = [
                f'unpushed full job {name}'
                for name in full_jobs
                ]
        if thumbs.inputs:
            self.unclean_report.append(
                    f'{thumbs.inputs} thumbnails in {thumbs.outputs} jobs'
                    )
        self.unclean_report += [
                f'unclassified path {name}'
                for name in unclassified
                ]
        # return fix classification
        if bad_job_id:
            return 'bad job id'
        if unclassified:
            return 'unclassified file pattern'
        assert self.repo_path.startswith(PathHelper.lts)
        r = LtsRepo(self.repo_name,LtsRepo.CURRENT_BRANCH)
        if r._current_branch() != PathHelper.cfg('lts_branch'):
            return 'wrong branch'
        return ''
    def check_job0(self):
        output = subprocess.check_output(
                    ['git','tag'],
                    cwd=self.repo_path,
                    encoding='utf8'
                    )
        if 'job0' not in output:
            return 'missing job0 tag'
            # this means the repo creation failed at some point;
            # you'll need to go through each creation step manually
            # after the last one that succeeded
    def check_sync(self):
        local_rev=subprocess.check_output(
                        ['git','rev-parse','@'],
                        cwd=self.repo_path,
                        encoding='utf8'
                        ).strip()
        try:
            remote_rev=subprocess.check_output(
                            ['git','rev-parse','@{u}'],
                            cwd=self.repo_path,
                            stderr=subprocess.STDOUT, # hide (expected) error
                            encoding='utf8'
                            ).strip()
        except subprocess.CalledProcessError:
            return 'not replicated'
            # This is the expected state for dev machines.
            # Since there's no 'fix_' method, this terminates the check.
        if local_rev != remote_rev:
            return 'not synced with remote'
    def check_s3(self):
        output = subprocess.check_output(
                        ['git','annex','list'],
                        cwd=self.repo_path,
                        encoding='utf8'
                        ).rstrip('\n')
        in_header = True
        remote_col = {}
        pc = PathCompressor()
        for line in output.split('\n'):
            if in_header:
                if line.endswith('|'):
                    in_header=False
                    path_col = max(remote_col.values())+2
                else:
                    col=0
                    try:
                        col+=(1+line.rindex('|'))
                    except ValueError:
                        pass
                    remote_col[line[col:]] = col
                continue
            if line[remote_col['s3']] != 'X':
                pc.add(line[path_col:])
        self.unpushed = pc.get()
        if self.unpushed:
            return 'some content not on S3'
    # If a repo contains stuff that isn't checked in, it can be dealt with
    # either by checking it in, or by deleting it. For production/master,
    # checking it in is always the correct thing to do. For dev environments,
    # it should be checked in if it was generated on the current branch, but
    # deleted if it was generated on a previous branch.
    #
    # The only way to be sure that something isn't from a previous branch is
    # to make sure all the repos are clean when a new branch is configured.
    # Since you'll typically never return to a previous dev branch, you can
    # safely delete things from the branch being left at that point. So the
    # procedure for switching to a new branch should be:
    # - run lts_status.py --remove --fix check
    # - configure the new branch in local_settings.py
    # - load the new DB snapshot
    # This is handled by import_prod_db.sh.
    def fix_not_clean(self):
        if self.remove:
            import os
            import shutil
            for x in self.untracked_files:
                target=os.path.join(self.repo_path,x)
                if os.path.isdir(target):
                    shutil.rmtree(target)
                else:
                    os.remove(target)
            return 'not clean - fixed via removal'
        else:
            # construct LtsRepo object
            assert self.repo_path.startswith(PathHelper.lts)
            repo_name = self.repo_path[len(PathHelper.lts):]
            r = LtsRepo(repo_name,LtsRepo.CURRENT_BRANCH)
            pc = PathCompressor()
            for x in self.untracked_files:
                pc.add(x)
            for x in pc.get():
                r.lts_push(x)
            return 'not clean - fixed via push'
    def fix_not_synced_with_remote(self):
        subprocess.check_call(
                    ['git','push'],
                    cwd=self.repo_path,
                    )
        return 'not synced - fixed via push'
    def fix_some_content_not_on_S3(self):
        #print('entering fix function')
        #print(len(self.unpushed),'directories to push')
        assert self.repo_path.startswith(PathHelper.lts)
        r = LtsRepo(self.repo_name,LtsRepo.CURRENT_BRANCH)
        for i,x in enumerate(sorted(self.unpushed)):
            print('sending',i+1,'of',len(self.unpushed),end='\r',flush=True)
            r.lts_send(x)
        #print('about to issue sync')
        r.lts_sync()
        #print('leaving fix function')
        return 'some content not on S3 - fixed via send'


