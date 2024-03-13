#!/usr/bin/env python3

import subprocess
from path_helper import PathHelper

# XXX To do:
# XXX - if this is run while a job is in progress, the log file gets updated
# XXX   through the symlink, and the content of the annex object no longer
# XXX   matches the hash. This needs to be fixed manually, after the job
# XXX   completes:
# XXX   - su to www-data
# XXX   - cd /home/www-data/lts/log#/###/#####/publish
# XXX   - git annex unlock bg_log.txt
# XXX   - git annex add bg_log.txt
# XXX   - git commit
# XXX   - then run another lts_status.py --repo log# --fix check
# XXX - Rather than fixing the above, once we've got MaintenanceLocks deployed,
# XXX   wrap any potentially problematic code with something that assures a
# XXX   MaintenanceLock is in place.

# XXX New LTS status checking architecture:
# XXX - a maintenance task mechanism that supports pausing the jobs queue
# XXX - a batch-based scanner that locates LTS repos that have problems,
# XXX   but doesn't fix them
# XXX - individual repo fix jobs that handle common cases found by the scanner
# XXX - manual repair of other individual repos (maybe somehow leveraging the
# XXX   maintenance task mechanism to pause the job queue for manual repairs)
# XXX
# XXX Action plan:
# XXX + implement locking mechanism, without maintenance task launch
# XXX + implement maintenance task launch
# XXX + implement manual maintenance lock
# XXX (at this point, we can safely do manual single-repo scans and fixes
# XXX by asserting a manual maintenance lock first)
# XXX + show active maintenance locks on job page
# XXX + create scan batched maintenance task
# XXX   + move RepoChecker and PathCompressor to dtk
# XXX - create cron job for kicking off scan task -- since there's a lot of
# XXX   repair accumulated on platform, we should first do a release to deploy
# XXX   manual operation with locking. Once we've used that to clean up the
# XXX   backlog of needed repairs, we can implement cron-based reporting for
# XXX   subsequent errors
# XXX (further automation could include integrating the manual lock into
# XXX this script for simpler manual operation, or integrating auto-fixing)
# XXX

banner = '#'*30

def get_repos(progress=True):
    from dtk.files import scan_dir
    from dtk.data import cond_int_key
    src = sorted(
            scan_dir(PathHelper.lts,output=lambda x:x),
            key = lambda x: cond_int_key(x.filename),
            )
    if progress:
        from tqdm import tqdm
        src = tqdm(src)
    for x in src:
        yield x

def get_jobtypes(repo_path):
    from dtk.files import scan_dir
    for jobtype in scan_dir(repo_path,output=lambda x:x):
        if jobtype.filename in ('empty','.git'):
            continue
        yield jobtype

def get_whereis_groups(repo_path):
    output = subprocess.check_output(
                    ['git','annex','whereis'],
                    cwd=repo_path,
                    encoding='utf8'
                    )
    lines = output.rstrip('\n').split('\n')
    start_idxs = [
            i
            for i,line in enumerate(lines)
            if line.startswith('whereis')
            ]
    for start,end in zip(start_idxs,start_idxs[1:]+[len(lines)]):
        yield lines[start:end]

def do_check(args):
    from dtk.lts import RepoChecker
    if args.repo:
        # single-repo case
        rc = RepoChecker(
                repo_path = PathHelper.lts+args.repo,
                fix = args.fix,
                remove = args.remove,
                )
        if args.force:
            rc.force = True
        print(rc.scan())
        if args.detail:
            for line in rc.unclean_report:
                print(line)
        return
    if PathHelper.cfg('machine_type') == 'platform' and args.fix:
        raise NotImplementedError('repair one-by-one')
    if args.force or args.detail:
        raise NotImplementedError('--force and --detail need --repo')
    total = 0
    replicated = 0
    errors = 0
    for repo in get_repos():
        if args.except_repo and repo.filename in args.except_repo:
            continue
        total += 1
        try:
            rc = RepoChecker(
                    repo.full_path,
                    fix = args.fix,
                    remove = args.remove,
                    )
            result = rc.scan()
            if result != 'ok':
                if not result.startswith('ok;'):
                    errors += 1
                print('repo',repo.filename,'-',result)
            else:
                replicated += 1
        except:
            print('got exception checking',repo.filename)
            raise
    print('checked %d repos; %d replicated; %d errors'%(total,replicated,errors))

s3_whereis_ending = ' -- [s3]'
here_whereis_ending = ' -- [here]'

def do_s3detail(args):
    repo_path = PathHelper.lts+args.repo
    total=0
    s3_cnt=0
    here_cnt=0
    for group in get_whereis_groups(repo_path):
        total += 1
        if not any([x.endswith(s3_whereis_ending) for x in group[1:-1]]):
            parts = group[0].split()
            print(parts[1],'not on s3')
        else:
            s3_cnt += 1
        if any([x.endswith(here_whereis_ending) for x in group[1:-1]]):
            here_cnt += 1
            if args.show_here:
                parts = group[0].split()
                print(parts[1],'here')
    print(total,'objects checked;',here_cnt,'here,',s3_cnt,'on s3')
    
def sync_repo(repo_name):
    # Make sure the repo has up-to-date information about S3 content.
    # For a replicated branch, all S3 content originates locally, and
    # we just need to be sure that content indexing is shared via a
    # git annex sync. For dev branches, all the non-local content we
    # care about should be known to the central bare repo when the local
    # branch was created, and it gets pulled down via a git fetch at that
    # time.  But if some problem has delayed the S3 write by production,
    # we can repeat the git fetch here after the problem is resolved.
    from dtk.lts import LtsRepo
    r = LtsRepo(repo_name,LtsRepo.CURRENT_BRANCH)
    if r.is_replicated():
        r.lts_sync()
    else:
        r._git_cmd(['fetch'])

def do_sync(args):
    if args.repo:
        # single-repo case
        sync_repo(args.repo)
        return
    for repo in get_repos():
        if args.except_repo and repo.filename in args.except_repo:
            continue
        sync_repo(repo.filename)

def do_jobcount(args):
    from dtk.files import scan_dir
    result = {}
    for repo in get_repos():
        if args.except_repo and repo.filename in args.except_repo:
            continue
        for jobtype in get_jobtypes(repo.full_path):
            d = result.setdefault(jobtype.filename,dict())
            d[repo.filename] = len(list(scan_dir(jobtype.full_path)))
    for jobtype,d in result.items():
        print(jobtype,'has',sum(d.values()),'jobs in',len(d),'repo(s)')
        # XXX add a detail option?

def do_rm_old(args):
    targets = {}
    import glob
    for cm in 'capp gesig path'.split():
        targets[cm+'_pub'] = glob.glob(PathHelper.publish+'*/'+cm)
        targets[cm+'_ws'] = glob.glob(PathHelper.storage+'*/'+cm)
    # path
    targets['path_detail'] = glob.glob(PathHelper.storage+'*/paths.*.tsv.gz')
    # log
    targets['log_pub'] = glob.glob(PathHelper.publish+'bg_logs')
    targets['log_ws'] = glob.glob(PathHelper.storage+'*/*/*/progress')
    # sig
    targets['sig_pub'] = [
            x
            for x in glob.glob(PathHelper.publish+'*_*')
            if not x.endswith('/bg_logs')
            ]
    targets['sig_ws'] = glob.glob(PathHelper.storage+'*/sig')
    for key in sorted(targets.keys()):
        print(key,'-',len(targets[key]),'objects to remove')
    if args.remove:
        import os
        import shutil
        for key in sorted(targets.keys()):
            for target in targets[key]:
                print('removing',target)
                if os.path.isfile(target):
                    os.remove(target)
                else:
                    shutil.rmtree(target)

def repo_branch(repo_path):
    import os
    return open(os.path.join(repo_path,'.git/HEAD')).read().strip()

def do_branch(args):
    if args.repo:
        # single-repo case
        print("%s: %s" % (
                args.repo,
                repo_branch(PathHelper.lts+args.repo)
                ))
        return
    result = {}
    prefix = 'ref: refs/heads/'
    for repo in get_repos():
        if args.except_repo and repo.filename in args.except_repo:
            continue
        branch = repo_branch(repo.full_path)
        if branch.startswith(prefix):
            branch = branch[len(prefix):]
        result.setdefault(branch,[]).append(repo.filename)
    from dtk.data import cond_int_key
    for branch in sorted(result.keys()):
        l = result[branch]
        print("%s: %d repos:\n  (%s)" % (
                branch,
                len(l),
                ' '.join(sorted(l,key=cond_int_key)),
                ))

def do_branch_detail(args):
    skip = set(['master','git-annex'])
    synced_branches = {}
    active_branches = {}
    all_branches = {}
    for repo in get_repos():
        if repo.filename in skip:
            continue
        output = subprocess.check_output(
                ['git','branch'],
                cwd=repo.full_path,
                text=True,
                ).rstrip('\n')
        synced_prefix = 'synced/'
        for line in output.split('\n'):
            active = (line[0] == '*')
            branch = line[2:]
            if branch.startswith(synced_prefix):
                branch = branch[len(synced_prefix):]
                synced_branches.setdefault(branch,set()).add(repo.filename)
            else:
                all_branches.setdefault(branch,set()).add(repo.filename)
                if active:
                    active_branches.setdefault(branch,set()).add(repo.filename)
    for branch in sorted(all_branches):
        out = branch
        if branch in active_branches:
            out += '; active in '+' '.join(sorted(active_branches[branch]))
        if branch in synced_branches:
            out += '; synced in '+' '.join(sorted(synced_branches[branch]))
        print(out)

def drop_unused(repo_path):
    subprocess.check_call(
            ['git','annex','unused'],
            cwd=repo_path,
            encoding='utf8'
            )
    subprocess.check_call(
            ['git','annex','dropunused','all','--force'],
            cwd=repo_path,
            encoding='utf8'
            )

def do_drop_unused(args):
    # this in normally done by lts_clean after it drops branches;
    # it exists here so we can force it even if a branch isn't dropped
    if args.repo:
        # single-repo case
        drop_unused(PathHelper.lts+args.repo)
        return
    for repo in get_repos():
        if args.except_repo and repo.filename in args.except_repo:
            continue
        drop_unused(repo.full_path)

def commit_config(repo_path):
    subprocess.check_call(
            ['git','config','--local','user.name','lts'],
            cwd=repo_path,
            encoding='utf8'
            )
    subprocess.check_call(
            ['git','config','--local','user.email','lts@twoxar.com'],
            cwd=repo_path,
            encoding='utf8'
            )

def set_origin(repo_name,ssh_dest):
    repo_path = PathHelper.lts+repo_name
    subprocess.check_call(
            ['git','remote','set-url','origin',
                    f'{ssh_dest}:2xar/lts/{repo_name}.git'
                    ],
            cwd=repo_path,
            encoding='utf8'
            )

def do_set_origin(args):
    from aws_op import Machine
    mch = Machine.name_index['lts']
    if args.repo:
        # single-repo case
        set_origin(args.repo,mch._ssh_dest())
        return
    for repo in get_repos():
        if args.except_repo and repo.filename in args.except_repo:
            continue
        set_origin(repo.filename,mch._ssh_dest())

def do_commit_config(args):
    if args.repo:
        # single-repo case
        commit_config(PathHelper.lts+args.repo)
        return
    for repo in get_repos():
        if args.except_repo and repo.filename in args.except_repo:
            continue
        commit_config(repo.full_path)

def rekey(path,key=None):
    # git annex stores AWS keys in a file with a UUID-based name in
    # .git/annex/creds for accessing the s3 special remote. Running
    # enableremote will update these stored keys.
    #
    import os
    env = dict(os.environ)
    if not key:
        from dtk.lts import LtsRepo
        key = LtsRepo.aws_key()
    env.update(
            AWS_ACCESS_KEY_ID=key[0],
            AWS_SECRET_ACCESS_KEY=key[1],
            )
    subprocess.check_call(
            ['git','annex','enableremote','s3'],
            cwd=path,
            env=env,
            encoding='utf8'
            )

def do_rekey(args):
    if args.aws_key:
        key = args.aws_key.split()
    else:
        key = None # use default from .lts_keys
    if args.repo:
        # single-repo case
        rekey(PathHelper.lts+args.repo,key)
        return
    for repo in get_repos():
        if args.except_repo and repo.filename in args.except_repo:
            continue
        rekey(repo.full_path,key)

if __name__ == '__main__':
    if PathHelper.cfg('machine_type') == 'platform':
        # running this in production requires:
        # - running as www-data
        # - HOME=/var/www (so git-annex can find .gnupg)
        # - the conda version of python, which includes all needed setup
        # Provide user with instruction if this is not the case
        import os
        if not all([
                os.environ[var] == val
                for var,val in (
                        ('USER','www-data'),
                        ('HOME','/var/www'),
                        )
                ]):
            py_path = PathHelper.venv+'envs/py3web1/bin/python'
            cmd_path = PathHelper.website_root+'scripts/lts_status.py'
            print(f'''
On platform, this command must be run as:
sudo -H -u www-data {py_path} {cmd_path}
''')
            import sys
            sys.exit(1)
    prefix = 'do_'
    cmds = [
            key[len(prefix):]
            for key in locals().keys()
            if key.startswith(prefix)
            ]
    import argparse
    parser = argparse.ArgumentParser(
            description="administer LTS repos"
            )
    parser.add_argument('--repo')
    parser.add_argument('--except-repo',nargs='*')
    parser.add_argument('--show-here',action='store_true')
    parser.add_argument('--fix',
            action='store_true',
            help='attempt to fix errors (for check)',
            )
    parser.add_argument('--remove',
            action='store_true',
            help='enable removal (for rm_old); remove uncommited files (for check --fix)',
            )
    parser.add_argument('--detail',
            action='store_true',
            help='show unclean detail (for single-repo check)',
            )
    parser.add_argument('--force',
            action='store_true',
            help='bypass extra fix protections (for single-repo check)',
            )
    parser.add_argument('--skip-s3',
            action='store_true',
            help='don\'t compare with S3 (for check)',
            )
    parser.add_argument('cmd',choices=cmds)
    parser.add_argument('--aws-key',
            help='access and secret keys, space-separated'
            )
    args = parser.parse_args()

    func = locals()[prefix+args.cmd]
    func(args)
