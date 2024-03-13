#!/usr/bin/env python3

program_description='''\
Manage LTS disk space usage.

In LTS, the contents of files themselves is stored in the 'annex', and only
symlinks to the files are stored in git. Freeing space is done by 'dropping'
annex files.

Annex files aren't versioned. Instead, the single annex holds all content on
all git branches. There are two strategies for dropping files:
- Files that originate in production, and so are backed up on S3, can just
  be explicitly dropped. The content will be recovered from S3 if it's needed
  again. The --keep-weeks option instructs lts_clean to drop files that haven't
  been accessed in a given number of weeks.
- Files that originate on a dev system aren't backed up, so they can only
  be removed after a new database snapshot is installed, and the branch
  associated with the old database has been removed. Removing the branch makes
  some content chunks 'unused', so git annex can drop them. The
  --prune-branches option invokes this functionality.

When run without options, this lists LTS repos in order by space consumption.
After using this to identify large space consumers, these can be reprocessed
with various cleanup options.
'''

# Other space-related lts_status commands:
# - space - report on space used by each lts repo
# - jobcount - report the number of jobs by CM type
# - branch - shows active branches, and which workspaces they're in
# - branch_detail - shows all branches, and whether they're active or synced
#   anywhere
# - drop_unused - may be used to drop unused content after manually
#   deleting branches or tags
# - rm_old - also space-related, but maybe obsolete; it appears to remove
#   old pre-lts files that were moved to lts

# TODO:
# - support cleanup of temporary job files outside LTS (ws/ws_id)
# - cleanup of non-annex uses of disk space
#   - this is an alternative to removing entire repositories (below)
#   - show_space now breaks down disk usage into annex, .git/objects, and
#     other (mostly the working tree itself).
#   - a symlink on ubuntu takes 4K, so even without any git annex content,
#     the repo can still consume significant space.  Checking out the
#     job0 tag can effectively eliminate this
#   - there are potential gains from 'git gc' and associated procedures
#     if a lot of space is used in the .git/objects directory
# - support removing a repo entirely from a dev machine (mostly this is
#   about not getting other nodes worried about there being copies of
#   annex content there)
#   - this should be recorded in the metadata by setting the trust to
#     'dead' as described here:
#     https://git-annex.branchable.com/git-annex-dead/
#     (Note that "repository" in that document is what we call a "remote".)
#   - this should be preceded by removal of any dev branches pushed to
#     the central machine

from path_helper import PathHelper
import subprocess
import os

def yield_file_leaves(root):
    # return each file path under root, except git internal files
    for name in os.listdir(root):
        if name == '.git':
            continue
        path = os.path.join(root,name)
        if os.path.isdir(path):
            for child in yield_file_leaves(path):
                yield child
        elif os.path.isfile(path):
            yield path

class JobFiles:
    # holds and returns info on all files for a job
    def __init__(self):
        self.raw_list = []
    def add_file(self,path):
        self.raw_list.append((
                path,
                os.path.getatime(path),
                os.path.getsize(path),
                ))
    def total_space(self):
        return sum([x[2] for x in self.raw_list])
    def most_recent_use(self):
        return sorted(self.raw_list)[-1]

def path_idxs(ws_root):
    # Both ws and log repos have a structure where the second layer is
    # a job id. This function returns indexes to the shard layer and
    # job id layer. Calls to this function mark places with assumptions
    # about the job id location in the path.
    shard_idx = len(ws_root.split('/'))
    return shard_idx,shard_idx+1

def make_per_job_index(ws_root):
    # returns a dict of JobFiles objects by job id
    result={}
    _,job_id_idx = path_idxs(ws_root)
    for path in yield_file_leaves(ws_root):
        if path == ws_root+'/empty':
            continue
        job_id = int(path.split('/')[job_id_idx])
        jf = result.setdefault(job_id,JobFiles())
        jf.add_file(path)
    return result

def clear_ws_repo(repo_name,cutoff,remove=False):
    ws_root=os.path.join(PathHelper.lts,repo_name)
    if not os.path.exists(ws_root):
        return
    # determine which jobs in a workspace have no files that have
    # been accessed inside the cutoff window
    d = make_per_job_index(ws_root)
    import datetime
    now = datetime.datetime.now()
    removed_space=0
    removed_jobs=[]
    for job_id in sorted(d):
        jf = d[job_id]
        path,access,size = jf.most_recent_use()
        t = datetime.datetime.fromtimestamp(access)
        if now - t > cutoff:
            removed_space += jf.total_space()
            removed_jobs.append(job_id)
    # show stats
    print('repo %s: will remove %d of %d jobs; will recover %0.0f MB' % (
            repo_name,
            len(removed_jobs),
            len(d),
            removed_space/(1024*1024.0),
            ))
    # optionally, drop files
    if not remove:
        return
    rel_paths = []
    shard_idx,job_id_idx = path_idxs(ws_root)
    for job_id in removed_jobs:
        jf = d[job_id]
        if jf.raw_list:
            l = ['/'.join(x[0].split('/')[shard_idx:job_id_idx+1])
                    for x in jf.raw_list
                    ]
            assert len(set(l)) == 1
            rel_paths.append(l[0])
    if rel_paths:
        from dtk.lts import LtsRepo
        r = LtsRepo(repo_name,LtsRepo.CURRENT_BRANCH)
        subprocess.check_call(
                ['git','annex','drop']+rel_paths,
                cwd=ws_root,
                env=r._aws_env(),
                )

def get_space_lookup(repo_names,suffix=''):
    assert repo_names,"No matching repos"
    usage = subprocess.check_output(
                    ['du','-s']+[x+suffix for x in repo_names],
                    cwd=PathHelper.lts,
                    encoding='utf8'
                    )
    result = {}
    for line in usage.split('\n'):
        if not line:
            continue
        size,root = line.split('\t')
        result[root] = int(size) * 1024
    return result

def show_space(repo_names):
    sub_parts = ['/.git/annex','/.git/objects']
    lookups = [get_space_lookup(repo_names,x) for x in sub_parts]
    recs = [(size,repo) for repo,size in get_space_lookup(repo_names).items()]
    recs.sort(key=lambda x:x[0])
    from dtk.text import fmt_size,print_table
    rows = [['repo','total size','annex','git objects','other']]
    totals = [0]*(2+len(sub_parts))
    for size,repo in recs:
        part_sizes = [d[repo+s] for d,s in zip(lookups,sub_parts)]
        other_size = size - sum(part_sizes)
        part_sizes.append(other_size)
        totals[0] += size
        for i,n in enumerate(part_sizes):
            totals[i+1] += n
        rows.append([
            repo,
            fmt_size(size),
            ]+[fmt_size(x) for x in part_sizes]
            )
    rows.append(['total']+[fmt_size(x) for x in totals])
    print_table(rows)

def check_names(repo_names):
    import os
    present = []
    for name in repo_names:
        if os.path.isdir(PathHelper.lts+name):
            present.append(name)
    print('found',len(present),'of',len(repo_names),'LTS repositories')
    return present

def parse_repo_range_item(item):
    if '-' in item:
        parts = item.split('-')
        assert len(parts) == 2, f"multiple hyphens in range '{item}'"
        import re
        m = re.search(r'([0-9]+$)',parts[0])
        assert m, f"character before hyphen must be numeric"
        start = m.group(0)
        prefix = parts[0][:-len(start)]
        return set(
                prefix+str(i)
                for i in range(int(start),int(parts[1])+1)
                )
    else:
        return set([item])

def prune_branches(repo_name):
    repo_path = PathHelper.lts+repo_name
    # We only want to delete branches that originated on this machine.
    # For dev machines, we extract the machine name from the lts_branch.
    # In production this will evaluate to 'master_', which will prevent
    # any branch dropping (but we also guard against ever getting here
    # in the first place for non-dev machines).
    local_branch_prefix = PathHelper.cfg('lts_branch').split('_')[0]+'_'
    # first make sure our list of local tracking branches is up-to-date,
    # so we don't get errors later trying to delete non-existent branches
    subprocess.check_call(
            ['git','remote','prune','origin'],
            cwd=repo_path,
            encoding='utf8'
            )
    # Assemble info on branches
    output = subprocess.check_output(
            ['git','branch','--all'],
            cwd=repo_path,
            text=True,
            ).rstrip('\n')
    blocked = set(['master','git-annex']) # never remove
    branch_info = {} # {stem:set(prefixes),...}
    for line in output.split('\n'):
        active = (line[0] == '*')
        branch = line[2:]
        try:
            idx = branch.rindex('/')
            prefix = branch[:idx+1]
            stem = branch[idx+1:]
        except ValueError:
            prefix = ''
            stem = branch
        branch_info.setdefault(stem,set()).add(prefix)
        if active:
            blocked.add(stem)
    print('LTS Repo',repo_name,'retaining:')
    # use list() because we modify the dict inside the loop
    for stem in list(branch_info):
        if stem in blocked or not stem.startswith(local_branch_prefix):
            print('  ',stem)
            del(branch_info[stem])
    # At this point, branch_info contains all the branches to be deleted.
    # Figure out which are tracking and which are local.
    remote = []
    remote_prefix = 'remotes/origin/'
    local = []
    for stem in branch_info:
        for prefix in branch_info[stem]:
            if prefix.startswith(remote_prefix):
                remote.append(prefix[len(remote_prefix):]+stem)
            elif prefix.startswith('remote/'):
                print('SKIPPING NON-ORIGIN REMOTE: '+prefix+stem)
            else:
                local.append(prefix+stem)
    if remote:
        print('Removing remote branches')
        for branch in remote:
            subprocess.check_call(
                    ['git','push','origin','--delete',branch],
                    cwd=repo_path,
                    encoding='utf8'
                    )
        subprocess.check_call(
                ['git','remote','prune','origin'],
                cwd=repo_path,
                encoding='utf8'
                )
    if local:
        print('Removing local branches:')
        for branch in sorted(local):
            print('  ',branch)
        subprocess.check_call(
                ['git','branch','-D']+local,
                cwd=repo_path,
                encoding='utf8'
                )
    if local or remote:
        print('Dropping unused content')
        remove_old_job_tags(repo_path)
        drop_unused(repo_path)

def remove_old_job_tags(repo_path):
    # The 'unused' command spends a lot of time cycling through all the
    # 'refs/tags/jobXXXXX' tags from the early days of LTS. So, drop those,
    # but retain the job0 tag, which is still used.
    output = subprocess.check_output(
            ['git','tag'],
            cwd=repo_path,
            text=True,
            ).rstrip('\n')
    tags_to_remove = [
            x
            for x in output.split('\n')
            if x.startswith('job') and x != 'job0'
            ]
    if tags_to_remove:
        print('removing old tags:',*sorted(tags_to_remove))
        subprocess.check_call(
                ['git','tag','-d']+tags_to_remove,
                cwd=repo_path,
                encoding='utf8'
                )

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

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=program_description,
            )
    parser.add_argument('--cur-branch',
            action='store_true',
            help='make sure configured branch is checked out',
            )
    parser.add_argument('--prune-branches',
            action='store_true',
            help='get rid of previous dev branches',
            )
    parser.add_argument('--keep-weeks',
            type=int,
            help='cutoff for old file checks',
            )
    parser.add_argument('--remove-old',
            action='store_true',
            help='remove old files (default is report only)',
            )
    parser.add_argument('repo_range',
            nargs='+',
            help='repo_name or repo_name-range_end_number',
            )
    args = parser.parse_args()
    # extract, assemble, and de-duplicate ws id list
    repo_names = set()
    for item in args.repo_range:
        repo_names |= parse_repo_range_item(item)
    from dtk.data import cond_int_key
    repo_names = sorted(repo_names,key=cond_int_key)
    # exclude any repos that don't actually exist (we don't want
    # this command to create any repos)
    repo_names = check_names(repo_names)
    # validate options
    actions_selected = set([
            a
            for a in [
                    'cur_branch',
                    'prune_branches',
                    'keep_weeks',
                    ]
            if getattr(args,a)
            ])
    if PathHelper.cfg('machine_type') != 'dev':
        bad_actions = actions_selected & set([
                'prune_branches',
                ])
        assert not bad_actions, 'Only allowed on dev machines: '+' '.join(
                sorted(bad_actions)
                )
    if not actions_selected:
        # an innocuous default action
        show_space(repo_names)
    # force any implied pre-reqs for specified actions
    if actions_selected and set(['keep_weeks','prune_branches']):
        # these imply the current branch must be selected, even if that's
        # not specified explicitly
        actions_selected.add('cur_branch')
    if 'cur_branch' in actions_selected:
        # current branch is a by-product of creating LtsRepo instance
        from dtk.lts import LtsRepo
        for repo_name in repo_names:
            r = LtsRepo.get(repo_name,PathHelper.cfg('lts_branch'))
    if 'prune_branches' in actions_selected:
        for repo_name in repo_names:
            prune_branches(repo_name)
    if 'keep_weeks' in actions_selected:
        # delete (or scan for) unused data
        # XXX Note that if keep_weeks is set to a value less than the time
        # XXX of the last snapshot, this may fail due to trying to delete
        # XXX locally-generated data not backed up on S3; we could check
        # XXX for this based on the branch name to produce a more helpful
        # XXX error message
        # express keep range as a timedelta
        import datetime
        cutoff = datetime.timedelta(weeks=args.keep_weeks)
        # process each workspace
        for repo_name in repo_names:
            clear_ws_repo(repo_name,cutoff,args.remove_old)
