import logging
logger = logging.getLogger(__name__)

etl_tiers={
        1:[
               'uniprot',
#                'd2ps',
                'disgenet',
                'openTargets',
                'string',
                'reactome',
                'pathways',
                'unichem',
                'drugnames',
                'matching',
                    'chembl',
                    'bindingdb',
                    'drugbank',
                    'cayman',
                    'med_chem_express',
                    'selleckchem',
                    'duma',
                ],
        2:[
                'aact',
                'duma_gwas',
                    'grasp',
                    'gwasCat',
                    'phewas',
                    'ukbb',
                'faers',
                    'name2cas',
                'meddra',
                'umls',
                'gene_sets',
                'targetScan',
                'ucsc_hg',
                'zinc',
                'efo',
                ],
        3:[
                'atc',
                'homologene',
                'aesearch_training',
                'orange_book',
                'surechembl',
                'global_protsets',
                'uniprot_dpi',
                'salmon',
                'CrossMap',
                ],
        4:[
                'dpi_bg_scores',
                'broad',
                'probeMiner',
                'sider',
                'stitch',
                ],
        5:[
                'cpdb',
                'cvarod',
                'MDF',
                'adrecs',
                'aeolus',
                'allTFs',
                'brenda',
                'offsides',
                'drpias',
                'geo',
                'global_protsets',
                'gottlieb',
                'humanProteinAtlas',
                'lincs',
                'ncats',
                'tcga_ge',
                'ttd',
                'uniprot_human_diseases',
                ],
        }


def get_all_etl_names():
    from path_helper import PathHelper
    from dtk.files import scan_dir
    from os.path import isdir
    return set(
                x.filename
                for x in scan_dir(
                        PathHelper.databases,
                        output=lambda x:x,
                        filters=[lambda x:isdir(x.full_path)]
                        )
                if x.filename not in {
                        'version_tools',
                        '__pycache__',
                        }
                )

def get_etl_name_lookup():
    # Each ETL source is known by its directory name. These names may be
    # mixed case, but the make variables representing dependencies (and
    # so, the versions.py attribute names) are all upper-case. This means
    # the original directory name can't be fully reconstructed from the
    # dependency attribute. This dict allows it to be looked up instead.
    all_etl_names = get_all_etl_names()
    return {x.lower():x for x in all_etl_names}

def get_etl_dir(name):
    from path_helper import PathHelper
    return PathHelper.databases+name

def get_versions_namespace(name):
    vpath = get_etl_dir(name)+'/versions.py'
    try:
        text = open(vpath).read()
    except IOError:
        return None
    out = dict()
    exec(compile(text, vpath, 'exec'),globals(),out)
    return out

def get_last_published_version(name):
    ppath = get_etl_dir(name)+'/last_published_version'
    try:
        return int(open(ppath).read().strip())
    except IOError:
        return None

def get_etl_data(name, name_lookup = None, version=None, ns=None):
    class Dummy:
        def __init__(self):
            # fill in defaults
            self.deprecated = False
            self.latest_version = 0
            self.description = ''
            self.published = ''
            self.source_version = ''
            self.dependencies = []
            self.other_info = []
            self.last_check = None
        def dep_dirs(self):
            return set(src for src,version in self.dependencies)
    result = Dummy()
    result.name = name
    l = [k for k,v in etl_tiers.items() if name in v]
    result.tier = l[0] if len(l) == 1 else 0
    # look for and load versions.py file
    if ns is None:
        ns = get_versions_namespace(name)
    if ns:
        if ns.get('deprecated'):
            result.deprecated = True
        result.months_between_updates = ns.get('months_between_updates',3)
        result.last_check = ns.get('last_checked')
        versions = ns['versions']
        if not name_lookup:
            name_lookup = get_etl_name_lookup()
        result.latest_version = get_last_published_version(name)
        if not result.latest_version:
            result.latest_version = max(versions)
            result.published = 'NOT PUBLISHED'
        else:
            import subprocess
            pubfile='last_published_version'
            cmd = ['git', 'status', '--porcelain',pubfile]
            cp = subprocess.run(cmd,
                    text=True,capture_output=True,
                    cwd=get_etl_dir(name),
                    )
            if cp.stdout.startswith(' M '):
                # it's locally modified -- use last modified time
                import os
                pubpath = os.path.join(get_etl_dir(name),pubfile)
                ts = os.path.getmtime(pubpath)
                import datetime
                dt=datetime.datetime.fromtimestamp(ts)
                result.published = dt.strftime('%Y-%m-%d')
            else:
                cmd = ['git','log',
                        '-n','1',
                        '--format=%ad','--date=short',
                        'last_published_version',
                        ]
                cp = subprocess.run(cmd,
                        text=True,capture_output=True,
                        cwd=get_etl_dir(name),
                        )
                result.published = cp.stdout.strip()
            if not result.published:
                logger.warning(f"can't retrieve date; cp:{cp}")
                raise RuntimeError(f'date retrieval error on {name}')
        if not version:
            # if a specific version isn't requested, return info on the
            # last published version
            version = result.latest_version
        result.version = version
        info = versions[version]
        result.description = info.get('description', '')
        ver_key = lambda x: x[:-4].lower()
        ver_id_keys = [x for x in info
                       if x.endswith('_VER') and ver_key(x) in name_lookup]
        shown = set(['description']+ver_id_keys)
        inputs = {
                name_lookup[ver_key(x)]:info[x]
                for x in ver_id_keys
                }
        result.source_version = inputs.get(result.name, '')
        result.dependencies = sorted([
                x for x in inputs.items()
                if x[0] != result.name
                ])
        result.other_info = sorted([
                x for x in info.items()
                if x[0] not in shown
                ])
    return result

def get_all_versioned_etl():
    result = {}
    name_lookup = get_etl_name_lookup()
    for src in name_lookup.values():
        info = get_etl_data(src,name_lookup)
        if info.latest_version and not info.deprecated:
            result[src] = info
    return result

def order_etl(etl_set):
    remaining = set(etl_set)
    done = set()
    passes = []
    try:
        matching = [x for x in etl_set if x.name == 'matching'][0]
    except IndexError:
        matching = None
    while remaining:
        this_pass = set()
        for info in remaining:
            if info.dep_dirs() - done:
                continue # can't do this one yet
            this_pass.add(info)
        if not this_pass:
            # We're not done, but we can't proceed normally. Check for one
            # special case of a mutual dependency between the matching
            # directory and an ETL source. In this case, we want to do the
            # anything which just depends on 'matching' during this pass,
            # and then we'll pick up the matching directory itself during
            # the next pass.
            for info in remaining:
                if info.dep_dirs()-done == set(['matching']) \
                        and matching \
                        and info.name in matching.dep_dirs():
                    this_pass.add(info)
        assert this_pass # else, there's a cycle and we can't progress
        passes.append(this_pass)
        done |= set(x.name for x in this_pass)
        remaining -= this_pass
    return passes

def get_etl_dependencies(name, version=None):
    all_data = get_etl_data(name, version=version)
    return all_data.dependencies

def latest_ucsc_hg_in_gwas():
    temp = [x[1] for x in get_etl_dependencies('duma_gwas')
               if x[0]=='ucsc_hg'
            ]
    assert len(temp) ==1, 'incorrect number of UCSC_HG dependencies in GWAS'
    return temp[0]


####
# Code for generated ETL stats in each ETL directory.
####
def make_single_diff_stat(before, after):
    def fmt_pct(cnt):
        if before == 0:
            return 0
        return f'{cnt*100/before:.1f}%'

    if isinstance(before, (int, float)) or isinstance(after, (int, float)):
        if isinstance(before, set):
            before = len(before)
        if isinstance(after, set):
            after = len(after)
        return dict(
            before=before,
            after=after,
            chg_pct=fmt_pct(after-before),
            )
    else:
        added = len(after - before)
        removed = len(before - after)
        kept = len(before & after)
        after = len(after)
        before = len(before)
        return dict(
            before=before,
            after=after,
            chg_pct=fmt_pct(after-before),
            added=added,
            added_pct=fmt_pct(added),
            removed=removed,
            removed_pct=fmt_pct(removed),
            kept=kept,
            kept_pct=fmt_pct(kept),
            )

def make_diff_stats(before_cols, after_cols):
    keys = before_cols.keys() | after_cols.keys()
    out = {}
    for key in keys:
        before = before_cols.get(key, set())
        after = after_cols.get(key, set())
        out[key] = make_single_diff_stat(before, after)
    return out

def run_stats_main(data_func, files):
    """
    data_func should generate a dict of sets & numbers, representing key collections & counts for this version.
    Those will be diffed against the previous version.

    files should be a list of files with a {version} fill-in.  We will find it either in the local directory
    or in the s3 versioned storage.
    The resolved filenames from files will be passed as arguments to data_func.
    """
    import argparse
    from dtk.log_setup import setupLogging
    setupLogging()
    parser = argparse.ArgumentParser(description='path utility')
    parser.add_argument('--version', type=int, help='Version number')
    parser.add_argument('--output', help='Where to write stats')
    args = parser.parse_args()

    file_class = files[0].split('.')[0]

    from path_helper import PathHelper
    def resolve(fn):
        """Finds which dir this file is in or fetches it, returning the path.

        Checks [cwd, 2xar/ws/<file_class>, s3 fileclass bucket].
        """
        import os
        if os.path.exists(fn):
            return fn
        fn = os.path.join(PathHelper.storage, file_class, fn)
        if os.path.exists(fn):
            return fn
        
        from subprocess import check_call, CalledProcessError
        try:
            # Fetch via move_s3_files; should this be in dtk?
            move_s3 = os.path.join(PathHelper.databases, 'matching', 'move_s3_files.py')
            check_call([move_s3, '', fn])
            return fn
        except CalledProcessError:
            import traceback as tb
            tb.print_exc()
            # We rely on this behavior when the set of files changes.
            return None


    def make_ver(version):
        ver_files = [resolve(x.format(version=version)) for x in files] 
        logger.info(f'For v{version} using files {ver_files}')
        return data_func(*ver_files)

    latest = make_ver(args.version)
    if args.version == 1:
        prev = {}
    else:
        prev = make_ver(args.version - 1)

    diff_stats = make_diff_stats(prev, latest)

    import pandas as pd
    df = pd.DataFrame(diff_stats)
    print(df)

    df.to_csv(args.output, sep='\t')



def generate_all_stats(etl_name):
    import os
    import pandas as pd
    import subprocess
    from subprocess import CalledProcessError
    etl_dir = get_etl_dir(etl_name)
    script = os.path.join(etl_dir, 'stats.py')
    if not os.path.exists(script):
        return [], []
    
    latest_ver = get_last_published_version(etl_name)

    dfs = []
    for ver in range(1, latest_ver+1):
        try:
            statsfile = f'statsfile.v{ver}.tsv'
            statspath = os.path.join(etl_dir, statsfile)
            if not os.path.exists(statspath):
                subprocess.check_call(['make', statsfile], cwd=etl_dir)
            dfs.append(pd.read_csv(statspath, sep='\t', index_col=0))
        except CalledProcessError:
            dfs.append(pd.DataFrame([]))

    

    cols = dfs[-1].columns.tolist()

    return cols, dfs


def stats_to_plots(col, dfs):
    def get_or_0(df, row):
        try:
            return float(df.loc[row, col])
        except KeyError:
            return 0

    values = [get_or_0(df, 'after') for df in dfs]
    added = [get_or_0(df, 'added') for df in dfs]
    removed = [-get_or_0(df, 'removed') for df in dfs]
    import plotly.express as px
    import plotly.graph_objects as go
    from dtk.plot import PlotlyPlot
    import plotly.io as pio
    pio.templates.default = 'none'
    x_range = list(range(1,len(dfs)+1))
    fig = px.scatter(title=col, x=x_range, y=values)
    fig.add_trace(go.Bar(x=x_range, y=added, name='added', marker_color='teal', marker_opacity=0.7))
    fig.add_trace(go.Bar(x=x_range, y=removed, name='removed', marker_color='red', marker_opacity=0.7))
    fig_d = fig.to_dict()
    fig_d['layout'].update({
        'xaxis': {
            'title': 'Version',
            },
        'yaxis': {
            'title': col,
            'rangemode': 'tozero',
        },
        'barmode': 'relative',
    })
    fig_d['data'][0].update({
        'mode': 'lines+markers',
    })
    pp = PlotlyPlot(fig_d['data'], fig_d['layout'])
    return pp
