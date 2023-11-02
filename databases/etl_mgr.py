#!/usr/bin/env python3

# XXX - remove 'publish' steps from all drug collections? the idea is
# XXX   they should only publish via 'matching' directory

import pwd,os
user=pwd.getpwuid(os.getuid())[0]
root='/home/%s/2xar/' % user

import sys
sys.path.insert(1,root+'twoxar-demo/web1/')

others={
        'license only':[
                'geo',
                ],
        'obsolete':[
                'cmap',
                'dpi',
                'lincs',
                'aeolus',
                'unichem',
                'humanProteinAtlas',
                ],
        }
for_future_reference={
        'clinical trials':[
                'aact',
                ],
        'disease':[
                'uniprot_human_diseases',
                'orange_book',
                ],
        'vocabularies':[
                'meddra',
                'uniprot',
                'umls',
                'atc',
                ],
        'gene expression':[
                'tcga_ge',
                'aesearch_training',
                ],
        'miRNA':[
                'targetScan',
                ],
        'adr':[
                'sider',
                'offsides',
                'gottlieb',
                ],
        }

class Refresh(object):
    grp='_'
    index = {}
    @classmethod
    def get(cls,name):
        try:
            return cls.index[name]
        except KeyError:
            pass
        return Refresh(name,notes=['details missing'])
    @classmethod
    def names(cls):
        return sorted(cls.index.keys())
    def __init__(self,name,**kwargs):
        self.name = name
        self.index[name] = self
        self.prereqs = kwargs.pop('prereqs',[])
            # datasets used when refreshing the current dataset; additional
            # steps may be necessary to bring these up to date
        self.notes = kwargs.pop('notes',[])
        self.steps = kwargs.pop('steps',[])
            # the operations to be performed to refresh the current dataset
        self.covers = kwargs.pop('covers',[])
            # datasets that are brought up to date as a byproduct of
            # updating the current dataset
        assert not kwargs
        if name.startswith(self.grp):
            # A refresh operation can only have steps if it has a
            # databases subdirectory where the steps can be executed.
            # Refresh objects that exist only for grouping should
            # only have prereqs.
            assert not self.steps
    def _show_children(self,indent,shown):
        for name in self.prereqs:
            if name in shown:
                print('  '*indent,'('+name+')')
            else:
                print('  '*indent,name)
                shown.add(name)
                r = self.get(name)
                r._show_children(indent+1,shown)
    def list_prereqs(self):
        shown = set([self.name])
        print(self.name)
        self._show_children(1,shown)
    def show_steps(self,shown=set()):
        for name in self.prereqs:
            if name not in shown:
                shown.add(name)
                r = self.get(name)
                r.show_steps(shown)
        if self.steps or self.notes:
            if not self.name.startswith(self.grp):
                print(divider())
                print(line('#','in databases/%s:'%self.name))
            for note in self.notes:
                print(line('###','NOTE:',note))
            if self.steps:
                print('\n'.join(self.steps))
    @classmethod
    def list_missing(cls):
        known = set(cls.index)
        for l in others.values():
            known |= set(l)
        for r in cls.index.values():
            known |= set(r.covers)
        unknown = []
        from dtk.files import is_dir, scan_dir
        for name in scan_dir(
                root+'twoxar-demo/databases',
                filters=[is_dir],
                output=lambda x:x.filename,
                ):
            if name not in known:
                unknown.append(name)
        if unknown:
            print('Unconfigured directories:')
            for d in unknown:
                print('  ',d)

def line(*args): return ' '.join(str(x) for x in args)
def divider(): return line('#'*20)
def cmd(*args): return line(*args)
def make(*args): return line('make',*args)
def manual(*args): return line('#',*args)

Refresh('drugbank',
        notes=[
                'create file needs manual upload on to platform',
                ],
        steps=[
                manual('remove download file to check for updates'),
                make('input'),
                make('build'),
                ],
        )

Refresh('ttd',
        # maybe remove this for data quality reasons, if chembl seems
        # like a better source of experimental combounds
        notes=[
                'currently disabled; latest version has new keys',
                'create file needs manual upload on to platform',
                ],
        )

Refresh('bindingdb',
        # they claim weekly updates; our version is 2 years old;
        # maybe just remove instead of updating
        steps=[
                manual('manually update URL in Makefile'),
                make('input'),
                make('build'),
                ],
        )

Refresh('chembl',
        notes=[
                '2 create files need manual upload on to platform',
                ],
        steps=[
                manual('manually load into MySQL and update Makefile for latest version'),
                make('input'),
                make('build'),
                ],
        )

Refresh('stitch',
        prereqs=[
                '_uniprot',
                ],
        notes=[
                'no updates published since 2016',
                'run on existing version to update uniprot mapping',
                ],
        steps=[
                manual('manually update Makefile for latest version'),
                make('input'),
                make('build'),
                ],
        )

Refresh('adrecs',
        notes=[
                'updated download now requires (free) account',
                'create file needs manual upload on to platform',
                ],
        )

Refresh('broad',
        prereqs=[
                '_uniprot',
                ],
        notes=[
                'no updates published since March 2017',
                'create file needs manual upload on to platform',
                'run on existing version to update uniprot mapping',
                ],
        steps=[
                manual('manually update Makefile for latest version'),
                make('input'),
                make('build'),
                ],
        )

Refresh('ncats',
        prereqs=[
                '_uniprot',
                ],
        notes=[
                'updates are years apart',
                'create file needs manual upload on to platform',
                'run on existing version to update uniprot mapping',
                ],
        steps=[
                manual('manually update Makefile for latest version'),
                make('input'),
                make('build'),
                ],
        )

Refresh('gene_sets',
        notes=[
                'update now requires login',
                'run on existing version to update uniprot mapping',
                ],
        steps=[
                make('input'),
                make('build'),
                make('publish_s3'),
                ],
        )

Refresh('uniprot',
        notes=[
                'updates happen about monthly',
                ],
        steps=[
                manual('remove download file (HUMAN_9606_idmapping.dat.gz) to check for updates'),
                make('input'),
                make('build'),
                make('publish_s3'),
                ],
        )

Refresh('_uniprot',
        # you need to re-load gene sets after re-loading protein sets
        # on the platform, so you might as well update both if you're
        # updating uniprots
        notes=[
                'proteins and pathways need manual upload onto platform',
                ],
        prereqs=[
                'uniprot',
                'gene_sets',
                ],
        )

Refresh('matching',
        notes=[
                'm. files need manual upload on to platform',
                ],
        prereqs=[
                'drugbank',
                #'ttd',
                'bindingdb',
                'chembl',
                'stitch',
                'adrecs',
                'broad',
                'ncats',
                ],
        covers=[
                'dpi_bg_scores',
                'd2ps',
                ],
        steps=[
                make('input'),
                make('build'),
                make('compare_ws_stats'),
                manual('verify changes are as expected'),
                make('publish_s3'),
                ],
        )
Refresh('duma_gwas',
        prereqs=[
                'grasp',
                'gwasCat',
                ],
        steps=[
                make('build'),
                manual('verify changes are as expected'),
                make('publish_s3'),
                ],
        )

Refresh('grasp',
        # they've been promising another version for at least a year
        prereqs=[
                '_uniprot',
                ],
        steps=[
                manual('manually update the version in Makefile'),
                manual('manually determine if a genome build update is needed, and if so update in the Makefile'),
                make('input'),
                make('build'),
                ],
        )
Refresh('gwasCat',
        prereqs=[
                '_uniprot',
                ],
        steps=[
                make('input'),
                make('build'),
                ],
        )

Refresh('bindingdb',
        # they claim weekly updates; our version is 2 years old;
        # maybe just remove instead of updating
        steps=[
                manual('manually update URL in Makefile'),
                make('input'),
                make('build'),
                ],
        )


Refresh('_ppi',
        prereqs=[
                'drpias',
                'string',
                'cpdb',
                ],
        )

Refresh('cpdb',
        notes=[
                'latest version is pulled automatically',
                ],
        steps=[
                make('input'),
                make('build'),
                make('publish_s3'),
                ],
        )

Refresh('string',
        notes=[
                'run on existing version to update uniprot mapping',
                ],
        steps=[
                manual('manually update Makefile for latest version'),
                make('input'),
                make('build'),
                make('publish_s3'),
                ],
        )

Refresh('drpias',
        prereqs=[
                '_uniprot',
                ],
        notes=[
                'no longer maintained, but rebuilds pull in uniprot updates',
                ],
        steps=[
                make('build'),
                make('publish_s3'),
                ],
        )

Refresh('_clin_ev',
        prereqs=[
                'faers',
                'cvarod',
                ],
        )

for name in ('faers','cvarod'):
    Refresh(name,
            prereqs=[
                    'drugbank',
                    'chembl',
                    #'ttd',
                    'meddra',
                    ],
            steps=[
                    make('input'),
                    make('build'),
                    make('publish'),
                    make('publish_s3'),
                    ],
            )

Refresh('_disease',
        prereqs=[
                'disgenet',
                'openTargets',
                ],
        )

Refresh('disgenet',
        prereqs=[
                '_uniprot',
                'meddra',
                ],
        steps=[
                make('input'),
                make('build'),
                make('publish'),
                make('publish_s3'),
                ],
        )

Refresh('openTargets',
        prereqs=[
                '_uniprot',
                ],
        notes=[
                'run on existing version to update uniprot mapping',
                ],
        steps=[
                make('build'),
                make('publish_s3'),
                ],
        )

Refresh('_ge',
        prereqs=[
                'arrayexpress_efo',
                ],
        )

Refresh('arrayexpress_efo',
        steps=[
                make('input'),
                make('build'),
                make('publish'),
                make('publish_s3'),
                ],
        )

Refresh('_other',
        prereqs=[
                'ucsc_hg',
                ],
        )

Refresh('ucsc_hg',
        steps=[
                manual('As needed, update the genome build version in path_helper.py <hg_version>'),
                make('input'),
                make('build'),
                make('publish_s3'),
                ],
        )

Refresh('_all',
        prereqs=[
                'matching',
                'duma_gwas',
                '_clin_ev',
                '_ppi',
                '_disease',
                '_ge',
                '_other',
                ],
        )

# XXX PLAT-2850:
# XXX Mods so far indicate there's not much of value left in this script.
# XXX
# XXX review/correct underlying ETL inconsistencies exposed so far:
# XXX   - what is dependency relationship between pathways and reactome?
# XXX   - devise an interface that lets an ETL directory expose non-standard
# XXX     dependencies (or, always encode these in the standard way, in
# XXX     addition to any non-standard encoding required by the Makefile)
# XXX   - disgenet includes a flavor as part of its uniprot dependency; should
# XXX     that be considered part of the standard?
# XXX   - non-standard source version attribute names: duma_gwas, faers,
# XXX     ucsc_hg
# XXX     attribute -- is that reasonable?
# XXX   - name2cas depends on non-versioned files; fix (PLAT-3226)
# XXX   - figure out a scheme for managing retired or dependent directories
# XXX     (gene_sets, grasp, gwasCat, d2ps)
# XXX   - review tiering (fix missing, adjust as needed, approximate future
# XXX     conversion dates)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='''\
Query a database of instructions for running ETL.

The database is organized as a tree of nodes, where each node corresponds
to a single ETL subdirectory, or a group of functionally related subdirectories.
By default, the command lists all the nodes in a tree structure.
''',
            )
    parser.add_argument('--missing',
            help='Instead of the normal output, list all the ETL subdirectories present in the filesystem but not yet represented in the database',
            action='store_true',
            )
    parser.add_argument('--start',
            default='_all',
            help='Start at the specified node instead of the root of the tree',
            metavar='NODE',
            choices=Refresh.names(),
            # XXX Note that in --versioned mode, the choices list may not
            # XXX align with actual active node names. If this script becomes
            # XXX important, maybe eliminate checking. 
            )
    parser.add_argument('--versioned',
            help='Use versions.py prereqs, not hard-coded',
            action='store_true',
            )
    parser.add_argument('--flat',
            help='Show in a list, not a tree',
            action='store_true',
            )
    parser.add_argument('--collapse',
            help='In --flat mode, eliminate artifical groupings',
            action='store_true',
            )
    parser.add_argument('--steps',
            help='Show refresh instructions instead of just node names',
            action='store_true',
            )
    parser.add_argument('--skip',
            help='skip instructions for the specified nodes (and their children)',
            nargs='*',
            metavar='NODE',
            choices=Refresh.names(),
            )
    args = parser.parse_args()
    
    if args.versioned:
        Refresh.index = {}
        from dtk.etl import get_etl_data,get_etl_name_lookup
        name_lookup = get_etl_name_lookup()
        for src in name_lookup.values():
            info = get_etl_data(src,name_lookup)
            if info.latest_version:
                Refresh(src,prereqs=[x[0] for x in info.dependencies])
    if args.missing:
        Refresh.list_missing()
    elif args.flat:
        groups = {}
        for name in Refresh.names():
            r = Refresh.get(name)
            if not r.prereqs:
                continue
            if args.collapse:
                if name.startswith('_'):
                    groups[name] = r.prereqs
                    continue
                # since names are sorted, by the time we get here, the
                # groups dict is completely built
                to_do = [r.prereqs]
                closure = set()
                while to_do:
                    for name2 in to_do[0]:
                        if name2.startswith('_'):
                            to_do.append(groups[name2])
                        else:
                            closure.add(name2)
                    to_do = to_do[1:]
            else:
                closure = r.prereqs
            print(r.name+': '+', '.join(sorted(closure)))
    else:
        r = Refresh.get(args.start)
        if args.steps:
            r.show_steps(set(args.skip or []))
        else:
            r.list_prereqs()
