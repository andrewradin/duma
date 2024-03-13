#!/usr/bin/env python3
import pwd
import os
import csv
import subprocess
from functools import reduce

'''Path management

NOTE: this file should not include import any django modules or any parts
of the web app, so it can be used by web1/settings.py

This file centralizes all information about the path layout of the various
components of the DUMA platform.  Within the web1 directory, the normal
Django conventions can be assumed, but any relationships between external
packages, or paths which differ between development and production, are
managed here.

In particular:
- various external components (like the R array processing, or machine
  learning) should be referenced through top-level variables defined
  here, rather than coding in assumptions about their relative locations
- output locations of files for display on the web should be determined
  using the various 'publish' members, and the URLs corresponding to those
  locations should be determined through url_of_file().
- this file can be invoked from the command line with a PathHelper
  member name to output the full path of that directory, so the data
  specified here is accessible to processes outside python
- this file can be invoked with the --dev-links or --prod-links options
  to create all necessary directories and symlinks
'''

def current_user_name():
    return pwd.getpwuid(os.getuid())[0]

# The AutoCompress stuff is defined here so that it can be used in
# the vbucket_spec setup below without requiring an import.
import enum
class AutoCompressMode(enum.Enum):
    # Can't use enum.auto here yet due to a bootstrapping issue - the base ubuntu
    # system only has python 3.5, which doesn't have that feature.
    # There are a couple of places in install.sh where this gets invoked with
    # the system python.
    NO_COMPRESS=object()
    COMPRESS_ONLY=object()
    COMP_AND_DECOMP=object()

class PathHelper(object):
    # external paths of interest

    # the install tree
    home = '/home/'+current_user_name()+'/'
    install_root = home + '2xar/'
    is_apache = (current_user_name() == 'www-data')
    # NOTE: is_apache will be true in the web code running under apache, and
    # in scripts explicitly executed as the www-data user (eg. dtk.lts under
    # the lts_status script relies on this).
    # It is NOT true when invoked from install.sh, which runs as ubuntu. That
    # script decides to use apache configuration from a machine_type of
    # 'platform' in local_settings.py. It is also not true in authkeys.py
    # (which runs on another machine entirely). That relies on a --apache
    # command line flag.
    if is_apache:
        code_root = '/home/ubuntu/2xar/'
    else:
        code_root = install_root
    venv = install_root + 'opt/conda/'
    R_base = install_root + "opt/conda/envs/r-env/"
    R_libs = R_base + "lib/R/library/"
    R_bin = R_base + "bin/"

    repos_root = install_root + 'twoxar-demo/'
    website_root = repos_root + 'web1/'

    weka = code_root+'weka-3-6-11/weka.jar'
    databases = code_root + 'twoxar-demo/databases/'
    uniprot = databases + 'uniprot/'
    # the root location in the filesystem to publish static web content;
    # this is a symlink to the actual location, which may
    # vary based on web installation
    publish = install_root+'publish/'
    MLpublish = publish+'ML/'
    history =  publish+'history/'
    authkeys_keydb = repos_root+'keydb/'
    @classmethod
    def ws_ml_publish(cls,wsid):
        return cls.MLpublish+str(wsid)+'/'
    @classmethod
    def ws_publish(cls,wsid):
        return cls.publish+str(wsid)+'/'
    @classmethod
    def pathsum_history(cls,wsid):
        return cls.history+str(wsid)+'/'
    @classmethod
    def ml_history(cls,wsid):
        return cls.MLpublish+str(wsid)+'/history/'
    @classmethod
    def dashboard_publish(cls):
        return cls.publish+'dashboard/'
    @classmethod
    def tmp_publish(cls,wsid,client):
        # This builds a directory name where individual web pages can store
        # .plotly and .png files. 'client' is a name identifying the web
        # page/view. Filenames within the directory should follow a
        # view-specific convention to encode any parameters that affect
        # the plot (so that only a plot generating matching parameters
        # will be rendered on the page).
        #
        # validate_plotdir() in dtk.plot will help build this directory,
        # and also provides a way to manage situations where the exact
        # data underlying a view parameter may change (for example, a
        # filename may encode the name of a drugset, but if the content
        # of the drugset changes, the plot needs to be regenerated).
        return cls.publish+'tmp/'+str(wsid)+'/'+client+'/'

    # root location for stored scores
    lts = install_root+'lts/'

    # this function returns the URL corresponding to an absolute
    # filesystem path
    @classmethod
    def url_of_file(cls,path):
        if path.startswith(cls.publish):
            return '/publish/' + path[len(cls.publish):]
        if path.startswith(cls.lts):
            return '/publish/lts/' + path[len(cls.lts):]
        raise AttributeError("'%s' is not a published path" % (path,))
    # this function returns the URL corresponding to a page wrapping
    # a plotly file
    @classmethod
    def url_of_plot(cls,wsid,path,headline=None):
        if path.startswith(cls.install_root):
            relpath = path[len(cls.install_root):]
            if relpath.startswith('lts/'):
                relpath = 'publish/'+relpath
            url = ('/cv/%d/plot/?path='%wsid) + relpath
            if headline:
                url += "&headline="+headline
            return url
        raise AttributeError("'%s' is not a valid path" % (path,))
    @classmethod
    def path_of_pubfile(cls,path):
        parts = path.split('/')
        if parts[0] != 'publish' or '..' in path:
            raise AttributeError("'%s' is not a valid path" % (path,))
        if parts[1] == 'lts' and parts[-2] == 'publish':
            # Under LTS, static web-viewable files (and plotly files)
            # are indicated by being in a subdirectory named 'publish'.
            # Examples are:
            # publish/lts/ws_id/job_type/job_id/publish/filename
            # publish/lts/log#/####/job_id/publish/filename
            #
            # To reach the file, replace the initial 'publish/lts'
            # with the LTS repo base.
            return (cls.lts,'/'.join(parts[2:]))
        # old-style publish directory files
        return (cls.publish,'/'.join(parts[1:]))

    # root location for data not visible on the website
    storage = install_root+'ws/'
    pidfiles = storage+'pidfiles/'
    timestamps = storage+'timestamps/'
    progress = storage+'progress/'
    downloads = storage+'downloads/'
    django_cache = storage+'django_cache/'
    goterm = storage+'goterm/'
    uphd = storage+'uphd/'
    d2ps_cache = storage+'d2ps_cache/'
    padre_pub = publish+'padre/'

    # This environment variable can be used to change the S3
    # cache directory, which is used during testing.  Otherwise,
    # it defaults to the same as storage.
    s3_cache_root = os.environ.get("S3_CACHE_DIR", storage)
    legacy_s3_file_classes = [
            'customdpi',
            'drugsets',
            'dpi',
            'd2ps',
            'ppi',
            'tox',
            'padre',
            'sigprot',
            'glee',
            'keys',
            ]
    # patch in legacy file class dirs, and build list
    s3_cache_dirs = []
    for file_class in legacy_s3_file_classes:
        path = s3_cache_root+file_class+'/'
        vars()[file_class] = path
        s3_cache_dirs.append(path)

    vbucket_specs = (
            # register formats of standard versioned files here:
            # fileClass, # of flavor name parts, list of roles.
            ('test',1,[
                    'both',
                    ('comp_only',AutoCompressMode.COMPRESS_ONLY),
                    ('neither',AutoCompressMode.NO_COMPRESS),
                    ]),
            ('uniprot',1,[
                    'Uniprot_data',
                    'Protein_Ensembl',
                    'Protein_EnsemblTRS',
                    'Protein_Entrez',
                    ('Protein_Names',AutoCompressMode.NO_COMPRESS),
                    ]),
            ('openTargets',0,[
                    ('data',AutoCompressMode.NO_COMPRESS),
                    'names',
                    'target_safety',
                    'tractability',
                    ]),
            ('drugnames',0,[]),
            ('aact',0,[
                    'disease_drug_matrix',
                    'sponsors',
                    'study_contacts',
                    'drugs',
                    'diseases',
                    'studies',
                    ]),
            ('duma_gwas',0,[
                    ('data',AutoCompressMode.NO_COMPRESS),
                    ('archivedata',AutoCompressMode.NO_COMPRESS),
                    'studies',
                    ('prev_rscd',AutoCompressMode.NO_COMPRESS)
                    ]),
            ('duma_gwas_v2d',0,[
                    ('data',AutoCompressMode.NO_COMPRESS),
                    ('archive',AutoCompressMode.NO_COMPRESS),
                    'studies',
                    ]),
            ('duma_gwas_v2g',0,[
                    ('data',AutoCompressMode.NO_COMPRESS),
                    ('archive',AutoCompressMode.NO_COMPRESS),
                    'd2g_summary',
                    'otarg_alleles',
                    ]),
            ('grasp',0,[
                    ('data',AutoCompressMode.NO_COMPRESS),
                    'studies',
                    'failed_snps',
                    ]),
            ('gwas_cat',0,[
                    ('data',AutoCompressMode.NO_COMPRESS),
                    'studies',
                    'failed_snps',
                    ]),
            ('phewas',0,[
                    ('data',AutoCompressMode.NO_COMPRESS),
                    'studies',
                    ]),
            ('ukbb',0,[
                    ('data_filtered',AutoCompressMode.NO_COMPRESS),
                    'studies',
                    ]),
            ('finngen',0,[
                    ('data_filtered',AutoCompressMode.NO_COMPRESS),
                    'studies',
                    ]),
            ('meddra',0,[]),
            ('mondo',0,[
                    'mappings',
                    'labels',
                    ]),
            ('umls',0,[
                    'to_meddra',
                    'to_ICD9_or_ICD10',
                    ]),
            ('agr',0,[
                    'human',
                    'models',
                    ]),
            ('monarch',0,[
                    'evidence',
                    'disease',
                    'gene',
                    ]),
            ('homologene',1,[
                    'entrez',
                    'ensembltrs',
                    ]),
            ('disgenet',0,[
                    'curated_meddra',
                    'curated_umls',
                    'disease_names',
                    'cui_disease_names',
                    ]),
            ('name2cas',0,[]),
            ('targetscan',0,[
                    'context_scores_human9606_ensembl'
                    ]),
            ('efo',0,[
                    'obo',
                    'terms',
                    'otarg_hier',
                    ]),
            ('salmon',1,[
                    ('kmer23',AutoCompressMode.NO_COMPRESS),
                    ('kmer31',AutoCompressMode.NO_COMPRESS),
                    ]),
            ('ucsc_hg',0,[
                    'chrom_sizes',
                    'prot_txn_sizes'
                    ]),
            ('faers',0,[
                    'drug_mat',
                    'drug_cols',
                    'indi_mat',
                    'indi_cols',
                    'demo_mat',
                    'date_mat',
                    'indi_drug_mat',
                    'dose_mat',
                    'indi_drug_dose_meta',
                    ]),
            ('chembl',1,[
                    'attributes',
                    'evidence',
                    'affinity',
                    'adme_assays',
                    'pc_assays',
                    'tox_assays',
                    'indications',
                    'raw_dpi_assays',
                    ]),
            ('drugbank',1,[
                    'attributes',
                    'evidence',
                    ]),
            ('bindingdb',1,[
                    'affinity',
                    'attributes',
                    'evidence',
                    ]),
            ('cayman',1,[
                    'attributes',
                    ]),
            ('selleckchem',1,[
                    'attributes',
                    ]),
            ('med_chem_express',1,[
                    'attributes',
                    ]),
            ('moa',1,[
                    'attributes',
                    ]),
            ('ncats',1,[
                    'attributes',
                    'evidence',
                    ]),
            ('duma',1,[
                    'attributes',
                    'evidence',
                    ]),
            ('pubchem',1,[
                    'attributes',
                    ]),
            ('lincs',1,[
                    'attributes',
                    'evidence',
                    'metadata',
                    ('expression', AutoCompressMode.NO_COMPRESS),
                    ]),
            ('globaldata',1,[
                    'attributes',
                    'evidence',
                    'parse_detail',
                    ]),
            ('matching',1,['dpimerge', 'clusters','props','ingredients']),
            ('string',1,[]),
            ('unichem',0,[
                    ('chembl_to_zinc',AutoCompressMode.NO_COMPRESS),
                    ('chembl_to_surechembl',AutoCompressMode.NO_COMPRESS),
                    ('drugbank_to_zinc',AutoCompressMode.NO_COMPRESS),
                    ('drugbank_to_surechembl',AutoCompressMode.NO_COMPRESS),
                    ('zinc_to_bindingdb',AutoCompressMode.NO_COMPRESS),
                    ]),
            ('gene_ontology',0,[
                    'hierarchy',
                    'genesets',
                    ]),
            ('reactome',0,[
                    'hierarchy',
                    'genesets',
                    ('diagrams', AutoCompressMode.NO_COMPRESS),
                    ('graphdb', AutoCompressMode.NO_COMPRESS),
                    ]),
            ('similarity', 1, [
                'fpdb',
                'struct_metadata',
            ]),
            ('pathways',2,[
                'genesets',
                'hierarchy',
                'gene_to_pathway',
                ]),
            ('zinc',0,[
                    ('biogenic+in-cells',AutoCompressMode.NO_COMPRESS),
                    ('endogenous',AutoCompressMode.NO_COMPRESS),
                    ('fda',AutoCompressMode.NO_COMPRESS),
                    ('for-sale+in-cells',AutoCompressMode.NO_COMPRESS),
                    ('in-man',AutoCompressMode.NO_COMPRESS),
                    ('in-trials',AutoCompressMode.NO_COMPRESS),
                    ('not-for-sale+in-vitro',AutoCompressMode.NO_COMPRESS),
                    ('for-sale+in-vitro',AutoCompressMode.NO_COMPRESS),
                    ('in-vivo',AutoCompressMode.NO_COMPRESS),
                    ('not-for-sale+in-cells',AutoCompressMode.NO_COMPRESS),
                    ('world',AutoCompressMode.NO_COMPRESS),
                    ]),
            ('uniprot_dpi',0, []),
            ('orthologs',0, []),
            ('mesh',0, ['diseases']),
            ('orange_book',0, [
                    'products',
                    'patent',
                    'use_codes',
                    ]),
        )

    # add vbuckets to s3_cache_dirs; this means they get created and
    # cleared the same way as legacy s3 cache directories
    for x in vbucket_specs:
        # Must match VBucket._cache_path_loader
        s3_cache_dirs.append(s3_cache_root+x[0]+'/')

    @classmethod
    def s3_cache_path(cls,file_class):
        if file_class in cls.legacy_s3_file_classes:
            return getattr(cls,file_class)
        raise AttributeError('Not a cache bucket')

    @classmethod
    def clean_stale_ws_files(cls):
        import glob
        for pattern in (
                'zinc_labels.*',
                '*_to_*.tsv.gz',
                'targetscan_*',
                'opentargets_*',
                'msigdb.*',
                'hg38.*',
                'efo_obo.pickle',
                'duma_gwas.*',
                'dpi.*.bg.tsv',
                'disGeNet.curated.*',
                'clin_ev.*',
                'uniprotToGOTerms.tsv',
                'GO.terms_alt_ids',
                '*Terms_allEvidence*',
                'aact_*',
                'c50.bindingdb.c50.tsv',
                'ki.bindingdb.ki.tsv',
                'drugNameMap.tsv',
                'SRAmetadb.sqlite',
                ):
            for path in glob.glob(cls.storage+pattern):
                print('removing',path)
                os.remove(path)

    s3_misc_file_patterns=(
            'dpi.*.bg.tsv',
            '__list_cache',
            )

    @classmethod
    def clear_s3_caches(cls,interactive=False):
        listcache_only=True
        # The above prevents deleting old cache files. At present, versioned
        # files are immutable, and non-versioned files aren't being updated
        # (they're being converted to versioning instead). So, there's no
        # reason to delete and force refetching other that reducing disk
        # usage. The access time option described below might be a better
        # way to achieve that. Alternatively, there are only a few directories
        # where the file download time is significant, so we could set this
        # to True in the loop below for those directories only.
        for d in cls.s3_cache_dirs:
            if interactive:
                rsp = input('clear %s? (y/N):'%d)
                if rsp.lower() != 'y':
                    continue
            for f in os.listdir(d):
                if listcache_only and f != '__list_cache':
                    # TODO: Rather than listcache_only, we can add an access
                    # time heuristic and clear out files that haven't been used
                    # in a long time
                    # (Currently disks are mounted relatime, so the timestamps
                    # aren't perfectly accurate, but still good to the day)
                    continue
                p = os.path.join(d,f)
                print('removing',p)
                os.remove(p)
        import glob
        import operator
        misc_files=reduce(operator.add,[
                glob.glob(cls.storage+pat)
                for pat in cls.s3_misc_file_patterns
                ])
        for f in misc_files:
            if interactive:
                rsp = input('clear %s? (y/N):'%f)
                if rsp.lower() != 'y':
                    continue
            print('removing',f)
            os.remove(f)

    # directories to create
    create_dirs = [
            pidfiles,
            timestamps,
            progress,
            downloads,
            django_cache,
            padre_pub,
            goterm,
            publish+'goterm/',
            uphd,
            d2ps_cache,
            ] + s3_cache_dirs

    __config=None
    local_settings_file=website_root+'local_settings.py'
    @classmethod
    def get_config(cls):
        '''Return all configuration which may vary between sites.'''
        if cls.__config is None:
            # The actual construction of the config dict is done in a
            # separate method to facilitate unit testing.
            cls.__config=cls._get_config(cls.local_settings_file)
        return cls.__config
    @classmethod
    def cfg(cls,attr):
        '''Return value of a single config parameter.'''
        return cls.get_config()[attr]
    @classmethod
    def _get_config(cls,path):
        # The result dict is pre-loaded with defaults, and then
        # site-specific overrides are applied
        result = {
                'worker_machine_name':'worker-test',
                'bigtmp':'/tmp', # override to another drive if needed
                'disk_space_limits': [], # override to check low disk space
                        # overridden list should contain (mountpoint,limit)
                        # pairs, where mountpoint is a string ("/", "/mnt2")
                        # and limit is a minimum size in MB, as an integer.
                'upgrade_on_install':True,
                'skip_R_packages':[],
                'can_publish_drugsets': False,
                # Can set to true on dev machines for easier dev, just be
                # sure to coordinate with others.
                'rsync_worker': False,
                'clinical_event_datasets':{
                        'default':['faers.v'],
                        # special per-workspace overrides, by workspace id
                        # (disabled to eliminate non-versioned cases)
                        #123:['XXX_dry_AMD','FAERS','CVAROD'],
                        #115:['FAERS+MDF','FAERS','CVAROD'],
                        },
                'monitor_AWS': False, # true on one machine running cronjob
                'https_hostname': '', # for apache certs and ServerName
                'LTS_role_keys': True, # use temporary tokens for S3 access
                'drug_edit_cluster_version':None, # dev workaround for the
                        # DrugProposals table falling behind the DpiMergeKey
                        # table (if a later merge file is uploaded)
                'lts_scan_slack_report': True, # to disable slack while testing
                }
        # execfile isn't in py3.
        # It has been ported, but we don't want to use external libraries here
        # because this function is called in install.sh at the very start,
        # so it has to be usable from vanilla python2.
        exec(compile(open(path).read(), path, 'exec'),globals(),result)
        if not 'version' in result:
            vers=subprocess.check_output(
                    ['git','describe','--tags'],
                    cwd=cls.repos_root,
                    ).decode('utf8')
            result['version'] = vers.strip()
        return result
    @classmethod
    def is_production(cls):
        return cls.cfg('https_hostname') == 'platform.twoxar.com'
    @classmethod
    def meta_progress(cls,geo_id):
        return cls.progress+"meta_"+geo_id
    @classmethod
    def sig_progress(cls,geo_id,tissue_id):
        return cls.progress+"sig_"+geo_id+str(tissue_id)
    @classmethod
    def path_progress(cls,ws_id):
        return cls.progress+"path_"+str(ws_id)

    # locations of external programs
    MLscripts = website_root+'ML/'
    Rscripts = website_root+'R/'
    deaScripts = Rscripts+'DEA/'
    fingerprint = repos_root+'moleculeSimilarity/fingerprint/'
    faers = website_root+'clinical/faers/'
    # and checked_in lookup tables
    exit_codes = website_root+'algorithms/exit_codes.tsv'

def build_one_symlink(source,link_name):
    if os.path.exists(link_name):
        os.remove(link_name)
    os.symlink(source,link_name)

def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_missing_files(root,filelist):
    missing = []
    for path in filelist:
        if not os.path.exists(root+path):
            missing.append(path)
    return missing

def check_data_directory(directory,filelist):
    data_missing = check_missing_files(directory,filelist)
    if data_missing:
        raise RuntimeError("\n\n\n"
                    +"DATA MISSING; into "+directory+", retrieve:\n"
            + "\n".join(data_missing)
            )

def build_dev_symlinks():
    # ws and publish have lots of special rules:
    # - on a worker machine, both should be symlinks to large drives
    #   (or, they can be local directories if root is a large drive)
    # - on dev machines, publish needs to symlink to web1/browse/static
    # - on platform, it also should do that, but it's the copy in www-data
    #   that really matters
    # this code creates them by assuming that, if they should be
    # externalized, the external directories will already exist
    ext_ws = '/mnt2/ubuntu/ws'
    ext_pub = '/mnt2/ubuntu/publish'
    if os.path.exists(ext_ws):
        build_one_symlink(ext_ws,PathHelper.storage[:-1])
    # else, a ws directory gets created implicitly below
    if os.path.exists(ext_pub):
        build_one_symlink(ext_pub,PathHelper.publish[:-1])
    else:
        make_directory(PathHelper.publish[:-1])
    for d in PathHelper.create_dirs:
        make_directory(d)
    check_local_settings()

def check_local_settings():
    path = PathHelper.local_settings_file
    if not os.path.exists(path):
        raise RuntimeError("\n\n\n"
                +"LOCAL SETTINGS FILE MISSING -- please create:\n"
                +path+"\n"
                )
def check_symlink(source,link_name):
    if source[-1] == '/':
        source = source[:-1]
    if link_name[-1] == '/':
        link_name = link_name[:-1]
    if os.path.exists(link_name):
        if os.path.islink(link_name):
            if os.path.realpath(link_name) == os.path.realpath(source):
                return # ok
    raise RuntimeError("\n\n\n"
                +"SYMLINK ERROR:\n"
                +"'%s' should be a symlink pointing to '%s'"
                % (link_name,source)
                )

def build_prod_symlinks():
    assert current_user_name() == 'www-data'
    check_symlink("/var/www/html/publish/",PathHelper.publish)
    # these are common with build_dev_symlinks
    for d in PathHelper.create_dirs:
        make_directory(d)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='path utility')
    parser.add_argument('--dev-links',action='store_true'
        ,help='create dev symlinks for the current user'
        )
    parser.add_argument('--prod-links',action='store_true'
        ,help='set up apache-specifc links and directories (run sudo www-data)'
        )
    parser.add_argument('--clear-s3-caches',action='store_true'
        ,help='remove all files in S3 cache directories, forcing re-fetch'
        )
    parser.add_argument('--clear-s3-caches-interactive',action='store_true'
        ,help='interactively clear S3 cache directories'
        )
    parser.add_argument('--clean-stale',action='store_true'
        ,help='remove old files from 2xar/ws'
        )
    parser.add_argument('--exit',metavar='code_or_symbol')
    parser.add_argument('dir',nargs='?',default=None)
    args = parser.parse_args()

    if args.clean_stale:
        PathHelper.clean_stale_ws_files()
    if (args.dev_links):
        build_dev_symlinks()
    if (args.prod_links):
        build_prod_symlinks()
    if (args.clear_s3_caches_interactive):
        PathHelper.clear_s3_caches(interactive=True)
    elif (args.clear_s3_caches):
        PathHelper.clear_s3_caches()
    if (args.exit):
        from algorithms.exit_codes import ExitCoder
        ec = ExitCoder()
        try:
            code = int(args.exit)
            print(ec.message_of_code(code))
        except ValueError:
            print(ec.encode(args.exit))
    if (args.dir):
        if args.dir == 'is_production':
            import sys
            sys.exit(0 if PathHelper.is_production() else 1)
        try:
            print(getattr(PathHelper,args.dir))
        except AttributeError:
            print(PathHelper.cfg(args.dir))
