from dtk.s3_cache import S3File
from functools import reduce
import logging
logger = logging.getLogger(__name__)

# TODO:
# - finish assessment of clustering performance and improvements
#   - false positive biotech mismatches
#     - chembl Unknown molecule_type does not mean biotech; change the ETL
#       to either bypass these or to not set the biotech attribute
#   - enumerate specific biotech cases and algorithm changes
#     - ideally a list of situation descriptions, with examples, estimates
#       of occurrance frequency, and how to handle in code
#     - only 465 clusters mix biotech and non-biotech (and this may decrease
#       with the chembl Unknown change)
#       - 247 have only one SMILES; 91 have no SMILES; 65 have 2, 12 have 3
#       - so focusing on these cases will give the most benefit; others may
#         get solved as natural fallout of the small-case algorithms, or
#         they could be dropped without much consequence
#   - superclusters: there are about 60 examples of clusters with > 10
#     std_smiles subsets; one of the motivations for over-clustering and then
#     breaking is that dealing with a smaller set of drugs makes common
#     patterns easier to identify and resolve. Do we need special handling
#     for these large clusters? Are there few enough that they can be
#     manually fixed via the falsehoods file? Are they clues to problems
#     in the std_smiles algorithm?
#     - this might be informed by counting the number of pairwise connections
#       between smiles codes; high counts are more likely to be real matches
#       with erroneous smiles standardization; low counts are more likely to
#       be errors in other attributes; this is probably worth waiting on the
#       new rdkit pipeline
# - add biotech changes
# - write script to extract examples of cluster issues for manual research
#   (by scanning log files, for example, and maybe integrating other data
#   to create actionable summaries with links)
# - try to reduce the number of dropped orphans in cluster breaking (but,
#   all subsequent tuning should be justified by stats)
# - remove old unused code and clean up

def tsv_reader(iterable):
    for line in iterable:
        yield line.strip('\n').split('\t')

def pair_set_as_tsv(s):
    '''
    Return a canonical string format for a set of string-valued pairs.
    '''
    return '\t'.join(sum(sorted(list(s)),tuple()))

def pair_set_as_delim(s):
    return "|".join(["%s:%s" % part for part in sorted(list(s))])

def assemble_pairs(fields):
    '''
    Return a list of tuples, each containing 2 adjacent fields
    '''
    # A hopefully clearer implementation of the zip(*[iter(rec)]*2)
    # example from the zip() documentation.
    # It works by creating a single iterator, and then accessing
    # it twice in each cycle through zip().
    i = iter(fields)
    return list(zip(i,i))

def all_props_of_drugset(s):
    props = set()
    for d in s:
        props |= d.prop_set
    return props

def match_score(s1,s2):
    if not s1:
        return 0
    raw = len(s1 & s2)
    return (
        raw,
        float(raw)/len(s1 | s2),
        )

def suspect_score(drugs,props):
    # Scores are going to be 10 or above if a cluster contains multiple
    # std_smiles values. The other factors influence the score, but are
    # all acceptable duplicate values given that we're folding different
    # forms together.
    counts = []
    for keytype in 'std_smiles inchi cas pubchem_cid'.split():
        counts.append((keytype,len(
            set([p[1] for p in props if p[0] == keytype])
            )))
    # include molecular formula as a key
    counts.append(('molform',len(
            set([p[1].split('/')[1] for p in props if p[0] == 'inchi'])
            )))
    score = sum([
            10 if x[0] == 'std_smiles' else 1
            for x in counts
            if x[1] > 1
            ])
    return score

def strip_virtual_inchi_key_matches(inchi_keys):
    # input is [('inchi_key',<value>),...]
    # This eliminates all entries that end with the generic suffix,
    # but otherwise match another key in the list
    generic_suffix = '-UHFFFAOYSA-N'
    first_parts = set([
                x[1][:14]
                for x in inchi_keys
                if not x[1].endswith(generic_suffix)
                ])
    result = []
    for x in inchi_keys:
        if x[1].endswith(generic_suffix) and x[1][:14] in first_parts:
            continue
        result.append(x)
    return result

from dtk.cache import cached_dict_elements

def _clusters_cache_loader(cls, version, base_keys):
    # If we're doing a mega-bulk lookup, not worth caching.
    if len(base_keys) > 5000:
        return False
    return True, version, base_keys, 'base_keys'

from dtk.lazy_loader import LazyLoader
class RebuiltCluster(LazyLoader):
    '''Extract cluster for web display based on versioned ETL files.

    Previously, cluster display was handled by loading the entire
    base_drug_clusters file. Now, we extract data supporting a
    single cluster directly from the ETL source data, which is
    faster, and doesn't require storing another version of the
    (huge) base_drug_clusters file each time the Duma collection
    is updated.

    Note that, unlike the original, this loads all attribute data
    that links drugs within the cluster, without using the falsehoods
    filtering or discarding attributes during cluster breaking. But
    since only the drugs that finally make it into the cluster are
    considered, and since only attributes common to two or more
    clustered drugs are returned, the results shouldn't be misleading.
    '''
    _kwargs=['base_key','version']
    # XXX Note that, if we try to use clustering information from
    # XXX outside the attributes files (unichem, for example) this
    # XXX would need to be extended to make those properties available
    # XXX as well.
    # for now, kegg has been removed from the match list; it's very
    # noisy in bindingdb, so much of what it adds gets taken back
    # out by cluster breaking
    # XXX At some point we might like to try the linked_ attributes,
    # XXX but this will take some additional work as a dummy attribute
    # XXX will need to be added for each drug indicating that it is
    # XXX 'linked' to itself.
    direct_cols='inchi cas pubchem_cid inchi_key std_smiles'.split()
    name_cols='canonical synonym'.split()
    @classmethod
    def name_remapper(cls,attr,val):
        # discard 1 and 2 character names (they're usually fragments from
        # parsing errors, and not helpful when intentional)
        if len(val) < 3:
            return (None,None)
        # discard 'generic' labels that probably span multiple molecules
        for suffix in (
                'conjugate',
                'complex',
                'analog',
                'analogue',
                'compound',
                'derivative',
                ):
            if val.endswith(' '+suffix):
                return (None,None)
        # fold all columns into a single 'name' column, and force lower case
        return ('name',val.lower())
    @classmethod
    def get_cluster_file_path(cls, version):
        from dtk.s3_cache import S3File
        s3f=S3File('matching',f'matching.full.v{version}.clusters.tsv')
        s3f.fetch()
        return s3f.path()
    def _cluster_file_path_loader(self):
        return self.get_cluster_file_path(self.version)
    def _match_inputs_loader(self):
        from dtk.s3_cache import S3File
        s3f=S3File('matching',f'matching.full.v{self.version}.ingredients.tsv')
        s3f.fetch()
        # ingredients file is a single line holding the space-separated
        # paths to each clustering input attributes file
        paths = open(s3f.path()).read().strip().split()
        split_fns = [x.split('/')[-1].split('.') for x in paths]
        # return {collection_name:'flavor.version',...}
        return {
                parts[0]:'.'.join(parts[1:3])
                for parts in split_fns
                }
    def get_match_inputs_version(self, collection):
        flavor, version = self.match_inputs[collection].split('.')
        assert version[0] == 'v'
        return int(version[1:])
    def _assay_files_loader(self):
        from dtk.files import VersionedFileName
        from dtk.s3_cache import S3Bucket,S3File
        result = []
        for file_class,choice in self.match_inputs.items():
            ip_version = self.get_match_inputs_version(file_class)
            meta = VersionedFileName.meta_lookup[file_class]
            if 'affinity' not in meta.roles:
                continue
            s3b = S3Bucket(file_class)
            for fn in s3b.list(cache_ok=True):
                vfn = VersionedFileName(meta=meta,name=fn)
                if vfn.version != ip_version:
                    continue
                if vfn.role != 'affinity':
                    continue
                result.append((file_class,vfn.flavor,S3File(s3b,fn)))
        return result
    def _raw_assay_files_loader(self):
        from dtk.files import VersionedFileName
        from dtk.s3_cache import S3Bucket,S3File
        result = []
        for file_class,choice in self.match_inputs.items():
            ip_version = self.get_match_inputs_version(file_class)
            meta = VersionedFileName.meta_lookup[file_class]
            if 'raw_dpi_assays' not in meta.roles:
                continue
            s3b = S3Bucket(file_class)
            for fn in s3b.list(cache_ok=True):
                vfn = VersionedFileName(meta=meta,name=fn)
                if vfn.version != ip_version:
                    continue
                if vfn.role != 'raw_dpi_assays':
                    continue
                result.append((file_class,vfn.flavor,S3File(s3b,fn)))
        return result

    @classmethod
    @cached_dict_elements(_clusters_cache_loader, list_style=True)
    def load_many_clusters(cls, version, base_keys):
        '''Return a RebuiltCluster for each base_key.

        base_keys is a list like [(coll_key_prop,coll_key_value),...]

        Returns a list, in the same order as base_keys. Each cluster has
        drug_keys pre-loaded, so subsequent use won't cause another cluster
        file scan. However, note that the "select=(col_keys, None)" in the
        get_file_records call below is much less efficient that the (arguably
        less robust) "grep -w" used by _drug_keys_loader, so this is only
        a win with larger numbers of base_keys.
        '''
        logger.info(f"Loading {len(base_keys)} molecule clusters")
        from dtk.files import get_file_records
        col_keys = [x[1] for x in base_keys]
        mapping = {}
        for rec in get_file_records(
                cls.get_cluster_file_path(version),
                select=(col_keys, None),
                allow_suspicious=True,
                ):
            rec = assemble_pairs(rec)

            for pair in rec:
                assert pair not in mapping, f"{pair} shows up multiple times"
                mapping[pair] = rec

        logger.info("Clusters loaded")

        out = []
        for base_key in base_keys:
            cluster_keys = mapping.get(base_key, None)
            if cluster_keys is None:
                cluster_keys = [base_key]

            rc = RebuiltCluster(version=version, base_key=base_key)
            # Pre-populate the drug keys.
            rc.drug_keys = cluster_keys
            out.append(rc)
        return out



    def _drug_keys_loader(self):
        from dtk.files import get_file_records
        from dtk.drug_clusters import assemble_pairs
        l = []
        for rec in get_file_records(
                self.cluster_file_path,
                grep=['-w',self.base_key[1]],
                ):
            rec = assemble_pairs(rec)
            if self.base_key in rec:
                l.append(rec)
        if len(l) == 0:
            return [self.base_key]
        assert len(l) == 1
        return l[0]
    def _src2keys_loader(self):
        from dtk.data import MultiMap
        def strip_suffix(s,suffix):
            assert s.endswith(suffix)
            return s[:-len(suffix)]
        return {
                strip_suffix(src,'_id'):keys
                for src,keys in MultiMap(self.drug_keys).fwd_map().items()
                }
    def best_duma_key(self):
        from dtk.prot_map import MoleculeKeySet
        duma_keys = [
                key
                for coll,key in self.drug_keys
                if coll == MoleculeKeySet.priority_dpi_coll
                ]
        if not duma_keys:
            return None
        return MoleculeKeySet.priority_duma_id(duma_keys)
    def _drug_prop_pairs_loader(self):
        result = []
        from dtk.s3_cache import S3File
        from dtk.files import get_file_records
        keep_cols=self.name_cols+self.direct_cols
        for file_class,keys in self.src2keys.items():
            if file_class not in self.match_inputs:
                # This is from a collection that wasn't part of clustering (usually moa).
                # Ignore.
                continue
            choice = self.match_inputs[file_class]
            s3f=S3File(file_class,f'{file_class}.{choice}.attributes.tsv')
            s3f.fetch()
            for key,attr,val in get_file_records(
                    s3f.path(),
                    keep_header=False,
                    select=(list(keys),0),
                    ):
                if attr not in keep_cols:
                    continue
                if attr in self.name_cols:
                    attr,val = self.name_remapper(attr,val)
                    if attr is None:
                        continue
                result.append(((file_class+'_id',key),(attr,val)))
        return result
    def _filtered_drug_prop_pairs_loader(self):
        result = []
        from dtk.data import MultiMap
        mm=MultiMap(self.drug_prop_pairs)
        for prop,drugs in mm.rev_map().items():
            if len(drugs) > 1:
                for drug in drugs:
                    result.append((drug,prop))
        return result

class FactChecker:
    def __init__(self,fn):
        self.bad_props = set()
        self.bad_drugs = set()
        self.falsehoods = set()
        with open(fn) as f:
            from dtk.readtext import comment_stripper,parse_delim
            f = comment_stripper(f)
            f = parse_delim(f)
            wildcard = ('*','*')
            for rec in f:
                assert len(rec) == 4, 'invalid record: %s'%repr(rec)
                rec = tuple(rec)
                if rec[:2] == wildcard:
                    self.bad_props.add(rec[2:])
                elif rec[2:] == wildcard:
                    self.bad_drugs.add(rec[:2])
                else:
                    self.falsehoods.add(rec)
    def check_fact(self,rec):
        assert len(rec) == 4, 'invalid record: %s'%repr(rec)
        rec = tuple(rec)
        if rec[2:] in self.bad_props:
            return False
        if rec[:2] in self.bad_drugs:
            return False
        if rec in self.falsehoods:
            return False
        return True

class Node:
    def __init__(self,drug):
        self.drug = drug
        self.subnodes = {}
    def get_size(self):
        beneath = 0
        for s in self.subnodes.values():
            for n in s:
                if n:
                    beneath += n.get_size()
        return beneath+1
    def add_child(self,matches,subnode):
        link_key = frozenset(matches)
        self.subnodes.setdefault(link_key,set()).add(subnode)
    def get_branches(self):
        branches = list(self.subnodes.items())
        branches.sort(key=lambda x:len(x[0]),reverse=True)
        return branches
    def html(self,backlink=None):
        result = '<li>%s</li>' % repr(self.drug)
        result += '<ul>'
        for matches,subnodes in self.get_branches():
            if matches == backlink:
                assert subnodes == set([None])
                continue
            sub_result = ""
            for subnode in subnodes:
                if subnode:
                    sub_result += subnode.html(matches)
            highlight = 'style="background-color:yellow"'
            if len(matches) > 1 \
                    or 'inchi' in [x[0] for x in matches] \
                    or not sub_result:
                highlight = ''
            result += '<li %s>%s</li>' % (
                    highlight,
                    pair_set_as_delim(matches),
                    )
            if sub_result:
                result += '<ul>%s</ul>' % sub_result
        result += '</ul>'
        return result
    def pretty_print(self,indent=0,backlink=None):
        result = "    "*indent
        result += repr(self.drug)+'\n'
        for matches,subnodes in self.get_branches():
            if matches == backlink:
                assert subnodes == set([None])
                continue
            result += "    "*indent+"  "
            result += pair_set_as_delim(matches)
            result += "->\n"
            for subnode in subnodes:
                if subnode:
                    result += subnode.pretty_print(indent+1,matches)
        return result
    def __repr__(self):
        return "(%s,%s)" % (repr(self.drug),repr(self.subnodes))

class OverlapChecker:
    def __init__(self):
        self.union = set()
        self.overlap = set()
    def add(self,s):
        self.overlap |= (s & self.union)
        self.union |= s

class Drug:
    def __init__(self,clusterer,key):
        self.clusterer = clusterer
        # XXX Eventually we might want to capture several properties that
        # XXX are not used for clustering, but are checked later. For now,
        # XXX just add special-case code for the biotech flag.
        self.biotech = None
        self.key = key
        self.prop_set = set()
        self.links = {}
    def __repr__(self):
        return "Drug(%s,%s)" % self.key
    def add_prop(self,prop):
        self.prop_set.add(prop)
        self.clusterer.drugs_by_prop.setdefault(prop,set()).add(self)
    def del_prop(self,prop):
        if prop in self.prop_set:
            self.prop_set.remove(prop)
            s = self.clusterer.drugs_by_prop[prop]
            s.remove(self)
            if not s:
                del(self.clusterer.drugs_by_prop[prop])
    def unhook(self):
        for prop in list(self.prop_set):
            self.del_prop(prop)
        del(self.clusterer.drugs_by_key[self.key])
    def build_links(self):
        self.links = {}
        for prop in self.prop_set:
            for drug in self.clusterer.drugs_by_prop[prop]:
                if drug != self:
                    self.links.setdefault(drug,set()).add(prop)
    def _add_recursive(self,s):
        if self not in s:
            s.add(self)
            for d in self.links:
                d._add_recursive(s)
    def get_cluster_as_set(self):
        result = set()
        try:
            self._add_recursive(result)
        except RecursionError:
            # at this point, we have some kind of distributed clustering
            # failure -- no single property is causing problems, but
            # chains of properties exceed the length python can handle.
            #
            print('Pathological cluster starting from',self)
            print('size exceeds',len(result),'-- see match_detail')
            self.report_pathological_cluster()
            raise
        return result
    def report_pathological_cluster(self):
        # do a non-recursive version of _add_recursive, tracing to the
        # log file, and give up after a while
        seen = set()
        pending = set([self])
        props_reported = set()
        backtrace = {}
        def report_backtrace(start):
            bt_seen = set()
            result = []
            d = start
            while d in backtrace:
                if d in bt_seen:
                    self.clusterer.log('pathologic_backtrace_loop',start,*result)
                    break
                bt_seen.add(d)
                parent = backtrace[d]
                result = [parent.links[d],d] + result
                d = parent
            assert d == self
            if result:
                self.clusterer.log('pathologic_backtrace',d,*result)
        while pending and len(seen) < 10000:
            print('starting cycle; seen:',len(seen),'pending:',len(pending))
            for d in list(pending):
                pending.remove(d)
                if d not in seen:
                    seen.add(d)
                    report_backtrace(d)
                    for d2,props in d.links.items():
                        if d2 in seen:
                            continue
                        if d2 not in backtrace:
                            backtrace[d2]=d
                        pending.add(d2)
                        if True:
                            continue # XXX skip prop logging
                        for prop in props - props_reported:
                            # XXX We could look up all the drugs for this
                            # XXX prop and record how many new ones there
                            # XXX are (not yet in seen or pending).
                            # XXX Or maybe report both the count of
                            # XXX overlapping and non-overlapping,
                            # XXX which will highlight the ones pushing us
                            # XXX into new territory.
                            self.clusterer.log('pathologic_prop',prop,'from',d2)
                            props_reported.add(prop)
    @classmethod
    def _expand(cls,node,done):
        new = set()
        for drug,matches in node.drug.links.items():
            if drug in done:
                subnode = None
            else:
                subnode = Node(drug)
                new.add(subnode)
                done.add(drug)
            node.add_child(matches,subnode)
        return new
    def get_cluster_as_tree(self):
        done = set([self])
        root = Node(self)
        new = set([root])
        while new:
            next_new = set()
            for node in new:
                next_new.update(self._expand(node,done))
            new = next_new
        return root

class Stat:
    def __init__(self,key,values,label=None):
        self.key = key
        self.values = values
        self.label = label if label else key
        self.children = []
    def find(self,key):
        if self.key == key:
            return self
        for child in self.children:
            match = child.find(key)
            if match:
                return match
        return None
    def add_child(self,stat):
        self.children.append(stat)
    def as_string(self,indent=0):
        result = '   '*indent
        result += self.label+": "+", ".join([str(x) for x in self.values])+'\n'
        for child in self.children:
            result += child.as_string(indent+1)
        return result
    def __repr__(self):
        return self.as_string()

class Clusterer:
    # Theory of operation:
    # - a drug key is a tuple like ('chembl_id','CHEMBL0001')
    # - a property is a tuple like ('cas','12345-5-32')
    # - multiple drug attribute files are read in, creating Drug objects
    #   for each drug, and registering them in the drugs_by_key and
    #   drugs_by_prop dicts
    #   - drugs_by_key is one-to-one: {key:Drug,...}
    #   - drugs_by_prop is one-to-many: {prop:set([Drug,...]),...}
    #   - add_drug_prop_pair handles setting these linkages up correctly,
    #     along with a prop_set member in the Drug object
    # - once the drugs and properties are in place, properties are
    #   examined to find and discard uninformative property values
    #   (which match too many drugs, or only one drug)
    # - then build_links is invoked to build local association structures
    #   linking each drug to other drugs which share property values
    # - clusters are built by reading out the build_links data recursively
    # - clusters are examined for signs of over-clustering, and problematic
    #   property values (linking seemingly unrelated drugs) are removed
    # - then the build_links/build_clusters step is repeated with the
    #   revised properties to produce the final clusters
    def __init__(self):
        self.drugs_by_key = {}
        self.drugs_by_prop = {}
        self.clusters = []
        self.fact_checker = None
        self.archive_name = 'base_drug_clusters.tsv'
        self.archive_s3 = S3File('drugsets',self.archive_name)
        self.outdir = 'stage_drugsets'
        self.logdir = '.'
        self._logfile = None
        self.collection_names = set()
    def log(self,*args):
        if self._logfile is None:
            self._logfile = open(self.logdir+'/match_detail.log','w')
        self._logfile.write('\t'.join([str(x) for x in args])+'\n')
    def archive_write_path(self):
        # This file is written to the current directory, rather than
        # to outdir, because it's a make step dependency. The Makefile
        # moves it into position if the entire makestep succeeds.
        return self.archive_name
        #return self.outdir + '/' + self.archive_name
    def get_drug(self,key):
        if key not in self.drugs_by_key:
            drug = Drug(self,key)
            self.drugs_by_key[key] = drug
        return self.drugs_by_key[key]
    def get_cluster_html(self,key):
        drug = self.get_drug(key)
        root = drug.get_cluster_as_tree()
        drugs = drug.get_cluster_as_set()
        props = set()
        for d in drugs:
            props |= d.prop_set
        ordered_props = sorted(props)
        ordered_drugs = sorted(drugs, key=lambda x: x.key)
        from dtk.html import pad_table
        return '<br>'.join([
                "Cluster size: "+str(root.get_size()),
                root.html(),
                pad_table(['drug']+ordered_props,[[d]+[
                                'X' if p in d.prop_set else ''
                                for p in ordered_props
                                ]
                            for d in ordered_drugs
                            ]
                        ),
                pad_table(['prop']+ordered_drugs,[[p]+[
                                'X' if p in d.prop_set else ''
                                for d in ordered_drugs
                                ]
                            for p in ordered_props
                            ]
                        ),
                ])
    def del_prop(self,prop):
        if prop in self.drugs_by_prop:
            drugs = list(self.drugs_by_prop[prop])
            for drug in drugs:
                drug.del_prop(prop)
    def trim_props(self,min_size=2,max_size=10,keep_types=set()):
        delete_list = []
        for prop,drugs in self.drugs_by_prop.items():
            if (min_size <= len(drugs) <= max_size):
                continue
            discard = prop[0] not in keep_types
            if len(drugs) > max_size:
                self.log('trim_props',discard,prop,drugs)
            delete_list.append(prop)
        for prop in delete_list:
            self.del_prop(prop)
    def trim_props2(self,max_size_per_collection=2,keep_types=set()):
        # instead of seeing how many drugs a prop hits overall, base
        # the promiscuity test on the number it hits within a single
        # collection
        delete_list = []
        for k,v in self.drugs_by_prop.items():
            from collections import Counter
            ctr = Counter()
            for d in v:
                ctr[d.key[0]] += 1
            if max(ctr.values()) > max_size_per_collection:
                delete_list.append(k)
        for prop in delete_list:
            if prop[0] in keep_types:
                continue
            drugs = self.drugs_by_prop[prop]
            self.log('trim_props2',prop,drugs)
            self.del_prop(prop)
    def trim_disconnected_drugs(self):
        delete_list = [
            k for k,v in self.drugs_by_key.items()
            if not v.prop_set
            ]
        stat = Stat('total_disconnected_drugs',[len(delete_list)])
        from collections import Counter
        ctr = Counter()
        for key in delete_list:
            ctr[key[0]] += 1
        for k,v in ctr.most_common():
            stat.add_child(Stat(k,[v]))
        for key in delete_list:
            del(self.drugs_by_key[key])
        return stat
    def trim_single_drug_clusters(self):
        keep_clusters = []
        delete_drugs = []
        for c in self.clusters:
            if len(c) == 1:
                delete_drugs += list(c)
            else:
                keep_clusters.append(c)
        self.clusters = keep_clusters
    def build_clusters(self):
        self.clusters = []
        done = set()
        for drug in self.drugs_by_key.values():
            if drug not in done:
                s = drug.get_cluster_as_set()
                done.update(s)
                self.clusters.append(s)
                if False and len(s) > 100:
                    tree = drug.get_cluster_as_tree()
                    print('huge drug cluster')
                    print(tree.pretty_print())
        self.trim_single_drug_clusters()
    # Here's a good description of inchi and inchi_key formats:
    # http://vamdc-standards.readthedocs.io/en/latest/inchi/
    # Some key points:
    # - There is a formal procedure for 'standardizing' both inchi and
    #   inchi_key to strip irrelevant detail and facilitate matching.  Whether
    #   they are standardized or not is explicitly coded in both inchi and
    #   inchi_key
    # - The last character of inchi_key is NOT a checksum as wikipedia
    #   claims, but an indication of protonation
    def is_suspect(self,cluster):
        cluster_props = all_props_of_drugset(cluster)
        biotech_mismatch = len(set(
                d.biotech for d in cluster if d.biotech is not None
                )) > 1
        values = [x for x in cluster_props if x[0] == 'std_smiles']
        smiles_mismatch = len(values) > 1
        if biotech_mismatch or smiles_mismatch:
            desc = 'mixed' if biotech_mismatch else 'unmixed'
            self.log('rebuild_clusters',
                    f'{desc} biotech with',len(values),'smiles',
                    len(cluster),'total drugs',
                    [d.key[1] for d in cluster if d.biotech is False],
                    [d.key[1] for d in cluster if d.biotech is True],
                    )
        if smiles_mismatch:
            return True
        return False
    def subset_by_attr(self,cluster,attr):
        subsets = {}
        orphans = set()
        for d in cluster:
            values = [x for x in d.prop_set if x[0] == attr]
            if len(values) > 1:
                self.log('rebuild_clusters',d,'has ambiguous values',values)
            elif len(values) == 1:
                s = subsets.setdefault(values[0],set())
                s.add(d)
            else:
                orphans.add(d)
        return subsets,orphans
    def rebuild_clusters(self):
        # - partition each cluster (by std_smiles or inchi_key)
        # - remove properties that span multiple partitions within a cluster
        # - recalculate linkages with those properties gone
        break_attempts = 0
        discarded_orphans = 0
        for c in self.clusters:
            if not self.is_suspect(c):
                continue
            # partition the drugs in the cluster
            # - subsets is a dict of sets of drugs; the key identifying
            #   each partition is arbitrary
            # - orphans is a set of drugs
            subsets,orphans = self.subset_by_attr(c,'std_smiles')
            sorted_subsets = sorted(subsets.items())
            subset_biotech = []
            for k,v in sorted(subsets.items()):
                kinds = set(d.biotech for d in v if d.biotech is not None)
                if len(kinds) > 1:
                    # XXX There are only 40 of these using v19 inputs.
                    # XXX Review again after chembl ETL changes.
                    self.log('mixed biotech subset',
                            [d.key[1] for d in v if d.biotech is False],
                            [d.key[1] for d in v if d.biotech is True],
                            )
                if len(kinds) == 1:
                    subset_biotech.append(next(iter(kinds)))
                else:
                    subset_biotech.append(None)
            for ((k,v),bt) in zip(sorted_subsets,subset_biotech):
                self.log('subset dump',
                        f'bt:{bt}',
                        k[1],
                        len(v),'drugs:',
                        sorted(d.key[1] for d in v),
                        )
            self.remove_conflicting_subset_properties(c,subsets)
            sorted_propsets = [
                    all_props_of_drugset(v)
                    for k,v in sorted_subsets
                    ]
            # Deal with any problematic orphans. Note that when the
            # clusters are reconstructed at the bottom of this file:
            # - any orphans with an affinity for only one subgroup will
            #   be assigned to that group
            # - orphans without an affinity for any group (which can happen
            #   if remove_conflicting_subset_properties deleted all their
            #   links into the cluster) will not be clustered.
            # So the problematic case is multiple non-zero affinity values.
            # XXX These are removed by delete_ambiguous_orphans() in the
            # XXX old code.
            #
            # XXX Incorporating biotech makes this complicated:
            # XXX - the basic idea is, we don't want to add a biotech
            # XXX   molecule to a non-biotech subset, or vice versa
            # XXX - since only std_smiles molecules make subsets, there
            # XXX   may not be a subset for a biotech molecule; if we
            # XXX   have biotech orphans and no biotech subset, we want to
            # XXX   make a new subset by picking an orphan; we need to break
            # XXX   any connection between that orphan and any existing
            # XXX   subset, while preserving connections to other biotech
            # XXX   orphans
            # XXX - to prevent an orphan from joining an incompatible
            # XXX   subset, we need to remove matching properties from
            # XXX   either the orphan or the subset
            # XXX - a biotech mismatch could either
            # XXX   - prevent an unambiguous property match from operating
            # XXX   - disambiguate an ambiguous match
            # XXX Based on the logs, good test cases are:
            # XXX - DB00052 / CHEMBL1201621 - these are two biotech orphans
            # XXX   which should be a new subgroup; DB00052 incorrectly links
            # XXX   to a SM subgroup via the name da-3002; CHEMBL1201621 has
            # XXX   that link and another incorrect link tev-tropin to a
            # XXX   different subgroup. They both have 7 links to each other.
            # XXX - DB14357 / CHEMBL2108918 - similar; new subgroup
            # XXX - CHEMBL3544994 - Oligonucleotide with 'free acid' synonym;
            # XXX   should probably just be dropped, but currently isn't
            # XXX - DB14691 - incorrectly linked via 'salix' to DB00695
            # XXX - CHEMBL3544989 - incorrectly linked to DB04785 via 'salt'
            # XXX - DB14212 - incorrectly linked to DB13616 via 'mel'
            # XXX - CHEMBL2108391 - in same cluster as above; doesn't link to
            # XXX   any std_smiles subset, but should link to DB14212
            # XXX - DB14298 - incorrectly linked to DB12039 via 'catechu'
            # XXX - DB01381 / CHEMBL2108466 - need new subgroup
            # XXX - DB03404 vs DB03934 - DB03404 is biotech w/o std_smiles,
            # XXX   but links to same std_smiles as DB03934 via CAY16487
            # XXX   and PUBCHEM455658??
            # XXX
            # XXX separately, we might want to disambiguate matches which
            # XXX are ambiguous as a result of only a single property
            # XXX
            # XXX In ChEMBL, 'Unknown' molecules should have biotech=None.
            # XXX see CHEMBL4297727 (or we shouldn't have extracted at all).
            # XXX 
            biotech_orphans = [d for d in orphans if d.biotech is True]
            if biotech_orphans:
                if all(x is False for x in subset_biotech):
                    # We need a new subset to hold biotech orphans.
                    for d in biotech_orphans:
                        self.log('biotech subset candidate',
                                d.key[1],
                                d.prop_set & all_props_of_drugset(
                                        x for x in biotech_orphans if x != d
                                        ),
                                )
                    # XXX select one here and convert to subset
                    # XXX Possible criteria:
                    # XXX - least prop overlap with existing subsets
                    # XXX - most prop overlap with other biotech orphans
            for d in orphans:
                sorted_overlaps = [d.prop_set & s for s in sorted_propsets]
                vec = [len(s) for s in sorted_overlaps]
                sorted_biotech_conflicts = [
                        (bt is True and d.biotech is False)
                        or
                        (bt is False and d.biotech is True)
                        for ov,bt in zip(vec,subset_biotech)
                        if ov
                        ]
                # XXX This is still not quite organized correctly. It ends
                # XXX up double-logging some orphans, and doesn't easily
                # XXX capture how prop matching and biotech matching interact.
                if sum(vec) == 0 or sum(bool(x) for x in vec) > 1:
                    # vecs with more than one non-zero value are the ones
                    # that would be removed by delete_ambiguous_orphans();
                    # vecs with all zero values will be silently dropped
                    # from the cluster (they only matched properties that
                    # were deleted by remove_conflicting_subset_properties()
                    self.log('problematic orphan affinity',
                            d.key[1],
                            vec,
                            [y for x,y in zip(vec,sorted_overlaps) if x],
                            )
                if any(sorted_biotech_conflicts):
                    self.log('problematic orphan biotech',
                            d.key[1],
                            vec,
                            sorted_biotech_conflicts,
                            [y for x,y in zip(vec,sorted_overlaps) if x],
                            )
            discarded_orphans += self.delete_ambiguous_orphans(subsets,orphans)
            break_attempts += 1
        self.log('rebuild_clusters','cluster breaks complete')
        # now recluster with the problematic properties removed
        print(break_attempts,'cluster breaks attempted,',
                discarded_orphans,'orphans discarded,',
                'rebuilding')
        # get rid of any drugs that don't match anything after cluster breaking
        self.trim_disconnected_drugs()
        self.build_links()
        self.build_clusters()
    def remove_conflicting_subset_properties(self,c,subsets):
        # see what properties overlap
        oc = OverlapChecker()
        for v in subsets.values():
            oc.add(all_props_of_drugset(v))
        # One hope here was that we'd be able to clearly determine which
        # subset an attribute values goes with by the number of occurances
        # in that subset.  But that number is often small (1-2), so it's
        # not that definitive.
        # The information might be useful for identifying bad attributes,
        # and SMILES standardization problems, so log it.
        for prop in sorted(oc.overlap):
            subset_matches = {}
            for k,v in subsets.items():
                matches = [x for x in v if prop in x.prop_set]
                if matches:
                    # XXX it might be worth listing the drug keys here instead
                    # XXX of just showing the length; this would let us see
                    # XXX the underlying pages more easily
                    subset_matches[k] = len(matches)
            self.log('rebuild_clusters','subset prop overlap'
                    ,prop
                    ,sorted(subset_matches.items())
                    )
        # now delete all overlapping properties
        if False:
            # from all the drugs in the subsets only
            for v in subsets.values():
                for d in v:
                    for p in oc.overlap:
                        d.del_prop(p)
        else:
            # from all drugs in the cluster
            for d in c:
                for p in oc.overlap:
                    d.del_prop(p)
    def delete_ambiguous_orphans(self,subsets,orphans):
        # Now, look for any cases where orphans overlap with multiple
        # subsets.  As an example, bindingdb_id,BDBM50344897 has the
        # names 'sodium borate' and 'sodium tetraborate', but these
        # associate with different std_smiles codes.
        # XXX We currently repair these by deleting the drug. Sometimes,
        # XXX the log shows the drug sharing multiple properties
        # XXX with one subset, but only a single one with the other. In
        # XXX that case, we could remove that single property instead.
        discarded_orphans = 0
        for d in orphans:
            subset_matches = {}
            for k,v in subsets.items():
                match = d.prop_set & all_props_of_drugset(v)
                if match:
                    subset_matches[k] = match
            if len(subset_matches) > 1:
                self.log('rebuild_clusters','orphan drug overlap'
                        ,d
                        ,subset_matches
                        )
                d.unhook()
                discarded_orphans += 1
        return discarded_orphans
    def collection_stats(self):
        stat = Stat('total_keys',[len(self.drugs_by_key)])
        from collections import Counter
        ctr = Counter()
        for key in self.drugs_by_key:
            ctr[key[0]] += 1
        for k,v in ctr.most_common():
            stat.add_child(Stat(k,[v]))
        return stat
    def link_stats(self):
        stat = Stat('total_props',[len(self.drugs_by_prop)])
        from collections import Counter
        ctr = Counter()
        for s in self.drugs_by_prop.values():
            ctr[len(s)] += 1
        l = list(ctr.items())
        l.sort(key=lambda x:x[0]) # output in 'histogram' order
        for k,v in l:
            stat.add_child(Stat('matching_%d_drugs'%k,[v]))
        return stat
    def cluster_stats(self):
        stat = Stat('total_clusters',[len(self.clusters)])
        stat.add_child(Stat('total_drugs',[sum(len(s) for s in self.clusters)]))
        from collections import Counter
        ctr = Counter()
        for s in self.clusters:
            ctr[len(s)] += 1
        l = list(ctr.items())
        l.sort(key=lambda x:x[0]) # output in 'histogram' order
        for k,v in l:
            stat.add_child(Stat('containing_%d_drugs'%k,[v]))
        ctr = Counter()
        for s in self.clusters:
            ctr2 = Counter()
            for drug in s:
                ctr2[drug.key[0]] += 1
            ctr[max(ctr2.values())] += 1
        l = list(ctr.items())
        l.sort(key=lambda x:x[0]) # output in 'histogram' order
        for k,v in l:
            stat.add_child(Stat(
                        'containing_at_most_%d_drugs_per_collection'%k,
                        [v],
                        ))
        ctr = Counter()
        for s in self.clusters:
            std_smiles_set=set()
            for drug in s:
                for prop in drug.prop_set:
                    if prop[0] == 'std_smiles':
                        std_smiles_set.add(prop)
            ctr[len(std_smiles_set)] += 1
        l = list(ctr.items())
        l.sort(key=lambda x:x[0]) # output in 'histogram' order
        for k,v in l:
            stat.add_child(Stat(
                        'containing_%d_std_smiles'%k,
                        [v],
                        ))
        ctr = Counter()
        for s in self.clusters:
            inchi_set=set()
            for drug in s:
                for prop in drug.prop_set:
                    if prop[0] == 'inchi':
                        inchi_set.add(prop)
            #if len(inchi_set) >= 5:
            #    print
            #    print s
            ctr[len(inchi_set)] += 1
        l = list(ctr.items())
        l.sort(key=lambda x:x[0]) # output in 'histogram' order
        for k,v in l:
            stat.add_child(Stat(
                        'containing_%d_inchis'%k,
                        [v],
                        ))
        ctr = Counter()
        for s in self.clusters:
            collection_set=set()
            for drug in s:
                collection_set.add(drug.key[0])
            ctr[len(collection_set)] += 1
        l = list(ctr.items())
        l.sort(key=lambda x:x[0]) # output in 'histogram' order
        for k,v in l:
            stat.add_child(Stat(
                        'containing_%d_collections'%k,
                        [v],
                        ))
        ctr = Counter()
        for s in self.clusters:
            props = all_props_of_drugset(s)
            score = suspect_score(s,props)
            ctr[score] += 1
        l = list(ctr.items())
        l.sort(key=lambda x:x[0]) # output in 'histogram' order
        for k,v in l:
            stat.add_child(Stat(
                        'suspect_score_%d'%k,
                        [v],
                        ))

        return stat
    def stat_overview(self):
        total = len(self.clusters)
        stat = Stat('total_clusters',[len(self.clusters)])
        stat.add_child(Stat('average_size',[
                    float(len(self.drugs_by_key))/len(self.clusters),
                    ]))
        cstats = self.cluster_stats()
        def stats_vec(template,low,high):
            vec = []
            for i in range(low,high+1):
                s = cstats.find(template%i)
                vec.append(s.values[0] if s else 0)
            vec.append(total - sum(vec))
            return vec
        stat.add_child(Stat(
                'max_drugs_per_collection',
                stats_vec('containing_at_most_%d_drugs_per_collection',1,2),
                ))
        stat.add_child(Stat(
                'inchis_per_cluster',
                stats_vec('containing_%d_inchis',0,2),
                ))
        num_collections=len(self.collection_names)
        stat.add_child(Stat(
                'collections_per_cluster',
                stats_vec('containing_%d_collections',1,num_collections),
                ))
        return stat
    def add_drug_prop_pair(self,prop,key):
        if self.fact_checker:
            if not self.fact_checker.check_fact(key+prop):
                print('skipping false association',key,prop)
                return
        # This forces protonation to be ignored
        if prop[0] == 'inchi_key':
            prop = (prop[0],prop[1][:-1]+'N')
        self.get_drug(key).add_prop(prop)
    def save_archive(self):
        with open(self.archive_write_path(),'w') as f:
            for prop,drugs in sorted(self.drugs_by_prop.items()):
                fields = list(prop)
                keys = [d.key for d in drugs]
                keys.sort()
                for key in keys:
                    fields += list(key)
                f.write(('\t'.join(fields)+'\n'))
    def write_cluster_dump(self):
        f = open(self.logdir+'/cluster_dump.out','w')
        rows = []
        for s in self.clusters:
            rows.append(pair_set_as_tsv( [d.key for d in s] ))
        rows.sort()
        for row in rows:
            f.write(row+'\n')
        f.close()
        return
    def load_fact_filter(self,fn):
        self.fact_checker = FactChecker(fn)
    def load_archive(self):
        self.archive_s3.fetch()
        with open(self.archive_s3.path()) as f:
            for rec in tsv_reader(f):
                pairs = assemble_pairs(rec)
                for drug_key in pairs[1:]:
                    self.add_drug_prop_pair(pairs[0],drug_key)
    def load_from_file(self,filename,prop_list,map_function=None):
        # extract collection id from filename
        path = filename.split('/')
        if path[-1].startswith('create.'):
            # old-style name
            from dtk.files import AttrFileName
            af = AttrFileName()
            af.set_name(path[-1])
            collection = af.collection
            master = af.is_master_create()
            key_name = af.key_name()
        else:
            # versioned file name
            from dtk.files import VersionedFileName
            file_class = path[-1].split('.')[0]
            vfn = VersionedFileName(file_class=file_class, name=path[-1])
            collection = f'{file_class}.{vfn.flavor}'
            master = vfn.flavor == 'full'
            key_name = file_class + '_id'
        self.collection_names.add(collection)
        keyset = set()
        header = None
        assert master,'clustering not supported on subsets'
        # scan file and record all properties
        with open(filename,'r') as f:
            rdr = tsv_reader(f)
            for rec in rdr:
                if not header:
                    header = rec
                    continue
                keyset.add(rec[0])
                if rec[1] == 'biotech':
                    key = (key_name, rec[0])
                    self.get_drug(key).biotech = (rec[2] == 'True')
                if rec[1] in prop_list:
                    if map_function:
                        # optionally remap attr and val
                        rec[1],rec[2] = map_function(rec[1],rec[2])
                        if rec[1] is None:
                            continue
                    self.add_drug_prop_pair(
                        (rec[1], rec[2]),
                        (key_name, rec[0]),
                        )
    def build_links(self):
        for drug in self.drugs_by_key.values():
            drug.build_links()

