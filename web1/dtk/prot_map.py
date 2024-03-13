from dtk.lazy_loader import LazyLoader
import six

import logging
logger = logging.getLogger(__name__)

class PpiMapping:
    '''Report characteristics of available PPI mappings'''
    file_classes = ['string', ]
    # NOTE: You probably don't want to use this.
    # Consider browse.default_settings.PpiDataset.value instead
    preferred = 'string.default.v1'
    default_evidence = 0.9
    def __init__(self,choice):
        if choice == 'string.default':
            # Legacy ppi choice, e.g. we got our PPI choice from the parms
            # of an old job.  Default to v1.
            choice = 'string.default.v1'

        self.choice = choice
        self.file_class = choice.split('.')[0]
        self.format = 'sqlsv'

        from dtk.tsv_alt import SqliteSv
        self.tsv = SqliteSv(self.get_path())

    def get_path(self):
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned(
                self.file_class,
                '.'.join(self.choice.split('.')[1:]),
                format=self.format,
                )
        s3f.fetch()
        return s3f.path()

    def get_data_records(self, min_evid = 0.0):
        if min_evid == 0.0:
            return self.tsv.get_records()
        else:
            return self.tsv.get_records(filter__evidence__gte=min_evid)

    def get_header(self):
        return self.tsv.get_header()

    def get_ppi_info_for_keys(self,uniprot_ids, min_evid = 0.0):
        '''Return a list of protein interactions for a protein.

        Find all interactions for a given Uniprot ID. Each
        interaction is represented by a namedtuple, which is guaranteed to
        have at least 2 uniprot_ids, evidence, and
        direction fields, and may have additional fields as well.
        '''
        PpiInteraction = self.tsv.get_namedtuple('PpiInteraction')
        rows = self.tsv.get_records(filter__prot1__in=uniprot_ids,
                                    filter__evidence__gte=min_evid)
        return [PpiInteraction(*row) for row in rows]
    def get_filtered_uniq_target(self, min_evid = default_evidence):
        to_ret = set()
        for frs in self.tsv.get_records(columns=['prot2'],
                                        filter__evidence__gte=min_evid):
            to_ret.add(frs['prot2'])
        return to_ret
    def get_uniq_target(self):
        from dtk.files import get_file_columns
        return set(x['prot2'] for x in self.tsv.get_records(columns=['prot2']))
    def get_uniq_target_cnt(self):
        return len(self.get_uniq_target())
    @classmethod
    def choices(cls):
        '''Return labels for all available PPI files formatted for a dropdown.
        '''
        choices = []
        from browse.models import Workspace
        for file_class in cls.file_classes:
            fc_choices = [(f'{file_class}.{x[0]}', f'{file_class}.{x[1]}')
                          for x in Workspace.get_versioned_file_choices(file_class)]
            choices.extend(fc_choices)
        choices.append(['string.default', 'string.default (legacy)'])
        return choices

class MoleculeKeySet:
    def __init__(self,rec):
        from dtk.drug_clusters import assemble_pairs
        from dtk.data import MultiMap
        self.coll2keys = MultiMap(assemble_pairs(rec))
    def collections(self):
        return self.coll2keys.fwd_map().keys()
    def keys(self,coll):
        return self.coll2keys.fwd_map().get(coll,[])
    coll_order = {
            x+'_id':i
            for i,x in enumerate((
                    'duma',
                    'drugbank',
                    'ncats',
                    'chembl',
                    'bindingdb',
                    ))
            }
    def best_key(self):
        # choose highest priority collection
        colls = list(self.collections())
        colls.sort(key=lambda x:self.coll_order.get(x,100))
        best_coll = colls[0]
        # return lexically lowest key within that collection
        return (best_coll,sorted(self.keys(best_coll))[0])
    # If a keyset contains duma drugs, they override consensus DPI,
    # and the highest-numbered duma drug should be the one used. This
    # basic definition lives here so it's available to dpi_merge,
    # and to the drug_edit logic.
    priority_dpi_coll = 'duma_id'
    @classmethod
    def priority_duma_id(cls, candidates):
        return sorted(candidates)[-1]

class DpiMapping:
    '''Report characteristics of available DPI mappings.

    Each mapping is represented by a file.  Each mapping has a short
    description suitable for use in a menu.  Each mapping has a pair
    of key columns that are used to associate the mapping with a
    Drug.
    '''
    # NOTE: You probably don't want to use this.
    # Consider browse.default_settings.DpiDataset.value instead
    preferred = 'dpimerge.DNChBX_ki'
    default_evidence = 0.5
    custom_dpi_prefix = 'cstm_'

    @classmethod
    def label_from_filename(cls, filename):
        import re
        m = re.match(r'dpi\.(.*)\.tsv$',filename)
        if m:
            # legacy filename
            return m.group(1)
        m = re.match(r'matching\.(.*)\.dpimerge\.tsv$',filename)
        if m:
            return m.group(1)
        m = re.match(r'(uniprot_dpi\.(.*))\.tsv$',filename)
        if m:
            return m.group(1)

        m = re.match(r'(cstm_.*)\.tsv$',filename)
        if m:
            return m.group(1)

        raise RuntimeError('unrecognized filename format')

    @classmethod
    def dpi_names(cls):
        '''Return labels for all available DPI files.

        For legacy files in the format dpi.<key>.<detail>.tsv,
        the label is the <key>.<detail> portion of the filename.
        For versioned files like matching.<detail>.<version>.dpimerge.tsv
        the label is <detail>.<version>.
        '''
        # the 'cache_ok' options lets us avoid a 1-2 sec delay hitting
        # AWS every time this gets called (most annoyingly, every time
        # a drug page is rendered); to force the code to pick up
        # a new dpi file, remove the "__list_cache" file from ws/dpi
        items = []
        from dtk.s3_cache import S3Bucket
        for bucket_name in ('matching','dpi','uniprot_dpi'):
            bucket = S3Bucket(bucket_name)
            for label in bucket.list(dezip=bucket_name=='dpi',cache_ok=True):
                try:
                    label = cls.label_from_filename(label)
                except RuntimeError:
                    continue # ignore other files in directory
                items.append(label)
        return items
    @classmethod
    def _get_key_of_name(cls,name):
        '''Return the key name for a DPI label.'''
        (key,detail) = name.split('.')
        return key
    @classmethod
    def choices(cls, ws=None):
        '''Return all DPI labels formatted for a dropdown.'''

        def parse(dpi_name):
            import re
            m = re.match(cls._versioned_typed_choice_RE, dpi_name)
            if m:
                dpi_type = m.group(1)
                version = int(m.group(2))
            else:
                dpi_type = None
                version = None

            return {'version': version, 'dpi_type': dpi_type}
        
        uid_to_name = {}
        uid_to_deprecated = {}
        if ws is None:
            choices = cls.dpi_names()
        else:
            from wsadmin.custom_dpi import custom_dpi_entries
            custom_entries = custom_dpi_entries(ws)
            uid_to_name.update({uid:name for uid, name, deprecated in custom_entries})
            uid_to_deprecated.update({uid:deprecated for uid, name, deprecated in custom_entries})
            choices = cls.dpi_names() + [uid for uid, name, deprecated in custom_entries]

        parsed = [(parse(x), x) for x in choices]

        # We pull out max version from the database instead of choices list.
        # Don't want people to select non-uploaded dpimerge files.
        from drugs.models import DpiMergeKey
        max_version = DpiMergeKey.max_version() or 0

        parsed = [x for x in parsed if x[0]['version'] is None or x[0]['version'] <= int(max_version)]

        max_uniprot_version = max([x[0]['version'] for x in parsed if x[1].startswith('uniprot_dpi')], default=0) # Uniprot is special


        group_order = [
            'WS-Custom',
            'Latest',
            'MoA',
            'Uniprot',
            'Combo',
            'Versioned',
            'Legacy',
            'Deprecated',
        ]

        def groups(data):
            parsed_info, choice = data
            out = []
            dpi_type, version = parsed_info['dpi_type'], parsed_info['version']

            if dpi_type:
                if dpi_type == '-moa':
                    out.append("MoA")
                elif dpi_type.startswith('+'):
                    out.append("Combo")
            
            if version == max_version or choice.startswith('uniprot_dpi') and version == max_uniprot_version:
                if uid_to_deprecated.get(choice, False):
                    out.append('Deprecated')
                elif choice in uid_to_deprecated:
                    out.append('WS-Custom')
                elif 'Combo' not in out:
                    # We don't include combos in latest.
                    out.append('Latest')
            

            if not version:
                out.append('Legacy')
            
            if choice.startswith('uniprot'):
                out.append('Uniprot')

            if version and not out:
                out.append("Versioned")

            assert out, f"No group for dpi {choice}, it won't appear"
            
            return out
        
        pairs = []
        for data in parsed:
            g = groups(data)
            for group_name in g:
                pairs.append((group_name, (data[1], data[0]['version'], data[0]['dpi_type'])))

        dpi_type_order = [
            None,
            '-moa'
            # Combos are implicitly last
        ]
        
        def format_entries(entries):
            def sort_key(x):
                choice, version, dpi_type = x
                version = version or -1
                dpi_type = dpi_type_order.index(dpi_type) if dpi_type in dpi_type_order else len(dpi_type_order)
                return (-version, dpi_type, choice)

            entries = sorted(entries, key=sort_key )
            return [(x[0], uid_to_name.get(x[0], x[0])) for x in entries]
        
        from dtk.data import MultiMap
        out = list((k,format_entries(v)) for k, v in MultiMap(pairs).fwd_map().items())
        out.sort(key=lambda x: group_order.index(x[0]))
        return out
    @classmethod
    def get_possible_mappings(cls,prop_names):
        '''Return DPI mappings applicable to a drug or drugset.

        prop_names is a set of the properties defined for the drug or
        drugset. A list of DPI labels that can be matched with any of
        those properties is returned.
        '''
        prop_names = set(prop_names)
        import re
        ok = set()
        for item in cls.dpi_names():
            if not re.match(cls._versioned_choice_RE,item):
                # legacy choices need to match prop names
                key = cls._get_key_of_name(item)
                key += '_id'
                if not prop_names & set([key,'m_'+key]):
                    continue
            ok.add(item)
        return list(ok)
    _versioned_choice_RE = r'[^.]+\.v([0-9]+)$'
    _versioned_typed_choice_RE = r'[^.]+?([+-][^.]+)?\.v([0-9]+)$'
    def __init__(self,choice):
        self.choice = choice
        import re
        m = re.match(self._versioned_choice_RE,choice)
        self.legacy = not m
        if m:
            self.version = int(m.group(1))
        else:
            self.version = None
        self.custom_dpi = choice.startswith(self.custom_dpi_prefix)
    def mapping_type(self):
        if not self.choice.startswith('uniprot'):
            return 'wsa'
        else:
            return 'uniprot'
    def get_path(self):
        from dtk.s3_cache import S3Bucket
        if self.legacy:
            bucket = S3Bucket('dpi')
            fn = f'dpi.{self.choice}.tsv'
        elif self.custom_dpi:
            from wsadmin.custom_dpi import custom_dpi_path
            return custom_dpi_path(self.choice)
        elif self.mapping_type() == 'uniprot':
            bucket = S3Bucket('uniprot_dpi')
            fn = f'{self.choice}.tsv'
        else:
            bucket = S3Bucket('matching')
            fn = f'matching.{self.choice}.dpimerge.tsv'
        from dtk.s3_cache import S3File
        f = S3File(bucket,fn)
        f.fetch(unzip=self.legacy) # make sure file is present
        return f.path()
    def get_db_path(self):
        if self.legacy or self.custom_dpi:
            return None
        if hasattr(self, 'db_path'):
            return self.db_path
        else:
            from dtk.s3_cache import S3Bucket
            bucket = S3Bucket('matching')
            fn = f'matching.{self.choice}.dpimerge.sqlsv'
            from dtk.s3_cache import S3File
            try:
                f = S3File(bucket,fn)
                f.fetch() # make sure file is present
                path = f.path()
            except OSError as e:
                # Haven't generated this file for everything, it is not required for now.
                path = None
            self.db_path = path
            return path
    # get_keyspace has been removed, since it only made sense for legacy
    # files (and calling code probably made invalid assumptions about
    # dpi key storage). The only remaining calls are in combo code, where
    # there are outdated dependencies on how base combo drugs represented
    # dpi information. Fix those when/if we revive the combo code.
    def get_dpi_keys(self,drug):
        if not self.legacy:
            from drugs.models import DpiMergeKey
            try:
                # if it's in a cluster, return the cluster key
                rec = DpiMergeKey.objects.get(drug=drug,version=self.version)
                return [rec.dpimerge_key]
            except DpiMergeKey.DoesNotExist:
                # else, return the native key
                return [getattr(drug,drug.collection.key_name)]
        keyspace = self._get_key_of_name(self.choice)
        match_list = []
        for key in (keyspace,'m_'+keyspace):
            try:
                match_list += getattr(drug,key+'_id_set')
            except AttributeError:
                pass
        return match_list
    @classmethod
    def _get_native_key_prop_qs(cls):
        from drugs.models import Collection,Prop
        name_set=set(Collection.objects.all().values_list('key_name',flat=True))
        return Prop.objects.filter(name__in=name_set)
    def get_filtered_key_target_pairs(self, min_evid = default_evidence):
        '''Yields (key,target) pairs above min_evid threshold.

        '''
        # Currently this is only used for collection stats, so older mapping
        # files aren't supported.
        # TODO: Re-enable this once end-to-end test is off of legacy.
        # assert not self.legacy
        from dtk.files import get_file_records
        for row in get_file_records(
                        self.get_path(),
                        keep_header=False,
                        ):
                if float(row[2]) >= min_evid:
                    yield (row[0],row[1])
    def get_file_dpi_keys(self):
        """Returns all dpi keys with entries in this DPI file.

        Note that this differs from several other methods in this class that
        return all keys that exist in the current matching, regardless of whether
        they have DPI entries.
        """
        seen = set()
        from dtk.files import get_file_records
        for row in get_file_records(
                        self.get_path(),
                        keep_header=False,
                        ):
            if row[0] not in seen:
                yield row[0]
            seen.add(row[0])

    def get_key_agent_pairs(self,agent_ids):
        '''Yields (dpikey,agent_id) pairs.

        Note that without a restricting set of agent ids, this would return
        lots of irrelevant information from old collections probably not used
        in any workspace. Usually get_key_wsa_pairs is more efficient.
        '''
        from drugs.models import DpiMergeKey,Tag,Prop
        if self.legacy:
            keyspace = self._get_key_of_name(self.choice)
            prop_ids = set(Prop.objects.filter(
                    name__in=(keyspace+'_id','m_'+keyspace+'_id'),
                    ).values_list('id',flat=True))
            qs = Tag.objects.filter(
                    prop_id__in=prop_ids,
                    drug_id__in=agent_ids,
                    )
            for pair in qs.values_list('value','drug_id'):
                yield pair
        else:
            qs = DpiMergeKey.objects.filter(
                    version=self.version,
                    drug_id__in=agent_ids,
                    )
            found_agents = set()
            # return all cluster keys
            for pair in qs.values_list('dpimerge_key','drug_id'):
                found_agents.add(pair[1])
                yield pair
            
            # return all non-cluster keys
            qs = Tag.objects.exclude(
                    drug__dpimergekey__version=self.version,
                    ).filter(
                    prop__name=Prop.NATIVE_ID,
                    drug_id__in=set(agent_ids) - found_agents,
                    )
            for pair in qs.values_list('value','drug_id'):
                yield pair
    def get_key_wsa_pairs(self,ws,keyset=None):
        '''Yields (dpikey,wsa) pairs.
        '''
        from browse.models import WsAnnotation
        from drugs.models import Prop
        if self.legacy:
            keyspace = self._get_key_of_name(self.choice)
            prop_qs = Prop.objects.filter(
                    name__in=(keyspace+'_id','m_'+keyspace+'_id'),
                    )
            qs = WsAnnotation.objects.filter(
                    ws=ws,
                    agent__tag__prop__in=list(prop_qs),
                    )
            if keyset is not None:
                qs = qs.filter(agent__tag__value__in=keyset)

            for pair in qs.values_list('agent__tag__value','id'):
                yield pair
        else:
            filters = dict(ws=ws, agent__dpimergekey__version=self.version)
            if keyset is not None:
                filters['agent__dpimergekey__dpimerge_key__in'] = keyset
            qs = WsAnnotation.objects.filter(**filters)
            for pair in qs.values_list('agent__dpimergekey__dpimerge_key','id'):
                yield pair
            qs = WsAnnotation.objects.filter(
                    ws=ws,
                    agent__tag__prop__name=Prop.NATIVE_ID,
                    ).exclude(
                    agent__dpimergekey__version=self.version,
                    )
            if keyset is not None:
                qs = qs.filter(agent__tag__value__in=keyset)
            for pair in qs.values_list('agent__tag__value','id'):
                yield pair
    def get_drug_bindings_for_prot(self, prot, min_evid = None):
        if min_evid is None:
            min_evid = self.default_evidence
        return self.get_drug_bindings_for_prot_list([prot], min_evid)
    def get_drug_bindings_for_prot_list(self, prot_list,min_evid = None):
        from collections import namedtuple
        if min_evid is None:
            min_evid = self.default_evidence
        DpiBinding = None
        l = []
        from dtk.files import get_file_records
        for row in get_file_records(
                    self.get_best_path(),
                    select=(prot_list,1),
                    keep_header=True,
                    ):
            if not DpiBinding:
                # process header
                DpiBinding = namedtuple('DpiBinding',row)
                continue
            if float(row[2]) >= min_evid:
                l.append(DpiBinding(*row))
        return l
    def get_dpi_info(self,drug, min_evid = 0.0):
        '''Return a list of protein bindings for a drug.

        Find all bindings that match any of the relevant binding keys
        for this drug.  (Note that, because of imperfect matching, more
        than one key could be associated with the same drug.)  Each
        binding is represented by a namedtuple, which is guaranteed to
        have at least a drug identifier, uniprot_id, evidence, and
        direction fields, and may have additional fields as well.
        '''
        return self.get_dpi_info_for_keys(self.get_dpi_keys(drug), min_evid = min_evid)
    def get_best_path(self):
        """Returns the sqlite if possible, otherwise tsv."""
        if self.get_db_path():
            path = self.get_db_path()
        else:
            path = self.get_path()
        return path
    def get_dpi_info_for_keys(self,keys, min_evid = 0.0):
        '''Return a list of DpiBinding tuples.

        Fields are (native_key, uniprot_id, evidence, direction).
        '''
        path = self.get_best_path()

        min_evid=float(min_evid) # deal w/ strings from URLs, etc.
        from collections import namedtuple
        DpiBinding = None
        l = []
        from dtk.files import get_file_records
        for row in get_file_records(
                        path,
                        select=(keys,0),
                        allow_suspicious=True,
                        keep_header=True,
                        ):
                if not DpiBinding:
                    # process header
                    DpiBinding = namedtuple('DpiBinding',row)
                    continue
                if float(row[2]) >= min_evid:
                    l.append(DpiBinding(*row))
        return l
    # get_drug_id_map was removed because it's inherently slow
    # (see comment in get_key_agent_pairs), and all uses were better
    # handled other ways.
    # Some calls in dead or experimental code were left, and should be fixed
    # when and if needed.

    def get_prot_id_map(self):
        out = {}
        from dtk.files import get_file_records
        for row in get_file_records(self.get_path(), keep_header=False):
            out[row[0]] = [row[1]]
        return out


    def get_wsa_id_map(self,ws):
        '''Return a map from DPI keys to lists of wsa ids.
        '''
        if self.mapping_type() == 'uniprot':
            return self.get_prot_id_map()

        from dtk.data import MultiMap
        mm = MultiMap(self.get_key_wsa_pairs(ws))
        return {k:list(v)
                for k,v in mm.fwd_map().items()
                }
    def get_wsa2dpi_map(self, ws, wsa_id_list, min_evid=None, include_dir=False):
        if min_evid is None:
            min_evid = self.default_evidence
        native2wsa = self.get_wsa_id_map(ws)
        # scan DPI file
        wsa2dpi = {}
        from dtk.files import get_file_records
        for row in get_file_records(self.get_path(),
                            keep_header=False,
                            ):
            native,uniprot,ev = row[:3]
            if float(ev) < min_evid:
                continue
            # build a set of uniprots for each WSA
            gen = (wsa_id
                   for wsa_id in native2wsa.get(native,[])
                   if wsa_id in wsa_id_list
                  )
            for wsa_id in gen:
                if include_dir:
                    wsa2dpi.setdefault(wsa_id,set()).add((uniprot, row[3]))
                else:
                    wsa2dpi.setdefault(wsa_id,set()).add(uniprot)
        return wsa2dpi

    def get_uniq_target_cnt(self):
        return len(self.get_uniq_target())
    def get_uniq_target(self):
        from dtk.files import get_file_columns
        return set(x.rstrip('\n') for x in get_file_columns(self.get_path(),[2], keep_header=False))
    def get_filtered_uniq_target(self, min_evid = default_evidence):
        from dtk.files import get_file_records
        to_ret = set()
        for frs in get_file_records(self.get_path(),
                    keep_header=False):
            if float(frs[2]) >= min_evid:
                to_ret.add(frs[1])
        return to_ret
    def get_uniq_mol_cnt(self):
        from dtk.files import get_file_columns
        return len(set(x.rstrip('\n') for x in get_file_columns(self.get_path(),[1], keep_header=False)))

    def is_combo(self):
        return '+' in self.choice

    def combo_name(self):
        import re
        return re.match(r'.*\+(.*?)\.', self.choice)[1]

    def get_noncombo_dpi(self):
        if not self.is_combo():
            return self
        # Remove the "+<basedrug>" portion.
        import re
        base_choice = re.sub(r'\+.*?\.', '.', self.choice)
        return DpiMapping(base_choice)
    
    def get_dpi_type(self):
        if self.is_combo():
            return 'combo'
        elif '-' not in self.choice:
            return 'baseline'
        else:
            import re
            return re.match(r'.*\-(.*?)\.', self.choice)[1]

    def get_baseline_dpi(self):
        import re
        baseline_choice = re.sub(r'[-+].*?\.', '.', self.choice)
        if baseline_choice == self.choice:
            return self
        else:
            return DpiMapping(baseline_choice)
        
    def dpimap_for_wsa(self, wsa):
        """Returns the correct moa/non-moa dpi variant of this dpi file for the given MoA."""
        wsa_moa = wsa.is_moa()
        self_moa = self.get_dpi_type() == 'moa'
        if wsa_moa == self_moa:
            return self

        if wsa_moa and not self_moa:
            from dtk.moa import moa_dpi_variant
            return DpiMapping(moa_dpi_variant(self.choice))
        else:
            return self.get_baseline_dpi()


# some clients of get_drug_id_map and get_wsa_id_map prefer the ids as
# strings rather than numbers; this will do the conversion
def stringize_value_lists(d):
    return {
            k:[str(x) for x in v]
            for k,v in six.iteritems(d)
            }

class MultiAgentTargetCache:
    """Wrapper around agent target caches.

    Primarily for switching between moa and non-moa variants.
    """
    def __init__(self, atc):
        self.atcs = [atc]
        if not atc.mapping.legacy and not atc.mapping.mapping_type() == 'uniprot':
            if atc.mapping.get_dpi_type() == 'moa':
                dpi2 = atc.mapping.get_baseline_dpi()
            else:
                from dtk.moa import moa_dpi_variant
                dpi2 = DpiMapping(moa_dpi_variant(atc.mapping.choice))
            
            atc2 = AgentTargetCache(
                mapping=dpi2,
                agent_ids=atc.agent_ids,
                dpi_thresh=atc.dpi_thresh
            )
            self.atcs.append(atc2)
    
    def atc_for_agent(self, agent_id):
        for atc in self.atcs:
            if agent_id in atc.agent2info:
                return atc

        # Return any instead of None, less special case handling.
        return self.atcs[0]

    def raw_info_for_agent(self, agent_id):
        return self.atc_for_agent(agent_id).raw_info_for_agent(agent_id)

class AgentTargetCache(LazyLoader):
    _kwargs=['mapping','agent_ids','dpi_thresh']
    @staticmethod
    def atc_for_wsas(wsas, ws=None, dpi_mapping=None, dpi_thresh=None):
        assert ws or (dpi_mapping and dpi_thresh), "Provide ws or dpi parms"

        from django.db.models.query import QuerySet
        if isinstance(wsas, QuerySet):
            agent_ids = wsas.values_list('agent_id', flat=True)
        elif wsas and isinstance(wsas[0], (int, str)):
            from browse.models import WsAnnotation
            wsas = WsAnnotation.objects.filter(pk__in=wsas)
            agent_ids = wsas.values_list('agent_id', flat=True)
        else:
            agent_ids = [x.agent_id for x in wsas]


        return AgentTargetCache(
                mapping=dpi_mapping or DpiMapping(ws.get_dpi_default()),
                dpi_thresh=dpi_thresh or ws.get_dpi_thresh_default(),
                agent_ids=list(agent_ids),
                )


    def raw_info_for_agent(self,agent_id):
        # return list of DpiBinding objects, as per get_dpi_info_for_keys
        return self.agent2info.get(agent_id,[])
    def full_info_for_agent(self,agent_id):
        # return list of (native_key,uniprot,gene,evidence,direction)
        return [
                (
                        x[0],
                        x.uniprot_id,
                        self.prot2gene.get(x.uniprot_id,'('+x.uniprot_id+')'),
                        float(x.evidence),
                        float(x.direction),
                )
                for x in self.agent2info.get(agent_id,[])
                ]
    def info_for_agent(self,agent_id):
        # return list of (uniprot,gene,direction) tuples, sorted by gene
        result = [
                (
                        x.uniprot_id,
                        self.prot2gene.get(x.uniprot_id,'('+x.uniprot_id+')'),
                        float(x.direction),
                )
                for x in self.agent2info.get(agent_id,[])
                ]
        return sorted(result,key=lambda x:x[1])
    def build_note_cache(self,ws,user):
        from browse.models import TargetAnnotation
        return TargetAnnotation.batch_note_lookup(
                ws,
                self.all_prots,
                user,
                )
    def _key2agent_loader(self):
        from dtk.data import MultiMap
        mm = MultiMap(
                self.mapping.get_key_agent_pairs(agent_ids=self.agent_ids)
                )
        return mm.fwd_map()
    def _agent2info_loader(self):
        result = {}
        for info in self.mapping.get_dpi_info_for_keys(
                list(self.key2agent.keys()),
                self.dpi_thresh,
                ):
            for agent_id in self.key2agent.get(info[0],[]):
                l = result.setdefault(agent_id,[])
                l.append(info)
        return result
    def _all_prots_loader(self):
        return set([
                x.uniprot_id
                for l in self.agent2info.values()
                for x in l
                ])
    def _prot2gene_loader(self):
        from browse.models import Protein
        return Protein.get_uniprot_gene_map(self.all_prots)

def protein_link(uniprot,gene,ws,
        note_cache={},
        direction=None,
        use_uniprot_label=False,
        consultant_link=False,
        ):
    '''Return html for a link to a protein page.

    The link text will be the gene name.
    If note_cache is set, it is assumed to be a dict as returned by
    TargetAnnotation.batch_note_lookup() and is used to append note
    icons as appropriate.
    If direction is not None, it is used to supply a direction icon.
    '''
    if not ws:
        # Use the test workspace if none provided.
        from browse.models import Workspace
        ws = Workspace.objects.get(pk=42)
    if isinstance(ws, (str, int)):
        from browse.models import Workspace
        ws = Workspace.objects.get(pk=ws)
    from dtk.html import link,join,glyph_icon,tag_wrap,nowrap
    page = 'protein' if not consultant_link else 'consultant_protein'
    parts = [link(
            uniprot if use_uniprot_label else gene,
            ws.reverse(page,uniprot),
            )]
    note_info = note_cache.get(uniprot)
    if note_info:
        txt = join(*[
                join(tag_wrap('b',name+': '),note)
                for name,(note_id,note) in sorted(six.iteritems(note_info))
                ],sep=u'<br>')
        # the direction arrows look better if there's no separation,
        # but the comment icon looks better if there's a space; so
        # do no separation by default, but pad here.
        parts.append(' ')
        parts.append(glyph_icon('comment',hover=txt,html=True))
    if direction is not None:
        from dtk.plot import dpi_arrow
        parts.append(dpi_arrow(direction))
    if len(parts) == 1:
        return parts[0]
    return nowrap(join(*parts,sep=''))


class AgentAssays(LazyLoader):
    # This class handles the parsing and formatting of assay information
    # from C50 and ki affinity files. The job of locating the files, and
    # determining the native molecule keys withing the files to be extracted,
    # is delegated to a 'cluster' object, which must have:
    # - an assay_files attribute, which is a list of (src,atype,s3f) tuples,
    #   where:
    #   - src is an ETL directory where the data was extracted (chembl,
    #     bindingdb, etc.)
    #   - atype is the assay type in the file (c50,ki)
    #   - s3f is an S3File object that can be used to retrieve and access
    #     the affinity file
    # - a src2keys attribute, which is a dict containing the set of keys
    #   matching this molecule for each source (like
    #   {'chembl':set(['CHEMBL25']),...}
    # For versioned dpi sources, the RebuiltCluster class in dtk.drug_clusters
    # provides this functionality. For legacy dpi sources, this is provided
    # by the AgentAssaysClusterProxy class below.
    _kwargs=['cluster']
    info_cols = [
                'assay_type',
                'drug_key',
                'protein',
                'gene',
                'direction',
                'nm',
                'assay_count',
                'std_dev',
                ]
    def assay_info(self):
        return [
                row[:3]+[self.prot2gene.get(row[2])]+row[3:]
                for row in self.assays
                ]
    def _prot2gene_loader(self):
        all_prots = set(x[2] for x in self.assays)
        from browse.models import Protein
        return Protein.get_uniprot_gene_map(all_prots)
    def _assays_loader(self):
        result = []
        from dtk.files import VersionedFileName
        from dtk.files import get_file_records
        for file_class,atype,s3f in self.cluster.assay_files:
            try:
                keys = self.cluster.src2keys[file_class]
            except KeyError:
                continue
            s3f.fetch()
            # input rec is:
            # (drugkey,uniprot,direction,c50_or_ki,n_measurements,std_dev)
            for rec in get_file_records(
                    s3f.path(),
                    select=(keys,0),
                    keep_header=False,
                    ):
                result.append([atype]+rec[:2]+[
                        int(rec[2]),
                        float(rec[3]),
                        int(rec[4]),
                        float(rec[5]),
                        ])
        return result

class AgentAssaysClusterProxy(LazyLoader):
    _kwargs=['agent']
    assay_types = ['c50','ki']
    def _s3b_loader(self):
        from dtk.s3_cache import S3MiscBucket
        return S3MiscBucket()
    def _assay_fns_loader(self):
        result = []
        for fn in self.s3b.list(cache_ok=True):
            parts = fn.split('.')
            if parts[0] in self.assay_types:
                result.append(fn)
        return result
    def _assay_files_loader(self):
        from dtk.s3_cache import S3File
        result = []
        for fn in self.assay_fns:
            parts = fn.split('.')
            atype = parts[0]
            file_class = parts[1]
            result.append((file_class,atype,S3File(self.s3b,fn)))
        return result
    def _src2keys_loader(self):
        result = {}
        keynames=set(x.split('.')[1] for x in self.assay_fns)
        for keyname in keynames:
            s = result.setdefault(keyname,set())
            sname = keyname+'_id_set'
            s |= getattr(self.agent,sname)
            s |= getattr(self.agent,'m_'+sname)
        return result

class RawAgentAssays(LazyLoader):
    # Like AgentAssays, but for the raw assays format.
    _kwargs=['cluster']
    info_cols = [
                'drug_key',
                'protein',
                'gene',
                'assay_type',
                'relation',
                'nm',
                'direction',
                'assay_format',
                ]
    def assay_info(self):
        return [
                row[:2]+[self.prot2gene.get(row[1])]+row[2:]
                for row in self.assays
                ]
    def _prot2gene_loader(self):
        all_prots = set(x[1] for x in self.assays)
        from browse.models import Protein
        return Protein.get_uniprot_gene_map(all_prots)
    def _assays_loader(self):
        result = []
        from dtk.files import VersionedFileName
        from dtk.files import get_file_records
        for file_class,atype,s3f in self.cluster.raw_assay_files:
            try:
                keys = self.cluster.src2keys[file_class]
            except KeyError:
                continue
            s3f.fetch()
            # input rec is:
            # ['chembl_id', 'uniprot_id', 'type', 'relation', 'value', 'direction', 'assay_format']
            for rec in get_file_records(
                    s3f.path(),
                    select=(keys,0),
                    keep_header=False,
                    ):
                result.append(rec[:4]+[
                        float(rec[4]),
                        int(rec[5]),
                        rec[6],
                        ])
        return result
