from django.db import models
from django.db import connection
from django.db import transaction
from django.conf import settings
from django.urls import reverse
from django.utils.safestring import mark_safe
from django.utils import timezone
from django.db.models import Min
import os
import csv
import pwd
import datetime
import numpy as np
import scipy.stats as sps
from tools import make_sure_dir_exists, copy_if_needed, Enum
from path_helper import PathHelper,make_directory
from notes.models import Note
from runner.models import Process
from drugs.models import Drug,Prop,Collection
from tools import sci_fmt,obfuscate
from functools import reduce

import logging
import six
logger = logging.getLogger(__name__)

####################################################################
# non-DB models
####################################################################
class MLScoreReader:
    def __init__(self,filename):
        self.filename = filename
        try:
            self.rdr=csv.reader(open(filename,"rb"))
            self.ok=True
        except Exception as e:
            logger.error(
                "MLScoreReader open problem '%s': %s"
                ,self.filename,repr(e)
                )
            self.ok=False
    def next(self,line,dbid):
        if self.ok:
            try:
                row = next(self.rdr)
                assert row[0] == dbid
                return row[1]
            except Exception as e:
                logger.error(
                    "MLScoreReader load problem '%s' line %d: %s"
                    ,self.filename,line,repr(e)
                    )
                self.ok = False
        return None

class Likelihood:
    def __init__(self,ws):
        self.target_val = WsAnnotation.indication_vals.KNOWN_TREATMENT
        self.ws = ws
        self.overall_ratio = (
                WsAnnotation.objects.filter( ws=ws
                        , indication=self.target_val
                        ).count(),
                WsAnnotation.objects.filter( ws=ws
                        ).count(),
                )
    def likelihood(self,ratio):
        total = np.array([float(self.overall_ratio[1]-self.overall_ratio[0])
                ,float(self.overall_ratio[0])
                ])
        observed = np.array([ratio[1]-ratio[0],ratio[0]])
        expected = total*np.sum(observed)/np.sum(total)
        (chi,prob) = sps.chisquare(observed,expected)
        return prob
    def eval_subset(self,qs):
        # return (qs_ratio,qs_likelihood,cutoff,cutoff_ratio,cutoff_likelihood)
        # each ratio is (KT count, total count)
        known = qs.filter(drug_ws__indication=self.target_val)
        known_count = known.count()
        qs_ratio = (known_count,qs.count())
        if self.overall_ratio[0] == 0 or qs_ratio[1] == 0 or known_count == 0:
            return ((qs_ratio),None,None,None,None)
        cutoff = known.aggregate(Min('direct_score'))['direct_score__min']
        cutoff_ratio = (known_count
                ,qs.filter(direct_score__gte=cutoff).count()
                )
        return (qs_ratio,self.likelihood(qs_ratio)
            ,cutoff
            ,cutoff_ratio,self.likelihood(cutoff_ratio)
            )
    def ratio_as_text(self,ratio):
        return "%d/%d known treatments (of %d/%d)" % (
                    ratio[0],ratio[1],self.overall_ratio[0],
                    self.overall_ratio[1],
                    )
    def as_text(self,struct):
        # returns a list of 1-3 text lines
        result = [self.ratio_as_text(struct[0])]
        if struct[1] is not None:
            result[0] += ", likelihood "+sci_fmt(struct[1])
            result.append("the minimum direct score for a known treatment is "
                    + sci_fmt(struct[2])
                    )
            if struct[2] != 0:
                    result[1] + "; above this are:"
                    result.append(self.ratio_as_text(struct[3])
                            + ", likelihood "+sci_fmt(struct[4])
                            )
        return result

from dtk.lazy_loader import LazyLoader
class WsCollectionInfo(LazyLoader):
    '''Retrieve information about the import status of a workspace.
    '''
    # This class uses the LazyLoader model to avoid doing any setup
    # work that isn't needed for the context it is called in.
    #
    # The latest incarnation of this class does all the calculations
    # based on agent ids. In some pathological cases, a workspace
    # may contain more WSAs than agents:
    # - past bugs resulted in multiple WSAs for the same agent
    # - the agent for a WSA is removed from the collection upload
    #   (the agent is still present and accessible, but doesn't appear
    #   in collection size counts)
    # These cases are called out separately on the impstat page to
    # simplify the import blocking logic.
    def __init__(self,ws_id,**kwargs):
        self.ws_id = ws_id
        self.clust_keys_cache = {}
        super(WsCollectionInfo,self).__init__(**kwargs)
    def import_counts(self,collection):
        all_agents = self.coll_to_all_agents.get(collection.id,set())
        ws_agents = self.coll_to_ws_agents.get(collection.id,set())
        blocked_agents = self.blocked_agents(collection)
        return (
                len(all_agents),
                len(ws_agents),
                len(all_agents - ws_agents - blocked_agents),
                len(blocked_agents - ws_agents),
                )
    def _coll_to_ws_agents_loader(self):
        from browse.models import WsAnnotation
        from dtk.data import MultiMap
        mm = MultiMap(WsAnnotation.objects.filter(
                ws_id=self.ws_id,
                agent__removed=False,
                ).values_list('agent__collection_id','agent'))
        return mm.fwd_map()
    def _coll_to_all_agents_loader(self):
        from dtk.data import MultiMap
        from drugs.models import Drug
        mm = MultiMap(Drug.objects.filter(
                removed=False,
                ).values_list('collection_id','id'))
        return mm.fwd_map()
    def _ws_agent_counts_loader(self):
        return {k:len(s) for k,s in self.coll_to_ws_agents.items()}
    def _all_agent_counts_loader(self):
        return {k:len(s) for k,s in self.coll_to_all_agents.items()}
    def clust_keys_for_collection_set(self,collection_ids):
        cache_key = frozenset(collection_ids)
        if cache_key not in self.clust_keys_cache:
            from drugs.models import DpiMergeKey
            from dtk.data import MultiMap
            mm = MultiMap(DpiMergeKey.objects.filter(
                    version=self.version,
                    drug__collection_id__in=collection_ids,
                    drug__removed=False,
                    ).values_list('dpimerge_key','drug_id'))
            self.clust_keys_cache[cache_key] = mm.fwd_map()
        return self.clust_keys_cache[cache_key]
    def clust_key_to_coll_agents(self,collection):
        '''Return {clust_key:{agent_id,...},...} for the specified collection.

        This will only include agents in the collection of interest, so only
        some clusters will be represented, and they won't have all their
        members.
        '''
        return self.clust_keys_for_collection_set([collection.id])
    def _ws_clust_key_to_agents_loader(self):
        ws_colls = set(self.coll_to_ws_agents.keys())
        all_ck2a = self.clust_keys_for_collection_set(ws_colls)
        ws_ck2a = {}
        for k,s in all_ck2a.items():
            in_ws = s & self.ws_agent_id_set
            if in_ws:
                ws_ck2a[k] = in_ws
        return ws_ck2a
    def _ws_clust_keys_loader(self):
        return set(self.ws_clust_key_to_agents.keys())
    def ws_has_different_subset(self,collection):
        for c in self.all_collections:
            if c.key_name != collection.key_name:
                continue
            if c.id == collection.id:
                continue
            if c.id in self.coll_to_ws_agents.keys():
                return True
        return False
    def blocked_agents(self,collection,ckey2agents=None):
        '''Return the set of blocked agent ids for the collection.

        Agents may be blocked because:
        - they are in a cluster already present in the workspace
        - they are in a cluster not already present, but which has
          multiple drugs from the collection in question, so only
          one should be imported.
        - they are the same molecule as an agent already in the workspace
          from a different collection subset
        The optional ckey2agents parameter allows the cluster
        map to be precalculated and reused for efficiency.

        Note that agents already in the workspace may or may not be
        included in the returned set, so further filtering may be
        mecessary depending on the application.
        '''
        if ckey2agents is None:
            ckey2agents = self.clust_key_to_coll_agents(collection)
        result = set()
        multi_drugs=0
        multi_clusters=0
        for k,s in ckey2agents.items():
            if k in self.ws_clust_keys:
                result |= s
            elif len(s) > 1:
                # This key isn't blocked by something already in the
                # workspace, but it contains multiple agent ids, and
                # we only want to allow one. Block all but the lowest.
                multi_clusters += 1
                multi_drugs += len(s)-1
                result |= set(sorted(s)[1:])
        if self.ws_has_different_subset(collection):
            # get all non-clustered ws agents
            # XXX this could be further limited by collection
            ck2a = self.ws_clust_key_to_agents
            clustered_agents = set.union(*ck2a.values()) if ck2a else set()
            non_clustered_agents = self.ws_agent_id_set - clustered_agents
            # get the native keys of those matching this collection
            from drugs.models import Tag
            in_ws_keys = set(Tag.objects.filter(
                    prop__name=collection.key_name,
                    drug__in=non_clustered_agents,
                    ).values_list('value',flat=True))
            # get this collection's agents with any of those keys
            subset_blocked_agents = set(Tag.objects.filter(
                    prop__name=collection.key_name,
                    value__in=in_ws_keys,
                    drug__collection_id=collection.id,
                    ).values_list('drug_id',flat=True))
            # add them to the blocked list
            subset_blocked_count = len(subset_blocked_agents - result)
            result |= subset_blocked_agents
        else:
            subset_blocked_count = 0
        logger.info("%s ws %d %s %s %s",
                collection.name,
                self.ws_id,
                f'{len(result)} blocked;',
                f'{multi_drugs} self blocked in {multi_clusters} clusters;',
                f'{subset_blocked_count} outside clusters but in subsets',
                )
        return result
    def count_used_drugs(self,collection):
        return self.used_drug_count.get(collection.id,0)
    def count_unused_drugs(self,collection):
        '''Return the number of unused drugs imported from a collection

        Conceptually, an 'unused' drug is one that nothing interesting
        has happened to in the workspace. See the code of
        unused_drug_criteria() to see the technical definition.
        '''
        return self.ws_agent_counts.get(collection.id,0) \
                - self.used_drug_count.get(collection.id,0)
    def clear_unused_drugs(self,collection,user):
        '''Mark unused drugs from the collection as invalid.
        '''
        from wsmgr.models import ImportAudit
        ia = ImportAudit.objects.create(
                ws_id=self.ws_id,
                collection=collection,
                operation='clear_unused',
                user=user,
                clust_ver=self.version,
                )
        self.unused_drug_filter.filter(
                agent__collection=collection,
                ).update(invalid=True)
        ia.succeeded = True
        ia.save()
    # below are on-demand loaders for the various data members used
    # by the methods above
    def _version_loader(self):
        '''Return clustering version (or None).'''
        ws = Workspace.objects.get(pk=self.ws_id)
        return ws.get_dpi_version()
    def _blocked_wsa_ids_loader(self):
        # Due to previous bugs, there are legacy cases in production
        # workspaces where two WSAs in a workspace point to the same
        # agent. We should allow at most one of these into the
        # workspace. So:
        # - gather all aliases (based on all_objects)
        # - for each:
        #   - if any wsa is already valid, block all others
        #   - else, block all but one (the oldest)
        from dtk.data import MultiMap
        mm=MultiMap(WsAnnotation.all_objects.filter(
                ws_id=self.ws_id,
                ).values_list('agent_id','id'))
        aliases={k:s for k,s in mm.fwd_map().items() if len(s)>1}
        aliased_wsas=set.union(*aliases.values()) if aliases else set()
        active_aliased_wsas=set(WsAnnotation.objects.filter(
                pk__in=aliased_wsas,
                ).values_list('id',flat=True))
        blocked_wsas = set()
        for k,s in aliases.items():
            if s & active_aliased_wsas:
                blocked_wsas |= s - active_aliased_wsas
            else:
                blocked_wsas |= set(sorted(s)[1:])
        return blocked_wsas
    def _ws_agent_id_set_loader(self):
        '''Return agent ids in the workspace.'''
        if self.coll_to_ws_agents:
            return set.union(*self.coll_to_ws_agents.values())
        return set()
    def _all_collections_loader(self):
        return Collection.import_ordering()
    def unused_drug_criteria(self):
        return dict(
                vote__id__isnull=True,
                drugset__id__isnull=True,
                dispositionaudit__id__isnull=True,
                indication=0,
                marked_on__isnull=True,
                indication_protection=0,
                study_note_id__isnull=True,
                )
    def _used_drug_filter_loader(self):
        # return a qs filtering used WSAs in the workspace
        return WsAnnotation.objects.filter(
                    ws_id=self.ws_id,
                    ).exclude(
                    **self.unused_drug_criteria()
                    )
    def _unused_drug_filter_loader(self):
        # return a qs filtering unused WSAs in the workspace
        return WsAnnotation.objects.filter(
                    ws_id=self.ws_id,
                    **self.unused_drug_criteria()
                    )
    def _used_drug_count_loader(self):
        # return a dict mapping a collection id to a count of unused drugs.
        from django.db.models import Count
        return {
            x['agent__collection_id']:x['id__count']
            for x in self.used_drug_filter.values(
                    'agent__collection_id'
                    ).annotate(Count('id'))
            }

####################################################################
# DB models
####################################################################
class Workspace(models.Model):
    name = models.CharField(max_length=256,default="")
    active = models.BooleanField(default=False)
    cross_compare = models.BooleanField(null=True)
    disease_note = models.ForeignKey(Note,null=True,blank=True, on_delete=models.CASCADE, related_name='note')
    hitsel_note = models.ForeignKey(Note,null=True,blank=True, on_delete=models.CASCADE, related_name='ws_hitsel_note')
    created = models.DateTimeField(auto_now_add=True,null=True)

    @property
    def eval_drugset(self):
        from browse.default_settings import EvalDrugset
        return EvalDrugset.value(self)

    @property
    def non_moa_eval_drugset(self):
        ds = self.eval_drugset
        prefix = 'moa-'
        if ds.startswith(prefix):
            return ds[len(prefix):]
        return ds

    def get_short_name(self):
        from browse.default_settings import DiseaseShortName
        return DiseaseShortName.value(self)

    def make_short_name(self):
        import re
        pieces = re.split(r'[ -]+', self.name)
        letters = [piece[0] for piece in pieces if len(piece)]
        if len(letters) == 1:
            return self.name[:4]
        elif len(letters) == 2:
            return '%s %s' % (pieces[0][:3], pieces[1][:3])

        return ''.join(letters)

    def get_disease_note_text(self):
        return Note.get(self,'disease_note','')
    def note_info(self,attr):
        if attr == 'disease_note':
            return {
                'label':"disease note for %s" %(
                        self.name,
                        ),
                }
        elif attr == 'hitsel_note':
            return {
                'label':"hit selection note for %s" %(
                        self.name,
                        ),
                }
        raise Exception("bad note attr '%s'" % attr)
    def __str__(self):
        return "%s (%s)" % (self.name,self.id)
    def ws_dir(self):
        dir =  PathHelper.storage+str(self.id)
        make_directory(dir)
        return dir
    def check_defaults(self):
        '''Make sure all parts of a workspace are present.

        This function acts somewhat like a constructor, making sure
        everything necessary in a workspace is present.  By doing
        this dynamically and idempotently, we can easily evolve
        workspaces whose definition has been extended since that
        instance was created.  This is mostly to support evolution
        during the development phase, but may be a reasonable
        strategy in the long term as well.
        '''
        path =  PathHelper.ws_publish(self.id)
        make_directory(path)
        # this used to pull things from the 2xar/data directory,
        # but that mechanism has been removed; there are no longer
        # any static files that need to get copied into each workspace
    def reverse(self,view,*args):
        '''Return the URL for a given view in this workspace'''
        return self.ws_reverse(view, self.id, *args)
    @classmethod
    def ws_reverse(cls,view,ws_id,*args):
        """This is faster than resolving a WS instance if you have an ID."""
        return reverse(view,args=tuple([ws_id] + list(args)))
    ##########
    # workspace data access
    ##########

    @classmethod
    def get_fixed_wsa_id_set_choices(cls):
        return [
                ( x[0], x[1] )
                for x in WsAnnotation.indication_groups()
                ]

    # drugsets
    def get_wsa_id_set_choices(self,fixed_only=False,train_split=False,test_split=False,retro=False,ds_only=False):
        ds_choices = [
                    ( 'ds%d'%rec.id, rec.name )
                    for rec in DrugSet.objects.filter(
                                    ws=self,
                                    ).order_by('-id')
                    ]
        if ds_only:
            return ds_choices
        # gather 'fixed' choices
        fixed = self.get_fixed_wsa_id_set_choices()
        fixed += self.get_ct_drugset_choices()
        result = list(fixed)
        # add in corresponding MoA-based choices
        result += [('moa-' + x[0], 'MoA ' + x[1]) for x in fixed]
        if not fixed_only:
            result += ds_choices
        base_result = list(result)
        if train_split:
            for ds_id, ds_name in base_result:
                result += [
                        ('split-train-%s' % ds_id, '%s (Train)' % ds_name)
                    ]
        if test_split:
            for ds_id, ds_name in base_result:
                result += [
                        ('split-test-%s' % ds_id, '%s (Test)' % ds_name)
                    ]
        if retro:
            result += [
                ( x[0], x[1] )
                for x in WsAnnotation.retro_groups()
                ]

        return result
    def get_wsa_id_set(self,name):
        '''Return a set of wsa_ids given a name.

        This should eventually replace everywhere in the system where
        we compare things to 'known treatments', so that we can easily
        substitute other target sets of drugs.
        '''
        if name.startswith('ds'):
            # it's a user-defined drugset
            ds_id = int(name[2:])
            return set(
                    DrugSet.objects.get(
                            ws=self,
                            id=ds_id,
                            ).drugs.values_list('id',flat=True)
                    )
        if name.startswith('wsas'):
            # it's an ad-hoc list of wsa ids
            l = [int(x) for x in name[4:].split(',')]
            return set(l)

        if name.startswith(self.ct_drugset_prefix):
            return self.get_ct_drugset(name[len(self.ct_drugset_prefix):])

        if name.startswith('split-'):
            from dtk.kt_split import get_split_drugset
            return get_split_drugset(name, self)

        if name.startswith('moa-'):
            from dtk.moa import make_wsa_to_moa_wsa
            base = self.get_wsa_id_set(name[4:])
            return set(make_wsa_to_moa_wsa(base, ws=self, pick_canonical=True).values())

        if name.startswith('retro_'):
            from dtk.retrospective import mol_selector
            return mol_selector(name[6:], self)
        fs_prefix = "flagset_"
        if name.startswith(fs_prefix):
            fs_id = int(name[len(fs_prefix):])
            qs = WsAnnotation.objects.filter(flag__run_id=fs_id)
            return set(qs.values_list('id', flat=True))

        if name == 'wseval':
            assert self.eval_drugset != name, "Can't set eval drugset to eval"
            return self.get_wsa_id_set(self.eval_drugset)

        if name.startswith('unreplaced-'):
            base_name = name[len('unreplaced-'):]
            wsas = self.get_wsa_id_set(base_name)
            from dtk.retrospective import unreplaced_mols
            return set(unreplaced_mols(wsas).values_list('id', flat=True))

        try:
            ind_vals = WsAnnotation.indication_group_members(name)
            return set(
                    WsAnnotation.objects.filter(
                            ws=self,
                            indication__in=ind_vals,
                            ).values_list('id',flat=True)
                    )
        except ValueError:
            pass
        raise Exception('unknown wsa_id_set name')
    # Drugsets based on clinical trial results
    ct_drugset_prefix = 'ct-'
    # XXX A method could be added to leverage the same data used below
    # XXX in order to identify problematic cases for review:
    # XXX - ph2 or ph3 drugs with no explicit status set
    # XXX - explicit status isn't what is implied by indication
    # XXX - ph2 failed but ph3 succeeded
    @classmethod
    def _get_ct_indication_data(cls):
        '''Return a dict of CT implications for indication values.

        Return value: {indication:(inferred ph2, inferred ph3),...}
        - inferred ph2 and ph3 are bools, indicating whether this
          indication implies passing or not.
        - presence of an indication as a dictionary key implies that
          some clinical trial data should exist:
          - (False, False) - explicit CT status should exist
          - (True, False) - explicit CT status should exist for ph3;
            ph2 should be Success if it exists
          - (True, True) - explicit CT status not needed; should both
            be Success if present
        '''
        iv = WsAnnotation.indication_vals
        return {
                iv.FDA_TREATMENT:(True,True),
                iv.KNOWN_TREATMENT:(True,True),
                iv.TRIALED3_TREATMENT:(True,False),
                iv.TRIALED2_TREATMENT:(False,False),
                iv.TRIALED_TREATMENT:(False,False),
                }
    @classmethod
    def get_ct_drugset_choices(cls):
        # Note that the labels and ordering below can be changed, but the
        # code must be preserved.
        return [
                (cls.ct_drugset_prefix+code_part, label)
                for label,code_part in [
                        ('PH3 Pass','ph3pass'),
                        ('PH3 Fail','ph3fail'),
                        ('PH3 Ongoing','ph3ongo'),
                        ('PH2 Pass','ph2pass'),
                        ('PH2 Fail','ph2fail'),
                        ('PH2 Ongoing','ph2ongo'),
                        ('CT Fail','ctfail'),
                        ]
                ]
    def _get_ct_explicit_set(self,include_test,explicit_test):
        '''Return two sets of wsa_ids matching passed-in criteria.

        include_test and explicit_test are callables taking a
        ClinicalTrialAudit record and returning a boolean.
        The result is a pair of sets of wsa_ids for which the
        CTA record returned True in the corresponding test.

        The intent is that include_test will flag all the CTAs
        that meet a particular criterion, and explicit_test will
        flag all the CTAs for which explicit results of the relevant
        type are available (and so, for which no implicit test should
        be done).
        '''
        result = set()
        explicit = set()
        from moldata.models import ClinicalTrialAudit
        for cta in ClinicalTrialAudit.get_latest_ws_records(self.id):
            if explicit_test(cta):
                explicit.add(cta.wsa_id)
            if include_test(cta):
                result.add(cta.wsa_id)
        return(result,explicit)
    def _get_ct_implicit_set(self,ind_test,explicit):
        '''Return a set of wsa_ids matching an implicit condition.

        This means all WSAs that:
        - are not in the passed-in explicit set
        - have an indication listed in _get_ct_indication_data where the
          associated data tuple matches ind_test
        ind_test is a callable taking a _get_ct_indication_data item value
        as a parameter.
        '''
        ind_vals = {
                k
                for k,v in self._get_ct_indication_data().items()
                if ind_test(v)
                }
        return set(
                WsAnnotation.objects.filter(
                        indication__in=ind_vals,
                        ws=self,
                ).exclude(
                        pk__in=explicit,
                ).values_list('pk',flat=True)
                )
    def get_ct_drugset(self,encoding):
        from moldata.models import ClinicalTrialAudit
        sv = ClinicalTrialAudit.ct_status_vals
        if encoding == 'ph3pass':
            result,explicit = self._get_ct_explicit_set(
                    lambda cta:cta.ph3_status == sv.PASSED,
                    lambda cta:cta.ph3_status != sv.UNKNOWN,
                    )
            result |= self._get_ct_implicit_set(
                    lambda idata:idata[1],
                    explicit,
                    )
        elif encoding == 'ph3fail':
            result,explicit = self._get_ct_explicit_set(
                    lambda cta:cta.ph3_status == sv.FAILED,
                    lambda cta:False,
                    )
        elif encoding == 'ph3ongo':
            result,explicit = self._get_ct_explicit_set(
                    lambda cta:cta.ph3_status == sv.ONGOING,
                    lambda cta:False,
                    )
        elif encoding == 'ph2pass':
            result,explicit = self._get_ct_explicit_set(
                    lambda cta:cta.ph2_status == sv.PASSED,
                    lambda cta:cta.ph2_status != sv.UNKNOWN,
                    )
            result |= self._get_ct_implicit_set(
                    lambda idata:idata[0],
                    explicit,
                    )
        elif encoding == 'ph2fail':
            result,explicit = self._get_ct_explicit_set(
                    lambda cta:cta.ph2_status == sv.FAILED,
                    lambda cta:False,
                    )
        elif encoding == 'ph2ongo':
            result,explicit = self._get_ct_explicit_set(
                    lambda cta:cta.ph2_status == sv.ONGOING,
                    lambda cta:False,
                    )
        elif encoding == 'ctfail':
            result,explicit = self._get_ct_explicit_set(
                    lambda cta:cta.ph2_status == sv.FAILED \
                        or cta.ph3_status == sv.FAILED,
                    lambda cta:False,
                    )
        else:
            raise NotImplementedError()
        return result

    def get_hidden_wsa_ids(self):
        return set(
                WsAnnotation.objects.filter(ws=self,
                        agent__hide=True,
                        ).values_list('id',flat=True)
                )
    # protein sets
    def get_uniprot_set_name(self, set_id):
        choices = self.get_uniprot_set_choices()
        for id, name in choices:
            if id == set_id:
                return name
        return None

    @classmethod
    def get_global_uniprot_set_choices(cls):
        return [
                ( 'globps_' + rec[0], rec[1] )
                for rec in ProtSet.get_global_protsets()
                ]

    def get_uniprot_set_choices(self, auto_dpi_ps=True):
        result = [('autops_none', 'None')]
        result += [
                ( 'ps%d'%rec.id, rec.name )
                for rec in ProtSet.objects.filter(
                                ws=self,
                                ).order_by('-id')
                ]

        result += [
                ('autops_wsunwanted', "Disease Unwanted (Intolerable + Non-Novel)")
                ]

        result += self.get_global_uniprot_set_choices()

        if auto_dpi_ps:
            for drugset_id, drugset_name in self.get_wsa_id_set_choices():
                result += [
                        ('ds_0.5_%s' % drugset_id, '%s DPI >= 0.5' % drugset_name),
                        ('ds_0.9_%s' % drugset_id, '%s DPI >= 0.9' % drugset_name),
                        ]

        return result

    def get_uniprots_from_drugset(self, drugset_name, threshold, mapping):
        wsa_ids = self.get_wsa_id_set(drugset_name)
        from dtk.prot_map import AgentTargetCache, DpiMapping
        dm = DpiMapping(mapping)
        wsas = self.wsannotation_set.filter(pk__in=wsa_ids)
        atc = AgentTargetCache.atc_for_wsas(wsas, dpi_mapping=dm, dpi_thresh=threshold)
        wsa_id2prot = {
                wsa.id:set([x.uniprot_id
                        for x in atc.raw_info_for_agent(wsa.agent_id)
                        if float(x.evidence) >= float(threshold)
                        ])
                for wsa in wsas
                }
        import operator
        uniprots = reduce(operator.or_,list(wsa_id2prot.values()), set())
        return set(uniprots)

    def get_uniprot_set(self,name):
        if name.startswith('ps'):
            # it's a user-defined protein set
            ps_id = int(name[2:])
            return set(
                    ProtSet.objects.get(
                            ws=self,
                            id=ps_id,
                            ).proteins.values_list('uniprot',flat=True)
                    )
        elif name.startswith('ds_'):
            # Protein set based on a drugset + DPI
            parts = name.split('_')
            thresh = float(parts[1])
            ds_name = '_'.join(parts[2:])
            wsa_ids = self.get_wsa_id_set(ds_name)
            return self.get_uniprots_from_drugset(
                    ds_name, thresh, self.get_dpi_default())
        elif name.startswith('globps_'):
            # Global protein set.
            globps_name = name[len('globps_'):]
            return ProtSet.get_global_protset(globps_name)
        elif name.startswith('autops_'):
            # Automatic protein set.
            parts = name.split('_')
            if parts[1] == 'none':
                return set()
            elif parts[1] == 'wsunwanted':
                return (self.get_uniprot_set(self.get_intolerable_ps_default()) |
                        self.get_uniprot_set(self.get_nonnovel_ps_default()))

        raise Exception('unknown uniprot_set name %s' % name)
    # versioned file data
    @classmethod
    def get_versioned_file_choices(self,file_class):
        from dtk.s3_cache import S3Bucket
        from dtk.files import VersionedFileName
        s3b = S3Bucket(file_class)
        return VersionedFileName.get_choices(
                file_class=file_class,
                paths=s3b.list(cache_ok=True),
                )
    def get_versioned_file_defaults(self):
        '''return {file_class:vdefault,...}'''
        return VersionDefault.get_defaults(self.id)
    # Prescreens
    def get_prescreen_choices(self):
        from dtk.text import limit
        return [
                (x.id,limit(x.marked_because()))
                for x in Prescreen.objects.filter(ws=self).order_by('-id')
                ]
    # clinical event datasets
    def get_cds_choices(self):
        # get choices for this workspace
        d = PathHelper.cfg('clinical_event_datasets')
        try:
            choices = d[self.id]
        except KeyError:
            choices = d['default']
        def rewrite(item):
            parts = item.split('.')
            if len(parts) == 2 and parts[1] == 'v':
                from browse.default_settings import Defaultable
                # supply default version for workspace
                value = Defaultable.lookup(parts[0]).value(ws=self)
                item = parts[0]+'.'+value
            return (item,item)
        return [ rewrite(x) for x in choices ]
    def get_cds_default(self):
        return self.get_cds_choices()[0][0]
    # combo therapies
    def get_combo_therapy_choices(self):
        result = []
        dirpath = self.ws_dir() + '/combo'
        try:
            for n in os.listdir(dirpath):
                parts = n.split('.')
                if len(parts) != 3 or parts[0] != 'fixed' or parts[2] != 'tsv':
                    continue
                d = self.get_combo_therapy_data(parts[1])
                result.append( (parts[1],d['name']) )
        except OSError:
            pass
        return result
    def get_combo_therapy_data(self,code):
        result = {}
        fn = self.ws_dir() + '/combo/fixed.%s.tsv' % code
        with open(fn) as f:
            for line in f:
                k,v = line.strip('\n').split('\t')
                result[k] = v
        return result
    def get_canonical_target_cache(self,agent_ids):
        '''Return an AgentTargetCache object for the specified agents.

        The cache will be built based on the default dpi options
        for this workspace, and so will provide a stable list of
        targets for use in the target portion of drug review.
        '''
        from dtk.prot_map import DpiMapping, AgentTargetCache
        return AgentTargetCache(
                mapping = DpiMapping(self.get_dpi_default()),
                agent_ids = agent_ids,
                dpi_thresh = self.get_dpi_thresh_default(),
                )
    # disease name in various ontologies
    def get_disease_aliases(self):
        pattern,detail=self.get_disease_default('WebSearch',return_detail=True)
        if detail and pattern:
            return pattern.split('|')
        results = set()
        import re
        for dd in DiseaseDefault.objects.filter(ws=self):
            for code in dd.value.split('|'):
                if not code:
                    # don't allow null codes (e.g. DisGeNet disable)
                    continue
                if re.search(r'\d\d',code):
                    # assume anything with 2 consecutive digits is a key
                    continue
                results.add(code.lower())
        if results:
            if len(results) == 1:
                return results.pop()
            return results
        return self.name
    def get_disease_default(self,vocab='',return_detail=False):
        try:
            dd=DiseaseDefault.objects.get(ws=self,vocab=vocab)
            result = (dd.value,dd)
        except DiseaseDefault.DoesNotExist:
            result = (self.name.lower(),None)
        if return_detail:
            return result
        return result[0]
    def set_disease_default(self,vocab,pattern,user):
        dd,new = DiseaseDefault.objects.get_or_create(
                ws=self,
                vocab=vocab,
                defaults={'value':pattern,'user':user}
                )
        if new:
            return
        # update existing record
        dd.value = pattern
        dd.user = user
        dd.save()
    # DPI settings
    def get_dpi_default(self):
        from browse.default_settings import DpiDataset
        return DpiDataset.value(self)
    def get_dpi_thresh_default(self):
        from browse.default_settings import DpiThreshold
        return DpiThreshold.value(self)
    def get_dpi_version(self):
        from dtk.prot_map import DpiMapping
        dm = DpiMapping(self.get_dpi_default())
        if dm.legacy:
            return None
        return dm.version
    # PPI settings
    def get_ppi_default(self):
        from browse.default_settings import PpiDataset
        return PpiDataset.value(self)
    def get_ppi_thresh_default(self):
        from browse.default_settings import PpiThreshold
        return PpiThreshold.value(self)
    def get_intolerable_ps_default(self):
        from browse.default_settings import IntolerableUniprotsSet
        return IntolerableUniprotsSet.value(self)
    def get_nonnovel_ps_default(self):
        from browse.default_settings import DiseaseNonNovelUniprotsSet
        return DiseaseNonNovelUniprotsSet.value(self)
    # previously-built feature matrices
    def get_feature_matrix_choices(self,exclude=set()):
        # This works as-is, and lets each FM-producing plugin supply
        # its own list of choices, but still depends on having a
        # pre-defined list of FM-producing plugins
        from runner.process_info import JobInfo
        jobnames = set()
        for plugin in set(['fvs','ml', 'fdf'])-exclude:
            uji = JobInfo.get_unbound(plugin)
            jobnames |= set(uji.get_jobnames(self))
        from runner.models import Process
        qs=Process.objects.filter(
                name__in=jobnames,
                status=Process.status_vals.SUCCEEDED,
                ).order_by('-id')
        result = []
        for job in qs:
            bji = JobInfo.get_bound(self,job)
            result += bji.get_feature_matrix_choices()
        return result or [(None,"No Feature Matrix in this workspace")]
    def get_feature_matrix(self,code):
        from runner.process_info import JobInfo
        job_id = JobInfo.extract_feature_matrix_job_id(code)
        bji = JobInfo.get_bound(self,job_id)
        bji.fetch_lts_data()
        return bji.get_feature_matrix(code)
    # tissue sets
    def get_tissue_sets(self):
        if hasattr(self,'_ts_cache'):
            return self._ts_cache
        qs = self.tissueset_set.all().order_by('id')
        if not qs.count():
            # this is a new workspace; create default tissue sets
            for name in ('default','miRNA'):
                ts = TissueSet.objects.create(
                        ws=self,
                        name=name,
                        miRNA=(name == 'miRNA'),
                        )
                logger.info("creating %s tissue set %d for ws %d"
                        ,name
                        ,ts.id
                        ,self.id
                        )
            qs = self.tissueset_set.all().order_by('id')
        self._ts_cache=qs
        return qs
    def invalidate_tissue_set_cache(self):
        if hasattr(self,'_ts_cache'):
            del(self._ts_cache)
    def get_tissue_set_choices(self):
        return [(x.id,x.name) for x in self.get_tissue_sets()]
    # previous successful runs of a given plugin
    def get_prev_job_choices(self,plugin):
        from runner.process_info import JobInfo
        ubi = JobInfo.get_unbound(plugin)
        names = ubi.get_jobnames(self)
        from dtk.text import fmt_time
        from runner.models import Process
        return [
                (p.id,'%s %d %s@%s'%(
                        p.role or p.job_type(),
                        p.id,
                        p.user,
                        fmt_time(p.completed),
                        ))
                for p in Process.objects.filter(
                        name__in=names,
                        status=Process.status_vals.SUCCEEDED,
                        ).order_by('-id')
                ]
    # GWAS datasets
    def get_gwas_dataset_qs(self):
        # this is like the built-in manager gwasdataset_set except
        # that it excludes rejected datasets
        return GwasDataset.objects.filter(ws=self,rejected=False)
    def get_gwas_dataset_choices(self):
        return [
                (x.id,"%s(%d)"%(x.phenotype.replace("_"," "),x.id))
                for x in self.get_gwas_dataset_qs()
                ]
    # gene expression datasets
    def imported_geo_ids(self):
        qs = Tissue.objects.filter(ws=self)
        result = []
        for val in qs.values_list('geoID', flat=True):
            if ',' in val:
                result += val.split(',')
            elif ':' in val:
                result.append(val.split(':')[0])
            else:
                result.append(val)
        return set(result)
    # selection rounds
    def elections(self):
        return Election.objects.filter(ws=self).order_by('-id')
    ##########
    # bulk access to drug properties
    ##########
    # The following methods facilitate creating external data structures
    # that hold information on multiple drugs. One of the main motivations
    # for this was that access to agent attributes from a WsAnnotation
    # didn't scale well if lots of drugs were being processed. With the
    # cleanup in PLAT-2612 and the new prefetch_agent_attributes method,
    # building external lookup structures may be less necessary.
    def get_agent2wsa_map(self):
        return dict(self.wsannotation_set.values_list('agent_id','id'))
    def get_default_wsas(self, dpi=None):
        from dtk.prot_map import DpiMapping
        from browse.default_settings import DpiDataset
        from browse.models import WsAnnotation
        dpi = DpiMapping(dpi or DpiDataset.value(ws=self))

        if dpi.get_dpi_type() != 'moa': 
            # This is the behavior we've typically had, where all imported WSAs are included.
            ws_wsas = WsAnnotation.objects.filter(ws=self, agent__removed=False)
        else:
            # For moa-style workflows, we want to subset to those that have entries in the dpi file.
            keys_with_dpi = set(dpi.get_file_dpi_keys())
            key_wsa_pairs = dpi.get_key_wsa_pairs(ws=self)
            wsas_with_dpi = {x[1] for x in key_wsa_pairs if x[0] in keys_with_dpi}
            ws_wsas = WsAnnotation.objects.filter(ws=self, agent__removed=False, pk__in=wsas_with_dpi)
        return ws_wsas
    def wsa_prop_pairs(self,prop,invalid=False):
        '''Return list of (wsa_id,value) tuples for a drug property.

        The returned list is suitable for initializing a dtk.data MultiMap.
        For a single-valued property, it's also suitable for initializing
        a dict. This can speed up bulk operations on large numbers of drugs
        in a workspace, without using wsa.agent.property access (which can
        cause a separate db query for each drug).

        By default, invalid wsa_ids are not included. Passing invalid=True
        returns only invalid ids. Passing invalid=None returns both.
        '''
        if isinstance(prop,str):
            prop = Prop.get(name=prop)
        cs = connection.cursor()
        sql = '''select wsa.id,value
                from browse_wsannotation as wsa,''' \
                +prop.value_table_name()+ \
                ''' as attr
                where wsa.agent_id = attr.drug_id
                and attr.prop_id = %s
                and wsa.ws_id = %s
                '''
        if invalid is not None:
            sql += ' and wsa.invalid=%d '% (1 if invalid else 0)
        cs.execute(sql,[prop.id,self.id])
        return cs.fetchall()
    def _build_wsa2name_map(self):
        from dtk.data import MultiMap
        mm = MultiMap(self.wsa_prop_pairs(Prop.NAME))
        mm.update(MultiMap(self.wsa_prop_pairs(Prop.OVERRIDE_NAME)))
        mm.update(MultiMap([
                (wsa,name+' (removed)')
                for wsa,name in self.wsa_prop_pairs(Prop.NAME,True)
                ]))
        mm.update(MultiMap([
                (wsa,name+' (removed)')
                for wsa,name in self.wsa_prop_pairs(Prop.OVERRIDE_NAME,True)
                ]))
        return dict(MultiMap.flatten(mm.fwd_map()))
    def get_wsa2name_map(self):
        from dtk.cache import Cacher
        cacher = Cacher('browse.models.Workspace.wsa2name')
        cache_key = cacher.signature((int(self.id),))
        return cacher.check(
                cache_key,
                self._build_wsa2name_map,
                )
    def clear_wsa2name_cache(self):
        from dtk.cache import Cacher
        cacher = Cacher('browse.models.Workspace.wsa2name')
        cache_key = cacher.signature((int(self.id),))
        cacher.delete_many([cache_key])
    @classmethod
    def clear_wsa2name_cache_by_agent(cls,agent_id):
        qs1=WsAnnotation.objects.filter(agent_id=agent_id)
        ws_id_list=qs1.values_list('ws_id',flat=True)
        from dtk.cache import Cacher
        cacher = Cacher('browse.models.Workspace.wsa2name')
        key_list = [
                cacher.signature((int(ws_id),))
                for ws_id in ws_id_list
                ]
        cacher.delete_many(key_list)
    ##########
    # bulk operations
    ##########
    def import_collection(self,collection,user):
        # The code below doesn't remove anything from the collection due to
        # it becoming blocked because of updated clustering. To achieve that
        # the import should be preceded with a call to clear_unused_drugs().
        # This happens in the col2 view, so that ALL collections can be
        # cleared before ANY collections are imported.
        #
        # Note that, since no WSA with meaningful history is ever invalidated,
        # it's possible for a workspace to have 2 WSAs in the same cluster.
        # This is now visible in the UI as a set of 'cluster mate' web links
        # on the annotate page.
        #
        # The duma collection is no longer handled specially. A drug clustered
        # with a duma drug will remain in the workspace if it is already
        # imported. But, it will inherit the DPI of the duma drug, since
        # that is the one prefered by dpimerge.
        #
        from drugs.models import UploadAudit
        logger.info("Importing collection %s", collection.name)
        coll_ver,ts,ok = UploadAudit.collection_status(collection.name)
        assert ok # don't import from damaged collection
        # we get the 'real' wci below, inside the transaction, but we need
        # one out here just to extract the cluster version
        tmp_wci = WsCollectionInfo(self.id)
        from wsmgr.models import ImportAudit
        ia = ImportAudit.objects.create(
                ws=self,
                collection=collection,
                operation='import',
                user=user,
                clust_ver=tmp_wci.version,
                coll_ver=coll_ver,
                )
        with transaction.atomic():
            # XXX If you import, say, chembl.full and then delete it and import
            # XXX chembl.adme, it would be possible for the code to re-use the
            # XXX existing invalidated WSA records (switching the agent id in
            # XXX the record to the agent id for the same chembl key in the new
            # XXX collection). This would tie together scores for the same
            # XXX substance from before and after the switch. It currently isn't
            # XXX done due to added complexity.
            all_ws_coll_qs = WsAnnotation.all_objects.filter(
                    ws=self,
                    agent__collection=collection,
                    )
            logger.info("%s ws %d has %d WSAs plus %d invalid %s",
                    collection.name,
                    self.id,
                    all_ws_coll_qs.filter(invalid=False).count(),
                    all_ws_coll_qs.filter(invalid=True).count(),
                    'at start of import',
                    )
            wci = WsCollectionInfo(self.id)
            ckey2agents = wci.clust_key_to_coll_agents(collection)
            blocked_agents = wci.blocked_agents(collection,ckey2agents)
            # unmark anything marked as invalid
            qs = WsAnnotation.all_objects.filter(
                    ws=self,
                    invalid=True,
                    agent__removed=False,
                    agent__collection=collection,
                    ).exclude(
                    agent_id__in=blocked_agents,
                    ).exclude(
                    # don't allow in extra copies of duplicated agents
                    pk__in=wci.blocked_wsa_ids
                    )
            qs.update(invalid=False)
            logger.info("%s ws %d has %d WSAs plus %d invalid %s",
                    collection.name,
                    self.id,
                    all_ws_coll_qs.filter(invalid=False).count(),
                    all_ws_coll_qs.filter(invalid=True).count(),
                    f'after unmark',
                    )
            # now bulk-import anything that isn't blocked
            # - reload wci to pick up agents unmarked above
            # - retain ckey2agents, since that's ws-independent
            wci = WsCollectionInfo(self.id)
            blocked_agents = wci.blocked_agents(collection,ckey2agents)
            # exclude anything already loaded
            skip = blocked_agents | wci.ws_agent_id_set
            total = 0
            skipped = 0
            al=[]
            from drugs.models import Drug
            for agent_id in Drug.objects.filter(
                            collection=collection,
                            removed=False,
                            ).values_list('id',flat=True):
                if agent_id in skip:
                    skipped += 1
                    continue
                a = WsAnnotation(ws=self,agent_id=agent_id)
                al.append(a)
                total += 1
                if len(al) >= 1000:
                    WsAnnotation.objects.bulk_create(al)
                    logger.info(
                        "%s ws %d import_collection pre-loading %d of %d drugs"
                        ,collection.name,self.id,len(al),total
                        )
                    al = []
            if al:
                WsAnnotation.objects.bulk_create(al)
            logger.info(
                    "%s ws %d import_collection complete"
                            "; added %d drugs, skipped %d"
                            "; %d WSAs plus %d invalid",
                    collection.name,self.id,total,skipped,
                    all_ws_coll_qs.filter(invalid=False).count(),
                    all_ws_coll_qs.filter(invalid=True).count(),
                    )

        ia.succeeded = True
        ia.save()
        # Clear the wsa2name cache now that we've finished reimporting.
        # Note that it is important that we do this after the transaction
        # has ended, to be sure the cache doesn't get rebuilt with the old
        # data.
        self.clear_wsa2name_cache()

    def import_single_molecule(self, agent_id, user):
        """Allows for one-off molecule imports.  Don't use this for bulk imports."""

        # Check that we don't already have a WSA for this agent or one of its cluster mates.
        version = self.get_dpi_version()
        agent_ids = Drug.matched_id_mm([agent_id], version).fwd_map()[agent_id]
        wsas = WsAnnotation.objects.filter(ws=self, agent__in=agent_ids)


        if wsas:
            raise Exception(f"WSA for {agent_id} already exists in {self}")

        # Check if we have an inactive WSA for this agent.
        wsas = WsAnnotation.all_objects.filter(ws=self, agent__in=agent_ids)

        if wsas:
            logger.info(f"Reactivating WSA for agent {agent_id} in {self}")
            assert wsas[0].invalid, "This should be invalid"
            wsas[0].invalid = False
            wsas[0].save()
            wsa = wsas[0]
        else:
            # No existing WSA, let's make a new one.
            wsa = WsAnnotation(ws=self, agent_id=agent_id)
            logger.info(f"Creating WSA for agent {agent_id} in {self}")
            wsa.save()

        from wsmgr.models import ImportAudit
        from drugs.models import UploadAudit
        parts = UploadAudit.collection_status(wsa.agent.collection.name)
        if parts is None:
            # Unversioned case, shouldn't really happen outside of tests.
            coll_ver = None
        else:
            coll_ver,ts,ok = parts
        ia = ImportAudit.objects.create(
                ws=self,
                collection=wsa.agent.collection,
                operation='single_import',
                user=user,
                clust_ver=version,
                coll_ver=coll_ver,
                wsa=wsa,
                )
        ia.succeeded = True
        ia.save()

        return wsa

class VersionDefault(models.Model):
    ws = models.ForeignKey(Workspace,null=True, on_delete=models.CASCADE)
    file_class = models.CharField(max_length=100)
    choice = models.CharField(max_length=100)

    class Meta:
        unique_together=[['ws', 'file_class']]
    @classmethod
    def _get_as_dict(cls,ws_id):
        # Filter to only known keys is mostly useful for development, where
        # you might end up with new types in the DB that don't exist on other
        # branches.
        known_keys = set(cls.default_global_defaults().keys())
        out = dict(x for x in cls.objects.filter(
                ws_id=ws_id,
                ).values_list('file_class','choice')
                if x[0] in known_keys)
        
        from browse.default_settings import Defaultable
        for name, subcls in Defaultable.get_subclasses(sort_output=False):
            if subcls.form_type == bool:
                out[name] = out[name] == 'True'
        return out
        
    @classmethod
    def set_defaults(cls,ws_id,change_list,user):
        import django
        audit_ts = django.utils.timezone.now()
        for file_class,choice in change_list:
            vd,new = cls.objects.get_or_create(
                    ws_id = ws_id,
                    file_class = file_class,
                    )
            vd.choice = choice
            vd.save()
            vda = VersionDefaultAudit(
                    ws_id = ws_id,
                    file_class = file_class,
                    choice = choice,
                    user = user,
                    timestamp = audit_ts,
                    )
            vda.save()

    @classmethod
    def ordering(cls):
        from browse.default_settings import Defaultable
        ordered_names = Defaultable.get_subclasses(sort_output=False)
        return {x[0]:idx for idx, x in enumerate(ordered_names)}

    @classmethod
    def default_global_defaults(cls):
        from browse.default_settings import Defaultable
        Types = [x[1] for x in Defaultable.get_subclasses() if x[0] != 'VDefaultable']
        return {Type.name(): Type.default_global_default() for Type in Types}

    @classmethod
    def get_defaults(cls,ws_id):
        from browse.default_settings import Defaultable
        # first check for missing records
        # - fill in missing ws-specific defaults from global defaults
        # - fill in missing global defaults from hard-wired list
        template = cls.get_defaults(None) if ws_id else cls.default_global_defaults()
        ws_qs=cls.objects.filter(ws_id=ws_id)
        keys = set(ws_qs.values_list('file_class',flat=True))
        missing = []
        for key in template:
            if key not in keys:
                if key not in Defaultable.get_all_names():
                    # This usually happens because a type got removed,
                    # which can often happen during development.  Just ignore.
                    continue
                def_cls = Defaultable.lookup(key)
                val = template[key]
                if ws_id and hasattr(def_cls, 'workspace_default'):
                    ws = Workspace.objects.get(pk=ws_id)
                    val = def_cls.workspace_default(ws)
                missing.append((key,val))
        if missing:
            cls.set_defaults(ws_id,missing,'auto_populated')
        # now return all records for selected ws as dict
        return cls._get_as_dict(ws_id=ws_id)

class VersionDefaultAudit(models.Model):
    ws = models.ForeignKey(Workspace,null=True, on_delete=models.CASCADE)
    file_class = models.CharField(max_length=100)
    choice = models.CharField(max_length=100)
    user = models.CharField(max_length=50)
    timestamp = models.DateTimeField()

class DiseaseDefault(models.Model):
    class Meta:
        unique_together=[['ws', 'vocab']]
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    vocab = models.CharField(max_length=100,blank=True,default="")
    value = models.TextField(blank=True,default="")
    user = models.CharField(max_length=50,blank=True,default="")
    timestamp = models.DateTimeField(auto_now=True,null=True)

class Demerit(models.Model):
    stage_vals = Enum([], [
                ('REVIEW',),
                ('PRECLINICAL',),
                ])

    desc = models.CharField(max_length=50)
    active = models.BooleanField(default=True)

    stage = models.IntegerField(
                choices=stage_vals.choices(),
                default=stage_vals.REVIEW,
                )

    def __str__(self):
        return self.desc + '' if self.active else ' (inactive)'

class KtAudit(models.Model):
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    user = models.CharField(max_length=50,blank=True,default="")
    timestamp = models.DateTimeField(auto_now_add=True,null=True)
    key = models.CharField(max_length=250,blank=True,default="")
    source = models.CharField(max_length=50,blank=True,default="")
    new_matches = models.IntegerField()
    old_matches = models.IntegerField()
    unmatched = models.IntegerField()

# Use an 'invalid' flag and custom manager rather than record deletion,
# so a wsa_id can't get reused, and an old score mis-applied.
class WsAnnotationManager(models.Manager):
    def get_queryset(self):
        return super(
                WsAnnotationManager,
                self,
                ).get_queryset().filter(invalid=False)

class WsAnnotation(models.Model):
    class Meta:
        index_together = [
                # We almost always query with the ws as a filter, so it
                # is way faster to include it as part of the other indexes.
                ['ws', 'indication'],
                ['ws', 'invalid'],
                ]
    objects = WsAnnotationManager()
    all_objects = models.Manager()
    invalid = models.BooleanField(default=False)
    def invalidate(self):
        self.invalid = True
        self.save()

    @classmethod
    def discovery_order_index(cls, value):
        try:
            return cls.discovery_order.index(value)
        except ValueError:
            # Not in the list, it is 'worse' than any values in the list.
            return -1

    @classmethod
    def screening_order_index(cls, value):
        try:
            return cls.screened_inds.index(value)
        except ValueError:
            # Not in the list, it is 'worse' than any values in the list.
            return -1

    def note_info(self,attr):
        if attr == 'study_note':
            return {
                'label':"Workspace note on %s for %s" %(
                        self.agent.canonical,
                        self.ws.name,
                        ),
                }
        raise Exception("bad note attr '%s'" % attr)
    @classmethod
    def grouped_choices(cls):
        iv = cls.indication_vals
        deprecated = [
                iv.CANDIDATE_PATENTED,
                iv.PATENT_PREP,
                iv.TRIALED_TREATMENT,
                ]
        treatments = [x for x in cls.ordered_kt_indications() if x not in deprecated]
        inds = [
                ("Inactive", [iv.INACTIVE_PREDICTION]),
                ("Predictions", [
                        iv.INITIAL_PREDICTION,
                        iv.REVIEWED_PREDICTION,
                        iv.HIT,
                    ]),
                ("Pre-Clinical", [
                        iv.IN_VITRO_1,
                        iv.IN_VITRO_2,
                        iv.IN_VIVO_1,
                        iv.IN_VIVO_2,
                        iv.LEAD_OP,
                    ]),
                ("Existing Treatments/Causes", treatments),
                ("Deprecated", deprecated),
            ]
        all_choices = iv.choices()

        choices = []
        included = set()
        for entry in inds:
            if isinstance(entry, int):
                choices.append(all_choices[entry])
                included.add(entry)
            else:
                group_choices = [all_choices[x] for x in entry[1]]
                choice_group = (entry[0], group_choices)
                choices.append(choice_group)
                included.update(entry[1])

        other_choices = []
        for choice in all_choices:
            if choice[0] not in included:
                other_choices.append(choice)

        if other_choices:
            choices.append(("Other", other_choices))

        return choices

    indication_vals = Enum([],
            [ ('UNCLASSIFIED',)
            , ('FDA_TREATMENT','FDA Approved Treatment')
            , ('FDA_CAUSE','FDA Documented Cause')
            , ('KNOWN_TREATMENT','Clinically used treatment')
            , ('KNOWN_CAUSE','Clinically indicated cause')
            , ('INITIAL_PREDICTION',)
            , ('CANDIDATE_CAUSE',)
            , ('EXP_TREATMENT','Researched as treatment')
            , ('EXP_CAUSE','Researched as cause')
            , ('INACTIVE_PREDICTION',)
            , ('CANDIDATE_PATENTED','Patent submitted')
            , ('PATENT_PREP','Preparing patent')
            , ('TRIALED_TREATMENT','Clinically investigated treatment')
            , ('HYPOTH_TREATMENT','Hypothesized treatment')
            , ('REVIEWED_PREDICTION',)
            , ('HIT',)
            , ('TRIALED1_TREATMENT','Phase 1 treatment')
            , ('TRIALED2_TREATMENT','Phase 2 treatment')
            , ('TRIALED3_TREATMENT','Phase 3 treatment')
            , ('IN_VITRO_1',)
            , ('IN_VITRO_2',)
            , ('IN_VIVO_1',)
            , ('IN_VIVO_2',)
            , ('REVIEWED_AS_MOLECULE',)
            , ('LEAD_OP',)
            ])
    # NOTE: When adding a new indication to the list above:
    # - it must be added at the end, to not break the integer offsets stored
    #   in the database
    # - it must be added to any relevant lists; these include:
    #   - the indication group logic below
    #   - the code in rvw/views that selects which indications to display
    #     for various flavors of the review page
    #     XXX this page should arguably leverage list logic here, to avoid
    #     XXX coupling and duplication
    #   - the grouped_choices method above
    # XXX It might be a nice idea to have a devoplers QC page that displays
    # XXX a table with an indication per row, with columns of groups, and
    # XXX checkboxes showing what groups each indication belongs to.
    screened_inds = [
        indication_vals.IN_VITRO_1,
        indication_vals.IN_VITRO_2,
        indication_vals.IN_VIVO_1,
        indication_vals.IN_VIVO_2,
        indication_vals.LEAD_OP,
        ]
    hit_inds = [
                indication_vals.HIT, # Passed final review
               ]
    selected_inds = screened_inds + hit_inds
    reviewed_inds = [
                indication_vals.INITIAL_PREDICTION, # Passed prescreen
                indication_vals.REVIEWED_PREDICTION, # Passed prelim review
                ]

    discovery_order = [
                indication_vals.UNCLASSIFIED,
            ] + reviewed_inds + hit_inds + screened_inds

# this name isn't actually correct any more, but close enough
    _selected_groups=[
            ('screened','Screened', screened_inds),
            ('selected','Selected', hit_inds),
            ('reviewed','Reviewed', reviewed_inds),
            ]
    reviewed_or_beyond = [x[0] for x in _selected_groups]
    _indication_groups=[
            ('kts','Known Treatments',[
                    indication_vals.FDA_TREATMENT,
                    indication_vals.KNOWN_TREATMENT,
                    ]),
            ('p3ts','Phase 3 Trialed or better',[
                    indication_vals.TRIALED3_TREATMENT,
                    ]),
            ('p2ts','Phase 2 Trialed or better',[
                    indication_vals.TRIALED2_TREATMENT,
                    ]),
            ('tts','Trialed or better',[
                    indication_vals.TRIALED1_TREATMENT,
                    indication_vals.TRIALED_TREATMENT, # legacy setting
                    ]),
            ('ets','Experimental or better',[
                    indication_vals.EXP_TREATMENT,
                    ]),
            ('hts','Hypothesized or better',[
                    indication_vals.HYPOTH_TREATMENT,
                    ]),
            ('related','Related to disease',[
                    indication_vals.FDA_CAUSE,
                    indication_vals.KNOWN_CAUSE,
                    indication_vals.EXP_CAUSE,
                    ]),
            ]
    def max_discovery_indication(self):
        # Intentionally not using values/values_list here, because that
        # interferes with prefetch_related.
        # If you're doing a single wsa, then perf probably doesn't matter,
        # and if you're doing multiple, you should do a prefetch_related.
        prev_indications = self.get_prev_indications()
        discovery_idxs = [self.discovery_order_index(x) for x in prev_indications]
        if len(discovery_idxs) == 0:
            # Everything is at least 'unclassified'.
            return self.discovery_order[0]

        max_idx = max(discovery_idxs)
        if max_idx == -1:
            return self.discovery_order[0]

        return self.discovery_order[max_idx]

    def count_screening_indications(self):
        prev_indications = self.get_prev_indications()
        screened_idxs = [self.screening_order_index(x) for x in prev_indications]
        cnt = sum([0]+[1 for x in screened_idxs if x > -1])
        return cnt

    def get_prev_indications(self):
        prev_indications = [x.indication for x in self.dispositionaudit_set.all()]
        prev_indications += [self.indication]
        return prev_indications

    @classmethod
    def ordered_selection_indications(cls):
        initial = [cls.indication_vals.INACTIVE_PREDICTION
                  ] + cls.reviewed_inds + cls.hit_inds + cls.screened_inds
        initial.reverse()
        return initial
    @classmethod
    def ordered_kt_indications(cls):
        result = []
        for code,label,members in cls._indication_groups:
            result += members
        return result
    @classmethod
    def all_groups(cls):
        options = cls.indication_groups()
        options += cls.retro_groups()
        return options

    @classmethod
    def retro_groups(cls):
        return [
                ('retro_reviewed', "Ever Reviewed"),
                ('retro_final_rev', "Ever Final Review"),
                ('retro_selected', "Ever Selected"),
                ('retro_screened', "Ever Screened"),
                ('retro_failed_screen', "Failed Screening"),
                ('retro_passed_first_scrn', "Passed First Screen"),
                ('retro_failed_first_scrn', "Failed First Screen"),
               ]

    @classmethod
    def indication_groups(cls):
        return [
                (x[0],x[1]) for x in cls._indication_groups
                ]+[
                ('classified','Any classification'),
                ('screened','Screening'),
                ('selected','Selected and Screening'),
                ('reviewed','Reviewed to Screening'),
                ('just_reviewed','Only Reviewed'),
                ]
    @classmethod
    def indication_group_members(cls,code):
        s = set()
        if code in cls.reviewed_or_beyond:
            groups = cls._selected_groups
        elif code == 'just_reviewed':
            groups = [('just_reviewed','Only Reviewed', cls.reviewed_inds)]
        else:
            groups = cls._indication_groups
        for grp in groups:
            s |= set(grp[2])
            if grp[0] == code:
                return s
        if code == 'classified':
            return set([x[0] for x in cls.indication_vals.choices() if x[0]])
        raise ValueError("unknown group code '%s'" % code)
    @classmethod
    def prefetch_agent_attributes(cls,qs,prop_types=None,use_id_prefetch=False,prop_names=None):
        '''Return a qs like the input qs, but with prefetched agent attributes.

        qs is queryset returning WsAnnotation objects.
        prop_type is an iterable returning the property types to fetch
        (default is TAG)
        use_id_prefetch invokes a modified algorithm, as follows:
        - normally, to prefetch N property types, modified versions of the
          passed-in qs are executed N+1 times; this isn't normally an issue
          since most queries are fast
        - if the passed-in query is slow, and if it returns a limited number
          of WSAs, a reasonable alternative is to execute it once and get
          back a list of WSA ids, and then do the N+1 main queries as an
          id__in=list filter; this is the option invoked by use_id_prefetch
        '''
        if use_id_prefetch:
            wsa_ids = list(qs.values_list('id',flat=True))
            qs = WsAnnotation.objects.filter(id__in=wsa_ids)
        from drugs.models import AttributeCacheLoader,Prop
        if not prop_types:
            prop_types = [Prop.prop_types.TAG]
        qs = qs.select_related('agent')
        for prop_type in prop_types:
            acl = AttributeCacheLoader(qs,'agent',prop_type,prop_names)
            for wsa in qs:
                acl.load_cache(wsa.agent)
        return qs
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    agent = models.ForeignKey(Drug,null=True,blank=True, on_delete=models.CASCADE)
    indication = models.IntegerField(choices=indication_vals.choices()
                ,default=indication_vals.UNCLASSIFIED
                )
    indication_href = models.CharField(max_length=1024,default="",blank=True)
    marked_by = models.CharField(max_length=50,blank=True,default="")
    marked_on = models.DateTimeField(null=True,blank=True)
    marked_because = models.CharField(max_length=1024,blank=True,default="")
    marked_prescreen = models.ForeignKey("Prescreen",null=True, on_delete=models.CASCADE)
    indication_protection = models.IntegerField(
                choices=Prop.protection_vals.choices(),
                default=Prop.protection_vals.NOT_SCREENED,
                )
    study_note = models.ForeignKey(Note,null=True,blank=True, on_delete=models.CASCADE)
    doc_href = models.CharField(max_length=1024,default="",blank=True)
    demerit_list = models.CharField(max_length=1024,default="",blank=True)
    review_code = models.CharField(max_length=50,blank=True,default="")
    replacements = models.ManyToManyField('WsAnnotation', related_name='replacement_for')

    txr_id = models.CharField(max_length=50,blank=True,default="")

    def get_assay_info(self):
        return self.agent.get_assay_info(self.ws.get_dpi_version())

    def get_raw_assay_info(self):
        return self.agent.get_raw_assay_info(self.ws.get_dpi_version())
    @classmethod
    def parse_demerits(cls, demerits_value):
        return set([
                int(x)
                for x in demerits_value.split(',')
                if x != ''
                ])
    def demerits(self):
        return self.parse_demerits(self.demerit_list)
    def demerit_text(self):
        s = self.demerits()
        if not s:
            return ''
        qs=Demerit.objects.filter(pk__in=s)
        return '; '.join(sorted(qs.values_list('desc',flat=True)))
    def update_indication(self,indication
                        ,demerits=None
                        ,user=""
                        ,detail=""
                        ,href=None
                        ,from_prescreen=None
                        ,allow_deprecated=False
                        ):
        indication=int(indication)
        # update the indication value while enforcing business rules
        # - if changing to INITIAL_PREDICTION, record who/when/why
        #   (XXX this may go away with the audit table, but it's currently
        #   displayed in various places)
        # - if PATENT_PREP, no demerits should be set
        # - if INACTIVE_PREDICTION, at least one demerit should be set
        cur_demerits = self.demerits()
        if demerits is None:
            demerits = cur_demerits
        else:
            demerits = set(demerits)
        if href is None:
            href = self.indication_href
        enum = self.indication_vals
        # these checks are done outside the update conditionals so they
        # pick up either demerit or indication changes (or even just
        # attempts to re-save an existing non-conforming record)
        if indication == enum.INACTIVE_PREDICTION:
            if not demerits:
                raise ValueError('At least one demerit must be set')
        if indication == enum.PATENT_PREP:
            if demerits:
                raise ValueError('No demerit can be set')
        if indication in self.indication_group_members('related'):
            if not href:
                raise ValueError('href required for treatments/causes')
        import re
        if href and not re.match(r'(http(s)?:)?//',href):
            raise ValueError('href must be a full URL')
        update = False

        if from_prescreen and from_prescreen != self.marked_prescreen:
            # TODO: Are there cases where we don't bother setting from_prescreen
            # but want to update other things?  I think so, so we ignore it...
            update = True
            self.marked_by = user
            self.marked_because = detail
            self.marked_on = timezone.now()
            self.marked_prescreen = from_prescreen

        if indication != self.indication:
            if indication == enum.TRIALED_TREATMENT and not allow_deprecated:
                raise ValueError('deprecated; use phase 1,2,3')
            if indication == enum.INITIAL_PREDICTION:
                self.marked_by = user
                self.marked_because = detail
                self.marked_on = timezone.now()
                if from_prescreen:
                    # This prevents you from removing a marked_prescreen, it can
                    # only be changed.  If we want to support removing, can
                    # remove the check, but have fixed problems in past with
                    # forgetting to propagate this.
                    self.marked_prescreen = from_prescreen
            self.indication = indication
            update = True
        if href != self.indication_href:
            self.indication_href = href
            update = True
        if demerits != cur_demerits:
            update = True
            self.demerit_list = ','.join([str(x) for x in demerits])
        if update:
            self.save()
            da = DispositionAudit(
                        wsa = self,
                        user = user,
                        timestamp = timezone.now(),
                        reason = detail,
                        indication = self.indication,
                        indication_href = self.indication_href,
                        demerit_list = self.demerit_list,
                        from_prescreen = from_prescreen
                        )
            da.save()
    def get_all_prscrn_ranks(self):
        from runner.process_info import JobInfo
        from dtk.scores import Ranker
        qs = Prescreen.objects.filter(ws=self.ws.id).order_by('-id')
        data = []
        for ps in qs:
            bji = JobInfo.get_bound(self.ws.id,ps.primary_job_id())
            cat = bji.get_data_catalog()
            ranker = Ranker(cat.get_ordering(ps.primary_code(),True))
            r = ranker.get(self.id)
            eff_bji = JobInfo.get_bound(self.ws.id,ps.eff_jid())
            cat = eff_bji.get_data_catalog()
            ranker = Ranker(cat.get_ordering(ps.eff_code(),True))
            eff_r = ranker.get(self.id)
            data.append([ps.name,
                         ps.id,
                         bji.job,
                         eff_bji.job,
                         ps.created.strftime("%Y-%m-%d %H:%M"),
                         ps.user,
                         r,
                         eff_r,
                        ])
        return data

# TODO we might want to use this approach for the scoreboard on the annotate page
    def get_marked_or_best_eff_jid(self):
        marked = self.get_marked_eff_jid()
        if marked is not None:
            return int(marked)
        jid = self.get_best_eff_jid()
        if jid is not None:
            jid = int(jid)
        return jid

    def get_best_eff_jid(self):
        rnks_data = self.get_all_prscrn_ranks()
        best = (None, None)
        for row in rnks_data:
            if best[0] is None:
                best = (row[3].id, row[7])
                continue
            if best[1] > row[7]:
                best = (row[3].id, row[7])
        return best[0]

    def get_marked_eff_jid(self):
        """Returns the efficacy (usually wzs) jid that marked this WSA."""
        pscr = self.marked_prescreen
        if not pscr:
            # We only set marked_because if we screened in.
            # If we screened it out, it will be in the disposition audit.
            das = DispositionAudit.objects.filter(wsa=self).exclude(reason='')
            if not das:
                return None

            pscr = das[0].from_prescreen

        if not pscr:
            return None

        return pscr.eff_jid()

    def is_moa(self):
        return self.agent.is_moa()

    def get_name(self,demo):
        label = self.agent.canonical
        if demo:
            enum = self.indication_vals
            kts = (
                    enum.FDA_TREATMENT,
                    enum.KNOWN_TREATMENT,
                    )
            if self.indication not in kts:
                label = obfuscate(label)
        if self.invalid:
            label += ' (removed)'
        if self.review_code:
            label = '['+self.review_code+'] '+label
        return label
### This was written as quickly as possible and needs to be firmed up at some point
    def get_surechembl_id(self):
        return self.get_id_from_unichem('surechembl')
    def get_zinc_id(self):
        return self.get_id_from_unichem('zinc')
    def get_id_from_unichem(self,src):
        from dtk.unichem import UniChem
        reference_collections = ['drugbank', 'chembl']
        uc = UniChem()
        new_ids=set()
        vdefaults = self.ws.get_versioned_file_defaults()
        for rc in reference_collections:
            id = getattr(self.agent, rc +'_id', None)
            if not id:
                continue
            d = uc.get_converter_dict(
                    rc,
                    src,
                    vdefaults['unichem'],
                    key_subset=[id],
                    )
            if id in d:
                new_ids.update(set(d[id]))
        return list(new_ids) if new_ids else None

    def get_patent_ids(self):
        patent_details = self.get_patent_details()
        patent_ids = [i[1] for i in patent_details]
        return patent_ids

    def get_patent_details(self):
        from dtk.s3_cache import S3MiscBucket, S3File
        from dtk.files import get_file_records
# I'd rather this wasn't hardcoded
        patent_file = 'patent.surechembl.full.tsv.gz'
        s3f = S3File(S3MiscBucket(), patent_file)
        s3f.fetch()
        patents = []
        surechembl_lst = self.get_surechembl_id()
        gen = get_file_records(s3f.path(),
                               select=(surechembl_lst,0),
                               keep_header = False
                              )
        for item in gen:
### TODO eventually we will expand outside of the US
            if item[1][0:2] != 'US':
                continue
            item[1] = item[1].replace('-','')
            patents.append(item)
        return patents
    ########
    # The next two methods manage review codes; currently these are
    # integers, but are stored as strings because they may at some point
    # get a suffix to document groups of related drugs. At that point,
    # only these two functions should need to change. set_review_code
    # will probably need an additional parameter to indicate the group.
    # review_code_sort_key might return a tuple to sort by group, and
    # then by drug within the group.
    ########
    def set_review_code(self):
        if self.review_code:
            return
        qs = WsAnnotation.objects.filter(ws_id=self.ws_id)
        qs = qs.exclude(review_code='')
        current_codes = qs.values_list('review_code',flat=True)
        next_code = 1
        if current_codes:
            next_code += max([int(x) for x in current_codes])
        self.review_code = next_code
    def review_code_sort_key(self):
        if self.review_code:
            return int(self.review_code)
        # when displaying an old election page, the drugs may not
        # have a review code; in this case sort by name
        return self.agent.canonical
    ########
    def str(self):
        return u"%s[%s]" % (self.agent,self.ws_id)
    def get_study_text(self):
        return Note.get(self,'study_note','')

    @property
    def max_phase(self):
        from drugs.models import Index
        ver = self.ws.get_dpi_version()
        drug_ids = Drug.matched_id_mm([self.agent_id], version=ver).fwd_map()[self.agent_id]
        max_phase = max(Index.objects.filter(
                            drug_id__in=drug_ids,
                            prop__name='max_phase',
                            ).values_list('value', flat=True), default=0)

        if max_phase == 4:
            term = 'Approved'
        elif max_phase == 0:
            term = 'Experimental'
        else:
            term = f'Ph.{max_phase}'
        term = f'{max_phase} ({term})'
        return term

    @classmethod
    def from_agent_ids(cls, agent_ids, ws, version):
        """Returns WSAs in a workspace that are clustered with these agent IDs."""
        from drugs.models import Drug
        from dtk.data import MultiMap
        id_mm = Drug.matched_id_mm(agent_ids, version=version)
        agent_to_src_agent = id_mm.rev_map()
        all_agents = agent_to_src_agent.keys()
        all_wsas = WsAnnotation.objects.filter(ws=ws, agent_id__in=all_agents)

        agent2wsa = []
        for wsa, dst_agent in all_wsas.values_list('id', 'agent_id'):
            for orig_agent in agent_to_src_agent[dst_agent]:
                agent2wsa.append((orig_agent, wsa))
        return MultiMap(agent2wsa)


    @classmethod
    def bulk_max_phase(cls, wsas):
        if not wsas:
            return {}

        from drugs.models import Flag, Index, Drug

        ws = wsas[0].ws
        ver = ws.get_dpi_version()
        agent_ids = wsas.values_list('agent_id', flat=True)
        drug_id_mm = Drug.matched_id_mm(agent_ids, version=ver)

        drug_ids = drug_id_mm.rev_map()
        max_phases = Index.objects.filter(
                            drug_id__in=drug_ids.keys(),
                            prop__name='max_phase',
                            ).values_list('drug_id', 'value')

        approved = Flag.objects.filter(
                            drug_id__in=drug_ids.keys(),
                            prop__name='approved',
                            ).values_list('drug_id', 'value')

        from dtk.data import MultiMap
        drug2phases = MultiMap(max_phases).fwd_map()
        drug2approved = MultiMap(approved).fwd_map()

        from dtk.aact import lookup_trials_by_molecules, phase_name_to_number
        wsa2trials = lookup_trials_by_molecules(wsas).fwd_map()

        out = {}
        for wsa in wsas:
            ids = drug_id_mm.fwd_map()[wsa.agent_id]
            trials = wsa2trials.get(wsa.id, [])
            phases = []
            approved = False
            for drug_id in ids:
                approved |= any(drug2approved.get(drug_id, []))
                phases.extend((x, drug_id) for x in drug2phases.get(drug_id, []))

            max_trial_phase = max(
                    [phase_name_to_number(x.phase) for x in trials],
                    default=0,
                    )

            max_phase = max(phases, default=(0, None))

            overall_max_phase = max(max_trial_phase, max_phase[0])
            if approved:
                overall_max_phase = 4


            out[wsa.id] = {
                # From drugbank, can mean approved in any region.
                "approved": approved,
                # From chembl.
                "max_phase": max_phase[0],
                "max_phase_drugid": max_phase[1],
                # From AACT.
                "trials": trials,
                "trials_max_phase": max_trial_phase,
                # Overall combined.
                "overall_max_phase": overall_max_phase,
            }
        return out


    def indication_link(self):
        text = self.indication_label()
        if self.indication == self.indication_vals.INACTIVE_PREDICTION:
            info = self.demerit_text()
            if info:
                from django.utils.html import format_html_join
                from dtk.html import glyph_icon
                text=format_html_join(' ','{}',[
                        (text,),
                        (glyph_icon('info-sign',hover=info),),
                        ])
        if self.indication_href:
            from dtk.html import link
            return link(text,self.indication_href,new_tab=True)
        else:
            return text
    def indication_label(self):
        return WsAnnotation.indication_vals.get('label',self.indication)
    def ind_prot_label(self):
        return Prop.protection_vals.get('label',self.indication_protection)
    def drug_url(self):
        return Workspace.ws_reverse('moldata:annotate',self.ws_id,self.id)
    def html_url(self):
        from dtk.html import link
        return link(self.agent.canonical, self.drug_url())
    def prescreen_flags(self):
        '''Return prescreen flag HTML, if any.
        '''
        from flagging.models import Flag
        flags = Flag.get_for_wsa(self)
        if not flags:
            return ''
        return Flag.format_flags(flags)
    def agent_strings_by_prop(self):
        l = list(self.agent.tag_set.all())
        l += list(self.agent.blob_set.all())
        l.sort(key=lambda x:x.prop.name)
        return l
    def get_possible_dpi_opts(self):
        # construct a list of all dpi mappings that have an entry for
        # this drug (by checking which keys this drug has)
        prop_names = set(
                self.agent.tag_set.values_list('prop__name',flat=True)
                )
        from dtk.prot_map import DpiMapping
        dpi_opts = DpiMapping.get_possible_mappings(prop_names)
        dpi_opts.sort()
        return dpi_opts
    @transaction.atomic
    def update_agent(self, new_agent, reason):
        WsaAgentAudit.objects.create(
                wsa=self,
                former_agent=self.agent,
                reason=reason,
                )
        self.agent = new_agent
        self.save()



class WsaAgentAudit(models.Model):
    wsa = models.ForeignKey(WsAnnotation, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now=True)
    former_agent = models.ForeignKey(Drug, on_delete=models.CASCADE)
    reason = models.CharField(max_length=1024)

class DispositionAuditManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(ignore=False)

class DispositionAudit(models.Model):
    objects = DispositionAuditManager()
    all_objects = models.Manager()

    wsa = models.ForeignKey(WsAnnotation, on_delete=models.CASCADE)
    user = models.CharField(max_length=50,blank=True,default="")
    # Rather than deleting history, we can mark audit entries as 'ignore',
    # and they won't be used for computing things.
    ignore = models.BooleanField(default=False)
    timestamp = models.DateTimeField(null=True,blank=True)
    reason = models.CharField(max_length=1024,blank=True,default="")
    indication = models.IntegerField()
    indication_href = models.CharField(max_length=1024,default="",blank=True)
    demerit_list = models.CharField(max_length=1024,default="",blank=True)
    from_prescreen = models.ForeignKey("Prescreen",null=True, on_delete=models.CASCADE)

    def indication_label(self):
        return WsAnnotation.indication_vals.get('label',self.indication)
    class Meta:
        # Normally we could just annotate with db_index, but this
        # is an alternative syntax that works with django's migrations.
        index_together = [
                ['indication'],
                ]

class ElectionFlavor:
    def __init__(self,flavor_string):
        enum = WsAnnotation.indication_vals
        self.flavor_string = flavor_string
        if flavor_string == '':
            self.label = 'Single-pass'
            self.input = enum.INITIAL_PREDICTION
            self.filter_dups = False
            self.output = enum.PATENT_PREP
        elif flavor_string == 'pass1':
            self.label = 'Preliminary'
            self.input = enum.INITIAL_PREDICTION
            self.filter_dups = True
            self.output = enum.REVIEWED_PREDICTION
        elif flavor_string == 'pass2':
            self.label = 'Final'
            self.input = enum.REVIEWED_PREDICTION
            self.filter_dups = True
            self.output = enum.HIT
        else:
            self.label = '??? (%s)'%flavor_string
    def top(self):
        enum = WsAnnotation.indication_vals
        return (enum.INACTIVE_PREDICTION, self.output)

class Election(models.Model):
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    due = models.DateTimeField()
    flavor = models.CharField(max_length=50,blank=True,default="")

    @property
    def flavor_info(self):
        return ElectionFlavor(self.flavor)

    def elec_label(self):
        flavor = ElectionFlavor(self.flavor)
        from dtk.text import fmt_time
        result = [flavor.label,fmt_time(self.due)]
        status = self.status()
        if status != 'Done':
            result.insert(0,status)
        return ' '.join(result)
    def sec_label(self):
        return "election"+str(self.id)
    # returns true if this Election has votes still outstanding
    def active(self):
        return self.vote_set.filter(
                                disabled=0,
                                recommended__isnull=True,
                                ).count()
    def status(self):
        '''Return status of the election.

        Active = voting still in progress
        Ready = voting complete, but wsa's not reclassified
        Done = all candidates reclassified
        '''
        flavor = ElectionFlavor(self.flavor)
        needs_reclassify = False
        enum = WsAnnotation.indication_vals
        for v in self.vote_set.filter(disabled=0):
            if v.recommended is None:
                return 'Active'
            if v.drug.indication == flavor.input:
                needs_reclassify = True
        return 'Ready' if needs_reclassify else 'Done'
    def update_note_permissions(self):
        # if Active, set all notes to private; else set to public
        enable = self.status() != 'Active'
        all_votes = self.vote_set.all()
        for v in all_votes:
            if v.note:
                if enable:
                    v.note.private_to = ''
                else:
                    v.note.private_to = v.reviewer
                v.note.save()
        # now update target notes
        # - this is currently a one-shot deal: once a target note is
        #   published, it remains public no matter what subsequently
        #   happens to that or other review rounds in the workspace
        if enable:
            all_agent_ids = [x.drug.agent_id for x in all_votes]
            targ_cache = self.ws.get_canonical_target_cache(all_agent_ids)
            all_note_ids = TargetAnnotation.all_note_ids(
                    self.ws,
                    targ_cache.all_prots,
                    )
            for note in Note.objects.filter(pk__in=all_note_ids):
                if note.private_to:
                    logger.info("publishing note %d %s ws %d",
                            note.id,
                            note.label,
                            self.ws.id,
                            )
                    note.private_to = ''
                    note.save()
    def _extract_counts_by(self,attr):
        d = {}
        for v in self.vote_set.filter(disabled=0):
            key = getattr(v,attr)
            l = d.setdefault(key,[0,0])
            l[1] += 1
            if v.recommended is not None:
                l[0] += 1
        return d
    def _format_counts(self,d,labels):
        # returns a sorted list of [label,done,assigned,%done]
        # with a total row at end. Input is a dict of {key:[done,assigned],...}
        # and a sorted list of [(label,key),...]
        result = []
        for label,key in labels:
            result.append( [label]+d[key] )
        # add total
        result.append( ['Total'
                        ,sum([x[1] for x in result])
                        ,sum([x[2] for x in result])
                        ] )
        # and add percent
        for l in result:
            pct = int(100*l[1]/l[2]) if l[2] else 0
            l.append(pct)
        return result
    def reviewer_progress(self):
        d = self._extract_counts_by('reviewer')
        labels = [(x,x) for x in d.keys()]
        labels.sort(key=lambda x:x[0].lower())
        return self._format_counts(d,labels)
    def candidate_progress(self,user,is_demo):
        d = self._extract_counts_by('drug')
        from dtk.html import link,nowrap
        from rvw.utils import get_needs_vote_icon

        def get_label(x):
            vote_icon = get_needs_vote_icon(x, user)
            drug_link = nowrap(link(
                            x.get_name(is_demo),
                            x.drug_url(),
                            new_tab=True,
                            ))
            from django.utils.html import format_html
            return format_html(u'{}{}',
                    vote_icon,
                    drug_link,
                    )

        labels = [(get_label(x), x) for x in d.keys()]
        labels.sort(key=lambda x:x[1].review_code_sort_key())
        return self._format_counts(d,labels)
    def candidate_results(self, is_demo):
        from browse.models import WsAnnotation
        # return:
        # [ [user,user,...] # in alphabetical order
        # , [score,druglink,wsa,[vote,vote,...]] # in order by score
        # ]
        # vote lists are in same order as user list
        d = {}
        users = set()
        for v in self.vote_set.filter(disabled=0):
            if v.recommended is None:
                return None
            drug = d.setdefault(v.drug.get_name(is_demo),{})
            users.add(v.reviewer)
            drug[v.reviewer] = v
        if not users:
            return None
        users = list(users)
        users.sort(key=lambda x:x.lower())
        drugs = []
        from dtk.html import link,nowrap
        for k,v in six.iteritems(d):
            recs = 0
            votes = []
            for u in users:
                try:
                    vote = v[u]
                    if vote.recommended:
                        recs += 1
                except KeyError:
                    vote = None
                votes.append(vote)
            drug = list(v.values())[0].drug
            color = ""
            from dtk.plot import Color
            if drug.indication == self.flavor_info.input:
                color = ""
            elif drug.indication == self.flavor_info.output:
                color = Color.default
            else:
                color = Color.highlight
            if color:
                style = 'border-left: 10px solid %s; border-right: 10px solid %s' % (color, color)
            else:
                style = ''
            drug_link = nowrap(link(k,drug.drug_url(),new_tab=True))
            drugs.append([recs,drug_link,drug,zip(users,votes),style])
        drugs.sort(key=lambda x: x[0],reverse=True)
        return [users,drugs]

class Vote(models.Model):
    @classmethod
    def needed_election_list(cls,user):
        election_ids = cls.objects.filter(
                                disabled=0,
                                recommended__isnull=True,
                                reviewer=user,
                                ).values_list(
                                'election',
                                flat=True,
                                ).distinct()
        return Election.objects.filter(id__in=election_ids).order_by('due')
    @classmethod
    def needs_this_vote(cls,wsa,user):
        return cls.objects.filter(
                                disabled=0,
                                recommended__isnull=True,
                                drug=wsa,
                                reviewer=user,
                                ).count()
    @classmethod
    def user_votes(cls,user):
        return cls.objects.filter(
                                disabled=0,
                                reviewer=user,
                                )

    def note_info(self,attr):
        if attr == 'note':
            return {
                'label':"%s's private note on %s for %s" %(
                        self.reviewer,
                        self.drug.agent.canonical,
                        self.drug.ws.name,
                        ),
                'private_to':self.reviewer,
                }
        raise Exception("bad note attr '%s'" % attr)
    election = models.ForeignKey(Election, on_delete=models.CASCADE)
    reviewer = models.CharField(max_length=50)
    drug = models.ForeignKey(WsAnnotation, on_delete=models.CASCADE)
    recommended = models.BooleanField(null=True)
    note = models.ForeignKey(Note,null=True,blank=True, on_delete=models.CASCADE)
    disabled = models.BooleanField(default=False)

    class Meta:
        unique_together = [['election', 'reviewer', 'drug']]

    def get_note_text(self, user=''):
        # this only works if the note has been made public or you specify
        # a user
        return Note.get(self,'note',user)


# allow tracking of ArrayExpress searches and accession disposition
class AeAccession(models.Model):
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    geoID = models.CharField(max_length=256)
    title = models.TextField(default='',blank=True)
    desc = models.TextField(default='',blank=True)
    src_default = models.IntegerField(default=0)
    pub_ref = models.TextField(default='',blank=True)
    alt_ids = models.TextField(default='',blank=True)
    # Added later, may not be filled in for earlier searches.
    experiment_type = models.TextField(blank=True, null=True)
    num_samples = models.IntegerField(default=None, null=True)
    class Meta:
        index_together = [
                ['geoID'],
                ]
    def alt_ids_as_links(self):
        if not self.alt_ids:
            return ''
        from dtk.html import link,nowrap,join
        ids = self.alt_ids.split(',')
        urls = []
        for alt_id in ids:
            if alt_id.startswith('PRJ'):
                url = Tissue.bio_url(alt_id)
            elif alt_id.startswith('GSE'):
                url = Tissue.geo_url(alt_id)
            else:
                url = Tissue.ae_url(alt_id)
            urls.append(nowrap(link(alt_id, url, new_tab=True)))
        return join(*urls)

    def search_site(self):
        if self.geoID.startswith('GSE'):
            return 'GEO'
        elif self.geoID.startswith('PRJ'):
            return 'BIO'
        else:
            return 'AE'
    def host_site(self):
        if self.geoID.startswith('GSE') or self.geoID.startswith('E-GEOD-'):
            return 'GEO'
        return 'AE'
    def link(self):
        if self.search_site() == 'AE':
            url=Tissue.ae_url(self.geoID)
        elif self.search_site() == 'BIO':
            url=Tissue.bio_url(self.geoID)
        else:
            url=Tissue.geo_url(self.geoID)
        from dtk.html import link,nowrap
        return nowrap(link(self.geoID, url, new_tab=True))
    def pub_link(self):
        if not self.pub_ref:
            return ''
        if self.search_site() == 'AE':
            url='https://doi.org/'+self.pub_ref
        else:
            url='https://www.ncbi.nlm.nih.gov/pubmed/'+self.pub_ref
        from dtk.html import link,nowrap
        return nowrap(link('publication', url, new_tab=True))
    def samples(self):
        site = self.search_site()
        fallback = f'({self.num_samples} samples)'
        if site not in ('AE', 'BIO'):
            return fallback
        from dtk.html import link,nowrap
        if site == 'BIO':
            if self.geoID[:5] != 'PRJNA':
                # We need the UID for the link below, and for non-PRJNA's it's not in the ID.
                # Presumably we could get it elsewhere, but it's easy enough to just go through
                # the project page in these cases.
                return fallback
            # PRJNA
            uid = int(self.geoID[5:])
            url = f'https://www.ncbi.nlm.nih.gov/sra?linkname=bioproject_sra_all&from_uid={uid}'
        else:
            url=Tissue.ae_url(self.geoID)
            url+="samples/?s_page=1&s_pagesize=2000"
        return nowrap(link(f"samples ({self.num_samples})", url, new_tab=True))
    def reject_text(self,mode):
        try:
            disp = self.aedisposition_set.get(mode=mode)
            return disp.rejected
        except AeDisposition.DoesNotExist:
            return ''
    def get_sample_attrs(self):
        from ge.models import SraRun, GESamples
        from collections import defaultdict
        import json
        if self.geoID.startswith('PRJ'):
            samples = [x.attrs_dict() for x in SraRun.objects.filter(bioproject=self.geoID)]
        else:
            try:
                samples = GESamples.objects.get(geoID=self.geoID).get_sample_attrs()
            except GESamples.DoesNotExist:
                samples = []
        return samples

class AeSearch(models.Model):
    LATEST_VERSION = 1

    mode_vals = Enum([], [
            ('CC','Case/Control'),
            ('TR','Treatment/Response'),
            ])
    species_vals = Enum(['latin'], [
            ('human','Human',None,'Homo sapiens'),
            ('mouse','Mouse',None,'Mus musculus'),
            ('rat','Rat',None,'Rattus norvegicus'),
            ('zebrafish','Zebrafish',None,'Danio rerio'),
            ('dog','Dog',None,'Canis lupus'),
            ('any','Any'),
            ])
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    term = models.CharField(max_length=256)
    mode = models.IntegerField(choices=mode_vals.choices()
                ,default=mode_vals.CC
                )
    species = models.IntegerField(choices=species_vals.choices()
                ,default=species_vals.human
                )
    version = models.IntegerField(default=0, null=True)
    when = models.DateTimeField()
    scores = models.ManyToManyField(AeAccession, through='AeScore')
    def mode_label(self):
        return self.mode_vals.get('label',self.mode)
    def species_label(self):
        return self.species_vals.get('label',self.species)
    @classmethod
    def latin_of_species(cls,species_code):
        latin = cls.species_vals.get('latin',species_code)
        if latin is None:
            label = cls.species_vals.get('label',species_code)
            raise RuntimeError(f"No latin for species '{label}'")
        return latin
    def _imported_geo_ids(self):
        # Return a list of all IDs that are already imported
        result = set()
        for geoID in self.ws.imported_geo_ids():
            result.add(geoID)
            # for GEO data, add AE label as well; search could have come
            # from either; note that the returned list is only used for
            # filtering, not counts, so it's ok to have multiple versions
            # of the same id
            if geoID.startswith('GSE'):
                result.add('E-GEOD-'+geoID[3:])
        return result
    def imported(self):
        return self.scores.filter(geoID__in=self._imported_geo_ids())
    def rejected(self):
        return self.scores.filter(
                        aedisposition__mode=self.mode,
                        ).exclude(
                        geoID__in=self._imported_geo_ids(),
                        )
    def unclassified(self):
        return self.scores.exclude(
                        aedisposition__mode=self.mode,
                        ).exclude(
                        geoID__in=self._imported_geo_ids(),
                        )

class AeDisposition(models.Model):
    accession = models.ForeignKey(AeAccession,on_delete=models.CASCADE)
    mode = models.IntegerField(choices=AeSearch.mode_vals.choices()
                ,default=AeSearch.mode_vals.CC
                )
    rejected = models.CharField(max_length=1024)

class AeScore(models.Model):
    search=models.ForeignKey(AeSearch,on_delete=models.CASCADE)
    accession=models.ForeignKey(AeAccession,on_delete=models.CASCADE)
    score = models.FloatField()
    def ae_highlighted_link(self):
        '''Highlighted link to the accession.

        Like AeAccession.link, except it highlights the search terms.
        '''
        if self.accession.search_site() != 'AE':
            return ''
        url=Tissue.ae_url(self.accession.geoID)
        url+='?query="%s"&organism=Homo+sapiens&exptype[]="rna+assay"'
        from dtk.html import link,nowrap
        return nowrap(link(
                self.accession.geoID,
                url % self.search.term,
                new_tab=True,
                ))
    def native_link(self):
        '''Link to host source, if not AE'''
        from dtk.html import link,nowrap
        if self.accession.search_site() == 'BIO':
            native_id=self.accession.geoID
            return nowrap(link(
                    native_id,
                    Tissue.bio_url(native_id),
                    new_tab=True,
                    ))
        elif self.accession.host_site() != 'AE':
            if self.accession.search_site() == 'AE':
                native_id='GSE'+self.accession.geoID.split('-')[2]
            else:
                native_id=self.accession.geoID
            return nowrap(link(
                    native_id,
                    Tissue.geo_url(native_id),
                    new_tab=True,
                    ))
        return ''

# allow flagging named subsets of drugs
class DrugSet(models.Model):
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    name = models.CharField(max_length=256)
    drugs = models.ManyToManyField(WsAnnotation)
    description = models.TextField(default='')

    created_on = models.DateTimeField(auto_now_add=True, null=True)
    created_by = models.CharField(max_length=256,default='')
    def __unicode__(self):
        return self.name

    @transaction.atomic
    def add_mols(self, mols, user):
        for mol in mols:
            self.drugs.add(mol)

        dsc = DrugSetChange.objects.create(
                drugset=self,
                user=user,
                description="Added molecules"
                )
        dsc.added.add(*mols)

    @transaction.atomic
    def remove_mols(self, mols, user):
        for mol in mols:
            self.drugs.remove(mol)

        dsc = DrugSetChange.objects.create(
                drugset=self,
                user=user,
                description="Removed molecules"
                )
        dsc.removed.add(*mols)

class DrugSetChange(models.Model):
    """Audit log for drugset changes."""
    description = models.TextField(default='')
    added = models.ManyToManyField(WsAnnotation, related_name='wsa_added')
    removed = models.ManyToManyField(WsAnnotation, related_name='wsa_removed')
    user = models.CharField(max_length=256)
    date = models.DateTimeField(auto_now_add=True)
    drugset = models.ForeignKey(DrugSet, on_delete=models.CASCADE)


# A pseudo-drugset, used as the backing store for manually-modified
# auto drugsets
class DrugSplit(models.Model):
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    name = models.CharField(max_length=256)
    drugs = models.ManyToManyField(WsAnnotation)
    # This drug split has manual edits and should be kept stable.
    manual_edits = models.BooleanField(default=False)
    class Meta:
        unique_together = ('ws', 'name')

class TargetAnnotation(models.Model):
    # This class represents disease-specific information about a gene
    # in a workspace. Initially it just anchors the TargetReview class
    # but will eventually hold a global note and an indication-like enum.
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    uniprot = models.CharField(max_length=56)

    class Meta:
        unique_together = ('ws', 'uniprot')


    @classmethod
    def all_note_ids(cls,ws,uniprots):
        '''Return an iterable over target note_ids.

        This is restricted to the specified workspace and uniprots,
        but unlike the function below does no user-based filtering,
        so it's suitable for code that needs to see restricted notes.
        '''
        return TargetReview.objects.filter(
                        target__ws=ws,
                        target__uniprot__in=uniprots,
                        note__isnull=False,
                        ).values_list('note_id',flat=True)
    @classmethod
    def batch_note_lookup(cls,ws,uniprots,user):
        '''Return {uniprot:{user:(note_id,text),...},...}.

        Includes all target notes for the requested uniprots and workspace
        that are accessible to the user. This allows a page with many
        uniprot links to get note-icon information for all of them efficiently.
        '''
        note2src = {
                note_id:(user,uniprot)
                for note_id,user,uniprot in TargetReview.objects.filter(
                        target__ws=ws,
                        target__uniprot__in=uniprots,
                        note__isnull=False,
                        ).values_list('note_id','user','target__uniprot')
                }
        result = {}
        for note_id,text in Note.batch_note_lookup(list(note2src.keys()),user):
            user,uniprot = note2src[note_id]
            d2 = result.setdefault(uniprot,{})
            d2[user] = (note_id,text)
        return result

class TargetReview(models.Model):
    # This class represents a particular reviewer's private information
    # about a target. For now it just holds a note.
    target = models.ForeignKey(TargetAnnotation, on_delete=models.CASCADE)
    user = models.CharField(max_length=70)
    note = models.ForeignKey(Note,null=True,blank=True, on_delete=models.CASCADE)

    class Meta:
        unique_together=[['user', 'target']]
    @classmethod
    def save_note(cls,ws,uniprot,user,text,private=True):
        '''Save a private note with any needed scaffolding.

        This allows a note to be saved without worrying about whether
        the TargetAnnotation and TargetReview objects exist yet.
        '''
        ta,created = TargetAnnotation.objects.get_or_create(
                ws=ws,
                uniprot=uniprot,
                )
        tr,created = cls.objects.get_or_create(target=ta,user=user)
        from notes.models import Note
        Note.set(tr,'note',user,text,private=private)
    def get_note_text(self, user=''):
        return Note.get(self,'note',user)
    def note_info(self,attr):
        if attr == 'note':
            return {
                'label':"target note for %s by %s" %(
                        self.target.uniprot,
                        self.user,
                        ),
                }
        raise Exception("bad note attr '%s'" % attr)

class ProteinUploadStatus(models.Model):
    timestamp = models.DateTimeField(auto_now=True)
    filename = models.CharField(max_length=256)
    ok = models.BooleanField(default=False)
    @classmethod
    def current_upload(cls):
        qs = cls.objects.order_by('-timestamp')
        try:
            latest = qs[0]
        except IndexError:
            return None
        if latest.ok:
            return latest.filename
        return None

class Protein(models.Model):
    class Meta:
        # Normally we could just annotate with db_index, but this
        # is an alternative syntax that works with django's migrations.
        index_together = [
                ['uniprot'],
                ['gene']
                ]
    uniprot = models.CharField(max_length=56)
    gene = models.CharField(max_length=56,default='')
    uniprot_kb = models.CharField(max_length=56,default='')
    def __unicode__(self):
        return self.uniprot
    def get_aliases(self):
        qs = self.proteinattribute_set.filter(attr__name='Alt_uniprot')
        return list(qs.values_list('val',flat=True))
    def get_pathways(self):
        qs = self.proteinattribute_set.exclude(attr__pathway_group='')
        return list(qs.values_list('attr__name',flat=True))
    def get_search_names(self):
        all_prot_names = [self.get_name()] + [self.gene] + self.get_alt_names()
        # Filter out EC names, we never use them.
        all_prot_names = [x for x in all_prot_names if not x.startswith('EC ')]
        return all_prot_names
    def is_dpi_druggable(self):
        from browse.utils import get_dpi_druggable_prots
        prots = get_dpi_druggable_prots()
        return self.uniprot in prots
    def is_small_mol_druggable(self):
        from dtk.open_targets import OpenTargets
        from browse.default_settings import openTargets
        otarg = OpenTargets(openTargets.latest_version())
        return otarg.is_small_mol_druggable(self.uniprot)
    @classmethod
    def get_all_pathways(cls):
        qs = ProteinAttributeType.objects.exclude(pathway_group='')
        return list(qs.values_list('name',flat=True))
    @classmethod
    def get_proteins_for_pathway(cls,pathway):
        '''Returns a queryset for all Proteins in a pathway.

        This would typically be used as a generator:
        s = set(Protein.get_proteins_for_pathway('KEGG_GALACTOSE_METABOLISM'))
        '''
        return cls.objects.filter(proteinattribute__attr__name=pathway)
    @classmethod
    def get_uniprots_for_pathway(cls,pathway):
        '''Returns a list of uniprot ids in a pathway.
        '''
        qs=cls.get_proteins_for_pathway(pathway)
        return qs.values_list('uniprot',flat=True)
    def get_prot_attrs(self):
        keys = {}
        for pa in self.proteinattribute_set.all():
            l = keys.setdefault(pa.attr.name,[])
            l.append(pa.val)
        return keys
    def ext_src_urls(self):
        keys = self.get_prot_attrs()
        result = []
        for key in keys.get('GeneID',[]):
            result.append( (f'DisGeNet ({key})',
                f'https://www.disgenet.org/browser/1/1/0/{key}/'
                , 'db' ))

        for key in keys.get('STRING',[]):
            result.append( (f'String PPI ({key})',
                "http://string-db.org/cgi/network.pl?all_channels_on=1&block_structure_pics_in_bubbles=0&direct_neighbor=1&hide_disconnected_nodes=0&hide_node_labels=0&network_display_mode=svg&network_flavor=evidence&targetmode=proteins&identifier=%s" % key,
                'other_db'))
        ens_keys = keys.get('Ensembl',[])
        for key in keys.get('GeneCards',[]):
            result.append( (f'GeneCards ({key})',
                "http://www.genecards.org/cgi-bin/carddisp.pl?gene=%s" % key,
                'db' ))
        for key in ens_keys:
            result.append( (f'ProteinAtlas ({key})',
                "http://www.proteinatlas.org/%s" % key,
                'other_db' ))
            result.append( (f'OpenTargets ({key})',
                "https://www.targetvalidation.org/target/%s" % key,
                'db' ))
        for key in keys.get('hgnc',[]):
            result.append( (f'AGR ({key})',
                "https://www.alliancegenome.org/gene/HGNC:%s" % key,
                'other_db',))
            result.append( (f'Monarch ({key})',
                "https://monarchinitiative.org/gene/HGNC:%s" % key,
                'other_db',))
        for key in keys.get('KEGG',[]):
            result.append( (f'KEGG ({key})',
                "http://www.genome.jp/dbget-bin/www_bget?%s" % key,
                'other_db',))
        for key in keys.get('MIM',[]):
            result.append( (f'OMIM ({key})',
                "https://omim.org/entry/%s" % key,
                'other_db',))
        for key in keys.get('Reactome',[]):
            result.append( (f'Reactome ({key})',
                "http://www.reactome.org/content/detail/%s" % key,
                'assoc',))
        return result
    @classmethod
    def search(cls,pattern,limit=None):
        from dtk.prot_search import search_by_any
        return search_by_any(pattern, limit)[0]

    @classmethod
    def get_canonical_of_uniprot(cls,uniprot):
        try:
            p=cls.objects.get(uniprot=uniprot)
            return p
        except cls.DoesNotExist:
            pass

        # In rare circumstances (e.g. https://www.uniprot.org/uniprot/P04745) a uniprot exists as an
        # alt_uniprot for multiple other prots.
        # Possibly this should be resolved on the backend and a canonical picked arbitrarily, but for now,
        # just pick one arbitrarily here.
        p=cls.objects.filter(
                proteinattribute__attr__name='Alt_uniprot',
                proteinattribute__val=uniprot,
                ).order_by('proteinattribute__val')
        if len(p) == 0:
            return None
        else:
            return p[0]
    @classmethod
    def get_gene_of_uniprot(cls,uniprot):
        p = cls.get_canonical_of_uniprot(uniprot)
        if p:
            return p.gene
        return ''
    @classmethod
    def get_uniprot_gene_map(cls,uniprots=None):
        # load alternate names first, so they'll be overwritten by
        # primary names in case of updates
        alt_qs=ProteinAttribute.objects.filter(
                        attr__name='Alt_uniprot',
                ).exclude(
                        prot__gene='',
                )
        qs=cls.objects.exclude(gene='')
        if uniprots:
            alt_qs=alt_qs.filter(val__in=uniprots)
            qs=qs.filter(uniprot__in=uniprots)
        d = dict(alt_qs.values_list('val','prot__gene'))
        d.update(qs.values_list('uniprot','gene'))
        return d
    @classmethod
    def get_gene_uniprot_map(cls,genes=None):
        qs=cls.objects.exclude(gene='')
        if genes:
            qs=qs.filter(gene__in=genes)
        return dict(qs.values_list('gene','uniprot'))
    @classmethod
    def make_gene_list_from_uniprot_list(cls,ids):
        uni_2_gene = cls.get_uniprot_gene_map()
        return [ uni_2_gene.get(id,id) for id in ids ]
    @classmethod
    def upload_pathway_from_gmt(cls,source,fn):
        from dtk.readtext import parse_delim
        wl = []
        for pathway,protlist in parse_delim(open(fn)):
            obj,new = ProteinAttributeType.objects.get_or_create(
                    name=pathway,
                    pathway_group=source,
                    )
            if not new:
                ProteinAttribute.objects.filter(attr=obj).delete()
            protlist = [
                    x.strip()
                    for x in protlist.split(',')
                    ]
            for prot in Protein.objects.filter(uniprot__in=protlist):
                pa = ProteinAttribute(prot=prot,attr=obj,val='Y')
                wl.append(pa)
                if len(wl) >= 2000:
                    ProteinAttribute.objects.bulk_create(wl)
                    wl = []
        if wl:
            ProteinAttribute.objects.bulk_create(wl)
    def get_uniprot_url(self):
        from dtk.html import link
        uniprot_url = 'https://www.uniprot.org/uniprot/'
        return link(self.gene if self.gene else self.uniprot,
                    uniprot_url + str(self.uniprot) + '/',
                    new_tab=True
                   )
    def get_name(self):
        attr_name = 'Protein_Name'
        try:
            return self.proteinattribute_set.get(attr__name=attr_name).val
        except ProteinAttribute.DoesNotExist as e:
            return ''

    def get_alt_names(self):
        attr_name = 'Alt_Protein_Name'
        try:
            return list(self.proteinattribute_set.filter(attr__name=attr_name).values_list('val', flat=True))
        except ProteinAttribute.DoesNotExist as e:
            return []
    def get_url(self, ws_id):
        return Workspace.ws_reverse('protein',ws_id,self.uniprot)
    def get_html_url(self, ws_id):
        from dtk.html import link
        name = self.gene
        # Some prots (e.g. A0A0J9YVX5) have no name
        if len(name) == 0:
            name = f'({self.uniprot})'
        return link(name, self.get_url(ws_id))

class GlobalTargetAnnotation(models.Model):
    target = models.OneToOneField(Protein, on_delete=models.CASCADE)
    note = models.ForeignKey(Note,null=True,blank=True, on_delete=models.CASCADE)
    def note_info(self,attr):
        if attr == 'note':
            return {
                'label':"global target note on %s (%s)" %(
                        self.target.gene,
                        self.target.uniprot,
                        ),
                }
        raise Exception("bad note attr '%s'" % attr)


class ProteinAttributeType(models.Model):
    class Meta:
        unique_together=[['name', 'pathway_group']]
    name = models.CharField(max_length=256)
    pathway_group = models.CharField(max_length=256,blank=True,default='')

class ProteinAttribute(models.Model):
    prot = models.ForeignKey(Protein, on_delete=models.CASCADE)
    attr = models.ForeignKey(ProteinAttributeType, on_delete=models.CASCADE)
    val = models.CharField(max_length=100)

    class Meta:
        # Normally we could just annotate with db_index, but this
        # is an alternative syntax that works with django's migrations.
        index_together = [
                ['val'],
                ]

# allow flagging named subsets of proteins
class ProtSet(models.Model):
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    name = models.CharField(max_length=256)
    proteins = models.ManyToManyField(Protein)

    description = models.TextField(default='')

    created_on = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    created_by = models.CharField(max_length=256,default='')

    default_unwanted_protset = 'globps_unwanted_tier1'
    default_nonnovel_protset = 'autops_none'
    def uniprot_sorted_proteins(self):
        return self.proteins.order_by('uniprot')
    def gene_sorted_proteins(self):
        return self.proteins.order_by('gene')
    def unique_genes(self):
        return set(self.proteins.all().values_list('gene', flat=True))
    def __unicode__(self):
        return self.name

    @classmethod
    def get_global_protset(cls, name):
        sets = cls.get_global_protsets()
        out = [prots for psname, label, prots in cls.get_global_protsets()
               if name == psname]
        assert len(out) == 1
        return out[0]

    @classmethod
    def get_global_protsets(cls):
        def genes_to_uniprots(genes):
            pqs=Protein.objects.filter(gene__in=genes)
            uniprots = []
            uniprots += pqs.values_list('uniprot', flat=True)
            return uniprots

        # Pull down the file
        from dtk.s3_cache import S3File, S3MiscBucket
        from dtk.files import get_file_records
        s3_file = S3File(S3MiscBucket(),'global_protsets.tsv')
        s3_file.fetch()
        from collections import defaultdict
        uniprot_sets = defaultdict(set)
        gene_sets = defaultdict(set)
        set_names = {}

        records = get_file_records(s3_file.path(), keep_header=False)
        for uniprot, gene, set_id, set_name in records:
            set_names[set_id] = set_name

            # We used to have some explicitly included uniprot entries, but they get
            # out-of-date with the canonical set of uniprots we use.
            # Rely instead on the gene mapping.
            if False:
                if uniprot.strip():
                    uniprot_sets[set_id].add(uniprot)

            if gene.strip():
                gene_sets[set_id].add(gene)

        for set_id, genes in gene_sets.items():
            uniprots = genes_to_uniprots(genes)
            uniprot_sets[set_id].update(uniprots)

        return sorted([
                (set_id, set_names[set_id], uniprot_sets[set_id])
                for set_id in uniprot_sets
            ])


Species = Enum([], [
    ('HUMAN',),
    ('MOUSE',),
    ('RAT',),
    ('DOG',),
    ('ZEBRAFISH',),
            ])

# The default tissue set for a workspace is created in
# Workspace.get_tissue_set_choices(), which also handles upgrading
# old tissue records. The default tissue is always the lowest
# numbered one (the first one returned by get_tissue_set_choices()).
class TissueSet(models.Model):
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    # XXX eventually drop ws column from Tissues?
    name = models.CharField(max_length=256)
    case_label = models.CharField(max_length=100,default='Case')
    control_label = models.CharField(max_length=100,default='Control')
    miRNA = models.BooleanField(default=False)
    species = models.IntegerField(choices=Species.choices(), default=Species.HUMAN)
    def tissue_ids(self):
        return self.tissue_set.values_list('id',flat=True)
    def ts_label(self):
        if self.name == 'default':
            return 'Case/Control'
        if self.species == Species.HUMAN:
            return self.name
        else:
            return f'{self.name} ({Species.get("label", self.species)})'
    def get_pathsum_jobname(self):
        from runner.process_info import JobInfo
        bji = JobInfo.get_unbound('path')
        return bji.get_jobname_for_tissue_set(self)

    def num_valid_tissues(self):
        count = 0
        qs = Tissue.objects.filter(tissue_set=self)
        for t in qs:
            _,_,_,total = t.sig_result_counts()
            if total:
                count += 1
        return count


# It's possible for mysql to re-assign ids from deleted tissues.  This
# causes confusion in the job history, and caused spurious failures in
# the selenium POST test.  So, rather than actually deleting the tissue,
# we mark it, and use a custom manager to filter out marked records.
# If we need to retrieve these at some point, we could create a second
# manager with a name like 'including_invalid'.
class TissueManager(models.Manager):
    def get_queryset(self):
        return super(TissueManager, self).get_queryset().filter(invalid=False)

class Tissue(models.Model):
    objects = TissueManager()
    def invalidate(self):
        Sample.objects.filter(tissue=self).delete()
        self.invalid = True
        self.save()
    def note_info(self,attr):
        if attr == 'note':
            return {
                'label':"tissue note on %s for %s" %(
                        self.name,
                        self.ws.name,
                        ),
                }
        raise Exception("bad note attr '%s'" % attr)
    # The 'source' field actually holds both information about where
    # the tissue data comes from and how it was imported.  Currently,
    # there are 4 possible sources:
    # - GEO
    # - AE
    # - EXT (processed externally and the final result uploaded from a file)
    # - COMBINED (multiple other tissues combined using GESig).
    # If the data comes from GEO or AE, it can be processed in one of 4 ways:
    # - normal
    # - fallback
    # - RNAseq
    # - Mixed
    # Also, unlike other enums, the value stored in the database is not
    # the numerical index, but the string metaGEO_parm.  This gives us
    # the flexibility to add new processing methods in the middle of the
    # enum, preserving the encoding and decoding patterns.
    # The methods immediately after the enum definition help encode and
    # decode source values.
    source_vals = Enum(['metaGEO_parm'],
            [ ('GEO','GEO',None,'geo')
            , ('AE','ArrayExpress',None,'ae')
            , ('GEO_ORIG','GEO fallback',None,'geo-orig')
            , ('AE_ORIG','AE fallback',None,'ae-orig')
            , ('GEO_SEQ','GEO RNAseq',None,'geo-seq')
            , ('AE_SEQ','AE RNAseq',None,'ae-seq')
            , ('GEO_MIX','GEO Mixed',None,'geo-mixed')
            , ('AE_MIX','AE Mixed',None,'ae-mixed')
            , ('EXT','External',None,'ext')
            , ('COMBINED','Combined',None,'comb')
            ])
    methods = ['Normal','Fallback','RNAseq','Mixed']
    fallback_idx = methods.index('Fallback')
    @classmethod
    def method_choices(cls,include_fallback=False):
        return [x
                for x in enumerate(cls.methods)
                if include_fallback or x[0] != cls.fallback_idx
                ]
    @classmethod
    def build_normal_source(cls,db_idx,method_idx):
        return cls.source_vals.get('metaGEO_parm',2*method_idx+db_idx)
    @classmethod
    def methodless(self,src_idx):
        return src_idx >= self.source_vals.EXT
    def parse_normal_source(self):
        src_idx = self.source_vals.find('metaGEO_parm',self.source)
        if self.methodless(src_idx):
            raise ValueError("no method for '%s' tissues"%self.source)
        return (src_idx%2,src_idx//2)
    def get_method_idx(self):
        return self.parse_normal_source()[1]
    def build_updated_source(self,method_idx,fallback_reason):
        # enforce rules:
        # - this can only be done on AE or GEO tissues
        # - if method is fallback, fallback reason must exist
        #   - but allow legacy missing values
        method_idx = int(method_idx)
        old_db,old_method = self.parse_normal_source()
        if method_idx == self.fallback_idx:
            if not fallback_reason and method_idx != old_method:
                raise ValueError('Fallback reason must be set')
        return self.build_normal_source(old_db,method_idx)
    @classmethod
    def get_db_idx(cls,geoID):
        if geoID.startswith('E-'):
            return 1
        if geoID.startswith('GSE') or geoID.startswith('GDS') or geoID.startswith('PRJ'):
            return 0
        raise ValueError('Must begin with E-, GSE, or GDS')
    @classmethod
    def geo_url(cls,base_id):
        return "http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc="+base_id
    @classmethod
    def ae_url(cls,base_id):
        return "http://www.ebi.ac.uk/arrayexpress/experiments/%s/"%base_id
    @classmethod
    def bio_url(cls, base_id):
        return f'https://ncbi.nlm.nih.gov/bioproject/{base_id}'
    def source_url(self):
        if self.is_bioproject():
            return Tissue.bio_url(self.geoID)
        stem = self.base_geo()
        if self.source.startswith('geo'):
            return self.geo_url(stem)
        if self.source.startswith('ae'):
            return "http://www.ebi.ac.uk/arrayexpress/experiments/"+stem
        return ''
    def source_label(self):
        enum = Tissue.source_vals
        return enum.get('label',enum.find('metaGEO_parm',self.source))
    def is_bioproject(self):
        return self.geoID.startswith('PRJ')

    from collections import namedtuple
    SigResultRec = namedtuple(
            'SigResultRec',
            'uniprot evidence direction fold_change',
            )
    def sig_bji(self):
        if not self.sig_result_job_id:
            raise ValueError("No sig results for tissue '%s'"%str(self))
        from runner.process_info import JobInfo
        return JobInfo.get_bound(self.ws,self.sig_result_job_id)
    def sig_fvs(self):
        bji = self.sig_bji()
        cat=bji.get_data_catalog()
        return cat.get_feature_vectors('ev','dir','fold')
    def fast_one_sig_result(self,uniprot):
        bji = self.sig_bji()
        from dtk.files import get_file_records
        for rec in get_file_records(
                bji.fn_dcfile,
                select=([uniprot],0),
                keep_header=False,
                ):
            # this should only match one record; return the first match
            return self.SigResultRec(
                    rec[0],
                    float(rec[1]),
                    int(rec[2]),
                    float(rec[3]),
                    )
        return None
    def sig_prots(self):
        """Returns all scored proteins, regardless of significance"""
        bji = self.sig_bji()
        from dtk.files import get_file_records
        out = []
        for rec in get_file_records(bji.fn_dcfile, cut='1', keep_header=False):
            out.append(rec[0])
        return out

    def one_sig_result(self,uniprot):
        try:
            # attempt an optimized retrieval based on assumptions
            # about run_sig.py, bypassing the data catalog
            return self.fast_one_sig_result(uniprot)
        except ValueError:
            return None
        except (AttributeError,IOError):
            pass
        try:
            cols,data=self.sig_fvs()
        except ValueError:
            return None
        for key,vec in data:
            if key == uniprot:
                return self.SigResultRec(key,*vec)
        return None
    def sig_results(self,
            over_only=True,
            ev_cutoff=None, # None means use default
            fc_cutoff=None, # None means use default
            ):
        try:
            cols,data=self.sig_fvs()
        except ValueError:
            return []
        if not over_only:
            return [
                    self.SigResultRec(key,*vec)
                    for key,vec in data
                    ]
        ev = self.ev_cutoff if ev_cutoff is None else ev_cutoff
        fc = self.fc_cutoff if fc_cutoff is None else fc_cutoff
        return [
                self.SigResultRec(key,*vec)
                for key,vec in data
                if vec[0] >= ev and vec[2] >= fc
                ]
    def _start_sig_result_job(self,job_id):
        self.sig_result_job_id = job_id
        self.total_proteins = 0
        self.over_proteins = 0
        self.save()
    def _recalculate_sig_result_counts(self):
        # must call save() for this method to take effect
        l = list(self.sig_results(over_only=False))
        self.total_proteins = len(l)
        self.over_proteins = len([
                1
                for r in l
                if r.evidence >= self.ev_cutoff
                and r.fold_change >= self.fc_cutoff
                ])
    def sig_result_counts(self):
        return (
                self.over_proteins,
                self.ev_cutoff,
                self.fc_cutoff,
                self.total_proteins,
                )
    def _prep_foreground_sig(self,user):
        # This is a hook which allows a tissue to populate its sig results
        # from somewhere other than run_sig. Given a sig record, it creates
        # a dummy job (needed because the data file path contains a job_id),
        # and then returns a bji for that job, ready for a call to
        # _convert_sig_result or the equivalent
        from runner.process_info import JobInfo
        uji = JobInfo.get_unbound('sig')
        import json
        job_id = Process.dummy_process(
                uji.get_jobname_for_tissue(self),
                settings_json=json.dumps({
                            'tissue_id':self.id,
                            }),
                user=user,
                )
        self._start_sig_result_job(job_id)
        bji = JobInfo.get_bound(self.ws,job_id)
        make_directory(bji.lts_abs_root)
        return bji
    @classmethod
    def sig_count_heading(cls):
        return "over (evid/fc) total"
    def sig_count_fmt(self):
        over,ev,fc,total = self.sig_result_counts()
        return '%d (%s/%s) of %d' % (
                over,
                sci_fmt(ev),
                sci_fmt(fc),
                total,
                )
    def legacy_outliers(self):
        '''Return any outlier ids stored in old format.'''
        path = PathHelper.publish+self.geoID+"/outlierGsms.csv"
        # file contains one outlier per line, with an 'x' in the first line
        result = []
        import re
        try:
            f=open(path,"r")
            next(f) # skip 'x' line
            for line in f:
                line = line.strip()
                if line == 'nr' or line == 'ng':
                    continue
                if self.source.startswith('geo'):
                    # extract GSM, and strip possible color prefix
                    m = re.match('(nr_|ng_|)(GSM[0-9]+).*',line)
                    if not m:
                        raise ValueError('bad outlier file at '+path)
                    line = m.group(2)
                result.append(line)
        except IOError:
            pass
        return result
    def outlier_count(self):
        # first handle legacy case
        count = len(self.legacy_outliers())
        if count:
            return count
        # extract from metadata file
        try:
            f=open(self.metadata_path())
            return sum([
                    1
                    for line in f
                    if line.endswith('\tOutlier: True\n')
                    ])
        except IOError:
            return 0
    def quality_links(self,jcc):
        result_debug = ""
        tissue_suffix = self.tissue_suffix()
        base_key = self.base_geo() + tissue_suffix
        key = self.geoID + tissue_suffix
        # calculate array link text, with outlier count
        array_link = "array"
        try:
            outliers = self.outlier_count()
            if outliers:
                array_link += "(%d)"%outliers
        except ValueError:
            array_link += " (BAD OUTLIER FILE; RE-RUN META)"
            jcc.bad_outliers = True
        # calculate sig_qc link text, with overall score
        sig_qc_link = "sig_qc"
        qc_scores = self.sig_qc_scores()
        overall = 'finalScore'
        if qc_scores.get(overall,'NA') != 'NA':
            # add score to link, if it's available and numeric
            sig_qc_link += '('+sci_fmt(qc_scores[overall])+')'
        from runner.process_info import JobInfo
        from runner.common import LogRepoInfo
        warnings = {}
        if self.sig_result_job_id:
            lri = LogRepoInfo(self.sig_result_job_id)
            lri.fetch_log()
            sig_log = lri.log_path()
            sig_bji = JobInfo.get_bound(self.ws,self.sig_result_job_id)
            warn = sig_bji.get_warnings()
            if warn:
                warnings['sig_log'] = f'{len(warn)} sig warning(s)'
        else:
            sig_log=None
        # get most recent meta log file
        meta_job = jcc.latest_jobs().get(self.get_meta_jobname())
        if meta_job:
            lri = LogRepoInfo(meta_job.id)
            lri.fetch_log()
            meta_log = lri.log_path()
            meta_bji = JobInfo.get_bound(self.ws,meta_job)
            warn = meta_bji.get_warnings()
            if warn:
                warnings['meta_log'] = f'{len(warn)} meta warning(s)'
        else:
            meta_log=None
        from dtk.html import link,glyph_icon
        from dtk.plot import Color
        for text,suffix in (
                # Almost everything in this list is for legacy directory
                # support. Modern jobs show only:
                # - for Meta, the array and meta_log links
                # - for Sig, the qc and log links
                ### Meta links
                (array_link,self.geoID+"/index.html"),
                ("meta_log", meta_log),
                ("meta_err", self.geoID+"/metaGEO_"+self.geoID+".err.txt"),
                ("meta_out", self.geoID+"/metaGEO_"+self.geoID+".log.txt"),
                ### Sig links
                ("MDS",key+"/"+base_key+"_MDS.png"),
                ("MA_plot",key+"/"+key+"_MAplot.png"),
                ("geoDE",key+"/"+base_key+"_geodePlot.pdf"),
                (sig_qc_link, None),
                ("sig_log", sig_log),
                ("sig_err", key+"/sigGEO_"+key+".err.txt"),
                ("sig_out",key+"/sigGEO_"+key+".log.txt"),
                ):
            if text == sig_qc_link:
                qc_url = self.sig_qc_url()
                if qc_url:
                    result_debug += (" "+link(text,qc_url))
            elif suffix == None:
                continue # for optional meta_log,sig_log cases
            else:
                if suffix.startswith('/'):
                    path = suffix
                else:
                    path = PathHelper.publish+suffix
                if os.path.isfile(path):
                    result_debug += (" "+link(
                                            text,
                                            PathHelper.url_of_file(path),
                                            ))
                    if text in warnings:
                        result_debug += glyph_icon(
                                'minus-sign',
                                color=Color.highlight,
                                hover=warnings[text],
                                )
        return mark_safe(result_debug)
    @classmethod
    def combine_tissues(cls,name,input_ids,user):
        '''create a single combined tissue from the specified inputs.

        Use GESig to combine the input tissue significant proteins into a
        single signature.  Create a new 'comb'-type tissue, and upload the
        combined signature as its sigprot list. Copy all input Sample records
        so Case/Control counts remain correct. Move all the input tissues
        to excluded, but record their ids so they can be restored.
        '''
        # verify all tissues exist
        inputs = list(cls.objects.filter(pk__in=input_ids))
        assert len(inputs) == len(input_ids)
        # verify all tissues are in the same tissue set
        ts = set([t.tissue_set_id for t in inputs])
        assert len(ts) == 1
        ts = list(ts)[0]
        assert ts
        # calculate combined signature
        from algorithms.run_gesig import assemble_tissue_data,combine_tissue_data
        tge_dd = assemble_tissue_data([(x,1.0) for x in input_ids])
        sig = combine_tissue_data(tge_dd,1)
        # create new tissue
        comb_tissue = cls(ws=inputs[0].ws,
                name=name.strip(),
                geoID=','.join([str(x) for x in input_ids]),
                source='comb',
                tissue_set_id=ts,
                )
        comb_tissue.save()
        # clear out any stale sample results
        Sample.objects.filter(tissue=comb_tissue).delete()
        # Copy Case/Control records
        for s in Sample.objects.filter(tissue_id__in=input_ids):
            copy = Sample(
                    tissue=comb_tissue,
                    sample_id=s.sample_id,
                    primary_factor=s.primary_factor,
                    secondary_factor=s.secondary_factor,
                    attributes=s.attributes,
                    classification=s.classification,
                    )
            copy.save()
        # upload signature
        bji = comb_tissue._prep_foreground_sig(user)
        rl = []
        for uni,vec in six.iteritems(sig):
            ev,fc = vec[:2]
            # direction is sign of ev (and fc)
            dirn = ev and (1,-1)[ev<0]
            rl.append((uni,abs(ev),dirn,abs(fc)))
        bji.write_sig_file(rl)
        bji.finalize()
        comb_tissue._recalculate_sig_result_counts()
        comb_tissue.save()
        # exclude input tissues
        for r in inputs:
            r.tissue_set=None
            r.save()
    def get_input_ids(self):
        """Gives the tissue IDs that make up this combined tissue."""
        return [int(x) for x in self.geoID.split(',')]
    def split_combined_tissue(self):
        '''effectively undo combine_tissues.

        Move this tissue out of its tissue set, and move the original
        input tissues back in.
        '''
        assert self.source == 'comb'
        assert self.tissue_set
        input_ids = self.get_input_ids()
        for r in Tissue.objects.filter(pk__in=input_ids):
            r.tissue_set=self.tissue_set
            r.save()
        self.tissue_set = None
        self.save()
    def set_name(self):
        if self.tissue_set:
            return self.tissue_set.name
        return '(Excluded)'
    def get_meta_jobname(self):
        from runner.process_info import JobInfo
        bji = JobInfo.get_unbound('meta')
        return bji.get_jobname_for_tissue(self)
    def get_sig_jobname(self):
        from runner.process_info import JobInfo
        bji = JobInfo.get_unbound('sig')
        return bji.get_jobname_for_tissue(self)
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    tissue_set = models.ForeignKey(TissueSet,null=True,blank=True, on_delete=models.CASCADE)
    name = models.CharField(max_length=256)
    geoID = models.CharField(max_length=256)
    source = models.CharField(max_length=20,default='geo')
    note = models.ForeignKey(Note,null=True,blank=True, on_delete=models.CASCADE)
    invalid = models.BooleanField(default=False)
    cc_selected = models.CharField(max_length=256,default='')
    fallback_reason = models.CharField(max_length=1024,default='')
    ev_cutoff = models.FloatField(default=0.95)
    fc_cutoff = models.FloatField(default=0)
    cutoff_job_id = models.IntegerField(null=True)
    sig_result_job_id = models.IntegerField(null=True)
    total_proteins = models.IntegerField(default=0)
    over_proteins = models.IntegerField(default=0)
    ignore_missing = models.BooleanField(default=False)
    def thresh_job_warning(self):
        if self.ev_cutoff != 0.95 or self.fc_cutoff != 0:
            if self.sig_result_job_id:
                if self.cutoff_job_id != self.sig_result_job_id:
                    return ' '.join([
                            "threshold set from",
                            'different' if self.cutoff_job_id else 'unknown',
                            'sig run',
                            ])
    def metadata_path(self):
        return PathHelper.publish+"%s/%s_metadata.tsv"%(self.geoID,self.geoID)
    def case_control_counts(self):
        return (
            self.sample_set.filter(classification=1).count(),
            self.sample_set.filter(classification=2).count(),
        )
    def base_geo(self):
        return self.geoID.split(':')[0]
    def tissue_suffix(self):
        return "_%d" % self.id
    @classmethod
    def concisify(cls,name):
        # XXX this method only exists for converting a legacy web page
        # XXX that only handles strings, but may depend on using the same
        # XXX truncation method as concise_name()
        from dtk.text import limit
        return limit(name,maxlen=30)
    def concise_name(self):
        return self.concisify(self.name)
    def sig_qc_url(self):
        if self.source in ('ext','comb'):
            return None
        return reverse('ge:sig_qc',args=tuple([self.ws_id,self.id]))
    def sig_qc_scores(self):
        if False:
            try:
                bji = self.sig_bji()
            except ValueError:
                return {}
            fn = bji.fn_sigqc
        else:
            # XXX a fast path which makes some assumptions about
            # XXX sig_qc file location, rather than consulting
            # XXX the plugin
            if not self.sig_result_job_id:
                return {}
            fn = os.path.join(
                    PathHelper.lts,
                    str(self.ws_id),
                    'sig',
                    str(self.sig_result_job_id),
                    'sigqc.tsv',
                    )
        try:
            with open(fn) as f:
                qc_scores=dict([
                        x.split('\t')
                        for x in f.read().split('\n')
                        if '\t' in x
                        ])
        except IOError:
            qc_scores = {}
        return qc_scores
    def get_note_text(self):
        return Note.get(self,'note','')
    def __str__(self):
        return self.name+' ('+str(self.id)+')'

class Sample(models.Model):
    group_vals = Enum([],
            [ ('IGNORE',)
            , ('CASE',)
            , ('CONTROL',)
            ])
    tissue = models.ForeignKey(Tissue, on_delete=models.CASCADE)
    sample_id = models.CharField(max_length=150)
    primary_factor = models.CharField(max_length=256)
    secondary_factor = models.CharField(max_length=256)
    attributes = models.TextField(blank=True)
    classification = models.IntegerField(choices=group_vals.choices())

# XXX - maybe, define Workspace-level functions that provide a single namespace
# XXX   for retrieving alternative sources of sigprot-equivalent data and/or
# XXX   collections of prot scorings (like tissue sets, all GWAS, etc.):
# XXX   - run_faers->dtk.disgenet
# XXX   - run_gpath._get_gwas_data
# XXX   - run_esga load_gwas_data
# XXX   - run_glee.get_gesig - this is the best example of named extraction
# XXX     from multiple sources; see all access to the 'input_score' parm
# XXX   - run_gwasig.get_gwas_data

class Scoreboard(models.Model):
    def note_info(self,attr):
        if attr == 'note':
            return {
                'label':"Saved Scoreboard note %s (%d,ws %d)" %(
                        self.name,
                        self.id,
                        self.ws_id,
                        ),
                }
        raise Exception("bad note attr '%s'" % attr)
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    user = models.CharField(max_length=70)
    name = models.CharField(max_length=256)
    score_list = models.TextField()
    enables = models.TextField()
    sort = models.CharField(max_length=512,blank=True,default='')
    filters = models.CharField(max_length=512,blank=True,default='')
    preselecting = models.BooleanField(default=False)
    prescreen_extra_enables = models.TextField(blank=True,default='')
    note = models.ForeignKey(Note,null=True,blank=True, on_delete=models.CASCADE)
    def get_note_text(self):
        return Note.get(self,'note','')
    def get_primary_score(self):
        if self.sort:
            job_id,code = self.sort.split('_')
            if job_id[0] == '-':
                job_id = job_id[1:]
        else:
            for source in self.enables.split('|'):
                parts = source.split(':')
                if len(parts) > 1:
                    job_id = parts[0]
                    code = parts[1]
                    break
        job_id = int(job_id)
        return job_id,code
    def preselect_rank(self):
        job_id,code = self.get_primary_score()
        from runner.process_info import JobInfo
        bji = JobInfo.get_bound(self.ws,job_id)
        cat = bji.get_data_catalog()
        hidden_wsa_ids = self.ws.get_hidden_wsa_ids()
        ordering = [
                x
                for x in cat.get_ordering(code,True)
                if x[0] not in hidden_wsa_ids
                ]
        non_unk = self.ws.get_wsa_id_set('classified')
        from dtk.scores import get_ranked_groups
        for ahead,tied in get_ranked_groups(ordering):
            if set(tied) - non_unk:
                break
        return 1+ahead+len(tied)//2
    def job_count(self):
        if not self.score_list:
            return 0
        return len(self.score_list.split('|'))
    def enable_count(self):
        return sum([
                len(s.split(':'))-1
                for s in self.enables.split('|')
                ])

class Prescreen(models.Model):
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    created = models.DateTimeField(default=timezone.now)
    user = models.CharField(max_length=70)
    name = models.CharField(max_length=256)
    primary_score = models.CharField(max_length=512)
    filters = models.CharField(max_length=512,blank=True,default='')
    extra_scores = models.TextField(blank=True,default='')
    old_prescreen = models.ForeignKey(Scoreboard,null=True,blank=True, on_delete=models.CASCADE)
    def primary_job_id(self):
        return int(self.primary_score.split('_')[0])
    def primary_code(self):
        return self.primary_score.split('_')[1]
    def eff_jid(self):
        """Returns the underlying efficacy job id."""
        code = self.primary_code()
        if code == 'selectability' or code == 'liveselectability':
            # The marking wzs is whatever the selectability job used.
            from runner.process_info import JobInfo
            sel_bji = JobInfo.get_bound(self.ws, self.primary_job_id())
            return sel_bji.wzs_jid()
        else:
            return self.primary_job_id()
    def eff_code(self):
        code = self.primary_code()
        if code == 'selectability' or code == 'liveselectability':
            # These are always wzs right now; track and report if that changes.
            return 'wzs'
        else:
            return code
    def extra_score_jids_and_codes(self):
        extras = self.extra_scores.split('|')
        return [x.split(':') for x in extras if x]


    def source_list_job_ids(self,cache=None):
        # the optional cache parameter is a dict which holds the result for
        # a given prescreen across multiple calls; this can speed up the
        # formatting of long drug lists
        primary_id = self.eff_jid()
        if cache and primary_id in cache:
            return cache[primary_id]
        from runner.process_info import JobInfo
        bji = JobInfo.get_bound(self.ws,primary_id)
        job_ids = bji.get_all_input_job_ids()
        job_ids.add(primary_id)
        if cache is not None:
            cache[primary_id] = job_ids
        return job_ids
    def source_list_jobs(self,cache=None):
        job_ids = self.source_list_job_ids(cache)
        return '|'.join(str(x) for x in job_ids)
    def preselect_rank(self):
        if '_' not in self.primary_score:
            return -1
        job_id,code = self.primary_score.split('_')
        from runner.process_info import JobInfo
        bji = JobInfo.get_bound(self.ws,job_id)
        cat = bji.get_data_catalog()
        hidden_wsa_ids = self.ws.get_hidden_wsa_ids()
        try:
            ordering = [
                    x
                    for x in cat.get_ordering(code,True)
                    if x[0] not in hidden_wsa_ids
                    ]
        except ValueError:
            logger.error(f"Couldn't code {code} from {job_id}")
            return 0
        non_unk = self.ws.get_wsa_id_set('classified')
        from dtk.scores import get_ranked_groups
        for ahead,tied in get_ranked_groups(ordering):
            if set(tied) - non_unk:
                break
        return 1+ahead+len(tied)//2
    def marked_because(self):
        return "prescreen #%d %s"%(self.id,self.name)

    @classmethod
    def get_marking_prescreen_id(cls,marked_because):
        import re
        # Older prescreens didn't have a '#', new ones do.
        m = re.match(r'prescreen #?([0-9]+)',marked_because)
        if not m:
            return None
        prescreen_id = int(m.group(1))
        return prescreen_id
    @classmethod
    def get_marking_prescreen(cls,marked_because):
        prescreen_id = cls.get_marking_prescreen_id(marked_because)
        if prescreen_id is None:
            return None
        return cls.objects.get(pk=prescreen_id)

class PrescreenFilterAudit(models.Model):
    prescreen = models.ForeignKey(Prescreen, on_delete=models.CASCADE)
    saved_filters = models.TextField(blank=True,default='')
    user = models.CharField(max_length=50)
    timestamp = models.DateTimeField(auto_now=True)
    @classmethod
    def latest_filter(cls,prescreen_id):
        try:
            pfa=cls.objects.filter(
                    prescreen_id=prescreen_id,
                    ).order_by('-timestamp')[0]
        except IndexError:
            return ''
        return pfa.saved_filters

class ScoreSet(models.Model):
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    created = models.DateTimeField(default=timezone.now)
    user = models.CharField(max_length=70)
    desc = models.TextField()
    wf_job = models.IntegerField(null=True)
    sort_score = models.CharField(max_length=70,blank=True,default='')
    saved_filters = models.TextField(blank=True,default='')
    migrated_from = models.CharField(max_length=70,blank=True,default='')
    def job_type_to_id_map(self):
        return dict(self.scoresetjob_set.values_list('job_type','job_id'))
    def enable_count(self):
        return sum(len(x.parsed_enables) for x in self.scoresetjob_set.all())
    def get_contributing_workflow_jobs(self):
        '''Return a list of workflow jobs that built the scoreset.'''
        from runner.models import Process
        wf_job = Process.objects.get(pk=self.wf_job)
        return [wf_job]+[p
                for p in Process.objects.filter(
                        id__gt=wf_job.id,
                        name=wf_job.name,
                        )
                if p.settings().get('resume_scoreset_id') == str(self.id)
                ]
    def source_list_source_string(self):
        from dtk.scores import SourceList
        def source_string_part(ssj):
            result = str(ssj.job_id)
            if ssj.label:
                result += SourceList.field_sep + ssj.label
            return result
        return SourceList.rec_sep.join(
                source_string_part(x)
                for x in self.scoresetjob_set.all()
                )
    def source_list_enable_string(self):
        from dtk.scores import SourceList
        return SourceList.rec_sep.join(
                SourceList.field_sep.join([str(x.job_id)]+x.parsed_enables)
                for x in self.scoresetjob_set.all()
                if x.enables
                )
    def build_dep_graph(self):
        # Builds a dependency graph from a scoreset.
        from runner.process_info import JobInfo
        import networkx as nx
        g = nx.DiGraph()

        for jt, jid in self.job_type_to_id_map().items():
            g.add_node(int(jid))

        for jt, jid in self.job_type_to_id_map().items():
            bji = JobInfo.get_bound(self.ws, jid)
            for input_jid in bji.get_input_job_ids():
                g.add_edge(int(input_jid), int(jid))

        return g

    def get_dependents(self, jids):
        """Returns the set of jids dependent on the specified jids."""
        from networkx.algorithms.dag import descendants
        g = self.build_dep_graph()
        deps = set()
        for jid in jids:
            deps |= set(descendants(g, jid))
        return deps

    @transaction.atomic
    def clone(self, ssjob_filter=None):
        """ssjob_filter takes a ScoreSetJob and returns whether to keep it in the clone."""
        new_ss = ScoreSet(
                ws=self.ws,
                user=self.user,
                desc=self.desc,
                sort_score=self.sort_score,
                saved_filters=self.saved_filters,
                )
        new_ss.save()
        for ssjob in ScoreSetJob.objects.filter(scoreset=self):
            if ssjob_filter and not ssjob_filter(ssjob):
                continue
            new_ssjob = ScoreSetJob(
                    scoreset=new_ss,
                    job_id=ssjob.job_id,
                    job_type=ssjob.job_type,
                    label=ssjob.label,
                    enables=ssjob.enables,
                    )
            new_ssjob.save()
        return new_ss

class ScoreSetJob(models.Model):
    scoreset = models.ForeignKey(ScoreSet, on_delete=models.CASCADE)
    job_id = models.IntegerField()
    job_type = models.TextField()
    label = models.TextField(blank=True,default='')
    enables = models.TextField(blank=True,default='')
    @property
    def parsed_enables(self):
        if not self.enables:
            return []
        return self.enables.split(':')
    @parsed_enables.setter
    def parsed_enables(self,code_list):
        self.enables = ':'.join(sorted(set(code_list)))

class RunSet(models.Model):
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    user = models.CharField(max_length=70)
    job_type = models.CharField(max_length=70,blank=True,default='')
    desc = models.CharField(max_length=256)
    common_config = models.TextField(blank=True,default='')
    score_idx = models.IntegerField(null=True)
    wf_job = models.IntegerField(null=True)
    def job_type_to_id_map(self):
        return dict(self.runsetjob_set.values_list('job_type','job_id'))

class RunSetJob(models.Model):
    runset = models.ForeignKey(RunSet, on_delete=models.CASCADE)
    job_id = models.IntegerField()
    job_type = models.TextField()

class UserAccess(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    host = models.CharField(max_length=70)
    access = models.CharField(max_length=70)
    created = models.DateTimeField(auto_now_add=True)
    class Meta:
        unique_together = [['user', 'host', 'access']]
    def mapped_host(self):
        from dtk.known_ips import KnownIps
        return KnownIps.hostname(self.host)

class GwasFilter(models.Model):
    def note_info(self,attr):
        if attr == 'note':
            return {
                'label':"GWAS PMID Filter note on %d" %(
                        self.pubmed_id,
                        ),
                }
        raise Exception("bad note attr '%s'" % attr)
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    pubmed_id = models.IntegerField()
    # note that, in GwasDataset, rejected means that we exclude that
    # phenotype; here, it means that we've rejected the filter, i.e.
    # we DON'T exclude the PMID; sorry, but the view code was easier
    # if they had the same name
    rejected = models.BooleanField(default=False)
    note = models.ForeignKey(Note,null=True,blank=True, on_delete=models.CASCADE)

    class Meta:
        unique_together = [['ws', 'pubmed_id']]
    def get_note_text(self):
        return Note.get(self,'note','')

class StageStatus(models.Model):
    class Meta:
        index_together = [['stage_name']]
        unique_together= [['stage_name', 'ws']]

    statuses = Enum([],
            [ ('NONE',)
            , ('ACTIVE',)
            , ('COMPLETE',)
            , ('VERIFIED',)
            ])
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    stage_name = models.CharField(max_length=128)
    status = models.IntegerField(
                    default=statuses.NONE,
                    choices=statuses.choices(),
                    )
    changed_on = models.DateTimeField(auto_now=True)

class StageStatusLog(models.Model):
    stage = models.ForeignKey(StageStatus, on_delete=models.CASCADE)
    status = models.IntegerField()
    date = models.DateTimeField(auto_now_add=True)
    user = models.CharField(max_length=50)

class GwasDataset(models.Model):
    def note_info(self,attr):
        if attr == 'note':
            return {
                'label':"GWAS dataset note on %s %s" %(
                        self.phenotype,
                        self.pubmed_id,
                        ),
                }
        raise Exception("bad note attr '%s'" % attr)
    ws = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    phenotype = models.CharField(max_length=1024)
    pubmed_id = models.CharField(max_length=256)
    rejected = models.BooleanField(default=False)
    note = models.ForeignKey(Note,null=True,blank=True, on_delete=models.CASCADE)
    # Only include variants whose study->variant p-values more significant than this.
    v2d_threshold = models.FloatField(default=1.0)

    # This should be unique on these fields, but MySQL has a limit on
    # index sizes which prevents us from declaring this.
    # Seems unlikely to hit an issue here, ignoring for now.
    #class Meta:
    #    unique_together = [['ws', 'phenotype', 'pubmed_id']]



    def get_note_text(self):
        return Note.get(self,'note','')
    def make_key(self):
        return '%s|%s' % (self.phenotype,self.pubmed_id)
    def make_path(self):
        import path_helper as ph
        dirname = ph.PathHelper.storage+'%d/gwas/' % self.ws_id
        ph.make_directory(dirname)
        return dirname+'gwds%d.tsv' % self.id
    def make_plot_path_prefix(self):
        import path_helper as ph
        dirname = os.path.join(
                ph.PathHelper.publish,
                str(self.ws_id),
                'gwas',
                )
        ph.make_directory(dirname)
        return os.path.join(dirname,'gwds'+str(self.id))
    def delete_data(self):
        out_fn = self.make_path()
        if os.path.exists(out_fn):
            os.remove(out_fn)
    @classmethod
    def get_master_S3File(self, ws):
        from browse.default_settings import duma_gwas_v2d
        return duma_gwas_v2d.get_s3_file(ws=ws, role='archive')
    
    def extract_data(self):
        from browse.default_settings import duma_gwas_v2d, duma_gwas_v2g
        v2d_fn = duma_gwas_v2d.get_s3_file(ws=self.ws, role='archive').path()
        v2g_fn = duma_gwas_v2g.get_s3_file(ws=self.ws, role='archive').path()

        out_fn = self.make_path()
        key = self.make_key()

        metadata = {
            'format_version': 1,
            'v2d': duma_gwas_v2d.value(self.ws),
            'v2g': duma_gwas_v2g.value(self.ws),
        }

        from dtk.gwas import extract_gwas_key_new
        from atomicwrites import atomic_write
        import json
        with atomic_write(out_fn, overwrite=True) as outf:
            outf.write('#' + json.dumps(metadata) + '\n')
            for line in extract_gwas_key_new(v2d_fn, v2g_fn, key):
                outf.write(line + '\n')

    def extract_data_old(self):
        from dtk.gwas import extract_gwas_key_old
        from browse.default_settings import duma_gwas
        s3_file = duma_gwas.get_s3_file(ws=self.ws, role='archivedata')
        out_fn = self.make_path()
        key = self.make_key()

        from atomicwrites import atomic_write
        with atomic_write(out_fn, overwrite=True) as outf:
            for line in extract_gwas_key_old(s3_file.path(), key):
                outf.write(line + '\n')

    from collections import namedtuple
    GwasResultRec = namedtuple(
            'GwasResultRec',
            'study_key snp chromosome base evidence uniprot v2g_evidence',
            )
    def load_study_data(self):
        self.get_study_data()
        self.num_variants = self.study_data[3]
        self.num_prots = self.study_data[4]
        self.total_samples = self.study_data[6]
        self.ancestry = self.study_data[7]
        self.chip_type = self.study_data[8]
        self.pub_date = self.study_data[9]
    def check_study_data(self):
        from dtk.gwas import search_gwas_studies_file
        key = self.make_key()
        self.matches = search_gwas_studies_file(self.ws, [key], exact=True)
        if len(self.matches) == 0 and key != key.lower():
            # See KeyError handler in extract_data about why we don't just lower() everything.
            key = key.lower()
            self.matches = search_gwas_studies_file(self.ws, [key], exact=True)

        return len(self.matches)
    def get_study_data(self):
        n=self.check_study_data()
        if n != 1:
            self.study_data = ['NA']*5+['Key matching error: '+ str(n)+' matches'] +['NA']*4
        else:
            self.study_data = self.matches[0]
    def get_data(self):
        in_fn = self.make_path()
        if not os.path.exists(in_fn):
            self.extract_data()
        from dtk.files import get_file_records
        meta = None
        for rec in get_file_records(in_fn, keep_header=True):
            if meta is None:
                # Newer data starts with a '#' header line with some json metadata.
                if rec[0].startswith('#'):
                    import json
                    meta = json.loads(rec[0][1:])
                    version = meta.get('format_version', 0)
                    continue
                else:
                    # Older data doesn't have any header, treat this line as real data.
                    meta = {}
                    version = 0

            if version == 0:
                # In the original gwds versions we don't have any v2g values, just hardcode to 1.0.
                v2g = 1
            else:
                v2g = float(rec[6])

            yield self.GwasResultRec(
                    rec[0],
                    rec[1],
                    rec[2],
                    rec[3],
                    float(rec[4]),
                    rec[5],
                    v2g
                    )

################################################################################
# utilities
################################################################################
def fix_fm_codes():
    from runner.models import Process
    qs=Process.objects.filter(settings_json__contains='fm_code')
    from runner.process_info import JobInfo
    from dtk.text import diffstr
    for p in qs:
        fm_code = p.settings()['fm_code']
        src_job_id = JobInfo.extract_feature_matrix_job_id(fm_code)
        src_p = Process.objects.get(pk=src_job_id)
        new_code = src_p.name.split('_')[0]+str(src_job_id)
        if new_code != fm_code:
            print('mismatch',p.id,p.name,src_p.name,new_code,fm_code)
            patch = p.settings_json.replace(fm_code,new_code)
            print(diffstr(p.settings_json,patch))
            p.settings_json = patch
            p.save()

def clean_removed_scoreboard_plugins():
    '''Remove non-existent plugins from saved scoreboards.

    If a plugin is removed from the system, attempting to access old scores
    for that plugin will throw an InportError from JobInfo.get_unbound().
    Since the plugin also disappears from the run menu, the only way to
    stumble over one of these scores is if it's in the source list for an
    active session (in which case, just log out and back in), or in a saved
    scoreboard.  This code tries to update scoreboards to eliminate this
    last case.
    '''
    from runner.process_info import JobInfo
    from runner.models import Process
    import re
    for sb in Scoreboard.objects.all():
        rm_list = set()
        if sb.score_list:
            src_jobs = [
                    Process.objects.get(pk=entry.split(':')[0])
                    for entry in sb.score_list.split('|')
                    ]
            for job in src_jobs:
                try:
                    bji = JobInfo.get_bound(sb.ws,job)
                except ImportError:
                    rm_list.add(job)
        if rm_list:
            print('sb %d, jobs %s' % (
                    sb.id,
                    ' '.join(['%d (%s)'%(x.id,x.name) for x in rm_list]),
                    ))
            rm_list = [x.id for x in rm_list]
            # verify sort column isn't affected
            sort_job_id,_ = sb.get_primary_score()
            assert sort_job_id not in rm_list
            # clean score_list
            sb.score_list = '|'.join([
                    x
                    for x in sb.score_list.split('|')
                    if int(x.split(':')[0]) not in rm_list
                    ])
            # clean enables
            sb.enables = '|'.join([
                    x
                    for x in sb.enables.split('|')
                    if int(x.split(':')[0]) not in rm_list
                    ])
            sb.save()

def migrate_saved_scoreboard_sorts():
    '''Convert from legacy sort formats.

    Originally, the only things sorted were scores, and the assumption
    was that they were in descending order by default.  The dtk.table
    code uses the more standard convention that a - prefix indicates
    a descending sort.  The Scoreboard table is the only place where
    the codes are stored persistently.  Convert legacy codes here.

    This relies on the fact that no ascending sorts were stored prior
    to the conversion.  This will prevent accidentally toggling back
    to the legacy format by re-running this function.
    '''
    for sb in Scoreboard.objects.all():
        if sb.sort:
            assert not sb.sort.startswith('-')
            sb.sort = '-'+sb.sort
            sb.save()

def migrate_ae_rejects():
    for acc in AeAccession.objects.all():
        if acc.old_rejected:
            if not acc.aedisposition_set.exists():
                for mode in [x[0] for x in AeSearch.mode_vals.choices()]:
                    disp = AeDisposition()
                    disp.accession = acc
                    disp.mode = mode
                    disp.rejected = acc.old_rejected
                    disp.save()

def group_adjacent_common_keys(iterable):
    last = None
    for key,attr,val in iterable:
        if key != last:
            if last is not None:
                yield (last,result)
            result = {}
            last = key
        s=result.setdefault(attr,set())
        s.add(val)
    if last is not None:
        yield (last,result)

def import_protein_names(choice):
    import json
    import gzip
    uniprot_to_info = {}
    print("Parsing names file")
    from dtk.s3_cache import S3File
    f = S3File.get_versioned(
            file_class='uniprot',
            choice=choice,
            role='Protein_Names',
            )
    f.fetch()
    fn=f.path()
    with gzip.open(fn, 'rb') as f:
        data = json.loads(f.read())
        for entry in data:
            for uniprot in entry['uniprots']:
                uniprot_to_info[uniprot] = entry

    print("Inserting name attributes")
    main_name_attr,new = ProteinAttributeType.objects.get_or_create(name="Protein_Name")
    alt_name_attr,new = ProteinAttributeType.objects.get_or_create(name="Alt_Protein_Name")
    from browse.models import Protein
    missing = set()
    buf = []
    total = 0
    for p in Protein.objects.all():
        info = uniprot_to_info.get(p.uniprot, None)
        if not info:
            missing.add(p)
            continue

        pa = ProteinAttribute()
        pa.prot = p
        pa.attr = main_name_attr
        # DB limit is 100, very spammy if we try to exceed
        pa.val = info['full_name'][:100]
        buf.append(pa)
        for alt_name in info['alt_names']:
            pa = ProteinAttribute()
            pa.prot = p
            pa.attr = alt_name_attr
            pa.val = alt_name[:100]
            buf.append(pa)

        if len(buf) >= 10000:
            total += len(buf)
            print('creating',len(buf),'attribute records; total',total)
            ProteinAttribute.objects.bulk_create(buf)
            buf = []

    if buf:
        total += len(buf)
        print('creating',len(buf),'attribute records; total',total)
        ProteinAttribute.objects.bulk_create(buf)
    print("Imported names, missing %d/%d uniprots" % (len(missing), len(Protein.objects.all())))


@transaction.atomic
def import_proteins(choice):
    """

    Imports a new set of protein data.
    All old attributes and proteins will be removed.

    We will attempt to remap old protein sets and old protein notes.
    Any protein in a set or note that is now an alt uniprot will be
    moved over to the new primary protein.
    """

    # retrieve main input file
    from dtk.s3_cache import S3File
    s3f = S3File.get_versioned(
            file_class='uniprot',
            choice=choice,
            role='Uniprot_data',
            )
    s3f.fetch()
    filepath=s3f.path()
    # record start of upload in status table (ok initially false)
    import os
    dirname,filename = os.path.split(filepath)
    from browse.models import import_proteins,ProteinUploadStatus
    stat = ProteinUploadStatus(filename=filename, ok=False)
    stat.save()
    # build uniprot-to-id map from existing records
    name2prot = {}
    for p in Protein.objects.all():
        name2prot[p.uniprot] = p
    print(len(name2prot),'existing protein records')
    unseen = set(name2prot.keys())
    # Pre-load AttributeTypes
    fixed = ['Gene_Name','UniProtKB-ID']
    attrs = [
            "KEGG",
            "MIM",
            "STRING",
            "Ensembl_PRO",
            "Gene_Synonym",
            "Ensembl_TRS",
            "GeneID",
            "Alt_uniprot",
            "Ensembl",
            "Reactome",
            "GeneCards",
            "hgnc",
            "Alt_UniProtKB-ID",
            ]
    attr2obj = {}
    for attr in attrs:
        obj,new = ProteinAttributeType.objects.get_or_create(name=attr)
        attr2obj[attr] = obj
        if new:
            print("created new attribute type",attr)
    # now scan file
    from dtk.files import get_file_records
    src = get_file_records(s3f.path(),keep_header=False)
    buf = []
    total = 0
    for uniprot,data in group_adjacent_common_keys(src):
        unseen.discard(uniprot)
        gene = data.get(fixed[0],set(['']))
        assert len(gene) == 1
        gene = gene.pop()
        kb = data.get(fixed[1],set(['']))
        assert len(kb) == 1
        kb = kb.pop()
        try:
            p = name2prot[uniprot]
            write = False
            if p.gene != gene:
                print('updating',uniprot,'gene from',p.gene,'to',gene)
                p.gene = gene
                write = True
            if p.uniprot_kb != kb:
                p.uniprot_kb = kb
                write = True
            if write:
                p.save()
        except KeyError:
            p = Protein()
            p.uniprot = uniprot
            p.gene = gene
            p.uniprot_kb = kb
            buf.append(p)
            if len(buf) >= 1000:
                total += len(buf)
                print('creating',len(buf),'protein records; total',total)
                Protein.objects.bulk_create(buf)
                buf = []
    if buf:
        Protein.objects.bulk_create(buf)
        total += len(buf)
        print('creating',len(buf),'protein records; total',total)
    used = set(ProtSet.objects.values_list('proteins__uniprot',flat=True))
    print(len(unseen),'unseen',len(unseen & used),'used')
    # now make a second pass to set attributes
    name2prot = {}
    for p in Protein.objects.all():
        name2prot[p.uniprot] = p
    print(len(name2prot),'existing protein records')
    # delete and re-load all attribute values for simplicity
    qs=ProteinAttribute.objects.all()
    print('deleting',qs.count(),'old attributes')
    qs.delete()
    print('completed attribute delete')
    src = get_file_records(s3f.path(),keep_header=False)
    buf = []
    total = 0
    for uniprot,data in group_adjacent_common_keys(src):
        prot = name2prot[uniprot]
        for k,v in data.items():
            if k in fixed:
                continue
            for val in v:
                pa = ProteinAttribute()
                pa.prot = prot
                pa.attr = attr2obj[k]
                pa.val = val
                buf.append(pa)
                if len(buf) >= 10000:
                    total += len(buf)
                    print('creating',len(buf),'attribute records; total',total)
                    ProteinAttribute.objects.bulk_create(buf)
                    buf = []
    if buf:
        total += len(buf)
        print('creating',len(buf),'attribute records; total',total)
        ProteinAttribute.objects.bulk_create(buf)
    import_protein_names(choice)
    # attempt to re-map used, unseen proteins before deletion
    for uniprot in unseen & used:
        base = list(Protein.objects.filter(
            proteinattribute__attr=attr2obj['Alt_uniprot'],
            proteinattribute__val=uniprot,
            ))
        if len(base) != 1:
            print(uniprot,'found',base)
        for ps in ProtSet.objects.filter(proteins__uniprot=uniprot):
            for p in base:
                ps.proteins.add(p)
                print('adding %s to %s ws %d for %s' % (
                        p.uniprot,
                        ps.name,
                        ps.ws_id,
                        uniprot,
                        ))

    # attempt to re-map proteins with protein notes.
    note_prots = set(TargetAnnotation.objects.all().values_list('uniprot', flat=True).distinct())
    changed_note_prots = unseen & note_prots
    logger.info(f"{len(changed_note_prots)} prots with notes have changed")
    for uniprot in changed_note_prots:
        base = list(Protein.objects.filter(
            proteinattribute__attr=attr2obj['Alt_uniprot'],
            proteinattribute__val=uniprot,
            ).values_list('uniprot', flat=True))
        assert len(base) == 1, f"No mapping for note prot {uniprot}"
        base = base[0]
        logger.info(f"Updating all notes on {uniprot} to {base}")

        for target_ann in TargetAnnotation.objects.filter(uniprot=uniprot):
            conflicts = TargetAnnotation.objects.filter(uniprot=base, ws=target_ann.ws)
            if len(conflicts) > 0:
                logger.warning(f'Already have an annotation for the new uniprot, just move over reviews')
                other_ann = conflicts[0]
                for target_review in TargetReview.objects.filter(target=target_ann):
                    user = target_review.user
                    rvw_conflicts = TargetReview.objects.filter(target=other_ann, user=user)
                    if len(rvw_conflicts) > 0:
                        # This target review is also conflicting...
                        rvw_conflict = rvw_conflicts[0]
                        combined = f"(merged from {uniprot}): " + target_review.get_note_text(user) + "\n(original):" + rvw_conflict.get_note_text(user)
                        TargetReview.save_note(ws=target_ann.ws, uniprot=base, user=target_review.user, text=combined)
                        logger.warning(f'Merging two prot notes: {combined}')
                    else:
                        target_review.target = other_ann
                        target_review.save()
            else:
                target_ann.uniprot = base
                target_ann.save()
    # delete any proteins not in new file
    qs=Protein.objects.filter(uniprot__in=unseen)
    print('deleting',qs.count(),'unseen proteins')
    qs.delete()
    # mark upload successful
    stat.ok=True
    stat.save()

def scan_collection_use(do_migrate=False):
    from drugs.models import Collection
    # for upgrade:
    # - if the workspace uses minus_drugbank
    #   - if it also uses full, handle remove case (or do this by hand)
    # - migrate minus -> full
    # for each collection, list the workspaces where it's used
    ws_by_coll = {}
    coll_name_to_id = {}
    for coll in Collection.objects.all():
        has = set()
        ws_by_coll[coll.name] = has
        coll_name_to_id[coll.name] = coll.id
        print(coll.name)
        for ws in Workspace.objects.all():
            cnt = ws.wsannotation_set.filter(agent__collection=coll).count()
            if cnt:
                print('  ',cnt,ws.name)
                has.add(ws.id)
    from_name = 'ttd.minus_drugbank'
    to_name = 'ttd.full'
    from_id = coll_name_to_id[from_name]
    to_id = coll_name_to_id[to_name]
    ws_set = ws_by_coll[from_name] - ws_by_coll[to_name]
    for ws_id in ws_set:
        if do_migrate:
            migrate_collection(ws_id,from_id,to_id)
        else:
            print('would migrate ws_id',ws_id)

def migrate_collection(ws_id=1,from_id=4,to_id=2):
    from drugs.models import Tag,Prop,Collection
    from_coll = Collection.objects.get(pk=from_id)
    to_coll = Collection.objects.get(pk=to_id)
    print('migrating ws_id',ws_id,'from',from_coll.name,'to',to_coll.name)
    assert from_coll.key_name == to_coll.key_name
    key_prop = Prop.get(from_coll.key_name)
    # build a map from the key to the new drug_id
    key2new_id = {t.value:t.drug_id
                for t in Tag.objects.filter(
                                    prop=key_prop,
                                    drug__collection=to_coll,
                                    )
                }
    print(len(key2new_id),'new ids for collection keys')
    # use it to build a map from the old id to the new id
    old2new = {}
    for t in Tag.objects.filter(
                        prop=key_prop,
                        drug__collection=from_coll,
                        ):
        if t.value in key2new_id:
            old2new[t.drug_id] = key2new_id[t.value]
    print(len(old2new),'old ids can be mapped')
    # now scan WSA table; if agent in map, update
    updated = 0
    skipped = 0
    for wsa in WsAnnotation.objects.filter(ws_id=ws_id):
        if wsa.agent_id in old2new:
            wsa.agent_id = old2new[wsa.agent_id]
            wsa.save()
            updated += 1
        else:
            skipped += 1
        if (updated+skipped) % 1000 == 0:
            print('updated',updated,'skipped',skipped)
    print('updated',updated,'skipped',skipped)


class WorkflowJob(models.Model):
    """Associates all jobs that are started by a workflow.

    Unlike scoresets, this includes in-progress and failed jobs.
    """
    wf_job = models.ForeignKey(Process,on_delete=models.CASCADE, related_name='wf_job')
    child_job = models.ForeignKey(Process,on_delete=models.CASCADE, related_name='child_job')
