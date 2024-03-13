
# In support of PLAT-3338, we'd like additional tools for gathering stats on
# drug collections to comparing one collection with another, and to quantify
# how a single collection evolves over time. Specifically, we'd like to measure
# size and degree of overlap for the collections as wholes, and for subsets of
# molecules within those collections.
#
# It's useful to be able to do this for collections on the platform and for
# experimental extractions in the ETL directory.
#
# This class is an attempt to abstract out the idea of partially overlapping
# collections of items with attributes, where some attributes are global and
# others are collection-specific. By separating this from any concept of
# molecules, and how molecule information is stored, we can uncouple:
# - code which populates this class from different ways that molecule data
#   is stored
# - code which gathers statistics and creates visualizations around the
#   relative structure of collections
# The separation should also allow us to re-use visualizations and statistics
# gathering for viewing our compound libraries as collections of MOAs or of
# structural cores, as well as collections of molecules.
#
# The basic theory of operation is:
# - supports multiple named collections
# - to allow determination of when an item in one collection is 'the same' as
#   an item in another, items are represented as keys. For our molecule
#   collections, dpimerge_ids are an appropriate key to use.
# - items can have attributes, either on a global level or on a per-collection
#   level. For example, a global attribute might classify a molecule as
#   'commercially available' or 'clinically investigated', and a per-collection
#   attribute might specify if DPI information on a molecule was available in
#   a specific collection.
# - arbitrarily complex filtering on attributes can be represented as lambda
#   functions. These functions take a namedtuple of attributes for a key,
#   and return a boolean indicating whether the attributes match the filter.
#   The namedtuple has data members for each global or local attribute defined
#   for any key in the collection; if the attribute is not defined for a
#   particular key, its value will be None.
# - collections and filtered collection subsets can be extracted as sets of
#   keys; set operations can determine overlap, and size of the sets can be
#   used to produce comparable statistics
class CollStats:
    class Collection:
        def __init__(self):
            self.keys = set()
            self.attr_names = set()
            self.attr_vals = {}
        def yield_vecs(self,global_names,global_vals):
            attr_keys = sorted(self.attr_names|global_names)
            from collections import namedtuple
            VecType = namedtuple('Attrs',attr_keys)
            for key in self.keys:
                d = self.attr_vals.get(key,{})
                g = global_vals.get(key,{})
                yield key,VecType(*[
                        d.get(name,g.get(name))
                        for name in attr_keys
                        ])
        def dump(self):
            print('keys:',self.keys)
            print('attr_names:',self.attr_names)
            print('attr_vals:',self.attr_vals)
    def __init__(self):
        self.collections = {}
        self.global_attr_names = set()
        self.global_attr_vals = {}
    def _get_or_create(self,name):
        return self.collections.setdefault(name,self.Collection())
    def keys(self,name):
        '''Return all keys in the named collection.'''
        return self.collections[name].keys
    def filter(self,name,func):
        '''Return keys in the named collection that match func.'''
        c = self.collections[name]
        return set(
                key
                for key,vec in c.yield_vecs(
                        self.global_attr_names,
                        self.global_attr_vals,
                        )
                if func(vec)
                )
    def add_keys(self,name,key_iter):
        '''Add keys to a collection without adding attributes.'''
        c = self._get_or_create(name)
        c.keys |= set(key_iter)
    def add_attrs(self,name,attr_names,key_attr_iter):
        '''Add keys and attributes to a collection.

        key_attr_iter is an iterator over (key,vec) tuples,
        where vec is a tuple of attribute values in the same
        order as attr_names.
        '''
        c = self._get_or_create(name)
        c.attr_names |= set(attr_names)
        for key,vec in key_attr_iter:
            c.keys.add(key)
            d = c.attr_vals.setdefault(key,dict())
            d.update(zip(attr_names,vec))
    def _load_shared_model_data(self):
        from drugs.models import UploadAudit
        status = UploadAudit.cluster_status()
        if not status:
            self.clust_ver = None
        else:
            self.clust_ver,ts,ok = status
            assert ok
        from browse.models import WsAnnotation
        qs = WsAnnotation.objects.all()
        enum = WsAnnotation.indication_vals
        for indication in (
                enum.UNCLASSIFIED,
                enum.INACTIVE_PREDICTION,
                ):
            qs = qs.exclude(indication=indication)
        from dtk.data import MultiMap
        si_agent2collkey_mm = MultiMap(
                qs.values_list('agent_id','agent__collection__key_name')
                )
        from drugs.models import DpiMergeKey,Tag
        si_agent2mergekey = dict(DpiMergeKey.objects.filter(
                drug_id__in=si_agent2collkey_mm.fwd_map().keys(),
                version=self.clust_ver,
                ).values_list('drug_id','dpimerge_key'))
        self.si_keys = set()
        for collkey,agents in si_agent2collkey_mm.rev_map().items():
            for agent,native_key in Tag.objects.filter(
                    drug_id__in=agents,
                    prop__name=collkey,
                    ).values_list('drug_id','value'):
                self.si_keys.add(si_agent2mergekey.get(agent,native_key))
    def add_from_model(self,coll,prop_names):
        from drugs.models import UploadAudit,DpiMergeKey,Tag
        coll_ver,ts,ok = UploadAudit.collection_status(coll.name)
        assert ok
        try:
            clust_ver = self.clust_ver
        except AttributeError:
            self._load_shared_model_data()
            clust_ver = self.clust_ver
        # get key data
        # - first load map for clusters
        agent2mergekey = dict(DpiMergeKey.objects.filter(
                drug__collection=coll,
                drug__removed=False,
                version=clust_ver,
                ).values_list('drug_id','dpimerge_key'))
        # - then build composite map (cluster key if available,
        #   falling back to native key if not)
        agent2key = {
                agent:agent2mergekey.get(agent,key)
                for agent,key in Tag.objects.filter(
                        drug__collection=coll,
                        drug__removed=False,
                        prop__name=coll.key_name,
                        ).values_list('drug_id','value')
                }
        # extract attribute data
        from drugs.models import Prop
        from dtk.data import MultiMap
        type2prop_mm = MultiMap(
                (p.prop_type,p)
                for p in Prop.objects.filter(name__in=prop_names)
                )
        def get_prop_agent_pairs():
            for type_code,props in type2prop_mm.fwd_map().items():
                PropClass=Prop.cls_from_type(type_code)
                for prop_name,agent in PropClass.objects.filter(
                        drug__collection=coll,
                        prop__name__in=prop_names,
                        ).values_list('prop__name','drug_id'):
                    yield (prop_name,agent)
        prop2agent_mm = MultiMap(get_prop_agent_pairs())
        attr_names = [
                'super_important',
                ]+['with_'+x for x in prop_names]
        self.add_attrs(coll.name,attr_names,(
                (key,[
                        key in self.si_keys,
                        ]+[
                            agent in prop2agent_mm.fwd_map().get(prop,set())
                            for prop in prop_names
                        ])
                for agent,key in agent2key.items())
                )
    def add_global_attrs(self,attr_names,key_attr_iter):
        '''Add global attributes for keys.

        key_attr_iter and attr_names are as in add_attrs().
        '''
        self.global_attr_names |= set(attr_names)
        for key,vec in key_attr_iter:
            d = self.global_attr_vals.setdefault(key,dict())
            d.update(zip(attr_names,vec))
    def get_key_coll_pairs(self):
        '''Return (key,collname) pairs for all keys in all collections.'''
        for name,c in self.collections.items():
            for key in c.keys:
                yield (key,name)
    def get_filtered_key_coll_pairs(self,filter_func):
        '''Return (key,collname) pairs for keys matching filter_func.'''
        for name in self.collections:
            for key in self.filter(name,filter_func):
                yield (key,name)
    def get_key2coll_mm(self,filter_func=None):
        '''Return a MultiMap from keys to collection names.'''
        if filter_func:
            src = self.get_filtered_key_coll_pairs(filter_func)
        else:
            src = self.get_key_coll_pairs()
        from dtk.data import MultiMap
        return MultiMap(src)
    def get_collset2key_mm(self,filter_func=None):
        '''Return a MultiMap from collection name combinations to keys.

        A collection name combination is a string of space-separated
        collection names in sorted order. It corresponds to a single
        cell in a Venn diagram of which keys belong to which collections.
        Although there are many keys per collection name combo, there
        is only one collection name combo per key (i.e. all the sets
        in result.rev_map().values() will have exactly one member).
        '''
        key2coll_mm = self.get_key2coll_mm(filter_func)
        from dtk.data import MultiMap
        return MultiMap(
                (' '.join(sorted(s)), k)
                for k,s in key2coll_mm.fwd_map().items()
                )

