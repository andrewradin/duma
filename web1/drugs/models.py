from django.db import models,transaction
from django.db.models import Prefetch
from tools import Enum
from notes.models import Note
from threading import Lock

import logging
logger = logging.getLogger(__name__)

# XXX Future enhancements:
# - convert 'agent' to non-nullable field
# - add audit fields on attributes?
# - add optional URL template to Prop class? (or use a django template?)
# - flag non-public Collections; restrict access?
# - add a PropSet class with a m2m relationship with Prop (so things like
#   which columns to match in the review search window can be configured
#   dynamically)?

class AttributeCacheLoader:
    @staticmethod
    def _nested_query(qs,path_to_drug,prop_type,prop_names=None):
        cls = Prop.cls_from_type(prop_type)
        id_label = 'id'
        if path_to_drug:
            id_label = path_to_drug+'_'+id_label

        drug_ids = qs.values_list(id_label, flat=True)
            
        kwargs = {}
        if prop_names:
            kwargs['prop__name__in'] = prop_names
        return cls.objects.filter(
                drug_id__in=drug_ids,
                **kwargs,
                ).values_list('drug_id','prop_id','value')
    def __init__(self,qs,path_to_drug,prop_type,prop_names=None):
        from collections import defaultdict
        self.cache_attr = Prop._cache_attr(prop_type)
        self.val_lists = defaultdict(list)
        src = self._nested_query(qs,path_to_drug,prop_type,prop_names)
        for drug_id,prop_id,value in src:
            self.val_lists[drug_id].append((prop_id,value))
    def load_cache(self,drug):
        setattr(drug,self.cache_attr,self.val_lists.get(drug.id,[]))

# Create your models here.
class Collection(models.Model):
    name = models.CharField(max_length=256,default="")
    key_name = models.CharField(max_length=256,default="")
    class Meta:
        unique_together = [['name', 'key_name']]
    foreign_key_prefix = 'm_'
    def __str__(self):
        return self.name
    default_collections = [
            'duma.full',
            'drugbank.full',
            'ncats.full',
            'chembl.adme_condensed',
            'pubchem.filtered',
            'bindingdb.full_condensed',
            'globaldata.full',
            'moa.full',
            'lincs.full',
            ]
    @classmethod
    def import_ordering(cls):
        partial_ordering = [
                'duma.full',
                'drugbank.full',
                'ncats.full',
                'chembl.adme_condensed',
                'bindingdb.full_condensed',
                'chembl.adme',
                'chembl.full',
                ]
        d = {x.name:x for x in cls.objects.all()}
        result = []
        for name in partial_ordering:
            if name in d:
                result.append(d[name])
                del d[name]
        result += sorted(d.values(),key=lambda x:x.name)
        return result
    @classmethod
    def get(cls,name):
        return cls.objects.get(name=name)
    @classmethod
    def get_or_create(cls,name,key_name):
        with transaction.atomic():
            c,new = cls.objects.get_or_create(name=name,key_name=key_name)
            if new:
                # create a property for the foreign key
                Prop.get_or_create(c.key_name
                                    ,Prop.prop_types.TAG
                                    )
                Prop.get_or_create(Prop.NATIVE_ID
                                    ,Prop.prop_types.TAG
                                    )
                Prop.get_or_create(cls.foreign_key_prefix+c.key_name
                                    ,Prop.prop_types.TAG
                                    ,multival=True
                                    )
            return c
    def get_or_create_drug(self,key,name=None):
        key_prop = Prop.get(self.key_name)
        if name is None:
            # just get, don't create
            return Drug.objects.get(collection=self
                                    ,tag__prop=key_prop
                                    ,tag__value=key
                                    )
        with transaction.atomic():
            d,new = Drug.objects.get_or_create(collection=self
                                        ,tag__prop=key_prop
                                        ,tag__value=key
                                        )
            if new:
                # need to set name and id
                d.set_prop(Prop.NAME,name)
                d.set_prop(key_prop,key)
                d.set_prop(Prop.NATIVE_ID,key)
            return d
    def static_properties_list(self):
        '''Return a list of all properties that should be cleared on reload.

        Basically, this is all the properties that we expect to be loaded
        from the creation file, and not subsequently changed.  Details of
        what doesn't qualify are in the code below.
        '''
        result = []
        props_to_keep = {
            # needed match drug_id to file key
            self.key_name,
            # Same with native ID.
            Prop.NATIVE_ID,
            Prop.NAME,
            'synthesis_protection',
            Prop.OVERRIDE_NAME,
            }
        for p in Prop.objects.all():
            if p.name in props_to_keep:
                continue
            if p.name.startswith(self.foreign_key_prefix):
                # these come from patch file; not restored by create
                continue
            result.append(p)
        return result
    def clear_properties(self,prop_list):
        '''Clear rhe specified properties for all drugs in the collection.
        '''
        for i in range(0,len(Prop.prop_types.members)):
            cls = Prop.cls_from_type(i)
            qs = cls.objects.filter(prop__in=prop_list,drug__collection=self)
            logger.debug("clearing %d %s records from %s"
                        ,qs.count()
                        ,Prop._type_label(i)
                        ,self.name
                        )
            qs.delete()
    @classmethod
    def fast_create_from_records(cls,inp,src_label,coll_name,key_name):
        logger.info(f"starting fast create for {coll_name} from {src_label}")
        c = cls.get_or_create(coll_name,key_name)
        # due to the 'unsafe' creation approach for the name and id props
        # in fast_load_from_records, in the event of an error, we can end
        # up with drug reccords with missing key and canonical fields; get
        # rid of these here, before attempting a reload
        for propname in (Prop.NAME,c.key_name):
            Drug.objects.filter(collection=c).exclude(
                                tag__prop__name=propname
                                ).delete()
        # get properties to preserve
        load_props = c.static_properties_list()
        orphans = cls.fast_load_from_records(src_label,c,load_props,inp)
        orphans_to_remove = Drug.objects.filter(id__in=orphans, removed=False)
        logger.debug("import returned %d drug ids no longer used"
                    ,len(orphans_to_remove)
                    )
        # mark all drugs no longer present in create file that aren't already
        for drug in orphans_to_remove:
            key = getattr(drug,c.key_name)
            name = getattr(drug,Prop.NAME)
            logger.info("drug_id %d '%s' '%s' removed",drug.id,key,name)
            drug.removed=True
            drug.save()
        logger.info(f"fast create from {src_label} complete")

    @classmethod
    def fast_load_from_records(cls,src_label,c,load_props,inp):
        # This does a fast load from an attr/val file, based on the idea that
        # the file contains all the values used in collection 'c' for the
        # properties listed in 'load_props'.  This allows us to pre-delete
        # all those value records and bulk-reload them, without needing to
        # get_or_create each individual value.
        header = next(inp)
        if 'attribute' not in header:
            raise Exception("only attr/val is supported")
        if 'value' not in header:
            raise Exception("attribute column without value column")
        has_href = ('href' in header)
        if len(header) != (4 if has_href else 3):
            raise Exception("wrong number of columns")
        key_name = c.key_name
        if key_name not in header:
            raise Exception("key column doesn't match collection name")
        c.clear_properties(load_props)
        load_prop_names = set([p.name for p in load_props])
        key2id = {}
        p = Prop.get(key_name)
        for v in p.cls().objects.filter(drug__collection=c,prop=p):
            key2id[v.value] = v.drug_id
        orphans = set(key2id.values())
        logger.debug("import pre-loaded %d drug records",len(key2id))
        id2name = {}
        p = Prop.get(Prop.NAME)
        for v in p.cls().objects.filter(drug__collection=c,prop=p):
            id2name[v.drug_id] = v.value
        native_prop = Prop.get(Prop.NATIVE_ID)
        queues = {}
        from dtk.readtext import convert_records_using_header
        matched_existing_drug_ids = set()
        for i,rec in enumerate(convert_records_using_header(inp,header)):
            if (i+1) % 2000 == 0:
                logger.debug("import scanning line %d from %s"
                                ,i+1
                                ,src_label
                                )
            key = getattr(rec,key_name)
            attr = rec.attribute.lower()
            if attr == Prop.NAME:


                # on 'canonical' records (which must be the first record
                # for each key), get drug record and add it to key2id hash
                if key in key2id:
                    # this is an existing drug; verify no name change
                    drug_id = key2id[key]
                    orphans.remove(drug_id)
                    matched_existing_drug_ids.add(drug_id)
                    old_name = id2name[drug_id]
                    new_name = p.cls().from_string(rec.value)
                    if old_name != new_name:
                        logger.debug("import key %s changed %s to %s from %s"
                                    ,key
                                    ,old_name
                                    ,new_name
                                    ,src_label
                                    )
                        p = Prop.get(Prop.NAME)
                        v = p.cls().objects.get(drug_id=drug_id,prop=p)
                        v.value = new_name
                        v.save()
                else:
                    # this is a new id; create drug record
                    if False:
                        # this is the safe way to do this, which wraps the
                        # creation of the drug in a transaction along with
                        # the external id and canonical properties; it also
                        # doesn't assume an insert, so it probes for the drug
                        # first
                        d = c.get_or_create_drug(key,rec.value)
                    else:
                        # In the interest of speed, since we know it's an
                        # insert, just append the drug, and push the props
                        # onto the batch update list.
                        d = Drug(collection=c)
                        d.save()
                        # Here we create any implied props in addition to the name,
                        # which includes the native_id and {collection}_id fields.
                        #
                        # NOTE: if you add something to this, you probably also want to
                        # add it to "static_properties_list" or it will be cleared on
                        # the next import.
                        for prop_name,val_str in (
                                    (Prop.NAME,rec.value),
                                    (c.key_name,key),
                                    (Prop.NATIVE_ID,key),
                                    ):
                            p = Prop.get(prop_name)
                            cls = p.cls()
                            value = cls.from_string(val_str)
                            rec = cls(drug_id=d.id,
                                        prop=p,
                                        value=value,
                                        )
                            q = queues.setdefault(p.prop_type,[])
                            q.append(rec)
                    key2id[key] = d.id
                    drug_id = d.id
            else:
                error = ''
                # guard against unexpected attributes
                if attr not in load_prop_names:
                    error = "bad attribute '%s'; key %s"%(rec.attribute,key)
                # tolerate missing keys (this allows a single patch file
                # to work for different collection subsets)
                elif key not in key2id:
                    error = "bad key %s"%key
                intolerant = True
                if error:
                    if intolerant:
                        raise Exception(error)
                    else:
                        continue
                # build appropriate type of value record
                drug_id = key2id[key]
                prop = Prop.get(attr)
                cls = prop.cls()
                value = cls.from_string(rec.value)
                rec = cls(drug_id=drug_id,
                        prop=prop,
                        value=value,
                        href=rec.href if has_href else ''
                        )
                q = queues.setdefault(prop.prop_type,[])
                q.append(rec)
                if len(q) >= 2000:
                    logger.debug("import writing %d %s records from %s"
                                ,len(q)
                                ,Prop.prop_types.get('label',prop.prop_type)
                                ,src_label
                                )
                    cls.objects.bulk_create(q)
                    del(queues[prop.prop_type]) # empty queue until next time
        # flush any non-empty queues
        for k,q in queues.items():
            cls = Prop.cls_from_type(k)
            logger.debug("import flushing %d %s records from %s"
                        ,len(q)
                        ,Prop.prop_types.get('label',k)
                        ,src_label
                        )
            cls.objects.bulk_create(q)
        logger.debug("import processed %d drug records from %s"
                    ,len(key2id)
                    ,src_label
                    )

        to_unremove_drugs = Drug.objects.filter(pk__in=matched_existing_drug_ids, removed=True)
        logger.info("%d previously removed drugs are now unremoved", len(to_unremove_drugs))
        to_unremove_drugs.update(removed=False)
        return orphans

    @classmethod
    @transaction.atomic
    def load_from_records(cls,filename,inp):
        from dtk.files import AttrFileName
        af = AttrFileName(filename)
        if not af.ok():
            raise Exception("can't extract collection name")
        if af.use == 'create':
            cls.fast_create_from_records(
                    inp,
                    af.get_name(),
                    af.collection,
                    af.key_name(),
                    )
            return
        elif af.use in ('pt','rt','m'):
            c = cls.get_or_create(af.collection,af.key_name())
            load_props = Prop.prefix_properties_list(af.use)
            cls.fast_load_from_records(af.get_name(),c,load_props,inp)
            return
        else:
            c = cls.get(af.collection)
        header = next(inp)
        key_name = c.key_name
        if key_name not in header:
            raise Exception("key column doesn't match collection name")
        if 'attribute' in header:
            # the file is in keyword/value format
            if 'value' not in header:
                raise Exception("attribute column without value column")
            has_href = ('href' in header)
            if len(header) != (4 if has_href else 3):
                raise Exception("wrong number of columns")
            d = None
            for i,rec in enumerate(convert_records_using_header(inp,header)):
                key = getattr(rec,key_name)
                attr = rec.attribute.lower()
                if d and getattr(d,key_name) != key:
                    d = None
                already_loaded = False
                if d is None:
                    # try to retrieve drug record; first file record
                    # must be 'canonical' in order to create
                    if attr == Prop.NAME:
                        d = c.get_or_create_drug(key,rec.value)
                        already_loaded = True
                    else:
                        d = c.get_or_create_drug(key)
                if not already_loaded:
                    prop = Prop.get(attr)
                    value = prop.cls().from_string(rec.value)
                    if has_href:
                        d.set_prop(prop,value,rec.href)
                    else:
                        d.set_prop(prop,value)
                if i and i % 100 == 0:
                    logger.debug("imported %d attr/val records from %s"
                                ,i
                                ,filename
                                )
            logger.debug("imported a total of %d attr/val records from %s"
                        ,i
                        ,filename
                        )
        else:
            # attributes are column names; values are column values
            skip = [c.key_name]
            create_ok = (Prop.NAME in header)
            if create_ok:
                skip.append(Prop.NAME)
            from dtk.readtext import convert_records_using_header
            for i,rec in enumerate(convert_records_using_header(inp,header)):
                # get the drug object
                key = getattr(rec,c.key_name)
                if create_ok:
                    d = c.get_or_create_drug(key,rec.canonical)
                else:
                    d = c.get_or_create_drug(key)
                # now iterate through all columns not in 'skip'
                # setting attributes on the drug object
                for attr_name in header:
                    if attr_name in skip:
                        continue
                    prop = Prop.get(attr_name)
                    value = getattr(rec,attr_name)
                    value = prop.cls().from_string(value)
                    d.set_prop(prop,value)
                if i and i % 100 == 0:
                    logger.debug("imported %d records from %s"
                                ,i
                                ,filename
                                )
            logger.debug("imported a total of %d records from %s"
                        ,i
                        ,filename
                        )
    @classmethod
    def get_drugset_bucket(cls):
        from dtk.s3_cache import S3Bucket
        return S3Bucket('drugsets')
    @classmethod
    def get_attr_upload_choices(cls):
        l = [];
        l.append(('','None'))
        for key in cls.get_drugset_bucket().list():
            l.append((key,key))
        return l
    @classmethod
    def _patch_tag_prefix(cls,prop_name,prefix):
        patched=0
        total=0
        for tag in Tag.objects.filter(prop__name=prop_name):
            total += 1
            if tag.value.startswith(prefix):
                continue
            tag.value = prefix + tag.value
            tag.save()
            patched += 1
        logger.info(f"patched {patched} of {total} {prop_name} tags")
    @classmethod
    def attr_loader(cls,filename,versioned=False):
        from dtk.files import VersionedFileName
        if versioned:
            # new-style versioned file name
            file_class=filename.split('.')[0]
            vfn = VersionedFileName(file_class=file_class,name=filename)
            collection_name = f'{file_class}.{vfn.flavor}'
            collection_key = file_class + '_id'
        else:
            file_class = cls.get_drugset_bucket()
        from dtk.s3_cache import S3File
        f=S3File(file_class,filename)
        f.fetch()
        from dtk.files import get_file_records
        src=get_file_records(f.path())
        if filename.startswith('prop_'):
            from dtk.readtext import convert_records_using_header
            Prop.load_from_records(convert_records_using_header(src))
        else:
            ua = UploadAudit(filename=filename)
            try:
                if versioned:
                    if collection_key == 'ncats_id':
                        # make sure NCATS keys are patched
                        cls._patch_tag_prefix('ncats_id','NCATS-')
                        cls._patch_tag_prefix('m_ncats_id','NCATS-')
                    cls.fast_create_from_records(
                            src,
                            filename,
                            collection_name,
                            collection_key,
                            )
                else:
                    cls.load_from_records(filename,src)
                ua.ok = True
            except Exception as ex:
                import traceback
                logger.error("got exception '%s' loading '%s'",
                        traceback.format_exc(),
                        filename,
                        )
                ua.ok = False
            ua.save()
            return ua.ok

class UploadAudit(models.Model):
    timestamp = models.DateTimeField(auto_now=True)
    filename = models.CharField(max_length=256)
    ok = models.BooleanField(default=False)
    @classmethod
    def collection_status(cls,collection_name):
        '''Return (version,timestamp,ok) tuple for the specified collection.

        Returns 'None' if no information is available for that collection
        name. This will happen if an invalid collection name is passed in,
        or a valid collection for which no versioned audit information is
        available.
        '''
        try:
            ua = cls.objects.filter(
                    filename__startswith=collection_name+'.'
                    ).order_by('-timestamp')[0]
        except IndexError:
            return None
        parts = ua.filename.split('.')
        return (int(parts[2][1:]), ua.timestamp, ua.ok)
    @classmethod
    def cluster_status(cls):
        '''Return (version,timestamp,ok) tuple for latest clustering data.'''
        try:
            ua = cls.objects.filter(
                    filename__endswith='.clusters.tsv'
                    ).order_by('-timestamp')[0]
        except IndexError:
            return None
        parts = ua.filename.split('.')
        return (int(parts[2][1:]), ua.timestamp, ua.ok)

from dtk.cache import cached_dict_elements
def _matched_id_mm_loader(cls,drug_ids,version):
    if version is None:
        # Can't cache for non-versioned, might have different matchings.
        return False
    else:
        return True, version, drug_ids, 'drug_ids'

class Drug(models.Model):
    def note_info(self,attr):
        if attr == 'bd_note':
            return {
                'label':'Global note on %s'%self.canonical,
                }
        raise Exception("bad note attr '%s'" % attr)
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE)
    bd_note = models.ForeignKey(Note,null=True,blank=True, on_delete=models.CASCADE)
    hide = models.BooleanField(default=False)
    removed = models.BooleanField(default=False)
    ubiquitous = models.BooleanField(default=False)
    def __str__(self):
        return "%s" % (self.canonical,)
    def get_key(self,with_key_name=False):
        native_drug_key = getattr(self,Prop.NATIVE_ID)
        if with_key_name:
            coll_key = self.collection.key_name
            # special format used by molecule matching
            return (coll_key,native_drug_key)
        return native_drug_key
    def get_molecule_matches(self,version):
        try:
            return self.rebuilt_cluster_cache[version]
        except AttributeError:
            self.rebuilt_cluster_cache = {}
        except KeyError:
            pass
        # not in cache; build, cache, and return
        from dtk.drug_clusters import RebuiltCluster
        rbc = RebuiltCluster(
                base_key=self.get_key(with_key_name=True),
                version=version,
                )
        self.rebuilt_cluster_cache[version] = rbc
        return rbc
    def get_assay_info(self,version):
        if version:
            mol_clust = self.get_molecule_matches(version)
        else:
            from dtk.prot_map import AgentAssaysClusterProxy
            mol_clust = AgentAssaysClusterProxy(agent=self)
        from dtk.prot_map import AgentAssays
        return AgentAssays(cluster=mol_clust)
    def get_raw_assay_info(self,version):
        if not version or version < 6:
            version = 6 # Min version with this data
        mol_clust = self.get_molecule_matches(version)
        from dtk.prot_map import RawAgentAssays
        return RawAgentAssays(cluster=mol_clust)
    def get_bd_note_text(self):
        return Note.get(self,'bd_note','')
    def set_prop(self,prop,value,href=''):
        if isinstance(prop,str):
            prop = Prop.get(name=prop)
        cls = prop.cls()
        if prop.multival:
            r,new = cls.objects.get_or_create(drug=self
                                        ,prop=prop
                                        ,value=value
                                        ,href=href
                                        )
        else:
            r,new = cls.objects.get_or_create(drug=self
                                        ,prop=prop
                                        ,defaults={"value":value
                                                ,"href":href
                                                }
                                        )
            if not new and (r.value != value or r.href != href):
                r.value = value
                r.href = href
                r.save()
        Prop.clear_cache(self,prop.prop_type)
        if prop.name == Prop.OVERRIDE_NAME:
            from browse.models import Workspace
            Workspace.clear_wsa2name_cache_by_agent(self.id)
    def del_prop(self,prop,value=None):
        if isinstance(prop,str):
            prop = Prop.get(name=prop)
        cls = prop.cls()
        qs = cls.objects.filter(drug=self,prop=prop)
        if value is not None:
            qs = qs.filter(value=value)
        qs.delete()
        Prop.clear_cache(self,prop.prop_type)
        if prop.name == Prop.OVERRIDE_NAME:
            from browse.models import Workspace
            Workspace.clear_wsa2name_cache_by_agent(self.id)
    # The idea here is to have 4 attributes that cache all the Tag, Flag,
    # Index, and Metric objects for this drug.  They can be filled either
    # by a prefetch call, or on the first access to an attribute of that
    # type.  Subsequent accesses are filled from the cache.  The affected
    # cache is deleted on set_prop calls, so it reloads the new value.  It
    # may also be deleted by request, in case the data is known or suspected
    # to be stale.
    #
    # If the attribute name ends in _set, a set is returned, which might
    # contain 0, 1, or more elements.  Otherwise, a single value or None is
    # returned.  We require the _set suffix on multivals to make sure the
    # client is prepared to handle multiple values.
    #
    # We support the _label suffix to translate index properties based on an
    # Enum class.
    # XXX It may eventually be useful to allow caching by individual property
    # XXX ids, rather than by prop_type. A filter like:
    # XXX       filter(agent__tag__prop__name='canonical')
    # XXX can result in an order of magnitude increase in performance during
    # XXX the fetch, and a corresponding decrease in data needing storage.
    # XXX This comes at the expense of more complex cache management. If we
    # XXX implement this, consider moving cache code into a separate class.
    def __getattr__(self,name):
        label_suffix = '_label'
        if name.endswith(label_suffix):
            name = name[:-len(label_suffix)]
            v = self.__getattr__(name)
            if v is None:
                v = 0
            # now, which enum do we use?
            enum_suffix = '_vals'
            match = None
            for e in dir(Prop):
                if e.endswith(enum_suffix):
                    test = e[:-len(enum_suffix)]
                    if name.endswith('_'+test):
                        match = e
                        break
            enum = getattr(Prop,match)
            return enum.get('label',v)
        multi_suffix = '_set'
        multiple = name.endswith(multi_suffix)
        if multiple:
            name = name[:-len(multi_suffix)]
        try:
            prop = Prop.get(name=name)
        except Prop.DoesNotExist:
            try:
                return super(Drug,self).__getattr__(name)
                # XXX while debugging something else, I noticed the above was
                # XXX throwing an Attribute error with the message:
                # XXX 'super' object has no attribute '__getattr__'
                # XXX I think the actual intent here was that, if super
                # XXX had the attribute in 'name', it would get returned,
                # XXX whether or not it implemented __getattr__. That means
                # XXX the code should be:
                # XXX   return getattr(super(Drug,self),name)
                # XXX This works, but I'm not sure I want to introduce such
                # XXX a deeply buried change without more research.
            except AttributeError:
                raise AttributeError("unknown property '%s'" % name)
        if prop.multival and not multiple:
            raise AttributeError("use '%s%s' for multival" %(name,multi_suffix))
        s = prop.get_vals_via_cache(self)
        if name == Prop.NAME:
            # check for a name override
            prop2 = Prop.get(name=Prop.OVERRIDE_NAME)
            s2 = prop2.get_vals_via_cache(self)
            if s2:
                s = s2
        if multiple:
            return s
        if len(s) == 0:
            return None
        return next(iter(s))
    @classmethod
    def get_linked_agents_map(cls,src_agent_ids,version,prop_type=None):
        '''Return {src_id:set([agent,agent,...]),...}'''
        linked_agent_id_mm = Drug.matched_id_mm(src_agent_ids,version)
        linked_agents_by_id = Drug.prefetched_drugs_by_id(
                list(linked_agent_id_mm.rev_map().keys()),
                prop_type
                )
        linked_agents_by_src = {}
        for src_id,id_set in linked_agent_id_mm.fwd_map().items():
            agent_set = set()
            linked_agents_by_src[src_id] = agent_set
            for agent_id in id_set:
                agent_set.add(linked_agents_by_id[agent_id])
        return linked_agents_by_src
    @classmethod
    @cached_dict_elements(_matched_id_mm_loader, multimap_style=True)
    def matched_id_mm(cls,drug_ids,version):
        '''Return a MultiMap of agent ids matched to each input agent id.
        '''
        from dtk.data import MultiMap
        if version is None:
            id2mval = MultiMap(Tag.objects.filter(
                    prop__name__startswith='m_',
                    drug_id__in=drug_ids,
                    ).values_list('drug_id','value'))
        else:
            drugs = list(Drug.prefetched_drugs_by_id(drug_ids).values())
            drug_keys = [drug.get_key(with_key_name=True) for drug in drugs]
            from dtk.drug_clusters import RebuiltCluster
            clusters = RebuiltCluster.load_many_clusters(version, base_keys=drug_keys)
            matches = []
            for drug, cluster in zip(drugs, clusters):
                matching_keys = cluster.drug_keys
                matches.extend((drug.id, x[1]) for x in matching_keys)
            id2mval = MultiMap(matches)

        coll_key_props = set(
                x.split('.')[0]+'_id'
                for x in Collection.objects.all().values_list('name',flat=True)
                )
        def id_pairs():
            for dest_id,dest_key in Tag.objects.filter(
                    prop__name__in=coll_key_props,
                    value__in=list(id2mval.rev_map().keys()),
                    ).values_list('drug_id','value'):
                for src_id in id2mval.rev_map()[dest_key]:
                    yield((src_id,dest_id))
            # The returned multimap is expected to include each drug in
            # as one of the matches to itself. This almost always happens
            # because the drug key shows up in m_dpimerge_id, but for NCATS
            # (and maybe other future cases) the dpimerge_id is different
            # from the native key. So, return all self-matches explicitly.
            for src_id in drug_ids:
                yield((src_id,src_id))
        return MultiMap(id_pairs())
    def any_wsa_url(self):
        try:
            return self.wsannotation_set.all()[0].drug_url()
        except IndexError:
            # There are no WSAs for this molecule.
            return ''
    def is_moa(self):
        return self.collection.name.startswith('moa.')


    @classmethod
    def prefetched_drugs_by_id(cls,drug_ids,prop_type=None):
        prop_type = prop_type or Prop.prop_types.TAG
        result = {}
        drug_qs=Drug.objects.filter(id__in=drug_ids)
        acl = AttributeCacheLoader(drug_qs,None,prop_type)
        for drug in drug_qs:
            acl.load_cache(drug)
            result[drug.id] = drug
        return result
    # external links
    def external_ids(self,sources,version):
        if isinstance(sources, str):
            sources = [sources]
        if version is None:
            out = set()
            for source in sources:
                source += '_id_set'
                try:
                    out |= getattr(self,Collection.foreign_key_prefix+source)
                except AttributeError:
                    # Newer collections may not have an m_ key, but they also won't
                    # be imported in a legacy context, so ignore.
                    pass

                try:
                    out |= getattr(self, source)
                except AttributeError:
                    # Some collections are not intended for import, so they
                    # have an m_ key, but not a native key. Just ignore these.
                    pass
            return out
        else:
            cluster = self.get_molecule_matches(version)
            id_sources = set(source + '_id' for source in sources)
            return set(x[1] for x in cluster.drug_keys if x[0] in id_sources)
    @classmethod
    def bulk_external_ids(cls,sources,version,drugs):
        if isinstance(sources, str):
            sources = [sources]
        if version:
            # pre-load the cluster cache for all drugs of interest with
            # a single scan of the clusters file
            drug_keys = list(Tag.objects.filter(prop__name=Prop.NATIVE_ID, drug__in=drugs).distinct().values_list('drug__collection__key_name', 'value'))
            from dtk.drug_clusters import RebuiltCluster
            clusters = RebuiltCluster.load_many_clusters(version, drug_keys)
            for drug, cluster in zip(drugs, clusters):
                try:
                    drug.rebuilt_cluster_cache[version] = cluster
                except AttributeError:
                    drug.rebuilt_cluster_cache = {version: cluster}
        # now just iteratively call external_ids
        return [drug.external_ids(sources, version) for drug in drugs]
    @classmethod
    def bulk_synonyms(cls, drugs, version):
        from dtk.data import MultiMap
        idmm = Drug.matched_id_mm(drugs, version=version)

        all_ids = idmm.rev_map().keys()

        canon_mm = MultiMap(Tag.objects.filter(drug__in=all_ids, prop__name='canonical').values_list('drug_id', 'value'))
        synonym_mm = MultiMap(Blob.objects.filter(drug__in=all_ids, prop__name='synonym').values_list('drug_id', 'value'))
        out = []
        for out_drug_id, clust_drug_ids in idmm.fwd_map().items():
            for drug_id in clust_drug_ids:
                for name in canon_mm.fwd_map().get(drug_id, []):
                    out.append((out_drug_id, name))

                for name in synonym_mm.fwd_map().get(drug_id, []):
                    out.append((out_drug_id, name))

        return MultiMap(out)


    @classmethod
    def bulk_prop(cls, drugs, version, prop_name, PropClass):
        from dtk.data import MultiMap
        idmm = Drug.matched_id_mm(drugs, version=version)
        all_ids = idmm.rev_map().keys()

        val_mm = MultiMap(PropClass.objects.filter(drug__in=all_ids, prop__name=prop_name).values_list('drug_id', 'value'))
        out = []
        for out_drug_id, clust_drug_ids in idmm.fwd_map().items():
            for drug_id in clust_drug_ids:
                for val in val_mm.fwd_map().get(drug_id, []):
                    out.append((out_drug_id, val))

        return MultiMap(out)

    def pubchem_cids(self, version):
        # This returns pubchem cids, either that are stored directly, or
        # that are derived from (new-style) stitch ids, or from clustering
        import re
        prefix_len = len('PUBCHEM')
        return getattr(self,'pubchem_cid_set') | set([
                re.match(r'CIDs0*(.*)',k).group(1)
                for k in self.external_ids('stitch', version)
                ]) | set([
                x[prefix_len:] for x in self.external_ids('pubchem',version)
                ])

    def std_smiles_or_bitfp(self):
        """Returns either std_smiles or a bit smiles fingerprint."""
        std_smiles = self.std_smiles
        if std_smiles[0] == '{':
            # Ono-style data where we're given the bitsmiles fingerprint
            # directly rather than the full smiles.
            import json
            return json.loads(std_smiles)
        else:
            return std_smiles

    def commercial_urls(self, version):
        result = []
        seen = set()
        for key in self.external_ids('med_chem_express', version):
            seen.add(key)
            result.append( (key+' (MedChemExpress)',
                'https://www.medchemexpress.com/search.html?q=' + key
                ) )

        for key in self.external_ids('selleckchem', version):
            id = key[3:]
            seen.add(id)
            result.append( (key+' (Selleckchem)',
                'https://www.selleckchem.com/search.html?searchDTO.searchParam=%s&sp=%s' % (id, id)
                ) )

        for key in self.external_ids('cayman', version):
            id = key[3:]
            seen.add(id)
            result.append( (key+' (Cayman)',
                'https://www.caymanchem.com/product/%s' % id
                ) )

        commlink_mm = self.bulk_prop([self.id], version, 'commercial_link', Blob)
        for link in commlink_mm.rev_map().keys():
            source, source_id, url = link.split('|')
            if source_id in seen:
                # The commercial_link field might have dupes with the explicit comm collections.
                continue
            result.append((f'{source_id} ({source})', url))


        return result

    def ext_src_urls(self, version):
        import dtk.url

        drug_db_links = self.commercial_urls(version)

        for key in self.external_ids('drugbank', version):
            drug_db_links.append( (key+' (drugbank)',
                dtk.url.drugbank_drug_url(key),
                ) )
        for key in self.external_ids('ncats', version):
            drug_db_links.append( (key+' (NCATS)',
                "http://ncats.nih.gov/files/%s.pdf" % key,
                ) )
        for key in self.external_ids('ttd', version):
            drug_db_links.append( (key+' (ttd)',
                "http://bidd.nus.edu.sg/group/TTD/ZFTTDDRUG.asp?ID=%s" % key,
                ) )
        for key in self.external_ids('chembl', version):
            drug_db_links.append( (key+' (chembl)',
                dtk.url.chembl_drug_url(key),
                ) )
        for key in self.external_ids('bindingdb', version):
            drug_db_links.append( (key+' (bindingdb)',
                dtk.url.bindingdb_drug_url(key)
                ) )
            drug_db_links.append( (key+' (purchasable page)',
                dtk.url.bindingdb_purchase_drug_url(key)
                ) )
        for key in self.external_ids('globaldata', version):
            drug_db_links.append( (key+' (gloabldata)',
                dtk.url.globaldata_drug_url(key)
                ) )

        pubchem_ids = self.pubchem_cids(version)
        for key in pubchem_ids:
            drug_db_links.append( ('cid '+key+' (pubchem)',
                "https://pubchem.ncbi.nlm.nih.gov/compound/%s" % key,
                ) )
        drug_search_links = []
        for key in pubchem_ids:
            drug_search_links.append( ('cid '+key+' (SIDER ADR)',
                "http://sideeffects.embl.de/drugs/%s" % key,
                ) )
        return drug_db_links, drug_search_links
    @staticmethod
    def prop_prop_pairs(key_prop,val_prop,collections=None):
        '''Return list of paired prop values that coexist in any drug.

        For example, find (name,cas) pairs.
        '''
        if isinstance(key_prop,str):
            key_prop = Prop.get(name=key_prop)
        if isinstance(val_prop,str):
            val_prop = Prop.get(name=val_prop)
        tables=[
                key_prop.value_table_name()+' as t1',
                val_prop.value_table_name()+' as t2',
                ]
        select_conditions=[
                't1.prop_id = %s',
                't2.prop_id = %s',
                ]
        fill_ins=[
                key_prop.id,
                val_prop.id,
                ]
        if collections:
            tables.append('drugs_drug')
            join_conditions=[
                    't1.drug_id = drugs_drug.id',
                    't2.drug_id = drugs_drug.id',
                    ]
            select_conditions.append('drugs_drug.collection_id in %s')
            fill_ins.append(collections)
        else:
            join_conditions=[
                    't1.drug_id = t2.drug_id',
                    ]
        sql = 'select t1.value,t2.value from %s where %s and %s'%(
                ','.join(tables),
                ' and '.join(join_conditions),
                ' and '.join(select_conditions),
                )
        from django.db import connection
        cs = connection.cursor()
        cs.execute(sql,fill_ins)
        return cs.fetchall()

# Previously, a Drug record could match DPI records either with its native
# key, or with an equivalent key defined in an m_ record. Now that we
# exclusively use dpimerge files, each drug record should have exactly
# one dpimerge key for matching DPI records (although the key may change
# from version to version of the dpimerge files, as clustering changes).
# This key can be accessed as:
#   agent.dpimergekey_set.get(version=1).dpimerge_key
# or
#   DpiMergeKey.objects.get(version=1,drug_id=98900).dpimerge_key
class DpiMergeKey(models.Model):
    drug = models.ForeignKey(Drug, on_delete=models.CASCADE)
    version = models.IntegerField()
    dpimerge_key = models.CharField(max_length=256)

    class Meta:
        # There are places where we look up via the dpimerge_key + ver, which
        # is super slow without this index.
        index_together = [
                ['version', 'dpimerge_key'],
                # This one is used in protmap key_agent_pairs.
                ['version', 'drug'],
                ]

    @classmethod
    def max_version(cls):
        from django.db.models import Max
        return DpiMergeKey.objects.aggregate(max_version=Max('version'))['max_version']

    @classmethod
    def load_from_keysets(cls,filename,inp,version):
        ua = UploadAudit(filename=filename)
        try:
            logger.info(f"starting DpiMergeKey upload from {filename}")
            cls._load_from_keysets(inp,version)
            logger.info(f"DpiMergeKey upload complete from {filename}")
            ua.ok = True
        except Exception as ex:
            import traceback
            logger.error("got exception '%s' loading '%s'",
                    traceback.format_exc(),
                    filename,
                    )
            ua.ok = False
        ua.save()
    
    @classmethod
    @transaction.atomic
    def fill_missing_dpimerge(cls,version):
        from drugs.models import Drug, Prop
        # The clusters file, used to create the dpimergekey table, contains only molecules that are clustered with something else
        # (regardless of whether any of them have any DPI entries).
        # Since it is a small minority of molecules that are unclustered with anything and it is easier to write code that
        # assumes that all molecules are in the dpimergekey table, we fill in values for all unclustered molecules here.
        # As of this writing, roughly 20% of drugs get filled in here.
        to_add = Drug.objects.exclude(dpimergekey__version=version).filter(tag__prop__name=Prop.NATIVE_ID).distinct().values_list('id', 'tag__value')
        logger.info(f"Filling in {len(to_add)} unclustered drugs")
        buf = []
        for drug_id, native_key in to_add:
            buf.append(cls(
                    drug_id=drug_id,
                    version=version,
                    dpimerge_key=native_key,
                    ))
            if len(buf) >= 2000:
                cls.objects.bulk_create(buf)
                buf = []
        if buf:
            cls.objects.bulk_create(buf)

    @classmethod
    @transaction.atomic
    def _load_from_keysets(cls,inp,version):
        total,by_type = cls.objects.filter(version=version).delete()
        type_count = by_type.get('drugs.DpiMergeKey',0)
        logger.info(f"replacing {type_count} old records")
        from dtk.prot_map import MoleculeKeySet
        from dtk.data import MultiMap
        coll_cache = {}
        buf = []
        line = 0
        for rec in inp:
            line += 1
            mks = MoleculeKeySet(rec)
            dpimerge_key = mks.best_key()[1]
            for key_name in mks.collections():
                try:
                    key2drug = coll_cache[key_name]
                except KeyError:
                    key2drug = MultiMap(Tag.objects.filter(
                                    prop__name=key_name,
                                    ).values_list('value','drug_id'))
                    logger.debug("cached %s keys for %s",
                            len(key2drug.fwd_map()),
                            key_name,
                            )
                    coll_cache[key_name] = key2drug
                for key in mks.keys(key_name):
                    for drug_id in key2drug.fwd_map().get(key,[]):
                        buf.append(cls(
                                drug_id=drug_id,
                                version=version,
                                dpimerge_key=dpimerge_key,
                                ))
                        if len(buf) >= 2000:
                            logger.debug(
                                f"writing {len(buf)} records, line {line}"
                                )
                            cls.objects.bulk_create(buf)
                            buf = []
        if buf:
            logger.debug(
                    f"writing {len(buf)} records, line {line}"
                    )
            cls.objects.bulk_create(buf)

class Prop(models.Model):
    # hardwired property names
    NAME = "canonical"
    OVERRIDE_NAME = "override_name"
    NATIVE_ID = "native_id"
    _create_checked = False
    prop_types = Enum([],
            [ ('TAG',) # string
            , ('FLAG',) # boolean
            , ('INDEX',) # int/enum
            , ('METRIC',) # float
            , ('BLOB',) # long string
            ])
    protection_vals = Enum([],
            [ ('NOT_SCREENED',)
            , ('NO_PATENT_FOUND',)
            , ('WEAK_PROTECTION',)
            , ('PROTECTED',)
            , ('STRONG_PROTECTION',)
            ])
    name = models.CharField(max_length=256,default="",)
    prop_type = models.IntegerField(choices=prop_types.choices())
    multival = models.BooleanField(default=False)
    _cache = None
    _id_cache = None
    class Meta:
        unique_together = [['name']]
    def prop_type_name(self):
        return self._type_label(self.prop_type)
    def rec_count(self):
        cls = self.cls()
        return cls.objects.filter(prop=self).count()
    @classmethod
    def prefix_properties_list(cls,prefix):
        prefix += '_'
        return [p for p in cls.objects.filter(name__startswith=prefix)
                #if p.name.startswith(prefix)
                ]
    @classmethod
    def reset(cls):
        # set static state back to startup
        cls._cache = None
        cls._id_cache = None
        cls._create_checked = False

    _cache_lock = Lock()
    @classmethod
    def _load_caches(cls):
        # cache all properties from DB
        if cls._cache is None:
            # Double-checked locking; multiple requests could come in
            # from multiple threads here, make sure they never see an empty
            # cache or try to load it twice.
            with cls._cache_lock:
                if cls._cache is None:
                    cls.load_hard_coded()
                    cache = {}
                    id_cache = {}
                    for p in Prop.objects.all():
                        cache[p.name] = p
                        id_cache[p.id] = p
                    cls._id_cache = id_cache
                    cls._cache = cache
    @classmethod
    def load_hard_coded(cls):
        # make sure all the hard-coded properties exist
        if not cls._create_checked:
            cls.get_or_create(cls.NAME,cls.prop_types.TAG)
            cls.get_or_create(cls.OVERRIDE_NAME,cls.prop_types.TAG)
            cls._create_checked = True
    @classmethod
    def load_from_records(self,inp):
        for rec in inp:
            p = Prop.get_or_create(rec.name
                                    ,int(rec.prop_type)
                                    ,bool(int(rec.multival))
                                    )
            # automatically changing the details of an existing property is
            # not supported; if it must be done, it can be done in SQL,
            # taking care to properly migrate (or delete) old values
            assert(p.prop_type == int(rec.prop_type))
            assert(p.multival == bool(int(rec.multival)))
    @classmethod
    def get(cls,name):
        cls._load_caches()
        try:
            return cls._cache[name]
        except KeyError:
            raise Prop.DoesNotExist('bad prop name: %s'%name)
    @classmethod
    def get_by_id(cls,id):
        cls._load_caches()
        try:
            return cls._id_cache[id]
        except KeyError:
            raise Prop.DoesNotExist('bad prop id: %d'%id)
    @classmethod
    def get_or_create(cls
                    ,name
                    ,prop_type=None
                    ,multival=False
                    ):
        if prop_type is None:
            prop_type = cls.prop_types.TAG
        p,new = cls.objects.get_or_create(name=name
                                ,defaults={'prop_type':prop_type
                                        ,'multival':multival
                                        }
                                )
        if new:
            cls.reset() # force cache reload if new prop created
        return p

    @classmethod
    def _type_label(cls,prop_type):
        """Return type-specific label ('tag', 'flag', ...)."""
        return cls.prop_types.get('label',prop_type).lower()
    @classmethod
    def _cache_attr(cls,prop_type):
        """Return type-specific attribute name.

        This attribute is attached to a Drug instance to cache all associated
        Tag, Flag, Index, or Metric objects.  The cache will be filled on the
        first call to get_vals_via_cache(), which in turn is invoked by an
        attribute reference on the Drug object.  If add_prefetch() is called
        for a QuerySet, the cache will be filled on all Drug objects as they're
        retrieved.  The cache is emptied by calling clear_cache(); it will
        reload on the next attribute fetch.  This is done automatically by
        the set_prop() and del_prop() operations.
        """
        return '_prop_%s_cache' % cls._type_label(prop_type)
    @classmethod
    def clear_cache(cls,drug,prop_type):
        attr = cls._cache_attr(prop_type)
        try:
            delattr(drug,attr)
        except AttributeError:
            pass
    def get_vals_via_cache(self,drug):
        attr = self._cache_attr(self.prop_type)
        cls = self.cls()
        try:
            l = getattr(drug,attr)
        except AttributeError:
            l = cls.objects.filter(drug=drug).values_list('prop_id','value')
            setattr(drug,attr,l)
        return set(x[1] for x in l if x[0] == self.id)
    @classmethod
    def cls_from_type(self,prop_type):
        if prop_type == Prop.prop_types.TAG:
            return Tag
        if prop_type == Prop.prop_types.FLAG:
            return Flag
        if prop_type == Prop.prop_types.INDEX:
            return Index
        if prop_type == Prop.prop_types.METRIC:
            return Metric
        if prop_type == Prop.prop_types.BLOB:
            return Blob
        return None
    def cls(self):
        return self.cls_from_type(self.prop_type)
    def value_table_name(self):
        return 'drugs_'+self._type_label(self.prop_type)

class Tag(models.Model):
    drug = models.ForeignKey(Drug, on_delete=models.CASCADE)
    prop = models.ForeignKey(Prop, on_delete=models.CASCADE)
    value = models.CharField(max_length=256,default="")
    href = models.CharField(max_length=1024,default="")
    class Meta:
        # Normally we could just annotate with db_index, but this
        # is an alternative syntax that works with django's migrations.
        index_together = [
                ['value'],
                ['prop', 'value'],
                ['prop', 'drug'],
                ]
    @classmethod
    def from_string(cls,string):
        if hasattr(string, 'decode'):
            return string.decode('utf8')
        else:
            return string

class Flag(models.Model):
    drug = models.ForeignKey(Drug, on_delete=models.CASCADE)
    prop = models.ForeignKey(Prop, on_delete=models.CASCADE)
    value = models.BooleanField(default=False)
    href = models.CharField(max_length=1024,default="")
    @classmethod
    def from_string(cls,string):
        return string.lower() in ('t','true','1')

class Index(models.Model):
    drug = models.ForeignKey(Drug, on_delete=models.CASCADE)
    prop = models.ForeignKey(Prop, on_delete=models.CASCADE)
    value = models.IntegerField()
    href = models.CharField(max_length=1024,default="")
    @classmethod
    def from_string(cls,string):
        return int(string)

class Metric(models.Model):
    drug = models.ForeignKey(Drug, on_delete=models.CASCADE)
    prop = models.ForeignKey(Prop, on_delete=models.CASCADE)
    value = models.FloatField()
    href = models.CharField(max_length=1024,default="")
    @classmethod
    def from_string(cls,string):
        return float(string)

class Blob(models.Model):
    """
    NOTE: This model has a custom index specified in raw SQL.
    It is created in drugs/migrations/0005_auto_20200323_1316

    This is needed because MySQL doesn't support arbitrary length indexes,
    and django doesn't support specifying prefix indexes.
    """
    drug = models.ForeignKey(Drug, on_delete=models.CASCADE)
    prop = models.ForeignKey(Prop, on_delete=models.CASCADE)
    value = models.TextField()
    href = models.CharField(max_length=1024,default="")
    @classmethod
    def from_string(cls,string):
        if hasattr(string, 'decode'):
            return string.decode('utf8')
        else:
            return string

class DrugProposal(models.Model):
    data = models.TextField()
    user = models.CharField(max_length=50)
    timestamp = models.DateTimeField(auto_now=True)
    drug_name = models.TextField(default='')

    # The ID that this drug will use within the twoxar drug collection.
    collection_drug_id = models.CharField(max_length=16, null=True, blank=True)

    # This proposed drug could replace a drug, or could be a new one entirely.
    ref_drug = models.ForeignKey(Drug, null=True, on_delete=models.CASCADE)
    # We could also be proposing an edit to a proposal that doesn't correspond
    # to a drug on the platform (yet).
    ref_proposal = models.ForeignKey("DrugProposal", null=True, on_delete=models.CASCADE)

    states = Enum([],
            [ ('PROPOSED',)
            , ('REJECTED',)
            , ('ACCEPTED',)
            , ('SKIPPED',)
            , ('OUT_OF_DATE',)
            ])
    active_states = (states.PROPOSED, states.ACCEPTED, )
    state =  models.IntegerField(choices=states.choices(), default=states.PROPOSED)

    @property
    def state_text(self):
        return self.states.get('label', self.state)

    def related_proposals(self):
        """Returns the set of proposals that are for the same drug as this."""
        related = set()
        if self.ref_drug:
            related.update(self.ref_drug.drugproposal_set.all())
        related.update(DrugProposal.objects.filter(collection_drug_id=self.collection_drug_id))
        related -= set([self])
        return related


###
# utilities
###

def set_agent(ws):
    from browse.models import WsAnnotation
    c = Collection.get_or_create("drugbank")
    qs = WsAnnotation.objects.filter(ws=ws)
    prop = Prop.get(c.key_name)
    for wsa in qs:
        print(wsa.pk,wsa.drug_id,prop.pk,c.pk)
        d = Drug.objects.get(collection=c
                            ,tag__prop=prop
                            ,tag__value=wsa.drug_id
                            )
        wsa.agent = d
        wsa.save()
