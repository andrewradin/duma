from django.db import models
from tools import Enum

################################################################################
# non-DB classes
################################################################################
class IndicationMapper:
    def _indication_info(self):
        # returns list of indication,phase_set pairs, ordered from least
        # to most restrictive indication
        from browse.models import WsAnnotation
        return [
                (WsAnnotation.indication_vals.EXP_TREATMENT,set([0])),
                (WsAnnotation.indication_vals.TRIALED1_TREATMENT,set([1])),
                (WsAnnotation.indication_vals.TRIALED2_TREATMENT,set([2])),
                (WsAnnotation.indication_vals.TRIALED3_TREATMENT,set([3])),
                (WsAnnotation.indication_vals.KNOWN_TREATMENT,set([4])),
                (WsAnnotation.indication_vals.FDA_TREATMENT,set([])),
                ]
    def indication_of_phase(self,phase):
        phase = int(phase)
        for ind,ps in self._indication_info():
            if phase in ps:
                return ind
        raise RuntimeError('No indication for phase %d',phase)
    def sort_key(self,indication):
        indication = int(indication)
        for i,(ind,_) in enumerate(self._indication_info()):
            if ind == indication:
                return i+1
        return 0

################################################################################
# DB models
#
# The schema here is a little complex.
# - a KtSearch is the top level; one is created whenever the user clicks the
#   'Search' button on the Treatment Search page. It just records who did
#   the search, when, and in what workspace, but its record id is then used
#   as the master id for a full set of results.
# - each such search queries multiple information sources (Chembl, Clinical
#   Trials, GlobalData, etc.). Each information source is represented by
#   a KtSource object hanging off the KtSearch, and holding source-specific
#   query parameters
# - each hit returned by a source is represented by a KtSearchResult. This is
#   basically the name of a treatment, a reference URL, the presumed
#   indication_val, and some fields used for disposition
#   - if the name can be associated with a wsa in the workspace, it will point
#     to a KtResultGroup
#   - if the name can't be associated, 'unmatch_confirmed' will be set to true
#   - if neither of these is true, the name still needs to be resolved
# - a KtResultGroup represents a wsa in the workspace that corresponds to
#   one or more KtSearchResults, and an indication of how the wsa was updated
#   in light of those results
################################################################################
class KtSearch(models.Model):
    ws = models.ForeignKey("browse.Workspace", on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    user = models.CharField(max_length=70)

    # the following are some common 'noise' drugnames; ignore them
    stoplist = set([
            'human',
            'acid',
            'mg',
            'sodium chloride',
            'in',
            'low',
            'sodium',
            'the',
            ])
    def _get_group(self,name):
        try:
            return self._name_lookup[name]
        except KeyError:
            pass
        wsa = self._match_name(name)
        grp = self._groups.get(wsa)
        print('processing new name',name,wsa,grp)
        if wsa and not grp:
            grp = KtResultGroup(search=self,wsa=wsa)
            grp.save()
            self._groups[wsa] = grp
        self._name_lookup[name] = grp
        return grp
    def add_item(self,query,name,href,ind_val,extra=''):
        if not hasattr(self,'_processed'):
            # build local caches to avoid duplicate processing
            # Note that originally these caches were always created empty,
            # on the assumption that all add_item calls happened on a
            # single KtSearch object. The initial implementation of name
            # resolution would issue additional add_item calls on new
            # instances, so populating the caches correctly became important.
            # The final implementation used get_or_create to manage
            # the KtResultGroup directly in the database, so the loading
            # code below no longer does anything, but has been left
            # in place.
            self._groups = {
                    g.wsa:g
                    for g in self.ktresultgroup_set.all()
                    }
            self._name_lookup = {}
            self._processed = set()
            for sr in KtSearchResult.objects.filter(query__search=self):
                if sr.group:
                    self._name_lookup[sr.drugname] = sr.group
                self._processed.add((sr.drugname,sr.href,sr.ind_val))
        # record a result in the database; there will typically be
        # multiple results for the same drug; we group them here
        # for matching purposes
        # XXX add any centralized canonicalization here, before matching;
        # XXX any per-source canonicalization should happen prior to the
        # XXX add_item call
        name = name.lower()
        # don't add noise
        if name in self.stoplist:
            return
        # don't add duplicate items
        key = (name,href,ind_val)
        if key in self._processed:
            return
        self._processed.add(key)
        grp = self._get_group(name)
        sr = KtSearchResult(
                query=query,
                group=grp,
                drugname=name,
                href=href,
                ind_val=ind_val,
                extra=extra,
                )
        sr.save()
    def _match_name(self,name):
        from drugs.models import Prop
        from browse.models import WsAnnotation
        name_prop=Prop.get(Prop.NAME)
        qs = WsAnnotation.objects.filter(ws=self.ws
                ,agent__tag__prop=name_prop
                ,agent__tag__value=name
                )
        # XXX maybe handle case of multiple matches better?
        if qs.count() == 1:
            return qs[0]
        # If there's no canonical name match, try again
        # using synonyms.  If that produces a unique match, use it.
        if qs.count() == 0:
            rewrite_from = None
            for suffix in (
                    'lauroxil',
                    'maleate',
                    'hydrochloride',
                    'hydrobromide',
                    'sodium',
                    'succinate',
                    'dimesylate',
                    'mesylate',
                    'pamoate',
                    'palmitate',
                    'fumarate',
                    ):
                if name.endswith(' '+suffix):
                    rewrite_from=name
                    name = name[:-len(suffix)-1]
                    break

            qs2 = WsAnnotation.objects.filter(ws=self.ws
                    ,agent__tag__value=name
                    ).distinct()
            if rewrite_from:
                print('from',rewrite_from,'to',name,'got',qs2)
            if qs2.count() == 1:
                return qs2[0]

            if qs2.count() == 0:
                found = None
                from browse.utils import drug_search
                search = drug_search(
                        version=self.ws.get_dpi_version(),
                        pattern=name,
                        pattern_anywhere='exact',
                        )
                dpi_ver=self.ws.get_dpi_version()
                for mergekey,_,_,_ in search:
                    wsas = WsAnnotation.objects.filter(
                            ws=self.ws,
                            agent__dpimergekey__version=dpi_ver,
                            agent__dpimergekey__dpimerge_key=mergekey,
                            )
                    if wsas.count() >= 1:
                        # If it's >1, this is a cluster mates situations,
                        # just pick one.
                        if found:
                            return None # more than one match; let user resolve
                        found = wsas[0]
                # return the unique wsa match in the workspace, or None
                # if drug_search returned nothing, or only returned non-imported
                # drugs
                return found



class KtSource(models.Model):
    search = models.ForeignKey(KtSearch, on_delete=models.CASCADE)
    source_type = models.CharField(max_length=70)
    config = models.TextField(default='',blank=True)
    def search_term(self):
        import json
        d=json.loads(self.config)
        return d['search_term']
    def source_label(self):
        from .sources import KtSourceType
        return KtSourceType.lookup(self.source_type).src_name

class KtResultGroup(models.Model):
    resolution_vals = Enum([], [
            ('UNRESOLVED',),
            ('ACCEPTED',),
            ('SKIPPED',),
            ('MATCHED_EXISTING',),
            ])
    # XXX allows null only to simplify migration
    search = models.ForeignKey(KtSearch,null=True,blank=True, on_delete=models.CASCADE)
    wsa = models.ForeignKey('browse.WsAnnotation', on_delete=models.CASCADE)
    resolution = models.IntegerField(
                choices=resolution_vals.choices(),
                default=resolution_vals.UNRESOLVED,
                )
    # who resolved the group, and when (excluding auto-resolve,
    # i.e. MATCHED_EXISTING)
    user = models.CharField(max_length=50,blank=True,default="")
    timestamp = models.DateTimeField(null=True)

    class Meta:
        unique_together = [['search', 'wsa']]
    def mark_resolver(self,user):
        '''Helper function for setting resolution user/timestamp.

        Note that you still need to call save().
        '''
        self.user = user
        from django.utils import timezone
        self.timestamp = timezone.now()
    def cache_evidence(self,im=None,items=None):
        im = im or IndicationMapper()
        items = items or self.ktsearchresult_set.all()
        # build map from source type name to usable_href flag
        from .sources import KtSourceType
        type_map = {
                name:cls.usable_href
                for name,cls in KtSourceType.get_subclasses()
                }
        # Order evidence by proposed indication, then usable_href flag.
        # This will sort the most restrictive indications. For the most
        # restrictive indication, usable sources will sort ahead of
        # unusable ones. If the most restrictive indication has only
        # unusable sources, this triggers special behavior in the view
        # to force manually setting an href.
        self.evidence = sorted(
                items,
                key=lambda x:(
                        im.sort_key(x.ind_val),
                        type_map[x.query.source_type],
                        ),
                reverse=True,
                )
        self.proposed_indication = self.evidence[0].ind_val
    def proposed_indication_label(self):
        return self.wsa.indication_vals.get('label',self.proposed_indication)
    def best_indication_href(self):
        return self.evidence[0].href
    def href_usable(self):
        return self.evidence[0].href_usable()
    def __str__(self):
        return '%s(%d)'%(self.__class__.__name__,self.id)

class KtSearchResult(models.Model):
    query = models.ForeignKey(KtSource, on_delete=models.CASCADE)
    group = models.ForeignKey(KtResultGroup,blank=True,null=True, on_delete=models.CASCADE)
    drugname = models.CharField(max_length=256,default="")
    href = models.CharField(max_length=1024,default="")
    extra = models.TextField(default='',blank=True)
        # An example function to retroatively populate the 'extra' field
        # may be retrieved from git. It was deleted in the same commit
        # in which this comment was added, because it used pre-versioned
        # files, and wasn't worth converting.
    from browse.models import WsAnnotation
    ind_val = models.IntegerField(
            choices=WsAnnotation.indication_vals.choices(),
            )
    unmatch_confirmed = models.BooleanField(default=False)
    def label(self):
        return self.drugname+(' (%s)'%self.query.source_label())
    def proposed_indication_label(self):
        from browse.models import WsAnnotation
        return WsAnnotation.indication_vals.get('label',self.ind_val)
    def href_usable(self):
        from .sources import KtSourceType
        st = KtSourceType.lookup(self.query.source_type)
        return st.usable_href
    @classmethod
    def ordered_unmatched_names(cls,search):
        unmatched = list(cls.objects.filter(
                query__search=search,
                group__isnull=True,
                unmatch_confirmed=False,
                ).values_list('ind_val','drugname','id'))
        im = IndicationMapper()
        # sort by indication, and by name within indication;
        # since we want indications to count down and names to
        # count up, we do a forward sort, but use the negative ind key
        unmatched.sort(
                key=lambda x:(-im.sort_key(x[0]),x[1])
                )
        from collections import namedtuple
        ReturnType=namedtuple('UnmatchedName','ind name id')
        return [ReturnType(*x) for x in unmatched]

################################################################################
# maintenance / testing functions
################################################################################
def relink_groups():
    qs=KtSearchResult.objects.all().order_by('query__search__id','id')
    plugged = 0
    for sr in qs:
        if sr.group and sr.group.search_id != sr.query.search_id:
            if not sr.group.search_id:
                sr.group.search_id = sr.query.search_id
                sr.group.save()
                plugged += 1
            else:
                grp,created = KtResultGroup.objects.get_or_create(
                        search_id=sr.query.search_id,
                        wsa_id=sr.group.wsa_id,
                        )
                print('reassigning',sr.group,'to',grp)
                sr.group=grp
                sr.save()
    print('plugged',plugged,'null pointers')

