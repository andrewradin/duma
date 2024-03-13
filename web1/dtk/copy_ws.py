
from dtk.subclass_registry import SubclassRegistry
from django import forms

################################################################################
# Support for copying workspace components
# - each component is represented by a PartCopier subclass, which implements
#   a copy() method
# - any state and configuration shared between components is held in a
#   CopyContext instance
# - any user configuration that influences a component's copy can be obtained
#   by overriding the add_form_fields() method in the PartCopier subclass
#   (by default, the base class provides a single checkbox to enable or disable
#   copying that component)
# - the external interface is via add_copy_options() and do_copies()
################################################################################

def get_ordered_subclasses():
    subclasses = [x[1] for x in PartCopier.get_subclasses()]
    # put list in declaration order; this allows control of the order of
    # the displayed fields (and the order of copy execution) based on the
    # class declaration order in this file
    import inspect
    subclasses.sort(key=lambda x:inspect.getsourcelines(x)[1])
    return subclasses

def add_copy_options(ff,default_enable):
    # add fields to form subclass-by-subclass
    for subclass in get_ordered_subclasses():
        subclass.add_form_fields(ff,default_enable)

from dtk.lazy_loader import LazyLoader
class CopyContext(LazyLoader):
    _kwargs = ['user','src_ws','dst_ws']
    def _wsa_map_loader(self):
        from browse.models import WsAnnotation
        src_qs = WsAnnotation.objects.filter(ws=self.src_ws)
        agent2src = {
                agent_id:src_id
                for src_id,agent_id in src_qs.values_list('id','agent_id')
                }
        result = {}
        dst_qs = WsAnnotation.objects.filter(ws=self.dst_ws)
        for dst_id,agent_id in dst_qs.values_list('id','agent_id'):
            if agent_id in agent2src:
                result[agent2src[agent_id]] = dst_id
        return result
    def invalidate_wsa_map(self):
        try:
            delattr(self,'wsa_map')
        except AttributeError:
            pass
    def copy_note(self,note_id):
        if not note_id:
            return None
        from django.db import transaction
        from notes.models import Note,NoteVersion
        with transaction.atomic():
            old_note = Note.objects.get(pk=note_id)
            old_vers = old_note._latest_version()
            if not old_vers.text:
                return None
            new_note = Note.objects.create(
                    label = old_note.label+f' (cloned to ws {self.dst_ws.id})',
                    private_to = old_note.private_to,
                    )
            NoteVersion.objects.create(
                    version_of=new_note,
                    created_by = old_vers.created_by,
                    text = f'(Cloned from ws {self.src_ws.id})\n\n'
                            + old_vers.text,
                    )
            return new_note.id
    ####
    # Unfortunately, creating a cloned job is quite CM-specific. The
    # following methods encapsulate the reusable parts.
    ####
    def copy_job_prep(self,job_id):
        '''Return a (src_bji,dst_job) tuple.

        src_bji is the bound JobInfo record for the job to be copied.
        dst_job is a template Process record to be modified and written.
        Both will be None if there's nothing worth copying.
        '''
        from runner.process_info import JobInfo
        src_bji = JobInfo.get_bound(self.src_ws,job_id)
        if src_bji.job.status != src_bji.job.status_vals.SUCCEEDED:
            return (None,None)
        # get a separate copy of the job record to modify
        from runner.models import Process
        dst_job = Process.objects.get(pk=job_id)
        dst_job.id = None
        # assure old job LTS stuff is fetched
        src_bji.fetch_lts_data()
        return (src_bji,dst_job)
    def copy_job_dummy_logfile(self,src_job_id,dst_job):
        '''Create and push dummy logfile and progress file for cloned job.'''
        from runner.common import LogRepoInfo
        lri = LogRepoInfo(dst_job.id)
        log_path = lri.log_path()
        import os
        from path_helper import make_directory
        make_directory(os.path.dirname(log_path))
        progress_path = lri.progress_path()
        desc = f'from ws {self.src_ws.id} job {src_job_id}'
        with open(log_path,'w') as fh:
            fh.write(f'cloned '+desc+'\n')
        with open(progress_path,'w') as fh:
            fh.write('cloning job\n')
            fh.write(f'complete ({desc})\n')
        lri.push_log()
    ####
    # CM-specific job copies
    ####
    def copy_sig_job(self,job_id):
        src_bji,dst_job = self.copy_job_prep(job_id)
        if not src_bji:
            return None
        assert src_bji.job_type == 'sig'
        d = dict(dst_job.settings())
        from browse.models import Tissue
        old_tissue = Tissue.objects.get(pk=d['tissue_id'])
        new_tissue = Tissue.objects.get(
                ws=self.dst_ws,
                name=old_tissue.name,
                )
        d['tissue_id'] = new_tissue.id
        import json
        dst_job.settings_json = json.dumps(d)
        dst_job.name = src_bji.get_jobname_for_tissue(new_tissue)
        dst_job.save()
        from runner.process_info import JobInfo
        dst_bji = JobInfo.get_bound(self.dst_ws,dst_job.id)
        # copy output tree and push to LTS. We can get away with an identical
        # copy because there are no wsa-keyed scores.
        import shutil
        shutil.copytree(src_bji.lts_abs_root,dst_bji.lts_abs_root)
        import os
        # need to re-name files whose names contain a tissue id
        base_dir = os.path.join(dst_bji.lts_abs_root,'publish')
        if os.path.exists(base_dir):
            from dtk.files import get_dir_file_names
            for old_name in get_dir_file_names(base_dir):
                old_path = os.path.join(base_dir,old_name)
                new_name = old_name.replace(
                        f'_{old_tissue.id}_',
                        f'_{new_tissue.id}_',
                        )
                new_path = os.path.join(base_dir,new_name)
                os.rename(old_path,new_path)
        # finish up
        dst_bji.finalize()
        self.copy_job_dummy_logfile(src_bji.job.id,dst_job)
        return dst_job.id

def do_copies(user,post_data,src_ws,dst_ws):
    ctx = CopyContext(
            user=user,
            src_ws=src_ws,
            dst_ws=dst_ws,
            )
    for subclass in get_ordered_subclasses():
        prefix = subclass.prefix()
        kwargs = {
                x[len(prefix):] : post_data[x]
                for x in post_data
                if x.startswith(prefix)
                }
        if kwargs.pop('enabled'):
            subclass.copy(ctx,**kwargs)

class PartCopier(SubclassRegistry):
    @classmethod
    def prefix(cls):
        return cls.__name__+'_'
    @classmethod
    def add_form_fields(cls,ff,default_enable):
        # by default, just add an enable field; this method can be
        # overridden if additional field info is required; if more
        # fields are defined, they should begin with class.__name__+'_',
        # and will show up as keyword arguments to the copy() method
        # (with the prefix removed from the argument name)
        ff.add_field(
                cls.prefix()+'enabled',
                forms.BooleanField(
                        label = 'Copy '+cls.label,
                        required = False,
                        initial = default_enable,
                        )
                )
    @classmethod
    def copy(self,ctx):
        raise NotImplementedError('copy() must be overridden by subclass')

class DiseaseNames(PartCopier):
    label = "Disease Names"
    @classmethod
    def copy(self,ctx):
        from browse.models import DiseaseDefault
        old = dict(DiseaseDefault.objects.filter(
                ws=ctx.dst_ws,
                ).values_list('vocab','value'))
        for dd in list(DiseaseDefault.objects.filter(ws=ctx.src_ws)):
            if old.get(dd.vocab) == dd.value:
                continue
            ctx.dst_ws.set_disease_default(dd.vocab,dd.value,ctx.user)

class Versions(PartCopier):
    label = "Versions and Defaults"
    @classmethod
    def copy(self,ctx):
        from browse.models import VersionDefault
        before = VersionDefault.get_defaults(ctx.dst_ws.id)
        src = VersionDefault.get_defaults(ctx.src_ws.id)
        VersionDefault.set_defaults(ctx.dst_ws.id,[
                (k,v)
                for k,v in src.items()
                if k != 'DiseaseShortName' and before[k] != v
                ],ctx.user)

class DrugImports(PartCopier):
    label = "Drug Imports"
    @classmethod
    def add_form_fields(cls,ff,default_enable):
        super(DrugImports,cls).add_form_fields(ff,default_enable)
        from browse.models import WsAnnotation
        ind_choices = WsAnnotation.indication_vals.choices()
        from dtk.html import WrappingCheckboxSelectMultiple
        ff.add_field(
                cls.prefix()+'ind_choices',
                forms.MultipleChoiceField(
                        label = '...copying these source indications',
                        choices = ind_choices,
                        initial = [x[0] for x in ind_choices],
                        required = False,
                        widget=WrappingCheckboxSelectMultiple,
                        )
                )
    @classmethod
    def copy(self,ctx,*,ind_choices):
        ind_choices = set(int(x) for x in ind_choices)
        from browse.models import WsAnnotation
        existingWsaByAgent = {
                wsa.agent:wsa
                for wsa in WsAnnotation.all_objects.filter(ws=ctx.dst_ws)
                # NOTE: this will silently drop duplicate agents
                }
        # Introspect through fields to get list of what to copy...
        fields_to_skip = set([
                'id',
                'invalid',
                'ws',
                'agent',
                'doc_href',
                'txr_id', # XXX workspace-specific?
                'replacements', # XXX map to new WSAs (in 2nd pass?)
                ])
        field_list = [
                x
                for x in WsAnnotation._meta.get_fields()
                if x.concrete and x.name not in fields_to_skip
                ]
        # ...and what to write during updates
        field_write_list = [
                x.name
                for x in WsAnnotation._meta.get_fields()
                if x.concrete and x.name not in (
                        'id',
                        'replacements', # can't do bulk updates on m2m
                        )
                ]
        create_list = []
        update_list = []
        def do_creates():
            WsAnnotation.objects.bulk_create(create_list)
        def do_updates():
            # this needs to use all_objects because otherwise invalid WSAs
            # won't get updated
            WsAnnotation.all_objects.bulk_update(update_list, field_write_list)
        for src_wsa in WsAnnotation.objects.filter(ws=ctx.src_ws):
            try:
                dst_wsa = existingWsaByAgent[src_wsa.agent]
                # there's already a wsa (maybe invalidated) for this agent
                dst_wsa.invalid = False
            except KeyError:
                # we need a new wsa
                dst_wsa = WsAnnotation(ws=ctx.dst_ws,agent=src_wsa.agent)
            for field in field_list:
                # default to source value
                attr = field.name
                val = getattr(src_wsa,attr)
                # do any field-specific special processing; this processing
                # may alter 'val' to set a different value, and 'attr' to
                # set the value to a different field. Setting 'attr' to
                # None suppresses the write at the bottom of the loop.
                if field.name == 'marked_prescreen':
                    # this is associated with the source ws; force to blank
                    val = None
                elif field.name == 'indication':
                    if dst_wsa.id: 
                        # don't alter existing
                        val = dst_wsa.indication
                    elif src_wsa.indication not in ind_choices:
                        val = WsAnnotation.indication_vals.UNCLASSIFIED
                elif field.name == 'demerit_list':
                    if dst_wsa.id: 
                        # don't alter existing
                        val = dst_wsa.demerit_list
                    elif src_wsa.indication not in ind_choices:
                        val = ''
                elif field.name == 'study_note':
                    val = ctx.copy_note(src_wsa.study_note_id)
                    attr = attr + '_id'
                if attr:
                    setattr(dst_wsa,attr,val)
            if dst_wsa.id is None:
                create_list.append(dst_wsa)
                if len(create_list) == 1000:
                    do_creates()
                    create_list = []
            else:
                update_list.append(dst_wsa)
                # update SQL is very complex; do smaller batches
                if len(update_list) == 100:
                    do_updates()
                    update_list = []
        if create_list:
            do_creates()
        if update_list:
            do_updates()
        ctx.invalidate_wsa_map() # force reload

class AESearches(PartCopier):
    label = 'AE Searches'
    # Model here is:
    # - AeSearch represents a (ws,term,mode,species) combo; repeated searches
    #   will re-use the record, updating the timestamp and deleting and
    #   re-creating the scores
    # - AeAccession objects represent a (ws,geoID), and are created as needed
    #   (as scores need something to point to)
    # - AeDisposition objects are tied to an AeAccession and a mode, and
    #   represent the rejection of an accession. Acceptance of an accession
    #   is represented by the presence of a tissue record (even if that record
    #   is subsequently excluded from processing)
    # What we want to accomplish here is:
    # - make sure any AeDispositions in the source also exist in the
    #   destination; this represents work done to reject an accession that
    #   won't need to be repeated
    # - make sure that any AeSearches in the source also exist in the
    #   destination; these represent search terms that have been tried in
    #   the source that won't need to be rediscovered.
    # - the normal use case after a copy will be to re-run all the searches
    #   to get new data, so copying of AeScore records is less important.
    #   If we create a new search in the destination, we'll copy the scores,
    #   which correspond to the 'when' field in the search; if we skip a
    #   search copy because the destination already has that one, we'll
    #   skip the score copy as well (which means the dest scores and the
    #   dest search.when field will always be aligned)
    @classmethod
    def copy(self,ctx):
        from browse.models import AeSearch,AeAccession,AeDisposition,AeScore
        # find any existing AeAccession records in the destination
        old_dst_acc_by_geo = {
                acc.geoID:acc.id
                for acc in AeAccession.objects.filter(ws=ctx.dst_ws)
                }
        # pre-copy AeAccession records (not search-specific)
        acc_map = {}
        for acc in AeAccession.objects.filter(ws=ctx.src_ws):
            try:
                # accession already exists in dst?
                acc_map[acc.id] = old_dst_acc_by_geo[acc.geoID]
            except KeyError:
                # no, need new accession
                old_acc_id = acc.id
                acc.id = None
                acc.ws_id = ctx.dst_ws.id
                acc.save()
                acc_map[old_acc_id] = acc.id
        # find any existing AeDisposition records in the destination
        old_dst_disp_by_acc_and_mode = {
                (disp.accession_id,disp.mode):disp.id
                for disp in AeDisposition.objects.filter(
                        accession__ws=ctx.dst_ws,
                        )
                }
        # pre-copy AeDisposition records (not search-specific)
        for disp in AeDisposition.objects.filter(accession__ws=ctx.src_ws):
            key = (acc_map[disp.accession_id],disp.mode)
            if key in old_dst_disp_by_acc_and_mode:
                continue
            disp.id = None
            disp.accession_id = acc_map[disp.accession_id]
            disp.save()
        # Now copy searches and scores
        for srch in AeSearch.objects.filter(ws=ctx.src_ws):
            if AeSearch.objects.filter(
                    ws=ctx.dst_ws,
                    term=srch.term,
                    mode=srch.mode,
                    species=srch.species,
                    ).exists():
                continue
            # copy AeSearch record
            src_srch_id = srch.id
            srch.id = None
            srch.ws_id = ctx.dst_ws.id
            srch.save()
            # copy AeScore records for search
            for score in AeScore.objects.filter(search_id=src_srch_id):
                score.id = None
                score.search_id = srch.id
                score.accession_id = acc_map[score.accession_id]
                score.save()

class GE(PartCopier):
    label = "GE Tissues"
    @classmethod
    def copy(self,ctx):
        # This pairs tissues and tissue sets by name, so if a tissue of
        # the same name already exists in the destination workspace, that
        # source tissue won't be copied, even if it has a different GeoID.
        #
        # To assure that this works, we need to make sure neither the source
        # or destination workspaces have multiple tissues with the same name.
        # We need to check this because no other place in the code really
        # cares, so duplicate tissue names can exist in the wild.
        from browse.models import Tissue,Sample
        from dtk.data import MultiMap
        def check_dup_tissue_names(ws):
            mm = MultiMap((
                    (name.lower()),id)
                    for name,id in Tissue.objects.filter(
                            ws=ws,
                            ).values_list('name','id')
                    )
            for k,s in mm.fwd_map().items():
                if len(s) > 1:
                    raise RuntimeError(
                            f'ws {ws.id} has {len(s)} tissues named {k}'
                                +' (some may be excluded)',
                            )
        check_dup_tissue_names(ctx.src_ws)
        check_dup_tissue_names(ctx.dst_ws)
        # build index to existing tissue sets; this makes sure default
        # tissue sets exist as a side effect
        dst_ts_by_name = {
                x.name:x
                for x in ctx.dst_ws.get_tissue_sets()
                }
        # copy all source tissue sets not already present
        # we call get_tissue_sets to force creation of defaults, but then
        # we invalidate the cache so we can modify the returned TS instances
        # without corrupting the cached copies
        #
        # At the end of this loop, ts_id_map will contain the destination
        # tissue set id for each source tissue set.
        src_ts_list = list(ctx.src_ws.get_tissue_sets())
        ctx.src_ws.invalidate_tissue_set_cache()
        ts_id_map = {}
        for ts in src_ts_list:
            src_pk = ts.pk
            if ts.name in dst_ts_by_name:
                ts_id_map[src_pk] = dst_ts_by_name[ts.name].pk
                continue
            ts.pk = None # force new record
            ts.ws = ctx.dst_ws # in destination ws
            ts.save()
            ts_id_map[src_pk] = ts.pk
        # get pre-existing destination tissues
        dst_tissue_by_name = {
                x.name:x
                for x in Tissue.objects.filter(
                        ws=ctx.dst_ws,
                        )
                }
        # now copy all tissues
        for tissue in Tissue.objects.filter(ws=ctx.src_ws):
            if tissue.name in dst_tissue_by_name:
                continue
            # XXX Samples get retrieved here, but are not yet copied
            src_samples = list(Sample.objects.filter(tissue=tissue))
            tissue.pk = None # force new record
            tissue.ws = ctx.dst_ws # in destination ws
            tissue.tissue_set_id = ts_id_map.get(tissue.tissue_set_id)
            tissue.note_id = ctx.copy_note(tissue.note_id)
            # make sure tissue exists before copying sig job
            tissue.save()
            if tissue.sig_result_job_id:
                # Make a dummy sig job and update tissue record. We don't
                # need to worry about copying cutoff_job_id because that's
                # never referenced -- it's just there to generate a warning
                # if it differs from sig_result_job.
                map_both = (tissue.cutoff_job_id == tissue.sig_result_job_id)
                new_job_id = ctx.copy_sig_job(tissue.sig_result_job_id)
                tissue.sig_result_job_id = new_job_id
                if map_both:
                    tissue.cutoff_job_id = new_job_id
                tissue.save()

class GWAS(PartCopier):
    label = "GWAS Datasets"
    @classmethod
    def copy(self,ctx):
        # actual data files are extracted on demand;
        # this code just need to create db records
        from browse.models import GwasDataset
        for src_gds in ctx.src_ws.get_gwas_dataset_qs():
            dst_gds,new = GwasDataset.objects.get_or_create(
                    ws = ctx.dst_ws,
                    phenotype = src_gds.phenotype,
                    pubmed_id = src_gds.pubmed_id,
                    note_id = ctx.copy_note(src_gds.note_id)
                    )

class DrugSets(PartCopier):
    label = "Drug Sets"
    @classmethod
    def copy(self,ctx):
        from browse.models import DrugSet
        for src_ds in DrugSet.objects.filter(ws=ctx.src_ws):
            try:
                dst_ds = DrugSet.objects.get(
                        ws=ctx.dst_ws,
                        name=src_ds.name,
                        )
            except DrugSet.DoesNotExist:
                dst_ds = DrugSet(ws=ctx.dst_ws,name=src_ds.name)
                dst_ds.created_by = f'copy from ws {src_ds.ws_id}'
            dst_ds.description = src_ds.description
            dst_ds.save()
            for src_id in src_ds.drugs.values_list('id',flat=True):
                dst_id = ctx.wsa_map.get(src_id)
                if dst_id:
                    dst_ds.drugs.add(dst_id)

class ProtSets(PartCopier):
    label = "Protein Sets"
    @classmethod
    def copy(self,ctx):
        from browse.models import ProtSet,Protein
        for src_ps in ProtSet.objects.filter(ws=ctx.src_ws):
            try:
                dst_ps = ProtSet.objects.get(
                        ws=ctx.dst_ws,
                        name=src_ps.name,
                        )
            except ProtSet.DoesNotExist:
                dst_ps = ProtSet(ws=ctx.dst_ws,name=src_ps.name)
                dst_ps.created_by = f'copy from ws {src_ps.ws_id}'
            dst_ps.description = src_ps.description
            dst_ps.save()
            for prot_id in src_ps.proteins.values_list('id',flat=True):
                dst_ps.proteins.add(prot_id)

