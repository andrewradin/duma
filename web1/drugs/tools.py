
from dtk.lazy_loader import LazyLoader

class CollectionUploadStatus(LazyLoader):
    cluster_bucket='matching'
    cache_ok=True # XXX speedup, but not as dynamic
    def _bucket_loader(self):
        from dtk.s3_cache import S3Bucket
        return S3Bucket(self.cluster_bucket)
    def _all_vfns_loader(self):
        from dtk.files import VersionedFileName
        return [
                VersionedFileName(file_class=self.cluster_bucket,name=x)
                for x in self.bucket.list(cache_ok=self.cache_ok)
                ]
    def _cluster_vfns_loader(self):
        return [ x for x in self.all_vfns if x.role == 'clusters' ]
    def _props_vfns_loader(self):
        return [ x for x in self.all_vfns if x.role == 'props' ]
    def _loaded_clusters_loader(self):
        from drugs.models import UploadAudit
        return set(UploadAudit.objects.filter(
                filename__startswith=self.cluster_bucket+'.',
                filename__contains='.clusters.',
                ok=True,
                ).values_list('filename',flat=True))
    def _needed_clusters_loader(self):
        return [x
                for x in self.cluster_vfns
                if x.to_string() not in self.loaded_clusters
                ]
    # As a first cut, hard-code the maintained collections here. For each
    # collection in this list, we assure the latest version has been uploaded.
    # XXX A more sophisticated approach would allow this list to change
    # XXX between versions of the cluster file, and would prevent a new
    # XXX version from being uploaded prior to being incorporated into
    # XXX clustering. Note that this list is different from the list in
    # XXX the 'ingredients' file; this list may contain only some of those
    # XXX sources, and different flavors.
    maintained_collections=[
            'drugbank.full',
            'ncats.full',
            'chembl.adme_condensed',
            'bindingdb.full_condensed',
            'duma.full',
            'med_chem_express.full',
            'pubchem.filtered',
            'moa.full',
            'lincs.full',
            'globaldata.full',
            ]
    def _needed_attrs_loader(self):
        from dtk.s3_cache import S3Bucket
        from drugs.models import UploadAudit
        from dtk.files import VersionedFileName
        fn_end = '.attributes.tsv'
        result = []
        for coll_name in self.maintained_collections:
            fn_start = coll_name+'.'
            file_class,flavor = coll_name.split('.')
            fn_meta = VersionedFileName.Meta(
                    prefix=file_class,
                    roles=['attributes'],
                    )
            s3b = S3Bucket(file_class)
            attr_files = [x
                    for x in s3b.list(cache_ok=self.cache_ok)
                    if x.startswith(fn_start)
                    and x.endswith(fn_end)
                    ]
            done = set(UploadAudit.objects.filter(
                    filename__startswith=fn_start,
                    filename__endswith=fn_end,
                    ok=True,
                    ).values_list('filename',flat=True))
            load_version = 0
            latest_done_version = 0
            for fn in attr_files:
                vfn = VersionedFileName(meta=fn_meta,name=fn)
                if fn in done:
                    latest_done_version = max(latest_done_version, vfn.version)
                if vfn.version > load_version:
                    load_version = vfn.version
            if load_version > latest_done_version:
                result.append(f"{fn_start}v{load_version}{fn_end}")
        return result

