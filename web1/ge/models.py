from django.db import models
from browse.models import AeAccession

class SraRun(models.Model):
    """For BioProject SRAs we know the list of SRX samples beforehand

    We store them in the database for reference as part of meta.
    """

    # Typically SRX###, but can be e.g. ERX or DRX.
    experiment = models.CharField(max_length=64)
    # Typically PRJNA###, but can be e.g. PRJEB or PRJDB
    bioproject = models.CharField(max_length=64, db_index=True)
    # Typically SAMN###, but can be e.g. SAMEA
    biosample = models.CharField(max_length=64)
    # Typically SRR###, but can be e.g. ERR or DRR.
    accession = models.CharField(max_length=64)

    # We pull these out indirectly from the attrs, may not always exist.
    # NOTE: This is the corresponding GEO/GDS ID for the BIOPROJECT/PRJ,
    # not the corresponding AeAccession geoID which will be PRJNA#####.
    geo_id = models.CharField(max_length=64,blank=True,null=True)

    # This will be a list of key-value pairs (potentially duplicate keys).
    attrs_json = models.TextField()
    class Meta:
        unique_together = [('experiment', 'bioproject')]

    def attrs_dict(self):
        import json
        from collections import defaultdict
        data = json.loads(self.attrs_json)
        seen = defaultdict(int)
        out = {}
        # Annoyingly the SRA attrs has repeated keys, so we have to handle that and rename them
        # to be unique.  They will always show up in a consistent order, at least.
        for k, v in data:
            idx = seen[k]
            seen[k] += 1
            if idx > 0:
                k = f'{k}_{idx}'
            out[k] = v
        return out




class GESamples(models.Model):
    """These are ArrayExpress or GEO samples.

    Similar use-case as the SraRun types above.
    Unlike those, however, we have a single entry for all samples for a given experiment.
    
    GEO and AE data has a lot of repeated long text fields, so this helps a lot with compression.
    """
    # We don't tie to a single accession because those are workspace-specific, and we can
    # instead share samples across worskpaces.

    # This corresponds to an AeAccession.geoID
    geoID = models.CharField(max_length=64, primary_key=True)

    # Compressed JSON list of samples, where each sample is a dict of key-value pairs
    attrs_json_gz = models.BinaryField()

    def get_sample_attrs(self):
        import isal.igzip as gzip
        import json
        uncmp = gzip.decompress(self.attrs_json_gz)
        return json.loads(uncmp.decode('utf8'))
    
    @classmethod
    def compress_sample_attrs(cls, attrs):
        if attrs:
            assert(isinstance(attrs[0], dict)), "Samples should be dicts"
        import isal.igzip as gzip
        import json
        byte_data = json.dumps(attrs).encode('utf8')
        cmp = gzip.compress(byte_data)
        return cmp
        
