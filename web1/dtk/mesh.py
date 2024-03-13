class MeshDiseases:
    _fallback_maps = [
            ('alias','alias_map'),
            ('variant','variant_map'),
            ]
    def __init__(self,choice):
        from dtk.s3_cache import S3File
        s3f=S3File.get_versioned('mesh',choice,role='diseases')
        s3f.fetch()
        self._load_terms(s3f.path())
    def match(self,term):
        term = term.lower()
        if term in self.base_terms:
            return (term,'exact')
        for label,map_attr in self._fallback_maps:
            try:
                return (getattr(self,map_attr)[term],label)
            except KeyError:
                pass
        return (None,None)
    def _load_terms(self,fn):
        from dtk.files import get_file_records
        self.base_terms = set()
        self.base_name_categories = dict()
        # name_alias_pairs yields each base name with each of its aliases,
        # which is used to construct the alias map. As a by-product, it
        # also builds the set of base names, and a map of the categories
        # for each base name.
        def name_alias_pairs():
            for rec in get_file_records(fn):
                categories = rec[0].split(',')
                # strip categories column from rec; now rec[0] holds preferred
                # disease name and other columns hold aliases, all lower-case
                rec = [x.lower() for x in rec[1:]]
                self.base_terms.add(rec[0])
                self.base_name_categories[rec[0]] = set(categories)
                for alias in rec[1:]:
                    yield (rec[0],alias)
        from dtk.data import MultiMap
        mm = MultiMap(name_alias_pairs())
        self.alias_map = dict(MultiMap.flatten(mm.rev_map()))
        # Now, construct variants for aliases and base terms that follow some
        # common patterns.
        def get_variants(s):
            result = set()
            import re
            single_comma = re.compile(r'([^,]+), ([^,]+)$')
            for x in s:
                m = single_comma.match(x)
                if m:
                    result.add(m.group(1)+' '+m.group(2)) # strip comma
                    result.add(m.group(2)+' '+m.group(1)) # strip comma and flip
            return result
        self.variant_map = {}
        for k,s in mm.fwd_map().items():
            v = get_variants(s)
            for x in v-s:
                self.variant_map[x] = k
        for k in self.base_terms:
            v = get_variants([k])
            v -= self.alias_map.get(k,set())
            for x in v:
                self.variant_map[x] = k

