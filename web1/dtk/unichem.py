

from dtk.cache import cached_dict_elements
def convert_cache_wrapper(self, k_src, v_src, version, key_subset=None):
    # Only cache if we're doing a specific key subsetting.
    if (key_subset is None) or len(key_subset) > 5000:
        return False, None, None, None

    return True, (k_src, v_src, version), key_subset, 'key_subset'

class UniChem(object):
    def __init__(self):
        self._load_src_map()
    def _load_src_map(self):
        import os
        from path_helper import PathHelper
        from dtk.files import get_file_records
        file_name='unichem_src_map.tsv'
        sm_file = os.path.join(
                               PathHelper.website_root,
                               'browse',
                               'static',
                                file_name
                               )
        self.source_map={}
        header = None
        for frs in get_file_records(sm_file):
            if header is None:
                header = frs
                continue
            self.source_map[frs[header.index('name')]] = int(frs[header.index('id')])

    # given 2 source names, the first to serve as a key and the second to be the value,
    # this will return a dictionary that allows the user to go from the first source to the 2nd.
    # note the values are lists, even if there is only one entry
    # e.g.:
    #   input: chembl, zinc
    #   output: {chembl_id:[zinc_id(s)]}
    # e.g. 2:
    #   input: zinc, chembl
    #   output: {zinc_id:[chembl_id(s)]}
    @cached_dict_elements(convert_cache_wrapper)
    def get_converter_dict(self, k_src,v_src, version, key_subset=None):
        from dtk.s3_cache import S3File
        assert k_src in self.source_map
        assert v_src in self.source_map
        assert v_src != k_src
        inds = self._get_order(k_src,v_src)
        names = [k_src,v_src]
        s3_file = S3File.get_versioned(
                'unichem',
                version,
                role=self._get_file_role(names[inds[0]], names[inds[1]]),
                format='tsv.gz',
                )
        s3_file.fetch()
        return self.get_converter_dict_from_file(
                s3_file.path(),
                inds,
                key_subset,
                )
    def get_converter_dict_from_file(self, fn, inds, key_subset=None):
        d = {}
        from dtk.files import get_file_records
        if key_subset is None:
            gen = (x for x in get_file_records(fn,
                                               keep_header=False,
                                              )
                   )
        else:
            gen = (x for x in get_file_records(fn,
                                               keep_header=False,
                                               select=(key_subset,inds[0]),
                                              )
                   )
        for frs in gen:
            k = frs[inds[0]]
            v = frs[inds[1]]
            if k not in d:
                d[k] = []
            d[k].append(v)
        return d
    def _get_file_role(self,s1,s2):
        return s1 + '_to_' + s2
    def _get_order(self,k_src,v_src):
        if self.source_map[k_src] < self.source_map[v_src]:
            return [0,1]
        return [1,0]
